# ember 面试准备手册（INTERVIEW.md）

> 用法：按「教学顺序」一个文件一个文件读。每个文件分三层：
> **(a) 骨架**——这个文件由哪几块组成、数据怎么流；**(b) 为什么**——设计动机和取舍；**(c) 语法**——看不懂的 Python/PyTorch 写法。
> 力气主要花在 (a)(b)，(c) 查到即可。每个 Stage 末尾有约 8 道面试题，答案都能在代码里找到出处。
> 术语第一次出现时给英文原词，之后中英混用。

---

## Stage 1 — 从零写 Qwen3 前向（Model from scratch）

### 读代码顺序
`ember/config.py` → `ember/model.py` → `ember/attention.py` → `ember/loader.py` → `ember/tokenizer.py` → `ember/simple_generate.py` → `tests/test_model.py` → `tests/test_parity_hf.py` → `benchmarks/parity_report.py` / `benchmarks/bench_single.py`

### 1. `ember/config.py`
**(a) 骨架**：一个冻结的 `ModelConfig` 数据类（dataclass）+ `from_pretrained_dir()` 读 HF 的 `config.json` + `tiny_config()` 造一个迷你模型配置给单元测试用。
**(b) 为什么**：引擎运行时不依赖 `transformers`，只需要前向用到的十几个超参数。`kv_bytes_per_token()` 是整个项目最重要的一行数学：每个 token 的 KV 缓存（KV cache）占 `2(K和V) × 层数 × kv头数 × head_dim × 每元素字节`。Qwen3-0.6B fp16：2×28×8×128×2 = 114,688 B ≈ 112 KiB/token。之后算「2 GiB 能放多少 token」全靠它。
**(c) 语法**：`@dataclass(frozen=True)` = 不可变的值对象（像 Java 的 record）；`@property` = 无参 getter；`@classmethod` = 工厂方法（像 Java 的 static factory）。

### 2. `ember/model.py`（核心）
**(a) 骨架**：`RMSNorm` → `RotaryEmbedding`（RoPE）→ `Attention`（GQA + q/k norm）→ `MLP`（SwiGLU）→ `DecoderLayer`（两个残差块）→ `Qwen3Model`（embedding → 28 层 → norm → logits）。
所有张量都是「打平的 token」布局 `[T, ...]`：一个 batch 里所有序列的 token 首尾相接（packed / varlen），只有注意力需要知道哪个 token 属于哪个序列——这个知识封装在传进来的 `AttentionBackend` 里。
**(b) 为什么**：
- **打平布局**：MLP、投影层对每个 token 独立，所以不同序列拼一起算没问题，也不需要 padding；这让后面「prefill 和 decode 混在一个 batch」变得自然（vLLM 的做法）。
- **RMSNorm 在 float32 里算**：和 HF 完全一致，半精度下平方求和容易损失精度。
- **RoPE 预计算 cos/sin 表**：按绝对位置索引，第 1000 个位置和第 0 个代价一样。
- **GQA（Grouped-Query Attention）**：16 个 query 头共享 8 个 KV 头，KV 缓存直接减半。
- **Qwen3 特有的 q_norm / k_norm**：每个头内部再做一次 RMSNorm（在 RoPE 之前），这是 Qwen3 相比 Qwen2 的变化，漏掉它输出就是错的。
- **融合 qkv_proj 和 gate_up_proj**：加载时把 3 个/2 个权重矩阵拼成一个，少几次 kernel launch，在 MPS 上很重要（单请求 decode 主要受 launch 开销限制）。
- **只对需要采样的行算 logits**：`[H, V] = 1024×151936` 是模型里最大的矩阵乘，prefill 时只算最后一个 token。
- **绑定词嵌入（tied embeddings）**：`logits = h @ embed_tokens.weight.T`，不单独存 lm_head。
**(c) 语法**：`nn.Module` 的 `forward` 通过 `module(x)` 调用；`register_buffer(..., persistent=False)` = 跟着 `.to(device)` 走但不进 state_dict；`x.chunk(2, -1)` 沿最后一维切两半；`F.silu` = x·sigmoid(x)。

### 3. `ember/attention.py`
**(a) 骨架**：`AttentionBackend` 协议（Protocol，像 Java interface）只有一个方法 `forward(layer_idx, q, k, v)`；两个实现：`NoCacheAttention`（无缓存，单序列因果注意力）和 `ContiguousKVCache`（为单序列预留 `max_len` 长度的连续缓存）。
**(b) 为什么**：把「KV 存哪、怎么读」从模型里抽出来，模型代码一行不改就能换成 Stage 2 的分页缓存（PagedAttention）。`ContiguousKVCache` 故意保留作为 naive 基线：它一开始就把 `max_len` 全部预留——这正是分页要消除的浪费。
因果 mask 的偏移：第 i 个新 query 的绝对位置是 `start_pos + i`，它能看到 key `j <= start_pos + i`。decode 时只有 1 个 query，可以看全部，所以 `mask=None`。
**(c) 语法**：`typing.Protocol` = 结构化类型（鸭子类型），不需要显式继承；`F.scaled_dot_product_attention(..., enable_gqa=True)` 让 PyTorch 自己处理 GQA，不用 `repeat_interleave` 复制 KV。

### 4. `ember/loader.py`
**(a) 骨架**：`resolve_model_dir`（手动解析 HF 缓存目录 `refs/main` → snapshot）→ `_checkpoint_name_map`（我们的参数名 → checkpoint 张量名列表）→ `load_model`（meta 设备上建模型 → 读 safetensors → 拼接融合 → `load_state_dict(assign=True)`）。
**(b) 为什么**：`torch.device("meta")` 上建模型不分配内存，再用 `assign=True` 直接把读进来的张量换进去，避免「先随机初始化一份、再拷贝」的双倍内存。RoPE 表是 buffer 不在 checkpoint 里，所以要 `_materialize_buffers` 重新在真实设备上建。checkpoint 里虽然有 `lm_head.weight`，但 tie=True 时我们不用它。
**(c) 语法**：`safe_open(path, framework="pt")` 是内存映射（mmap）读取，按需加载单个张量。

### 5. `ember/tokenizer.py`
**(a) 骨架**：`Tokenizer`（包装 `tokenizers` 库 + 用 jinja2 渲染模型自带的 chat template）和 `IncrementalDetokenizer`（流式输出时把 token 增量变成文本）。
**(b) 为什么**：BPE token 不是一一对应字符，一个中文字/emoji 的 UTF-8 字节可能被拆到两个 token 里，单独解码会得到 `U+FFFD`（�）。所以末尾是 � 时先不输出，等后续 token 补全。测试里验证了我们的 chat template 渲染和 HF `apply_chat_template` 字符串完全一致。
**(c) 语法**：`ImmutableSandboxedEnvironment` = 安全的 jinja2 沙箱（模板来自模型仓库，不可信）。

### 6. `ember/simple_generate.py`
**(a) 骨架**：一次 prefill（整段 prompt 前向）→ 循环 decode（每次 1 个 token，复用缓存）→ greedy argmax。
**(b) 为什么**：最简单、最可信的参考实现。后面批处理引擎的输出必须和它逐 token 相同（测试里就是这样比的）。
**(c) 语法**：`@torch.inference_mode()` 比 `no_grad` 更彻底地关闭 autograd 记录。

### Stage 1 实测结论（数字出处见 BENCHMARK.md）
- fp32 和 bf16（CPU）下，28 层每层隐藏状态和 logits 与 HF transformers 的最大绝对误差都是 **0.0**（逐位相同）。
- 8 个 prompt × 64 token greedy：fp32 CPU 8/8 相同；**MPS fp16 8/8 相同**；MPS bf16 只有 3/8 与 fp32 参考相同（即使和 HF 自己的 MPS bf16 比也只有 6/8）→ 所以 MPS 默认用 fp16。

### Stage 1 面试题
1. **Q：KV cache 每个 token 占多少内存？怎么算？**
   A：`2 × num_layers × num_kv_heads × head_dim × bytes`。Qwen3-0.6B fp16 = 2×28×8×128×2 = 112 KiB/token，2 GiB 约能放 18.7k token。出处 `config.py: kv_bytes_per_token`，测试 `test_kv_bytes_per_token_for_qwen3_0_6b_shape`。
2. **Q：为什么有 KV cache？没有它 decode 复杂度是多少？**
   A：自回归每步只新增 1 个 token，但注意力需要所有历史 token 的 K/V。不缓存就要每步重算整段前缀，生成 n 个 token 总计 O(n²) 次 token 前向；缓存后每步只算 1 个 token 的投影，注意力读历史 K/V。代价是显存随长度线性增长。
3. **Q：GQA 和 MHA/MQA 的区别？对推理有什么好处？**
   A：MHA 每个 query 头有自己的 K/V；MQA 所有头共享一组；GQA 折中（Qwen3-0.6B：16 Q 头 / 8 KV 头）。KV cache 和读带宽按 KV 头数缩小（这里减半），decode 是访存瓶颈，所以直接提速。
4. **Q：RoPE 的核心性质是什么？你怎么测试的？**
   A：把位置编码成旋转，使 q·k 只依赖相对距离 (m−n)。`test_rope_scores_depend_only_on_relative_distance` 验证 (10,3) 和 (110,103) 的分数相同；位置 0 时旋转是恒等变换。
5. **Q：怎么证明你的实现是对的？**
   A：三层：单元测试（RMSNorm 公式、RoPE 性质、增量 decode == 全量前向）；与 HF 逐层对比（28 层最大误差 0.0）；greedy 生成与 HF `generate` 逐 token 相同（8 prompt × 64 token）。
6. **Q：为什么 bf16 在 MPS 上和参考结果不一致，而 fp16 一致？**
   A：bf16 只有 8 位尾数（fp16 有 11 位），logits 里前两名差距很小时，矩阵乘累加顺序的微小不同就会翻转 argmax，之后整段生成就分叉。融合 qkv 会改变 MPS 矩阵乘的分块/累加顺序，所以连和 HF 的 bf16 也只有 6/8 一致。Qwen3-0.6B 激活值在 fp16 范围内，fp16 更准，所以选 fp16。
7. **Q：单请求 decode 速度是被什么限制的？**
   A：batch=1 时每步只有约 1.2 GB 权重读取，M2 的带宽理论上能到上百 token/s，但我们实测约 40 tok/s——瓶颈是 PyTorch eager 模式在 MPS 上的 kernel launch 和 Python 开销（每层十几个小 kernel × 28 层）。这正是批处理（Stage 3）能大幅提升吞吐的原因：一次 launch 服务多个序列。
8. **Q：为什么要融合 QKV / gate-up？有什么代价？**
   A：3 个矩阵乘变 1 个，kernel 数减少、矩阵更大更好利用硬件；代价是加载时要拼接权重（名字映射稍复杂），以及在低精度下累加顺序改变导致和参考不再逐位一致（见第 6 题）。

---

## Stage 2 — 分页 KV 缓存（Paged KV cache）

### 读代码顺序
`ember/sequence.py` → `ember/block_manager.py` → `ember/paged_attention.py` → `ember/model_runner.py` → `tests/test_block_manager.py` → `tests/test_paged_attention.py` → `benchmarks/bench_memory.py`

### 1. `ember/sequence.py`
**(a) 骨架**：`Sequence` 数据类：`token_ids`（prompt + 已生成）、`num_computed_tokens`（已经写进 KV 缓存的 token 数）、`block_table`、状态（WAITING / RUNNING / FINISHED）。
**(b) 为什么**：整个调度器只看一个差值 `num_tokens - num_computed_tokens`：>1 是 prefill（可能分块），=1 是 decode。分块 prefill（chunked prefill）、前缀缓存命中（prefix cache hit，直接把 computed 往前推）、抢占重算（preemption by recompute，把 computed 归零）全都是改这一个计数器——同一条代码路径。
**(c) 语法**：`field(default_factory=list)` 避免可变默认值共享；`field(init=False)` + `__post_init__` 在构造后派生字段；`eq=False` 让对象按身份比较（`seq in list` 不会逐字段比）。

### 2. `ember/block_manager.py`
**(a) 骨架**：`Block(block_id, ref_count)` 列表 + `free_block_ids` 双端队列（deque，空闲链表）。核心方法：`num_new_blocks_needed` / `can_allocate` / `allocate_slots` / `free` / `fork`。逻辑位置 t 的物理槽位（slot）= `block_table[t // bs] * bs + t % bs`。
**(b) 为什么**：
- 和操作系统虚拟内存一样：逻辑连续、物理不连续。按需分配，一个序列最多浪费 `block_size-1` 个槽（内部碎片，internal fragmentation），任何空闲块都能接到任何序列后面，所以没有外部碎片（external fragmentation）。
- `allocate_slots` 失败时**没有副作用**（先检查再分配），调度器可以放心先问后做。
- `free` 逆序释放：序列尾部最不可能被共享，先回到空闲队列被复用（Stage 4 前缀缓存时有用）。
- 引用计数（ref count）支持共享块：`fork` 让子序列共享父序列的块；只共享**满的**块，所以不需要写时复制（copy-on-write）。
**(c) 语法**：`-(-n // k)` 是整数向上取整；`deque.popleft()` O(1)。

### 3. `ember/paged_attention.py`（核心）
**(a) 骨架**：`PagedKVCache`（每层一个扁平的 `[num_blocks*bs, kv_heads, d]` K 池和 V 池）；`PagedAttentionMetadata.build`（每步构建一次、所有层共享：`slot_mapping`、`gather_slots`、`query_index`、`mask`）；`PagedAttention.forward` 三步：① `index_copy_` 按 slot 写入新 K/V ② 按 `gather_slots` 把每个序列的上下文收集成 `[B, kv_heads, S, d]` ③ 把打平的 query 散到 `[B, H, max_q, d]`，一次批量 SDPA，再按 `query_index` 收回来。
**(b) 为什么**：
- mask 同时编码两件事：key 存在（`j < context_len`）和因果（`j <= query 的绝对位置`）。query i 的绝对位置是 `context_len - query_len + i`——这个公式让同一套代码处理 prefill、分块 prefill、decode。
- padding 的 query 行永远至少能看到 key 0，所以不会出现全 mask → softmax NaN（我一开始写了保护代码，推导后发现是死代码，删了，测试里留了断言）。
- 代价（诚实说）：没有融合的 paged attention kernel（vLLM 用 CUDA kernel 直接按块表读），我们每层都把 K/V 真的复制出来，访存 ∝ batch × 最长上下文。
**(c) 语法**：`tensor[index_tensor]` 是高级索引（gather，会复制）；`index_copy_(0, idx, src)` 原地按行写；`@classmethod build` 是工厂方法。

### 4. `ember/model_runner.py`
**(a) 骨架**：`ScheduledSeq(seq, num_new_tokens)` + `ModelRunner.execute(batch)`：拼 `input_ids`/`positions` → 建 metadata → 前向 → 只对 `completes_sequence` 的序列取最后一个 token 算 logits。
**(b) 为什么**：分块 prefill 的中间块不需要采样，就不做 `[H,V]` 投影（省最大的那个 matmul）。
**(c) 语法**：`torch.inference_mode()` 装饰器。

### Stage 2 实测（`results/stage2_memory.json`）
64 条真实生成（prompt 17–34 token，输出 11–256，37 条遇 EOS 停）：逐步平均利用率 paged(bs=16) **93.3%**，预留 prompt+max_tokens **53.7%**，预留 max_model_len=2048 **7.3%**；2 GiB KV 能放 85 / 67 / 9 个这样的请求。

### Stage 2 面试题
1. **Q：为什么需要分页？naive 预分配浪费在哪？**
   A：naive 必须在请求开始时按最大可能长度预留连续显存，但输出长度事先未知，多数请求早停（我们 64 条里 37 条遇 EOS）。实测预留 prompt+max_tokens 平均只有 53.7% 的槽被用到；分页按需分配，93.3%。
2. **Q：分页的碎片有多少？block size 怎么选？**
   A：每个序列只有最后一个块可能不满，浪费 < block_size。实测 bs=8/16/32 利用率 96.7%/93.3%/87.9%。块越小浪费越少，但块表更长、gather 索引更多、kernel 效率下降；取 16（vLLM 默认）。
3. **Q：逻辑 token 位置怎么映射到物理位置？**
   A：`slot = block_table[pos // bs] * bs + pos % bs`，见 `BlockManager.slot` 和 `PagedAttentionMetadata.build`。
4. **Q：你的 paged attention 和 vLLM 的有什么不同？**
   A：数据布局相同（块表 + slot mapping），但 vLLM 用 CUDA kernel 在注意力内部按块表直接读 K/V；我在 PyTorch 里先 gather 成 padded 张量再调 SDPA，多了一次复制和 padding。原因：MPS 上没有现成的 paged kernel。这是已知的性能差距。
5. **Q：怎么测试 block manager 的正确性？**
   A：hypothesis 状态机测试：随机交错 add / decode / fork / free，每步检查不变量——ref_count 等于出现在多少个块表里；空闲链表恰好是 ref_count==0 的块且无重复；每个序列的块数正好是 ⌈computed/bs⌉。`derandomize=True` 保证可复现。
6. **Q：prefill 和 decode 混在一个 batch 里，mask 怎么构造？**
   A：每个序列 b 有 `query_len` 和 `context_len`，query i 的绝对位置 = `context_len - query_len + i`，`mask[b,i,j] = (j <= 该位置) & (j < context_len)`。decode 时 query_len=1，就是看全部历史。
7. **Q：共享块为什么不需要 copy-on-write？**
   A：我们只共享**已经写满**的块（前缀缓存按满块哈希），满块之后不会再被写；新 token 总是写进序列自己的新块。只有共享未满的块（比如 beam search 从中间分叉）才需要 CoW。
8. **Q：KV 池大小怎么定？**
   A：`num_blocks = 预算字节 / (kv_bytes_per_token × block_size)`（`PagedKVCache.num_blocks_for_budget`）。2 GiB fp16 → 18,724 token → 1,170 块。我第一次压测按 64×2048 预留直接 OOM（14 GiB），正好说明了 naive 的问题。

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

---

## Stage 3 — 连续批处理调度器（Continuous batching scheduler）

### 读代码顺序
`ember/scheduler.py` → `ember/engine.py` → `tests/test_scheduler.py` → `benchmarks/micro_gather.py` → `benchmarks/bench_decode_batch.py` → `benchmarks/bench_batching.py`

### 1. `ember/scheduler.py`（核心）
**(a) 骨架**：`SchedulerConfig`（`max_num_seqs` 并发上限、`max_num_batched_tokens` 每步 token 预算、`chunked_prefill`、`policy`）；`Scheduler` 持有 `waiting`（deque）和 `running`（list）两个队列；`schedule()` 每步从零决定这一步跑谁、每个跑几个 token，分两段：① 先排 running（decode 1 个 token，或未完成 prefill 的下一块）② 剩余预算按 FCFS 接纳 waiting。`_preempt()` 抢占；`finish()` 结束并释放块。
**(b) 为什么**：
- **迭代级调度（iteration-level scheduling）**：每生成一个 token 就重新调度一次，序列一结束马上空出位置给新请求，不用等整批最长的那个——这就是「连续」批处理（Orca 论文的思想）。静态批处理（static batching）要等最长的序列跑完，短序列结束后它的槽位一直空着。
- **decode 优先**：running 先排，保证已经在出字的请求的 inter-token latency（ITL）稳定；新 prompt 用剩余预算。
- **token 预算 + 分块 prefill**：一个 2000 token 的 prompt 如果一步算完，这一步所有人的 decode 都要等它；切成块后每步最多 `max_num_batched_tokens` 个 token，把卡顿摊开。代价是长 prompt 的 TTFT 会变长一点。
- **抢占选最新到达的**：先来先服务的公平性；被抢占的放回 waiting **队首**，优先恢复。
- **重算（recompute）而不是换出（swap）**：vLLM 支持把 KV 换到 CPU 内存。在统一内存的 Mac 上「换到 CPU」没有意义（同一块物理内存），而 prefill 是计算密集、很快的，所以重算更简单划算。把 `num_computed_tokens` 归零，下次把 prompt+已生成的 token 当成新 prompt 重新 prefill。
- **抢占的那一步不接纳新请求**：内存已经紧张，再接纳只会抖动（thrashing）。
- `policy="static"` 用同一套代码做出静态批处理基线，保证对比只差调度策略（同样的 kernel、同样的模型）。
**(c) 语法**：`deque.appendleft` 放回队首；`@dataclass(frozen=True)` + `__post_init__` 做参数校验。

### 2. `ember/engine.py`
**(a) 骨架**：`LLMEngine.step()` = `schedule()` → `runner.execute()` → `sample()` → 更新每个序列（`num_computed_tokens += n`，需要的话 append 新 token、判断停止条件、`finish` 释放）。`add_request` / `abort` / `generate`（离线批量接口）。
**(b) 为什么**：引擎是**同步、单线程**的，不知道 HTTP 的存在；Stage 5 的服务器在一个单独线程里循环调用 `step()`。这样引擎可以独立测试，也不会被 asyncio 事件循环卡住（GPU 前向是阻塞调用）。
**(c) 语法**：`itertools.count()` 生成自增 id；`iter()` + `next()` 按顺序消费采样结果。

### 3. 性能排查：`benchmarks/micro_gather.py`
第一次跑批处理时 B=32 每步 102 ms，远高于计算量的预期。逐项计时发现 81 ms 花在 K/V gather 上：MPS 上高级索引 `pool[slot_ids]`（2D 索引张量）只有约 7 GB/s。换成在 `[num_blocks, bs*kv*d]` 视图上按**块**做 `index_select`，同样的拷贝从 81 ms 降到 12.7 ms（约 6.4 倍）；B=32 decode 吞吐 311.8 → 517.8 tok/s。

### Stage 3 实测（`results/stage3_batching.json`，48 请求、输出长度 U[16,256]、3 次重复）
continuous B=32 **377.3 tok/s**，static B=32 308.2，sequential 41.4（9.1 倍）；p99 e2e 18.9 s / 23.1 s / 171.1 s；TPOT 随 batch 从 24 ms 涨到 62 ms。KV 池只给 160 块时抢占 34 次，全部完成，吞吐降 26%。
fp16 下 38–43/48 个输出和顺序执行逐 token 相同（batch 形状改变浮点舍入）；fp32 完全相同。

### Stage 3 面试题
1. **Q：continuous batching 和 static batching 的本质区别？什么时候差距最大？**
   A：调度粒度——static 以「请求批」为单位，continuous 以「一次迭代（一个 token）」为单位。输出长度方差越大差距越大：static 批里短请求结束后槽位空转直到最长请求结束。
2. **Q：吞吐（throughput）和延迟（TTFT/TPOT）怎么权衡？**
   A：批越大，每步服务的序列越多，吞吐上升；但每步更慢，TPOT 变大；排队的请求要等槽位，TTFT 变大。`max_num_seqs` 和 `max_num_batched_tokens` 就是这个旋钮。
3. **Q：为什么 decode 优先而不是 prefill 优先？**
   A：prefill 优先（老版 vLLM）能更快接纳新请求、TTFT 好，但 prefill 步会让所有正在 decode 的请求停顿，ITL 抖动。decode 优先 + 分块 prefill 让 decode 每步都有份，prefill 用剩余预算。
4. **Q：KV 块用完了怎么办？**
   A：抢占最近到达的 running 序列：释放它的全部块，`num_computed_tokens=0`，放回 waiting 队首，之后重算。代码 `Scheduler._preempt`；测试 `tight_memory` 场景断言发生了抢占且输出与参考逐 token 相同。
5. **Q：重算和换出（swap）怎么选？**
   A：swap 需要 CPU 内存和 PCIe 带宽，适合 prompt 很长、重算很贵的情况；重算只花 FLOPs，prefill 是计算密集，在 GPU 上很快。M 系列芯片 CPU/GPU 共享内存，swap 没有意义。
6. **Q：分块 prefill 的块大小（token 预算）怎么选？**
   A：太大：长 prompt 一步占满，其他请求的 ITL 尖峰；太小：prefill 拆成很多步，每步固定开销（kernel launch、权重读取）被重复支付，TTFT 变长、吞吐下降。
7. **Q：为什么批处理能提升吞吐？瓶颈在哪？**
   A：decode 时每步都要读一遍全部权重（~1.2 GB），batch=1 时这次读取只服务 1 个 token；batch=B 时服务 B 个，权重读取和 kernel launch 开销被摊薄。实测见 BENCHMARK.md 的 decode 吞吐 vs batch。批再大以后，注意力的 K/V 读取（随 batch × 上下文增长）和 logits 投影成为主要开销。
8. **Q：你怎么发现并解决性能瓶颈的？**
   A：先算理论值（计算量/访存量），发现实测离谱 → 逐个算子计时（`torch.mps.synchronize()` 保证计时准确）→ 定位到 gather 占 2/3 → 写微基准比较 4 种 gather 实现（结果逐位相同）→ 换成 block 级 `index_select`。有前后对比数据（`results/stage3_decode_batch_before_gather_fix.json` vs `results/stage3_decode_batch.json`）。

---

## Stage 4 — 前缀缓存（Prefix caching）

### 读代码顺序
`ember/block_manager.py`（后半部分：`match_prefix` / `cache_full_blocks` / `_take_free_block` / `_release`）→ `ember/scheduler.py`（admission 段）→ `ember/engine.py`（`step` 里的 `cache_full_blocks`）→ `tests/test_prefix_cache.py` → `benchmarks/bench_prefix.py`

### 1. `ember/block_manager.py`（前缀缓存部分）
**(a) 骨架**：每个 `Block` 多了 `block_hash` 和 `token_ids`；`cached: dict[hash → block_id]`；空闲链表从 deque 换成 `OrderedDict`（可以 O(1) 删除中间元素）。`match_prefix(seq)`：新请求到达时，逐个满块算链式哈希去查；`cache_full_blocks(seq)`：每步之后，把新写满的块登记进缓存；`_take_free_block`：从空闲链表头取块，如果它带着缓存内容就驱逐（evict）。
**(b) 为什么**：
- **链式哈希** `h_i = hash((h_{i-1}, 本块 tokens))`：一个 token 的 K/V 取决于它前面所有 token，所以两个块「内容相同」必须连前缀也相同。链式哈希把整个前缀编码进去。
- **存 token_ids 做二次校验**：哈希碰撞时也不会读到错误的 K/V（测试 `test_hash_collision_is_rejected_by_token_check`）。
- **只在 K/V 真正写入之后登记**（step 之后调 `cache_full_blocks`）：否则同一步里另一个序列可能命中一个还没写的块。
- **至少留 1 个 token 不命中**：prompt 完全命中时也得算最后一个 token 的 logits 才能采样（`test_never_matches_the_final_token`）。
- **空闲链表 = LRU 驱逐队列**：ref_count 归零的块带着哈希回到链表尾部（最近使用），分配从头部取（最久未用）；命中一个空闲的缓存块就把它从链表里「捞回来」。没有缓存内容的空块放到**头部**优先复用——这是写测试时发现的：纯按释放时间排序会先驱逐有价值的缓存块。
- **ref count 共享 + 不需要 CoW**：只共享满块，满块不会再被写。
- 和抢占的协同：被抢占的序列释放块后，块还在缓存里，重算时大概率命中，抢占代价变小。
**(c) 语法**：`OrderedDict.popitem(last=False)` 弹出最早插入的；`move_to_end(k, last=False)` 移到头部；`hash((a, tuple))` 对 int 元组是确定性的（不受 PYTHONHASHSEED 影响，字符串才会随机化）。

### 2. scheduler 的接纳（admission）段
**(a) 骨架**：`match_prefix` → `num_computed_tokens = 命中数` → 只给剩余部分分配块 → 失败就 `_undo_prefix_match` 回滚。
**(b) 为什么**：命中的块可能本来在空闲链表里（ref=0），拿走它们会减少空闲块数，所以必须「先占住命中块，再检查剩余能否分配」，不行就回滚，保证没有副作用。

### Stage 4 实测（`results/stage4_prefix_cache.json`）
（最终代码重跑：命中 TTFT 359.8 → 39.6 ms；批处理 63.1 → 153.5 tok/s，2.4 倍——因为 Stage 6 的分组注意力修复让「不开缓存」的 prefill 也变快了，相对收益变小。面试时用最终数字。）
587-token 共享 system prompt + 32 个不同问题：命中请求的 TTFT **361.5 → 39.5 ms（9.2 倍）**，93% 的 prompt token 来自缓存；16 并发批处理吞吐 **48.6 → 151.0 tok/s（3.1 倍）**。批处理时命中率只有 84%：前 16 个请求在同一步被接纳，那时前缀块还没写完、不能共享。fp16 下缓存开/关 greedy 一致 29–31/32。

### Stage 4 面试题
1. **Q：prefix caching 解决什么问题？适用什么场景？**
   A：多个请求共享相同前缀（system prompt、few-shot 示例、多轮对话历史）时，前缀的 K/V 只算一次。直接减少 prefill 计算 → TTFT 下降、吞吐上升。
2. **Q：为什么用哈希而不是前缀树（radix tree，SGLang 的做法）？**
   A：块级哈希实现简单、查找 O(块数)、和分页天然对齐（vLLM 的做法）；radix tree 能做 token 级匹配、更适合多轮对话树状分叉，但实现和驱逐更复杂。代价：我们只能按整块命中，最后不满的块永远要重算。
3. **Q：驱逐策略？为什么 LRU？**
   A：LRU 近似「最近被用过的前缀更可能再被用」（热门 system prompt）。实现上复用空闲链表，不需要额外数据结构。空块优先复用。
4. **Q：怎么保证缓存开/关输出一致？**
   A：tiny 模型 fp32 上，缓存开/关输出逐 token 相同（包括小 KV 池、发生驱逐的情况，`test_engine_outputs_identical_with_cache_on_and_off`）。真模型 MPS fp16 上，命中时 prefill 只算后缀，矩阵形状不同导致浮点累加顺序不同，greedy 可能在接近平局处分叉——实测一致率见 BENCHMARK.md，诚实报告。
5. **Q：什么时候命中率高但收益小？**
   A：输出很长、prefill 占比小时，省下的 prefill 在总时间里占比低；或者 prefill 本来就很快（短 prompt）。我们的基准是 587-token system prompt + 32 token 输出，prefill 占比大，所以收益明显。
6. **Q：哈希碰撞怎么办？**
   A：块里存了 token_ids，命中时逐个比较；碰撞就当未命中。
7. **Q：多个请求同时写同一个前缀会怎样？**
   A：同一步里两个请求都 miss，各自计算，第一个登记成功，第二个发现哈希已存在就不登记（它的块保持私有）。不会出错，只是少共享一次。
8. **Q：前缀缓存和 KV 内存的冲突？**
   A：缓存块只占「空闲」块，分配时随时可以被驱逐，所以不会降低可用容量——缓存是免费的，只要有空闲块就留着。

---

## Stage 5 — 服务层（Serving API）

### 读代码顺序
`ember/server/protocol.py` → `ember/server/metrics.py` → `ember/server/async_engine.py` → `ember/server/api.py` → `ember/server/__main__.py` → `tests/tiny_tokenizer.py` → `tests/test_server.py`

### 1. `ember/server/async_engine.py`（核心）
**(a) 骨架**：`AsyncLLMEngine` = 一个**引擎线程**（`_loop`：收件箱 → `engine.step()` → 把每个 `StepOutput` 投递到对应请求的 `asyncio.Queue`）+ 事件循环这边的 `generate()` 异步生成器（发 add 命令 → 等自己的队列 → 增量反分词、停止串检查 → `yield RequestOutput`）。
**(b) 为什么**：
- **GPU 前向是阻塞调用**，放在事件循环里会卡住所有 HTTP 连接。所以引擎独占一个线程；**只有这个线程碰引擎**，引擎本身不需要锁。
- 线程 → asyncio 只能用 `loop.call_soon_threadsafe(queue.put_nowait, item)`；asyncio 对象不是线程安全的。
- 反分词和停止串在事件循环里做，让引擎线程的 step 循环尽量紧凑。
- **背压（backpressure）**：`inflight >= max_inflight` 直接抛 `QueueFullError` → 429。无界排队只会让所有人的延迟一起爆炸，早拒绝让客户端可以重试/换实例。
- **取消**：`generate` 的 `finally` 只要没正常结束（断连、超时、停止串、异常）就发 abort，序列的 KV 块在下一步就被释放。
- **停止串**：流式输出时不能把停止串的一部分先发出去——`_stop_holdback` 计算文本末尾有多长是某个停止串的前缀，这部分先扣住。
- **超时**：用 `asyncio.wait_for` 加剩余时间，超时抛 `RequestTimeoutError` → 504。
**(c) 语法**：异步生成器（`async def` + `yield`）的 `finally` 在 `aclose()` / 被取消 / 正常结束时都会执行；`queue.Queue`（线程安全）vs `asyncio.Queue`（协程安全）。

### 2. `ember/server/api.py`
**(a) 骨架**：`create_app()` 返回 FastAPI app：`/v1/completions`、`/v1/chat/completions`、`/v1/models`、`/health`、`/metrics`。`run()` 是两个生成接口共用的骨架；`_stream_body` 生成 SSE（`data: {...}\n\n`，最后 `data: [DONE]`）。lifespan 负责启动/关闭引擎线程。
**(b) 为什么**：
- 流式请求**先拿到第一个输出再返回响应头**，这样 429/400 能作为真正的 HTTP 状态码返回（响应头发出后就只能在流里报错了）。代价：响应头在 TTFT 之后才到。
- 断连检测：流式由 Starlette 的 `StreamingResponse` 自己监听 `http.disconnect` 并取消生成器；非流式用一个轮询 `request.is_disconnected()` 的 watcher 任务。（我一开始流式也加了 watcher，调试时发现问题其实出在测试里：`next(resp.iter_lines())` 丢弃迭代器会关闭连接——服务端正确地把它当成断连 abort 了。）
- 错误映射：429 背压、504 超时、503 引擎挂了、400 参数错误、422 schema 校验（FastAPI 自动）。
**(c) 语法**：`@asynccontextmanager` 写 lifespan；`StreamingResponse(async_gen, media_type="text/event-stream")`。

### 3. `ember/server/metrics.py`
TTFT、ITL（相邻 token 间隔）、TPOT（首 token 之后平均每 token 时间）、e2e 用 Histogram；队列深度、running 数、in-flight、KV 使用率、前缀命中率用 Gauge；请求数按 finish_reason 分的 Counter。每个 app 一个独立 `CollectorRegistry`，测试里可以起多个 server。

### 4. 测试为什么起真的 uvicorn
in-process ASGI transport 模拟不出「客户端关闭 TCP 连接」。`tests/test_server.py` 在线程里起 uvicorn、随机端口，用 tiny 随机模型 + `tests/tiny_tokenizer.py` 造的 512 词表字节级 tokenizer，CI 不需要下载任何东西。

### Stage 5 面试题
1. **Q：为什么引擎要单独一个线程，而不是直接在 async handler 里调用？**
   A：`engine.step()` 是同步阻塞的 GPU 调用（几十毫秒），在事件循环里跑会阻塞所有连接的读写。单独线程 + 消息传递，事件循环永远不阻塞；而且只有一个线程碰引擎，不需要锁。
2. **Q：TTFT、TPOT、ITL、e2e 的区别？**
   A：TTFT = 发请求到第一个 token（含排队 + prefill）；ITL = 相邻两个 token 的间隔；TPOT = (e2e − TTFT)/(输出数 − 1)，即 ITL 的平均；e2e = 整个请求。交互场景关心 TTFT 和 TPOT，离线批处理关心吞吐。
3. **Q：客户端断开后会发生什么？**
   A：Starlette 取消响应生成器 → `pump` 任务取消 → `generate()` 的 finally 发 abort → 引擎线程在下一轮收件箱处理时 `engine.abort` 释放块。测试 `test_client_disconnect_aborts_sequence` 断言序列被移除、已用块数归零、abort 计数增加。
4. **Q：为什么要背压？429 和排队怎么选？**
   A：超过容量后排队只会让 TTFT 无限增长（开环负载下队列线性增长）；429 让上游负载均衡器/客户端去重试或换实例。阈值 `max_inflight` 应该按容量（见 docs/CAPACITY.md）设定。
5. **Q：SSE 是什么？为什么不用 WebSocket？**
   A：Server-Sent Events，基于普通 HTTP 响应的单向流（`text/event-stream`，`data: ...\n\n`）。OpenAI API 用它；简单、兼容代理和负载均衡；生成是单向的，不需要 WebSocket 的双向通道。
6. **Q：停止串在流式输出里有什么坑？**
   A：停止串可能跨多个 token 出现，如果每个 token 就立刻发，停止串的前半截已经发给客户端了。解决：扣住「可能是停止串前缀」的尾部文本（`_stop_holdback`），确认不是再发；命中后截断并 abort 序列。
7. **Q：结构化日志记什么？**
   A：每个请求结束一行 JSON：request_id、status、finish_reason、prompt/completion token 数、TTFT、e2e。可以直接被日志系统解析聚合。
8. **Q：这个服务器的扩展性瓶颈在哪？**
   A：单进程单 GPU；引擎线程和 Python GIL：HTTP 处理、反分词和引擎调度共享 GIL，高并发时 CPU 侧开销会和 step 抢 GIL。生产系统（vLLM）把 API server 和引擎拆成不同进程用 ZMQ 通信，就是为了这个。

---

## Stage 6 — 压测与基准（Load testing & benchmarks）

### 读代码顺序
`benchmarks/common.py` → `benchmarks/loadgen.py` → `benchmarks/hf_server.py` → `benchmarks/bench_serving.py` → `BENCHMARK.md` Stage 6 节

### 1. `benchmarks/loadgen.py`
**(a) 骨架**：`make_workload`（种子固定的 log-normal 输入/输出长度）→ `send`（一个流式请求，记录首个非空文本块的时间 = TTFT、结束时间、`usage` 里的真实 token 数）→ `open_loop`（泊松到达）/ `closed_loop`（C 个用户轮流发）→ `summarize_run`（吞吐 + TTFT/TPOT/e2e 的 p50/p90/p99）。
**(b) 为什么**：
- **开环 vs 闭环（open-loop vs closed-loop）**：闭环负载会「自动减速」——服务器慢了，用户发得也慢，永远不会过载，所以测不出排队；开环（泊松到达，间隔 ~ Exp(1/λ)）不管服务器快慢照样到达，过了容量队列就线性增长，TTFT 爆炸——这才是真实流量的样子，也是找「拐点（knee）」的方法。
- **泊松过程**：大量独立用户的到达近似泊松；指数分布间隔。
- **`ignore_eos` + 用 `usage` 计数**：控制输出长度，各后端可比；不支持 ignore_eos 的后端（Ollama）会提前停，所以永远用服务器报告的真实 token 数算吞吐。
- **合成数据要写清楚来源**：没有下载 ShareGPT，用 log-normal 并写明参数。
**(c) 语法**：`asyncio.gather` 并发等待；`httpx.AsyncClient.stream` + `aiter_lines()` 读 SSE。

### 2. `benchmarks/hf_server.py`
朴素基线：`transformers.generate` + `TextIteratorStreamer` + 全局 `asyncio.Lock`（一次一个请求）。这就是「随手写的服务」会有的样子，对比它能看出调度 + 批处理带来的提升。

### 3. `benchmarks/bench_serving.py`
起服务器子进程 → 等 `/health` → 预热 → 每个到达率跑一次（每点请求数 = max(30, λ × 40)，保证高负载下能到达稳态）→ 每次之前 `wait_for_quiet` 负载门控 → 重复 3 次 → 关服务器。

### Stage 6 实测（`results/stage6_serving_*.json`，3 次重复）
- ember 拐点在 λ=2 到 3 req/s 之间：λ=2 时 TTFT p99 0.47 s，λ=3 时 15.5 s；吞吐平台 ~201 tok/s（~1.74 req/s）。
- 朴素 HF 服务器峰值 54.1 tok/s → ember **3.7 倍**；Ollama（4-bit Q4_K_M + llama.cpp Metal kernel）217.3 tok/s，ember 达到其 ~93%，但单用户 TPOT Ollama 9 ms vs ember 26 ms。
- **这个压测抓到一个真 bug**：第一次扫描只有 ~128 tok/s。原因是 prefill 块和 decode 混在一个 SDPA 调用里，decode 行被 padding 到 prefill 块长度（1 个 1024 块 + 31 个 decode ≈ 32 倍注意力计算）。改成 decode 行和 prefill 行分组各自 padding 后，混合步从 4479.9 ms 降到 687.5 ms。这也是题目要求「prefill/decode 分开」的真正原因。

### Stage 6 面试题
1. **Q：怎么找到服务的饱和点（knee）？**
   A：开环扫到达率 λ，画吞吐和 TTFT 对 λ 的曲线：吞吐随 λ 线性增长直到容量上限后变平；TTFT 在接近上限时开始急剧上升（排队）。两者拐的地方就是 knee，生产上把 SLO 设在 knee 之前。
2. **Q：为什么不能只用闭环测？**
   A：闭环中服务器越慢请求越少（coordinated omission 的一种），会严重低估过载时的延迟。
3. **Q：p99 为什么比均值重要？**
   A：用户体验和 SLO 都看尾部；批处理系统里少数被长 prefill 卡住、被抢占、排队久的请求决定 p99。
4. **Q：TPOT 随负载怎么变？为什么？**
   A：负载越高，running batch 越大，每步要服务的序列越多，每步变慢 → TPOT 上升；这是用延迟换吞吐。
5. **Q：和 HF generate 基线比，提升来自哪里？**
   A：（1）连续批处理让一次权重读取服务多个序列；（2）分页 KV 让并发数不受预分配限制；（3）调度器在请求结束的那一步就补位。HF 基线一次一个请求，过载时 TTFT 全是排队时间。
6. **Q：你的 benchmark 有哪些局限？**
   A：单机单卡 MPS、localhost（没有网络延迟）；合成长度分布不是真实 ShareGPT；随机 token 拼的 prompt 没有语义；机器上同时有别的进程（所以记录 load、门控、重复 3 次）；Ollama 用的是量化 GGUF 权重和 llama.cpp 的 Metal kernel，不是同精度对比。
7. **Q：怎么保证结果可复现、可信？**
   A：固定随机种子、记录机器/日期/负载、原始 JSON 全部入库、每个关键点重复 3 次报告变异系数、命令写在 BENCHMARK.md；数字只来自实际运行。
8. **Q：讲一个你通过 benchmark 发现的问题。**
   A：见上：开环压测吞吐远低于离线 decode 吞吐 → 用 micro benchmark 复现单个混合步 → 定位到 decode 行被 pad 到 prefill 长度 → 分组 attention → 4479.9 → 687.5 ms，拐点从 λ<2 移到 2~3 req/s。前后日志都在 `results/`。
9. **Q：Little's law 在这里怎么用？**
   A：L = λ × W：系统中平均在途请求数 = 到达率 × 平均停留时间。比如已知容量下的 e2e，可以估算需要多大的 `max_num_seqs` / `max_inflight`（见 docs/CAPACITY.md）。

---

## Stage 7 — 采样与 int8 量化（Sampling & weight-only int8）

### 读代码顺序
`ember/sampling.py` → `ember/engine.py`（`add_request` 里的 generator）→ `ember/quant.py` → `tests/test_sampling.py` → `tests/test_quant.py` → `benchmarks/bench_quality.py`

### 1. `ember/sampling.py`
**(a) 骨架**：`SamplingParams`（temperature / top_p / top_k / seed / max_tokens / ignore_eos + 校验）；`apply_top_k_top_p`（排序 → 按排名去掉 top-k 之外 → 在剩下的里面算 softmax、累加 → 去掉「排在它前面的概率和已经 ≥ p」的 token → scatter 回原顺序）；`sample`（greedy 行直接 argmax；其余行在设备上批量做温度/top-k/top-p，再每行从自己的 `torch.Generator` 取一个均匀随机数 u，在设备上批量 `cumsum` + `searchsorted` 做逆 CDF 采样）。
**(b) 为什么**：
- 顺序：温度 → top-k → top-p（和 HF 一致）。top-p 用「前面的累计概率」判断，保证概率最高的 token 永远保留。
- **每个请求自己的随机数生成器**：带 seed 的请求输出只取决于自己的 logits 和自己的随机流，和同 batch 里有谁无关；generator 存在 `Sequence` 上，被抢占重算时**不会重置**（重算只重建 KV，不重新采样已生成的 token）。测试 `test_seeded_request_is_independent_of_batch_and_preemption` 验证这两点。
- **逆 CDF 采样**：第一版在 CPU 上逐行 `torch.multinomial`（151,936 词表），batch 32 时采样一步 77 ms 里占 54 ms，top-p 吞吐只有 greedy 的 44%；改成每行只取一个随机数、其余在设备上批量完成后，22 ms，吞吐 235.8 → 398.5 tok/s。u 取 (0,1]，保证概率为 0（被 mask）的 token 永远选不到。
- 排序整个 151,936 词表是有代价的，所以只对非 greedy 且设置了 top-k/top-p 的行做；吞吐代价见 BENCHMARK.md。
- 停止串（stop sequences）在服务层基于文本实现（Stage 5），max_tokens 在引擎里。
**(c) 语法**：`torch.searchsorted(cdf, u)` 返回第一个 ≥ u 的位置；`torch.rand((), generator=g)` 用指定生成器取一个标量；`tensor.scatter_(dim, index, src)` 把排序后的值放回原位置。

### 2. `ember/quant.py`
**(a) 骨架**：`Int8Linear`：按输出通道对称量化，`scale = max|W_row| / 127`，`Wq = round(W/scale)`；前向调用 `torch._weight_int8pack_mm(x, Wq, scale)`（kernel 内部反量化）。`quantize_int8(model)` 把 28 层里的 4 个投影全部替换。
**(b) 为什么**：
- **W8A16（weight-only）**：小 batch decode 受权重读取带宽限制，权重字节数减半理论上能加速；激活保持 fp16，精度损失小。
- **per-channel 对称**：每行一个 scale，比整个矩阵一个 scale 精度高得多；对称（无 zero-point）让 kernel 简单。
- embedding（和 LM head 共享）、norm、KV cache 不量化。
- 测试：逐元素误差 ≤ scale/2（舍入误差上界）；tiny 模型 logits 相对误差 < 5%。
**(c) 语法**：`torch._weight_int8pack_mm` 是 PyTorch 私有 op（下划线开头），MPS 上有实现，是 torchchat 用的那个。

### Stage 7 实测（`results/stage7_quality_speed.json`）
困惑度 fp32 11.2848 / fp16 11.2859 / int8 11.3000（+0.13%）；权重 1.19 → 0.75 GB（−37%，embedding 保持 fp16）；decode batch 1：42.4 → 51.0 tok/s（+20%），batch 32 无提升（538 vs 532）。

### Stage 7 面试题
1. **Q：temperature、top-k、top-p 分别做什么？**
   A：temperature 缩放 logits（<1 更尖锐、>1 更平）；top-k 只保留概率最高的 k 个；top-p（nucleus）保留累计概率达到 p 的最小集合，候选集大小随分布自适应。
2. **Q：怎么做到「固定 seed 结果可复现」？在批处理里有什么坑？**
   A：每个请求一个独立的、用 seed 初始化的 generator，只被自己的采样推进；如果用全局 RNG，同 batch 里别的请求会消耗随机数，结果就取决于 batch 组成。剩下的坑是 logits 本身：fp16 下 batch 形状不同 → logits 有微小差异 → 可能采到不同 token（Stage 3 的 batch invariance 问题），fp32 下完全可复现。
3. **Q：抢占重算会不会改变采样结果？**
   A：不会。重算只重建 KV；已生成的 token 保留在 `token_ids` 里，generator 状态保留，接着往下采。测试覆盖。
4. **Q：weight-only 量化为什么能加速 decode？什么时候不加速？**
   A：decode 小 batch 时每步要把全部权重读一遍，是访存瓶颈，权重字节减半 → 读取时间减半。batch 大或 prefill 时变成计算瓶颈，反量化反而增加开销；在我们的 eager MPS 上 batch=1 还受 kernel launch 限制。实测见 BENCHMARK.md（如实报告，包括没有加速的情况）。
5. **Q：怎么评估量化的精度损失？**
   A：固定文本上的困惑度（perplexity = exp(平均负对数似然)）：fp32 参考、fp16、int8 三者对比。语料是 CPython 自带的 `pydoc_data.topics` 文档文本，本地、固定、可复现；不是标准 benchmark，只做相对比较。
6. **Q：per-tensor、per-channel、per-group 量化的区别？**
   A：一个 scale 管的元素越少，越能适应局部的数值范围、误差越小，但 scale 存储和计算开销越大。per-group（如每 128 个元素一组，GPTQ/AWQ 常用）在 4-bit 时几乎必需；8-bit per-channel 通常就够了。
7. **Q：为什么不做 speculative decoding？**
   A：Qwen3 没有更小的同词表 draft 模型可用（0.6B 已经是最小的）；n-gram/prompt-lookup 的投机解码在分页缓存上需要回滚 KV slot，时间有限下选了能完整做完、能量化评估的 int8。这是明确的取舍。
8. **Q：top-p 实现里为什么用「之前的累计概率」而不是「包括自己的累计概率」？**
   A：用包括自己的累计 ≤ p 会把「跨过 p 的那个 token」也去掉，可能一个都不剩；用之前的累计 < p 保留它，保证集合的总概率 ≥ p 且至少保留 1 个 token。测试用暴力参考实现做了 200 个随机例子的对比。

---

## Stage 8 — 工程化（Ops polish）

### 读代码顺序
`Dockerfile` → `.dockerignore` → `pyproject.toml`（`[tool.uv.sources]`、ruff/mypy/pytest 配置）→ `.github/workflows/ci.yml` → `tests/conftest.py`（`real_model` 自动跳过）→ `docs/ARCHITECTURE.md` → `docs/CAPACITY.md`

### 1. `Dockerfile`
**(a) 骨架**：`python:3.15-slim` → 拷贝 uv → 先只拷 `pyproject.toml`/`uv.lock` 装依赖（缓存层）→ 再拷代码 → 非 root 用户 → HEALTHCHECK 打 `/health` → ENTRYPOINT 启动 CPU 服务。
**(b) 为什么**：
- **依赖层和代码层分开**：改代码不会重装 torch，构建快。
- **Linux 上 torch 走 PyTorch CPU 索引**（`[tool.uv.sources]` 带 `sys_platform == 'linux'` marker）：PyPI 上 Linux 版 torch 会拉几个 GB 的 CUDA 包；改完后 lock 文件里 nvidia 包数为 0，镜像内容 539 MB。macOS 仍用 PyPI 的 MPS 版。
- **模型不打进镜像**：挂载只读的 HF 缓存（`-v ~/.cache/huggingface:/models:ro`，`HF_HUB_CACHE=/models/hub`），镜像小、权重和代码分开发布。我第一次写的挂载路径是 `/root/...`，但容器里是非 root 用户——跑起来才发现，改成显式的 `HF_HUB_CACHE`。
- 实测：在 colima 里构建（1 分 44 秒）并跑通了一次真实模型的 chat 请求（`results/stage8_docker_*`）。
**(c) 语法**：`COPY --from=ghcr.io/astral-sh/uv:0.12.15 /uv /usr/local/bin/uv` 是多阶段拷贝二进制；`uv sync --frozen --no-dev --no-install-project` 只装依赖不装项目本身。

### 2. CI（`.github/workflows/ci.yml`）
ruff check → ruff format --check → mypy（strict）→ pytest。CI 没有模型：`tests/conftest.py` 在 HF 缓存里找不到 Qwen3-0.6B 时，给所有 `@pytest.mark.real_model` 的测试加 skip；其余测试用 tiny 随机权重模型 + 生成的 512 词表 tokenizer，CPU 上 6 秒跑完。本机用空的 `HF_HUB_CACHE` 模拟了 CI：64 passed / 6 skipped；有模型时 70 passed（`results/stage8_ci_local.txt`）。没有远端仓库，所以 CI 本身没真正跑过——这一点要诚实说。

### 3. 严格类型检查的取舍
`mypy --strict` 全开，只关了 `warn_return_any`（torch 的 `nn.Module.__call__` 返回 `Any`，每个 `return self.proj(x)` 都会报）；对没有类型的第三方库（safetensors、transformers）在 loader/tests/benchmarks 里允许 untyped call。

### Stage 8 面试题
1. **Q：为什么镜像里不放模型权重？**
   A：权重 1.2 GB、更新节奏和代码不同；挂载卷/对象存储让同一个镜像服务不同模型版本，镜像拉取更快。代价是部署时要保证卷存在（HEALTHCHECK 的 start-period 留了 120 s 给加载）。
2. **Q：CI 没有 GPU、没有模型，怎么保证测试有意义？**
   A：核心逻辑（block manager、调度器、分页注意力、前缀缓存、采样、HTTP 层）都用 tiny 随机模型在 fp32 CPU 上测「和参考实现逐 token 相同」，不依赖真实权重；真实模型的数值对齐测试本地跑，CI 自动跳过。
3. **Q：怎么保证测试不 flaky？**
   A：所有随机性都有种子（`torch.manual_seed` autouse fixture、`random.Random(seed)`、hypothesis `derandomize=True`）；服务器测试用随机空闲端口；等待用有超时的轮询而不是固定 sleep。最后把整个测试集连续跑 10 次，结果存在 `results/stage8_test_10x.txt`。
4. **Q：容量规划怎么做？**
   A：见 `docs/CAPACITY.md`：KV 每 token 112 KiB → 2 GiB 放 18,724 token；实测服务拐点 ~2 req/s；用 Little's law（L = λW）估在途请求数来设 `--max-inflight`；chunk 大小在 ITL 平滑和 prefill 效率之间取舍（1024 块的混合步 687.5 ms vs 128 块 129.7 ms）。
5. **Q：如果要上生产，下一步做什么？**
   A：API server 和引擎拆进程（避开 GIL）；CUDA 上用 FlashAttention/FlashInfer 的 paged kernel 和 CUDA graphs；多副本 + 负载均衡按 KV 使用率/队列深度路由；按 Prometheus 指标做告警（TTFT p99、队列深度、拒绝率）；请求级限流。
6. **Q：Prometheus 指标里最重要的几个？**
   A：TTFT p99（用户体验）、队列深度和 429 拒绝数（过载信号）、KV 使用率和抢占次数（内存压力）、step 时长（引擎健康）、前缀命中率（缓存是否有效）。

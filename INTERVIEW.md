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

# Progress log

## 2026-11-14 — Stage 1: model from scratch
- Implemented Qwen3 (RMSNorm, RoPE, GQA with per-head q/k RMSNorm, SwiGLU, tied embeddings) in plain PyTorch,
  loading `model.safetensors` directly (fused qkv / gate-up at load time). No `transformers` in `ember/`.
- Attention is behind an `AttentionBackend` protocol (no-cache, contiguous cache now; paged later).
- Verified: per-layer and logits diff vs HF = 0.0 (fp32 & bf16 CPU); greedy 8×64 tokens identical in fp32 CPU and
  fp16 MPS. Found bf16 on MPS diverges (3/8 vs fp32 reference, 6/8 vs HF bf16 MPS) → fp16 default on MPS.
- Measured single-request decode: MPS fp16 45.9 tok/s, CPU bf16 35.3, CPU fp32 18.6.
- Problem: first benchmark run had CPU rows with load1 > 4 (another job on the machine) → reran with the gate;
  first run kept as `results/stage1_single_request.run1.json`.

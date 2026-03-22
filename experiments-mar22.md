# autoresearch/mar22 — Experiment Log

**Platform:** NVIDIA H100 80GB
**Baseline val_bpb:** 1.041946 (50.3M params, 44GB VRAM, 25.9% MFU)
**Best val_bpb:** 1.009015 (**-0.033**, same params/VRAM)

## Results

| # | val_bpb | VRAM GB | Status | Change |
|---|---------|---------|--------|--------|
| 1 | 1.0419 | 44.0 | **keep** | Baseline |
| 2 | OOM | — | crash | Depth 8→12 (n_embd 768) |
| 3 | 1.0681 | 33.8 | discard | Depth 8→10 + batch 64 |
| 4 | 1.0807 | 66.5 | discard | Depth 8→10 + batch 128 |
| 5 | 1.0609 | 44.0 | discard | SiLU activation |
| 6 | 1.0546 | 44.5 | discard | SwiGLU MLP |
| 7 | 1.0470 | 44.0 | discard | Warmdown 0.5→0.25 |
| 8 | **1.0090** | 43.9 | **keep** | **Batch 2^19→2^18 (no grad accum)** |
| 9 | 1.0113 | 22.2 | discard | Batch 2^17 + batch 64 |
| 10 | 1.0583 | 19.3 | discard | Grad checkpoint + depth 12 |
| 11 | 1.0099 | 43.9 | discard | Matrix LR 0.04→0.05 |
| 12 | 1.0119 | 44.9 | discard | Full attention (SSSL→L) |

## Key Insight

**More optimizer updates >> bigger models** in a 5-minute fixed budget.

Halving `TOTAL_BATCH_SIZE` from 2^19 to 2^18 eliminated gradient accumulation (grad_accum 2→1), doubling optimizer steps from ~625 to ~1232 while processing the same total tokens. This single change gave a **3.2% relative improvement** — larger than any architecture or activation change.

## What Didn't Work and Why

- **Scaling depth:** Larger models are slower per step → fewer tokens/updates in 5 min → net loss. Even with gradient checkpointing (which saved memory but added recomputation overhead), only 388 steps fit.
- **SwiGLU/SiLU:** relu² is hard to beat for throughput on H100. The non-power-of-2 FFN dimensions in SwiGLU tank MFU.
- **Full attention:** Sliding window (SSSL) is a better compute/quality tradeoff at this scale. Full attention adds FLOPs without proportional quality gain.
- **LR/schedule tuning:** The baseline hyperparams were already well-tuned. Small changes (±LR, warmdown ratio) had marginal or negative effects.
- **Batch 2^17:** Going too small makes gradients too noisy — there's a sweet spot around 2^18.

## Next Directions Worth Trying

- Cosine LR schedule instead of linear warmdown
- Depth=9 with ASPECT_RATIO=57 (same n_embd=512, one more layer, minimal slowdown)
- Label smoothing or different softcap values
- Higher EMBEDDING_LR or UNEMBEDDING_LR with the smaller batch
- Downloading more data shards for diversity

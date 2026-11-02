# Fuse BF16 K=V=64 GLA intra-tile output on gfx936

> Status: confirmed. This note was generated only after the KDA candidate passed correctness and its configured benchmark gates.

## Target

- Backend/device/architecture: `hygon` / `BW` / `gfx936`
- Operator: `flag_attn.runtime.backend._hygon.FLA.gla.chunk_gla`
- Implementation: `src/flag_attn/runtime/backend/_hygon/FLA/gla/chunk_gla.py`
- Language: `triton`
- Candidate: `c003` from task `flagattention-chunk-gla-bw1000`

## Confirmed result

- Metric: `latency` (minimize, `ms`)
- Aggregate improvement: **33.83%**
- Aggregation: `geomean_ratio`; directional ratio `1.338259`

| Shape | Candidate | Baseline | Improvement | Status |
|---|---:|---:|---:|---|
| `B=1,T=1024,H=8,K=64,V=64,bf16` | 0.3126 ms | 0.4414 ms | +41.20% | pass |
| `B=2,T=1024,H=8,K=64,V=64,bf16` | 0.3109 ms | 0.4396 ms | +41.40% | pass |
| `B=4,T=1024,H=8,K=64,V=64,bf16` | 0.3078 ms | 0.4518 ms | +46.78% | pass |
| `B=4,T=2048,H=16,K=64,V=64,bf16` | 0.413 ms | 0.5018 ms | +21.50% | pass |
| `B=8,T=2048,H=32,K=64,V=64,bf16` | 1.4113 ms | 1.7584 ms | +24.59% | pass |
| `B=8,T=4096,H=8,K=64,V=64,bf16` | 0.8051 ms | 0.9939 ms | +23.45% | pass |
| `B=16,T=512,H=8,K=64,V=64,bf16` | 0.3201 ms | 0.4493 ms | +40.36% | pass |

## Mechanism

The fused producer computes a full 64x64 intra-chunk A tile and immediately consumes it in A@V within the same CTA. For K=V=64 there is one K tile and one V tile, so this avoids writing the FP32 A intermediate to global memory, reading it again in the output kernel, and launching that separate output stage without duplicating A computation.

## Applicability

Validated on Hygon BW/gfx936, BF16 inference with K=64 and V=64, no gradients, no initial state, no final-state output, non-varlen input, and state_v_first=False. Seven shapes varying B, T, and H passed output checks and three alternating-order A/B rounds.

## Transferable lesson

For small square GLA head dimensions where a CTA can hold one complete intra-chunk A tile and one V tile, test fusing A@V with output to remove the global intermediate and launch. Match BC/K/V tile widths so the fused path does not decompose a baseline-wide tile into repeated subtiles. Always compare end-to-end latency and resource pressure; larger V can make fusion slower.

## Limitations and guardrails

Do not enable for K/V=128: the broad fusion candidate was 1.5–1.7x slower on tested cases. FP16, FP32, varlen, initial/final state, state_v_first, backward, and dimensions other than K=V=64 are not validated. rocprofv2 kernel-trace did not finish, so no PMC/LDS-bottleneck claim is made. The full public benchmark improvement was limited to the eligible D=64 case, while the seven-case held-out family produced the recorded geomean gain.

## Evidence

- `runs/candidates.jsonl`
- `runs/benchmark.csv`
- `docs/evidence-c003.md`

Recorded at: `2026-10-27T06:31:58+00:00`

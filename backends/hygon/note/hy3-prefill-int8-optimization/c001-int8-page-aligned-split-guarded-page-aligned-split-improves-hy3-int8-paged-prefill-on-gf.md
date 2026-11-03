# Guarded page-aligned split improves HY3 INT8 paged prefill on gfx936

> Status: confirmed. This note was generated only after the KDA candidate passed correctness and its configured benchmark gates.

## Target

- Backend/device/architecture: `hygon` / `BW` / `gfx936`
- Operator: `hy3-prefill-int8-optimization`
- Implementation: `src/flag_attn/runtime/backend/_hygon/hy3/prefill/core.py`
- Language: `triton`
- Candidate: `c001-int8-page-aligned-split` from task `hy3-prefill-int8-optimization`

## Confirmed result

- Metric: `HY3 INT8 median latency (ms)` (minimize, `ms`)
- Aggregate improvement: **35.06%**
- Aggregation: `geomean_ratio`; directional ratio `1.350624`

| Shape | Candidate | Baseline | Improvement | Status |
|---|---:|---:|---:|---|
| `q127_kv127_dense_causal_nhd` | 0.3482 ms | 0.3496 ms | +0.40% | pass |
| `q512_kv4096_dense_causal_nhd` | 1.0339 ms | 1.7964 ms | +73.75% | pass |
| `q512_kv32768_dense_causal_nhd` | 6.0138 ms | 10.8023 ms | +79.63% | pass |
| `q127_kv127_dense_causal_hnd` | 0.3403 ms | 0.343 ms | +0.79% | pass |
| `q512_kv4096_dense_causal_hnd` | 1.0318 ms | 1.7994 ms | +74.39% | pass |
| `q512_kv32768_dense_causal_hnd` | 5.963 ms | 10.5847 ms | +77.51% | pass |
| `q127_kv127_random_block_nhd` | 0.379 ms | 0.3882 ms | +2.43% | pass |
| `q512_kv4096_random_block_nhd` | 0.5562 ms | 0.7337 ms | +31.91% | pass |
| `q512_kv32768_random_block_nhd` | 1.9472 ms | 2.8187 ms | +44.76% | pass |
| `q127_kv127_random_block_hnd` | 0.3797 ms | 0.3899 ms | +2.69% | pass |
| `q512_kv4096_random_block_hnd` | 0.5533 ms | 0.7293 ms | +31.81% | pass |
| `q512_kv32768_random_block_hnd` | 1.9436 ms | 2.767 ms | +42.36% | pass |

## Mechanism

Extending the existing 64x64 page-aligned split path to INT8 lets the paged attention work over more independent KV segments while reusing the kernel's INT32 QK accumulation, explicit QK/V scales, BF16 partial state, and softmax-aware reducer. The trace showed the candidate paged kernel with lower LDS use (8 KiB) and VGPR use (176) than the reference grouped path (32 KiB and 256 VGPR).

## Applicability

Hygon BW/gfx936, HY3 INT8 paged prefill, page size 64, NHD/HND, Q=512 and KV=4096/32768 under the existing metadata-based guard: unmasked or sparsity bucket 2, supported KV layout, at least two default Q tiles, and default CTA count below the existing device-wave threshold.

## Transferable lesson

When the paged Triton kernel and reducer already support INT8 scaling and partial-state merging, a guarded page-aligned split can be shared across BF16 and INT8 without changing the attention math. Keep the original fallback for short Q, other page sizes/layouts, and unvalidated sparsity classes; benchmark each precision separately.

## Limitations and guardrails

Measured on one BW/gfx936 environment with batch 1, 32 query heads, 4 KV heads, 64-token pages, Q lengths 127/512 and KV lengths 127/4096/32768. All 12 tested rows improved with a 1.3506x geomean latency speedup. Longer contexts, batch sizes above one, other page sizes, scales beyond benchmark defaults, and other sparsity buckets remain unvalidated. Random-block FA comparison uses an approximate window mask, not the identical mask.

## Evidence

- `runs/candidates.jsonl`
- `runs/benchmark.csv`
- `runs/int8-page-split-baseline-paired.log`
- `runs/int8-page-split-candidate-paired.log`
- `runs/int8-page-split-candidate-correctness.log`
- `runs/int8-page-split-profile-comparison.txt`

Ledger digests at publication (a changed ledger means this note is stale):

- `runs/candidates.jsonl` sha256 `7292b7bbab5ff840a288d03da0a4adbee4fb570af10b36b2c4a76c8ff2ee69e0`
- `runs/benchmark.csv` sha256 `db8bfd683c35b4bf35f6d56a2f19a1b8e1ae8950e304daffa343685a63c1decf`

Task workspace: `D:/Desktop/codex/.hygon-control`

Recorded at: `2026-11-03T07:54:45+00:00`

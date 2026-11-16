# S60-TORCH-002: sparse chunk buckets and compact output reordering

## Scope and hypothesis

Pure Torch finite-input GCU inference, S60/GCU300, Torch 2.10.0+cpu with
`torch_gcu`, Python 3.12.3. Status: separate numerical tests and full-call
timings; no hardware profiling or compiler dispatch proof.

When query-panel batching fixes launch costs but dense attention still grows
quadratically, group selected query edges by KV chunk. Reuse each chunk's KV
in batched BMM rather than gathering a KV payload for every individual query
or computing every dense query-key pair. Test packing, sort and merge costs
as part of the full operator.

## Observed implementation pattern

1. Preserve routes: mandatory current chunk and selected earlier chunks,
   deterministic ties, packed sequence isolation and GQA head mapping.
2. Bucket edges `(head, query, route-slot)` by `(head, KV-chunk)`. Sort edge
   IDs on the device and count buckets. The observed implementation transfers
   small counts to the host; include this synchronization in timing.
3. Group buckets by power-of-two query capacity (less than 2x valid-row
   padding for nonempty buckets). Bound temporary matrices by workspace bytes.
4. Gather query rows and chunk KV; apply valid-row, KV-tail and causal masks.
   Compute FP32 QK, softmax/PV and per-route logsumexp (LSE).
5. Build a **one-dimensional integer row permutation** with scatter, concatenate
   contiguous outputs/LSEs, then gather rows to restore route-slot order.
6. Merge disjoint chunk outputs using normalized LSE weights in FP32. This is
   a reordering of attention, not averaging independently normalized outputs.

For chunks j with output O_j and log-normalizer L_j:

```text
L = logsumexp_j(L_j)
O = sum_j(exp(L_j - L) * O_j)
```

Mask empty/padded rows safely, exclude them from the final output, and preserve
the reference behavior for all-masked cases. Test rounding against the frozen
tolerances; algebraic equivalence is not bitwise equivalence.

## Failure evidence that changed the design

| Candidate | 4096 / 8192 / 16384 / 32768 ms | Decision in original task |
|---|---|---|
| Fresh dense r303 baseline | 76.877 / 213.375 / 736.586 / 3135.878 | Comparator |
| s401: scatter full `[edges, D]` outputs | 441.169 / 896.710 / 1815.372 / 3876.809 | Rejected: every shape slower |
| s402: compact row-map scatter, concat/gather | 105.095 / 184.356 / 344.598 / 731.913 | Rejected: short case regressed 36.7% |
| s403: also batch heads by real payload budget | 81.045 / 164.756 / 341.605 / 732.638 | Kept under explicitly approved short-case tradeoff |

The s401/s402 comparison supports testing compact output reordering; it is
not a hardware-counter attribution to a particular scatter unit. s403 uses
a general 512 MiB real FP32 route-output budget and 128 MiB score/packed
budget. Head groups derive from L, route count, D and element size, without
exact-length dispatch. These individual budgets do not cap total memory;
padding, concatenate/gather overlap, inputs/outputs and minimum-one-tile
exceptions also consume memory.

## Measurement and generalization

Formal class: BF16 B2, Hq=Hkv16, D128, chunk128, Top-K8, q32/head4 CLI;
synchronized wall-clock median, warmup1, repeat3, inner1. Four-shape latency
sum fell 68.29%. The 4096 case was 5.42% slower and required explicit user
acceptance in that task. **Do not reuse that regression permission elsewhere.**

Two untuned guards using the same implementation improved:

| Input class | Baseline -> s403 ms |
|---|---:|
| BF16 packed lengths [12288], Hq8/Hkv2, D64, chunk64, Top-K4 | 98.988 -> 24.839 |
| FP16 packed lengths [24576,12289], Hq12/Hkv3, D96, chunk128, Top-K6 | 772.937 -> 207.809 |

All 12 frozen CPU/GCU tests and separate full-shape/class/boundary checks
passed. Source fingerprint and exact checks are in the original proof/report.
The 32768 allocation-only probe reported 2.810 GiB PyTorch peak allocated
(inputs/outputs/intermediates); this is neither allocator reserved nor total
device usage, and says nothing about training memory.

Attention work is approximately O(H L R C D), plus sort/packing, for R selected
chunks of C tokens. Dense chunk-mean routing still costs O(H L^2 D/C). Total
operator latency is not guaranteed linear. Longer inputs, different route
densities, host metadata overhead and generic/training paths need fresh timing.

Sources: `torch-sparse-report`, `torch-sparse-candidates`,
`torch-sparse-benchmark`, `torch-sparse-class-benchmark`, `torch-sparse-proof`
and `torch-sparse-memory` in [sources.json](sources.json).

Portability follow-up: [S60-TRITON-003](triton-streaming-score-tiles.md)
records that Triton already used route buckets/compact row maps, while an
additional Torch-style vectorized rank reduction regressed and was rejected.
Transfer mechanisms through fresh candidates, not assumed implementation parity.


## Causal routing panels: prune before scoring

For selection restricted to **complete strictly earlier chunks**, query rows
in [s,e) with chunk size C can select only chunk IDs below floor((e-1)/C).
Crop chunk-mean columns to that prefix before a panel's score BMM and repeated
max/scatter. Retain the per-query historical mask inside the panel; the panel
bound alone is insufficient. If the prefix is empty, produce invalid route
slots directly. If fewer chunks than requested exist, unused slots must remain
invalid, never duplicate a chunk. Preserve first-index tie policy and score
precision. Bounds derive from sequence-local positions, including packed tails.

This transfers causal pruning from attention token panels to chunk-selection
panels. It removes columns that were entirely masked after being computed and
scanned. It does not reduce the complexity class of dense chunk routing and
does not help a panel whose end already makes every chunk eligible. The exact
bound depends on strict historical-chunk semantics; do not reuse it for
bidirectional attention or routing that permits current/future chunks.

On the scoped Torch2.10/torch_gcu2.10+3.7 S60 environment, separate preserved-
implementation/dense-reference checks, forced tiny routing budgets, GQA,
strides, ties and gradient checks passed at unchanged tolerances. Full-call
eight-shape measurements support testing the mechanism after sparse attention.
Raw samples, source changes and per-shape effects remain in the task report.
No physical compute-unit or bandwidth attribution follows from pruning score
elements. New dtypes, SDKs, panel sizes and selection implementations need
fresh numerical and timing validation.

Sources: `torch-routing-prefix-report`, `torch-routing-prefix-numerics`,
`torch-routing-prefix-benchmark`, `torch-routing-prefix-proof` and
`torch-routing-prefix-patch` in [sources.json](sources.json).

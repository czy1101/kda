# S60-TORCH-001: batch dispatches and reuse shared payloads

## Scope and status

Observed in pure Torch MoBA on S60/GCU300, Torch 2.10.0+cpu with `torch_gcu`,
Python 3.12.3. Torch-GCU/SDK build identifiers were not captured in these
summaries; re-probe them. Status: separate correctness and full-call timing.
No compiler-callee or hardware-counter attribution was collected.

## Mechanisms worth testing

| Measured/source symptom | Coherent candidate | Main risk |
|---|---|---|
| Repeated routing max/scatter in many small query tiles | Batch routing scores under a byte budget; reuse selected indices | Larger temporary matrices |
| KV gather, cast and zeroing repeated per query | Reuse FP32 KV head tiles across queries | Extra dense arithmetic at long lengths |
| Token masks materialized before chunk selection | Compact chunk-level masks, expand only for the active score panel | Tail and fallback semantics |
| Many tiny attention BMM calls | Batch query panels using workspace capacity | Peak memory and synchronization |
| Dense panels include wholly future KV columns | Crop KV prefix to the final query in each causal panel | Off-by-one causal boundaries |
| Repeated query scale operation | Scale attention Q once in FP32; keep unscaled routing Q | Rounding and routing changes |

These mechanisms were separate candidates, not one prescribed rewrite.
Historical routing used 32 MiB score batches; attention panels used 32 then
128 MiB. Those are experimental budgets, not S60 hardware limits or universal
defaults. Derive capacities from live tensor sizes, element sizes and the
algorithm; include minimum-tile exceptions and overlapping temporaries.

Causal example: a panel for queries 256..511 needs at most KV 0..511. Each
query still needs its own causal mask (query 256 cannot attend 257..511).
The next panel obtains its own prefix. This eliminates inaccessible future
work; it does not remove history, truncate input tensors or change routing.

## Historical evidence

All rows below are paired within their own run: BF16 B2, Hq=Hkv16, D128,
chunk128, Top-K8; synchronized wall-clock median, warmup1, repeat3, inner1.
Each combined row contains several individually recorded candidates.

| Round / mechanism group | 4096 before -> after ms | 8192 before -> after ms |
|---|---:|---:|
| c001 -> c003: routing batches and shared FP32 attention, original comparator | 3765.003 -> 516.694 | 7859.014 -> 1037.124 |
| r201 -> r202: compact masks and query panels, fresh c003 comparator | 512.242 -> 111.887 | 1028.775 -> 431.474 |
| r301 -> r303: causal cropping, Q pre-scale, larger panels, fresh r202 comparator | 111.968 -> 76.701 | 431.386 -> 213.640 |

Do not multiply these paired ratios into a new measured result. r304's added
contiguous Q copy regressed both cases (77.457/214.990 vs 76.701/213.640 ms)
and was rejected. Contiguity is a hypothesis with copy costs, not an automatic
optimization. Dense arithmetic still scales quadratically at long lengths;
see [sparse buckets](torch-sparse-buckets.md).

## Validation and applicability limits

The fast path was finite-input GCU inference for FP16/BF16/FP32. Generic
CPU, FP64, autograd and nonfinite behavior retained a selected-token fallback.
Masking logits does not necessarily isolate invalid NaN values in later
arithmetic; keep tested isolation semantics. Any finite-input guard, metadata
transfer and dispatch cost belongs inside full-call timing.

Validate packed tails, GQA/MQA head mapping, strides, equal-score ties,
Top-K1/saturation, dtype tolerances, future NaNs and generic-path gradients.
Force small workspace capacities to exercise batching boundaries. Reject
changes that pass a small primitive probe but fail complete operator tests.
Benchmark several representative lengths and untuned shape classes before
claiming generality. Graph-capture support is not established: CPU metadata
and `.item()` guards synchronize.

Sources: `torch-round1-report`, `torch-round2-report`, `torch-round3-report`
and their benchmark/candidate ledgers in [sources.json](sources.json).

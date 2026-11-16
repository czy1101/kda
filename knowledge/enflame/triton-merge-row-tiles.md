# S60-TRITON-004: amortize output/LSE merge over contiguous rows

## Scope and status

Observed2026-10-16, S60/GCU300, Triton3.6.0, Torch2.10.0+cpu with torch_gcu.
Status: frozen forward/backward tests, independent full-shape and boundary
checks, event timings, synchronized phase timings and allocated-memory peaks.
No hardware-counter attribution or compiler-resource comparison was collected.

## Mechanism

MoBA combines current-chunk and selected historical-chunk outputs using their
FP32 log-normalizers. In the retained rank-loop merge, a program processes a
contiguous tile of output rows. Candidate m601 changed only `block_rows`
from64 to256 in the merge wrapper. For the formal aligned cases this requests
one quarter as many logical programs, each processing four times as many rows.

FP32 normalization/accumulation, rank iteration, native staged payload gathers,
validity masks and invalid-route NaN clearing are retained. No three-dimensional
rank vectorization was introduced. The shape boundary is tile tails, not an
exact sequence-length dispatch. Correctness tolerances were unchanged.

The hypothesis is amortized scheduling and per-program work over longer
contiguous slices. Timings establish that this tile helped, but do not isolate
scheduling, compiler layout or transfer effects. Do not label it bandwidth- or
occupancy-bound without compatible profiling evidence.

## Measured effect

Formal BF16 B2 H16 D128 C128 Top-K8; same frozen event timer, warmup25ms,
rep100ms. Three complete unchanged benchmark commands per baseline/candidate,
with a per-shape median. All numerical checks were separate commands.

| Length | Original event median ms | m601 event median ms |
|---|---:|---:|
| 4096 | 122.690 | 109.314 |
| 8192 | 252.095 | 226.553 |
| 16384 | 644.361 | 507.418 |
| 32768 | 4247.430 | 2526.165 |

**32768 event variability matters:** original formal samples ranged2859.196
to4256.594ms; candidate samples were all faster than the fastest original
formal sample. Comparing the slowest candidate2543.587ms against that lowest
original2859.196ms still shows about11% reduction. The median comparison alone
would overstate a stable gain. Preserve ranges and corroborating phase evidence;
the source of event variability was not resolved. The repeat policy was fixed
before candidate deployment; the isolated2855.926ms diagnostic repeat was
preserved but excluded from formal three-run aggregation.

| Shape | Original merge phase ms | m601 merge phase ms | Merge calls before/after |
|---|---:|---:|---|
| B2x4096 | 44.101 | 31.362 | 1/1 |
| B2x8192 | 89.684 | 64.337 | 1/1 |
| B2x32768 | 729.378 | 413.273 | 4/4 |

These are synchronized diagnostics, with native gathers included in the parent
merge cost. Short shapes use three cached diagnostic repeats;32768 one.
Do not add nested gather costs again or derive exact event-time percentages.
The route-output budget remained512MiB. Tensor peaks were unchanged at
0.616/1.231/2.819GiB for these three shapes. These are PyTorch allocated tensor
peaks including inputs/outputs/workspace, not all device or compiler-local
resources. No claim of unchanged register/shared pressure is established.

Two untuned strided GQA classes passed original-merge comparisons and measured
performance guards: BF16 packed lengths[24577,16385], Hq12/Hkv3 D96 C128 K8
improved1516.933->1445.214ms; FP16 length24577, Hq12/Hkv3 D64 C64 K16 improved
2391.889->2339.204ms. These are single paired guard measurements, not broad
performance guarantees for all dtypes/dimensions/Top-K values.

## Correctness and limits

All39 frozen executable tests passed; six unavailable CUDA graph/FlashMoBA
comparisons skipped. Separate comparisons used a hash-pinned original merge
and original512MiB pass budget for all four complete formal inputs and held-out
classes. Independent merge probes cover3/7/63 routes, NaN-invalid entries and
row/dimension tails. Strided packed GQA/MQA and forced multipass guards passed.
Final restored cached suite passed again. Training latency/memory was unmeasured.

The10% aggregate target and2% formal/class caps passed; only m601 was attempted.
Increasing the pass budget or using still larger row tiles was not tested in
this round. Do not present those prepared ideas as verified improvements.
The earlier rank-vectorized t501 regression remains contrary evidence:
[S60-TRITON-003](triton-streaming-score-tiles.md). Larger tiles are hypotheses
requiring current-environment numerical, latency and memory gates, not defaults
for every S60 reduction kernel. Routing remains a separate long-input bottleneck.

Sources in [sources.json](sources.json): `merge-row-report`,
`merge-row-candidates`, `merge-row-baseline-samples`, `merge-row-best-samples`,
`merge-row-evaluation`, `merge-row-phase-evaluation`, `merge-row-source-proof`,
`merge-row-best-source`, and raw original/final short/long phase logs.


## Buffer fusion must preserve effective work organization (2026-10-26)

On the reviewed GCU300/S60 stack, eliminating an intermediate reorder buffer
does not guarantee a full-call improvement. One tested expression combines
scalar-base regular D segments with per-row normalization; numerical tests
and actual slice/DTE paths pass, but complete-call latency regresses. A second
expression consumes a routed attention result in its producing query tile.
It preserves the original low-precision intermediate rounding before FP32
merge and uses unique single-rank query ownership to update initialized
output/LSE rows. Descriptor coverage checks and source-pinned matched-pass
comparisons validate the tested ownership and numerical semantics.

The producer expression introduces BF16 payload gather/scatter and FP32 LSE
gather/scatter callees while retaining native matrix calls. It also regresses
complete calls; larger tested classes abort with SIP/async-printf diagnostics.
The cause of those failures is unisolated. Neither fewer allocated tensors,
native matrix selection nor a correct small class establishes large-class
runtime support or a performance win. No OOM, hardware limit, physical engine
utilization or individual-instruction cost is inferred. Restored original
source passes separate numerical and large-execution recovery checks.

Single-owner semantics are a prerequisite for the tested in-place formulation:
one rank assigns at most one expert to a query/head, descriptor ranges partition
all valid entries without duplication, and no other query row is read. Do not
extend that update to repeated queries or multiple ranks without a new proof.
Generalizable lessons are ownership, intermediate rounding and transfer/work
organization; workload-specific timings and choices stay in task artifacts.

Sources: moba-generic-fusion-scalar-source, moba-generic-fusion-producer-source,
moba-generic-fusion-ownership, moba-generic-fusion-parity,
moba-generic-fusion-actual-paths, moba-generic-fusion-large-failure and
moba-generic-fusion-recovery in sources.json.

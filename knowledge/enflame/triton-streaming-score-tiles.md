# S60-TRITON-003: inspect fallback score tiles and validate Torch transfers

## Scope and status

Observed 2026-10-16, S60/GCU300, Enflame Triton 3.6.0, Torch 2.10.0+cpu
with torch_gcu. SDK/driver build identifiers remain unestablished. Status:
compiler-observed matrix calls, separate numerical checks, formal four-shape
timing and two untuned class comparisons. No hardware counter attribution.

## Mechanism: a fallback branch can reintroduce a slow matrix path

The earlier MoBA direct routing branch (at most128 chunk slots per sequence)
already padded narrow scores to at least64 columns. Its longer-sequence
streaming branch still used `max(32, block_topk)` columns. At B2x32768,
chunk128, seven historical routes, that branch used outer64x32x128 BF16 dots.
The bounded streaming candidate changed the minimum column tile to64, without
changing route count, scores, causal context or the direct-branch threshold.

| Compiled variant | Outer M,N,K | Actual LLVM callee |
|---|---|---|
| Original BF16 streaming | 64,32,128 | `tops_matmul_bf16_fp32` |
| Retained BF16 streaming | 64,64,128 | `tops_convgen_m32_n32_k128_bf16_fp32` |
| Original FP16 class reference | 64,32,64 | `tops_matmul_fp16_fp32` |
| Retained FP16 class | 64,64,64 | `tops_convgen_m32_n32_k64_fp16_fp32` |
| Additional FP16 compiled guard | 64,64,128 | `tops_convgen_m32_n32_k128_fp16_fp32` |

Warps4/stages1/vector_length512. BF16 shared-memory metadata rose from41984
to66560 bytes; these are kernel resource metadata, not device capacity.
Current source and hash-pinned original-reference locations, cache metadata,
TTIR dot shapes, LLVM calls and LLVM hashes are preserved in the exports.
This extends the observed cases in [S60-TRITON-001](triton-convgen.md); it is
not a general support matrix or a universal switch to enable convgen.

The wider tile also reduces loop iterations and repeated Top-K state merges.
The experiment does not isolate the speedup contribution of each effect.
Streaming storage remains bounded by fixed score tiles; no full sequence
score matrix was introduced. Larger Top-K values already producing a64-column
tile do not receive this width change. Other query tiles/layouts still need
independent lowering and timing checks.

## Performance and correctness evidence

Formal BF16 B2 H16 D128 chunk128 Top-K8, same data generator and event-timer
helper:25ms warmup,100ms repetition, median. Numerical checks were separate.

| Length | Fresh original ms | Retained ms |
|---|---:|---:|
| 4096 | 123.952 | 121.922 |
| 8192 | 255.792 | 252.291 |
| 16384 | 652.610 | 645.039 |
| 32768 | 15522.509 | 2856.387 |

The first three use the unchanged direct branch; their small differences are
not evidence of an additional optimization. The long case improved5.43x.
Its synchronized routing diagnostic fell from14134.377 to1492.716ms; this is
one warm instrumented sample per side, not the formal timer or a hardware
utilization measurement. Tensor peak allocated remained3026425856 bytes
(2.819GiB), including inputs/output/workspace; compiler shared metadata did
increase. Do not equate this allocator value to all device memory.

Two untuned full-operator classes, with the same timer and a hash-pinned
original routing comparator, improved:

| Class | Original -> retained ms |
|---|---:|
| BF16 lengths[16641,1025], Hq4/Hkv2, D128, C128, Top-K8, strided Q/K | 813.260 ->197.197 |
| FP16 lengths[8385], Hq6/Hkv2, D64, C64, Top-K5, strided Q/K | 244.228 ->60.660 |

All39 executable frozen tests passed; six skips remain unavailable CUDA graph
API / NVIDIA FlashMoBA comparisons. Separate checks cover NaN-invalid routes,
row/dimension/rank tails, GQA/MQA, packed strides and forced rank multipass.
Long exact-score routing matched independent CPU stable ordering (including
BF16 ties); long complete outputs matched the original routing implementation
at unchanged tolerances. No benchmark/test source changes or shortened context.

## Negative transfer: Torch-style vectorization can regress Triton

Before widening streaming tiles, candidate t501 loaded bounded groups of
route payloads as `[rank,row,dimension]`, applied FP32 LSE weights and reduced
rank in Triton. It retained native row gathers and passed all correctness.
However, four formal times were157.158/320.420/765.837/15834.695ms versus
123.952/255.792/652.610/15522.509ms. Every shape regressed; it was rejected
and the original checkpoint restored before the next candidate.

The existing Triton implementation already grouped KV-chunk routes and used
compact bijective row maps, so copying Torch bucketing again was unnecessary.
Torch's successful native reduction did not predict Triton's tensor-layout
performance. The exact compiler/runtime reason for t501's regression was not
isolated; do not invent a stall, bandwidth or hardware-unit explanation.

## Reuse and revalidation

Inspect every direct/fallback branch around a performance discontinuity.
Treat tile width as one hypothesis; preserve deterministic Top-K ordering,
valid/causal masks, finite/nonfinite semantics and reference tolerances.
Probe installed compilation in the contracted environment, check actual call
sites, test independent long/tail/GQA cases, then benchmark the full call and
memory. Use only KernelPilot for remote actions. Dense chunk-mean routing is
still quadratic in length divided by chunk size; lengths beyond32768 were
not formally timed in this run. Training performance remains unmeasured.

Sources in [sources.json](sources.json): `triton-transfer-report`,
`triton-transfer-compiler-analysis`, `triton-transfer-artifacts`,
`triton-transfer-candidates`, `triton-transfer-original-benchmark`,
`triton-transfer-best-benchmark`, `triton-transfer-class-performance`,
`triton-transfer-proof`, `triton-transfer-original-phases`,
`triton-transfer-best-phases`, and `triton-transfer-rejected-source`.

## Rejected unconditional64-to128 streaming width, 2026-10-16

Fresh retained-m601 phases measured routing1496.207ms at32768 and19196.906ms
at131072, respectively59%/69% of synchronized top-level sums. Already-fast
convgen dots did not remove dependent Top-K extraction and merge work.
Candidate r701 doubled only the fixed streaming score columns64->128,
preserving direct branch, masks, route ordering and precision. It passed39
frozen tests/six unavailable skips, independent CPU stable/tie routing through
131072 including Top-K17/63, original-routing full outputs on all six formal
shapes and strided packed BF16/FP16 GQA output comparisons.

Single formal invocation per shape, frozen event helper25ms warmup/100ms rep:

| Length | Original ms | Candidate ms |
|---|---:|---:|
| B2x4096 | 110.139 | 110.031 |
| B2x8192 | 226.530 | 226.505 |
| B2x16384 | 507.609 | 507.228 |
| B2x32768 | 2542.864 | 2608.986 |
| B2x65536 | 7971.486 | 7778.162 |
| B2x131072 | 27978.020 | 25463.691 |

131072 routing fell to16824.447ms, but32768 routing rose to1564.899ms.
Complete32768 regressed2.60%; untuned BF16/FP16 classes regressed16.66%/7.23%.
Both performance caps failed, so reject despite the131072 improvement and
aggregate ratio0.932835. Best m601 restored; no routing change retained.
The actual launched N128 dot still called convgen; triggering or retaining
that entry is insufficient to predict net loop performance. Wider reductions,
padding, loop frequency and resource changes are competing effects; no
hardware counters isolated the cause. Smaller streaming pools may amortize
less, but a capacity-aware width heuristic remains unvalidated.

At user request, the sampling policy changed to one complete invocation,
repeating only independent obvious anomalies. Reviews found no anomalies:
fresh phases supported the regressions, so no repeat was run. All raw samples
remain. This does not supersede previous tasks' historical three-run data.
User then prioritized a separate Q-reorder memory-path experiment; do not
claim this rejected route attempt as an optimization win.

Sources: `routing-rejected-report`, `routing-rejected-evaluation`,
`routing-rejected-candidates`, `routing-rejected-phases`,
`routing-rejected-compiler`, and `routing-rejected-source-proof`.

## Global convergence of per-tile Top-K extraction (2026-10-23)

For a retained Top-K set and a score tile whose candidates are extracted in
descending score/ascending index order, extraction can stop if **every row**
rejects its best remaining tile candidate against the retained worst pair.
Later candidates from that tile are no better under the same total ordering.
This is a semantic reasoning rule, not a validated backend optimization.
Stopping because only some rows reject is invalid. Skipping later score tiles
is also invalid. Preserve invalid/zero keys, unsigned score-key ordering,
inactive retained lanes, tails and deterministic tie handling.

On the reviewed GCU300/Triton3.6.0 stack, replacing the fixed extraction loop
inside a compound native-matrix/unsigned-Top-K kernel by a dynamic loop with a
row-global integer-sum continuation condition terminates the first inference
test process with SIGSEGV/exit139. The retained fixed-loop source passes its
frozen suite and independent CPU merge-state cases (empty, no replacement,
mixed rows, ties and high unsigned keys). The dynamic candidate did not pass
numerical validation and has no performance result. The exact compiler/runtime
failure location is unresolved: do not generalize this to all dynamic loops,
integer sums, or physical hardware support, and do not claim the theorem proves
that particular lowering correct. Preserve failure logs and restore known-good
source. A future implementation needs compiler-class numerical gates before
timing; version/layout changes require fresh validation.

This complements the compound integer-reduction linking boundary in
[S60-TRITON-009](triton-six-unit-survey.md), with a different observed failure
stage rather than an asserted common cause. Sources in [sources.json](sources.json):
`moba-remain5-convergence-source`, `moba-remain5-convergence-failure`,
`moba-remain5-retained-state`, `moba-remain5-retained-suite`, and
`moba-remain5-environment`.

## Exact partitioned Top-K needs bounded state and a common total order (2026-10-23)

Historical score candidates can be divided into disjoint intervals, reduced to
local Top-K sets and merged into the exact global Top-K. A discarded local
candidate has at least K candidates no worse under the same total order, so it
cannot enter the global set. Score-descending and index-ascending tie order
must be identical in both stages. Preserve global IDs, causal/valid masks and
score precision; recomputing selected scores with a different rounding path is
not part of this equivalence. A partition's final partial interval must be
masked rather than counted as another valid chunk range.

A tested GCU300/Triton3.6.0 implementation stores unsigned score-key bits and
signed indices in separate int32 partial buffers. Explicit bitcasts retain key
ordering across the sign bit. Padded rank lanes may remain unwritten only when
every consumer masks them to invalid key/index sentinels. Per-row partial
candidate IDs differ from a shared chunk-index vector: the global merge must
use those per-row IDs for both tie-breaking and removal. Wider unions can
stream bounded candidate tiles with the established integer-reduction layout.

Bound workspace before allocating both buffers; derive partition count from
chunk capacity, padded ranks, query/head cardinality and a fixed storage bound.
Reduce partitions or retain original streaming when that bound cannot split.
This bounds explicit partial state, not total allocator/backend memory. Local
direct extraction can remove retained-worst updates, but extra launches,
partial traffic, padded causal work and global selection can outweigh savings.
The total-order proof establishes semantics, not speed or backend compatibility.

Independent CPU cases validate ragged partition tails, GQA, strided FP16/BF16,
low/high ranks and multi-tile partitions. Random complete selected-index tensors
match a hash-pinned original module reconstructed without replacing live source.
Frozen tests pass separately. Whole-call timing and untuned regression caps
remain mandatory; actual retained-source compiler captures establish selected
paths only, without physical-unit utilization or isolated stall attribution.
Revalidate compiler/layout classes and resource bounds on another stack.

Evidence in sources.json: `moba-part7-r701-source`,
`moba-part7-r701-correctness`, `moba-part7-r701-state`,
`moba-part7-r701-evaluation` when available, `moba-part7-environment` and
`moba-part7-matched-paths`. Exact tuned constants and timings stay in task data.


The reviewed partitioned candidate is numerically correct but rejected by
complete-call and untuned performance gates. A separate source-pinned stage
diagnostic shows the partition-local score/selection/write stage itself
dominates; final union is not the only added cost. Actual local matrices retain
convgen and structured payload calls, while global partial reads select integer
gather despite contiguous allocation. Flattened partition/rank expressions with
runtime strides do not guarantee affine transfer recognition. Do not infer
benefit from native matrix presence, fewer retained-worst updates or contiguous
allocation alone. Separate unchanged-reference phases from rejected-candidate
captures. No hardware counter isolates the physical cause. Sources:
`moba-part7-candidate-paths`, `moba-part7-candidate-source`,
`moba-part7-stage-diagnostic`.

## Choose partial-state layout jointly for producer and consumer (2026-10-23)

A partitioned selection algorithm's partial workspace must be organized for
both stages. A producer emitting one rank at a time can expose contiguous query
segments by putting token last, instead of issuing a strided column write into
token-major partial state. Its global consumer then needs the exact matching
head/token/partition/rank strides and a single affine candidate axis. Producer
contiguity alone does not establish efficient consumer transpose/load lowering.

For owned padded-rank workspace, semantic rank validity can be applied after
loading allocated cells. Preserve both physical token-row and candidate-column
allocation bounds: padding a reduction tile wider than the actual candidate
pool does not allocate the extra columns. The successful isolated post-load
mask example cannot authorize reading those missing cells. Keep original IDs,
unsigned key bitcasts, deterministic ties, causal candidates and score precision.
Changing layout must update every producer/consumer stride consistently.

Actual candidate captures remove indexed gather from both partition stages
and retain native matrix instructions and structured slice/deslice transfers.
Nevertheless the complete operator fails the performance gate: selecting a
structured transfer is insufficient when local selection, staging and changed
decomposition dominate. This is contrary evidence against promoting a layout
from a synthetic consumer benchmark or native-path markers alone.

This is one bounded producer/consumer organization hypothesis on the reviewed
GCU300/Triton stack, not a universal recommended dimension order. Numerical
checks, actual generated transfer/reduction paths and complete-call formal and
untuned caps decide whether a configuration is useful. Local writes, global
loads, synchronization and memory peaks compete; source contiguity and matrix
instruction presence do not establish physical utilization or end-to-end gain.
Task results and exact constants remain outside this card. Evidence pointers:
moba-layout8-r801-source, moba-layout8-r801-correctness,
moba-layout8-r801-state when executed, moba-layout8-r801-evaluation when measured,
moba-layout8-environment and moba-layout8-matched-paths.

## Exact local tournaments trade reduction width for candidate-state work (2026-10-23)

Partition a score tile into disjoint small groups. The maximum of the current
group maxima is the maximum of all remaining candidates. After extraction,
replace only that group's winner with its next remaining candidate, or clear
the extracted original entry and recompute local winners. This invariant gives
exact repeated selection under a common unsigned-key-descending, signed-index-
ascending total order; it does not justify dropping selections or approximating
scores. Invalid zero keys must not acquire an index, exhausted groups remain
invalid, and every original candidate must be retained until selected or safely
discarded by the full Top-K rule.

For adjacent pairs on the reviewed GCU300/Triton3.6.0 stack, split/reshape can
represent the winner and its only spare, with a fixed compare/select insertion
into bounded retained row vectors. Derive both original IDs from the scalar
contiguous chunk base; flipping a global index bit assumes an aligned base and
is wrong for odd global bases. Spare depletion and per-row index updates need
their own exact oracle. Independent ordered-union and integrated dot-producing
packed/GQA/strided selector checks pass for the tested pair expression.

Matched actual returned compound kernels retain the native matrix/structured
payload paths and declared shared storage. They emit fewer integer reduction
call sites but add INT32 three-dimensional transfer/transpose and synchronization
sites. The complete-call gates reject the pair expression as slower. Static
sites are not dynamic traffic or isolated stall costs; the observation supports
counting local-state/layout work as part of the tradeoff, without identifying
which changed instruction causes the timing difference.

An alternative fixed-four-vector expression passes smaller retained-state
oracles but fails a larger high-unsigned/tied-key index-order oracle while its
key comparison passes. It is rejected before production deployment or timing.
The cause is unisolated; neither the grouping proof nor small-state success
establishes every compiler/state class. Preserve the failure, exact staged
script bytes and live-helper source binding rather than weakening the oracle.
Revalidate after layout, dtype, retained-state or compiler changes. Task choices,
counts, timings and candidate history stay in external moba-pair13-* evidence.

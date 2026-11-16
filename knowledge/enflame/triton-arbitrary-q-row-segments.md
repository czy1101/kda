# S60-TRITON-006: arbitrary Q rows can use regular per-row payload segments

## Gradient-fold follow-up (2026-10-20)

The scalar-base segment mechanism also applies when a flattened output
index maps across token/KV-head boundaries during grouped-head gradient
folding. A vector of flattened destinations can induce BF16 payload gather
calls even though each individual head's D segment is regular. Select one
scalar output row, derive its exact token/head source base, and load each
query-head contribution as a regular D segment. Bounded sequential rows
per program can retain similar output work without assembling discontinuous
row addresses into a vector gather.

Executed GCU300/Triton3.6.0 probes changed actual gradient payload calls
from `tops_gather_load_bf16` to `tops_slice_async_1_bf16`, with padded-slice,
deslice and DTE-wait calls. The group contributions still accumulate in the
original serial FP32 order. Separate exact comparisons cover BF16/FP16,
non-power-of-two head dimensions, group sizes and final-row tails; frozen
operator checks also pass. This is compiler and correctness evidence, not
a claim about physical bandwidth or full-operator speedup. Retain complete
call timing and regression gates before promoting any implementation.
Sources: `parallax-native-baseline-artifacts`,
`parallax-fold-segments-artifacts`, and `parallax-fold-segments-guards`
in [sources.json](sources.json).

## Scope and status

Original observation2026-10-16 on S60/GCU300, Enflame Triton3.6.0, Torch2.10.0+cpu
with torch_gcu: training-path MoBA Q reorder. The gradient-fold follow-up
above has its own evidence scope. Full frozen39 MoBA tests pass,
six unavailable backend skips, independent CPU copy/gradient checks and three
formal row classes. Actual launched baseline/candidate compiler artifacts
verify the payload switch. No hardware counters or physical-engine attribution.

## Mechanism and semantic boundary

S60-TRITON-005's64-row page optimization required an actual shared source-page
mapping. MoBA selected query rows can be arbitrary, scattered and duplicated.
Do not replace those IDs by a different whole-page selection. Instead, exploit
the regular head-dimension addresses *within each selected row*:

1. Load one scalar original row ID, preserving its position and multiplicity.
2. Derive scalar token/head base using real input strides.
3. Copy the D-dimension contiguous/strided segment using regular dimension
   offsets, preserving masks and output row position.
4. Statically group eight independent row copies per program, one warp and
   one stage, to amortize scheduling without asserting adjacent source rows.

The prior vector index load formed a2D arbitrary row-by-dimension payload
pointer tensor. Candidate q801 instead selects a regular payload transfer for
each scalar base. Fixed8 is a tested scheduling choice, not a hardware page
size or an optimum for every shape. More scalar transfers can still regress;
verify lowering and whole-phase/full-forward performance before promotion.

| Payload | Original actual callee | Retained actual callee |
|---|---|---|
| BF16 D128 contiguous | tops_gather_load_bf16 | tops_slice_async_1_bf16 |
| FP16 D96, Q/index stride2 | tops_gather_load_fp16 | tops_slice_async_2_fp16 |

All three formal input classes have paired actual CompiledKernel captures
(six launches), including BF16 aligned rows and FP16 tails/strides. Retained
payload gather_load is absent; slice_pad/deslice and tops_wait_dte are present.
The old kernel already had slice calls for index handling and DTE waits, so
presence alone was insufficient. New tops_memcpy_async reads the scalar IDs;
Q payload uses slice. Resource shared metadata512->0 bytes is compiler data,
not total device memory or chip capacity. Static call counts are not dynamic
traffic, runtime utilization or proof of useful compute/transfer overlap.

## Performance and correctness

Frozen event helper:25ms warmup/100ms rep/median, one complete invocation by
explicit user preference; repeat only obvious independently supported anomalies.
Numerical checking is in separate commands. Every process capped600s including
compilation. Old autotuning/warmup remain in the original comparator. No anomaly
repeat was needed. These are Q-packing forward times, not full MoBA timings.

| Row class | Original ms | Retained ms | Speedup |
|---|---:|---:|---:|
| Q-reorder T8192 H16 D128 R917504 bf16 contiguous | 540.908 | 83.939 | 6.444x |
| Q-reorder T16384 H16 D128 R1835008 bf16 contiguous | 1101.684 | 166.610 | 6.612x |
| Q-reorder T8193 H6 D96 R245790 fp16 strided | 144.549 | 30.351 | 4.763x |

Separately, one warmup/one timed complete grad-enabled forward (Q/K/V require
grad) uses GCU events plus synchronized wall. Compare only within this table;
this boundary differs from frozen no-grad benchmarks. Backward speed unmeasured.

| Shape, BF16 H16 D128 C128 K8 | Original ms | Retained ms | Speedup |
|---|---:|---:|---:|
| B2x4096 | 1451.028 | 1044.510 | 1.389x |
| B2x8192 | 3877.687 | 3028.269 | 1.280x |

Target>=10% aggregate row improvement and<=2% per-row-class/full-forward
regression passed; q801 retained, original/best checkpoints preserved. The
sole retained mechanism preserves API, precision and arbitrary mapping. The
existing FP32 index_add backward is unchanged. Independent CPU tests cover
BF16/FP16 exact copy, duplicate gradients, D63/96/128/129/256, multiple heads,
empty outputs, row/dimension tails and Q/index stride2. Output tolerance0;
gradient atol4e-3/rtol1e-3. Full frozen integration/output/backward tests pass.

Inference uses separate native index_select and never calls this training
gather. Its route/attention/merge functions are unchanged; no six-shape no-grad
speedup is claimed. Long inference routing remains the dominant measured phase.
The prior r70164->128 streaming-score experiment was rejected on regressions
and restored; do not combine its transient timings with q801's parent.

## Revalidation and evidence

Revalidate dtype, layout, D stride and width, indices/duplicates, rows per
program, warp count, output masks, compiled payload callsites and complete
cost. Native index_select remains opaque to this Triton artifact audit. Larger
dimensions/dtypes/production distributions, full backward time and physical
DTE utilization are unmeasured. No matched GCU profiler tool was available;
prior explicit Topsprof captures were unusable. No CUDA tools or TLE.

Raw exports remain in container /tmp/moba_q_reorder_20261016_artifacts/;
local structured source/callsite hashes and raw logs remain in the task workspace.
Sources in sources.json: q-segments-report/evaluation/paths/source-proof,
q-segments-candidates/source, q-segments-baseline-benchmark/best-benchmark,
q-segments-baseline-train/best-train, q-segments-cpu-checks/integration-tests,
q-segments-formal-audit.

## Follow-up: inference already has regular BF16 payload transfers

A read-only runtime-linked audit of retained q801/m601 BF16 no-grad B2x4096
H16 D128 C128 K8 captured routing, current/historical attention and output
merge. Every captured kernel had actual BF16 slice-family calls and DTE waits.
Inference already used this regular transfer path; q801's training Q gather
is not evidence of a new inference improvement. Merge still has three static
tops_gather_load_fp32 callsites. Native index_select staging remains outside
this Triton artifact audit, with its internal lowering unestablished. Do not
claim that every access is a regular slice or extrapolate this shape-only
audit to other specializations. Source/test/benchmark hashes stayed unchanged.
Sources: inference-memory-report, inference-memory-structured,
inference-memory-raw in sources.json.

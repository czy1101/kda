# Enflame S60 / GCU300 optimization knowledge

Last reviewed: 2026-11-26. Scope: local S60/GCU300 observations, not every
Enflame device or future compiler release. Searchable metadata lives in
[catalog.json](catalog.json); original artifact paths and SHA256 digests live
in [sources.json](sources.json). Raw logs are not copied into this repository.

Start with [../../docs/bottleneck-localization.md](../../docs/bottleneck-localization.md)
when the question is "where is the time going", and with
[../../docs/enflame-compiler-diagnostics.md](../../docs/enflame-compiler-diagnostics.md)
when the question is "what did the compiler emit". The cards below supply the
observations those two procedures consume.

## Select a starting point

| Symptom / question | Card | Language | Evidence scope |
|---|---|---|---|
| Which compiled artifact did the timed call actually use, and what does a cache key prove? | [S60-COMPILER-001: artifact identity](compiler-artifact-identity.md) | Triton | Five-stage chain across three compilers, in-process versus external boundaries, specialization-attribute cache entries, registered versus executed versus effective passes |
| Which local dot shapes select convgen; BF16 final storage versus dot-result type | [S60-TRITON-001: convgen selection](triton-convgen.md) | Triton | Bounded N/K32 or N/K16 rule; masked tails, loops and storage controls; frontend/local-IR gates distinguished |
| Compute/DTE paths; affine payloads and structural metadata coordinates | [S60-TRITON-002: computation and memory paths](triton-compute-memory-paths.md) | Triton | Runtime-linked calls/ISA; immutable bases, scalar offsets and safe post-load masks; bounded tl.load/tle.load linked-code equivalence; physical utilization unmeasured |
| Which expressions trigger matrix, vector, special-function and structured-page transfer paths? | [S60-TRITON-005: independent hardware trigger probes](triton-hardware-trigger-probes.md) | Triton | 53 configurations; matched calls/ISA; stable page-transfer comparison and unstable matrix repeats |
| Arbitrary Q rows or flattened gradient folds use gather; can scalar-base D segments select regular payload copies? | [S60-TRITON-006: scalar-row segments](triton-arbitrary-q-row-segments.md) | Triton | Paired actual payload calls, CPU mapping/backward tests, row timings and exact-order gradient guards |
| Long-sequence fallback has a latency cliff; can a Torch mechanism transfer? | [S60-TRITON-003: streaming score tiles and transfer failures](triton-streaming-score-tiles.md) | Triton | Matched matrix calls, long class timings, rejected vectorization and bounded dynamic-convergence failure |
| Output/LSE merge dominates; many small programs or repeated rank passes | [S60-TRITON-004: contiguous merge row tiles](triton-merge-row-tiles.md) | Triton | Full-call/class/phase timings, unchanged tensor peaks and event variability |
| MSA already uses convgen; can query grouping or a live score tile improve it? | [S60-TRITON-007: MSA query groups and bounded score tiles](triton-msa-live-tiles.md) | Triton | Actual launched kernels, LLVM/ISA, separate correctness and fourteen-shape timing |
| MSA already uses convgen but remains slow; known strides, singleton query axes and PR timing boundaries | [S60-TRITON-008: MSA static geometry](triton-msa-static-geometry.md) | Triton | Separate correctness, full14-case timing, real-launch artifacts and39 matched dimensions |
| Many small dispatches, repeated KV gather/cast, oversized masks | [S60-TORCH-001: batching and reuse](torch-batching.md) | Torch | Separate correctness + end-to-end timing |
| Long-sequence attention remains quadratic; scattered payloads; future routing work | [S60-TORCH-002: sparse buckets](torch-sparse-buckets.md) | Torch | Sparse buckets, compact reorder and strict causal routing-prefix pruning |
| Native SDPA available, but is it a valid replacement? | [S60-TORCH-003: native API failures](torch-native-api-pitfalls.md) | Torch | Rejected numerical candidate + API probe |
| No profiler, cold runs, incompatible timers, unknown hardware path | [S60-EVIDENCE-001: evidence discipline](evidence-and-measurement.md) | Both | Reusable interpretation rules |
| Device SIP busy/occupancy monitoring and unusable DRAM counters | [S60-EVIDENCE-002: efsmi GPM](efsmi-gpm-monitor.md) | Both | Installed read-only help, idle and sustained MSA samples |
| Systematic matrix/vector/on-chip/transfer/pipeline/scheduling probes | [S60-TRITON-009: six-category survey](triton-six-unit-survey.md) | Triton | Separate numerical tests, compiled calls/ISA, matched probes, compound integer-reduction link limits and tuple-reduction gates |
| Sparse metadata, stable device radix, packed exact prefixes, temporary capacity, dot/precision gates and compiler limits | [S60-TRITON-011](triton-gcu-indexing-and-compiler-limits.md) | Both | Version-scoped frontend/local-IR and runtime Tensor-tuple launcher gates; broadcast-mask runtime assertion and verified stride specialization |
| Bounded live tiles, prefix ranks, mask layout, shared-destination ownership and weighted-mean normalization | [S60-TRITON-010](triton-live-tiles-and-output-ownership.md) | Triton | Reusable mechanisms; resource limits, epsilon rescaling, probability-output boundaries, split-call costs and external numerical evidence |
| Many small programs, one tile per program; is the grid count itself the cost? | [S60-TRITON-012: per-program cost](triton-per-program-cost.md) | Triton | Three kernels at two grid sizes, one-unit batching A/B, batch-4 regression, neutral warp/transfer ablations and a context-dependent single-point replay |
| Key-major FP32 gate tile materializes a 32K+32K shared transpose pair; time-major leaves only the BF16 pair | [S60-TRITON-014: shared transpose materialization](triton-shared-transpose-materialization.md) | Triton | Same-worktree TTIR/GCUIR/LLIR/fatbin, shared-allocation ledger kept separate from metadata `shared`, 14.75% kernel A/B and a state-scan hash-only control |
| Consumer re-masks a producer-masked operand, or reads a transposed tile through a strided key-major pointer | [S60-TRITON-013: operand preparation](triton-operand-preparation.md) | Triton | Exact-rewrite numerics, per-kernel A/B deltas and related neutral load-spelling evidence |

```bash
# Controller-side lookup; no remote connection or optional Wiki required.
rg -n -i 'convgen|padding|bf16' <kda>/knowledge/enflame
rg -n -i 'scatter|bucket|workspace|sdpa' <kda>/knowledge/enflame
```

## Use during a KDA run

1. Read the task contract first. Match backend, architecture, language,
   dtype, shape *class*, layout, grad mode, numerical semantics and versions.
2. Read relevant cards, including their limits and rejected alternatives.
   Record card IDs and environment differences in `docs/plan.md`. With no
   matching card, record that fact and use the normal KDA loop.
3. Convert a mechanism into one falsifiable candidate hypothesis. State what
   compiler artifact, phase cost or full-call metric should change.
4. Check the installed API/compiler in the contracted environment through
   KernelPilot. Do not install tools, bypass controlled SSH, enable TLE, or
   apply CUDA-specific assumptions because a card mentions another backend.
5. Run numerical tests in a separate command. Performance-only benchmarks
   retain warmup, compilation policy and device synchronization, but contain
   no accuracy assertions, reference comparisons or NaN/Inf output checks.
6. Promote only from fresh comparable measurements under the task's goals
   and regression rules. Check representative tails, GQA, strides and memory
   where relevant; avoid dispatch boundaries based only on measured lengths.

An index match does not declare a profiler available or override backend
profile/skill routing. It is useful even when no GCU profiler profile exists.
The generated KernelPilot optimization prompt links this index for GCU tasks,
including tasks without a matching backend profile. The task-side device,
measurement, and TLE procedure is
[../../docs/enflame-s60-runbook.md](../../docs/enflame-s60-runbook.md).

## Add or revise knowledge

Admission rule: publish only reusable mechanisms, interfaces, applicability
conditions and evidence-backed backend limitations. Explain what another
operator or workload can reuse, and the version/layout/semantic boundaries.
A result from one task is not itself a general rule. Keep operator-specific
winners, tuned constants, shape tables, timing tables, candidate ledgers and
run narratives in that task's workspace. Evidence pointers and hashes belong
in this index; copies of experiment reports do not. Do not auto-publish them.

Use [CARD_TEMPLATE.md](CARD_TEMPLATE.md), assign a stable ID, and update the
catalog. Separate observed facts, mechanisms inferred from those facts, and
unverified ideas. Summarize the environment, semantic boundaries, failure modes
and revalidation triggers needed for reuse. Link external evidence for exact
comparators, timing policies, validation scope and raw measurements instead of
copying task reports. Preserve older contrary evidence instead of
silently rewriting it as a universal rule. New versions need fresh evidence;
the review date is not a compatibility guarantee. Missing local artifacts
reduce reproducibility and must be reported, not replaced by invented logs.

During subsequent Enflame optimization runs, maintain this knowledge base
when a new reusable mechanism, computation/data-transfer path, applicability
condition or instructive failure is discovered. This is a standing user
request. Add or revise the relevant card, catalog entry and source pointers
with hashes for local artifacts; keep raw logs in the task workspace. Record
the knowledge update in the final report. Hypotheses remain explicitly
unverified until supported by current evidence; contrary results narrow or
supersede earlier guidance rather than being discarded.

Knowledge maintenance must respect repository write access and frozen task
files. If the KDA repository cannot be written in the current execution
environment, save a proposed card/update in the task's docs and report its
pending destination. Do not change tests, benchmarks, contracts or remote
allowlists to perform a knowledge update.

# Verify why a performance change works

Use this procedure when explaining an observed gain, especially CTA ordering or work distribution changes and cases where PCM counters cannot be collected. The goal is a reproducible causal argument, not a plausible hardware story. Adapt the experiments to the kernel and authorized budget; do not run every diagnostic on every candidate.

## Separate three questions

1. **Does it improve latency?** Use repeated, uninstrumented parent/candidate measurements and the required correctness and formal benchmark checks.
2. **Which software factor causes the gain?** Isolate mapping, grid dimensionality, traversal direction, pointer representation, and compilation effects rather than attributing their combined change to one factor.
3. **What execution mechanism mediates the gain?** Observe the relevant execution behavior and test whether the gain survives removal of the proposed cause.

An answer to question 1 or 2 does not automatically answer question 3. Equal register counts or static instruction counts do not establish identical dynamic execution. A program ID is a label, not proof of physical issue order. Grid size divided by processor count is not a measured number of execution waves.

Before measuring, name competing mechanisms, the observation that would distinguish them, and an outcome that would weaken the preferred explanation. For example, ordering may change load balance, intra-kernel reuse, memory contention, or generated instructions. Preserve useful negative results instead of selecting only supporting observations.

## Establish the work and dependency model

Derive a logical task's loop bounds, valid work, input footprint, outputs, and producer dependencies from the actual launched specialization. Distinguish:

- total work across the grid from work per CTA;
- independent CTAs reading already prepared overlapping inputs from CTAs consuming another CTA's output;
- logical tile/head identity from physical program ID;
- static instruction sites from dynamically repeated operations.

For a causal triangular kernel, a later query tile may execute more prefix iterations than an earlier tile even though both produce the same output tile size. Reordering preserves total work but can change its allocation and overlap. This is an example to verify from loop bounds, not an assumption for all attention kernels, window configurations, or operators.

## Isolate ordering from compilation

First compare the smallest source change that isolates the factor, such as swapping dimensions without also reversing traversal. Preserve the same mathematical task set, CTA count, inputs, outputs, tile, warps, stages, and numerical contract.

If source remapping changes compilation, consider a diagnostic kernel reading logical task coordinates from a runtime mapping table. Compare permutations using the same compiled kernel, grid, table layout, input/output addresses, and launch options. Verify the final binary hash for every permutation; one Python JIT object or equal compiler metadata is insufficient because specialization and alignment can create different binaries. Check that each table is a bijection over the intended logical tasks and validate outputs before timing.

The table load can prevent constant folding and increase spills. Therefore compare permutations **within the table implementation**, and compare native variants separately. Reproduce the direction on native kernels before extending the diagnostic result to production. Archive both compiler paths; do not compare their absolute latency as though only task ordering differed.

Use interleaved or randomized version order, precompiled kernels, preallocated outputs, synchronized device timing, enough real samples, and an independent-process retest for the key comparison. A fixed-shape diagnostic constant or mapping is not a production specialization or authorization to hard-code benchmark cases.

## Observe execution without PCM

Probe the installed backend and compiler before using device intrinsics. Some PPU FlagTree versions expose `triton.language.extra.ppu.globaltimer` and `smid`; availability and meaning must be checked locally. Do not substitute CUDA intrinsics merely because their names match.

Start with a small independent probe, then instrument isolated copies of the actual kernel. A useful per-CTA record is:

```text
physical program ID, logical task coordinates, execution-unit ID,
entry timestamp, exit timestamp
```

Keep raw records for multiple launches, not only aggregates or a screenshot. Explain exactly which thread/warp writes the record, where timestamps are taken, and which synchronization establishes the measured interval. Use only synchronization valid for the kernel; an oversubscribed grid-wide spin barrier can deadlock and must not be added to a real kernel.

Validate instrumentation before interpreting the trace:

- Check timestamp monotonicity, wraparound, units, and cross-unit comparability. A small synchronization/handshake probe can detect obvious causal inconsistencies; it does not prove zero clock skew. Keep raw ticks if conversion is unverified.
- Check whether execution IDs are stable within a CTA or differ between warps. Do not assume IDs are dense or that an observed subset proves other units are disabled.
- Inspect IR and, where available, final ISA to confirm the reads surround the intended work and have not been hoisted, merged, or eliminated. A requested dump producing a header alone is not successful disassembly.
- Compare registers, spills, shared memory, correctness, and latency with and without tracing. The uninstrumented gain should remain recognizable in the traced variants. If instrumentation changes the behavior materially, reduce its scope, sample fewer tasks, or keep the result inconclusive.

Call these **instrumented execution intervals**. Entry occurs after hardware launch, exit precedes full retirement, and timestamp/log writes introduce overhead. They are not exact issue/retirement events. Generic kernel-level timeline records cannot substitute for per-CTA records.

## Test the proposed execution mechanism

For a scheduling or tail hypothesis, calculate from the records:

- task membership of early overlapping cohorts; do not infer a strict total issue order from close timestamps;
- algorithmic work allocated to each observed execution ID, using verified loop counts or another explicit work model;
- per-task duration distributions conditioned on logical work, rather than pooling short and long tasks indiscriminately;
- full interval span, sum of CTA interval durations, and their ratio, the mean number of overlapping intervals;
- the overlap curve near completion and the spread of each execution ID's last completion time.

If the total interval duration changes little while the full span contracts, temporal overlap accounts for much of the difference. More even allocated work and more concentrated completion times support load-balance/tail explanations. This arithmetic decomposition is not a percentage attribution of independent hardware causes. Use a shared time scale in comparison plots; disclose whether a tail window is absolute or relative.

Software overlap, interval coverage, and task counts are not hardware occupancy, active warps, issue utilization, or CU Active. Do not normalize by an assumed processor count and rename the result a measured hardware percentage. Compare the same observed execution-ID population and state unresolved ID semantics.

## Remove the hypothesized cause

Design a counterfactual that weakens the proposed mechanism and predict its result before running it. For heterogeneous task lengths, an isolated variant can make loop work approximately uniform while masking extra results to preserve the output. For another hypothesis, use a control that removes the relevant dependency, reuse opportunity, or intermediate transfer while keeping other factors as stable as feasible.

Verify that the compiler still executes the intended diagnostic work: inspect loop bounds, branches, side effects, and consumed outputs, and check observed task-duration distributions. Source-level padding alone does not establish equal work. Validate numerical results before collecting performance data. If a microbenchmark intentionally changes semantics, label it accordingly and never treat it as a correct operator candidate.

Compare orderings within the same counterfactual binary. A gain that persists in the real kernel but disappears after the hypothesized cause is reduced supports that cause. A surviving or reversed gain is evidence to revise the explanation, not to hide the control.

Counterfactuals usually introduce confounders. Uniform-loop padding can change memory traffic, cache state, register allocation, and pipeline behavior; it does not completely exclude cache mechanisms. Combine the control with real-kernel traces and same-binary experiments instead of declaring uniqueness from one result. Likewise, a large cache-perturbation buffer does not prove a cache was flushed or measure its hit rate.

## Report the evidence boundary

Keep a compact chain in the existing task report:

```text
measured improvement -> isolated software factor -> observed execution change
-> counterfactual prediction and result -> supported mechanism
-> remaining alternatives and unmeasured quantities
```

Record source/binary hashes, specialization, device, input construction, timing scope, sample count, order, raw trace paths, instrumentation changes, correctness, and independent retest. Show representative regressions and exceptions as well as successes. Distinguish direct observations from a mechanism supported by converging controls and from an untested explanation.

When the controls support it, move beyond "mapping changed and got faster" to the specific observed distribution/overlap/tail change. When they do not, retain the measured gain and report the mechanism as unresolved. Do not invent cache, DRAM, stall, or occupancy counters to make the explanation sound complete.

Keep operator-specific harnesses, data, plots, credentials, and benchmark values in the task workspace. This reference records reusable decisions, not one experiment's conclusions or a mandatory PPU tuning recipe.

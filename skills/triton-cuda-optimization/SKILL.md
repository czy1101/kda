---
name: triton-cuda-optimization
description: Optimize Triton kernels on NVIDIA CUDA after correctness and a reproducible baseline exist. Use for candidate selection, launch/layout tuning, pass fusion or splitting, temporary-workspace decisions, and diagnosing shape-dependent regressions. Do not use as a replacement for the task's tests, benchmark contract, or profiler evidence.
---

# Triton CUDA optimization

Read the task contract and the selected backend profile first. Treat its
correctness command, required shapes, aggregation rule, regression limit, and
allowed paths as hard constraints.

## Candidate loop

1. Classify the current limit using measured evidence: launch overhead,
   redundant global-memory traffic, temporary workspace, occupancy/register
   pressure, synchronization, or arithmetic throughput.
2. State one falsifiable hypothesis and the shapes it should help or hurt.
3. Estimate memory traffic and every temporary allocation before changing pass
   structure. Include batch, heads, partitions, dtype bytes, and simultaneous
   buffers; leave device-memory headroom.
4. Change one coherent mechanism. Keep tunable thresholds named and explain
   the safety condition they encode.
5. Run correctness, then the full required benchmark matrix. A fast subset is
   useful for triage but cannot promote a candidate.
6. Record failures and regressions. Revert to the best correct candidate before
   starting the next hypothesis.

Do not create exact-shape branches, expected-value shortcuts, or thresholds
chosen only to fit the published benchmark matrix. Shape-dependent routing is
acceptable only when permitted by the task and justified by a stable
algorithmic crossover, resource limit, or input property. Run configured guard
shapes or generalization checks before keeping such a candidate.

It is valid to finish without reaching the target when measurements show no
safe actionable bottleneck, the best implementation is already at reference
parity, or the remaining gap depends on capabilities unavailable to Triton or
the selected backend. Keep the best correct implementation and state the
evidence and remaining gap instead of forcing another change.

For choices between one-pass and partitioned/two-pass kernels, read
`references/workspace-and-pass-selection.md`.

Use the profiler skill selected by the backend profile only when counters are
needed to distinguish competing explanations. Do not profile merely because a
tool is available.

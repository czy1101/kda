---
name: triton-cuda-optimization
description: Optimize Triton kernels on NVIDIA CUDA after correctness and a reproducible baseline exist. Use for candidate selection, launch/layout tuning, pass fusion or splitting, temporary-workspace decisions, and diagnosing shape-dependent regressions. Do not use as a replacement for the task's tests, benchmark contract, or profiler evidence.
---

# Triton CUDA optimization

Read the task contract and the selected backend profile first. Treat its
correctness command, required shapes, aggregation rule, regression limit, and
allowed paths as hard constraints.

## Shared-memory and resource policy

Apply the user's KDA coding convention before implementing a candidate:

- Reject explicit shape/size-only empty shared-memory allocation followed by
  stores/copies, including scratch and communication buffers. Later filling
  every location, or using dummy initialization, does not satisfy this rule.
- Permit a documented interface that takes actual algorithm data and constructs
  or copies its shared representation; derive storage from that data/layout.
  Compiler-managed staging and live intermediate registers remain permitted.
- Do not tune independent register-prefix budgets (`REG`), empty shared-memory
  byte budgets (`CACHE_BYTES`), or standalone physical register-count caps.
  `REG` counts retained elements, not physical registers. Normal algorithmic
  tiles, warp/thread counts, and stages remain eligible for measured tuning.
- Packing metadata such as `WORD_BLOCK` does not allocate memory by itself;
  inspect the complete consumer/data flow rather than parameter names.
- Verify the real backend/version API; do not invent an initialization method.
  If no permitted interface exists, use an allowed ordinary path or report the
  missing capability according to the task's extension policy.
- Correctness or speed cannot override this convention. Reject noncompliant
  candidates and restore the best compliant correct checkpoint; report when no
  such checkpoint exists. Preserve frozen files and historical evidence.

This is a user convention, not a hardware-validity claim. In KDA work, also read
`<kda>/docs/shared-memory-and-resource-policy.md` for the detailed review rules.

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

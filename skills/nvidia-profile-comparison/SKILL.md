---
name: nvidia-profile-comparison
description: Analyze and compare structured NVIDIA GPU profiler artifacts for a candidate and an optional high-performance reference. Use when a CUDA task has NCU-derived exports or low-level compiler artifacts and needs an evidence-backed optimization hypothesis. Do not use on non-NVIDIA backends or as a substitute for end-to-end benchmark results.
---

# NVIDIA profile comparison

Use the task contract and backend profile to determine which artifacts are
available. Preserve the original profiler report, but reason from deterministic
structured exports rather than assuming an opaque report is directly readable.

Before comparing candidate and reference, verify that they used the same task
inputs, shapes, device, warmup, capture method, and software environment. If
comparability is not established, report the mismatch and analyze each report
separately.

Compare end-to-end execution first. CUDA and Triton implementations may split
work into different kernels, so do not force a one-to-one kernel mapping. Then
compare aligned kernels or equivalent groups using the checklist in
`references/comparison-checklist.md`.

Read `references/artifact-contract.md` when validating report completeness,
performing direct-AI fallback, or deciding whether PTX/SASS evidence is usable.

Produce one evidence-backed bottleneck explanation and one falsifiable
candidate hypothesis. State which shapes should benefit, the expected metric
movement, and risks such as register pressure, occupancy loss, extra launches,
workspace growth, or shape-specific regressions. Do not infer an optimization
win from counters alone; correctness and the contracted benchmark remain the
promotion authority.

When end-to-end performance and the meaningful normalized indicators are
already within the task's reference-parity tolerance, report parity as a valid
stop reason. Do not manufacture a low-level difference or recommend an unsafe
rewrite merely to produce another candidate. Likewise, when the remaining gap
cannot be connected to an actionable Triton or backend mechanism, report the
analysis as inconclusive or capability-exhausted and preserve the best result.

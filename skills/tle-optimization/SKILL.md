---
name: tle-optimization
description: Use project-documented TLE primitives for a kernel optimization only after the task explicitly enables TLE and its tle-wiki plus API probe have passed. Use for deciding whether TLE can express a measured bottleneck and for recording unsupported primitives or TLE failures. Do not activate for default Triton-only tasks or infer APIs from another TLE version.
---

# TLE optimization

Load this skill only when `analysis.extensions.tle` is `auto` or `required` and
`runs/tle-assessment.json` confirms both the operator project's configured
`tle-wiki` directory and the documented API probe succeeded in the contracted
environment.

Treat the project Wiki as the authority for the installed TLE version, imports,
primitives, constraints, and examples. Do not substitute memory of another TLE
project or assume that a conda environment named `tle` exposes the API.

Start from a measured bottleneck and identify the exact operation Triton cannot
express adequately. Select a TLE primitive only when the Wiki documents its
semantics and it addresses that operation. State the expected benefit and
failure risks before implementing. TLE candidates use the normal allowed-path,
correctness, benchmark, generalization, candidate-record, and rollback gates.

Record API, compilation, lowering, runtime, correctness, and performance
failures with the attempted primitive and error. If the required operation has
no documented TLE primitive, do not invent or emulate a nonexistent interface.
Use a safe non-TLE fallback when available; otherwise preserve the best correct
implementation and add a concrete missing-API request to the final report,
including the bottleneck evidence and the semantics the new primitive would
need.

When the task policy is `required`, a missing Wiki/API, undocumented required
primitive, or failed TLE implementation is a terminal blocker for that run.
Restore the best correct checkpoint and report it; do not silently continue as
a Triton-only optimization. Under `auto`, record the same evidence and continue
through the ordinary optimization route.

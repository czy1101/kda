# Final Optimization Report

## Outcome

- Status:
- Target reached:
- Stop reason:
- Best correct candidate:
- Deployed or restored revision:
- Best-checkpoint restore result:

## Performance

- Comparator:
- Metric, direction, and unit:
- Required-shape results:
- Aggregate result:
- Remaining gap:
- Reference-parity assessment:

## Correctness and generalization

- Correctness result:
- Generalization policy:
- Guard-shape or held-out evidence:
- Shape-dependent logic and algorithmic justification:

## Candidate summary

List every candidate, including rejected and failed attempts, with its
hypothesis, result, decision, reason, and rollback status.

## Strategy and interface assessment

- Diagnostic layer each candidate targeted, and its change class:
- Implementation means used, per candidate:
- Capability gates reached for that means — documented semantics, local interface,
  target compile, correct run, workload gain — and the gate that failed when one did:
- Means considered and not used, with the measured reason:
- For a means beyond the operator's existing implementation: the version it was
  validated against, and whether anything was compiled or measured on a device

## Analysis summary

- Detected backend and environment:
- Analysis tools probed, versions, and availability:
- Selected tool and selection reason:
- Tool failures and fallback path:
- Bottleneck evidence:
- Profile or compiler artifacts used:
- Skill or direct-AI fallback used:
- Remaining actionable bottlenecks:
- Exhausted or unavailable backend capabilities:

## TLE assessment

- Task policy (`disabled`, `auto`, or `required`):
- Project `tle-wiki` path and availability:
- TLE API probe and availability:
- TLE primitives used:
- TLE errors and fallback:
- Missing TLE API or primitive required by the diagnosed bottleneck:

## Workflow execution record

List workflow errors, failed stages, retries, interruptions, recoveries, and
early exits in execution order. Do not omit recovered errors.

## Integrity statement

Confirm that measurements came from recorded commands, no test or benchmark
was altered, no benchmark-specific values were hard-coded, and the remote
implementation is the reported best correct candidate or restored baseline.

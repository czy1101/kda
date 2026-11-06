# Final Optimization Report

Adapt this template to the user's requested deliverable using
`skills/operator-optimization-report/SKILL.md`. For a single-document
successful-method summary, expand only retained mechanisms, embed each method's
complete benchmark comparison and key code evidence, and omit audit-only
sections. Keep all attempts and workflow events in the internal ledger.

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

For an experiment-audit report, list every candidate, including rejected and
failed attempts, with its hypothesis, result, decision, reason, and rollback
status. For a successful-method summary, replace this section with retained
methods in cumulative order; include the full required matrix for each adjacent
comparison and keep negative rows visible.

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

Include this section for an execution-audit deliverable: list workflow errors,
failed stages, retries, interruptions, recoveries, and early exits in execution
order. Preserve all events internally even when a successful-method deliverable
omits this audit section.

## Integrity statement

Confirm that measurements came from recorded commands, no test or benchmark
was altered, no benchmark-specific values were hard-coded, and the remote
implementation is the reported best correct candidate or restored baseline.

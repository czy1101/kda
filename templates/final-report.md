# Final Optimization Report

## Outcome

- Status:
- Target reached:
- Stop reason:
- Best correct candidate:
- Deployed or restored revision:
- Best-checkpoint restore result:

## Stop-budget report

- Candidates attempted vs `stop.max_candidates`:
- Profiling captures vs `stop.max_profile_runs`:
- Consecutive candidates without a new best vs `stop.no_improvement_rounds`:
- Action taken when a budget or streak was reached:

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

## PR quality gates

- Public API import/export result:
- Module-layering result:
- Operator name and registry-id result:
- pytest marker collection result:
- Unified dispatch result:
- Quality evidence: `runs/quality.json`

## Candidate summary

List every candidate, including rejected and failed attempts, with its
hypothesis, result, decision, reason, and rollback status.

## Reusable optimization knowledge

- Final best candidate reviewed:
- Knowledge published (Markdown / JSON / index entry), or skipped:
- Eligibility evidence and aggregate gain:
- Skip or publication error reason, if applicable:

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

# Kernel Optimization Task

Read, in order, the task workspace `AGENTS.md`, `.kernelpilot/task.yaml`, the
KDA workflow at `workflows/kernel-optimization.yaml`, the selected backend
profile, and only the Skills routed for the current stage.

Treat the workflow as the intended execution contract for one complete run.
Use its declared transition for each observed stage result. If a transition,
action, tool, or required artifact cannot be completed, record the deviation
and recovery in `runs/stage-events.jsonl` and the final report.

## Execution and safety

- Use KernelPilot wrappers for environment preflight, tool discovery, baseline,
  correctness, benchmark, profiling, checkpoint, and remote operations.
- For SSH tasks, never invoke `ssh`, `scp`, `rsync`, or an unrestricted remote
  shell. Edit only the controlled local mirror and synchronize only
  `constraints.allowed_paths` through KernelPilot.
- Preserve the baseline checkpoint before editing. Save a new best checkpoint
  only after correctness, performance comparison, and any configured
  generalization checks pass. Restore the best after every rejected candidate.
- Do not modify the task comparator, reference, tests, benchmark, or forbidden
  paths after measurement begins.
- Never fabricate measurements, hide failures, or force an optimization when
  evidence does not support one. Do not hard-code benchmark shapes, expected
  values, or public test cases. Shape-dependent logic needs task authorization,
  an algorithmic reason, and configured generalization evidence.

## Candidate loop

1. Inspect the operator and reference, then write `docs/draft.md` and
   `docs/plan.md` before implementation.
2. Establish correctness and the configured original/reference baseline, and
   record the baseline candidate before evaluating optimized candidates.
3. Diagnose the current bottleneck from source and measurements. Profile only
   when it can distinguish concrete competing hypotheses.
4. State one falsifiable hypothesis and implement one coherent candidate.
5. Run correctness. Benchmark only correct candidates across the full required
   shape set, then run configured generalization checks.
6. Record every kept, revised, rejected, failed, or regressed candidate with a
   reason and rollback result. Save the checkpoint for a correct improved
   candidate, then evaluate the configured goal from raw benchmark records.
7. Continue until the target or a configured evidence-backed stop condition is
   reached. A truthful target-not-reached result is acceptable.

The task's metric and direction are authoritative. Do not assume the metric is
latency: minimized and maximized metrics use the same evidence gates. Exactly
one of `goal.relative_to_baseline` and `goal.target_value` defines success, and
`goal.aggregation` plus `goal.max_regression` controls multi-shape promotion.

## Backend and performance analysis

If `target.profile` is present, validate and use it. Otherwise use the unique
backend profile matching the declared backend/device/architecture/language. Do
not guess when multiple profiles match; continue without vendor-specific
guidance when none matches.

Run backend tool discovery in the exact task environment. A backend declaration
is a probe allowlist, not proof that a tool is installed or usable. Select only
tools whose live probe succeeded, and preserve their versions and failures in
`runs/analysis-capabilities.json`.

- Use kernel counters for execution-pipeline or memory questions, timelines for
  launch/synchronization/host gaps, framework tools for operator attribution,
  and compiler artifacts for lowering or generated-code questions.
- A successful executable probe establishes visibility only. Record capture,
  permission, export, or parsing failures and try another compatible discovered
  capability or the benchmark/source/compiler fallback.
- Analyze structured exports rather than treating opaque binary reports as
  model-readable evidence. Candidate/reference captures must use comparable
  inputs, shapes, device conditions, warmup, and capture policy.
- Use routed profiling Skills first. If they are missing or inconclusive and
  `ai_profile_fallback` is enabled, analyze attributable structured artifacts
  directly and record the fallback and confidence.
- Never install, upgrade, reconfigure, or elevate permissions for a profiler
  unless the task separately authorizes that system change.

## TLE

`analysis.extensions.tle` defaults to `disabled`.

- `disabled`: do not inspect `tle-wiki`, probe the API, load the TLE Skill, or
  implement with TLE.
- `auto`: assess the configured project `tle-wiki` and task-declared read-only
  API probe. If either is unavailable, record the reason and continue without
  TLE.
- `required`: if the Wiki/API assessment fails, the necessary primitive is
  absent, or TLE execution cannot produce a correct candidate, stop and restore
  the best correct checkpoint rather than silently downgrading the requirement.

For remote tasks, list and read Wiki files only through KernelPilot's controlled
read-only Wiki commands. Treat that project's Wiki as the authority for the
installed version and API; do not infer interfaces from another TLE version or
from a conda environment name. Use only documented primitives. Record TLE API,
compilation, lowering, runtime, correctness, and performance errors. If a
measured bottleneck needs an operation that neither Triton nor the available
TLE API can express, report the concrete missing primitive and required
semantics instead of inventing an interface.

## Evidence and completion

Before each stage, update `runs/workflow-state.json`. After each stage, append
the action, result, evidence, and selected transition to
`runs/stage-events.jsonl`. Keep the candidate and benchmark ledgers consistent
with their schemas.

Finish with `docs/final-report.md` using `templates/final-report.md`, and return
the structured result required by `schemas/run-result.schema.json`. The report
must identify the restored/deployed best correct candidate, comparator, metric,
per-shape and aggregate result, correctness, generalization evidence, remaining
gap, stop reason, rejected candidates, unavailable capabilities, TLE assessment,
and every workflow error or recovery in execution order.

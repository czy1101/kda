# Kernel Optimization Task

Read, in order, the task workspace `AGENTS.md`, `.kernelpilot/task.yaml`, the
KDA workflow at `workflows/kernel-optimization.yaml`, the selected backend
profile, and only the Skills routed for the current stage.

Treat the workflow as the intended execution contract for one complete run.
Use its declared transition for each observed stage result. If a transition,
action, tool, or required artifact cannot be completed, record the deviation
and recovery in `runs/stage-events.jsonl` and the final report.

## Driver contract

This prompt is driver-neutral; KernelPilot may launch Codex or OpenCode. Codex
returns the structured result through its output schema. OpenCode must write the
same JSON object to the final-result path named in the generated prompt, even
when the run stops without a performance win. Do not start a nested `codex` or
`opencode` session. OpenCode has no Codex-style filesystem sandbox, so
`constraints.allowed_paths` and `constraints.forbidden_paths` are binding
without exception.

## Bottleneck localization

Diagnosis is two steps, and the order is binding. At every `diagnose` stage:

1. **Screen six layers briefly** and write `runs/bottleneck-screen.json`
   (`schemas/bottleneck-screen.schema.json`): `algorithm_numerics`,
   `operator_dataflow`, `kernel_implementation`, `launch_config_autotune`,
   `compiler_codegen`, `runtime_system`. For each, record the evidence you
   actually have, whether it is worth continuing, the single most
   decision-relevant unknown, and what the next experiment would let you decide.
   The six layers are viewpoints, not six mandatory device experiments; mark a
   layer `not_applicable` or record insufficient evidence rather than inventing
   an experiment for it.
2. **Select one deep dive** and state its hypothesis, control, expected
   observable change, both falsification conditions, the next action for each
   outcome, and its budget and stop condition.

Follow `docs/bottleneck-localization.md` for the layer table, the deep-dive
routing rules, the four-link evidence chain, the five change classes and the
attribution-strength levels. Do not promote a symptom into a layer's conclusion,
and do not run an experiment whose two outcomes lead to the same next action.

When the question is about generated code, also read
`docs/enflame-compiler-diagnostics.md`. Classify the candidate as `kernel`,
`config`, `compiler` or `runtime` in the screen; a source rewrite whose gain
partly comes from the autotuner switching configuration is two changes and needs
a fixed-config control.

Existing evidence that already supports a candidate is enough. Stop collecting
once the decision is made, and report the attribution level you reached instead
of the one the story suggests.

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

## Shared-memory and resource policy

Before implementing a candidate, read
`<kda>/docs/shared-memory-and-resource-policy.md` in the resolved KDA repository.
Reject explicit empty shared-memory allocation followed by stores/copies,
including dummy initialization, and independent register-prefix/shared-memory
budget tuning or standalone register caps. Permit documented interfaces taking
actual data to construct/copy shared storage, compiler-managed staging, and
normal algorithmic tile/warp/stage tuning. Check the full data flow, not just
parameter names; passing tests or a speedup does not waive this user convention.
If no permitted API exists, use an allowed ordinary path or report the missing
capability under the task policy. Restore only a compliant correct checkpoint;
report when none exists and preserve frozen files and historical evidence.

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

For GCU tasks, read KDA `knowledge/enflame/README.md` and only relevant cards.
Also read KDA `docs/enflame-s60-runbook.md`. Record card IDs, environment
differences and the resulting falsifiable hypothesis in `docs/plan.md`. This
applies even without a matching backend profile. The knowledge index is a
reference, not a profiler/skill declaration; historical results require fresh
validation and do not expand task permissions.
In the final report, record confirming or contrary evidence for reused cards.
When new reusable Enflame mechanisms, paths or failure modes emerge, update
the knowledge cards, catalog and source pointers with their evidence and
limits. Keep raw artifacts in the task workspace and report the update.
If repository write access is unavailable, preserve a proposed update in task
docs; knowledge maintenance does not permit changes to frozen task files.

If `target.profile` is present, validate and use it. Otherwise use the unique
backend profile matching the declared backend/device/architecture/language. Do
not guess when multiple profiles match; continue without profile-routed skills
when none matches. Applicable local knowledge cards remain reference material.

Run backend tool discovery in the exact task environment. A backend declaration
is a probe allowlist, not proof that a tool is installed or usable. Select only
tools whose live probe succeeded, and preserve their versions and failures in
`runs/analysis-capabilities.json`.

- Use kernel counters for execution-pipeline or memory questions, timelines for
  launch/synchronization/host gaps, framework tools for operator attribution,
  and compiler artifacts for lowering or generated-code questions.
- Even without a device, compiler artifacts can be compared by identity:
  `scripts/compile_artifacts.py index` and `compare` report stage hints, sizes,
  hashes, container kinds, cache keys and bundle targets, and list what they
  cannot resolve. They are not evidence about performance.
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

## S60 execution rules

For Enflame S60/GCU300 tasks, `docs/enflame-s60-runbook.md` is binding:

- Use only a confirmed idle physical card 6 or 7. Set
  `TOPS_VISIBLE_DEVICES=<actual-card>` on every process that can launch device
  work. For non-interactive `docker exec`, repeat `-e TOPS_VISIBLE_DEVICES=...`
  and `-w <workspace>` on every invocation; do not rely on a prior shell.
- Fix semantics, shape, dtype, layout, device, correctness threshold, baseline,
  measurement scope, and rollback condition before editing.
- Measure the full operator, preprocessing, and hotspot in stages before making
  one falsifiable hypothesis. Keep source, compiler IR, intrinsic, final
  instruction, and hardware-counter evidence distinct. An emitted intrinsic does
  not prove unit utilization or a speedup.
- Run correctness first. Compare A/B on the same card, input, layout, warmup,
  and timed region. Save raw samples, spread, compilation configuration, and
  failed candidates. Re-run the required shape set before promotion.
- TLE stays disabled unless the task enables it. Verify the installed Wiki,
  API probe, minimal compilation, and correctness before using it on a hotspot.
  Missing counters do not prove a TLE gap.

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

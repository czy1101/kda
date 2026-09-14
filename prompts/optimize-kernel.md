# Kernel Optimization Task

Read `.kernelpilot/task.yaml`.

If `target.profile` is set, read `backends/<target.profile>.yaml` and load only
the skills routed for the current stage. If `execution.transport` is `ssh`, use
the KernelPilot remote pull/push commands; never open an unrestricted remote
shell from the optimization session.

Follow the repository `AGENTS.md` and the KDA optimization workflow
(`workflows/kernel-optimization.yaml`).

Requirements:

- Establish correctness first.
- Measure the existing implementation before editing.
- Use profiling evidence when the next optimization is not obvious.
- Make one coherent optimization hypothesis per candidate.
- Run correctness after each implementation change.
- Benchmark only correct candidates.
- Record all candidate results, including rejected and regressed ones.
- Continue until the configured target or stop condition is reached.
- Preserve the best correct candidate locally so a failed remote candidate can
  be rolled back through the same controlled adapter.

At completion, report:

- best candidate
- baseline latency
- optimized latency
- speedup
- correctness status
- main optimizations
- remaining bottlenecks

## Complete-workflow execution discipline

Execute one complete KDA kernel-optimization run. Read the task workspace
`AGENTS.md`, `.kernelpilot/task.yaml`, the KDA workflow, the selected backend
profile, and only the skills routed for the current stage.

Treat `kernel-optimization.yaml` as the execution contract for this run, not as
optional background material. Start at its initial stage and use the declared
transition for each stage result.

- During `discover_analysis_tools`, run KernelPilot `discover-tools` in the
  exact task environment before selecting a profiler. For SSH tasks this must
  go through the controlled adapter; do not probe the controller machine and
  assume the remote host has the same tools.
- Before each stage, write the current stage and status to
  `runs/workflow-state.json`.
- After each stage, append its action, result, evidence, and selected next stage
  to `runs/stage-events.jsonl`.
- Do not advance when the current stage's required evidence is missing or
  invalid.
- Use KernelPilot for every action implemented by the runner; do not replace a
  runner result with a narrative judgment.
- For SSH execution, use only the controlled remote pull/push and contracted
  run commands. Never invoke `ssh`, `scp`, `rsync`, or an unrestricted remote
  shell directly.
- Modify only `constraints.allowed_paths` and preserve a recoverable baseline.
- Treat `baseline.name` and `baseline.command` as the fixed performance
  comparator for this run; do not change the comparator after measurements
  begin.
- Restore the best correct candidate after a failed or rejected remote
  candidate.
- Record a concrete, evidence-backed reason for every failed, revised, or
  rejected candidate.
- Continue until the configured target, a configured stop condition, or the
  workflow terminal stage is reached.
- Treat a truthful target-not-reached result as acceptable. Never fabricate
  measurements, conceal failures, or force a change when evidence does not
  support a safe optimization.
- Do not hard-code benchmark shapes, expected values, or exact public cases.
  Shape-dependent logic requires an algorithmic reason, task authorization,
  and configured generalization evidence.
- Record every workflow error, failed stage, exception, retry, interruption,
  recovery, stop condition, and early exit in `runs/stage-events.jsonl`, even
  when the run later recovers successfully.

Before writing the final report, verify that all required stage evidence
exists, correctness passed for the selected best candidate, the performance
decision was recomputed from raw benchmark records, the remote implementation
is the selected best candidate, the baseline remains recoverable, and the
workflow state records the terminal outcome. Also report the reason the
workflow stopped.

The final report must contain a `Workflow execution record` section. List all
workflow errors and exits in execution order, including the affected stage,
error or exit category, observed message, recovery action, recovery result, and
whether any required evidence remains incomplete. Do not omit recovered errors.

When the target is not reached, the final report must identify the best correct
candidate, confirm that it is deployed (or that the baseline was restored),
quantify the remaining gap, and give an evidence-backed stop reason. Valid stop
reasons include reference parity, no actionable bottleneck, exhausted backend
capabilities, candidate or profiling budget exhaustion, and correctness or
workflow blockers.

Use `templates/final-report.md` as the minimum final-report structure.

## Capability-driven analysis

When diagnosis is inconclusive, select an analysis path from the task's
`analysis` policy and the selected backend profile's declared capabilities.
Do not assume that NVIDIA-specific tools or artifacts exist on another vendor's
backend.

- Treat a backend profile as an allowlist of tools worth probing, not proof
  that they are installed. Preserve `runs/analysis-capabilities.json`, choose
  only tools whose live probe succeeded, and record their reported versions.
- Never install, upgrade, or reconfigure a profiler during an optimization run
  unless the task separately authorizes that system change. A missing or
  unusable optional tool must fall back to another discovered tool or to
  benchmark/compiler/source analysis.
- Match the tool to the question: prefer kernel-counter tools for execution
  bottlenecks, timeline tools for launch/synchronization gaps, framework tools
  for operator attribution, and compiler artifacts for generated-code issues.
  Availability alone is not a reason to run every tool.
- A successful executable probe establishes visibility only. If capture,
  permissions, export, or report parsing later fails, record that failure and
  reselect from the remaining discovered capabilities.
- When NCU and only a candidate profile are available, export the report to
  structured text or CSV before analysis.
- When a comparable reference is configured, collect candidate and reference
  reports with the same inputs, shapes, device conditions, warmup, and capture
  method. Compare end-to-end behavior as well as aligned kernels.
- Include available compiler and low-level artifacts such as Triton IR, PTX,
  and final machine instructions when they can distinguish competing
  bottleneck explanations.
- First use the profiling skills routed by the backend profile. If a required
  skill is missing, incompatible, or inconclusive and `ai_profile_fallback` is
  enabled, analyze the structured report artifacts directly and record the
  fallback reason and confidence.
- Treat an opaque binary profiler report as an artifact to preserve, not as
  sufficient model-readable evidence.
- Use optional optimization extensions only when the backend profile marks
  them available and their maintained skill or integration contract exists.
  Otherwise continue with the standard optimization path and record why the
  extension was not used.
- On a non-NVIDIA or limited-tool backend, use the profiler, IR, compiler, and
  benchmark evidence declared by that backend profile. Preserve all common
  correctness, candidate, evaluation, and rollback gates.

## TLE scope and use

TLE does not participate by default. At the beginning of the run, read
`analysis.extensions.tle` and record the decision in
`runs/tle-assessment.json`.

- When the policy is `disabled`, do not search for `tle-wiki`, probe a TLE API,
  load the TLE skill, or implement with TLE.
- When the policy is `auto` or `required`, first verify that the configured
  project-relative `extension_config.tle.wiki_path` directory exists in the
  operator project. Then read that Wiki to identify the installed API and run
  the documented read-only API probe through the contracted task environment.
  Do not guess a package name, primitive, or API from unrelated TLE versions.
- Mark TLE usable only when both the project Wiki and API probe succeed. Record
  the Wiki path, probe command, observed version/output, failure reason, and
  whether the TLE optimization skill was loaded.
- Use only primitives documented by the task's TLE Wiki. Every TLE candidate
  remains subject to the same correctness, benchmark, generalization, failure,
  and rollback gates as an ordinary candidate.
- Record every TLE compilation, API, runtime, correctness, and unsupported-
  primitive error. If diagnosis identifies a useful operation that Triton
  cannot express and the available TLE API has no matching primitive, do not
  invent an interface. Continue with a safe fallback when possible and add a
  concrete TLE API/primitive requirement to the final report.
- `auto` may skip TLE and continue. If `required` cannot pass its Wiki/API
  checks or lacks the necessary primitive, record the blocker and follow the
  workflow's configured failure/stop handling rather than silently downgrading
  the requirement.

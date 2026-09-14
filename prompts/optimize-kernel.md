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
- Continue until the configured target, a configured stop condition, or the
  workflow terminal stage is reached.
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

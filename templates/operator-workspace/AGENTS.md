# Operator Workspace Instructions

This workspace is one kernel optimization task and follows the Kernel Design
Agents (KDA) workflow.

## Task contract

Read `.kernelpilot/task.yaml` before doing anything else. It defines the
objective, target, implementation path, correctness command, benchmark command,
baseline, performance goal, constraints, and stop conditions.

If the contract is missing, or a required field is still a placeholder, stop and
ask instead of guessing.

Run contracted correctness, benchmark, and profile commands through
`scripts/kernelpilot.py run` when that runner is available. It applies the
contract's `environment.setup` first, so conda activation, library paths, and
device selection are reproduced instead of merely selecting a Python binary.

When `execution.transport` is `ssh`, Codex remains on the controller. Never
invoke `ssh`, `scp`, or `rsync` directly. Use `kernelpilot.py remote pull` to
materialize allow-listed files under `remote-worktree/`, edit that mirror, and
use `kernelpilot.py remote push` to deploy. The adapter checks both the path
allow-list and the last observed remote hash. All test and benchmark commands
still go through `kernelpilot.py run`.

When `target.profile` is set, read the matching file under KDA `backends/` and
load only the skills routed for the current stage.

Before selecting a performance-analysis tool, run KernelPilot `discover-tools`
in the contracted environment and preserve `runs/analysis-capabilities.json`.
For SSH tasks the probes run on the executor through the controlled adapter.
Use only backend-declared tools whose probe succeeded; do not install or
reconfigure a missing tool as part of the optimization run.

## Principles

1. Correctness has priority over performance.
2. Never claim a performance improvement without measured evidence.
3. Establish a reproducible baseline before optimizing.
4. Prefer profiling evidence over speculative optimization.
5. Every optimization attempt is a candidate.
6. Record rejected and regressed candidates with a reason; never discard them silently.
7. Do not modify the reference implementation, the correctness command, or the benchmark command.
8. Target-not-reached is acceptable when supported by evidence. Never invent
   measurements, suppress failures, or continue changing code only to claim a win.
9. Record a concrete reason for every failed or rejected candidate and restore
   the best correct candidate before continuing or reporting.
10. Do not hard-code benchmark shapes or values. Shape-dependent optimization
    needs an algorithmic justification, task authorization, and generalization
    evidence when configured.

## Loop

1. Inspect the implementation and reference; write `docs/draft.md`.
2. Do not start implementation until `docs/draft.md` exists.
3. Verify correctness and record the baseline measurement.
4. Diagnose the bottleneck; profile only when the next change is not already justified by evidence.
5. Implement one coherent hypothesis per candidate.
6. Deploy through the configured transport, run correctness after each change,
   and benchmark only correct candidates.
7. Record every attempt in `runs/candidates.jsonl` and `runs/benchmark.csv`.
8. Keep the best candidate; continue until the target or a stop condition is reached.
9. Produce the final report.

After a rejected remote candidate, restore the best correct mirror and push it
through the same adapter before continuing.

Stop honestly when the reference is already within the configured parity
tolerance, no actionable bottleneck remains, the backend exposes no further
safe analysis capability, or a configured budget is exhausted. Record the stop
reason and remaining gap in the final report.

Never silently broaden a single number into a multi-shape performance claim.
Use `goal.direction`, `goal.aggregation`, and `goal.max_regression` to decide
promotion, and retain the per-shape measurements in `runs/benchmark.csv`.

## Evidence files

| File | Contents |
|---|---|
| `docs/draft.md` | Initial plan draft. Required before implementation starts. |
| `docs/plan.md` | The executable plan. |
| `runs/candidates.jsonl` | One JSON record per candidate: id, parent, hypothesis, changes, correctness, latency, baseline, speedup, decision, reason. |
| `runs/benchmark.csv` | Per-shape measurements with columns `candidate,shape,metric_value,baseline_value,unit,status`. The runner recomputes promotion from these values. |
| `runs/analysis-capabilities.json` | Backend-declared tools probed in this exact environment, with availability, versions, and default selection. |
| `profile/` | Profiler output or report summaries. |

A future reader must be able to reconstruct what changed, what was measured, and
why a candidate was promoted.

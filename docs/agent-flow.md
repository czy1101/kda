# Agent Flow

Kernel Design Agents is a repeatable loop for agent-driven implementation work. The loop is useful when a task needs both exploration and evidence-based promotion decisions.

## Principle

Keep the reusable workflow separate from the task workspace. This repository explains the flow. The task workspace owns code, tests, datasets, benchmark scripts, private rules, and generated artifacts.

The controller and executor may also be separate. The agent driver (Codex or
OpenCode), prompts, workflow state, and evidence stay on the controller. A GPU
server that cannot run either driver is an execution-only host reached through
the controlled SSH adapter. The adapter runs only task-contracted commands and
synchronizes only allow-listed implementation files.

MLSys-style benchmark work is one possible application of this loop. The same flow can also be used for compiler passes, runtime kernels, infrastructure changes, or other performance-sensitive tasks.

## The nine-step operator optimization loop

This is the canonical form of the loop. The workflow at
[../workflows/kernel-optimization.yaml](../workflows/kernel-optimization.yaml)
implements it; the stage column below is the mapping, and any stage whose action
cannot be completed is a recorded deviation rather than a silent skip.

| Step | What it settles | Workflow stages | Done when |
|---|---|---|---|
| 1 | **Fix the task.** Semantics, reference, shape class, dtype and accumulation, layout and strides, tolerance, device, correctness command, baseline identity, measurement scope, warmup and timer, target, regression limit, rollback condition | `init`, `select_backend`, `prepare_execution` | No contract field is a placeholder |
| 2 | **Verify correctness of the unmodified implementation.** A candidate cannot be compared against a baseline nobody proved correct | `correctness` | The reference implementation passes the contracted check |
| 3 | **Establish the baseline** under the same conditions every later candidate will be measured in | `baseline`, `benchmark`, `evaluate` | A recorded baseline candidate and its raw samples exist |
| 4 | **Locate the bottleneck.** Screen the six layers, then select one deep dive that can change a decision | `diagnose`, and the profiling stages it selects | `runs/bottleneck-screen.json` states the selected layer and a falsifiable deep dive |
| 5 | **Select the strategy and propose candidates.** Name the change class and the implementation means, then answer the five questions and fill the eight items | `select_analysis_strategy`, `select_optimization_strategy`, `resolve_optional_extensions` | `docs/plan.md` holds the candidate with its means, its expected signature and its acceptance criterion, written before implementation |
| 6 | **Validate the capability, implement, and check.** Establish that the means works in this environment before the candidate depends on it; then implement one coherent candidate and run correctness | `optimize_with_extension` or `optimize`, `deploy_candidate`, `correctness_candidate` | The required capability gates pass, and the candidate is correct |
| 7 | **Measure under the same conditions.** A/B against the current best on the same card, input, layout, warmup and timed region | `benchmark_candidate`, `compare_candidate`, `generalization_check` | Paired raw samples exist and the required shape set was re-run |
| 8 | **Decide keep or drop,** with a reason and a rollback result for every outcome | `keep_candidate`, `reject_candidate`, `restore_best`, `check_stop` | The best correct checkpoint is the deployed implementation |
| 9 | **Commit and accumulate.** Report what was established, at which verification level, and whether anything is reusable beyond this operator | `promote` or `finalize_without_target`, then `report` | `docs/final-report.md` exists and matches the structured result |

Steps 4–6 are the ones that decide whether the run is worth its budget, and they
are the ones most often merged. Keep them separate:

- Step 4 answers *where the cost is* and does not name a change.
- Step 5 answers *what kind of edit* and *in what interface*, and does not
  implement anything.
- Step 6 answers *does that interface work here* before it answers *is the
  candidate correct*.

A run may stop at step 4 with an evidence-backed "no actionable bottleneck". That
is a complete run, not a failure.

For an Enflame target, the means and the capability gates are in
[enflame-interface-assessment.md](enflame-interface-assessment.md); the layers and
change classes stay in [bottleneck-localization.md](bottleneck-localization.md).
The three vocabularies are separate and must not be merged into one enum.

**Split controller and executor.** For SSH tasks, materialize allow-listed source
into `remote-worktree/` before inspection, deploy a candidate through the adapter
before correctness, and use the recorded remote hash to reject concurrent or
out-of-band overwrites. KernelPilot snapshots the initial allowed source and the
current approved best; the best checkpoint is restored on normal completion,
failure, or an interrupt that reaches the runner's cleanup path.

## Seven rules that stay in force

Short forms, each with the place that carries the detail. They hold for every
step above.

1. **Choose the observation depth from the bottleneck, not from the tool.**
   History and cost, not sophistication — [bottleneck-localization.md](bottleneck-localization.md).
2. **Keep JIT, autotune, first load, and warm execution apart.** A cold number and
   a warm number answer different questions; say which one the metric includes.
3. **Compare single-point, configuration and combination gains separately, and do
   not add them.** A gain with two sources needs three numbers and a statement of
   whether they add up.
4. **The expected change must correspond to the layer being changed.** A predicted
   timing drop with no predicted artifact or execution signature is not a
   hypothesis.
5. **Distinguish a cache key, a static operation count, a device image, an actual
   execution and a counter.** Each establishes a different link in the evidence
   chain, and none substitutes for another.
6. **Correctness type, disable policy, device constraints and modification
   permission remain in force.** A speedup, a passing test, or a new interface
   waives none of them.
7. **An evidence-backed stop is allowed.** Reference parity, no actionable
   bottleneck, exhausted capabilities and budget exhaustion are honest terminal
   outcomes; none of them is relabelled as a promotion.

## Task Contract

Each task should state:

- Objective.
- Inputs and outputs.
- Correctness requirements.
- Constraints on implementation language, dependencies, APIs, or deployment.
- Validation command.
- Evaluation command, if different from validation.
- Promotion criteria.

## Evidence Records

Use simple files in the task workspace:

- `docs/draft.md` for the first plan draft.
- `docs/plan.md` for the executable plan. It also carries the per-candidate
  implementation means and the capability-gate results described in
  [enflame-interface-assessment.md](enflame-interface-assessment.md); no ledger
  field records them, so the plan is the only place they exist.
- `benchmark.csv` or another tabular log for measurable results.
- `candidates.jsonl` for candidate names, parent links, and status.
- `profile/` for profiler output or report summaries.
- `runs/analysis-capabilities.json` for live backend-tool discovery evidence.
- `runs/bottleneck-screen.json` for the two-step diagnosis record described in
  [bottleneck-localization.md](bottleneck-localization.md).
- `runs/tle-assessment.json` for the one machine-readable interface assessment
  that exists, and only when the task enables TLE.
- `runs/compile-artifacts.json` and `runs/compile-artifact-diff.json` for
  compiler-artifact identity and layered comparison.
- `runs/` or `outputs/` for generated artifacts.

The exact format is less important than consistency. A future reader should be able to reconstruct what changed, what was measured, and why a candidate was promoted.

## Promotion Rule

Promote a candidate only when it satisfies the task contract and has evidence that it improves or preserves the target metric. If a candidate is rejected, record the reason instead of silently discarding it.

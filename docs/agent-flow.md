# Agent Flow

Kernel Design Agents is a repeatable loop for agent-driven implementation work. The loop is useful when a task needs both exploration and evidence-based promotion decisions.

## Principle

Keep the reusable workflow separate from the task workspace. This repository explains the flow. The task workspace owns code, tests, datasets, benchmark scripts, private rules, and generated artifacts.

The controller and executor may also be separate. Codex, prompts, workflow
state, and evidence stay on the controller. A GPU server that cannot run Codex
is an execution-only host reached through the controlled SSH adapter. The
adapter runs only task-contracted commands and synchronizes only allow-listed
implementation files.

MLSys-style benchmark work is one possible application of this loop. The same flow can also be used for compiler passes, runtime kernels, infrastructure changes, or other performance-sensitive tasks.

## Minimal Loop

1. Define the task contract.
2. Select the backend profile and discover its analysis tools in the real task environment.
3. Let the agent inspect the local workspace.
4. Make the agent write `docs/draft.md`.
5. Convert the draft into an executable plan.
6. Implement the first candidate.
7. Validate correctness.
8. Run configured PR quality gates when present.
9. Measure the target metric when applicable.
10. Record evidence and decide whether to keep, revise, or reject the candidate.
11. Repeat until the promotion criteria are met or the remaining blockers are explicit.

For SSH tasks, materialize allow-listed source into `remote-worktree/` before
inspection, deploy a candidate through the adapter before correctness, and use
the recorded remote hash to reject concurrent or out-of-band overwrites.
KernelPilot snapshots the initial allowed source and the current approved best;
the best checkpoint is restored on normal completion, failure, or an interrupt
that reaches the runner's cleanup path.

## Task Contract

Each task should state:

- Objective.
- Inputs and outputs.
- Correctness requirements.
- Constraints on implementation language, dependencies, APIs, or deployment.
- Validation command.
- Evaluation command, if different from validation.
- Promotion criteria.

KernelPilot always runs a built-in AST import-path audit, even without a
`quality` block. It checks `implementation.path` plus any configured extra
source paths and rejects relative imports such as `from .index import ...`;
use the complete repository package path instead. An optional `quality` block
adds five task-local PR gates: public API import/export, module layering,
operator name and registry consistency, pytest marker collection, and unified
operator dispatch. Their commands run in the same environment as correctness
and benchmark.

## Evidence Records

Use simple files in the task workspace:

- `docs/draft.md` for the first plan draft.
- `docs/plan.md` for the executable plan.
- `benchmark.csv` or another tabular log for measurable results.
- `candidates.jsonl` for candidate names, parent links, and status.
- `profile/` for profiler output or report summaries.
- `runs/analysis-capabilities.json` for live backend-tool discovery evidence.
- `runs/quality.json` for configured public API, module-layer, absolute-import,
  naming/registry, pytest-marker, and unified-dispatch checks.
- `runs/` or `outputs/` for generated artifacts.

The exact format is less important than consistency. A future reader should be able to reconstruct what changed, what was measured, and why a candidate was promoted.

## Promotion Rule

Promote a candidate only when it satisfies the task contract and has evidence that it improves or preserves the target metric. If a candidate is rejected, record the reason instead of silently discarding it.

At every run's finalization—whether the goal was reached or a truthful
target-not-reached result was recorded—the default workflow reviews the final
best correct candidate for reusable knowledge. If the gain is material and
the applicability is understood, run
[`scripts/record_knowledge.py`](../scripts/record_knowledge.py) with the task
workspace and candidate id. Place the note under the matching
`backends/<vendor>/note/` directory when one exists, such as
`backends/hygon/note/`; retain `knowledge/` as the generic fallback.

The recorder verifies correctness, candidate decision, per-shape benchmark,
aggregate gain, and regression gates before writing a Markdown note plus JSON
record and updating the index. It requires the author to state the mechanism,
validated scope, transfer lesson, and limitations; benchmark correlation alone
is not accepted as a causal explanation. The default material-gain threshold
is 5%, and it can be raised for noisy or high-impact tasks. If any gate fails
or the result is not transferable, record the specific skip reason in the stage
events and final report; do not publish it as confirmed knowledge. Publication
failure must not affect restoration or reporting of the best correct candidate.

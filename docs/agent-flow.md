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
2. Let the agent inspect the local workspace.
3. Make the agent write `docs/draft.md`.
4. Convert the draft into an executable plan.
5. Implement the first candidate.
6. Validate correctness.
7. Measure the target metric when applicable.
8. Record evidence and decide whether to keep, revise, or reject the candidate.
9. Repeat until the promotion criteria are met or the remaining blockers are explicit.

For SSH tasks, materialize allow-listed source into `remote-worktree/` before
inspection, deploy a candidate through the adapter before correctness, and use
the recorded remote hash to reject concurrent or out-of-band overwrites.

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
- `docs/plan.md` for the executable plan.
- `benchmark.csv` or another tabular log for measurable results.
- `candidates.jsonl` for candidate names, parent links, and status.
- `profile/` for profiler output or report summaries.
- `runs/` or `outputs/` for generated artifacts.

The exact format is less important than consistency. A future reader should be able to reconstruct what changed, what was measured, and why a candidate was promoted.

## Promotion Rule

Promote a candidate only when it satisfies the task contract and has evidence that it improves or preserves the target metric. If a candidate is rejected, record the reason instead of silently discarding it.

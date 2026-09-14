# Prompt Templates

This directory stores generic prompts for the Kernel Design Agents workflow.

The templates are intentionally task-agnostic. Fill in the task objective, constraints, validation command, and promotion criteria before starting an agent session in a separate implementation workspace.

## Available Templates

| Path | Purpose |
|---|---|
| `basic-flow.md` | Minimal prompt for research, planning, implementation, validation, and iteration. |
| `optimize-kernel.md` | Entry point for a full optimization run driven by `.kernelpilot/task.yaml`. |
| `analyze-kernel.md` | Understand the operator, its layouts, and its baseline without editing code. |
| `profile-kernel.md` | Produce profiling evidence and one falsifiable hypothesis from it. |
| `final-review.md` | Summarize the best candidate, the evidence, and the remaining bottlenecks. |

## How To Use

1. Create the task contract: copy `../templates/task.yaml` to
   `<task-workspace>/.kernelpilot/task.yaml` and fill it in.
2. Enter the task implementation workspace.
3. Copy the relevant template content into the agent session. For a full run,
   start from `optimize-kernel.md`; it reads the task contract and delegates the
   method to `AGENTS.md`, `workflows/kernel-optimization.yaml`, and the Skills.
4. Ask the agent to read the workspace and write `docs/draft.md`.
5. Convert that draft into an executable plan.
6. Run the implementation loop with validation after each meaningful change.

The prompts stay thin on purpose: they trigger a task. The method lives in the
workflow and the Skills, and the task-specific facts live in `.kernelpilot/task.yaml`.

Task-specific prompts should live with the task they describe. Do not add benchmark-specific datasets, acceptance tables, or private evaluator details to this generic repository.

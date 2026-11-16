# KDA Codex Instructions

This repository defines a reusable workflow for GPU kernel optimization.

## Core principles

1. Correctness has priority over performance.
2. Never claim a performance improvement without measured evidence.
3. Establish a reproducible baseline before optimization.
4. Prefer profiling evidence over speculative optimization.
5. Every optimization attempt is a candidate.
6. Failed or regressed candidates must be recorded rather than silently discarded.
7. Keep the task workspace separate from reusable KDA workflow assets.
8. Keep Codex on the controller when an execution host cannot run it; use the
   controlled adapter for contracted commands and allow-listed source sync.
9. Do not bypass the configured transport with direct `ssh`, `scp`, or `rsync`.
10. An evidence-backed run that does not reach its performance target is an
    acceptable outcome. Never fabricate an improvement or hide a failed attempt.
11. Every failed or rejected candidate needs a concrete reason and must leave
    the remote implementation at the best correct candidate.
12. Do not hard-code public benchmark shapes, values, or dispatch boundaries
    solely to pass the measured cases. Shape specialization requires an
    algorithmic justification and task authorization.
13. Treat backend analysis-tool declarations as a probe allowlist, not proof of
    installation. Select profiling only from current task-environment discovery
    evidence; do not install or reconfigure tools during the run.
14. Save only correctness- and benchmark-approved candidates as the best
    checkpoint. KernelPilot restores that checkpoint when the run exits.
15. Keep metric handling generic: the task's metric, direction, unit, comparator,
    aggregation, and target mode define promotion.
16. For GCU tasks, consult `knowledge/enflame/README.md` and relevant cards
    before choosing a candidate, even without a matching profiler profile.
    Record card IDs and applicability in the task plan. Revalidate historical
    observations; knowledge never overrides task constraints or evidence gates.
17. Maintain the Enflame knowledge base when optimization reveals a new
    reusable mechanism, path, limitation or instructive failure. Update cards,
    catalog and evidence pointers; distinguish observations from hypotheses
    and report the update. Preserve raw artifacts in the task workspace. If
    repository writes are unavailable, leave a proposed update in task docs;
    do not bypass permissions or modify frozen tests/benchmarks.
18. Admit only generalizable knowledge: a reusable mechanism, interface,
    applicability condition, or evidence-backed backend limitation. State its
    version/layout/semantic boundaries; generalizable does not mean valid on
    every device. Keep operator-specific winners, tuned constants, shape
    tables, timings, candidate ledgers and run narratives in the task workspace.
    A task result alone is not a knowledge card. External evidence pointers and
    hashes are allowed; they support the reusable claim without copying the
    experiment report into the knowledge base. Do not auto-publish run reports.

## Shared-memory and resource policy

Apply the user's [shared-memory and resource policy](docs/shared-memory-and-resource-policy.md).
Do not explicitly allocate empty shared memory and fill it later, or tune
independent register-prefix/shared-memory budgets or standalone register caps.
Use documented interfaces initialized from actual algorithm data or compiler-
managed storage. Normal algorithmic tile/warp/stage tuning remains permitted.
Review this before implementing a candidate; speed or correctness does not
waive the convention. Preserve task permissions and historical evidence.

## Optimization loop

For every kernel optimization task:

1. Read the task contract.
2. Inspect the implementation and reference.
3. Verify correctness.
4. Establish baseline performance.
5. Profile the current bottleneck when needed.
6. Write an optimization hypothesis.
7. Implement one coherent candidate.
8. Run correctness.
9. Run benchmark.
10. Record evidence.
11. Keep, revise, or reject the candidate.
12. Repeat until the target or stop condition is reached.

## Repository rules

- Use English for repository-facing files, comments, documentation, prompts, and commit messages.
- Keep task-specific prompts, datasets, validators, generated implementations, benchmark logs, and candidate artifacts out of this repository.
- Treat benchmark competitions, including MLSys-style work, as downstream applications of KDA rather than as the scope of this repository.
- Put generated outputs in `runs/`, `outputs/`, or `profile/`; these paths are ignored by git.
- Prefer documenting reusable workflow mechanics over documenting one task's private harness or acceptance thresholds.

## Where each kind of content belongs

| Content | Location |
|---|---|
| "Optimize this specific kernel" | `.kernelpilot/task.yaml` in the task workspace |
| Performance goal and stop conditions | [schemas/task.schema.yaml](schemas/task.schema.yaml), `.kernelpilot/task.yaml` |
| "Run correctness after every change" | this file, [workflows/kernel-optimization.yaml](workflows/kernel-optimization.yaml) |
| "How to discover/select/profile tools" / "how to analyze a benchmark" | a Skill |
| "How to optimize SM90 / Triton tiling" | a Skill or its `references/` |
| Which skills apply to a chip, architecture, and language | `backends/<id>.yaml` |
| Local versus execution-only SSH placement | `.kernelpilot/task.yaml` `execution` block |
| Remote command dispatch and source synchronization | `scripts/controlled_ssh.py` |
| "Which candidate is fastest right now" | `runs/candidates.jsonl`, `runs/benchmark.csv` |
| Which stage the task is currently in | the task workspace, not this repository |
| Reusable, evidence-scoped Enflame optimization observations | `knowledge/enflame/` (raw task artifacts stay in their workspaces) |

The reusable loop is defined here; task-specific prompts, harnesses, and results
stay in the task workspace. See [docs/agent-flow.md](docs/agent-flow.md) for the
underlying loop and [prompts/README.md](prompts/README.md) for how to start a
session.

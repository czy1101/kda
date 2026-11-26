# KDA Agent Instructions

This repository defines a reusable workflow for GPU kernel optimization.

## Core principles

1. Correctness has priority over performance.
2. Never claim a performance improvement without measured evidence.
3. Establish a reproducible baseline before optimization.
4. Prefer profiling evidence over speculative optimization.
5. Every optimization attempt is a candidate.
6. Failed or regressed candidates must be recorded rather than silently discarded.
7. Keep the task workspace separate from reusable KDA workflow assets.
8. Keep the agent on the controller when an execution host cannot run it; use
   the controlled adapter for contracted commands and allow-listed source sync.
   KernelPilot supports Codex and OpenCode; select the driver explicitly for a
   reproducible run.
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

## Driver notes

Codex and OpenCode share the same task contract, workflow, evidence files, and
checkpoint gates. Codex receives a JSON output schema; OpenCode is instructed to
write the same structured result to a contracted file because its CLI has no
`--output-schema` equivalent. OpenCode does not provide Codex's filesystem
sandbox, so the task allowlist, forbidden-path hashes, and best-checkpoint
restore are mandatory rather than advisory. Do not start a nested agent session.

For S60/GCU300 work, also follow
[docs/enflame-s60-runbook.md](docs/enflame-s60-runbook.md): use only the
confirmed idle physical device, set `TOPS_VISIBLE_DEVICES` on every launching
process, and do not infer hardware utilization from the presence of an
intrinsic or a compiler artifact.

## Shared-memory and resource policy

Apply the user's [shared-memory and resource policy](docs/shared-memory-and-resource-policy.md).
Do not explicitly allocate empty shared memory and fill it later, or tune
independent register-prefix/shared-memory budgets or standalone register caps.
Use documented interfaces initialized from actual algorithm data or compiler-
managed storage. Normal algorithmic tile/warp/stage tuning remains permitted.
Review this before implementing a candidate; speed or correctness does not
waive the convention. Preserve task permissions and historical evidence.

## Optimization loop

Nine steps, defined in [docs/agent-flow.md](docs/agent-flow.md) with their stage
mapping. They settle different questions and are not interchangeable:

1. Fix the task: semantics, shapes, dtypes, layout, tolerance, device, commands,
   baseline identity, measurement scope, target, regression limit, rollback.
2. Verify correctness of the unmodified implementation.
3. Establish the baseline under the conditions later candidates will be measured in.
4. Locate the bottleneck: screen the six layers, then select one deep dive that
   can change a decision — see
   [docs/bottleneck-localization.md](docs/bottleneck-localization.md).
5. Select the strategy: the change class and the implementation means, with the
   candidate's expected signature and acceptance criterion written down first.
6. Validate the capability, implement one coherent candidate, and run correctness.
7. Measure under the same conditions as the baseline.
8. Decide keep or reject, with a reason and a rollback result.
9. Report what was established, at which verification level, and whether it is
   reusable beyond this operator.

Step 4 names no change; step 5 implements nothing; step 6 establishes that the
interface works before it establishes that the candidate is correct. A run may end
at step 4 with an evidence-backed "no actionable bottleneck".

Three vocabularies stay separate: the six diagnostic **layers**, the five
**change classes** (`kernel`, `config`, `compiler`, `runtime`, `measurement`), and
the **implementation means**. A layer does not choose a class and a class does not
choose a means. For Enflame targets the means table, the eight items each
candidate records in `docs/plan.md`, and the five capability gates are in
[docs/enflame-interface-assessment.md](docs/enflame-interface-assessment.md);
none of them is machine-checked. Choosing a means does not move the bottleneck
between layers.

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
| OpenCode host/controller entry points | `docs/opencode.md` |
| S60 device, measurement, and TLE rules | `docs/enflame-s60-runbook.md` |
| How to choose a diagnosis direction and stop | `docs/bottleneck-localization.md` |
| How to choose an implementation means and prove it works before a candidate depends on it | `docs/enflame-interface-assessment.md` |
| How to read the S60 compile chain and artifacts | `docs/enflame-compiler-diagnostics.md` |
| "Which candidate is fastest right now" | `runs/candidates.jsonl`, `runs/benchmark.csv` |
| Which stage the task is currently in | the task workspace, not this repository |
| Reusable, evidence-scoped Enflame optimization observations | `knowledge/enflame/` (raw task artifacts stay in their workspaces) |

The reusable loop is defined here; task-specific prompts, harnesses, and results
stay in the task workspace. See [docs/agent-flow.md](docs/agent-flow.md) for the
underlying loop and [prompts/README.md](prompts/README.md) for how to start a
session.

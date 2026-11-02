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
16. Always run KernelPilot's built-in import-path audit for each optimization
    task. Use repository-qualified absolute package imports for Python modules in
    submitted code. For example, write
    `from flag_attn.runtime.backend._hygon.FLA.index import ...`, not
    `from .index import ...`. This is a Python module path, not a filesystem or
    server path. The import audit is mandatory even when no `quality` block is
    present. When PR quality gates are declared, also require passing public
    API, module-layer, naming/registry, pytest-marker, and unified-dispatch
    checks before benchmarking or promoting a candidate.
17. At every optimization run's finalization, review the best correct candidate
    for reusable knowledge. By default, publish a note with
    `scripts/record_knowledge.py` when its correctness, benchmark, material-gain,
    regression, and transferability gates pass; otherwise record why it was
    skipped. Include the device, architecture, operator, applicability,
    limitations, and evidence. Never publish rejected, failed, inconclusive,
    or unsupported candidates as confirmed knowledge. A rejected candidate that
    would otherwise be retried belongs in `scripts/record_rejection.py` negative
    knowledge instead: state what was measured, why it failed, and the condition
    that would justify a retry.
18. After confirming the execution backend and device from the task contract and
    live preflight, inspect the matching vendor notes under `backends/` before
    forming optimization hypotheses. For Hygon, read
    `backends/hygon/note/INDEX.md`; for another domestic accelerator, read the
    corresponding `backends/<vendor>/note/INDEX.md` when present. Apply findings
    only within their recorded scope and state when no matching notes exist.
    Treat a `provisional` note as a hypothesis to re-measure, and check
    `revalidate_after` before reusing a note after a DTK, driver, or device
    change. For Hygon, route the measured bottleneck through
    `backends/hygon/note/METHODS.md`, check `backends/hygon/note/rejected/`
    before retrying an experiment, and use
    `backends/hygon/note/OPERATOR-PORTING.md` for a new operator.
19. For Hygon performance analysis, require a usable profiler capture before
    making bottleneck claims or promoting a candidate. Prefer rocprofv2 kernel
    traces; use rocprof only after its task-local capture smoke test succeeds.
    Record device-state and architecture probes as supporting context.

## Optimization loop

For every kernel optimization task:

1. Read the task contract.
2. Inspect the implementation and reference.
3. Verify correctness.
4. Run the mandatory import-path audit and any configured PR quality gates.
5. Establish baseline performance.
6. Profile the current bottleneck when needed.
7. Write an optimization hypothesis.
8. Implement one coherent candidate.
9. Run correctness and configured PR quality gates.
10. Run benchmark.
11. Record evidence.
12. Keep, revise, or reject the candidate.
13. Repeat until the target or stop condition is reached.

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
| Public API, module layers, naming/registry, pytest marker, and unified dispatch | `quality` in `.kernelpilot/task.yaml`, `runs/quality.json` |
| Absolute package imports in submitted Python files | Mandatory built-in AST audit on every `run quality`; optional extra paths under `quality.import_paths`, evidence in `runs/quality.json` |
| Which stage the task is currently in | the task workspace, not this repository |
| Confirmed, reusable optimization lesson | `scripts/record_knowledge.py`; vendor-specific notes under `backends/<vendor>/note/`, with `knowledge/` as the generic fallback |
| Measured negative result worth not repeating | `scripts/record_rejection.py` → `<note-root>/rejected/` |
| A note's lifecycle state, scope, and re-validation date | YAML front matter on guides/rejections; `status`/`version`/`updated_at` in result JSON |
| The note catalog for a backend | Generated `INDEX.md` via `scripts/knowledge_index.py` (never hand-edited) |
| "Which notes are stale or contradict each other?" | `scripts/knowledge_audit.py` (`--strict` before a release) |
| Retiring a superseded note | `--revalidate-of` / `--supersede` on the recorders; never delete a note |

The reusable loop is defined here; task-specific prompts, harnesses, and results
stay in the task workspace. See [docs/agent-flow.md](docs/agent-flow.md) for the
underlying loop and [prompts/README.md](prompts/README.md) for how to start a
session.

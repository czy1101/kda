# Extending KDA for a backend

KDA separates execution policy from optimization knowledge. Add a backend in
small, independently testable layers instead of growing the task prompt.

## Files and ownership

| Concern | Location |
|---|---|
| Device/language matching and skill routing | `backends/<id>.yaml` |
| Reusable optimization judgment | `skills/<name>/SKILL.md` |
| Detailed backend facts and procedures | `skills/<name>/references/` |
| Deterministic parsing or report conversion | `skills/<name>/scripts/` |
| Task-specific host, environment, paths, commands, shapes, and target | `.kernelpilot/task.yaml` in the control workspace |
| Stage ordering and gates | `workflows/kernel-optimization.yaml` |
| Agent entry request | `prompts/optimize-kernel.md` |

Backend profiles should also declare optional analysis capabilities instead of
assuming that every device supports NVIDIA tools:

```yaml
analysis_tools:
  - id: <vendor-profiler>
    kind: kernel-profiler
    priority: 100
    probe: command -v <vendor-profiler> >/dev/null 2>&1
    version_command: <vendor-profiler> --version
    capabilities: [kernel_counters, structured_export]
    skills: [skills/<vendor-profiler-analysis-skill>]

capabilities:
  profilers: [ncu]
  structured_profile_export: true
  reference_profile_comparison: true
  ai_profile_fallback: true
  low_level_artifacts: [triton-ir, ptx, sass]

extensions:
  tle:
    status: experimental
    required: false
    skill: skills/tle-optimization
    project_marker: tle-wiki
    reason: Requires explicit task enablement plus successful Wiki and API checks.
```

Task policy decides whether a capability is `auto`, `disabled`, or `required`.
An unavailable `auto` capability falls back and is recorded; an unavailable
`required` capability fails the analysis stage.

Keep optional extensions disabled in the task template unless the user wants
them involved. Backend `experimental` means “eligible for task-local
validation,” not “installed.” Extension-specific knowledge belongs in its
Skill; project/version-specific API truth comes from the operator project's
Wiki and probe results.

Discovery probes must be read-only, bounded, and safe to run after
`environment.setup`. They may check an executable, Python module, version, or
device visibility. Do not install packages, change drivers, alter clocks, or
request elevated permissions from a probe. Declare several tools when they
answer different questions; priority selects a default but does not require
running all of them.

## Add a backend

1. Copy `templates/backend.yaml` to `backends/<backend>-<arch>.yaml`.
2. Give the profile a stable id and narrow match fields. Do not encode a
   particular operator's shapes or scores in it.
3. Create one skill for knowledge that changes optimization decisions on that
   backend. Keep `SKILL.md` concise and route detailed material into
   `references/`.
4. Add profiler guidance separately and list it under `when_profiling`; do not
   force expensive profiling on every candidate.
5. Add read-only `analysis_tools` probes and declare profiler,
   structured-export, reference-comparison, low-level, and extension
   capabilities truthfully. Leave vendor-specific capabilities empty when the
   backend does not provide them.
6. Set `target.profile` in the task contract and test one known operator end to
   end. Record which guidance was used and whether it improved candidate
   quality.
7. Promote repeated, cross-task lessons into the skill. Leave one-off findings
   in the task's evidence ledger.

Validate a new skill with the skill-creator validator:

```bash
python3 ~/.codex/skills/.system/skill-creator/scripts/quick_validate.py \
  skills/<name>
```

## Selection rule

Load `skills.always` before diagnosis. Load `when_discovering` only while
interpreting tool discovery and selection. Load `when_profiling` only when the task
has a valid profile command and profiling is justified. Load
`when_researching` only when local evidence and existing references are
insufficient. This keeps context focused while allowing backend-specific depth.

## Acceptance rule

A backend extension is ready when its profile validates, its required skills
validate, a task can select it without prompt edits, and at least one complete
run proves correctness, scope enforcement, benchmark aggregation, rollback,
and evidence generation.

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

## Add a backend

1. Copy `templates/backend.yaml` to `backends/<backend>-<arch>.yaml`.
2. Give the profile a stable id and narrow match fields. Do not encode a
   particular operator's shapes or scores in it.
3. Create one skill for knowledge that changes optimization decisions on that
   backend. Keep `SKILL.md` concise and route detailed material into
   `references/`.
4. Add profiler guidance separately and list it under `when_profiling`; do not
   force expensive profiling on every candidate.
5. Set `target.profile` in the task contract and test one known operator end to
   end. Record which guidance was used and whether it improved candidate
   quality.
6. Promote repeated, cross-task lessons into the skill. Leave one-off findings
   in the task's evidence ledger.

Validate a new skill with the skill-creator validator:

```bash
python3 ~/.codex/skills/.system/skill-creator/scripts/quick_validate.py \
  skills/<name>
```

## Selection rule

Load `skills.always` before diagnosis. Load `when_profiling` only when the task
has a valid profile command and profiling is justified. Load
`when_researching` only when local evidence and existing references are
insufficient. This keeps context focused while allowing backend-specific depth.

## Acceptance rule

A backend extension is ready when its profile validates, its required skills
validate, a task can select it without prompt edits, and at least one complete
run proves correctness, scope enforcement, benchmark aggregation, rollback,
and evidence generation.

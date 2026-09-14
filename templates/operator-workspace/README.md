# Operator Workspace Template

Copy this directory into the repository that contains the kernel you want to
optimize. The workspace stays task-specific, while the reusable workflow stays
in the KDA repository.

```bash
mkdir -p <operator-repo>/.kernelpilot
cp <kda>/templates/task.yaml   <operator-repo>/.kernelpilot/task.yaml
cp <kda>/templates/operator-workspace/AGENTS.md <operator-repo>/AGENTS.md
```

Then edit `.kernelpilot/task.yaml`: replace every placeholder, and validate it
against `<kda>/schemas/task.schema.yaml`.

Start the session from the operator repository:

```bash
cd <operator-repo>
codex
```

and give the agent the prompt from `<kda>/prompts/optimize-kernel.md`.

Expected layout after the first rounds:

```text
<operator-repo>/
  AGENTS.md
  .kernelpilot/
    task.yaml
  docs/
    analysis.md        # optional, from prompts/analyze-kernel.md
    draft.md
    plan.md
  runs/
    candidates.jsonl
    benchmark.csv
  profile/
```

`runs/`, `outputs/`, and `profile/` are generated evidence and should not be
committed to the KDA repository.

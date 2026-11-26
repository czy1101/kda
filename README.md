# Kernel Design Agents

Kernel Design Agents (KDA) is an agent-centric workflow for using coding agents to research, implement, verify, and iterate on performance-sensitive CUDA kernel tasks.

This repository documents an early research prototype and remains under active development. Community feedback and contributions are welcome. If you are interested in HAN Lab Mafia's solutions ranking #1–3 on tracks at the MLSys Kernel Contest, see [mit-han-lab/mlsys2026-flashinfer-contest](https://github.com/mit-han-lab/mlsys2026-flashinfer-contest) for performance evaluation and reproduction.

## Contents

| Path | Purpose |
|---|---|
| `docs/agent-flow.md` | Minimal end-to-end KDA workflow. |
| `AGENTS.md` | Repository-facing agent instructions for Codex and OpenCode. |
| `CLAUDE.md` | Repository-facing agent instructions for Claude Code. |
| `prompts/README.md` | How to use prompt templates. |
| `prompts/basic-flow.md` | Generic starter prompt for a new task. |
| `prompts/optimize-kernel.md` | Entry point for a full optimization run. |
| `prompts/analyze-kernel.md` | Understand the operator without editing code. |
| `prompts/profile-kernel.md` | Produce profiling evidence and one hypothesis. |
| `prompts/final-review.md` | Summarize the best candidate and the evidence. |
| `workflows/kernel-optimization.yaml` | Declarative definition of the optimization loop. |
| `schemas/task.schema.yaml` | Task contract schema. |
| `schemas/candidate.schema.json` | Candidate ledger record schema. |
| `schemas/run-result.schema.json` | Structured result returned by automated agent runs. |
| `schemas/backend.schema.yaml` | Contract for backend-to-skill routing profiles. |
| `schemas/bottleneck-screen.schema.json` | Six-layer screen and selected deep dive written to `runs/bottleneck-screen.json`. |
| `docs/bottleneck-localization.md` | Two-step diagnosis: screen six layers, then run one decision-changing experiment. |
| `docs/opencode.md` | OpenCode entry points on an S60 host or a controller with SSH. |
| `docs/enflame-s60-runbook.md` | S60 device, measurement, TLE, and evidence rules. |
| `docs/enflame-compiler-diagnostics.md` | S60 compile-chain confirmation, artifact identity, pass localization, and the compiler-change gate. |
| `docs/evidence-format.md` | Canonical benchmark and candidate ledger format. |
| `docs/backend-extensions.md` | How to add a chip backend without growing prompts. |
| `backends/` | Backend/architecture matching and stage-specific skill routing. |
| `knowledge/` | Evidence-scoped optimization reference, including Enflame S60/GCU300 compiler paths and Torch mechanisms. |
| `templates/task.yaml` | Fill-in task contract. |
| `templates/operator-workspace/` | Files to drop into an operator repository. |
| `scripts/kernelpilot.py` | Minimal task validator, environment-aware command runner, and Codex/OpenCode driver. |
| `scripts/compile_artifacts.py` | Offline artifact identity manifest and layered comparison; no device or framework import. |
| `scripts/controlled_ssh.py` | Allow-listed, hash-guarded adapter for execution-only GPU hosts. |
| `CONTRIBUTING.md` | Contribution process and DCO sign-off requirements. |
| `THIRD_PARTY_NOTICES.md` | Third-party component and license disclosures. |
| `third_party_licenses/` | Verbatim upstream license files for distributed third-party components. |

## Community Kernel Wishlist

Have a kernel that needs optimization? [Submit a request](https://github.com/NVlabs/kda/tree/wishlist#submit) directly as a pull request to `wishlist`, with a reproducible definition, representative workloads, and your best-known baseline implementation. No issue is required. The community wishlist currently supports NVIDIA B200 and B300 GPUs only.

[Browse and upvote requests](https://github.com/NVlabs/kda/pulls?q=is%3Apr%20base%3Awishlist), [discuss an idea](https://github.com/NVlabs/kda/issues/new?template=kernel-request.yml) if you need help preparing the files, or visit the [project website](https://nvlabs.github.io/kda/). Keep request details in `requests/<github-username>-<kernel-name>/README.md`; the pull request description can be brief. Merging records the request, while optimization progress and result links remain on the original pull request. Website source lives on `pages`; active issue forms are maintained on `main`.

## Getting Started
Install the agent workflow dependencies before starting the agent session:

```bash
git clone --recurse-submodules https://github.com/mit-han-lab/kernel-design-agents.git
cd kernel-design-agents

# link skills
mkdir -p ~/.claude/skills
ln -s "$(pwd)/skills/ncu-report-skill" ~/.claude/skills/ncu-report-skill
ln -s "$(pwd)/skills/KernelWiki" ~/.claude/skills/KernelWiki

# or clone the independently licensed ncu skill directly
mkdir -p ~/.claude/skills && cd ~/.claude/skills
git clone https://github.com/mit-han-lab/ncu-report-skill.git
```

Use KernelWiki from this repository's pinned submodule. A direct upstream
checkout may contain artifact snapshots governed by additional terms that are
omitted from this distribution.

## Using KDA with Codex or OpenCode

The workflow assets here are agent-agnostic: Markdown prompts, a declarative
workflow, JSON Schemas, and `SKILL.md` directories. KernelPilot supports two
non-interactive drivers without changing the task contract or evidence gates:

| Purpose | Claude Code | Codex | OpenCode |
|---|---|---|---|
| Repository instructions | `CLAUDE.md` | `AGENTS.md` | `AGENTS.md` |
| Skills | symlink into `~/.claude/skills/` | symlink into `~/.codex/skills/` | project `skills` config or `.opencode/skills/` |
| Start a task | paste `prompts/basic-flow.md` | paste `prompts/optimize-kernel.md` | `kernelpilot.py prompt` or `optimize --agent opencode` |
| Non-interactive driver | not provided here | `codex exec` | `opencode run --format json` |

`--agent auto` (the default) honours `KERNELPILOT_AGENT`, then prefers an
installed Codex executable, then an installed OpenCode executable. Explicit
selection is recommended for reproducible runs. Existing scripts that pass
`--codex-bin` continue to use Codex.

Install the skills for Codex:

```bash
mkdir -p ~/.codex/skills
ln -sfn "$(pwd)/skills/KernelWiki"       ~/.codex/skills/KernelWiki
ln -sfn "$(pwd)/skills/ncu-report-skill" ~/.codex/skills/ncu-report-skill
ln -sfn "$(pwd)/skills/backend-profiler-discovery" ~/.codex/skills/backend-profiler-discovery
ln -sfn "$(pwd)/skills/triton-cuda-optimization" ~/.codex/skills/triton-cuda-optimization
ln -sfn "$(pwd)/skills/nvidia-profile-comparison" ~/.codex/skills/nvidia-profile-comparison
ln -sfn "$(pwd)/skills/tle-optimization" ~/.codex/skills/tle-optimization
```

Skills are user-level for Codex: they become available to every Codex session,
not only sessions started in this repository. Verify inside a Codex session with
`/skills`.

For OpenCode, `kernelpilot.py init --agent opencode` writes
`<operator-repo>/.opencode/opencode.jsonc` with the KDA `skills/` directory on
the project skill path. OpenCode also discovers `.opencode/skills/`,
`.claude/skills/`, and `.agents/skills/` if you prefer to link individual skills.

Initialize the task in the operator repository, not here, then let KernelPilot
construct the full agent prompt with absolute KDA paths:

```bash
python3 <kda>/scripts/kernelpilot.py init <operator-repo>
# OpenCode project setup:
python3 <kda>/scripts/kernelpilot.py init --agent opencode <operator-repo>
# Fill <operator-repo>/.kernelpilot/task.yaml, then:
python3 <kda>/scripts/kernelpilot.py optimize --workspace <operator-repo>
python3 <kda>/scripts/kernelpilot.py optimize --agent opencode \
  --model <provider/model> --workspace <operator-repo>
```

The repository includes a deliberately small execution layer. It validates the
contract, applies the declared environment setup, launches the selected agent
non-interactively, captures events, checks required evidence, and rejects runs
that modify explicitly forbidden paths. The agent still owns optimization
judgment; the runner owns reproducibility and gates.

```bash
# Validate the task and its environment preflight.
python3 <kda>/scripts/kernelpilot.py validate --workspace <operator-repo>

# Run a contracted command with the exact environment setup.
python3 <kda>/scripts/kernelpilot.py run correctness --workspace <operator-repo>

# Discover backend tools and inspect the selected analysis route.
python3 <kda>/scripts/kernelpilot.py discover-tools --workspace <operator-repo>
python3 <kda>/scripts/kernelpilot.py strategy       --workspace <operator-repo>

# Inspect the selected agent invocation without running it.
python3 <kda>/scripts/kernelpilot.py optimize --agent codex --workspace <operator-repo> --dry-run
python3 <kda>/scripts/kernelpilot.py optimize --agent opencode \
  --model <provider/model> --workspace <operator-repo> --dry-run

# Print the same prompt for a manually driven OpenCode session.
python3 <kda>/scripts/kernelpilot.py prompt --agent opencode --workspace <operator-repo>

# Run the optimization loop. Agent authentication must already be configured.
python3 <kda>/scripts/kernelpilot.py optimize --agent opencode \
  --model <provider/model> --workspace <operator-repo>
```

### OpenCode on the S60 host

Use OpenCode directly in an independent FlagAttention worktree when the host can
run OpenCode and the S60 container is reachable through `docker exec`. This is
the simplest path for interactive work. See [docs/opencode.md](docs/opencode.md)
for the exact worktree, device-variable, and measurement commands, and
[docs/enflame-s60-runbook.md](docs/enflame-s60-runbook.md) for the device rules.

### OpenCode on a controller with a remote S60 executor

The remote GPU server does not need OpenCode. Put the task contract and evidence
in a small local control workspace, set `execution.transport: ssh`, and declare
the SSH alias and absolute remote repository path. KernelPilot materializes only
`constraints.allowed_paths` under `remote-worktree/` and rejects a push when a
path is not allow-listed or its remote hash changed since pull.

```bash
python3 <kda>/scripts/kernelpilot.py remote inspect --workspace <control-workspace>
python3 <kda>/scripts/kernelpilot.py remote pull    --workspace <control-workspace>
# The OpenCode agent edits <control-workspace>/remote-worktree/<allowed-path>
python3 <kda>/scripts/kernelpilot.py remote push    --workspace <control-workspace>
python3 <kda>/scripts/kernelpilot.py run correctness --workspace <control-workspace>
python3 <kda>/scripts/kernelpilot.py run benchmark   --workspace <control-workspace>
# When TLE is enabled, inspect its Wiki through the read-only adapter.
python3 <kda>/scripts/kernelpilot.py remote list-wiki --workspace <control-workspace>
```

The adapter is the normal execution boundary, not a substitute for operating
system isolation: the agent instructions and agent permission policy must also
disallow direct `ssh`, `scp`, and `rsync` calls.

Codex runs with `--sandbox workspace-write`, JSONL event capture, and a
structured final-output schema. OpenCode runs with `opencode run --format json`
and the same structured result schema written to a contracted file. OpenCode
does not provide Codex's filesystem sandbox; KernelPilot still enforces the
allowlist and forbidden-path hashes and restores the best checkpoint. See
[docs/opencode.md](docs/opencode.md) for the verified CLI behavior and limits.

## Submodules

The following third-party projects are included as Git submodules. The source URLs and revisions below are intentional and match the gitlinks recorded in this repository.

| Path | Source | Pinned revision | License |
|---|---|---|---|
| `skills/KernelWiki` | [mit-han-lab/KernelWiki](https://github.com/mit-han-lab/KernelWiki.git) | Based on [`76d27b56f804e7e7295d4c570e1e5d7eef4b0a75`](https://github.com/mit-han-lab/KernelWiki/commit/76d27b56f804e7e7295d4c570e1e5d7eef4b0a75), with the repository-maintained release sanitization recorded by the gitlink | MIT for original material; embedded artifacts retain their upstream terms; restricted CuTe DSL artifacts are omitted |
| `skills/ncu-report-skill` | [mit-han-lab/ncu-report-skill](https://github.com/mit-han-lab/ncu-report-skill.git) | [`d1887948c7d53690cfe6605f59c1329b8a1c6bb5`](https://github.com/mit-han-lab/ncu-report-skill/commit/d1887948c7d53690cfe6605f59c1329b8a1c6bb5) | MIT |

These submodules are not covered by this repository's first-party license. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the applicable license text and additional details.

Install the `humanize` Claude Code plugin from the Claude Code plugin UI:

```text
/plugin marketplace add PolyArch/humanize
/plugin install humanize@PolyArch
```

## Minimal Flow

1. Create a separate implementation workspace for the target task.
2. Define the task contract: objective, constraints, validation command, and promotion criteria.
3. Start an agent session in the implementation workspace.
4. Give the agent `prompts/basic-flow.md`, filled in with the task-specific details.
5. Ask the agent to write a short plan draft to `docs/draft.md` in the implementation workspace.
6. Convert the draft into an executable plan, either manually or with a planning tool such as Humanize.
7. Implement in small iterations, verifying after each meaningful change.
8. Record candidates, benchmark or evaluation results, profiling evidence, and final promotion decisions.

The workflow is intentionally independent of any single benchmark harness or hardware target. A downstream task can add its own evaluator, datasets, profiling tools, and domain-specific references.

## Recommended Workspace Layout

Use this repository as reference material, then do implementation work elsewhere:

```text
task-workspace/
  docs/
    draft.md
    plan.md
  runs/
  outputs/
  profile/
  runs/
    benchmark.csv
    candidates.jsonl
    workflow-state.json
    stage-events.jsonl
```

The exact files can change by domain. The important rule is that the agent records enough context for another engineer to understand what was tried, what passed validation, and why the final candidate was selected.

## Contributing

This project accepts outside contributions under the Developer Certificate of Origin process. See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

## License

Except for the third-party submodules identified above, first-party documentation, prompts, skills-style content, and assets are licensed under the [Creative Commons Attribution 4.0 International License](LICENSE), and first-party source code is licensed under the [Apache License 2.0](LICENSE).

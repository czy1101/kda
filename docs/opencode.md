# Driving KDA with OpenCode

This page describes the two supported OpenCode entry points for an Enflame S60
operator task. Both keep the KDA task contract, allowlisted source paths,
forbidden-path hashes, correctness gate, benchmark ledger, checkpoint/rollback,
and final-review evidence unchanged. Only the agent driver changes.

## Verified CLI baseline

The adapter was implemented against the OpenCode installed on the S60 host on
2026-11-16:

```text
opencode --version
# opencode 2.2.7

opencode run --help
# used: --standalone, --format json, --agent, --model, --title
# available but opt-in: --auto
```

`opencode run --format json` emits one JSON object per line. The observed record
types were `step_start`, `text`, and `step_finish`; every record carried a
`sessionID`. The OpenCode CLI has no `codex exec` subcommand, no
`--output-schema`, and no `--output-last-message`. KDA therefore asks OpenCode to
write the same `schemas/run-result.schema.json` object to a path named in the
generated prompt. If that file is missing, KernelPilot can recover the object
from the emitted text, but the file is the preferred evidence.

The runner uses `--standalone` so a scripted run does not depend on another
OpenCode client's shared service state. It does not pass `--auto` by default.
Pass `--auto-approve` only if your OpenCode permission configuration would
otherwise wait for an interactive approval. Do not use it to bypass a deliberate
`deny` rule.

## Agent selection

```bash
# Default: KERNELPILOT_AGENT, then Codex if installed, then OpenCode.
python3 <kda>/scripts/kernelpilot.py optimize --workspace <workspace>

# Explicit drivers.
python3 <kda>/scripts/kernelpilot.py optimize --agent codex    --workspace <workspace>
python3 <kda>/scripts/kernelpilot.py optimize --agent opencode \
  --model <provider/model> --workspace <workspace>

# Inspect the exact argv and prompt without starting an agent.
python3 <kda>/scripts/kernelpilot.py optimize --agent opencode \
  --model <provider/model> --workspace <workspace> --dry-run
```

Environment variables:

| Variable | Purpose |
|---|---|
| `KERNELPILOT_AGENT` | `codex`, `opencode`, or `auto`. |
| `KERNELPILOT_CODEX` | Codex executable (default `codex`). |
| `KERNELPILOT_OPENCODE` | OpenCode executable (default `opencode`). |
| `KERNELPILOT_OPENCODE_MODEL` | Default `provider/model` for OpenCode. |
| `KERNELPILOT_OPENCODE_AGENT` | OpenCode primary agent (default `build`). |

`--codex-bin` is retained for existing scripts. `--agent-bin` overrides the
executable after selection. `--model` is OpenCode-only and is rejected for
Codex.

## Entry point A: OpenCode on the S60 host

Use this when the host can run OpenCode and the S60 container is reachable with
`docker exec`. OpenCode runs on the host; the operator code is in an independent
FlagAttention worktree. The container does not need OpenCode.

1. Create or reuse an independent FlagAttention worktree. Do not disturb other
   worktrees or uncommitted work.

   ```bash
   git -C /data/hjq/FlagAttention fetch origin
   git -C /data/hjq/FlagAttention worktree add \
     /data/hjq/FlagAttention-<operator> \
     -b feat/enflame-<operator> origin/main
   ```

2. Initialize the KDA task in that worktree and fill the contract.

   ```bash
   python3 /data/hjq/kda-opencode/scripts/kernelpilot.py init \
     --agent opencode /data/hjq/FlagAttention-<operator>
   # Edit /data/hjq/FlagAttention-<operator>/.kernelpilot/task.yaml
   python3 /data/hjq/kda-opencode/scripts/kernelpilot.py validate \
     --workspace /data/hjq/FlagAttention-<operator>
   ```

   `init --agent opencode` writes `.opencode/opencode.jsonc` with the KDA
   `skills/` directory on OpenCode's project skill path. OpenCode reads the
   workspace `AGENTS.md` automatically.

3. Run OpenCode. Interactive:

   ```bash
   cd /data/hjq/FlagAttention-<operator>
   opencode
   ```

   Non-interactive through KernelPilot:

   ```bash
   python3 /data/hjq/kda-opencode/scripts/kernelpilot.py optimize \
     --agent opencode --model <provider/model> \
     --workspace /data/hjq/FlagAttention-<operator>
   ```

4. Contracted commands must enter the S60 container with the device variable and
   working directory set every time. On the host, the worktree is
   `/data/hjq/FlagAttention-<operator>`; inside `hjq-flagattention` the same
   bind mount is `/workspace/hjq/FlagAttention-<operator>`. For example:

   ```bash
   docker exec -e TOPS_VISIBLE_DEVICES=6 \
     -w /workspace/hjq/FlagAttention-<operator> \
     hjq-flagattention bash -lc '<correctness or benchmark command>'
   ```

   Put that full `docker exec` command in the task's `correctness.command`,
   `benchmark.command`, and baseline/profile commands. Do not assume a shell
   variable or working directory persists between `docker exec` calls.

## Entry point B: OpenCode on a controller, S60 execution over SSH

Use this when OpenCode runs on a separate control machine and the S60 machine is
execution-only. The remote host does not need OpenCode. KernelPilot's controlled
SSH adapter copies only `constraints.allowed_paths` into
`<control-workspace>/remote-worktree/`, checks the remote hash before writing,
and runs contracted commands on the remote host.

1. Create the control workspace and task contract.

   ```bash
   mkdir -p /path/to/control-workspace
   python3 <kda>/scripts/kernelpilot.py init --agent opencode /path/to/control-workspace
   ```

   Set, at minimum:

   ```yaml
   execution:
     transport: ssh
     host: <ssh-config-host-for-s60>
     workspace: /data/hjq/FlagAttention-<operator>
   ```

   `execution.workspace` is the host path that the S60 container sees through
   its bind mount. Use the corresponding `/workspace/hjq/...` path inside
   `docker exec` commands.

2. Materialize, edit, deploy, and measure.

   ```bash
   python3 <kda>/scripts/kernelpilot.py remote inspect --workspace /path/to/control-workspace
   python3 <kda>/scripts/kernelpilot.py remote pull    --workspace /path/to/control-workspace
   # OpenCode edits /path/to/control-workspace/remote-worktree/<allowed-path>
   python3 <kda>/scripts/kernelpilot.py remote push    --workspace /path/to/control-workspace
   python3 <kda>/scripts/kernelpilot.py run correctness --workspace /path/to/control-workspace
   python3 <kda>/scripts/kernelpilot.py run benchmark   --workspace /path/to/control-workspace
   ```

3. Run the full loop with OpenCode on the controller:

   ```bash
   python3 <kda>/scripts/kernelpilot.py optimize \
     --agent opencode --model <provider/model> \
     --workspace /path/to/control-workspace
   ```

   The prompt tells OpenCode to edit only the local mirror and to use
   `kernelpilot.py remote pull/push`. Direct `ssh`, `scp`, and `rsync` remain
   forbidden by the agent instructions and the KDA safety rules.

## Manual fallback

If automatic driving is unavailable, the same workflow can be driven manually
with an interactive OpenCode session:

```bash
python3 <kda>/scripts/kernelpilot.py prompt --agent opencode \
  --workspace <workspace> > /tmp/kda-prompt.txt
opencode run --standalone --format json --agent build \
  --model <provider/model> "$(cat /tmp/kda-prompt.txt)"
```

After the session, run `kernelpilot.py verify --workspace <workspace>` and inspect
`runs/kernelpilot-state.json`, `runs/candidates.jsonl`, `runs/benchmark.csv`,
and `docs/final-report.md`. This manual path still enforces the task contract
and evidence checks; it does not provide automatic checkpoint restore unless
the session itself uses the `checkpoint` commands.

## Limits and unverified claims

- The adapter was validated with OpenCode 2.2.7 on the S60 host. Re-check
  `opencode run --help` after an upgrade; do not assume the flags or event
  schema are stable across major versions.
- OpenCode does not provide an OS sandbox equivalent to Codex
  `--sandbox workspace-write`. The allowed/forbidden path checks are evidence
  gates, not a substitute for operating-system isolation.
- No S60 operator performance result is claimed by this adapter change. A dry
  run or a correctness-only task does not prove a speedup.
- The controlled SSH entry point was validated at the adapter/command level, not
  against a live remote S60 SSH alias in this change. Verify
  `remote inspect` on the target host before a production run.

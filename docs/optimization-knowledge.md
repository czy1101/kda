# Evidence-gated optimization knowledge

KDA keeps reusable optimization lessons separate from task-local code, logs,
and benchmark harnesses. By default, every optimization run ends with a
knowledge-eligibility review. A correctness-passing, materially improved,
transferable best candidate is published as a device- and operator-specific
note with `scripts/record_knowledge.py`; an ineligible candidate remains only
in the task ledger and its skip reason is reported.

## Promotion gates

This is a standard finalization stage in `workflows/kernel-optimization.yaml`,
and `prompts/optimize-kernel.md` requires the agent to perform it for both
target-reached and evidence-backed target-not-reached runs. The workflow does
not publish by copying benchmark results: the agent must explain a validated
mechanism and scope, then the recorder enforces the evidence gates below.

The command refuses to publish a note unless all of the following hold:

1. The candidate appears exactly once in `runs/candidates.jsonl`, is marked
   `decision: keep`, and has `correctness: pass`.
2. Every configured benchmark shape has a passing row in
   `runs/benchmark.csv`; all rows use positive values and one unit.
3. The task's configured aggregation shows at least 5% improvement by default.
   `--min-improvement` can raise the bar for noisy or high-impact tasks; it
   should not be lowered simply to create a note.
4. No required shape exceeds the task's configured maximum-regression limit.
5. The author supplies the mechanism, validated applicability, transferable
   lesson, and limitations. The command does not infer a causal explanation
   from latency numbers.

Published notes go under `backends/<vendor>/note/<operator>/` when that vendor
note directory exists; otherwise they go under
`knowledge/<backend>-<device>-<arch>/<operator>/`. `--note-dir` overrides the
operator directory so a note can join an existing family (for example `gla`).
Each consists of a readable Markdown report and a JSON sidecar validated by
`schemas/optimization-knowledge.schema.json` before anything is written. The
JSON also records `task_workspace` and the sha256 of `runs/candidates.jsonl`
and `runs/benchmark.csv`, so a note can be traced back to the ledgers it was
computed from and a later ledger edit is detectable. Existing notes are never
overwritten silently; `--update` revises one in place and increments its
`version`, keeping `created_at` and setting `updated_at`.

`INDEX.md` is generated from the notes by `scripts/knowledge_index.py`, never
hand-edited. CI runs it with `--check`, so a stale catalog fails the build.

## Updating, retiring, and auditing

Three operations keep the library from rotting:

| Goal | Command | Effect |
|---|---|---|
| Correct a published result | `record_knowledge.py --update ...` | Rewrites the note in place: `version` +1, `created_at` kept, `updated_at` set, every evidence gate re-run |
| Replace a result with a re-measured one | `record_knowledge.py --revalidate-of <old-id-or-path> ...` | Publishes the new note, then marks the old one `superseded` with `superseded_by` |
| Retire a note without a replacement | `record_knowledge.py --supersede <id-or-path> ...` (or `record_rejection.py --supersede`) | Marks the note `superseded`; it stays readable and listed |

A note is never deleted: `superseded` keeps the history and the index moves it
under `## Superseded results` (or marks the rejection line). An identifier is a
recorded `id`, a relative path, or a bare `.md` filename; an ambiguous id is an
error rather than a guess.

`scripts/knowledge_audit.py` answers "what should I re-check":

```bash
python3 scripts/knowledge_audit.py                 # local report
python3 scripts/knowledge_audit.py --strict        # warnings fail too
python3 scripts/knowledge_audit.py --as-of 2026-12-01
```

It reports `ledger-changed` errors, `revalidate-overdue`, `provisional-stale`,
`workspace-missing`, and `conflicting-confirmed` warnings, plus
`ledger-unbound` info for records written before evidence binding existed.
`ledger-changed` means the note's recorded sha256 no longer matches
`runs/benchmark.csv` or `runs/candidates.jsonl`, so the note must be re-published
with `--update` or superseded. CI runs the default mode; a local `--strict` run
is the pre-release check.

Example:

```bash
python3 scripts/record_knowledge.py \
  --workspace /path/to/task-workspace \
  --candidate c004 \
  --title "Reuse adjacent query rows in MMAC" \
  --mechanism "Explain the measured device/operator mechanism." \
  --applicability "List the tested dtype, layout, shape range, and dispatch guard." \
  --lesson "State what another kernel author should measure or derive." \
  --limitations "List untested cases and any measured regressions." \
  --evidence profile/c004.md
```

For example, Hygon reports are collected under `backends/hygon/note/`. Add a
vendor note directory before publishing confirmed knowledge for a new backend
family if its reports should be collected with that profile.

## Negative knowledge

A rejected approach is still knowledge: it answers "we tried this and it was
slower, retry only when X". Record it with `scripts/record_rejection.py`, which
writes a short front-mattered note under `<note-root>/rejected/` and regenerates
the index:

```bash
python3 scripts/record_rejection.py \
  --title "GLA intra-tile fusion is slower at K=V=128" \
  --device BW --arch gfx936 \
  --operator flag_attn.runtime.backend._hygon.FLA.gla.chunk_gla \
  --scope "Fusing the intra-chunk A tile with the output stage at K=128, V=128" \
  --hypothesis "Removing the FP32 intermediate should help every head size." \
  --observed "1.5x to 1.7x slower than baseline on the tested cases." \
  --reason "The wide tile decomposes into repeated subtiles; LDS pressure dominates." \
  --retry-when "A wide-tile schedule fits 64 KiB LDS with the measured VGPR use."
```

A rejection is not evidence-gated like a confirmed result — its value is the
documented failure — but it must state what was tried, what was measured, why it
failed, and the concrete condition that would justify a retry. Rejections are
never cited as wins: they live under `## Rejected approaches` in the index.

Rejected, inconclusive, correctness-failing, and insignificant candidates are
still not promoted into confirmed guidance. This prevents KDA from teaching an
unverified correlation as a general device rule, while keeping the dead end on
record so the next run does not repeat it.

## Metadata and lifecycle

Notes without a JSON sidecar (guides and rejections) carry YAML front matter:

```yaml
---
id: hygon-mmac-fma-path-selection
kind: guide            # guide | rejection
status: confirmed      # confirmed | provisional | superseded | rejected
device: BW
arch: gfx936
operator: "hy3 decode attention"
scope: "Choose MMAC or vector FMA from effective tile occupancy"
last_validated: 2026-10-19      # optional: when evidence was collected
revalidate_after: 2027-01-31    # optional: re-check after a DTK or device change
---
```

`status: provisional` marks method that is not yet backed by a recorded result
note; `superseded` (results) and `rejected` (rejections) close a note's life.
Re-validation is a new note or an `--update`, not a silent edit, so the index
always reflects what was actually measured when.


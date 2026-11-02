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

Published notes go under `backends/<vendor>/note/<backend>-<device>-<arch>/<operator>/`
when that vendor note directory exists; otherwise they go under
`knowledge/<backend>-<device>-<arch>/<operator>/`. Each consists of a readable
Markdown report and a JSON sidecar validated by
`schemas/optimization-knowledge.schema.json`; the selected note root's
`INDEX.md` is updated for discovery. Existing notes are never overwritten.

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

Rejected, inconclusive, correctness-failing, and insignificant candidates stay
in the task's candidate ledger; they are not promoted into reusable
optimization guidance. This prevents KDA from teaching an unverified
correlation as a general device rule.

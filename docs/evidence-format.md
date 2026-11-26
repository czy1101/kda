# KernelPilot evidence format

KernelPilot uses plain JSONL and CSV so a run can be audited without a service
or database.

## Analysis capability record

`runs/analysis-capabilities.json` follows
`schemas/analysis-capabilities.schema.json`. KernelPilot generates it by
running only backend-declared, read-only probes after the task environment is
activated. It records both available and unavailable tools, versions when
obtainable, the selected default, and an environment fingerprint. Rerun
discovery when the task target, environment setup, executor, or workspace
changes; stale evidence is rejected.

## Benchmark ledger

`runs/benchmark.csv` has one row per candidate and required shape:

```csv
candidate,shape,metric_value,baseline_value,unit,status
baseline,B2x8192 H16 D128,1.894,1.894,ms,pass
c001,B2x8192 H16 D128,1.887,1.894,ms,pass
```

`shape` must exactly match the strings in `benchmark.shapes`. Metric and
baseline values must use the same positive unit. `baseline_value` is the value
from the comparator selected by `baseline.name` in the task contract, such as
`original` or `reference`. KernelPilot recomputes the aggregate ratio using
`goal.aggregation`; it does not trust a speedup written by the agent. The
aggregation policy may be `sum_ratio`, `mean_ratio`, `geomean_ratio`, `all`, or
`any`; it is selected per task rather than imposed globally.

## Candidate ledger

`runs/candidates.jsonl` contains one object per attempt and follows
`schemas/candidate.schema.json`. Include rejected correctness failures and
unsafe probes as well as kept candidates. Every rejected or revised candidate
must include a non-empty reason and rollback result. Use
`generalization_evidence` when a candidate introduces shape-dependent behavior.
New records use `metric`, `direction`, `metric_value`, `baseline_value`, and
`unit`; legacy latency-only fields remain accepted for older ledgers.

## Bottleneck screen

`runs/bottleneck-screen.json` follows
`schemas/bottleneck-screen.schema.json` and records the two-step diagnosis from
[bottleneck-localization.md](bottleneck-localization.md): one entry for each of
the six layers with its evidence, disposition, key unknown and the decision the
next experiment would support, plus at least one deep dive naming a falsifiable
hypothesis, its control, both outcomes and its stop condition.

`analysis.bottleneck_screen` controls enforcement. The default `auto` validates
the file only when it exists, so task workspaces written before this record
existed keep passing; `required` fails verification when it is missing;
`disabled` ignores it. Validation is structural. It can show that all six layers
were addressed and that a selected deep dive is falsifiable in form. It cannot
decide whether the cited evidence is sufficient or whether a natural-language
attribution overreaches, and it must not be reported as if it could.

The check runs in `verify` and at the end of `run`, so it gates completion and
promotion, not the order in which the agent works. It is not a pre-flight
interlock and must not be described as one.

Two optional blocks record attribution honestly:

- `attribution` keeps the two axes apart. `control_level` is `L0`–`L3` (how well
  the comparison was controlled); `observation` is `none`, `static-counts`,
  `device-image` or `counter` (how deep the evidence reaches). A `counter`
  reading requires `scope_note`, because a counter says what a unit did inside
  its documented scope, not why the time moved.
- `decomposition` records the three numbers for a gain with two sources:
  `source_at_fixed_config`, `config_at_fixed_source`, `deployed_combination`,
  plus `relationship` (`additive`, `not-additive`, or `unknown`). `unknown` is
  the honest default; subtracting one delta from another is not a decomposition.

## Compiler-artifact identity

`scripts/compile_artifacts.py` writes a manifest of artifact identity and a
layered comparison, typically to `runs/compile-artifacts.json` and
`runs/compile-artifact-diff.json`. It records size, SHA-256, a filename-derived
`stage_hint`, a byte-level container kind, cache keys, offload-bundle targets,
missing stages and unsupported files. It never writes to its inputs, and it
reports no operation counts and no semantic equivalence.

`stage_hint` is a filing label, not proof of IR dialect, and a differing byte
range is not proof of differing machine instructions. Both are stated in the
output's `unresolved` list. Pairing is reported as `match_quality`: an
`exact-path` pair shares its relative path, while a `name-only` pair was matched
across directories because the filename was unique on both sides. A `name-only`
pair carries an `unresolved` entry saying the two files may be different
specializations of one kernel; one kernel name can produce several cache entries
whose artifacts are not interchangeable. A filename that stays repeated on
either side is never paired and appears in `summary.ambiguous_names`.
When neither side contributes a device image — the
dump stopped before linking, or the fatbin was never captured — the comparison
adds an `unresolved` entry saying no device binary was compared, so an
`identical` count on text stages is never read as an instruction-level result.
Fixtures used by its tests are test data, not device evidence.

## Checkpoints

KernelPilot stores baseline and best snapshots under
`.kernelpilot/checkpoints/`. Only a correct candidate that has passed the
configured benchmark and generalization decision may replace the best
checkpoint. KernelPilot restores best when the agent process exits and rejects
a final result whose reported best candidate does not match that checkpoint.

## Structured final result

Automated Codex and OpenCode runs use `schemas/run-result.schema.json`. Codex
writes it through `--output-last-message`; OpenCode writes it to the path in the
generated prompt, with event-text extraction as a fallback. The runner verifies
that `best_candidate` exists in the candidate ledger and that
`target_reached` agrees with the benchmark ledger.

A valid final result may have `target_reached: false`. It must still report the
best correct candidate, a structured `stop_reason`, every workflow issue, the
remaining performance gap, and whether rollback restored the best candidate.
Reference parity, no actionable bottleneck, exhausted backend capabilities, and
configured budget exhaustion are honest stop outcomes rather than promotions.

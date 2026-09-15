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

## Checkpoints

KernelPilot stores baseline and best snapshots under
`.kernelpilot/checkpoints/`. Only a correct candidate that has passed the
configured benchmark and generalization decision may replace the best
checkpoint. KernelPilot restores best when the Codex process exits and rejects
a final result whose reported best candidate does not match that checkpoint.

## Structured final result

Automated Codex runs use `schemas/run-result.schema.json`. The runner verifies
that `best_candidate` exists in the candidate ledger and that
`target_reached` agrees with the benchmark ledger.

A valid final result may have `target_reached: false`. It must still report the
best correct candidate, a structured `stop_reason`, every workflow issue, the
remaining performance gap, and whether rollback restored the best candidate.
Reference parity, no actionable bottleneck, exhausted backend capabilities, and
configured budget exhaustion are honest stop outcomes rather than promotions.

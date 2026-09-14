# KernelPilot evidence format

KernelPilot uses plain JSONL and CSV so a run can be audited without a service
or database.

## Benchmark ledger

`runs/benchmark.csv` has one row per candidate and required shape:

```csv
candidate,shape,metric_value,baseline_value,unit,status
baseline,B2x8192 H16 D128,1.894,1.894,ms,pass
c001,B2x8192 H16 D128,1.887,1.894,ms,pass
```

`shape` must exactly match the strings in `benchmark.shapes`. Metric and
baseline values must use the same positive unit. KernelPilot recomputes the
aggregate ratio using `goal.aggregation`; it does not trust a speedup written by
the agent.

## Candidate ledger

`runs/candidates.jsonl` contains one object per attempt and follows
`schemas/candidate.schema.json`. Include rejected correctness failures and
unsafe probes as well as kept candidates.

## Structured final result

Automated Codex runs use `schemas/run-result.schema.json`. The runner verifies
that `best_candidate` exists in the candidate ledger and that
`target_reached` agrees with the benchmark ledger.

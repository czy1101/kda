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

## PR quality gates

`kernelpilot run quality` always writes `runs/quality.json` and runs the
mandatory import-path gate. When a task declares `quality`, it also executes
the five configured PR gates. The checks are:

- `public_api`: import/export and public API behavior;
- `module_layers`: separation of API, implementation, tests, and benchmarks;
- `import_paths`: built-in AST audit that rejects relative imports in Python
  files at `implementation.path` (or the optional configured paths);
- `naming`: one stable operator name shared by the API, pytest marker, and registry id;
- `pytest`: collection or execution of the declared marker;
- `dispatch`: lookup through the repository's unified operator dispatcher.

The five project-specific commands remain repository-specific. The absolute
import gate is mandatory for every task; when the optional quality contract is
required, all six checks must pass before final evidence can be verified.

## Generalization evidence

When the contract configures `generalization.command`, run it through
`kernelpilot run generalization`. The runner executes it in the contracted
environment, exports `generalization.max_regression` as
`KDA_GENERALIZATION_MAX_REGRESSION`, and writes `runs/generalization.json`:

```json
{
  "version": 1,
  "command": "python bench/guard_panel.py",
  "max_regression": 0.02,
  "started_at": "2026-11-03T02:10:00+00:00",
  "finished_at": "2026-11-03T02:12:31+00:00",
  "exit_code": 0,
  "passed": true,
  "stdout_tail": "...",
  "stderr_tail": ""
}
```

Final verification fails when the evidence is missing, when `exit_code` is not
zero, or when it was recorded for a different command. The panel owns the
tolerance decision; a non-zero exit is the contract's "guard shape regressed".

## Profile budget ledger

Every capture started through `kernelpilot run profile` appends one line to
`runs/profile-runs.jsonl`:

```json
{"version": 1, "finished_at": "…", "tool": "rocprofv2", "variant": "candidate", "exit_code": 0}
```

`verify` compares the line count with `stop.max_profile_runs` and fails the run
when the budget is exceeded. The ledger is a best-effort budget rather than an
audit trail: a capture invoked by hand outside the wrapper is not counted.

`stop.no_improvement_rounds` is computed from the candidate and benchmark
ledgers at verification time — the trailing number of candidates that did not
beat the best aggregate ratio. It produces a printed warning and a
`stop_report` block in `runs/kernelpilot-state.json`; it does not fail the run,
because only the agent can act on it by re-diagnosing.

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
checkpoint: `kernelpilot checkpoint save-best --candidate <id>` refuses unless
`runs/candidates.jsonl` records that candidate once with `decision: keep` and
`correctness: pass`, and `runs/benchmark.csv` supplies a passing, machine-
verifiable row for every configured shape. The accepted checkpoint stores the
recomputed aggregate in `promotion_evidence`. KernelPilot restores best when the
Codex process exits and rejects a final result whose reported best candidate does
not match that checkpoint.

## Structured final result

Automated Codex runs use `schemas/run-result.schema.json`. The runner verifies
that `best_candidate` exists in the candidate ledger and that
`target_reached` agrees with the benchmark ledger.

A valid final result may have `target_reached: false`. It must still report the
best correct candidate, a structured `stop_reason`, every workflow issue, the
remaining performance gap, and whether rollback restored the best candidate.
Reference parity, no actionable bottleneck, exhausted backend capabilities, and
configured budget exhaustion are honest stop outcomes rather than promotions.

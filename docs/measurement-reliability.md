# Reliable performance evidence

## Why this layer exists

Source-bound evidence answers which program was tested. It does not establish
that an apparent speedup survives noise, an occupied device or measurement drift.
The optional measurement policy adds controller-driven repetitions, raw-sample
checks, a frozen original baseline and interleaved best/candidate confirmation.
No operator shapes, chip-specific probes or numeric promotion targets are built
into the core. Policies and input identities belong to the task; hardware probes
belong to a backend/task adapter.

## Example policy (illustrative thresholds, not universal defaults)

```yaml
baseline:
  name: original
  role: initial_implementation
benchmark:
  command: python3 benchmark_adapter.py
  result_path: runs/benchmark.csv
  shapes: [forward/fp16/workload-a, backward/bf16/workload-b]
measurement:
  conditions:
    warmup: 10
    iterations: 30
    timing_scope: device-kernel-only
    timer: device-events
    input_id: seed42-and-input-manifest-sha256
  process_repeats: 3
  min_samples: 5
  max_relative_mad: 0.02
  min_improvement: 0.02
  confirmation_pairs: 3
  guard_command: python3 device_guard.py
  samples_path: runs/samples.json
```

The goal must also declare max_regression and the usual direction/aggregation.
The policy is frozen with the complete task hash; changing it requires a new
round. Existing tasks without measurement retain legacy behavior, so they must
not be described as having passed this reliability gate.

## Harness protocol

The contracted CSV still needs run_id, candidate, shape, metric_value,
baseline_value, unit and status. The controller checks its metric_value against
the raw-sample median. In strict mode the CSV's baseline_value does NOT select
the comparator: aggregated baseline values are rebuilt from the frozen original
implementation measurement, preventing an external reference from replacing it.

The benchmark must write raw JSON to samples_path:

```json
{
  "run_id": "value of KERNELPILOT_RUN_ID",
  "candidate": "value of KERNELPILOT_CANDIDATE",
  "conditions": {"warmup":10,"iterations":30,"timing_scope":"device-kernel-only","timer":"device-events","input_id":"seed42-and-input-manifest-sha256"},
  "unit": "ms",
  "samples": {
    "forward/fp16/workload-a": [1.0,1.01,0.99,1.0,1.0],
    "backward/bf16/workload-b": [2.0,2.01,1.99,2.0,2.0]
  }
}
```

KERNELPILOT_MEASUREMENT_CONDITIONS contains the canonical JSON conditions. The
adapter must actually apply them; echoing metadata alone is not proof of correct
timing or input construction. Freeze the adapter and input manifests as protected
files. All samples must be positive finite numbers. Units, complete workload
coverage, conditions and run identity must match. Sample counts must remain equal
per workload across processes, candidates and confirmation arms.

The guard runs in the same declared execution environment before and after a
benchmark command, outside the harness timing scope. Its stdout must be JSON:

```json
{"idle":true,"environment_id":"stable-device-driver-runtime-clock-policy-id","details":{}}
```

Only boolean true with a non-empty stable environment_id is accepted. The backend
adapter owns how to check device ownership and runtime identity, including how to
exclude its own diagnostic activity. Never use a time-varying field such as free
memory as the stable ID. Busy/unknown results are retained and rejected. KDA does
not terminate processes, alter clocks or silently retry until a fast run appears.
A task may wait and explicitly rerun after the device becomes available.

## Controller workflow

```sh
kernelpilot run correctness --candidate baseline
kernelpilot measure --candidate baseline
# Generalization, when configured, follows the last benchmark:
kernelpilot run generalization --candidate baseline
# Record pass/keep in candidates.jsonl, then approve the measured baseline:
kernelpilot checkpoint save-best --candidate baseline

# Edit and deploy one candidate through the ordinary controlled interface.
kernelpilot run correctness --candidate c001
kernelpilot measure --candidate c001
kernelpilot confirm --candidate c001
kernelpilot run generalization --candidate c001  # when configured
# Record the keep decision only after checking all evidence.
kernelpilot checkpoint save-best --candidate c001
```

measure invokes the benchmark command in process_repeats separate invocations;
it does not merely duplicate an existing sample list. It aggregates the median
of each process's sample median. Both within-process sample dispersion and
between-process median dispersion use MAD / median and must fit the policy.
The first baseline record is frozen with exclusive creation under
.kernelpilot/measurement-baseline.json; it cannot be overwritten by a rerun or
by a new best. Per-run raw logs and samples remain under runs/command-evidence/.
Aggregated summaries and constituent run hashes are under runs/measurements/.

confirm executes current approved best -> candidate for each configured pair.
It deploys only allowed implementation paths, uses existing source-bound
correctness, runs both arms under the same task, checks environment and sample
counts, and saves pair CSVs and run hashes under runs/confirmations/. Every pair
must meet min_improvement under the goal's aggregation/direction and satisfy
per-workload max_regression. Each arm's across-pair medians must also meet the
dispersion threshold. This conservative policy does not average away a bad pair.

A failed or interrupted confirmation reaching Python cleanup restores best and
records the rollback outcome. A passing confirmation leaves the candidate for
explicit save-best; it is not itself a promotion. New candidate measurements,
new benchmark runs, changed source, or a changed best invalidate confirmation.
If generalization is configured, run it after the final confirmation benchmark.
Initial screening also compares against the frozen baseline and current best;
interleaving does not waive those checks or the final delivery target report.

## Outcomes and boundaries

- pass: this measurement policy was satisfied, not statistical certainty.
- inconclusive/noise: dispersion exceeded the configured limit.
- inconclusive/contaminated_or_unknown: the guard could not establish idle state.
- inconclusive/environment_changed: comparable runtime identity was not maintained.
- error: invalid evidence, command failure or confirmation rejection; the concrete
  reason and rollback status remain recorded.

MAD is robust dispersion, not a confidence interval or hypothesis test. It can
underrepresent rare outliers; raw min/max and samples are retained. Sequential
best->candidate pairing does not eliminate every order/caching bias. Guard checks
at boundaries can miss transient interference during a kernel. Stronger device
monitoring must be supplied by the backend; there is no physical GPU lease or
cross-controller lock here. Declared metadata and a trusted harness remain part
of the measurement contract. No claim of theoretical peak utilization follows.

Tests use CPU-only synthetic timing data, plus the real controlled script paths
with a simulated SSH executor. They prove workflow decisions and evidence
handling, not actual GPU speedups. Real-task migration needs a new round and a
reviewed benchmark/guard adapter; old reports remain historical evidence.

## Attempt traceability

Repeat and confirmation records include an `attempts` array for every process
invocation, including rejected guards, nonzero exits and noisy measurements.
Each bound attempt records its command run ID, status and record hash, with
repeat or confirmation pair/arm labels. Successful `run_ids` remain the only
inputs to aggregation; linking failed evidence never makes it admissible.
Failures before a bound run exists are explicitly marked `unbound_error`.
The trace survives errors that reach Python cleanup; a controller crash or
forced termination may still require inspecting command-evidence directly.

## Ordered sample diagnostics

MAD discards order: warming and cooling sequences can have the same median
and MAD. Summaries also report first/last window medians; each window has
`max(1, sample_count // 3)` samples. These are descriptive, not a new
stationarity test or an extra promotion threshold. For repeated processes,
the sequence is the process-median sequence. Raw observations remain saved.
A noise rejection retains per-shape summaries and all exceeded shapes in
`diagnostics`, after checking the complete raw sample/CSV contract. A later
malformed shape remains invalid evidence, rather than being hidden by an
earlier noisy shape. Existing acceptance and regression limits are unchanged.

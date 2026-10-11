# Source-bound candidate evidence

## Problem and effect

A passing test and a fast CSV are insufficient if they belong to an older source
revision. The checkpoint writer previously accepted a candidate label without
checking the source or the execution evidence. A stale local mirror could also
be copied while the remote host had executed different code.

Bound candidate runs now connect declared file contents, the task contract,
command logs, run IDs, benchmark summaries and best-checkpoint approval. This is
operator/backend-independent. The implementation contains no workload names,
precision rules or hardware thresholds.

## Commands and benchmark adapter contract

```sh
kernelpilot run correctness --candidate baseline
kernelpilot run benchmark --candidate baseline
# Run this only when generalization.command is configured:
kernelpilot run generalization --candidate baseline
# Record the candidate's pass/keep decision in runs/candidates.jsonl, then:
kernelpilot checkpoint save-best --candidate baseline
```

Use a new candidate identifier after making one coherent implementation change.
Run the same sequence for that identifier. Diagnostic commands without
--candidate remain supported but cannot supply evidence for save-best.

A benchmark adapter must produce a CSV at benchmark.result_path (default
runs/benchmark.csv) on the execution host. It must be under runs/ or outputs/.
In addition to the existing summary columns, each candidate row must contain:

- candidate: value of KERNELPILOT_CANDIDATE, set by the runner.
- run_id: value of KERNELPILOT_RUN_ID, unique to this invocation.

Each run must replace the candidate's old summaries, not append repeated rows for
that same candidate. Raw repetitions belong in separate artifacts. The runner
copies the returned CSV into a per-run controller evidence directory; it does
not mutate the task's production tests or benchmark scripts automatically.

## What is enforced

- Before/after content snapshots cover allowed implementation paths, declared
  forbidden paths and reference.path. The canonical JSON task hash includes
  command text, environment setup, goal and transport configuration.
- SSH snapshots read actual executor files through ControlledSSH, not cached
  remote-state hashes. The executor must provide python3 in the declared setup.
- Bound benchmark cannot execute without the latest successful correctness run
  for that candidate and exact identity. A later failure supersedes an earlier
  success. Generalization similarly follows correctness and benchmark.
- Every started bound invocation has a uniquely named evidence directory under
  runs/command-evidence/. Failures and exceptions remain recorded. Returned
  command stdout/stderr and CSV files have SHA256 checksums.
- A successful process exit is insufficient for benchmark success: the CSV must
  identify this invocation and pass the generic evaluator's coverage, unit,
  finite-value and status checks.
- Save-best validates current identity, latest correctness/benchmark linkage,
  optional generalization, frozen artifact hashes, and a pass/keep ledger entry.
  The source copied into the checkpoint must match actual validated source.
- A measured baseline must be approved first. Later candidates must strictly
  improve the current best under the declared aggregation/direction and respect
  per-workload regression limits against best and the selected comparator.
  Baseline values, units, workload identities, contract and protected inputs
  cannot silently change between approvals. Changes require a separate round.
- Delivery target attainment is reported separately: a useful incremental best
  may still miss the final delivery target. This is not automatic target success.
- New checkpoints include source hashes, evidence IDs and approval metadata.
  Source integrity is checked before restore. Initial rollback snapshots are
  explicitly recovery_only, not benchmark-approved best. Existing checkpoints
  are retained on optimize startup rather than overwritten.

## Scope limits and migration

Historical logs and checkpoints remain available for inspection/recovery. They
cannot be retroactively treated as bound evidence. Use an independent task round
and run the declared baseline through the new interface before new promotion.
Do not fabricate run IDs for old measurements.

Only declared files are hashed. List imported helpers, tests, harnesses and
reference inputs in forbidden_paths when they affect validation. Whole-directory
snapshots are strict: generated cache files inside a protected directory count
as changes. Prefer specific input files, or direct caches outside frozen paths.
Symlinked source/protected paths are rejected for bound execution.

This increment does not cryptographically authenticate an adversarial harness,
prove a benchmark's timing semantics, freeze installed libraries or live GPU
conditions, prevent every transient/concurrent out-of-band edit, implement a
cross-controller lock, or by itself establish measurement reliability. The
optional measurement layer adds repeated sampling and interleaved confirmation. The task
contract hash binds configured environment settings, not observed hardware
state. Per-run files are content-checked on use, not a signed append-only store.
A killed host/process can leave a running record; it is not accepted as success.

The checkpoint storage writer is an internal implementation primitive for
recovery snapshots, not a public promotion bypass. Automatic rollback after
every failed manual command and fully resumable lifecycle transactions remain
separate follow-up work.

## Validation

CPU-only integration tests execute independent synthetic workloads and cover
latency minimization, throughput maximization, local execution, and a simulated
SSH executor that runs the same controlled scripts in a temporary directory.
They test stale source/task/test files, later failures, missing correctness,
stale CSVs, artifact edits, changed comparator values, checkpoint corruption,
remote mirror mismatch, and optional generalization. Simulated SSH coverage is
not a real GPU benchmark or a claim of hardware performance improvement.

## Measurement reliability follow-up

The optional `measurement` policy now adds a frozen initial-implementation
baseline, raw sample validation, repeated invocations, device/environment guards
and interleaved confirmation before best promotion. See
[measurement reliability](measurement-reliability.md) for the current protocol
and limitations; this supersedes earlier notes describing that layer as pending.
It remains opt-in for legacy tasks and requires harness migration in a new round.

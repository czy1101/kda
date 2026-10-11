# Evidence-driven optimization

KDA owns the reproducible optimization lifecycle; agents own diagnosis and
candidate design. Task workspaces own operator code, workload matrices,
tolerances and promotion thresholds. Backend adapters own tool commands and
hardware-specific metric semantics. No operator-specific rules are added to the
core workflow.

## Components and effects

| Component | Problem addressed | Result |
| --- | --- | --- |
| [Source-bound evidence](source-bound-evidence.md) | Old tests or CSVs can be attributed to different code. | Source, protected inputs, task, invocation and artifacts are checked before promotion. |
| [Measurement reliability](measurement-reliability.md) | A single noisy or contaminated sample can look faster. | Optional raw samples, process repeats, guards and interleaved confirmation constrain promotion. |
| [MetaX profiling](metax-profiling.md) | Finding a profiler executable does not establish usable capture. | Controlled collection validates workload completion, exports, metric coverage and artifact hashes. |
| [Compiler evidence and hypotheses](compiler-hypotheses.md) | A source edit may not produce its intended compiler effect. | Captures compare explicit workload/kernel mappings and assess registered predictions. |

Compiler observations and profiler data are diagnostic evidence, not benchmark
results or proof of performance benefit. Delivery target attainment and approval
of an incremental best remain separate decisions.

## Strict metric evaluation

Run `python3 scripts/kernelpilot.py evaluate --workspace <task> --candidate <id>`.
This reads the local benchmark ledger without executing workloads or changing a
checkpoint. Exit codes are 0 for target reached, 1 for valid evidence below the
target, and 2 for invalid evidence.

Each selected workload needs exactly one passing summary with finite positive
values and a consistent nonempty unit. Missing workloads, duplicate summaries,
invalid goals and aggregation overflow fail closed. Raw repetitions belong in
separate artifacts. Workload identifiers are opaque: encode phase, dtype, shape
and layout where they change the operation. No missing measurements or hardware
utilization figures are inferred.

## Adoption and limits

Bound correctness and benchmark runs are required for new best-checkpoint
approvals. Historical unbound evidence cannot be retroactively approved; begin
a new task round and measure the baseline. Measurement reliability and required
compiler hypotheses are opt-in task policies. Enabling them requires the
documented harness and probe contracts, not changes to production test logic.

CPU-only tests cover unrelated synthetic workloads, latency minimization,
throughput maximization, local execution and simulated controlled SSH. They
validate workflow mechanics, not GPU speedups. Real profiler versions must pass
their own capture/export checks.

Evidence files are content-checked, not signed or adversary-proof. Only declared
inputs are frozen. Guard snapshots can miss transient interference, and no
device lease or cross-controller lock is provided. Fully resumable lifecycle
transactions and stronger environment attestation remain separate work. Failed
manual commands do not universally restore source; interleaved confirmation and
the optimization driver's existing cleanup provide their documented recovery.

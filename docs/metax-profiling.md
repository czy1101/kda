# MetaX profiling through KernelPilot

This adapter connects MetaX counter and timeline exports to KDA's existing task,
transport and evidence workflow. It is independent of the kernel language,
operator, shape and device model. It does not install tools, modify drivers or
promote optimization candidates.

## Tools and evidence levels

| Tool ID | Purpose | Validated export |
| --- | --- | --- |
| `mcprofiler` | Hardware counter collection | `*.txt.json` section/metric/value reports |
| `mctracer` | Kernel and runtime API timeline | JSON `traceEvents`, complete events with device/queue and correlation identifiers |

`discover-tools` records executable availability as `declared_unverified`.
A successful `run profile` proves workload execution, readable exports, source
and environment checks, and supported normalization. `profile-report` verifies
saved artifact hashes and reports the capabilities actually observed. These are
three different evidence levels; finding a binary is not proof that all its
metrics work on the device.

The capture protocol was exercised with mcProfiler 3.9.0.29 and mcTracer
3.7.2.0. Other versions must pass their own capture validation; unknown export
formats fail rather than being treated as zero-valued measurements.

## Configure a task

Merge the following **partial example** into a valid `.kernelpilot/task.yaml`.
Keep the task's correctness, benchmark, execution and source constraints. The
paths and commands below are placeholders supplied by each task.

```yaml
target:
  backend: maca
  profile: maca
  device: MetaX GPU

environment:
  setup: |
    # Retain the task's existing runtime activation here.
    export KDA_MCPROFILER_BIN=/path/to/mcProfiler
    export KDA_MCTRACER_BIN=/path/to/mcTracer

profile:
  tool: mctracer
  tools:
    mctracer:
      adapter: metax
      workload_command: python profile_workload.py
      guard_command: python device_guard.py
      timestamp_unit: ns
      timeout_seconds: 300
      max_artifact_bytes: 67108864
      # Optional: exactly one explicit, completed measurement region.
      capture_range:
        start_api: mcProfilerStart
        stop_api: mcProfilerStop
    mcprofiler:
      adapter: metax
      workload_command: python profile_workload.py
      guard_command: python device_guard.py
      metrics: [Total Cycles]
      counts: 10
      start_from_api: true
      timeout_seconds: 300
      # Optional vendor kernel-name filter:
      # kernel_names: [my_kernel]
      # Optional environment for the profiled child only:
      # runtime_environment:
      #   LD_LIBRARY_PATH: /task/compatible/runtime/lib
      #   LD_PRELOAD: /task/compatible/runtime/lib/libmcToolsExt.so
```

Without binary overrides, the adapter uses `mcProfiler` on PATH and
`${MACA_PATH:-/opt/maca}/bin/mcTracer`. Environment overrides belong in the task,
not the backend profile. Never replace system libraries to make collection work.

The guard must print exactly one JSON object and exit successfully:

```json
{"idle": true, "environment_id": "stable-device-and-runtime-identity"}
```

Implement the guard for the actual environment. It should identify the target
device, other workloads and stable hardware/runtime properties. Do not include
the current time in `environment_id`. Busy or unknown state must use
`idle: false` or fail. KDA checks before and after capture, but these snapshots
cannot rule out all transient interference during a run.

Declare the profiling workload, guard, reference and relevant tests as concrete
protected files in `constraints.forbidden_paths`. Keep only implementation
sources in `allowed_paths`. Source binding covers declared paths, not arbitrary
undeclared dependencies. The task is responsible for choosing representative
inputs and verifying correctness; the profile command does not automatically
replace or execute the full correctness suite.

## Run and verify

From the task controller directory:

```bash
./kernelpilot validate
./kernelpilot discover-tools
./kernelpilot strategy
./kernelpilot run correctness
./kernelpilot run profile --tool mctracer
./kernelpilot run profile --tool mcprofiler
./kernelpilot profile-report --run <profile_id>
```

`run profile` uses the same configured local or controlled SSH execution adapter
as other contracted commands. It returns a unique profile ID and evidence path.
No manual SSH, SCP, report copy or remote profiler installation is required.
`profile-report` works offline and verifies **historical evidence**, not the
current remote implementation. Rerun discovery after environment/backend profile
changes. Compare a reference using a separate frozen task and the same workload
protocol; the MetaX adapter currently supports the candidate variant only.

Use separate captures for timelines and counters. Their instrumentation and
replay behavior differ. Do not run them concurrently on the same device.

## Workload boundaries and timing semantics

Without `capture_range`, the timeline includes initialization, compilation,
warmup, validation and measured work. For a scoped timeline, place exactly one
start marker after warmup and device synchronization, then synchronize measured
GPU work before the stop marker. The configured API names must appear in the
trace. Selection requires unique API correlation and rejects GPU work that
crosses a boundary. `start_from_api` controls mcProfiler's `--custom` mode; it
does not by itself trim an mcTracer export.

The supported mcTracer schema uses **nanoseconds**, verified against the
installed MCPTI activity timestamp contract. Do not apply the Chrome trace
microsecond default. Epoch timestamps remain integers; precision-losing floating
point timestamps are rejected. Other unit conventions require an explicit new
export adapter, not changing this task field to guess a scale.

The parser distinguishes:

- Kernel duration: one complete GPU kernel event's `dur`.
- Kernel interval union: covered kernel intervals per device **within one trace
  file**, accounting for overlapping queues.
- Same-queue kernel gap: time between disjoint kernel intervals on one queue.
- API end to kernel start: signed difference, only with unambiguous correlation.
- Submit to start: difference when an explicit submit timestamp exists.

A kernel gap is not proof of device idle time: copies or other queues may run.
An API-to-kernel difference can include queueing and dependencies; it is not
proof of host submission overhead. Trace `pid` is a display track, not an OS
process ID. The adapter does not infer missing identities. It reports matched
kernel/total kernel counts; multi-process or duplicate correlation IDs remain
unlinked. Counter values retain their vendor units and report scopes; cycles
are never converted into latency using an assumed frequency.

Profiler timings are diagnostic only. Keep independent, uninstrumented formal
benchmarks for candidate promotion and interleaved performance confirmation.

## Evidence and failure behavior

Each capture creates `runs/profile-evidence/<id>/`:

| Artifact | Meaning |
| --- | --- |
| `record.json` | Status, tool discovery, before/after source identities, guards, validated capabilities and artifact hashes |
| `task.json` | Exact contracted configuration |
| `collector.py`, `parser.py`, `profile-runner.py` | Collection/interpretation implementation snapshot |
| `guard-*.log`, `transport.stderr.log` | Guard and transport evidence |
| `raw/capture.json`, `raw/argv.json`, `raw/version.*.log` | Actual binary hash, version output, arguments and capture status |
| `raw/target-exits.jsonl` | Workload completion for each recorded replay |
| `raw/tool.*.log` | Profiler stdout/stderr, including failures |
| `raw/trace/` or `raw/vendor-export/` | Original vendor exports |
| `analysis/normalized.json` | Structured interpretation and limitations |
| `analysis/{kernels,apis,gaps,devices,metrics}.csv` | Applicable tables; absent tables indicate unsupported/not-applicable data |

A zero exit code is insufficient. Missing exports, swallowed workload errors,
unknown schemas, missing requested counters in any report scope, invalid units,
changed sources, changed environment, or a busy guard reject success. Collected
raw data is retained when validation fails; it does not become benchmark data.
`metric_coverage` lists requested, available and missing metrics by report.

The collector uses a relative `mcTracer --odname` because the validated version
incorrectly prefixes its working directory to an absolute output path and can
still exit zero without a trace. mcProfiler writes next to its executable; KDA
copies only the new output directory identified in this invocation's completion
message, checks its location and size, and leaves previous vendor output intact.
Concurrent captures in the same installation are not supported.

Artifacts use a bounded archive (64 MiB by default) with path, member and SHA-256
checks. A transfer failure or oversized export may leave the complete data only
on the executor, with a failed local record; do not report such a run as passed.
A timeout stops only the newly created collector-owned process group, never
unrelated workloads. No history is deleted automatically.

Current limitations include one API-delimited range per trace file, required
recognized kernel events in every parsed trace JSON, no cross-file/process GPU
interval merging, and export-specific counter parsing. Hardware counter
availability must be verified for each requested metric; successful `Total
Cycles` collection does not establish support for every stall/cache/bank metric.
Hash checking detects accidental artifact changes, not a malicious rewrite of
both files and their unsigned manifest.

## Validation for maintainers

```bash
python3 -m unittest discover -s tests -v
```

`tests/test_metax_profiling.py` covers executable discovery, fake-tool full
capture, missing exports despite zero exit, workload failure, guards, source
changes, units, correlation ambiguity, interval unions, capture boundaries,
metric coverage, archive safety and offline tamper detection. Keep synthetic
fixtures in KDA and device-specific real captures in the task workspace. A new
backend/export version needs a real idle-device smoke capture in addition to
these CPU tests.

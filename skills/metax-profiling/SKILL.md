---
name: metax-profiling
description: Collect and interpret MetaX counter and timeline evidence through KDA without mixing profiling and benchmark metrics.
---

# MetaX profiling

Read the task contract and run `discover-tools` then `strategy`. Only use
successful environment probes. `mcprofiler` is a counter profiler;
`mctracer` is a timeline candidate. Discovery is not proof of capture support.
Set binary locations in task environment using KDA_MCPROFILER_BIN and
KDA_MCTRACER_BIN, or use the normal executable search path. Do not install,
change global libraries, clocks, drivers or privileges during analysis.

Keep correctness first. Capture into a unique run directory with source and
protected-input hashes before/after, command logs and all raw exports.
Stop on a busy/unknown device. Preserve the actual profiler version and errors.
Do not use replayed/instrumented duration as the benchmark promotion metric.

Normalize only recognized, explicit timestamp units and schemas. Preserve
unknown formats and fail closed; missing timestamps or stream/process IDs
must not silently become zero. Counter reports cannot imply event order.
For timelines, separate per-kernel duration, interval union and same-queue
gaps. Overlapping queues must not be counted as serial gaps. Do not label a
gap as host delay without matched API correlation and compatible clocks.
When API linkage is missing or ambiguous, state that causality is unknown.

Record discovered, capture-validated and analysis-supported capabilities
separately. Tool failures are capability evidence, not kernel regressions.

Use the built-in `adapter: metax` task configuration described in
[MetaX profiling](../../docs/metax-profiling.md). Run `run profile --tool mctracer`
for timelines or `run profile --tool mcprofiler` for counters, then verify the
returned ID with `profile-report --run <id>`. Protect the workload and guard
files in the task contract. Explicitly scope measured work with `capture_range`
when initialization/warmup would contaminate analysis. Report requested counter
coverage and API correlation coverage; partial linkage does not prove host
causality. Read the implementation's documented limitations before interpreting
an export. Real device reports belong in the task workspace, not KDA.

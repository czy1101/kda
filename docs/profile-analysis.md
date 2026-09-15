# Capability-driven profile analysis

KDA keeps the correctness and benchmark loop backend-independent. Profiling,
compiler artifacts, and optimization extensions are selected from the task's
`analysis` policy and the selected backend profile's declared capabilities.

## Live tool discovery

Backend declarations are candidates, not proof of installation. Before
strategy selection, run:

```bash
python3 scripts/kernelpilot.py discover-tools --workspace <task-workspace>
```

Each backend profile supplies read-only probes. KernelPilot executes them after
the task's `environment.setup`, on the remote executor for SSH tasks, and writes
`runs/analysis-capabilities.json` in the controller workspace. The record
includes available and unavailable tools, versions when obtainable, and the
selected default. Its structure is defined by
`schemas/analysis-capabilities.schema.json`. Discovery never installs or
changes a profiler.

`analysis.preferred_tools` can reorder tools for a particular task. Otherwise
the backend profile's priority applies. `profile.tool` may name one explicit
tool or use `auto`. Tool-specific capture and export commands belong under
`profile.tools.<tool-id>` because those commands depend on the workload.

Presence is only the first gate. Capture permissions, useful kernel matching,
export support, and report readability are verified when the selected tool is
used. If that stage fails, record the failure and try another discovered tool
or the benchmark/compiler/source fallback.

Profiler-specific skills should be attached to the corresponding
`analysis_tools[].skills` entry, so discovering a framework or timeline tool
does not accidentally load guidance written for a different report format.

## Strategy order

When the next candidate is already justified, proceed without profiling. When
diagnosis is inconclusive:

1. Compare candidate and reference profiles when both are configured and the
   backend supports comparable profiling.
2. Otherwise analyze a candidate profile when a profiler and capture command
   are available.
3. Use a routed profiler skill first. If it is missing, incompatible, or
   inconclusive, use direct AI analysis only on deterministic structured
   exports and record the fallback.
4. Include attributable IR or instruction artifacts when they distinguish
   competing explanations.
5. Fall back to source and benchmark evidence when backend tooling is limited.

The `.ncu-rep` file is preserved evidence, not assumed to be model-readable.
The task should provide an export command that produces stable text or CSV for
analysis.

## Reference comparison

Candidate and reference captures must use identical semantic inputs, shapes,
device conditions, warmup, and capture policy. Compare end-to-end behavior
before individual kernels. Different implementations may fuse or split work,
so a one-to-one kernel comparison is optional rather than assumed.

## Optional extensions

An extension is selected only when the backend profile marks it `available` or
`experimental` and the task policy does not disable it. `required` fails when
the extension is unavailable; `auto` records the reason and continues through
the standard optimization route.

TLE is disabled by default in the task template. Changing its policy to `auto`
or `required` triggers a task-local assessment: the operator project must
contain the configured `tle-wiki` directory, and a read-only API probe derived
from that Wiki must succeed in the contracted environment. Only then may the
TLE optimization skill be loaded and documented primitives be used. A conda
environment named `tle` is not proof that the extension API is available.

Write the assessment to `runs/tle-assessment.json`. Record missing Wiki/API,
version mismatch, compilation/runtime failures, and fallback. When the
diagnosed bottleneck needs an operation neither Triton nor the available TLE
API can express, report the specific missing primitive as an extension request
instead of inventing an API.

For SSH tasks, use `kernelpilot remote list-wiki` and `remote read-wiki --path`
to read the configured Wiki directly through the controlled adapter. These
operations cannot write Wiki content and reject paths outside the configured
root. An `auto` assessment may fall back; a failed `required` assessment stops
the run and restores the best checkpoint.

## Inspection command

After filling the task contract, inspect the deterministic route with:

```bash
python3 scripts/kernelpilot.py strategy --workspace <task-workspace>
```

This command selects from the recorded live capabilities; it does not run the
profiler.

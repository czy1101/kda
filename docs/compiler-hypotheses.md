# Compiler evidence and falsifiable hypotheses

This increment makes compiler observations reproducible and candidate intent
checkable. It does not generate kernel optimizations, infer a bottleneck from a
resource count, or change correctness/performance promotion requirements.

## Separation of responsibilities

- Core: source/contract binding, input identity matching, artifact hashing,
  byte/text comparison, explicit metric semantics and prediction assessment.
- Backend adapter: resource field interpretation with supporting provenance.
  `compiler_backends.py` contains the optional MetaX normalization; no MetaX rule
  is applied to another backend. Generic producers supply explicit metrics.
- Task: input specifications, probe, compiled instances, hypothesis, desired
  metric changes, and original benchmark/acceptance policy. Core contains no
  operator names, shape lists, dispatch rules or hardware resource thresholds.
- Agent: interpret the evidence, design a coherent candidate, and decide whether
  a disproved mechanism is worth timing for another documented tradeoff.

## Contract (opt-in; use a fresh round when changing a frozen contract)

```yaml
analysis:
  hypotheses: required  # optional is the backward-compatible default
  compiler:
    probe: diagnostics/compiler_probe.py
    guard_command: python diagnostics/device_guard.py
    adapter: generic  # or metax-triton with target.backend: maca
    workloads: [forward/fp32/a, backward/fp16/b]
    max_artifact_bytes: 67108864
constraints:
  # Keep the task's existing source allowlist and protected files.
  forbidden_paths:
    - diagnostics/compiler_probe.py
    - diagnostics/device_guard.py
```

The probe must be a protected file already present on the executor. Use the
ordinary controlled KDA task setup to prepare it; do not add diagnostics to the
operator source allowlist. Capture neither installs packages nor rewrites a
backend or task. The device guard returns JSON with `idle: true` and a stable,
nonempty `environment_id`. It runs before/after the probe, outside any timing.
Boundary checks do not prove the device was idle throughout execution.

`compile capture` runs the probe via the task's environment and the same
ControlledSSH adapter as other remote work. A bundled pure-Python exporter is
injected as the `kda_compiler` dictionary; no KDA package install is needed on
the executor. The probe constructs exactly the declared cases and emits one
JSON envelope to stdout. Send extra diagnostics to stderr.

A probe for a Triton backend may use:

```python
# The task prepares inputs, obtains the exact compiled instance, and ensures
# runtime resource metadata is available. Do not infer input identity from an
# unordered compiler cache. This snippet assumes those objects already exist.
case = kda_compiler['triton_case'](
    workload_id, input_specification_sha256, compiled_instance,
    driver_source=installed_driver_source_if_available,
)
kda_compiler['emit']([case], compiler={
    'name': compiler_name, 'version': exact_compiler_version,
    'backend': backend_name,
})
```

For a non-Triton compiler, emit cases with `workload`, `kernel`, `input_id`,
`config`, `artifacts`, `metrics` and optional `raw_resources`. `artifacts` maps
stage names to `kda_compiler['artifact'](text_or_bytes)`. `metrics` maps names to
`{value, unit, scope, definition, provenance}`. Values must be nonnegative finite
numbers, and semantics must be explicit. Metrics are producer assertions, not
hardware-counter measurements. The producer and its dependencies must be frozen
as protected inputs. The raw protocol is illustrated by the synthetic tests.

`input_id` is the SHA256 of a canonical input specification including shape,
dtype, strides/layout, relevant scalar arguments and seed/input construction.
It is not a hash of source or just a friendly case name. The trusted probe must
actually use that specification; echoing a hash is not independent attestation.
Each workload/kernel pair must appear exactly once and the workload set must
match the task. Multiple kernels for a workload are allowed; changed kernel
coverage cannot silently be compared as equivalent. A fusion or rename needs an
explicit future mapping or a separate diagnostic, not a guessed match.

## Commands and ordering

```sh
kernelpilot compile capture --candidate baseline
# Save the returned run_id as the parent_capture in a hypothesis JSON file.
# Register while canonical source still matches that parent capture:
kernelpilot hypothesis register --spec hypothesis.json

# Edit/deploy candidate c001 using ordinary allowed-source operations.
kernelpilot run correctness --candidate c001
kernelpilot compile capture --candidate c001
kernelpilot compile compare --left PARENT_CAPTURE_ID --right CANDIDATE_CAPTURE_ID
kernelpilot hypothesis assess --candidate c001 --capture CANDIDATE_CAPTURE_ID

# Existing measurement and promotion requirements are unchanged:
kernelpilot measure --candidate c001
# confirm and checkpoint save-best only if ordinary gates are satisfied.
```

Capture itself is diagnostic and may precede correctness when debugging a
compiler. It never supplies correctness evidence. The normal optimization loop
runs full correctness before timing. In required mode, bound correctness and
capture require registration; benchmark and save-best also require an assessment
matching the exact current source/contract. The initial `baseline` identifier is
exempt from hypothesis registration. Existing baseline freezing and evidence
checks remain in force.

Registration cannot overwrite a prior ID. It requires unchanged parent source,
so a post-hoc explanation cannot masquerade as a pre-registered hypothesis.
Missing metrics require more evidence first, not an invented zero. Legacy tasks
remain optional; reports must not claim the new gate was used unless enabled.

## Hypothesis format

```json
{
  "id": "c001",
  "parent_capture": "RETURNED_PARENT_RUN_ID",
  "statement": "The temporary allocation may limit useful parallelism.",
  "change": "Shorten the lifetime of one temporary.",
  "expected_effect": "Reduce temporary storage without increasing repeated work.",
  "risks": "Earlier results might increase another live range.",
  "disproof": "Storage does not decrease, or the compiler undoes the intended change.",
  "predictions": [
    {"workload":"forward/fp32/a", "kernel":"kernel_entry", "kind":"metric",
     "name":"temporary_bytes", "relation":"decrease"}
  ]
}
```

Metric relations: `decrease`, `increase`, `equal`, `nonincrease`, `nondecrease`.
Artifact relations: `changed`, `unchanged` for a named stage. Comparisons are exact
static values; no statistical threshold or chip limit is hard-coded. Human
statements, risks and disproof text remain human reasoning. V1 automatically
assesses only the declared observable compiler predicates, not source liveness,
causal bottlenecks or final machine scheduling.

Predictions return `supported`, `not_supported`, or `unknown` (missing metric or
incomparable semantics). A negative prediction dominates the combined result;
otherwise unknown dominates supported. Per-prediction outcomes always remain.
The report always says `performance_status: not_assessed` and
`promotion_authorized: false`. Unsupported/unknown predictions are not hidden;
they do not by themselves prohibit a correctly validated candidate from being
timed for a tradeoff. They never bypass ordinary benchmark/promotion gates.

## What comparison proves and does not prove

Captures must match task/protected hashes, environment ID, compiler identity,
collector/adapter implementation, workload/kernel mapping and input IDs.
Candidate implementation and launch configuration may differ and are reported.
Every raw artifact is content-checked on use; changed text gets a unified diff.
Metric deltas require identical unit, scope and definition. Missing stages or
metrics are unknown, not zero. Raw fields and provenance are always retained.

Identical bytes mean identical saved artifacts, not necessarily identical runtime
performance. Different bytes may only reflect debug locations or embedded
metadata. These comparisons are not semantic equivalence checks. Registers,
shared memory and local bytes are not latency, occupancy or dynamic spill counts.
No resource increase automatically rejects a candidate or proves a compiler bug.

For `metax-triton`, local memory bytes are reconstructed from `n_spills * 4`
only when attached driver source matches the known `MC_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES`
assignment followed by division by four. Without that evidence the normalized
metric stays unavailable. Raw `n_spills` remains visible. This verifies a source
pattern, not that an adversarial binary was built from that source. Backend
versions with other formulas need another reviewed adapter rule and tests.

The exporter does not promise MIR, assembly or every IR stage. Only returned
artifacts are recorded. TLE importability or an async attribute in IR is not
proof of asynchronous hardware execution.

## Evidence, limits and validation

- `runs/compiler-evidence/<id>/`: before/after identities, guarded environment,
  raw stdout/stderr, task/exporter/adapter/collector source and hashed artifacts.
- `runs/compiler-comparisons/<id>/`: per-input resource/config/artifact deltas,
  text diffs and links/hashes of both parent captures.
- `runs/hypotheses/<candidate>/`: pre-registered statement and predictions tied
  to the parent capture and source identity.
- `runs/hypothesis-assessments/<id>/`: predicate outcomes tied to exact captures.

Failed captures remain recorded. Files are content-checked, not cryptographically
signed; trusted task producers and local storage remain part of the model.
Configured artifact limits bound accepted payloads after command completion,
not streaming subprocess memory. No physical device lease or automatic source
rollback is added by these diagnostic commands. Existing checkpoint/recovery
workflow remains responsible for deployment recovery.

CPU-only tests exercise two unrelated synthetic workloads, the injected exporter,
local execution, simulated ControlledSSH, stale identities, mutation/tampering,
missing/invalid artifacts, unit mismatches, unknown observations and required
workflow gates. They validate mechanics, not GPU speedups or real-server capture
compatibility. Historical operator artifacts can be inspected separately but
must not be relabeled as newly captured or pre-registered evidence.

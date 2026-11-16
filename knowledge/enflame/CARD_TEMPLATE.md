# STABLE-ID: specific mechanism or failure

## Generalization gate

- Reusable mechanism, interface or bounded backend limitation:
- Another operator/workload class to which it applies:
- Independent probe, mathematical argument or cross-workload evidence:
- Conditions that invalidate the claim:

If these cannot be stated, keep the finding in the task workspace. Do not
publish task winners, tuned constants, workload/timing tables or candidate
history. Evidence pointers may identify the original experiment; the card
must describe what can be reused without that experiment's tuning choices.

## Scope and status

- Observation date and review date:
- Device/backend/architecture:
- Framework, compiler, SDK/driver and device-library versions (unknowns explicit):
- Language, dtype/accumulation, M/N/K, strides/layout, launch options:
- Shape class, inference/training mode and numerical semantics:
- Status: hypothesis / compiler-observed / numerically validated /
  performance-observed / rejected / superseded (choose explicitly):

## Observed facts

Give mechanism-relevant before/after facts with artifact hashes/source IDs.
Separate compiler path proof, phase diagnostics and full-call performance.
Keep raw artifacts in the task workspace, not this repository.

## Mechanism and candidate hypothesis

Explain the proposed cause. State one expected change and what falsifies it.
Label inferred or unverified details. Show minimal pseudocode if useful.

## Reproduction and validation

Link the original commands and measurement policy in the task evidence.
Describe the independent reproduction/validation needed for another workload,
including numerical semantics, boundary cases and memory accounting. Put
task-specific comparator settings and measurement tables in that task's
report. Use KernelPilot for remote work.

## Limits, failures and revalidation

List negative results, regressions, fallback behavior, unknown hardware
mapping, shape/semantic restrictions and changes requiring a fresh probe.
Never inherit another task's regression permission or acceptance thresholds.

## Sources

Add source IDs to sources.json with repository-relative external artifact
paths, SHA256 digests and availability. Update catalog.json with this card's
ID, path, language, keywords, status and source IDs. Missing evidence must
remain explicit.

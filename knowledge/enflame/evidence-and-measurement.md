# S60-EVIDENCE-001: distinguish paths, runtime costs and hardware claims

## Evidence levels

| Evidence | Supports | Does not establish |
|---|---|---|
| Source / operation counts | Candidate mechanism, repeated work, complexity | Dominant runtime cost or hardware utilization |
| TTIR and GCU IR | Dot and lowered operand/result dimensions, transforms | Executed machine instruction or throughput |
| LLVM call site + matched specialization | Selected device-library entry | Physical unit mapping or occupancy |
| Synchronized phase diagnostic | Cost in that diagnostic under its instrumentation | Unperturbed end-to-end percentages |
| Same-contract full-call timing | Performance change on measured inputs | Generality to unmeasured classes |
| Successfully probed compatible profiler capture | Only counters/trace fields actually exported | Unsupported metrics or absent hardware details |

In the historical Torch run, per-operation synchronization increased the
instrumented total to 9433.55 ms vs the formal 3765.00 ms. Phase totals could
overlap or nest; do not add them into formal percentages. In the Triton
convgen case, a routing diagnostic improved about 21x; that does not make the
entire MoBA operator 21x faster.

## Execution and measurement checklist

- Use only KernelPilot for SSH inspection, artifacts and contracted commands.
  A knowledge card never expands the remote allowlist or permits direct SSH.
- Run declared live tool discovery in the exact environment. A previous task
  having no GCU profile means only that no declared matching profiler was
  available there; it does not prove that Enflame has no profiling tools.
  Do not substitute Nsight/CUDA assumptions or install tools during a run.
- Associate source hashes, cache keys, kernel names, compile options and
  framework/compiler/SDK versions with generated artifacts. Cache presence
  alone does not prove that the timed source executed that specialization.
- Separate cold compilation/startup from warm cached execution. Report both
  where requested. Pytest correctness-suite duration is not benchmark duration
  or operator latency; one benchmark item can internally sweep many shapes.
- Freeze timer type, synchronization, warmup, repeats, call boundary, inputs,
  comparator and device conditions. Synchronized wall time includes Python
  and host gaps; device-event timing measures a different boundary. Historical
  numbers with different contracts cannot establish Torch/Triton parity.
- Keep numerical correctness in separate tests. Performance commands can
  compile, warm up, synchronize, check options/devices/dependencies and time;
  omit output assertions, reference comparisons and accuracy/NaN/Inf checks.
- Report per-shape regression, aggregate metric and extra memory. Individual
  workspace budgets are not peak-memory limits. Say allocated vs reserved,
  tensor-only vs whole-device, and inference vs training.
- Test a justified shape class, with untuned tails/layouts/head ratios where
  relevant. Reconfirm routing complexity after attention becomes sparse.

## Knowledge maintenance

Record card IDs and current applicability in the plan; record confirming,
contradicting or inconclusive evidence in the final report. Keep raw task
artifacts outside KDA. Only update shared cards when repository writes are
authorized; otherwise leave a proposed card in the task's docs. Missing local
artifacts or stale software versions are reasons to revalidate, not reasons
to present remembered results as fresh measurements.

Sources: all report sources in [sources.json](sources.json), plus KDA's
[optimization prompt](../../prompts/optimize-kernel.md) and
[evidence format](../../docs/evidence-format.md).


## Training boundaries and nested-event follow-up

The executed GLA integration illustrates that a public training benchmark can
include logits transformation,gradient zeroing and framework autograd work
outside named GLA stage intervals. Isolated phase timing and event intervals
captured inside the actual public call are different observations. Preserve
the input/gradient mode and measurement boundary; do not sum nested intervals
as uninstrumented percentages or identify a particular library operation from
an unattributed residual alone. The diagnostic's instrumentation also has cost.

Repeated timing variation occurred even in paths whose source mechanism was
unchanged by the fallback experiment. The cause was not established. Retain
all configured alternating samples and report the per-shape gate,without
assuming a causal regression or relaxing the rule merely to deploy a large
class-specific win. Task-specific numbers and decisions remain external.

Sources:`gla-training-hierarchy`,`gla-training-hierarchy-raw`,
`gla-native-matrix-confirmation` and `gla-optimization-final` in
[sources.json](sources.json).


## Same-process branch and null-control follow-up (2026-10-22)

For an execution-option candidate, first verify that the changed option is
consumed by the workload's actual branch. On GCU300/Triton3.6.0, one public
operator's alternate branch bypasses the candidate option: paired execution
in the same process produced identical ordered cache hashes, linked binary
hashes, launch metadata, scalar arguments and tensor layouts. The changed
fallback kernels did not launch. This rules out a direct device-code/path
change on those measured inputs, not indirect effects in every environment.

Use fixed input tensors, warm both options, alternate order and add equally
instrumented same-setting null labels. Identical-code null groups showed
material timing differences in this environment. Such evidence establishes
measurement/execution variability; it does not identify clock, contention,
host scheduling or any other physical source without further observation.
Retain all trials and the original acceptance decision. An option comparison
and an unchanged-code timing fluctuation are different causal claims.

A time-budgeted benchmark with a single call longer than the budget yields
only one observation per round. Report that sampling policy explicitly.
Training boundaries can include expensive framework nonlinearities in both
forward and backward. Bound the actual public-call stages and verify suspected
primitive costs separately. Retained-graph primitive timing has a different
boundary from full training; gradient hooks also perturb execution. Do not
assign a nested-event residual wholly to an operation or sum independent
medians into exact percentages.

Sources: `gla-same-process-launch-controls`,
`gla-same-process-launch-controls-raw`, `gla-isolated-framework-primitives`
and `gla-native-application-diagnosis` in [sources.json](sources.json).

## Single-point vs cumulative attribution (2026-11-19)

When a multi-change candidate is reported as a single percentage, re-derive each
component from the common frozen baseline before attributing the gain. In the
GLA second round, the cumulative output-batching candidate (built on the
intra-batching candidate) measured 0.9736 vs baseline, but the same output
batching built directly on the baseline measured 1.00743 (0.61% slower); the
intra batching alone was 0.97890. A component's cumulative delta is therefore
not its isolated effect. Measure each single change with the same inputs,
device, warmup, alternating order and four-pair batch, keep the final
combination as a separate stage, and compare the combination with the product
(or sum) of the single-point deltas. When a change alters the autotuner's
selected configuration, add a fixed-config control so that the semantic change
and the configuration switch are not conflated. Retain all raw samples and label
unresolved mechanisms as pending rather than inferring them from the timing.

Sources: `gla-r2-attr-report`, `gla-r2-attr-manifest`,
`gla-r2-attr-ab-summary`, `gla-r2-attr-profile-summary`.

## Compile hash vs device code (2026-11-20)

Do not treat a changed Triton cache key as a device-code change. In the GLA
c009 attribution, `_chunk_fwd_state_scan_kernel` changed its Triton hash from
`f847a01a...` to `776d68c5...`, but its GCUIR, LLIR and fatbin were
byte-identical; the TTIR differed only in source-line numbers (230 -> 233) and
SSA/`#loc` numbering. Compare GCUIR/LLIR/fatbin (or disassembly) before
attributing a kernel-hash change to a code change. A changed source path or a
line-number shift is enough to change the cache key.

Sources: `gla-r2-attr-compiler-doc`, `gla-r2-attr-compiler-ir-baseline`,
`gla-r2-attr-compiler-ir-c009`.

## The four-link chain and the five change classes (2026-11-26)

Treat a causal claim as a chain with four links: **source**, **compiled
artifact**, **actual execution**, **performance change**. Each link needs its own
proof, and a claim may not borrow strength from a link that was never
established:

| Link | Proof needed | Typical break |
|---|---|---|
| Source | File, revision, `constexpr` and launch values | Two revisions of the same kernel name |
| Artifact | Identity of the specialization that ran, plus artifact hashes | "The" cache entry was a different specialization |
| Execution | Launch record tied to that same specialization | Artifact changed but the timed call used the old one |
| Performance | Same-contract A/B, same device, input, warmup, timed region | Warm candidate versus cold baseline |

Classify every candidate before reporting it, because the four classes are
measured differently and one is not the other: **kernel** (source),
**config** (tile/warps/stages/autotune selection or launcher options),
**compiler** (version, flags, passes, rules), **runtime** (submission,
allocation, synchronization, module load, host launcher, timed region). Add
**measurement** for the case where only the timer, repeats or warmup changed.
A source rewrite whose gain partly comes from the autotuner switching
configuration is kernel plus config: report the source delta at fixed
configuration, the configuration delta at fixed source, and the deployed
combination, and say whether the three add up. They need not — source,
configuration and code generation interact — so a subtraction is not a
decomposition.

Separate the three strengths of a numerical result: variants agreeing with each
other is **consistency**, the repository suite passing is **regression**, and an
independent high-precision reference passing is **accuracy**. They are not
interchangeable, and a scope that was only checked for finite outputs and
plausible magnitude is none of the three. A check that failed, was skipped, or
still needs re-running is never written as PASS; a criterion that was set too
strictly is reported as a criterion error, not as a numerical failure.

Depth of observation is not attribution strength. A hardware counter reports
activity of the measured unit within that counter's documented scope; the
reading does not establish that the activity caused the timing change, and an
uncontrolled counter reading is weaker evidence than a clean A/B with alternating
order. Record the two axes separately.

Negative results and their bounds belong in the record with the same weight as
wins: an unchanged median under a changed spelling, a rejected vectorization, a
measured regression all constrain what the next candidate may assume. State the
range in which a result holds (shape, dtype, tile, compiler version) instead of
generalizing from the one configuration that was measured.

Sources: the measurement sections above; the classification and chain wording
follow the runbooks in `docs/bottleneck-localization.md` and
`docs/enflame-compiler-diagnostics.md`.

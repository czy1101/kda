---
name: operator-optimization-report
description: Write or revise an operator optimization report with complete benchmark comparisons and attributable code and profiler evidence. Use for final reports and summaries of retained optimization methods, including cumulative step comparisons; do not treat report writing alone as authorization for new GPU experiments.
---

# Operator optimization reports

Read the task contract, the user's requested deliverable, and the recorded
commands and source versions before writing. Use the user's reporting language.
Repository instructions are maintained in English; they do not force an English
user deliverable.

## Report scope and delivery

- Keep one authoritative document at the user-selected path. Update it in place
  rather than creating a report per round or a bundle of documents.
- For a single-document deliverable, embed the facts needed to review every
  conclusion. Do not require another Markdown file, CSV, JSON, screenshot,
  profiler report, archive, or source file to understand the result. Commands,
  implementation locations, and revision hashes may appear as provenance, but
  they are not substitutes for inline evidence.
- A **successful-method summary** expands only mechanisms retained in the final
  implementation. It is not a list of all attempts. Keep rejected candidates,
  failed experiments, compiler-debugging history, and workflow errors in the
  internal evidence ledger. Include them in the deliverable only when the user
  requests an experiment or execution audit.
- Preserve all required rows in the formal benchmark comparison, including rows
  that became slower. Restricting the report to retained methods does not permit
  filtering out adverse results for those methods.
- Summarize correctness and development checks. Do not paste every passing or
  failing test case, full source files, or unfiltered tool logs to make the report
  look complete. Length is acceptable when it carries necessary evidence, not
  when it repeats raw artifacts.
- Use descriptive method titles and human-readable version labels, such as
  "first cumulative version" or the user's preferred equivalent. Keep machine
  candidate IDs as provenance rather than the main narrative. Use established
  technical terms; do not invent labels for ordinary implementation choices.

## Complete benchmark evidence for each method

For a retained-method report, make the comparator explicit for every method.
When the user requests cumulative comparisons, build the following sequence:

1. Preserve and measure the runnable original implementation as version 0.
2. Version 1 adds only the first retained mechanism to version 0.
3. Each subsequent version retains the preceding changes and adds only the next
   mechanism. Save the source snapshot, delta, and exact parent before measuring.
4. Check that the final cumulative version implements the retained best version;
   record any ordering-dependent change instead of presenting a different
   implementation as the same endpoint.
5. Run the entire task-required benchmark matrix for the baseline and every
   cumulative version. Compare version 1 with version 0, then version i with
   version i-1. Also report the cumulative comparison with version 0.

These reconstructed reporting versions are not necessarily the original trial
numbers. Explain their relationship once. Do not reuse the timing of an
unrelated parent or subtract isolated experiment improvements to fabricate a
cumulative result.

Lock the standard benchmark script and configuration across the sequence:
required shapes, dtypes, providers, layout, physical mapping, seeds, algorithm
parameters, backend, device, warmup, repeat interval, sample count, synchronization,
and measured scope. State the actual command, working directory, relevant
environment, and any non-default settings. Use the task's standard defaults;
do not hard-code one operator's shapes, timing values, or accelerator into this
skill. Separate end-to-end API measurements from preallocated kernel timing.
Run comparisons serially on the same device to avoid mutual interference.

Representative-shape or shorter-duration screening remains screening. It cannot
stand in for a complete standard matrix for a retained method, even if the final
combined implementation has a complete result. If coverage is missing, say so;
obtain authorized measurements before declaring that method fully benchmarked.
Do not infer a full table from a few successful examples.

Respect existing authorization and the task's cost and execution constraints.
Use KDA's controlled interface for remote commands and source synchronization.
A report-only request does not independently authorize expensive reruns. An
explicit request to supplement benchmarks does authorize those measurements;
do not ask again for work already authorized. During benchmark-only reruns, an
explicit user instruction to reuse prior correctness evidence takes precedence
for that rerun: state that correctness was not rerun, distinguish historical
validation from the reconstructed versions' assumed status, and do not turn
this into a general policy for promoting untested new implementations.

For every required row, include both versions' absolute values and all available
benchmark statistics: median or the contract's statistic, min/max, CV or other
reported dispersion, unit, and sample protocol. Identify the full shape, dtype,
mode, and path, either per row or through unambiguous table headings and fixed
configuration. For latency:

- adjacent speedup = previous latency / current latency;
- adjacent reduction = (1 - current latency / previous latency) * 100%;
- cumulative speedup = original latency / current latency.

For a higher-is-better metric, use current / comparator and label the direction.
Report an aggregate only with its weighting and definition. An equal-weight sum
of per-shape medians is a comparison of the matrix, not the duration of one joint
execution. Keep per-row regressions and noise visible; do not claim that every
shape improves because the sum improves. Reuse the official stability criterion,
and qualify small changes relative to the observed variation.

## Explain each retained mechanism

Use this reading order, adapting headings to the user's reference style:

1. **Problem and evidence:** identify the overhead or bottleneck and affected
   workloads. Separate measured facts from the proposed explanation.
2. **Concrete modification:** explain what changed and why it could help.
   Embed the necessary before/after code or complete logical block/function,
   including conditions, masks, fallback behavior, and meaningful launch
   arguments. Do not use ellipses in a critical branch or replace code with a
   source-file link. Do not dump the entire implementation when a complete
   relevant block is enough.
3. **Complete benchmark comparison:** put the full matrix immediately after
   the method, with its actual parent and protocol. A brief per-mode or per-path
   summary can explain where the aggregate gain comes from.
4. **Interpretation and boundary:** explain the observed incremental effect,
   interaction with earlier methods, and remaining uncertainty. Do not assign
   the final combined gain to an isolated mechanism.

Embed the profiler and compiler indicators used by these explanations, including
units, configuration, workload, launch selection, and source version. Preserve
all relevant collected numerical indicators in readable tables; do not just
paste the favorable counters or refer readers to raw files. Include key generated
instruction or error excerpts only when needed to support a reported conclusion.
State "not collected" when per-step evidence does not exist. A profile of the
original and final kernels explains their combined change, not every intermediate
step. Label historical profiles explicitly when only benchmarks were rerun.
Profiler replay duration is not a benchmark speedup. Independently measured
stages are not automatically additive. API availability is not proof that the
compiler emitted a primitive. Do not say all conflicts disappeared when only one
conflict counter improved, or call an explored best the hardware's absolute limit.

## Review and final state

- Verify full matrix coverage, row pairing, units, ratio calculations, stability,
  and association with the recorded source revision and command.
- Check that code excerpts contain the entire necessary logic and that the
  document has no external evidence dependency or unrelated raw dumps.
- State which correctness checks were newly run, reused, or explicitly skipped.
- Preserve the original baseline. After temporary measurement variants, restore
  the approved best implementation and verify it against its checkpoint without
  overwriting a concurrent edit.
- Report final deployed/restored identity and measured remaining bottlenecks.
  Keep ordinary syntax or parser-debugging history out of a successful-method
  document, while retaining it in internal workflow records.

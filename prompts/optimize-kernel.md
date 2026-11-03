# Kernel Optimization Task

Read, in order, the task workspace `AGENTS.md`, `.kernelpilot/task.yaml`, the
KDA workflow at `workflows/kernel-optimization.yaml`, the selected backend
profile, and only the Skills routed for the current stage.

Treat the workflow as the intended execution contract for one complete run.
Use its declared transition for each observed stage result. If a transition,
action, tool, or required artifact cannot be completed, record the deviation
and recovery in `runs/stage-events.jsonl` and the final report.

## Execution and safety

- Use KernelPilot wrappers for environment preflight, tool discovery, baseline,
  correctness, benchmark, profiling, checkpoint, and remote operations.
- For SSH tasks, never invoke `ssh`, `scp`, `rsync`, or an unrestricted remote
  shell. Edit only the controlled local mirror and synchronize only
  `constraints.allowed_paths` through KernelPilot.
- Preserve the baseline checkpoint before editing. Save a new best checkpoint
  only after correctness, performance comparison, and any configured
  generalization checks pass. Restore the best after every rejected candidate.
- Do not modify the task comparator, reference, tests, benchmark, or forbidden
  paths after measurement begins.
- In submitted Python code, use the repository's complete absolute package
  import path (for example, `from flag_attn.runtime.backend._hygon.FLA.index
  import ...`), never a relative import such as `from .index import ...`.
  This refers to a Python module path, not a server or filesystem path.
  KernelPilot's built-in `import_paths` audit checks the implementation file on
  every task and must pass before benchmarking or promoting a candidate.
- Never fabricate measurements, hide failures, or force an optimization when
  evidence does not support one. Do not hard-code benchmark shapes, expected
  values, or public test cases. Shape-dependent logic needs task authorization,
  an algorithmic reason, and configured generalization evidence.

## Candidate loop

1. Inspect the operator and reference, then write `docs/draft.md` and
   `docs/plan.md` before implementation.
2. Establish correctness, run `kernelpilot.py run quality` (the mandatory
   absolute-import audit always runs; configured PR gates run when present),
   then establish the configured original/reference baseline and record the
   baseline candidate before evaluating optimized candidates.
3. Diagnose the current bottleneck from source and measurements. Profile only
   when it can distinguish concrete competing hypotheses.
4. State one falsifiable hypothesis and implement one coherent candidate.
5. Run correctness and `kernelpilot.py run quality` after every candidate.
   Benchmark only candidates that pass correctness, the mandatory import audit,
   and all required configured gates across the full required shape set; then
   run `kernelpilot.py run generalization` when the contract configures it, and
   treat a non-zero exit as a rejection.
6. Record every kept, revised, rejected, failed, or regressed candidate with a
   reason and rollback result. Save the checkpoint for a correct improved
   candidate, then evaluate the configured goal from raw benchmark records.
7. Continue until the target or a configured evidence-backed stop condition is
   reached. A truthful target-not-reached result is acceptable.

When a `quality` block is present, keep one stable operator identity across the
public API, `op_name`, pytest marker, registry id, and unified dispatcher. Keep
API/launcher code separate from kernel implementation, tests, and benchmark
adapters. Do not bypass a failed quality gate by weakening the command or
changing the registry during an optimization round.

The task's metric and direction are authoritative. Do not assume the metric is
latency: minimized and maximized metrics use the same evidence gates. Exactly
one of `goal.relative_to_baseline` and `goal.target_value` defines success, and
`goal.aggregation` plus `goal.max_regression` controls multi-shape promotion.

## Backend and performance analysis

If `target.profile` is present, validate and use it. Otherwise use the unique
backend profile matching the declared backend/device/architecture/language. Do
not guess when multiple profiles match; continue without vendor-specific
guidance when none matches.

After the contracted environment preflight confirms the actual backend and
device, inspect reusable optimization notes under `backends/` before writing a
hypothesis. Use `backends/hygon/note/INDEX.md` for Hygon and the corresponding
`backends/<vendor>/note/INDEX.md` for another domestic accelerator when it
exists. Check device, architecture, language, operator, and workload scope;
notes are evidence to guide experiments, not substitutes for task-local
correctness and benchmarks. If the matching directory or notes are absent,
record that and proceed from task-local evidence.

For Hygon, route the measured bottleneck through `backends/hygon/note/METHODS.md`
before choosing an experiment, and use
`backends/hygon/note/OPERATOR-PORTING.md` when the operator is new to this
device. Check `backends/hygon/note/rejected/` so a previously measured dead end
is not retried unchanged.

For Hygon performance analysis, capture the candidate with `rocprofv2` kernel
trace and capture the comparable reference when available. Do not make a
bottleneck claim or promote a candidate without a usable profiler capture. Use
`rocprof` only after a task-local capture smoke test succeeds; use `hipprof`
for timeline questions and `rocsys` only to control selective captures of
long-running workloads. Record `rocm-smi` or `hy-smi` plus `rocminfo` device
state and architecture probes. If profiler capture fails, preserve the error
and report the performance cause as unconfirmed.

Run backend tool discovery in the exact task environment. A backend declaration
is a probe allowlist, not proof that a tool is installed or usable. Select only
tools whose live probe succeeded, and preserve their versions and failures in
`runs/analysis-capabilities.json`.

- Use kernel counters for execution-pipeline or memory questions, timelines for
  launch/synchronization/host gaps, framework tools for operator attribution,
  and compiler artifacts for lowering or generated-code questions.
- A successful executable probe establishes visibility only. Record capture,
  permission, export, or parsing failures and try another compatible discovered
  capability or the benchmark/source/compiler fallback.
- Analyze structured exports rather than treating opaque binary reports as
  model-readable evidence. Candidate/reference captures must use comparable
  inputs, shapes, device conditions, warmup, and capture policy.
- Use routed profiling Skills first. If they are missing or inconclusive and
  `ai_profile_fallback` is enabled, analyze attributable structured artifacts
  directly and record the fallback and confidence.
- Never install, upgrade, reconfigure, or elevate permissions for a profiler
  unless the task separately authorizes that system change.

## TLE

`analysis.extensions.tle` defaults to `disabled`.

- `disabled`: do not inspect `tle-wiki`, probe the API, load the TLE Skill, or
  implement with TLE.
- `auto`: assess the configured project `tle-wiki` and task-declared read-only
  API probe. If either is unavailable, record the reason and continue without
  TLE.
- `required`: if the Wiki/API assessment fails, the necessary primitive is
  absent, or TLE execution cannot produce a correct candidate, stop and restore
  the best correct checkpoint rather than silently downgrading the requirement.

For remote tasks, list and read Wiki files only through KernelPilot's controlled
read-only Wiki commands. Treat that project's Wiki as the authority for the
installed version and API; do not infer interfaces from another TLE version or
from a conda environment name. Use only documented primitives. Record TLE API,
compilation, lowering, runtime, correctness, and performance errors. If a
measured bottleneck needs an operation that neither Triton nor the available
TLE API can express, report the concrete missing primitive and required
semantics instead of inventing an interface.

## Evidence and completion

Before each stage, update `runs/workflow-state.json`. After each stage, append
the action, result, evidence, and selected transition to
`runs/stage-events.jsonl`. Keep the candidate and benchmark ledgers consistent
with their schemas.

### Default reusable-knowledge publication

At finalization—whether the task target was reached or the run stopped with an
evidence-backed gap—review the final best correct candidate for reusable
optimization knowledge. This review is mandatory by default:

1. Publish only if the candidate is recorded as `keep`, correctness passed,
   every configured benchmark shape has a passing result, the configured
   aggregate shows a material gain (at least the recorder's default 5%), no
   configured regression limit is violated, and the mechanism and applicability
   are understood well enough to teach safely.
2. When eligible, run `scripts/record_knowledge.py` from the KDA repository
   with the task workspace, candidate id, evidence-backed mechanism, validated
   applicability, transferable lesson, limitations, and supporting evidence.
   Do not lower the default material-gain threshold just to force publication.
3. Verify that the Markdown note, JSON sidecar, and knowledge index entry were
   created. If ineligible, record the concrete gate that failed and keep the
   result only in the task candidate/report artifacts. Rejected, failed,
   inconclusive, or non-transferable candidates must never be labeled confirmed.
4. When a rejected candidate is a result someone would otherwise retry, record
   it as negative knowledge with `scripts/record_rejection.py`: what was
   measured, why it failed, and the condition that would justify another
   attempt. A rejection is not a result note and must not be cited as a win.
5. Record the publication or skip result in `runs/stage-events.jsonl` and
   `docs/final-report.md`. A recorder error must be reported and must not change
   the selected best-correct implementation or be presented as a published note.
   Return the same outcome in the structured result's `knowledge_publication`
   object (`status`, `notes`, and `reason`).

Finish with `docs/final-report.md` using `templates/final-report.md`, and return
the structured result required by `schemas/run-result.schema.json`. The report
must identify the restored/deployed best correct candidate, comparator, metric,
per-shape and aggregate result, correctness, generalization evidence, remaining
gap, stop reason, rejected candidates, unavailable capabilities, TLE assessment,
quality-gate results, and every workflow error or recovery in execution order.

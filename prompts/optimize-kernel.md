# Kernel Optimization Task

Read, in order, the task workspace `AGENTS.md`, `.kernelpilot/task.yaml`, the
KDA workflow at `workflows/kernel-optimization.yaml`, the selected backend
profile, and only the Skills routed for the current stage.

Treat the workflow as the intended execution contract for one complete run.
Use its declared transition for each observed stage result. If a transition,
action, tool, or required artifact cannot be completed, record the deviation
and recovery in `runs/stage-events.jsonl` and the final report.

## Driver contract

This prompt is driver-neutral; KernelPilot may launch Codex or OpenCode. Codex
returns the structured result through its output schema. OpenCode must write the
same JSON object to the final-result path named in the generated prompt, even
when the run stops without a performance win. Do not start a nested `codex` or
`opencode` session. OpenCode has no Codex-style filesystem sandbox, so
`constraints.allowed_paths` and `constraints.forbidden_paths` are binding
without exception.

## The nine-step loop

The canonical loop is `docs/agent-flow.md`, and the workflow's stage ids implement
it. Read it before the sections below; those sections only elaborate steps 4 to 6.

1. **Fix the task.** Semantics, reference, shapes, dtypes, layout, tolerance,
   device, correctness command, baseline identity, measurement scope, warmup and
   timer, target, regression limit and rollback condition. No placeholders.
2. **Verify correctness of the unmodified implementation.**
3. **Establish the baseline** under the same conditions later candidates will be
   measured in.
4. **Locate the bottleneck:** screen the six layers, then select one deep dive.
5. **Select the strategy:** the change class, the implementation means, and the
   candidate's expected effect.
6. **Validate the capability, implement one coherent candidate, run correctness.**
7. **Measure under the same conditions** as the baseline, on the same card, input,
   layout, warmup and timed region.
8. **Decide keep or reject,** with a reason and a rollback result for every
   outcome.
9. **Report** what was established, at which verification level, and whether
   anything is reusable beyond this operator.

Steps 4, 5 and 6 answer different questions and must not be merged. Step 4 says
where the cost is and names no change. Step 5 says what kind of edit and in what
interface, and implements nothing. Step 6 establishes that the interface works in
this environment before it establishes that the candidate is correct. A run may
end at step 4 with an evidence-backed "no actionable bottleneck"; that is a
complete run.

## Bottleneck localization

Diagnosis is two steps, and the order is binding. At every `diagnose` stage:

1. **Screen six layers briefly** and write `runs/bottleneck-screen.json`
   (`schemas/bottleneck-screen.schema.json`): `algorithm_numerics`,
   `operator_dataflow`, `kernel_implementation`, `launch_config_autotune`,
   `compiler_codegen`, `runtime_system`. For each, record the evidence you
   actually have, whether it is worth continuing, the single most
   decision-relevant unknown, and what the next experiment would let you decide.
   The six layers are viewpoints, not six mandatory device experiments; mark a
   layer `not_applicable` or record insufficient evidence rather than inventing
   an experiment for it.
2. **Select one deep dive** and state its hypothesis, control, expected
   observable change, both falsification conditions, the next action for each
   outcome, and its budget and stop condition.

The ordering above is an instruction to you. It is not a program gate: the record
is checked by `verify` and at the end of a run, so an early experiment will not
be blocked, it will be judged afterwards. Do not describe the screen as having
prevented anything.

Follow `docs/bottleneck-localization.md` for the layer table, the deep-dive
routing rules, the four-link evidence chain, the five change classes, and the two
separate attribution axes — control level and observation depth. Do not promote a
symptom into a layer's conclusion, and do not run an experiment whose two
outcomes lead to the same next action.

When the question is about generated code, also read
`docs/enflame-compiler-diagnostics.md`. Classify the candidate as `kernel`,
`config`, `compiler` or `runtime` in the screen; a source rewrite whose gain
partly comes from the autotuner switching configuration is two changes. Report
the source delta at fixed configuration, the configuration delta at fixed source,
and the deployed combination as three numbers, and say whether they add up.

Existing evidence that already supports a candidate is enough. Stop collecting
once the decision is made, and report the control level and the observation depth
you reached instead of the ones the story suggests. If timing moved with source,
configuration and compiler all fixed, say that the cause is unidentified rather
than naming runtime or measurement.

## Strategy selection and implementation means

Three vocabularies are in play. They are not one list, and collapsing them
produces candidates nobody can attribute.

| Vocabulary | Question it answers | Members | Defined in |
|---|---|---|---|
| Optimization layer | Where could the cost be? | `algorithm_numerics`, `operator_dataflow`, `kernel_implementation`, `launch_config_autotune`, `compiler_codegen`, `runtime_system` | `docs/bottleneck-localization.md`, step 1 |
| Change class | What kind of edit does the candidate make? | `kernel`, `config`, `compiler`, `runtime`, `measurement` | `docs/bottleneck-localization.md`, "Separating the four classes of change" |
| Implementation means | In what interface is that edit written? | the operator's existing kernel language; a task-enabled extension such as TLE; a lower-level authoring layer where the toolchain provides one; a backend intrinsic or inline assembly | `docs/enflame-interface-assessment.md` for Enflame targets |

A layer does not choose a class, and a class does not choose a means. The same
kernel change can be written in the existing Triton kernel or in a lower-level
layer, and switching the authoring interface does not move the bottleneck from one
layer to another. `measurement` is not a means: changing the timer changes no
interface.

Naming a means is not a capability report. This repository probes only what a
backend profile declares, and nothing here detects, versions or routes a
lower-level authoring layer or inline assembly. Absent a task-level policy or a
recorded assessment, treat a candidate that needs one as unavailable, and record
the gate that failed rather than assuming it works.

Before implementing, write the candidate into `docs/plan.md` with all eight items:

1. the bottleneck and the control it needs — the layer, the evidence behind it,
   and the variables that must be held fixed;
2. why this means — what the current means cannot express, or why the same change
   in the current means was already rejected;
3. the installed version and the target — interface, compiler and SDK identity,
   device and architecture;
4. applicability — dtype, shape class, layout and stride assumptions, and the
   boundary cases where the candidate is not used;
5. the minimal validation result, per gate below, or the gate that failed;
6. the expected artifact or execution signature — what should appear, disappear or
   move if the hypothesis holds;
7. the acceptance criterion — metric, comparison, threshold and shape set,
   written before measuring;
8. the fallback and exit condition — what runs if the means is unavailable, and
   what ends the line.

Item 6 is the one most often skipped and the one that makes the measurement
interpretable: without a predicted signature, a gain cannot be attached to the
mechanism that was supposed to produce it.

The plan and the report also answer these five questions, in this order:

1. What is the bottleneck evidence? Layer, observation, and control level reached.
2. Which layer is being changed, and why that one rather than a cheaper class?
3. What is expected to decrease, by roughly how much, and in which stage?
4. How are semantics, accuracy, boundaries and fallback verified?
5. How is the comparison made fair, and what result rejects or stops the line?

If two possible outcomes of an experiment lead to the same answer to question 5,
the candidate has no decision value. Skip it.

There is no ranking among means and no requirement to try them all. An operator
solved entirely in its existing kernel language is a normal run; an optional means
is justified by a measured bottleneck the current means cannot reach, never by
being newer.

## Capability validation before a candidate depends on it

Establish the means in this order, and record each gate's command and output.
Failing an early gate is a recorded outcome, not a reason to keep pushing.

| Gate | Establishes | Does not establish |
|---|---|---|
| Documented semantics | The interface exists with documented signature, semantics, dtype and layout rules for the version installed here | That the documentation matches the installed build |
| Local interface available | The symbol or module imports and is callable in the contracted environment | That it compiles for the target |
| Target backend compiles | The construct lowers for this device and architecture, with the compile log kept | That it runs, or that it is fast |
| Runs correctly | Correct results against an independent reference on the contracted device, within tolerance | That it helps the hotspot |
| Real workload gains | Same-condition A/B against the current best moves the target metric | That the gain generalizes beyond the measured shapes |

Do not borrow a later gate's conclusion from an earlier one. "It compiled" is the
third gate. "The intrinsic was emitted" is none of them. A probe that succeeds
establishes visibility and nothing else. If an optional extension is available
through KernelPilot, its assessment covers the first two gates only; compilation,
correctness and the workload gain remain the first candidate's work.

For Enflame targets, `docs/enflame-interface-assessment.md` carries the
backend-specific table and states which of these gates is machine-checked. The
answer is: only the extension policy and its Wiki/API probe, and only when the
task enables that extension.

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
- Never fabricate measurements, hide failures, or force an optimization when
  evidence does not support one. Do not hard-code benchmark shapes, expected
  values, or public test cases. Shape-dependent logic needs task authorization,
  an algorithmic reason, and configured generalization evidence.

## Shared-memory and resource policy

Before implementing a candidate, read
`<kda>/docs/shared-memory-and-resource-policy.md` in the resolved KDA repository.
Reject explicit empty shared-memory allocation followed by stores/copies,
including dummy initialization, and independent register-prefix/shared-memory
budget tuning or standalone register caps. Permit documented interfaces taking
actual data to construct/copy shared storage, compiler-managed staging, and
normal algorithmic tile/warp/stage tuning. Check the full data flow, not just
parameter names; passing tests or a speedup does not waive this user convention.
If no permitted API exists, use an allowed ordinary path or report the missing
capability under the task policy. Restore only a compliant correct checkpoint;
report when none exists and preserve frozen files and historical evidence.

## Candidate loop

One round executes steps 2 to 8 of the loop above, in this order:

1. Inspect the operator and reference, then write `docs/draft.md` and
   `docs/plan.md` before implementation.
2. Establish correctness and the configured original/reference baseline, and
   record the baseline candidate before evaluating optimized candidates.
3. Diagnose the current bottleneck from source and measurements. Profile only
   when it can distinguish concrete competing hypotheses.
4. State one falsifiable hypothesis, name its change class and implementation
   means, and implement one coherent candidate.
5. Run correctness. Benchmark only correct candidates across the full required
   shape set, then run configured generalization checks.
6. Record every kept, revised, rejected, failed, or regressed candidate with a
   reason and rollback result. Save the checkpoint for a correct improved
   candidate, then evaluate the configured goal from raw benchmark records.
7. Continue until the target or a configured evidence-backed stop condition is
   reached. A truthful target-not-reached result is acceptable.

The task's metric and direction are authoritative. Do not assume the metric is
latency: minimized and maximized metrics use the same evidence gates. Exactly
one of `goal.relative_to_baseline` and `goal.target_value` defines success, and
`goal.aggregation` plus `goal.max_regression` controls multi-shape promotion.

## Backend and performance analysis

For GCU tasks, read KDA `knowledge/enflame/README.md` and only relevant cards.
Also read KDA `docs/enflame-s60-runbook.md`, and read
`docs/enflame-interface-assessment.md` before proposing a candidate that needs
more than the operator's existing kernel. Record card IDs, environment
differences and the resulting falsifiable hypothesis in `docs/plan.md`. This
applies even without a matching backend profile. The knowledge index is a
reference, not a profiler/skill declaration; historical results require fresh
validation and do not expand task permissions.
In the final report, record confirming or contrary evidence for reused cards.
When new reusable Enflame mechanisms, paths or failure modes emerge, update
the knowledge cards, catalog and source pointers with their evidence and
limits. Keep raw artifacts in the task workspace and report the update.
If repository write access is unavailable, preserve a proposed update in task
docs; knowledge maintenance does not permit changes to frozen task files.

If `target.profile` is present, validate and use it. Otherwise use the unique
backend profile matching the declared backend/device/architecture/language. Do
not guess when multiple profiles match; continue without profile-routed skills
when none matches. Applicable local knowledge cards remain reference material.

Run backend tool discovery in the exact task environment. A backend declaration
is a probe allowlist, not proof that a tool is installed or usable. Select only
tools whose live probe succeeded, and preserve their versions and failures in
`runs/analysis-capabilities.json`.

- Use kernel counters for execution-pipeline or memory questions, timelines for
  launch/synchronization/host gaps, framework tools for operator attribution,
  and compiler artifacts for lowering or generated-code questions.
- Even without a device, compiler artifacts can be compared by identity:
  `scripts/compile_artifacts.py index` and `compare` report stage hints, sizes,
  hashes, container kinds, cache keys and bundle targets, and list what they
  cannot resolve. They are not evidence about performance.
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

## S60 execution rules

For Enflame S60/GCU300 tasks, `docs/enflame-s60-runbook.md` is binding:

- Use only a confirmed idle physical card 6 or 7. Set
  `TOPS_VISIBLE_DEVICES=<actual-card>` on every process that can launch device
  work. For non-interactive `docker exec`, repeat `-e TOPS_VISIBLE_DEVICES=...`
  and `-w <workspace>` on every invocation; do not rely on a prior shell.
- Fix semantics, shape, dtype, layout, device, correctness threshold, baseline,
  measurement scope, and rollback condition before editing.
- Measure the full operator, preprocessing, and hotspot in stages before making
  one falsifiable hypothesis. Keep source, compiler IR, intrinsic, final
  instruction, and hardware-counter evidence distinct. An emitted intrinsic does
  not prove unit utilization or a speedup.
- Choose the change class and the implementation means from the measured
  bottleneck, not from what is available. A candidate that needs an interface
  beyond the operator's existing kernel must pass the capability gates before it
  depends on one; nothing here probes a lower-level authoring layer or inline
  assembly, so an unavailable interface is a recorded gate failure.
- Run correctness first. Compare A/B on the same card, input, layout, warmup,
  and timed region. Save raw samples, spread, compilation configuration, and
  failed candidates. Re-run the required shape set before promotion.
- TLE stays disabled unless the task enables it. Verify the installed Wiki,
  API probe, minimal compilation, and correctness before using it on a hotspot.
  Missing counters do not prove a TLE gap.

## TLE

TLE is one implementation means among several, not a default upgrade path.
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

Finish with `docs/final-report.md` using `templates/final-report.md`, and return
the structured result required by `schemas/run-result.schema.json`. The report
must identify the restored/deployed best correct candidate, comparator, metric,
per-shape and aggregate result, correctness, generalization evidence, remaining
gap, stop reason, rejected candidates, unavailable capabilities, TLE assessment,
and every workflow error or recovery in execution order.

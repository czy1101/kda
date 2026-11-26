# Selecting an implementation means, and assessing it before use

Why this document exists: a run that reads "the bottleneck is in the kernel" and
immediately reaches for the most advanced available authoring interface has
skipped the question that decides the candidate. The bottleneck question comes
first, the change class second, the implementation means third. This page
describes the third one for Enflame targets, and how to establish that a means
actually works before a candidate depends on it.

It is guidance for the agent, recorded in `docs/plan.md`. No program reads it:
there is no schema, no evidence gate and no validator for the content below. The
one machine-readable assessment that exists here is TLE's, and it is described in
[its own section](#what-is-actually-machine-checked-and-what-is-not). Nothing on
this page establishes a speedup, and nothing here was measured on an S60 device in
this revision.

## Three enumerations that must stay separate

They answer different questions, have different vocabularies, and a run that
collapses them into one list produces candidates nobody can attribute. Do not
build a single "strategy" enum that mixes them.

| Enumeration | Question it answers | Members | Canonical definition |
|---|---|---|---|
| Optimization layer (diagnostic viewpoint) | *Where* could the cost be? | `algorithm_numerics`, `operator_dataflow`, `kernel_implementation`, `launch_config_autotune`, `compiler_codegen`, `runtime_system` | [bottleneck-localization.md](bottleneck-localization.md), step 1 |
| Change class (what an edit touches) | *What kind of edit* will the candidate make? | `kernel`, `config`, `compiler`, `runtime`, `measurement` | [bottleneck-localization.md](bottleneck-localization.md), "Separating the four classes of change" |
| Implementation means (how it is written) | *In what authoring interface* is that edit expressed? | standard Triton, TLE, Gluon, intrinsic or inline assembly | this page |

The relations are many-to-many and are not derivable in either direction:

- A `kernel` change class can be written in standard Triton, TLE, Gluon or inline
  assembly. The class does not choose the means.
- A means does not choose the layer. Writing the same kernel in Gluon does not
  move the bottleneck from `operator_dataflow` to `kernel_implementation`.
- `measurement` is not a means at all. Changing the timer changes no interface.
- One candidate may carry two change classes (source rewrite plus an autotuner
  selection), but it is still written in exactly one means per kernel body.

Record the three separately. A screen entry names a layer; `change_class.kind`
names the class; the means belongs in `docs/plan.md` next to the candidate,
because the ledger has no field for it.

## The means available on this backend

Four are named because they are the plausible ways to write a kernel here. Naming
one is not evidence that it is installed, documented or usable on this device; the
gates below are what decide that. This is not a ranking, and the order is not a
preference.

| Means | What it is | When it is the right answer | What KDA actually supports today |
|---|---|---|---|
| Standard Triton | The operator's own `@triton.jit` kernel, tiles, masks, `tl.dot`, `tl.load`/`tl.store` | The bottleneck is expressible with the documented Triton operations, which is most `kernel` and `operator_dataflow` work | The default. Backend-neutral loop, correctness and benchmark wrappers, artifact tooling |
| TLE | The vendor's Triton extension layer, with its own documented primitives | A measured bottleneck needs an operation that the available Triton path cannot express well, and the installed TLE version documents a primitive for it | A task policy (`analysis.extensions.tle`), a schema, and `kernelpilot.py assess-tle`. Optional and disabled by default |
| Gluon | Triton's lower-level authoring layer, where the kernel states layouts and memory explicitly | The cost is in a layout conversion or placement decision that the Triton-level expression cannot control, and the installed toolchain provides Gluon for this target | **Nothing automated.** No probe, no schema, no routed skill, no contract field |
| Intrinsic or inline assembly | A backend intrinsic or an inline assembly block for one operation | The operation must map to a specific instruction and no higher-level interface reaches it | **Nothing automated.** Same status as Gluon |

For an operator whose implementation is a framework call rather than a Triton
kernel, the default means is that framework's documented API, and the remaining
rows apply unchanged.

The last two rows are the important ones. KDA does not detect, probe, version-check
or route Gluon or inline assembly. `kernelpilot.py discover-tools` probes only
what a backend profile declares, and no profile here declares them; `backends/`
has no Enflame profile at all. A prompt that mentions them is naming a possibility,
not reporting a capability. If a candidate uses either, the record must say the
interface was established by hand, by the steps below, in that task's environment.

**Not every means must be tried.** A run that uses standard Triton for every
candidate is a normal, complete run. Trying an optional means is justified by a
measured bottleneck that the current means cannot reach, never by the means being
newer. There is no fixed priority order to climb; the table is a lookup, not a
ladder.

## The five capability gates

A means is usable for a candidate only after all five gates pass, in this order.
Each gate is cheap enough to fail early, and failing an early gate is a recorded
outcome rather than a reason to keep pushing.

| Gate | What it establishes | How it is established | What it does not establish |
|---|---|---|---|
| G1 Documented semantics | The interface exists, with a documented signature, semantics, dtype and layout rules, for the version installed here | The project's own Wiki or the vendor's documentation for the installed version, read in this environment | That the documentation matches the installed build |
| G2 Local interface available | The symbol or module is importable and callable in the contracted environment | A read-only import or signature probe, recorded with its version and its failure output | That it compiles for the target |
| G3 Target backend compiles | The construct lowers for this device and architecture | A minimal unit compiled through the real toolchain, with the compile log kept | That it runs, or that it is fast |
| G4 Runs correctly | The compiled unit produces correct results on the device | Correctness against an independent reference, on the contracted card, within the contracted tolerance | That it helps the hotspot |
| G5 Real workload gains | Using it in the measured hotspot improves the target metric | Same-condition A/B against the current best, as described in [enflame-s60-runbook.md](enflame-s60-runbook.md) §4 | That the gain generalizes beyond the measured shapes |

Two habits:

- **Do not borrow a later gate's conclusion from an earlier gate.** "It compiled"
  is G3. "It is faster" is G5. The S60 material contains both directions of this
  error: an emitted intrinsic read as unit utilization, and a successful probe read
  as an available optimisation.
- **A failed gate is the result.** Record which gate failed and its output. G2 or
  G3 failing ends the line for that means; G4 failing ends the candidate, not the
  means; G5 failing ends the candidate and is a normal benchmark outcome.

For TLE the gates map onto existing machinery: G1 is the Wiki read, G2 is the
task-declared API probe, and G3–G5 are the first candidate's compile, correctness
and benchmark. `assess-tle` covers G1 and G2 only, and its `status: available`
means exactly that.

For Gluon and inline assembly, all five gates are manual work in the task
environment. Nothing in this repository performs them, and no evidence file is
produced for them. If a task needs them, say so in `docs/plan.md` with the gate
outputs pasted in, and expect the reader to treat the claim as self-reported.

## What every candidate must state before it is implemented

Written into `docs/plan.md` (or the plan section of `docs/draft.md` when the
candidate is still being proposed), next to the candidate id. Eight items, no
exceptions. A candidate that cannot fill them is not ready to implement.

| # | Item | What it must contain | Failure mode it prevents |
|---|---|---|---|
| 1 | Bottleneck and the control it needs | The layer the screen selected, the measured or read evidence behind it, and which variables must be held fixed to see the change | A candidate whose improvement cannot be attributed |
| 2 | Why this means | Why the current means cannot express the change, or why the same change in the current means was already rejected | Reaching for a new interface as a substitute for a hypothesis |
| 3 | Installed version and target | The version, package or commit of the interface, the compiler and SDK identity, and the device and architecture it is aimed at | "It works on the vendor's newer release" |
| 4 | Applicability | dtype, shape class, layout and stride assumptions, and the boundary cases where it is not used | A specialization silently applied to shapes it does not cover |
| 5 | Minimal validation | The G1–G3 results with their commands and outputs, or the gate that failed. G4 and G5 need the candidate, so they are filled in after it exists | An interface assumed to work because it is documented |
| 6 | Expected artifact or execution signature | What should appear or disappear in the generated code or the execution record if the hypothesis holds | A result that "looks plausible" for any hypothesis |
| 7 | Acceptance criterion | The metric, the comparison, the threshold and the shape set that decide keep or reject, written before measuring | Moving the target after seeing the number |
| 8 | Fallback and exit condition | What runs if the meant interface is unavailable at G2, G3 or G4, and the budget or condition that ends the line | A run that cannot continue and cannot stop |

Item 6 is the one most often skipped. It is also the one that makes G5
interpretable: without a predicted artifact or execution signature, a measured
gain cannot be attached to the mechanism that was supposed to produce it.

## The five questions every candidate answers

Short form, for the plan and the final report. The eight items above are the
detail; these five are the summary a reviewer reads first.

1. **What is the bottleneck evidence?** Layer, observation, and the control level
   reached — not the story that fits.
2. **Which layer is being changed, and why that one?** The change class chosen
   within that layer, and why the cheaper classes were ruled out or already
   rejected.
3. **What is expected to decrease, and by roughly how much?** The observable in a
   named stage, with the artifact or execution signature from item 6.
4. **How are semantics, accuracy, boundaries and fallback verified?** The
   correctness type used (consistency, regression or accuracy — they are
   different results, see [bottleneck-localization.md](bottleneck-localization.md)),
   the shape and dtype boundaries, and what happens when the means is unavailable.
5. **How is the comparison made fair, and what result rejects or stops it?** The
   A/B conditions, and both outcomes with the action each one triggers.

A candidate whose answer to question 3 and question 5 could support either
outcome has no decision value. Skip it rather than running it.

## What is actually machine-checked, and what is not

| Item | Checked by | Enforcement |
|---|---|---|
| Six-layer screen and one deep dive | `schemas/bottleneck-screen.schema.json`, `verify_evidence` | Structural only, at `verify` and at the end of `run` |
| TLE policy, Wiki presence, API probe | `schemas/tle-assessment.schema.json`, `kernelpilot.py assess-tle` | Structural; `required` fails when unavailable |
| Change class of the selected direction | `schemas/bottleneck-screen.schema.json` (`change_class`) | Structural only. The candidate ledger has no `change_class` field, so a per-candidate class is prose in `docs/plan.md` |
| The eight items above, the five questions, the five gates, the means | nothing | Agent guidance only. Recorded in `docs/plan.md`; no gate reads it |

That last row is deliberate. A doc that promises a gate it does not have is worse
than a doc that states the limit, and this repository has already been corrected
once for describing an ordering instruction as program-level enforcement.

The workflow has no separate stage for this step. Strategy selection happens in
`select_optimization_strategy` and, when the cause is still unknown,
`select_analysis_strategy`; the assessment outputs belong in `docs/plan.md`,
which every stage already reads.

## What this page does not claim

- No interface described here was validated on an S60 device in this revision.
  Gates G3–G5 are procedure, not results.
- Gluon and inline assembly are not supported, probed or routed. Mentioning them
  in a prompt is not a capability report.
- A passed gate is not a performance claim, and a failed one is not a backend
  limitation — it is one environment's result, at one version, on one card.

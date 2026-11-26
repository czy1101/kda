# Bottleneck localization: screen first, then dig once

Why this document exists: a run that collects every available artifact spends
its budget on evidence nobody reads. A run that jumps from the first number to a
cause produces a confident wrong answer. The method below makes the first step
cheap and the second step deliberate.

Two rules carry most of the value:

1. Screen six layers before choosing any experiment.
2. Then run **one** diagnosis that can change a decision, and stop when it has.

The screen is a set of viewpoints, not six mandatory device experiments. Marking
a layer `not_applicable`, or recording that its evidence is insufficient, is a
valid and common outcome.

Machine-readable form: `runs/bottleneck-screen.json`, validated against
`schemas/bottleneck-screen.schema.json`. Set `analysis.bottleneck_screen:
required` when the task must produce one; the default `auto` validates it only
when present. See [evidence-format.md](evidence-format.md).

## Step 1: the six-layer screen

For each layer record four things: current evidence, whether it is worth
continuing, the single most decision-relevant unknown, and what the next
experiment would let you decide. Do not write a plan into the evidence field.

| Layer id | Question it answers | Cheap evidence to consult first | What it cannot decide |
|---|---|---|---|
| `algorithm_numerics` | Is there algebraically redundant work, or a precision contract that forces a slow path? | Source, shapes, dtypes, accumulation type, tolerance, reference implementation | Where the time goes. A lower operation count is not a faster kernel |
| `operator_dataflow` | Is the cost in this kernel, or in what is materialized, re-read or re-laid-out around it? | Staged timings (full call, preprocessing, hotspot), intermediate tensor shapes and dtypes, workspace buffers | Which instruction is slow |
| `kernel_implementation` | Does the kernel body do the right work with the right partitioning and access pattern? | Kernel source, grid, tile `constexpr`s, masks, per-kernel timing | Whether the compiler generated what the source implies |
| `launch_config_autotune` | Which configuration actually ran, and was it chosen or pinned? | Cache entry metadata, the recorded launch arguments, fixed-config control | Whether that configuration is near optimal for this hardware |
| `compiler_codegen` | Did the generated code deviate from the source's intent? | Stage artifacts and their hashes, bundle targets, disassembly — see [enflame-compiler-diagnostics.md](enflame-compiler-diagnostics.md) | Physical unit utilization |
| `runtime_system` | Is the gap outside the kernel: submission, allocation, synchronization, module load, timing? | Timeline or host-gap measurement, cold-versus-warm comparison, first-call behavior | Which kernel is slow |

Two habits keep this step honest:

- **Do not promote a symptom to a layer's conclusion.** "The hot kernel is 70% of
  the call" is a dataflow observation, not a kernel-implementation finding.
- **Prefer the layer with the cheapest discriminating evidence**, not the layer
  that sounds most advanced. Reading a dump costs a device run; re-reading the
  source and the recorded launch arguments costs minutes.

A worked shape for the record:

| Layer | Evidence | Continue | Key unknown | Decision the next experiment supports |
|---|---|---|---|---|
| `algorithm_numerics` | Same math as reference; FP32 accumulation already used | no | none | Nothing: no algebraic slack left at this level |
| `operator_dataflow` | Hotspot is 62% of the full call; one FP32 workspace tensor of 32 MiB is written then re-read | yes | Does the consumer need FP32, or is it a producer-precision choice? | Whether to attack the workspace or the kernel |
| `kernel_implementation` | Two masked loads feed one elementwise op | no | — | Deferred until dataflow says the kernel matters |
| `launch_config_autotune` | Autotune picked `BK=128`; a fixed-config control was not run | yes | Does the deployed win survive at fixed config? | Whether the gain is semantic or a config switch |
| `compiler_codegen` | No artifacts captured yet | yes | Is the FP32 tile transposed through shared memory? | Whether a source rewrite can remove a real conversion |
| `runtime_system` | Cold 31 s, warm 2.7 s; compile cached | no | none | Nothing: compilation is already outside the timed region |

## Step 2: select the deep dive

Pick the direction the screen points to. Each row states what to collect, what
the result decides, and when to stop. Collecting more than the row names needs a
reason written down.

| Direction | Enter when the screen shows | Collect exactly this | Result decides | Stop when |
|---|---|---|---|---|
| `algorithm` | Operation count or precision contract is suspect | Operation count for this shape class, numeric contract, reference comparison for the same inputs | Whether a rewrite can remove work at all, or the cost is irreducible | The contract is confirmed or a cheaper formulation is identified |
| `dataflow` | Cost sits between kernels: materialized workspace, repeated preprocessing, layout conversion | Per-stage timings on identical inputs, tensor shapes/dtypes/liveness, bytes written and re-read | Which side of the boundary to change | One boundary is shown to carry the cost, or none does |
| `kernel` | One kernel dominates and its body is improvable | Kernel source, grid, tile sizes, access pattern; single-kernel A/B with alternating order | Whether the kernel rewrite is worth a full-call measurement | The kernel delta is measured and the mechanism is stated at its real strength |
| `config` | Deployed result may depend on the autotuner's choice | Deployed config, a fixed-config control, and the config-selection record | Whether the effect is the source change or the config switch | Both numbers exist and are reported separately |
| `compiler` | Source intent and generated code may differ | Stage artifacts, hashes, bundle targets, and only the pass interval the hypothesis names — see [enflame-compiler-diagnostics.md](enflame-compiler-diagnostics.md) | Whether to change the source, the config, or (rarely) propose a compiler rule change | The deviation is located to a stage, or shown not to exist |
| `runtime` | First call is slow, or host gaps, allocation, synchronization or module load are visible | Cold versus warm, `_init_handles`-class one-time work, submission gaps, timer boundary | Whether the target metric even includes the cost | The cost is placed inside or outside the timed region |

`none` is a legitimate selection when the screen shows no actionable direction:
record it and finish with an evidence-backed stop rather than inventing a
candidate.

## The experiment contract

Every experiment states six things before it runs. The schema requires them for
each deep dive; a run that cannot fill them does not need the experiment.

| Field | What it must contain | Failure mode it prevents |
|---|---|---|
| `hypothesis` | One falsifiable statement about the current bottleneck | A direction ("look at shared memory") that no result can contradict |
| `control` | Variables held fixed, and the comparison that isolates the changed one | Comparing a warm candidate against a cold baseline |
| `expected_change` | The observable that should move, in which stage, and roughly by how much | A result that "looks plausible" for any hypothesis |
| `supports_if` / `refutes_if` | Both acceptance conditions, written before measuring | Moving the goal after seeing the number |
| `next_action` | What happens after a supporting result **and** after a refuting one | Continuing a dead line because the result was ambiguous |
| `stop` | Budget and the condition that ends this line | Twelve control groups for one decision |

Two budget rules follow from the existing stop block (`stop.max_candidates`,
`stop.max_profile_runs`, `stop.no_improvement_rounds`):

- If two possible outcomes lead to the same next action, the experiment has no
  decision value. Skip it.
- Once the decision is made, stop collecting. A confirmatory control group is
  justified only when it separates two hypotheses that are still both alive.

## The four-link evidence chain

The chain is `source → compiled artifact → actual execution → performance
change`. Each link proves a different thing, and a claim may not borrow strength
from a link that was never established.

| Link | Established by | Does not establish |
|---|---|---|
| Source | The exact file, revision and `constexpr`/launch values | What the compiler emitted |
| Compiled artifact | Artifact identity for the specialization that ran: kernel name, shapes, dtypes, strides, grid, warps, stages, selected config, compiler/SDK identity, and artifact hashes | That this artifact is the one in the timed region |
| Actual execution | A launch record tied to that same specialization (kernel name, cache entry, arguments), or a control showing the alternative did not launch | That the timed region was dominated by it |
| Performance change | Same-contract A/B on the same device, input, warmup and timed region | Which mechanism inside the kernel caused it |

Breakage signatures, each seen in the S60 material:

- **Cache entry exists, but the timed run used another specialization.** One
  kernel name produced two cache entries that differ only by a specialization
  attribute; the artifacts are not interchangeable.
- **Artifact changed, but the timed code did not.** A cache key changed while
  the device binary stayed byte-identical
  ([S60-TRITON-014](../knowledge/enflame/triton-shared-transpose-materialization.md)).
- **Performance changed, but the named mechanism is aggregate.** A 14.75% kernel
  gain attributed to "fewer transposes" is a sum over transposes, waits,
  barriers and a smaller footprint that were never ablated.
- **Numerics pass, but at the wrong scope.** "Output is finite" and "matches an
  independent float64 reference" are different results
  ([S60-EVIDENCE-001](../knowledge/enflame/evidence-and-measurement.md)).

## Separating the four classes of change

Most attribution errors on this backend come from collapsing these into "the
change". Record the class in `change_class.kind`.

| Class | What actually changed | Typical evidence | How to keep it separate |
|---|---|---|---|
| `kernel` | Source code of the kernel body or its expression structure | Same config, same inputs, source-only diff | Fix the config; a config switch is a different class |
| `config` | Tile sizes, warps, stages, autotuner selection, launcher options | Deployed config record plus a fixed-config control | Compare against the same source at the same pinned config |
| `compiler` | Compiler version, flags, passes, rules | Compiler identity plus artifact-level diff at fixed source and config | Needs an independent build; see the gate below |
| `runtime` | Submission, allocation, synchronization, module load, host launcher, timing boundary | Timeline or host-gap evidence at fixed kernel and config | Keep the timed region identical, or report that it moved |

`measurement` exists for the case where only the measurement changed — timer
type, repeats, warmup, synchronization. That is a different claim from a
performance change and must not be reported as one.

The practical test: if the source and the config and the compiler are all fixed
and the number still moves, the change is in runtime or in measurement. Say
which, and if you cannot, say that you cannot.

## Attribution strength

State the level reached. Do not claim above it.

| Level | What is shown | The strongest honest sentence |
|---|---|---|
| L0 | Correlation in one configuration | "The change is associated with a lower median in this configuration" |
| L1 | Single-variable A/B, alternating order, repeats, raw samples | "With this variable changed and the rest held fixed, the median moved by X" |
| L2 | L1 plus a fixed-config control separating the semantic change from a config switch | "Of the X, Y comes from the source change and Z from the configuration switch" |
| L3 | L2 plus an artifact-level observation that matches the predicted mechanism | "The named stage product changed in the predicted direction" |
| L4 | A resource or counter observation inside its documented scope | "The measured unit activity changed within the scope of that counter" |

L4 is unavailable in many S60 runs. That is a capability limit, not a reason to
promote an L1 result to a mechanism claim.

Inferences this repository has already made and corrected — do not repeat them:

- Sum of allocation sizes reported as a peak footprint.
- A generated intrinsic read as unit utilization.
- A Triton cache-key change read as a device-code change.
- Differing bytes read as differing machine instructions.
- A registered pass read as an executed pass; an executed pass read as a change
  that produced different output.
- An absent token in an IR text read as an absent operation. Counting
  `triton_gcu.matmul` border markers as missing matrix work is the canonical
  failure.
- One component's cumulative delta read as its isolated contribution; a combined
  candidate compared against its own intermediate instead of the frozen baseline.
- A total gain assigned to the one mechanism that is easiest to describe.

## Verification levels

Keep these distinct in the plan, the report and the commit message. Never write
the first three as the fourth.

| Level | Meaning | What it does not prove |
|---|---|---|
| Static | Documents, schemas, prompts and cross-references are consistent | Anything about a real run |
| Offline tool | A deterministic tool passes its tests on fixtures | Real compiler or device behavior |
| Dry-run / fake runner | The workflow and gates execute end to end without a device | That the device path works |
| Device | Correctness and timing on the contracted hardware | Generality beyond the measured shapes |

A fixture is test data. It is never S60 compilation evidence.

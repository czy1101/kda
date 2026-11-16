# Enflame S60 operator optimization runbook

This runbook is the task-side operating procedure for Kernel Design Agents on
the local Enflame S60/GCU300 host. It complements
[knowledge/enflame/README.md](../knowledge/enflame/README.md) and the generic
[agent flow](agent-flow.md). It is not a performance result and does not claim
that any particular operator is faster.

## 1. Fix the task before changing code

Record all of the following in `.kernelpilot/task.yaml` and `docs/plan.md`:

- Operator semantics and exact reference implementation.
- Shape class, including batch/head/layout and boundary cases.
- Input/output dtype, accumulation dtype, and numerical tolerance.
- Layout and strides, including non-contiguous or packed variants.
- Physical device: only card 6 or card 7, and the actual `TOPS_VISIBLE_DEVICES`
  value used by every process.
- Correctness command and threshold.
- Baseline identity and baseline command.
- Measurement scope: full operator, preprocessing, and hotspot kernel.
- Warmup, repeat count, synchronization policy, and timer.
- Promotion target, regression limit, and rollback condition.

Do not begin implementation while any of those fields are placeholders.

## 2. Confirm the device is allowed and free

At the start of a session:

```bash
efsmi
docker ps --filter name=hjq-flagattention
```

Use only physical card 6 or 7, and only when `DUsed` is 0 and no known task is
using it. Preserve other users' containers, processes, worktrees, untracked
files, and experiment directories. Do not kill or restart another process to
free a card.

Inside the S60 container the device list is controlled by
`TOPS_VISIBLE_DEVICES`. The value is the physical card number, not an index
inside an already-filtered list. A quick visibility check is:

```bash
docker exec -e TOPS_VISIBLE_DEVICES=6 -w /workspace/hjq \
  hjq-flagattention bash -lc \
  'python -c "import torch; print(torch.gcu.device_count())"'
```

Expect `1` for a single-card task. Set `TOPS_VISIBLE_DEVICES` on every process
that can start device work. For non-interactive `docker exec`, repeat both the
environment variable and the working directory every time:

```bash
docker exec -e TOPS_VISIBLE_DEVICES=6 \
  -w /workspace/hjq/FlagAttention-<operator> \
  hjq-flagattention bash -lc '<command>'
```

Do not rely on a previous interactive shell's exports, and do not run a
multi-card sweep on an unconfirmed card.

## 3. Measure in stages

For each candidate, measure and save:

1. **Full operator** under the task contract, including host-side launch and
   synchronization policy.
2. **Preprocessing** if the operator reshapes, pads, transposes, sorts, or
   materializes workspace.
3. **Hotspot kernel** under the same input and warmup conditions.

Only then state one falsifiable hypothesis. It should name the expected change
in a measured stage, for example "the hotspot's output-merge phase decreases
while full-operator latency decreases by at least the measured share of that
phase". A hypothesis that cannot be contradicted by the next measurement is not
ready.

Keep the evidence classes separate:

| Evidence | What it supports | What it does not support |
|---|---|---|
| Python/Triton source | The intended algorithm and data flow | Generated hardware behavior |
| Compiler IR / LLVM | A lowering or optimization path | Physical unit utilization |
| Intrinsic or generated call | The compiler emitted a construct | The unit was busy or the operator is faster |
| Final instruction / ISA | The selected instruction sequence | Runtime bottleneck by itself |
| Hardware counter or monitor | The measured unit activity, within its scope | Causal explanation without a matched experiment |

The Enflame knowledge base deliberately distinguishes these levels. Do not turn
"an intrinsic was generated" into "the matrix or vector unit utilization
improved". If counters are unavailable, report that limitation instead of
inventing utilization.

If the task explicitly uses the read-only `efsmi dmon` monitor, revalidate its
installed help, device mapping, metric indices, and sampling interval first.
The historical S60 card [S60-EVIDENCE-002](../knowledge/enflame/efsmi-gpm-monitor.md)
observed SIP utilization/occupancy and DRAM read/write fields; DRAM stayed zero
under active loads, so zero means "unavailable", not "no traffic". SIP busy does
not measure matrix-pipeline, DTE, stall, or overlap behavior. Device-level
samples are not a per-kernel profiler and do not replace matched A/B timing.

## 4. Correctness and A/B measurement

For every candidate:

1. Run correctness first. A failing candidate is recorded, not benchmarked as a
   winner.
2. Compare against the selected baseline on the same physical card, same input,
   same layout, same warmup, and same timed region.
3. Alternate baseline and candidate runs (A/B) rather than measuring all
   baseline runs first and all candidate runs later.
4. Save raw samples, not only an aggregate. Record median/mean, spread,
   compilation configuration, whether compilation was included, and failures.
5. Re-run the full required shape set before promotion. If a full run is too
   expensive, state the risk and the subset used; do not silently substitute a
   favorable shape.
6. Roll back the candidate if correctness or the regression limit fails. The
   best correct checkpoint is the implementation that remains deployed.

Do not claim a speedup from one noisy sample. Do not compare a warm candidate
against a cold baseline, or a candidate measured with one device/timer against a
baseline measured with another.

## 5. TLE

TLE is disabled by default. If the task enables it:

1. Read the project's `tle-wiki` for the installed version; do not infer an API
   from another version or from a conda environment name.
2. Run the task-declared API probe in the contracted environment.
3. Compile a minimal TLE example and validate its correctness before using it in
   the hotspot.
4. Record API, compilation, lowering, runtime, correctness, and performance
   errors separately.
5. Use TLE only if a measured bottleneck needs an operation that the available
   Triton path cannot express well. A missing counter is not evidence of a TLE
   gap, and a successful intrinsic probe is not evidence of a speedup.

If no documented primitive expresses the required semantics, report the concrete
missing primitive and its required behavior instead of inventing an interface.

## 6. Knowledge base updates

After a run, update KDA knowledge only when the result is reusable across
operators: a mechanism, interface, applicability condition, or evidence-backed
backend limitation. Use
[knowledge/enflame/CARD_TEMPLATE.md](../knowledge/enflame/CARD_TEMPLATE.md),
update `catalog.json` and `sources.json`, and keep raw logs, operator-specific
winning constants, shape tables, absolute timings, candidate ledgers, and run
narratives in the FlagAttention worktree.

A task result alone is not a knowledge card. Preserve contrary evidence and
narrow an earlier card rather than rewriting it as a universal rule. If the KDA
repository is not writable, put the proposed update in the task's `docs/` and
report the pending destination; do not modify frozen tests, benchmarks,
contracts, or remote allowlists to publish knowledge.

## 7. Task parameters for a new S60 operator

A minimal contract starts like this (replace every placeholder):

```yaml
version: 1
task:
  name: <operator-name>
  description: <one-line goal>

target:
  backend: gcu
  device: S60
  arch: gcu300
  language: triton          # or torch, depending on the implementation

execution:
  transport: local          # use ssh when OpenCode runs on a controller
  # host: <ssh-config-host-for-s60>
  # workspace: /data/hjq/FlagAttention-<operator>

environment:
  shell: bash
  setup: |
    export TOPS_VISIBLE_DEVICES=6
  preflight:
    command: python -c "import torch; print(torch.gcu.device_count())"
    expect_stdout_contains: "1"

implementation:
  path: src/flag_attn/<operator>.py
reference:
  path: <reference path>

correctness:
  command: docker exec -e TOPS_VISIBLE_DEVICES=6 -w /workspace/hjq/FlagAttention-<operator> hjq-flagattention bash -lc '<correctness command>'
benchmark:
  command: docker exec -e TOPS_VISIBLE_DEVICES=6 -w /workspace/hjq/FlagAttention-<operator> hjq-flagattention bash -lc '<benchmark command>'
  warmup: 20
  repeat: 100
  shapes:
    - <required shape>

baseline:
  name: original

goal:
  metric: latency
  direction: minimize
  aggregation: mean_ratio
  relative_to_baseline: 0.95
  max_regression: 0.02

constraints:
  allowed_paths:
    - src/flag_attn/<operator>.py
  forbidden_paths:
    - tests
    - benchmark
    - <reference path>

stop:
  max_candidates: 20
  max_profile_runs: 10
  no_improvement_rounds: 5
```

When OpenCode runs on the host and commands enter the container, the
`docker exec` lines must set `-e TOPS_VISIBLE_DEVICES=<6-or-7>` and `-w` every
time. When the task runs inside the container (`transport: local` from the
container's perspective), set `TOPS_VISIBLE_DEVICES` in `environment.setup` and
keep the command in the container's working directory. Either way, the task
contract is the authority; the example is not a performance claim or a ready
solution for a specific operator.

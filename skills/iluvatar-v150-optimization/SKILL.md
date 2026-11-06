---
name: iluvatar-v150-optimization
description: Optimize operators on Iluvatar BI-V150 / ivcore11 by selecting native compute, matrix staging, memory, Warp64, and resource mechanisms, then verifying their trigger conditions in compiler IR, assembly, and relevant ixKN counters. Use before choosing V150 optimization candidates and when continuing optimization after migration.
---

# Iluvatar V150 hardware-aware optimization

Read the task contract and selected backend profile. Before choosing the first
candidate, assess how the operator can use this chip's efficient execution and
data paths. Repeat that assessment when the bottleneck changes. Correctness and
the reproducible baseline still precede candidate promotion.

## Start with the hardware opportunity

Map the operator's expensive operations and data movement to these choices:

| Work | Assess first |
|---|---|
| Matrix products and quantized matrix products | TCU through `tl.dot`; operand dtype, accumulator precision, instruction tile, and layout |
| Matrix operand preparation | SME global-to-shared staging; host tensor rank, real stride, alignment, dot consumer, and completion waits |
| Elementwise affine/quantization/activation work | Packed FP16/BF16 arithmetic and explicit `tl.fma`; conversion and packing costs |
| Small integer dot products or packed metadata | C++ `__dp4a`, byte permutation, bit operations; only within permitted implementation languages |
| Softmax and normalization | Actual SFU paths and numerical requirements; distinguish native instructions from expanded functions |
| Repeated or irregular memory access | Coalescing, true alignment, cache working set, reuse, uniform address expressions, and fusion |
| Reductions and inter-thread reuse | Warp64 shuffle/vote, shared layout, bank conflicts, synchronization, and atomic aggregation |
| Low utilization or resource pressure | VRF/SRF, spills, shared budget, stages, block residency, grid size, masked lanes, and tail waves |
| Host gaps and offload | Launch/Graph overhead, pinned transfers, lifetime, dependencies, and measured overlap |

Do not force a matrix or specialized path onto unsuitable work. State why a
promising mechanism applies or why it is ruled out; do not exhaustively test
unrelated features. Use measured bottlenecks to choose among viable mechanisms.

Read [native-paths.md](references/native-paths.md) for trigger conditions and
known failures relevant to the operation. The maintained detailed sources are
[the hardware reference](../Iluvatar/天数V150_硬件特性与计算搬运单元.md) and
[the knowledge base](../Iluvatar/天数V150_知识库.md); search the cited topic instead
of loading all historical operator experiments.

## Make the trigger a falsifiable hypothesis

Before implementing a hardware-oriented candidate, put the following in the
task's existing plan or candidate record:

- The expensive operation and proposed hardware/data path.
- The exact source/API change and required dtype, rank, stride, alignment,
  mask, consumer, launch options, and software-version conditions.
- Expected TTGIR/LLIR and machine instructions, removed instructions, resource
  changes, or dynamic counters. Specify what observation would refute the idea.
- Numerical, synchronization, residency, and preparation costs, including any
  preprocessing that the task requires timing.

For example, `stride=` on a matrix load is a hypothesis about SME selection;
it is not proof of selection or of a speedup. BF16 `a*b+c` is not proof of a
packed FMA. More stages are not proof of overlapping transfers.

## Verify selection, execution, and benefit separately

1. Preserve attributable compiler artifacts for the exact launched
   specialization: source identity, dtype/shape/layout, launch options, compiler
   versions, TTIR/TTGIR/LLIR, native assembly, and resource metadata as available.
   Use the launched kernel's `asm` first; cached IR must be linked to that
   specialization, not selected by its timestamp or kernel name alone.
2. Inspect the actual instruction sequence and its dependencies. Check loads,
   conversions, shared operations, waits, and spills around the desired
   instruction, not just a matching mnemonic. Metadata and SDK declarations
   describe candidates or interfaces, not completed hardware execution.
3. Run the contracted correctness checks, including affected tails, layouts,
   masks, integer signs, and precision boundaries. Exclude incorrect paths from
   performance comparisons.
4. If static artifacts cannot answer a dynamic question, use the discovered
   ixKN capability with [iluvatar-ixkn-analysis](../iluvatar-ixkn-analysis/SKILL.md).
   Use instruction counters for execution mix, memory counters for traffic and
   conflicts, and occupancy/stall data for resource or scheduling hypotheses.
   Counter collection is conditional on the question and task policy.
5. Benchmark correct candidates without profiler interference over the full
   required workload. Record trigger success and performance success separately;
   a specialized instruction can execute correctly yet lose overall.

If a path is uncertain, use a bounded minimal probe in the task workspace and
the contracted execution environment, through KDA's controlled transport. Isolate
selection variables and preserve failures. Do not invent an API or support claim.
No available counter or dump means an evidence limit, not a reason to claim the
path executed. Keep the best correct checkpoint after rejected candidates.

## Scope and reporting

This skill does not enable C++, TLE, extra tooling, or system changes by itself.
Respect allowed languages/paths and the task's optional-extension policy. TLE
requires its routed skill and project Wiki/API assessment even when historical
V150 notes show a working primitive.

Record evidence in existing task-local `runs/`, `outputs/`, or `profile/` paths,
not beside reusable skills. For each hardware claim distinguish: proposed,
generated in artifacts, observed dynamically, correctness-verified, and measured
benefit. Explain rejected paths and remaining uncertainty without claiming
unmeasured throughput or overlap.

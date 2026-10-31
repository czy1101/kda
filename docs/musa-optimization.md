# Evidence-driven Triton optimization on MUSA

Use the `musa-mp31` backend profile for MTT S5000 tasks. Keep task-specific
kernels, reproduction inputs, compiler dumps and timings in the operator
workspace. This guide records portable investigation methods, not a claim that
a tile size, compiler workaround or instruction is correct on every release.

## Establish what the machine actually supports

Record the device properties, driver, PyTorch/torch_musa and Triton versions in
the task environment. Discover tools there; a GPU name or importable profiler
does not prove that device events can be captured. The MUSA profile probes the
framework's supported activities and never selects NVIDIA profilers implicitly.
If no device profiler is available, time isolated stages with synchronized MUSA
events and retain end-to-end measurements. Isolated stage totals need not equal
end-to-end latency because launch overhead and cache state differ.

Probe documented optional APIs by executing the documented availability check.
An exported Python function can still raise `NotImplementedError`. Do not turn
that stub into a fabricated primitive. TLE requires the task-local Wiki and API
assessment described in `skills/tle-optimization/SKILL.md`; compiler support for
SQMMA alone does not establish the TLE API contract.

## Inspect the generated implementation

For the hot specialization, save source identity, launch parameters, compiler
metadata, Triton GPU IR and LLVM IR. Distinguish scalar/ordinary dot lowering
from SQMMA lowering using metadata and IR. Hardware capability is not evidence
that a particular kernel uses its matrix units. Cross-check metadata flags
against instruction-bearing IR: a `uses_sqmma: false` flag can disagree with
`llvm.musa.sqmma.*` calls in the final LLVM IR. Preserve both artifacts and
report the contradiction; do not conclude that matrix units are absent from a
single metadata field. Backend optimization options may depend on that field,
so inspect the effective compiler command as well as the intended option. Do not transplant SM90 warp,
shared-memory, PTX or SASS assumptions.

Larger tiles may reduce launches yet worsen register pressure, spills, layout
conversions or library-call costs. Compare them on identical inputs with three
or more warmed measurements; never promote a tile based only on nominal matrix
throughput. Change one coherent dataflow or lowering decision per candidate.

## Separate numerical bugs from performance problems

When a kernel compiles but fails, compare forward and each gradient separately
against an independent high-precision reference, on CPU if necessary. Preserve
adversarial boundaries, masks, noncontiguous inputs, partial tiles and low-
precision types. Inspect spatial error patterns: periodic wrong rows can reveal
a layout/conversion or compiler issue that a global mean obscures.

For an unsupported vector math instruction, compare the generated IR before
and after a documented math-library alternative. Preserve logarithm bases,
scales, infinity/mask semantics and tolerances. A fallback that compiles is not
yet correct. If a low-precision product miscomputes, an IEEE FP32 product can
establish a correct checkpoint; explicitly record its performance limitation.
Keep such workarounds backend-local and version-evidenced, never universal.

A bounded-domain elementary-function replacement is a separate algorithmic
candidate. First state its range reduction, truncation bound, floating-point
rounding behavior, NaN/infinity semantics and underflow policy. Sweep the domain
against an independent higher-precision reference, including representational
boundaries. Then run the complete operator and gradient checks at unchanged
tolerances. A softmax-specific underflow argument does not justify reusing the
same approximation in an arbitrary recurrent gate or state update. Keep the
mathematical validation script with the operator evidence, not in this guide.

A minimal reproducer should include the dtype, dimensions, strides, masks,
launch parameters and expected errors for each output. Save compiler failures
and wrong-result cases alongside performance regressions in the candidate log.

## Optimize the algorithm around measured traffic

Account for intermediates by tensor size and lifetime. For sparse operators,
include routing indices, sorting/prefix scans, materialized payloads and result
merges as well as matrix arithmetic. Prefer eliminating a whole intermediate
or synchronization stage over tuning a small launch. Preserve native grouped
heads rather than expanding K/V when the algorithm permits it. Fusion must not
introduce nondeterministic floating-point atomics or erase masking contracts.

Check the main bottleneck after each accepted change: reducing arithmetic may
expose routing or merge traffic. A timing-only smoke test does not replace the
correctness gate. Keep the best correct checkpoint when compilation, numerical
validation, stability, or generalization fails. Record a no-improvement result
rather than promoting a noisy gain.

## Read-only artifact cross-check

Run `python scripts/inspect_musa_artifacts.py <kernel.json> <kernel.llir>`
on artifacts saved by the operator task. The report separates the metadata flag
from executable SQMMA intrinsic calls and highlights a contradictory false flag.
Unused LLVM declarations are not counted. This is a diagnosis aid, not a
substitute for final machine-code inspection or measured performance.

## Validate pipeline changes and cache isolation

Treat pipeline depth as a per-kernel correctness decision, including separate
forward and backward checks. A setting that passes forward can still produce
wrong gradients. If restricting the setting repairs correctness, benchmark that
restricted candidate independently; overlap is not evidence of a speedup.

Batching independent reductions can increase live state and layout-conversion
costs. Removing launches or intermediate writes is a hypothesis, not proof of
end-to-end benefit. Retain paired, alternating-order measurements and reject
small gains when the task explicitly requires substantial improvements.

When an unchanged kernel behaves inconsistently, preserve the failing source,
device selection, compiler versions and cache identity. Rerun with a separate
task-local compiler cache; do not delete a shared cache. Change one variable at
a time to attribute the cause. If both device and cache change, report the
confounding factors and do not label the cache defective. Re-establish baseline
and candidate in the same environment before comparing timings.

## Audit native sparse-library contracts before substitution

Distinguish an official primitive from an official implementation of the full
algorithm. Inspect the installed version as well as public source; a newer
wrapper does not prove that the current environment provides the operation.
Follow the wrapper through normalization into the low-level dispatch. Check
head ratios, route indexing granularity, fixed slot counts, page dimensions,
LSE layout, masking, and gradient registration. Wrapper signatures may accept
a broader layout than the kernel actually supports.

For grouped-query sparse attention, verify whether routes are independent per
query head or shared per KV group. Slicing one route out of each head group
changes algorithms that route per query head. A small CPU counterexample can
prove this incompatibility before an expensive GPU port. Score definitions
are equally important: max-score routing and mean-key routing are not
interchangeable. Keep the router fixed when testing a new attention primitive.

Test noncontiguous packed KV explicitly. A contiguous copy can diagnose a
layout mismatch, but its copy cost belongs in end-to-end timing. Continue using
the original correctness tolerances even if only a few low-precision elements
fail. Preserve rejected adapters and do not report an unvalidated primitive
as an accepted baseline. Set the library's documented workspace/cache root
to the authorized task directory before importing JIT-enabled code.

## Correctness-gate autotune candidates

A full operator test may fail only when timing selects a particular compiled
configuration. Isolate each candidate on an input with an exact invariant,
such as an output merge with no valid historical routes: it must reproduce
the local output. Sweep dtype, reduction width and dimensions. Exclude
wrong-result configurations before timing; do not treat a fresh cache or a
different winning configuration as a correctness fix. Retain the reproducer
and a regression check for the pruning policy.

When adopting native matrix collectives, keep routing semantics independent
from the matrix tiling. Device descriptors can schedule one query-head/block
expert without sharing head-specific choices or synchronizing counts to the
host. Include packing and merge costs in the end-to-end metric. If inference
and training forward use different lowering paths, validate their original
strict agreement and gradients; sharing the validated native collective may
resolve rounding differences without relaxing tolerances. Version-gate private
collective APIs and preserve a correct fallback for unsupported configurations.


## Prune causally impossible work before retuning tiles

For a causal router, bound the candidate scan by what the last valid query
in a query tile can see. Skip candidate tiles that are invisible to every
query, while preserving the score arithmetic, per-head decisions, and tie
ordering inside the remaining tiles. Handle empty query tiles, sequences with
no historical block, and partial final blocks explicitly. Derive bounds from
the actual metadata contract rather than assuming all packed sequences have
the maximum length.

Check the pruned and unpruned route indices exactly, including tied scores,
packed lengths, grouped heads, and non-power-of-two block sizes. When the
attention arithmetic is unchanged, test bitwise output equivalence as well
as the independent reference checks. Measure the eliminated stage and the
full operator separately: an attention-dominated workload may barely benefit
from faster routing. Preserve that limitation in the performance report.

An exported profiler JSON is not sufficient evidence of GPU kernel timing.
Confirm actual device execution events; host launch API records are not
kernel durations. If capture lacks those events, use synchronized device
events for stage timing and record that limitation. Check other GPU processes
while the task is active: an idle keepalive may automatically yield to real
work, whereas unrelated active workloads invalidate controlled comparisons.

Set the task device explicitly before importing the accelerator runtime; do not
inherit an unrelated container visibility setting. Verify the benchmark PID
on the authorized physical device, and record both visibility and physical
device identity. Reject and retain separately any run that used the wrong
device, even if its latency looks plausible.


## Validate compiler proposals against final lowering

Do not infer missing matrix instructions from a summary metadata boolean when
the lowered code contains the instructions. Distinguish an asynchronous
global-to-shared copy from a tensor-memory descriptor path. A larger software
pipeline can double shared-memory use and regress the full operator even when
register counts decrease. Record negative configurations as regression cases.

Trace apparent redundant layout conversions through LLVM and, when available,
machine code before claiming that they survive optimization. A duplicated
loop-carried representation in an intermediate dialect may already collapse
to one accumulator later. Likewise, compiler-reported spills are evidence for
investigation, not proof of the entire slowdown's cause. Match dtype, shape,
and compilation hash when comparing resources; JIT caches can accumulate
specializations across repeated module loads.


## Distinguish output legalization from accumulator spills

When a compiler reports private memory for a low-precision kernel, locate the
actual accesses in target assembly before attributing them to matrix register
pressure. A vector output-store lowering can introduce a private-memory
store/reload round trip even when the matrix loop has no such access. Compare
otherwise matched dtypes, use debug locations, and retain assembly plus the
exact code-generation flags. If the installed disassembler cannot decode the
binary, assembly emitted by the same compiler is useful evidence; label it as
re-emitted assembly rather than a decoded binary.

A possible pure-DSL workaround is to apply the original rounding first and
then pack adjacent low-precision bit patterns into a wider integer store.
Require contiguous aligned output rows, explicit handling of tails and odd
widths, preserved element order and endianness, and an unchanged fallback.
Verify exact output bits, including guard dimensions, and measure the full
operator. Reduced private-memory size alone is not a performance result.

## Verify host descriptors before claiming TME

A compiler capability check is only a starting point. On a given MP31 Triton build,
`tl.make_tensor_descriptor` inside a kernel can fail legalization even when the
host-side `triton.tools.tensor_descriptor.TensorDescriptor` path compiles. Test
both paths with a small copy kernel, compare the copied values exactly, and
inspect final LLVM IR for `llvm.musa.tme.ld.tile.*`. A descriptor in source or
TTGIR alone does not prove a hardware TME load was emitted.

For packed token/head/dimension tensors, a rank-three host descriptor can
load a regular K/V tile while irregular query gathers and per-head routing
stay unchanged. Check contiguous layout, element/stride alignment, tail
padding and masks. Keep the original pointer-load path for unsupported
strides, devices or toolchains.

Host descriptor creation and encoding are part of operator latency. Benchmark
both reused input pointers and fresh K/V allocations with compilation and
input generation excluded from timing. A transfer mechanism that helps long
loops may lose on short calls. If a larger pipeline stage count is tried,
record shared-memory allocation and end-to-end latency; TME plus additional
stages can raise shared use substantially and lose despite correct results.

## Test direct TME operands with a minimal dot

On the tested MP31 Triton/MUSA stack, a rank-three
`[64, 1, 128]` V descriptor followed by reshape to `[64, 128]`
and a direct `tl.dot(A, V)` fails in
`TritonMUSAGPUOptimizeDotOperands`. Adding `tl.where` makes that
kernel compile, but its TTGIR contains TME -> ordinary shared ->
`local_load` -> `select` -> `local_alloc` -> SQMMA: the V operand is
copied through a thread tensor into a second shared layout.

A rank-two descriptor `[64, 128]` with no `tl.where` compiles and
matches a FP32 reference within 3.81e-6. Its TTGIR shows TME writing
directly to the SQMMA B shared operand, with no V `local_load` or
second `local_alloc`. The final LLVM IR contains
`llvm.musa.tme.ld.tile.2d` and SQMMA calls. Therefore the compiler
can express the direct layout for this rank-two case; the failure is
specific to the tested rank-three load/reshape path.

For packed contiguous `[token, head, dim]` V data, a zero-copy view
as `[token, head*dim]` lets a rank-two descriptor select head `h`
at column `h*dim`. Use this only for aligned, full tiles whose
bounds mask is truly redundant; keep the original masked path for
other dimensions or partial tiles. In a MoBA D128, 16x64 route, this
single-variable change made original and candidate outputs bitwise
equal on 15 representative cases and reduced end-to-end time by
about 6-13% on several D128 cases. In long chunk-1024 GQA, the
routed-stage MUSA Event fell from 123.26 to 109.85 ms while full
latency fell from 180.78 to 167.34 ms. D64/D256 remained on the
original path. This does not account for the remaining gap to MATE.

A useful compiler extension would make the direct SQMMA-compatible
TME destination layout available for rank-three descriptors and
reshape, with explicit swizzle, alignment, barrier ownership and tail
zero filling. A failed mapping should produce a specific diagnostic.
Check generated TTGIR/LLVM IR and numeric output before asserting
that a descriptor removed a layout conversion.

## Check the whole profiler software stack

Moore Perf Compute 1.3.1 requires MUSA SDK and Linux Driver 5.2.0;
Moore Perf Compute 1.2.1 requires the 5.1.0 stack. One tested container
reported MUSA toolkits 4.3.5, MUPTI 1.0.5, `mcc` 5.1.0 and driver
3.3.5-server. The newer `mcu` created SQLite reports with target kernel
records and raw counter blobs, but its official viewer displayed empty
Duration and memory-counter values, even with an explicit device filter.
A generated `.mcu-rep` is therefore not proof that the requested hardware
metrics are readable. Record the toolkit, MUPTI, compiler, driver and MCU
versions separately, confirm nonempty rendered metric values, and fall back
to synchronized device events plus IR when they are absent.

## Choose TME or LSU per operand

A successful TME lowering is not evidence that every matrix operand should use TME. In a controlled MP31 MoBA experiment, holding routing and tile shape fixed, pointer-loaded K with TME-loaded V improved D128 end-to-end latency, while the same choice regressed D256 severely. Inspect each operand's final IR path and benchmark it separately. A useful ablation matrix is K/V = TME/TME, LSU/TME, TME/LSU, LSU/LSU, with identical semantic inputs and correctness checks.

When a reference implementation packs data before a native attention kernel, time Q packing, KV packing, and the native kernel individually. Do not compare only the native kernel with an unpacked Triton operator, and do not reuse route descriptors across implementations with different query tile sizes. Preserve complete operator timing and test all supported shapes before applying a conditional choice. A shared-memory conversion seen in IR establishes an extra data movement, not its fraction of wall time.

## Verify profiler events and API stubs on MUSA

On the tested Torch 2.7.1 / torch_musa 2.7.1 / Triton 3.6.0 MP31 stack, `torch.profiler.supported_activities()` includes `PrivateUse1`, but one exported MoBA trace contained CPU operations and MUSA driver launches without device-kernel duration events. Inspect the trace categories and actual device-event durations before calling the framework profiler a GPU hotspot tool. When absent, retain synchronized MUSA Event measurements and compiler IR, and state that hardware-counter attribution remains unavailable.

The same Triton MUSA build exports `libdevice.fast_expf` in Python, yet calling it from a JIT kernel raises `NotImplementedError`. Probe a minimal compiling kernel before relying on a visible symbol. MATE's vector fast-exp2 implementation is useful as a compiler feature request, but its accuracy, special-value behavior and performance must be validated independently on the target compiler.

Moving a TME V load earlier in source did not create an asynchronous pipeline in the tested MoBA kernel; shared allocation rose and latency regressed. Compare final IR ordering, shared lifetime and end-to-end timing before treating source-order prefetch or a larger tile as a hardware optimization.


## Keep routing descriptor tiles consistent with kernel tiles

A MoBA route scheduler may build one descriptor per query tile while a
separate attention function chooses the SQMMA tile. Both call sites must
share one launch-configuration decision. On the tested stack, changing
only the attention `BLOCK_M` from 16 to 32 while leaving route descriptors
at 16 produced redundant work and a large slowdown, despite outputs
passing coarse reference tolerances. This was a configuration bug, not
GPU-load evidence. Pass the same chunk size, input layout and capability
checks to both sites; inspect descriptor counts and run matched AB
benchmarks after changing a tile.

A 32x64 D128 route became faster only when combined with a rank-two V
descriptor whose TME target was already SQMMA B shared. A rank-two K
descriptor also compiled through `tl.trans(K)` directly into QK SQMMA
B shared. For contiguous packed `[token, head, 128]` K/V, flattening
the last two dimensions is a zero-copy view; the head offset is
`head*128`. Require full 64-token historical tiles and retain the
masked fallback for irregular chunks or strides. In one same-card,
same-input run, full MoBA `long-c1024-k8` fell from 179.97 to 123.74
ms, while the MATE reference was 111.23 ms. In that final variant,
the routed attention MUSA Event was 69.17 ms, compared with 76.33 ms
when only V used the direct two-dimensional path. Static TTGIR/LLVM
IR proves the removed conversion and emitted 2D TME/SQMMA; hardware
counter values were unavailable, so no machine-cycle attribution was
made. Do not assume that larger tiles or TME alone cause the gain.


For a matched long D128/chunk-1024 MoBA workload, the old 16-query
scheduler emitted 462,848 route descriptors and the coordinated
32-query scheduler emitted 233,472, matching the MATE route-CTA
count. This removes the confirmed twofold descriptor/work duplication;
it does not prove equal per-CTA efficiency. Once grid counts match,
compare operand loads, softmax, waits and overlap with trustworthy
device counters or controlled kernel-stage timings.


In a final same-input MUSA Event breakdown for D128/chunk-1024,
current Triton routed attention took 68.97 ms against MATE's
55.47 ms; full calls were 125.52 and 111.95 ms. Local attention,
route construction and merge were near equal. After matching the
233,472 route descriptors and making K/V TME operands direct, the
remaining measurable gap is per-CTA routed-kernel execution. Event
timings do not distinguish wait instructions, Q gathers, softmax,
register pressure or native pipeline overlap; obtain readable
hardware counters or controlled single-variable kernel ablations
before naming an instruction-level cause.


## Apply direct rank-two TME operands across head dimensions

The zero-copy `[token, head*dim]` descriptor technique is useful for
aligned D64 as well as D128. In an MP31 MoBA D64 route, changing
32x32 to 32x64 while letting both K and V TME loads feed SQMMA shared
operands directly removed their `local_load` conversions. BF16 D64
fell from 4.12 to 3.40 ms and FP16 D64 from 4.04 to 3.32 ms in
same-card paired full-operator timing. Independent BF16/FP16 references
with more than 2048 tokens passed, and irregular 96-token chunks used
the unchanged path. An FP16 D64 benchmark still trailed MATE at
3.34 vs 2.94 ms; its routed-attention Event was 1.263 vs 0.784 ms.
The remaining gap is inside each route CTA, not proof that the
descriptor layout failed.

Do not infer a useful pipeline from a larger `num_stages`: on the
direct K/V D128 route, stages 1 -> 2 increased shared memory from
28,672 to 90,112 B/CTA and latency from 123.7 to 271.8 ms for
chunk-1024. A 64 -> 128 KV tile failed the existing numerical
tolerance on the first representative case. Keep these as rejected
experiments; specify an explicit resource budget and a correctness
gate for compiler work on asynchronous TME/SQMMA overlap.

A concrete compiler request is a legal MP31 vector FP32 exp2 lowering.
Current Triton MoBA compiles `libdevice.exp(x*ln2)` through
`__nv_expf`; MATE assembly uses `AF32.EXP2.BST4`. The tested
Triton/MUSA build failed to lower vector `tl.exp2`, and exposed
`fast_expf` raised `NotImplementedError` when JIT compiled.
Define accuracy and special-value behavior first, then test
independent softmax/MoBA correctness and same-input route-kernel
timings. Static instruction differences do not quantify how much of
the measured gap exp2 accounts for.

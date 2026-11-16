# S60-TRITON-011: device indexing and compiler capability boundaries

## Scope and status

Version-scoped observations on S60/GCU300, Triton3.6.0 and
torch_gcu2.10.0+3.7.20260408. Reusable across device-side indexing, sparse
metadata construction and layout transformations. A limitation observed in
this compiler/launcher is not a statement about physical hardware capability
or every compiler release. Reprobe after a stack change.

## Atomic reservation is a compiler capability gate

The isolated INT32 tl.atomic_add(..., sem="relaxed") reservation expression
fails before execution because tt.atomic_rmw cannot be legalized. Do not
build an algorithm around this path without a fresh compile-and-execute
probe. This does not establish that the chip has no atomics or that every
atomic variant fails.

## Sorted INT32 values can support non-atomic sparse metadata

Native device torch.sort passed independent exact comparisons of INT32
values, including repeated maximum-value sentinels. Its returned permutation
reports an int64 type while the backend warns about Long-to-Int replacement;
the validated metadata construction consumes sorted values only. Index
permutations require their own range and correctness checks.

For an integer encoding bucket*capacity+item, bound every valid key strictly
below the reserved sentinel and check workspace/index extents before dispatch.
Sorted values, per-bucket lower/upper bounds and prefix-scanned counts can
construct CSR-like lists without atomic reservation. Preserve multiplicity,
padding, empty buckets and the required item order. This is a functional
alternative, not a claim that sorting is faster than atomics or another
algorithm; include metadata and workspace cost in the consuming operation.

## Launch dimensions have independent limits

The installed GCU launcher enforces grid.z<=255. Flattening independent task
coordinates or moving the large coordinate to another supported dimension
can avoid that limit while preserving the index mapping. Check every launch
dimension, flattening arithmetic, tail and grid bound on the target stack;
do not assume the same limit for grid.x/y or another backend.

## Local assembly can introduce an unsupported layout

The tested order-preserving join/transpose/reshape assembly fails GCU
legalization of tt.trans for a join-derived linear layout, including a
flattened two-dimensional spelling. Ordinary matrix transpose already works
on the same stack. Therefore diagnose the exact expression, layout and
generated IR rather than labeling transpose hardware unavailable.

Inspect installed API signatures before constructing an alternative; accepted
front-end syntax does not establish successful backend lowering. A staged
regular-memory implementation can avoid the local assembly, but its copies
and allocation must be counted before recommending it.

## Attribute memory paths to operands

The presence of a gather/scatter call does not establish that a large payload
uses it. Trace argument provenance and element extent: a kernel can transfer
its payload through slice/deslice routines while gathering or scattering
only small indices or normalization statistics. Conversely, incidental
metadata slice calls do not prove regular payload transfer. See
S60-TRITON-006 for scalar-base segment expressions. Static callees establish
lowering, not dedicated-engine utilization or complete-operation speedups.

## Dot interfaces and local-IR acceptance are different gates

On the recorded GCU300/Triton3.6.0 stack, installed semantic code and complete
compilation probes establish these frontend boundaries:

- `tl.dot(..., out_dtype=tl.bfloat16)` is rejected. Generate an allowed
  FP32/FP16 intermediate and convert for final BF16 storage instead; tested
  BF16-input/BF16-output kernels pass separate device-reference checks.
- The target allows only `input_precision='ieee'` in the tested complete
  kernels. TF32, TF32x3, BF16x3 and BF16x6 requests are rejected, even though
  the shared enum contains those names.
- FP8 E4M3/E5M2 operand types are rejected for this architecture by the
  tested Triton interface. BF16/FP16/FP32 mixed operand pairs are rejected
  because the two dot inputs must have the same type.
- For BF16 or FP32 inputs, requesting FP16 dot output did not change the
  actual local result from FP32 in the tested kernels; final FP16 storage
  remains possible. Always inspect the generated result type.

Synthetic local matmul IR can accept additional result types, FP8 operands,
TF32/TF32x3 and dynamic memrefs that are not demonstrated executable through
the normal frontend. Local BF16x3/BF16x6 properties and mixed operand types
are also rejected. Never infer an end-to-end capability from an enum entry
or one successful lowering stage. Record the failing layer and exact error,
and revalidate after stack changes. See [S60-TRITON-001](triton-convgen.md)
for bounded local alignment and the distinction from final storage type.

Sources: `convgen-extension-semantic`, `convgen-extension-compiler`,
`convgen-extension-local`, `convgen-extension-device` and
`convgen-extension-report` in [sources.json](sources.json).

## Evidence and limits

Catalog source IDs link isolated API/compiler probes, exact numerical checks
and operand-attributed artifacts outside this repository. Operator-specific
layouts, tuned constants, latency tables and candidate decisions remain in
their task reports. These observations require revalidation on a new stack
and do not authorize changing a task's precision or semantic contract.


## Dynamic zero strides require specialization in the tested mask path

On GCU300/S60 with Enflame Triton 3.6.0, a four-dimensional broadcast mask
view supplied runtime zero strides to a tiled attention load. The process
aborted during device execution with a SIP assertion: dynamic stride zero
was unsupported, with an explicit request to make the stride constexpr.
Ordinary non-broadcast masks had already passed the same public/reference
tests. Thus a successful compile and non-broadcast test do not establish
that a zero-stride view is executable through this lowering path.

Making only the relevant mask strides `tl.constexpr` passed separate
CPU-reference tests covering batch/head/query/key broadcasting, partial
blocks, boolean/additive masks and fully masked rows, without allocating
an expanded mask payload. Keep payload strides dynamic where possible;
S60-TRITON-008 records regressions from broader specialization. This is a
bounded remedy for the tested masked-load expression, not proof that every
GCU operation rejects runtime zero strides or that specialization is free.
Full-call timing must include host validation and dispatch; small masked
calls can become slower despite correct broadcast execution.

Sources: `sage-cleanup-dynamic-zero-stride-failure`,
`sage-cleanup-broadcast-correctness`, `sage-cleanup-attention-source` and
`sage-cleanup-comparison` in [sources.json](sources.json).


## Native sort wrappers and large-input performance require separate evidence


On the recorded torch_gcu2.10.0+3.7.20260408 stack, default and stable sort
wrappers reach a registered GCU sort.values_stable kernel. A composite wrapper's
dispatch-table entry alone therefore does not establish CPU fallback. Inspect
the invoked overload chain and actual device behavior; registration still does
not expose the internal sorting algorithm or physical execution unit.

Separate CPU checks validate INT32 sorted values, permutation bijection and
index-to-value mapping through large repeated-sentinel inputs; requested stable
ordering is also checked within ties. Reported index dtype and backend Long-to-
Int warnings do not replace range/bijection validation before inverse scatter.

Performance-only probes show a sharp length-dependent cliff in this installed
native sort. Default, explicit stable=False and stable=True all share the slow
large class; changing the API argument does not repair it. Small functional or
timing success cannot establish large-workload performance. Preserve complete
call costs and do not infer missing hardware or recommend an untested segmented
replacement. Exact extents, samples and workload-specific choices remain in
task artifacts. Sources: moba-round3-sort-checks, moba-round3-sort-timing and
moba-round3-report. Revalidate after backend/compiler changes.



## Disjoint key intervals permit native segmented routing sort

On the recorded GCU300/Triton3.6.0 and torch_gcu stack, independent head groups
can use native batched sort over shorter rows instead of one flattened sort.
Assign each group a disjoint ordered key interval so concatenated sorted rows
remain globally ordered. If invalid slots stay allocated, reserve an invalid
bucket inside each interval, include it in offsets, and suppress its consuming
descriptors. A shared global sentinel mixed between groups is not equivalent.

The sort permutation is local to each row. Inverse scatter must use that local
axis, followed by the exact original rank/token/head output mapping. The consumer
must decode the same new expert encoding. Preserve all multiplicities, arbitrary
selected IDs, masks, int32 bounds and precision; generic training routes need
not change. This is one algorithm/layout change, not a length dispatch table.

Independent CPU invariants cover sorted membership, empty/invalid buckets,
strides, inverse bijection, query IDs and descriptor coverage, including a large
sort class. Frozen integration and complete-call caps pass. Runtime-returned
consumer kernels retain matrix and structured payload-transfer paths. Native
sort registration remains GCU; its internal algorithm and physical unit remain
unmeasured. This avoids an observed native-sort size cliff without proving a
universal optimum or a first hardware-engine activation. Revalidate after stack
or layout changes. Exact sizes, tuned choices and timings remain in task data.

Sources: moba-route4-head-checks, moba-route4-sort-probe,
moba-route4-actual-paths, moba-route4-evaluation and moba-route4-report.

## Local rank shifts using tt.gather need an executable gate (2026-10-23)

On the reviewed GCU300/Triton3.6.0 stack, a bounded local retained-state shift
expressed with tl.gather can fail to legalize tt.gather during compilation.
An isolated sorted-insertion probe passes exact CPU state/order checks when
the rank extent is one, then fails at a nontrivial rank extent. The compiler
process exits139 after the legalization error; this is not evidence that the
generated kernel executed and accessed invalid global memory.

Two integrated forms with different final output stores pass the original
suite but both fail a separate packed streaming selector. The isolated probe
contains no attention dot, production route output or external payload gather,
so the failure is reproducible in the local-state expression itself. This does
not prove every tl.gather form is unsupported or that the chip lacks indexed
memory transfers: tops_gather_load on actual tensor payloads is a different
lowering path. A rank-one success can be constant-folded and does not establish
nontrivial local gather support. Probe exact shape/dtype/layout before relying
on it; static rank vectors or other supported expressions remain hypotheses.

Candidates were rejected before performance measurement and original restored.
Source, separate frozen/packed gates and isolated diagnostics use the
moba-sorted11-* evidence prefix; constants and run narrative remain external.

## Statically indexed rank vectors are a bounded alternative, not an unrestricted guarantee (2026-10-23)

The reviewed GCU300/Triton3.6.0 local-gather limitation can be avoided in an
ordered retained-state algorithm by representing ranks as statically indexed
row-vector tuples. Carry a candidate through fixed compare/select swaps,
preserving unsigned key order and ascending index ties, and store the already
ordered rank vectors directly. This removes the need for a local tensor gather
and repeated worst-entry/slot reductions; matrix and score arithmetic need not
change. Whether the larger static expression is useful is a separate gate.

Independent CPU union/order checks execute exactly for the selected bounded
tuple-state classes, including empties, ties, high unsigned keys, no-replacement
and mixed rows. An initial unrestricted probe nevertheless produces an exact
state mismatch at a higher rank extent after smaller classes pass. The cause
of that mismatch is not isolated and no hardware capacity or general compiler
miscompile is inferred. Preserve the failing oracle instead of treating small
tuple success as a universal capability claim.

A conservative bound on independent retained key/index vectors can select the
new representation, with the established tensor-state algorithm retaining the
larger-state path. This guard is about representation size, not sequence lengths
or public expected values. Both selected bounded classes and the complete
fallback selector need independent exact validation. Performance promotion
requires complete-call formal and untuned regression gates even after these
tests pass. No physical utilization, isolated instruction cost or universal
rank/group recommendation follows. Source, raw capability failure, separate
bounded checks and complete-operator outcome use moba-rank12-* source pointers.


The bounded integrated implementation also passes separate packed/strided and
high-rank fallback selector checks, including exact random indices against its
pinned parent. Source-bound actual captures retain the native matrix callee,
matrix ISA and declared shared storage. Retained-state unsigned-minimum calls
are removed while some other emitted calls and wait sites increase. The full
call and separately instrumented selection phase improve under the task gates;
this supports the combined algorithm/state change without attributing the gain
to one instruction or treating wait counts as utilization measurements.

## Simpler diagnostic bodies can encounter a different lowering failure (2026-10-23)

On the reviewed GCU300/Enflame Triton3.6.0 stack, a tiled BF16 dot stream with
unsigned-key candidate reductions compiled and executed with an inner repeated
selection loop. Removing that loop and retaining a single maximum-key checksum,
or a single maximum-key/minimum-index checksum, did not complete compilation
within the observed bounded runs. Python stack samples for the second form
locate the wait inside toolkit.run called by compiler.make_gcuir. This is a
compilation-stage observation, not device execution latency or a proved
particular pass bug; preserve the bounded duration and termination records.

Keeping the one selection inside a runtime one-iteration inner loop compiled
and passed separate exact CPU checksum checks. Actual returned artifacts retain
the native matrix callee and declared shared extent. The evidence supports
testing loop structure when a simplified diagnostic fails to compile; it does
not establish that every flat dot/reduction fails or that adding loops improves
production performance. The runtime bound must be justified by the input
contract, not a measured shape value. Revalidate on other compiler versions.

For performance ablations, keep every removed result live with a bounded
checksum and inspect the resulting loop/resource/matrix path. Even when these
paths match, subtracting separately compiled variants measures an intervention
effect, not exact additive instruction time: storage, scheduling and overlap
may change. Sources use the moba-breakdown-* prefix in sources.json; task timing
tables and failed scripts remain external.


## Device-only stable radix metadata requires separate capacity evidence (2026-10-26)

On the recorded GCU300/S60, Triton3.6.0 and torch_gcu2.10.0+3.7.20260408
stack, a stable device-only LSD radix partition is an executable alternative
to native sorting for the tested nonnegative int32 metadata. Per-block digit
histograms, a hierarchical exclusive scan in segment/bin/block order and
local cumulative ranks yield unique non-atomic destinations. Carry original
absolute row IDs with keys through every pass to retain stable multiplicity
and construct an exact inverse map. Disjoint head key intervals include their
own invalid sentinel, whose consumer descriptor count must be suppressed.
Preserve int32 extent bounds, segment tails, global ordering, invalid slots
and both producer/consumer expert encodings.

Separate CPU-oracle tests cover stable keys and payloads, hierarchical scan
boundaries, invalid/empty buckets and both head-major/global route layouts.
The frozen integration suite and separate packed/GQA/strided/streaming checks
pass. Runtime inference dispatch contains allocation, slices and views rather
than native Torch numerical operations. These are bounded functionality and
dispatch observations, not proof for every int32 range, layout or backend.
Exact extents, source hashes and workload timings remain in the task evidence.

Removing native operations does not establish a faster or lower-memory whole
operator. Materialized reordered Q, routed output and ordered merge payloads
can dominate temporary capacity. End-of-pass Python references can retain a
consumed route output while the next call's right-hand side allocates Q.
Explicitly releasing those references passed a separate forced-multipass
weak-reference/numerical guard, yet the large end-to-end warmup still OOMed.
Allocator diagnostics reported lower live allocated bytes after this change;
aggregate free bytes did not establish that the requested allocation would
succeed. No physical fragmentation cause or universal allocation limit was
isolated. This is contrary evidence against recommending lifetime deletion
alone as a complete capacity repair.

A subsequent disjoint-row scratch-reuse/direct-merge candidate did not finish
its bounded numerical gate and was terminated; the failing layer was not
isolated. It supplies no correctness, speedup or unsupported-hardware claim.
Both attempts were rejected and the task-entry source restored. Probe an
exact small ownership/layout class and complete the numerical gate before
timing an in-place or indirect-merge expression. A later read-only inspection
also found an unrelated workload in the shared container; do not attribute
earlier variation to it without matching timing evidence or terminate it.

Sources: moba-pure-radix-source, moba-pure-radix-checks,
moba-pure-operator-audit, moba-pure-frozen-gate, moba-pure-capacity-failure,
moba-pure-lifetime-check, moba-pure-lifetime-oom and
moba-pure-workspace-timeout in sources.json. Whole-call timing, allocator
messages, rejected source and run narrative remain outside the knowledge base.


## Packed exact digit prefixes reduce indexed-store sites (2026-10-26)

On GCU300/S60, Enflame Triton3.6.0 and torch_gcu2.10.0+3.7.20260408,
two independent small prefix counters can share one int32 scan. Encode one
predicate as one and the other as 2**16, then extract the corresponding low
or high 16-bit inclusive prefix. The semantic proof requires no low-lane
carry and no signed int32 overflow: each lane must remain below 2**16 and
the high lane below 2**15, including all intermediate partial prefixes. The
tested bounded block extent satisfies both. Revalidate larger blocks, signed
increments, different integer widths and different scan lowering separately.

For a four-bucket stable digit partition, two packed scans replace four
independent scans. Select the element's exact bucket prefix and scalar global
bucket/block base, then issue one unique key/index store pair per valid row.
Preserve stable original positions, segmented global offsets, invalid-key
slots, digit width, pass count and final-tail allocation masks. The selected
destination remains irregular: fewer indexed scatter sites are not a switch
to affine DMA or proof that a new physical engine was activated.

Independent CPU oracles validate stable sorted values and original-row
payloads across block tails, repeated keys, high key ranges and segmented
head/global route layouts. Integration and independent operator-purity gates
are separate from timing. Actual run-returned callee/source hashes support
the compiler-path comparison. Operator-specific constants, timings, promotion
decisions and capacity outcomes stay in the external task report. Callee counts
do not measure physical utilization, traffic or overlap.

A fresh unchanged-source run on another selected card completed a capacity
case that previously OOMed. This narrows the earlier capacity observation to
its execution environment; it does not isolate contention, fragmentation or
a hardware/card difference as its cause. Source identity and a fresh capacity
guard are necessary before transferring an OOM conclusion to another run.

Sources: moba-native-packed-source, moba-native-packed-primitives,
moba-native-packed-boundaries, moba-native-packed-frozen,
moba-native-packed-paths and moba-native-card1-baseline in sources.json.


## Public sorting expressions are not dedicated-unit evidence (2026-10-26)

Installed Enflame Triton3.6.0 source inspection shows tl.sort and tl.topk
calling a JIT sort_impl that constructs a bitonic comparison network through
hypercube reshapes, compare/select and reductions. The inspected installation
does not expose tl.sort_pair or tl.argsort. Its inspected GCU libdevice and
backend source declarations provide no verified dedicated stable key/value
sorting entry. This is an API/source observation, not an execution probe,
universal interface limit, physical-unit absence, or claim that a closed-source
compiler cannot specialize the generated network. Inspect the installed version
before relying on upstream API names. A public sort expression alone neither
establishes stable permutation semantics nor dedicated hardware activation.

Native Torch sort registration, pure Triton implementation and physical sorting
unit attribution are different evidence questions. Use actual executed compiler
artifacts for lowering and separate correctness/full-call gates for replacement
algorithms. Source: moba-native-sort-interface-audit in sources.json.


## Host SDK sort and device expressions have different invocation boundaries (2026-10-26)

On the reviewed S60 stack, installed topsaten_ops.h declarations and
libtopsaten.so exports expose topsatenSort and topsatenArgSort. A small C++
host bridge using installed topsatenTensor descriptors, int32 values/indices,
stable=True and the current GCU stream executes exact stable sorting. CPU
oracles and device radix equivalence validate repeated keys, tails, sentinels,
segmented head intervals and local-to-absolute permutation conversion in the
tested bounded classes. This establishes an executable host library path,
not an in-kernel Triton primitive or distinct physical sorting unit.

Separate actual-returned int32 tl.sort kernel captures contain comparison/
select, structured memory, broadcast/transpose and synchronization calls;
no dedicated tops_sort callee appears in those bounded variants. Public
bitonic syntax and exported host library routines are different evidence.
Do not call a host descriptor/stream routine through an elementwise device
extern interface. Revalidate SDK ABI, dtype/index bounds, stability, stream
and complete consuming-operation latency after any stack/layout change.
Workload-specific timings and implementation choices remain in task artifacts.

Sources: moba-sdk-sort-binding, moba-sdk-sort-equivalence and
moba-tl-sort-executed-lowering in sources.json.


## Match supplementary references before diagnosing a regression (2026-10-26)

Changing pass grouping in a supplementary test can change the order of
intermediate BF16/FP16 stores and normalization. A CPU error threshold chosen
for another grouping may fail the unchanged source as well. A newly added
class can likewise expose an inherited baseline error without an optimization
regression. Preserve the failed candidate and baseline controls. Pin reference
source bytes and match pass geometry, require exact/no-worse baseline parity
where appropriate, and retain the original frozen/default CPU-reference gates
and tolerances. Do not silently relax existing correctness criteria or present
baseline parity as an absolute accuracy improvement. Exact values and input
classes belong in the external task report, not a universal tolerance rule.

A fresh executable gate also reconfirms failure to legalize the tested INT32
relaxed atomic reservation expression. This is the same bounded compiler
capability observation above, not evidence of absent hardware atomics. Larger
indexed-payload execution failures after small classes pass require their own
runtime/address/layout gate; compiler acceptance alone is insufficient.

Sources: moba-generic-atomic-reprobe, moba-generic-baseline-forced-control,
moba-generic-baseline-dense-control, moba-generic-fusion-parity and
moba-generic-fusion-dense-parity in sources.json.

## Runtime Tensor tuples and Python positional expansion are distinct (2026-10-27)

In the recorded GCU300/Enflame Triton3.6.0 stack, passing a runtime tuple of
two BF16 Tensor pointers as one JIT launch argument reaches launcher
generation but fails in `ty_to_cpp` with `KeyError: ('*bf16', '*bf16')`.
The tested kernel selects tuple elements with `tl.static_range`; eliminating
pointer-valued select expressions does not remove this independent launcher
signature gate. Classify the failing stage separately from Triton frontend
or device-IR legalization.

A tuple constructed inside JIT from independently passed pointer arguments
is a different interface and works in the independently checked version.
At the Python call site, padding a tuple of Tensor objects and expanding it
with `*pointers` supplies scalar arguments to the launcher, rather than a
tuple-valued runtime argument. This copy-free host-side assembly passes
separate numerical checks; actual-launch compiler artifacts preserve the
device kernel relative to equivalent named pointer arguments. The Python
container itself does not concatenate or copy residual tensor payloads.

This observation covers the recorded scalar-pointer ABI, its source masks
and launch options on this installed stack. It does not imply that all
Python/Triton tuples are unsupported, that constexpr tuples fail, or that
newer compiler releases have the same launcher mapping. Reprobe a real
multi-pointer runtime tuple after a stack update. A packed-device-buffer
alternative changes allocation/data-movement work and must be timed with
that cost included; its operator-specific outcomes remain in task artifacts.

Sources: `residual-abi-smoke`, `residual-abi-source-equivalence`,
`residual-abi-launches` and `residual-abi-numerics` in [sources.json](sources.json).

## Validate complete pointer representation before indirect loads (2026-10-27)

On the recorded GCU300/Triton3.6.0/Torch-GCU2.10.0+3.7.20260408 stack,
a CPU int64 table of real residual Tensor addresses did not survive an
exact CPU-to-GCU-to-CPU value check. The transfer emitted the GCU Long-to-Int
replacement warning. The experiment rejected this candidate before any
indirect pointer dereference. Advertised tensor dtype alone is insufficient
evidence that a device address table preserves all address bits.

A separate split-int32 pointer-table attempt ended with a reported Sip
exception and process exit137. Its parent did not record a completed check,
so the exact failing transfer/cast/indirect-load operation is unconfirmed.
Do not infer universal inttoptr, uint64 or hardware-indirection support or
absence from this failure. The candidate was not retried. A fresh process
subsequently passed ordinary allocation/synchronization/readback without a
device reset, and a conventional contiguous-buffer fallback passed separate
numerical and frozen project tests.

For address-table mechanisms, validate full values before dereference,
isolate runtime probes, preserve partial logs on native failure, and keep
representation validation distinct from a successfully executable indirect
kernel. A contiguous packed payload avoids that pointer-table mechanism,
but allocation and full data-copy cost belong inside its complete-call
benchmark. Exact source shapes, performance and operator selection remain
in the external task artifacts, not in this reusable capability note.

Sources: `pointer-table-interrupted-smoke`, `pointer-table-recovered-smoke`,
`pointer-table-fresh-context-health` and `pointer-table-fallback-numerics`
in [sources.json](sources.json).

## Generated scalar launch wrappers can preserve readable in-JIT tuples (2026-10-27)

In the recorded GCU300/Enflame Triton3.6.0 stack, a small deterministic
import-time Python factory successfully generated a launch wrapper with
independent scalar Tensor-pointer arguments, then passed their tuple to a
readable helper inside JIT. This does not pass a runtime Tensor tuple through
the GCU launcher: its inspected actual signature still has separate pointer
slots, while the helper tuple exists inside compiled code. Separate numerical
and unchanged project tests passed. Actual launched specialization hashes and
paired timings are retained externally; this note does not publish tuning.

Triton decorates functions by inspecting Python source. A synthetic filename
registered in linecache before compile/exec supplied inspect.getsource without
writing a temporary source file. Repeated factory construction produced stable
source and argument names. The generated text used only fixed parameter names
and constexpr annotations derived from the helper signature, not tensor data,
addresses or benchmark-specific shapes. Wrapper construction occurred at
module import, not inside the timed public call; no payload packing was needed.

Scope: this installed Triton/Python inspection behavior and the recorded
scalar-pointer ABI. It does not establish generated-source portability for
every toolchain, cold-import latency, tuple launch support, arbitrary argument
annotations or performance of unrelated wrappers. Keep source recovery and
runtime argument binding as explicit checks after a stack change. Use this
organization only when source generation is authorized; it intentionally keeps
the enumerated runtime ABI rather than claiming to eliminate it.

Sources: `generated-abi-source-inspection`, `generated-abi-numerics`,
`generated-abi-actual-launches` and `generated-abi-structure`
in [sources.json](sources.json).


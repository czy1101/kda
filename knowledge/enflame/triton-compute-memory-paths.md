# S60-TRITON-002: computation, data movement and storage paths

Independent device follow-up:
[S60-TRITON-005](triton-hardware-trigger-probes.md) tests 53 standalone
configurations, resolves representative final instructions and demonstrates
a same-mapping scalar-page versus vector-row gather comparison. It preserves
profiler-capture failures and unstable matrix timing repeats.

## Scope and evidence

Reviewed 2026-10-16. Initial compiler observations: S60/GCU300, Enflame
Triton3.6.0, five cached MoBA routing variants in source `convgen-probe`.
Subsequent runtime-linked forward audits are scoped in the sections below;
no hardware-counter profile was available. User-provided summary:
`s60-hardware-paths-note` in [sources.json](sources.json).

Enflame's [Hot Chips 33 presentation](https://hc33.hotchips.org/assets/program/conference/day2/Enflame_Final%20Deck.pdf)
(slides 5-6) describes GCU-CARE 1.0 as programmable and supporting tensor
computation. GCU-CARE can be compared conceptually with Tensor Cores for its
matrix/tensor acceleration role. This is a functional analogy, not an
instruction-level equivalence. That presentation concerns the earlier DTU 1.0;
its dimensions, capacities and peak throughput are not S60 specifications.
Mapping current `tops_*` calls to specific S60 physical units remains unverified.

## Computation path

Express matrix multiplication using `tl.dot`; dtype, matrix shapes, layout
and launch configuration influence backend selection. No general switch to
enable `convgen` has been verified. In the existing MoBA routing kernel:

```python
# BF16 inputs, FP32 accumulation; a: [64, 128], b: [128, 64]
scores = tl.dot(a, b, out_dtype=tl.float32)
# Observed launch: num_warps=4, num_stages=1
```

| Outer M x N x K | Lowered local M x N x K | Actual LLVM callee |
|---|---|---|
| 64 x 32 x 128 | 32 x 16 x 128 | `tops_matmul_bf16_fp32` |
| 64 x 64 x 128 | 32 x 32 x 128 | `tops_convgen_m32_n32_k128_bf16_fp32` |

These are observed configurations in that kernel. A standalone dot of the
same size still requires artifact inspection. K is the multiplication
reduction dimension, not routing Top-K. See [S60-TRITON-001](triton-convgen.md)
for padding masks, exact metadata, timings and reproduction requirements.
Inspect routing, attention QK and attention PV independently; one call site
does not prove that every dot in the operator follows the same path.

## Data movement and storage paths

| Engine, path or resource | Role | Triton expression / organization | Markers in current artifacts |
|---|---|---|---|
| DTE data-transfer mechanism, including asynchronous interfaces | Copy data and wait for transfer completion | Express data access with loads/stores; backend chooses lowering | `tops_memcpy_async`, `tops_wait_dte`, `tops_private_dte_ctr` |
| Slice / padded-slice path | Obtain tiles and handle boundaries | Regular block access, valid-element masks and zero padding | `tops_slice_async_*`, `tops_slice_pad_async_*` |
| Transpose path | Form layouts needed by computation | Change access orientation or express a transpose | `tops_transpose_async_*` |
| Local / Shared storage resources | Stage intermediate data for reuse and exchange | Organize program-local reuse; backend determines placement | `tops_get_local_memory`, `tops_get_shared_memory` |

These markers are software entry points or allocation interfaces, not proof
that each operation has a separate dedicated physical unit. A logical
transpose may fold into indexing without an explicit transfer. Source-level
reuse does not prove register residency or a particular physical cache.
Do not infer S60 bank counts, memory capacity or sharing scope from names.

Pointer-based masked loads with `other=0` are a candidate way to express tail
padding; inspect actual lowering. Regular intra-block addresses can make
slice/copy patterns easier to recognize. Neither source syntax nor
contiguity alone guarantees a particular `tops_*` call or a speedup.

## What the existing artifacts actually show

All five cached narrow/padded routing variants contain the following static
call-site counts:

| Marker | Count per variant |
|---|---:|
| `tops_memcpy_async` | 2 |
| `tops_wait_dte` | 34 |
| `tops_slice_async_2_bf16` | 7 |
| `tops_transpose_async_2_bf16` | 1 |
| `tops_get_local_memory` / `tops_get_shared_memory` | 1 / 1 |

Padded-slice, deslice and synchronization markers also appear. These counts
are static, not dynamic transfer counts, bytes or wait durations. This sample
does **not** demonstrate a transfer-entry switch between the narrow and padded
variants. The confirmed distinction is their matrix-computation entry. Equal
static counts do not establish equal traffic or runtime.

## Candidate mechanisms and validation

- Preserve block-local regularity in sparse access: resolve a selected block
  base, then use actual strides to access its token/head-dimension tile.
  Blocks can be noncontiguous; their internal access must be analyzed from
  the real layout rather than assumed contiguous.
- Reuse a loaded KV tile across queries when traffic dominates. Measure the
  extra mask work, temporary storage and reduced program-level parallelism.
- Compare layout alternatives only when they change a concrete transfer or
  dot bottleneck; account for padding, transpose and packing costs.
- Treat prefetch/double buffering as an unverified pipeline hypothesis until
  lowering and runtime evidence show useful overlap. An `async` suffix,
  earlier source load or increased `num_stages` is insufficient evidence.

Through KernelPilot, retrieve matching source/metadata and actual IR call
sites, then compare shapes, addresses/layout transforms and synchronization.
Do not count declarations as executed calls. Run numerical tests separately
and measure complete operator cost under the frozen timer contract. Profile
transfers/overlap only with compatible tools that pass live discovery.
Reject an optimization when correctness, net latency or regression gates fail.

Sources: `convgen-probe`, `convgen-analysis`, `s60-hardware-paths-note`, and
`enflame-hotchips33-care` in [sources.json](sources.json). The local summary
contains additional historical ideas; those are not promoted to verified
S60 mechanisms by this card.

## Runtime-linked forward audit, 2026-10-16

A subsequent read-only audit ran the retained MoBA inference on BF16 B2 H16
D128 C128 Top-K8 at lengths4096/8192/16384/32768. It wrapped the existing
launch function in process, called the unchanged original, and inspected the
**returned CompiledKernel** TTIR/LLVM for that launch. This avoids matching
the current shape to an unrelated cached specialization. All16 captured
launches had convgen calls at every observed dot site, including every
historical rank pass; no tested shape lacked that compiler path.

| Stage | Observed outer M,N,K | Actual LLVM callee |
|---|---|---|
| Routing4096/8192/32768 | 64,64,128 | `tops_convgen_m32_n32_k128_bf16_fp32` |
| Routing16384 | 64,128,128 | `tops_convgen_m32_n64_k128_bf16_fp32` |
| Current-chunk QK | 64,64,128 | `tops_convgen_m32_n32_k128_bf16_fp32` |
| Current-chunk PV | 64,128,64 | `tops_convgen_m32_n64_k64_bias_bf16_fp32` |
| Historical-chunk QK | 128,128,128 | `tops_convgen_m64_n64_k128_bf16_fp32` |
| Historical-chunk PV | 128,128,128 | `tops_convgen_m64_n64_k128_bias_bf16_fp32` |

These include **outer N128** cases, so the earlier N64 recipe is not a
necessary condition. The `bias` suffix is part of the observed accumulator
entry and does not imply an added mathematical attention-bias input.
Do not generalize these observations to other precision, dimensions, layouts,
training or larger untested lengths. Non-dot sorting/gathers, Top-K selection
and merging are outside this matrix call-site coverage. All dots using
convgen does not establish that matrix work dominates end-to-end latency.

Compiled hashes, launch constants, source locations, TTIR/LLVM text hashes
and whole-call synchronization bind this evidence to the executed forward.
It still does not provide GCU-CARE hardware counters, physical-instruction
attribution or utilization. Source/test/benchmark hashes were unchanged.
Sources: `compute-audit-report`, `compute-audit-structured`, the four
`compute-audit-log-*` records, and `compute-audit-source-proof` in sources.json.

## Longer retained-m601 inference audit, 2026-10-16

A separate extension audited BF16 B2 H16 D128 C128 total Top-K8 at65536
and131072 with the retained256-row merge (m601). The actual returned compiled
kernels show convgen at all34 dot callsites across18 launched kernels. Each
shape launched one streaming routing kernel, one current-attention kernel and
all7 historical-attention rank passes. Routing remained outer64x64x128;
attention QK/PV shapes and callees match the table above. No narrower fallback
appeared at these two longer lengths, so there was no missing path to repair
and no speculative code change or new speedup claim.

| Length | Full-call event median ms | Three-invocation min..max ms | Missing convgen stages |
|---|---:|---:|---|
| 65536 | 7971.301 | 7943.343..7975.614 | none |
| 131072 | 28094.160 | 27801.543..28103.217 | none |

Timer and data helpers are frozen:25ms warmup/100ms rep/event median, then
median of three complete invocations fixed before timing; all raw samples are
retained. Each shape process is capped600s including compilation. These are
same-implementation timings, not an optimization comparison or a Torch timer
comparison. Frozen suite39 passed/6 skipped; independent CPU stable routing
ordering also matched every token for both long B2 lengths with Hq2/Hkv1,
D128/C128/K8 BF16, including rounded ties. Full new-length output/reference
and training performance were not separately verified. Source/test/benchmark
hashes remained unchanged across local/mirror/checkpoint/host/container.

All-dot convgen coverage still says nothing about physical GCU-CARE counters,
utilization, non-dot Top-K, sorting, gather/merge or host overhead. Routing
score work remains quadratic in length/chunk size.262144 and524288 remain
unmeasured; do not infer path coverage or measured latency for those shapes.
No profiler profile/tool allowlist was available. Sources: `long-moba-report`,
`long-moba-audit`, `long-moba-samples`, `long-moba-proof`,
`long-moba-audit-65536`, `long-moba-audit-131072`,
`long-moba-routing-65536`, and `long-moba-routing-131072`.

## Generic integer matmul can contain matrix instructions (2026-10-21)

Executed GCU300/Triton3.6.0 INT8-input/INT32-output probes use the generic
`tops_matmul_si8_si32` entry and contain matching linked
`m.vmm2.mode19.s.acc0` instructions. Separate CPU integer references match
exactly. Therefore a generic `tops_matmul_*` callee does not universally
exclude matrix instructions; identify the input type and inspect the matching
binary. Earlier BF16 generic-path observations remain scoped to their tested
specializations. This does not identify physical execution-unit utilization
or establish a speedup. See [S60-TRITON-001](triton-convgen.md) for the separate
floating-point convgen predicate and output-type boundaries.

Sources: `convgen-extension-device`, `convgen-extension-analysis` and
`convgen-extension-report` in [sources.json](sources.json).

## Preserve affine bases across masked loops (2026-10-21)

On GCU300/Triton3.6.0, an executed regular K/V loop with boolean-mask skip
control carried complete two-dimensional pointer tensors between iterations.
Its masked K INT8 and V FP16 payloads lowered to `gcu.gather_load` and
`tops_gather_load_*`, including indexed-load instructions in the matching
binary. The original dense branches already used native matrix calls and
structured transfer; this was a bounded masked-payload issue.

Rebuilding each iteration's tile pointer from an immutable base plus a
scalar loop offset restored affine slice/padding transfer lowering and
removed those payload gather calls in the sampled specialization. Matching
convgen and `m.vmm2.mode17.f` remained. Linked binaries retain `l.idma`,
`l.wait` and `l.wait.dec`. Independent numerical guards and separately timed
boolean-mask cases support reuse of this source-organization mechanism.

Preserve the actual physical strides, row-major versus tile-major block
offsets, per-block scale ownership, first-tile base, valid tails and next-tile
prefetch bounds. An arbitrary row-ID lookup is not made affine by spelling
the same irregular mapping differently. Check generated address provenance
and actual payload extents rather than labeling every gather as metadata.

The measured additive-mask cases did not share the large boolean-mask gain;
the original additive path did not have the same skip-controlled pointer
problem. Dense performance was near parity. A blanket constexpr geometry
candidate also removed gather but regressed dense full-call timing; see
[S60-TRITON-008](triton-msa-static-geometry.md). Static waits even increased
in the useful masked specialization, so neither fewer waits nor gather
removal alone proves an end-to-end win. Physical-unit utilization, dynamic
traffic and transfer/compute overlap remain unmeasured.

Sources: `sage-attention-path-baseline`, `sage-attention-affine-path`,
`sage-attention-static-path`, `sage-attention-path-comparison`,
`sage-attention-confirmation`, `sage-attention-correctness`,
`sage-attention-integrity` and `sage-attention-report` in
[sources.json](sources.json). Exact workloads, tuned choices and timings stay
in the external task workspace.

## Recurrent GLA payload audit follow-up (2026-10-22)

Actual returned CompiledKernel objects from GCU300/Triton3.6.0 GLA launches
show structured slice/padded-slice, deslice/transpose and DTE-wait calls for
sampled dense BF16 and FP16 tail/variable-length forward/backward paths.
GCU IR traces K/V pointer arguments through scalar address arithmetic and
memref reinterpretation to payload slice operations. This attributes real
operands, rather than treating a small metadata memcpy as payload evidence.
No gather/scatter callsites were observed within this bounded sample.

The recurrent fallback's computation-path activation retains these transfer
families, while execution-group splitting changes local tile decomposition.
Thus a missing matrix path and an already structured memory path can coexist
in the same kernel. Neither library names, static waits nor linked DMA code
establish physical DTE utilization, bandwidth, efficient scheduling or useful
overlap. The parent correctness failure prevented performance promotion;
exact operand traces, workloads and artifacts stay outside the knowledge base.

Sources: `gla-runtime-path-audit`, `gla-path-analysis` and
`gla-fallback-trigger-probe` in [sources.json](sources.json).

## Keep address safety separate from semantic validity (2026-10-22)

On GCU300/Triton3.6.0, a regular row-by-dimension tile can lose structured
transfer lowering when a per-lane `where` clamps its row index, even if the
original bounds mask remains on the load. Runtime-linked BF16 chunk-mean
payloads show actual gather calls in that case. Removing redundant pointer
clamping under unchanged valid-element masks restores affine slice/padding
transfer. Resolve the row tile from scalar bases and genuine strides; do not
remove a safety predicate or reinterpret arbitrary row IDs as adjacent rows.

A related case has affine ordered LSE storage but an arbitrary route-validity
load mask. If the wrapper has allocated every bounded row/rank slot, loading
with allocation bounds first and applying semantic validity afterward can
expose the structured path. Invalid values must be replaced before maxima,
exponentials, products or reductions; invalid staged NaNs must not contribute.
This requires an allocation proof, not an assumption that masked addresses
are safe. Real tails still need their original memory masks.

The paired executed artifacts, independent CPU selector/merge guards and full
workload gates are linked below. Address or mask simplification alone does not
prove a speedup: a standalone affine-address candidate failed a short-class
regression gate despite improving longer cases. Static DMA or matrix entries
still do not measure physical unit utilization, traffic or overlap. Revalidate
dtype, strides, allocation ownership, numerical sentinels and whole-call cost.

Sources: `moba-affine-lowlevel`, `moba-affine-paths`, `moba-affine-guards`,
`moba-affine-performance` and `moba-affine-report` in [sources.json](sources.json).


## Scalar structural coordinates can expose regular metadata transfers


On GCU300/Triton3.6.0, extracting structural coordinates from every lane of a
flattened index and recombining their physical strides can retain indexed
INT32 loads even for contiguous token storage. A program with scalar rank/head
coordinates and affine token offsets instead selects slice/padded-slice payload
calls. Preserve original slots, arbitrary input strides and invalid sentinels;
do not replace an arbitrary selection with adjacent rows.

The same principle applies to head-major FP32 normalization statistics. A
flattened token/head tile emits gathered reads and scattered writes, whereas
one scalar head with a token tile exposes structured LSE slice/deslice paths.
Large payloads then have different real row strides, so removing small metadata
gathers alone is insufficient: verify all payload addresses, tails, invalid NaN
shielding, independent precision checks and complete-call regression caps.

Runtime-returned formal kernels and independent numerical guards confirm these
bounded transformations. The vector special-function and matrix paths were
already active. Gather/scatter also run on GCU; the switch is not evidence of a
previous CPU path or first physical-engine activation. Callees/ISA do not measure
utilization, bandwidth or overlap. Exact tuning choices and timings stay in the
external task data. Sources: moba-round3-paths, moba-round3-route-checks,
moba-round3-merge-checks, moba-round3-performance and moba-round3-report.

## Move validity filtering after safe allocated-padding loads (2026-10-23)

On the reviewed GCU300/Triton3.6.0 stack, a periodic per-column rank-validity
mask on an otherwise contiguous internal-workspace load selects integer gather.
Simplifying partition/rank address arithmetic alone retains gather and does
not improve the paired kernel. A separate control loads the complete allocated
row tile with only memory-boundary masking, then applies the same semantic
rank-validity predicate to loaded values. Actual run-return artifacts switch
from tops_gather_load_si32 to structured tops_slice_pad_async_2_si32 and
tops_slice_async_2_si32. Reduction calls, tie semantics and valid outputs remain
the same; resource, staging, select and fence code can still change.

The mechanism requires ownership of every loaded padding element inside the
allocation. Never remove actual allocation-boundary checks or read beyond
physical rows/columns. Keep invalid keys/indices masked before selection and
preserve unsigned key bit patterns. Loading allocated padded lanes can increase
physical bytes; semantic work being identical does not mean traffic is equal.
An arbitrary sparse address mapping cannot be made contiguous by moving its
mask, and a source rewrite is not proof that this lowering will be selected.

Separate full-output equality plus independent CPU rows pass for the bounded
int32-workspace class. Both timing orders support a substantial removable cost
from the original masked-load organization and resulting gather lowering.
This does not isolate physical gather-instruction latency: address generation,
library path and associated generated code change together. Captured call sites
are not dynamic hardware counters. Synthetic precomputed partial candidates
and one merge-kernel class do not establish a full-operator gain or compatibility
for other dtypes/layouts, and the rejected partition algorithm still has a
separately dominant local stage. No deployed operator change is implied.
Sources in sources.json: moba-gather-cause-source, moba-gather-cause-correctness,
moba-gather-cause-paths, moba-gather-cause-timing and moba-gather-cause-environment.

## Load API spelling and async flags can compile identically (2026-10-27)

On GCU300 with Enflame Triton 3.6.0, one controlled masked-row load ablation
independently compiled a standard `tl.load` path and an existing `tle.load`
path. Both used a scalar row base plus contiguous hidden offsets, a masked
non-power-of-two BF16 tail with zero padding, `eviction_policy="evict_first"`,
one execution group and one stage. The TLE variant included both synchronous
loads and a bounded source-level `is_async=True` case. Arithmetic, masks,
pointer ABI and launch geometry were preserved.

The returned actual-launch CompiledKernels had distinct source/cache hashes
for the two variants, while paired raw GCU IR, LLVM IR and final linked
fatbin hashes were identical throughout the contracted cases. Independent
CPU-reference tests passed for both paths. Thus different frontend load APIs
and an async source flag need not introduce different executable code in this
version/layout. Inspect the generated specialization before attributing a
runtime change or useful transfer/computation overlap to that spelling.

This is a bounded compiler observation, not a general equivalence theorem
for TLE and Triton. It does not cover other TLE primitives, strided/gather
layouts, other launch layouts, other compiler versions, TLE-free package
installation, physical occupancy or hardware-counter measurements. Exact
operator shapes, paired timings, snapshot diff and test coverage remain in
the external task artifacts rather than this reusable card.

Sources: `attnres-load-ablation-launches`, `attnres-load-ablation-equivalence`,
`attnres-load-ablation-numerics` and `attnres-load-ablation-comparison` in
[sources.json](sources.json).


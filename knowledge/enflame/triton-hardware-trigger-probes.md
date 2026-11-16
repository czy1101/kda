# S60-TRITON-005: standalone compute and structured-memory triggers

## Scope and evidence

Observed 2026-10-16 on GCU300, Triton 3.6.0, torch_gcu
2.10.0+3.7.20260408, TopsCC v1.9.17 / compiler 5.3.2. This independent
microkernel experiment extends S60-TRITON-001/002. There are 53 passing
configuration tests and three shared-input equivalence tests. Numerical
checking is separate from performance programs. Each inspected specialization
has source, TTIR, GCU IR, LLVM-dialect IR, metadata and fatbin hashes. Nineteen
representative binaries also have extracted GCU300 ELF and disassembly.

These are selected software/ISA paths, not an enumeration of physical cores,
resource capacities or theoretical peaks.

## Compute expressions and final instructions

Default outer tile is M64/N64/K128, warps4/stages1.

| Source expression / input | Observed callee | Observed final mnemonic |
|---|---|---|
| BF16 `tl.dot`, N64 | `tops_convgen_m32_n32_k128_bf16_fp32` | `m.vmm.mode0.bf.vsel0` |
| BF16 `tl.dot`, N32 | `tops_matmul_bf16_fp32` | `m.mop.cvt.da.rne.bf16f32.vsel0`, `m.mop.mdm.f32.qa`, `m.mop.madd.f32.qa` |
| FP32 `tl.dot`, N64, input_precision=ieee | `tops_convgen_m32_n32_k128_fp32_fp32` | `m.vmm2.mode18.f` |
| INT8 to INT32 `tl.dot`, N64 | `tops_matmul_si8_si32` | `m.vmm2.mode19.s.acc0` plus other operations |
| FP32 `tl.fma` | `tops_fma_v4_fp32` | `m.mop.mac.f32.qa` |
| FP32 `tl.exp` | `tops_exp_v4_fp32` | `m.msf.mode0` |
| FP32 `tl.rsqrt(x+2)` | `tops_rsqrt_v4_fp32` | `m.msf.mode7` |
| FP32 `tl.sum` | shifts and vector additions | `m.mop.madd.f32.va` |
| FP32 softmax maximum | `tops_redmaxnumf_v4_fp32` and related entries | `v.vmaxa.f32` and other max operations |

The INT8 sample was slow despite containing a matrix-family instruction.
Neither the generic matmul callee nor the absence of convgen proves that
matrix hardware is unused. FP32 datatype/precision-request metadata and
random-input tolerance tests do not establish exhaustive IEEE accuracy.

BF16/FP16 M64/N64 at K32/64/128/256 selected corresponding convgen entries.
Changing warps1/4/8 for M64/N64/K128 selected local tiles m64_n64/m32_n32/
m16_n32 respectively. FP16, M32, M128, N128 and transposed-B variants were
also tested. These cases do not define a universal supported-size threshold.

For an N32 logical problem, load a padded N64 B tile with `mask=n<32,
other=0`, run the N64 dot and store only the N32 result. B retains its physical
K-by-32 layout; no separate packing kernel is required. This selected BF16
convgen and `m.vmm`. Shared-input narrow/padded outputs matched each other
and a CPU reference. Preserve downstream invalid-score masking for routing
and softmax. The padded microkernel measured approximately 0.256 ms in two
rounds for 256 independent tiles, including masked loads and stores.

## Structured memory: scalar page ID versus vector row IDs

The comparator uses exactly the same source-row mapping: 256 pages of
64-by-128 BF16, page permutation `(37*p+11)%256`.

1. Load one ID per source row and form a 2-D pointer tensor. This selects
   `tops_gather_load_bf16`; final code includes `m.vldxda.h.vsel0/1`.
2. Load one scalar page ID per program, calculate its base and use affine
   row/column offsets within the page. Payload selects slice/deslice without
   gather. Its small `tops_memcpy_async` copies the page ID, not the payload.

In the same repeat process, row mapping took 4.87321 ms and page mapping
0.19555 ms, approximately 24.9x. Earlier measurements were 4.87075 / 0.18789
ms. Independent shared-input tests establish identical output for the full
shape and a 131-row/117-column/stride256 tail case.

This transformation requires real block structure. It cannot replace arbitrary
row selections by different pages. This does not establish availability of a
general indexed-row DMA primitive or a GCU400-only gather operation.

Affine pointer and block-pointer loads produced the same slice-family entries
and nearly equal timings. Physical transpose generated
`tops_transpose_async_2_bf16`. Fixed row stride256 and a one-BF16 base offset
retained slice calls. Removing source masks still left slice_pad operations
because lowering also handles internal boundaries. A flat copy did not become
one large payload memcpy. No universal alignment or page-size hardware
threshold was established.

GCU IR includes `memref_ext.slice_start`, `slice_pad_start`,
`slice_deslice_start`, `deslice_start` and `memref.dma_wait`, matched to actual
LLVM calls. Do not infer a separate physical engine for every operation name.

## Measurement limits and rejected interpretations

Timing uses five warmups and nine trials of ten launches, GCU events plus
synchronized wall time. Inputs repeat without cache flushing. Effective
bandwidth counts logical payload read/write bytes, excluding indices,
temporary transfers and bus transactions; it is not HBM peak bandwidth.

The first low-MAD sweep showed narrow BF16 dot at 43.85693 ms versus about
0.256 ms for padding. Two later narrow-dot repeats had roughly 112 ms medians
and 28-37% MAD. Streaming matmul stages1/2/3 also became highly variable,
with second-scale outliers. Cause was not established. Preserve contrary
results; do not promote a stable 171x matrix speedup or stage3 performance
rule. The first streaming sweep was 0.68037/0.67686/0.52620 ms, but stable
overlap or stage3 benefit remains unconfirmed. Padding and page-transfer fast
cases remained low-MAD in the paired repeat.

No matching GCU analysis-tool profile existed in the checkout. Live explicit
queries discovered Topsprof 1.9.17 and Operator/Memcpy/CDTE/SDTE domains.
All five capture attempts aborted at TopsProfiler initialization with
`status == DTU_SUCCESS (-160 vs. 0)`, exit134; CSV files were empty. These
captures are discovered but unusable. No driver, permissions, clocks or
privileged counters were changed.

The installed `gcuobjdump` wrapper failed on raw bundles in topscc mode and
misparsed its single-hyphen `-mcpu` flag in clang mode. Extract the GCU300 ELF
member using the clang bundle directory and run:

```bash
/opt/tops/bin/llvm-objdump -d --triple=dtu --mcpu=gcu300 device.elf
```

Plain `-d` incorrectly decodes the architecture. Explicit GCU300 decoding
reveals the cited instructions, but some runtime/printf routines still fail
decoding. Full-file static counts include runtime and are not dynamic
utilization or occupancy. Unit counts and actual overlap remain unverified.

## Follow-up source audit: production candidates, not measured improvements

A source-only review of 57 Python files in FlagAttention's Enflame backend
found the vector-page expression in the active shared HY3 static BF16/FP8
decode kernels. The scheduler creates 64-token-aligned task starts and the
producer processes 64 tokens per iteration, making scalar page selection a
candidate to verify. Check every accepted task-map source and tail before
changing it; no production lowering change or speedup was established.

The portable HY3 prefill kernel also uses vector page IDs, while the GCU
page64 specialized branch already uses a scalar page ID and block pointers.
Dynamic decode, MiniMax/MSA and NSA selected-block paths reviewed already
use scalar page/block bases. MoBA's grad-enabled Q reorder has vector row
indices, but arbitrary selected query rows cannot be replaced by a whole
page without preserving their mapping. Its inference path stages selected
rows using native index_select before affine attention/merge loads; source
alone does not establish that library operator's internal lowering.

Read the audit's dispatch details and source hashes before choosing a
candidate. Sources: `hardware-trigger-backend-audit`,
`hardware-trigger-backend-scan`, `hardware-trigger-backend-audit-validation`.

## MSA follow-up: six representative BF16 compiler specializations

Independent compilation of exact local MSA JIT function snapshots confirms
convgen calls and `m.vmm2.mode17.bf` for all six sampled w1/s1 configurations:

| Specialization | Observed convgen tile family |
|---|---|
| Sparse prefill direct, GQA8, D128, K128 | m8_n128_k128, plus accumulator/bias entry |
| Sparse prefill grouped8, GQA8 | m64_n128_k128, plus bias |
| Sparse decode grouped2, query length4 | m16_n128_k128, plus bias |
| Sparse decode direct, query length1 | m8_n128_k128, plus bias |
| Fused index prefill score/TopK4, query tile64 | m64_n128_k128 |
| Index decode, 32 index heads | m32_n128_k128 |

The four attention samples have `tops_exp2_v4_fp32`, `m.msf.mode0`,
multiply-accumulate and maximum/sum reductions. Index scoring samples have
matrix and comparison/reduction work without attention softmax exp2.
All six have actual slice/transpose/memcpy/DTE-wait call sites and no
`tops_gather_load_bf16` call. The source selects scalar physical pages before
affine K/V tiles, including the grouped candidate-union path.

These are compile-only snapshots with representative runtime arguments and
explicit heuristic constants. JIT warmup compiled without launching kernels.
Do not infer production execution, numerical correctness, performance,
occupancy or overlap. FP8 and unsampled shapes need separate inspection.
M8 selecting convgen here narrows any assumed minimum-M rule; dispatch still
depends on layout, tile, warp splitting and compiler/device library.

Sources: `hardware-trigger-msa-report`, `hardware-trigger-msa-summary`,
`hardware-trigger-msa-source-proof`, `hardware-trigger-msa-artifact-hashes`.

## Sources and revalidation

Sources: `hardware-trigger-report`, `hardware-trigger-summary`,
`hardware-trigger-artifact-hashes`, `hardware-trigger-equivalence`,
`hardware-trigger-final-code`, `hardware-trigger-capture-failures` and
`hardware-trigger-environment` in [sources.json](sources.json).

Revalidate source, dtype, layout/stride, tile, launch options, compiler/device
library, precision semantics and input class. Use the isolated experiment's
controller for contracted SSH. Keep numerical tests separate from benchmarks,
and preserve repeated samples rather than selecting only fast runs.

Follow-up: [S60-TRITON-006](triton-arbitrary-q-row-segments.md) validates
regular D-segment transfers for arbitrary MoBA Q row IDs while preserving their
full mapping. It is a per-row transformation, not an assumed whole-page mapping.

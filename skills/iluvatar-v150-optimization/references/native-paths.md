# V150 native paths and selection boundaries

These are optimization decisions distilled from the maintained
[hardware reference](../../Iluvatar/天数V150_硬件特性与计算搬运单元.md) and
[knowledge base](../../Iluvatar/天数V150_知识库.md), measured on BI-V150 OAM /
ivcore11 with CoreX/Driver 4.4.0, PyTorch 2.7.1, Triton 3.6.0, as of
2026-10-15. Recheck the live environment; do not extend results to other chips,
compiler versions, shapes, masks, or numerical ranges without evidence.

## Compute selection

| Mechanism | Trigger and use | Generated evidence / caveat |
|---|---|---|
| TCU FP16/BF16/FP32 | `tl.dot`, legal operands/layout; tested instruction fragments `16x16x16`, FP32 accumulation | `ml_matrix_mad_f32_f16_m16n16k16`, BF16 or FP32 variants; tested GEMM tile `32³`, 4 warps, 1 stage |
| TCU INT8 | `tl.dot` with INT8 inputs, INT32 accumulator and `out_dtype=tl.int32`; tested K fragment 32 | `ml_matrix_mad_i32_i8_m16n16k32`; check signedness, padding, and overflow |
| Packed FMA | FP16/BF16 operands kept in that type; explicit `tl.fma(a,b,c)` | `ml_fma_v2f16` / `ml_fma_v2bf16`; two independent elementwise FMAs in a 32-bit register, not a reduction |
| Packed ordinary arithmetic | FP16 `+`, `*`, `tl.minimum/maximum` with pairable thread layout | `ml_pk_add/mul/min/max_f16`; measured BF16 multiplication packs, but ordinary BF16 add/min/max widen and convert |
| Integer dot4 | CoreX C++ signed `__dp4a(int,int,int)`, four signed INT8 bytes in each packed input plus INT32 accumulator | Runtime correctness and `llvm.bi.idot4` confirmed; native mnemonic incompletely decoded, peak not measured |
| Integer rearrangement | `triton.language.extra.corex.libdevice.byte_perm/popc/brev/mulhi` | `ml_byte_permute_b32`, `ml_bcnt1_b32`, `ml_brev_b32`, signed INT32 `ml_mul_i64_i32`; thread-local byte rearrangement is not shuffle |
| SFU math | FP32 `tl.exp2/sqrt/rsqrt`; reciprocal entry depends on required precision | `ml_exp_f32`, `ml_sqrt_f32`, `ml_rsq_f32`, `ml_rcp_f32`; correctly rounded reciprocal also used FMA correction |
| Precision conversion | `.to(tl.float16/bfloat16)` | `ml_cvt_f16_f32` / `ml_cvt_bf16_f32`, plus packing/reordering; include conversion instructions and temporary registers |

FP16 ordinary `a*b+c` fused in the tested layout with
`enable_fp_fusion=True`; disabling it produced separate packed multiply/add.
BF16 ordinary `a*b+c` did not fuse under either setting; use explicit `tl.fma`
and inspect the output. The FP16 FMA probe differed from FP64-then-FP16 reference
at one of 65536 elements by 0.0009765625. Read the knowledge base's
“半精度打包 FMA” numerical example before relying on bitwise equivalence.

FP8 storage is software-decoded into FP16/BF16 matrix computation on the tested
path. Confirm decoding, scale, layout, and register costs rather than assigning
native FP8 throughput. FP32 `ieee/tf32/tf32x3` generated the same FP32 matrix
instruction in the measured cases; `bf16x3/x6` increased work and registers.
Names alone do not imply a faster precision mode.

Measured `tl.log2` expanded to arithmetic/comparisons; `libdevice.tanh` expanded
to exp/reciprocal/arithmetic/branches. Do not count either as one native
instruction. `exp2` increased SFU counters; BFU counters also occur in a copy
baseline and do not stand for the number of exponentials.

## SME matrix staging

SME prepares matrix tiles in shared memory for TCU consumers. Automatic frontend
candidates require CoreX, SME not disabled, FP16/BF16/FP32/INT8 tensors with
original and squeezed rank at least two. Row-contiguous candidates need the
last dimension divisible by `64/element_size`; column candidates use the
noncontiguous tensor's penultimate dimension. This is a compiler selection
granularity, not a cache-line or universal transaction-size claim.

The backend still needs a suitable dot operand consumer, layout, and real row
stride. Explicit `tl.load(pointer_tile, stride=actual_row_stride)` supplies the
stride in elements, but does not force SME on arbitrary loads.

For the independently tested FP16 `32³` dot:

- Rank-two inputs selected SME automatically. Flattening the host tensors
  disabled automatic selection although addresses still formed the same tile.
- Supplying correct `stride=32` on those flattened dot operands restored SME.
- Elementwise `a+b` consumers used no SME, with or without explicit stride.
- K<29 masked loads with zero fill remained correct and used SME plus fixup
  operations. Do not extend this to arbitrary masks or zero overhead.
- Launch `use_sme=` was recomputed by the frontend. Its metadata is an argument
  candidate bitmask, potentially including output tensors, not execution proof.
- A fresh process with `TRITON_DISABLE_SME=1` disabled the automatic path.
  Separate processes avoid confusing JIT/cache specialization effects.

Look for `sl_sme_load_16x1b64_rowxfb16`, SME intrinsic calls in LLIR,
`sl_wait ... g2scnt(0)`, and the matrix consumer. Check the entire sequence and
runtime correctness. In another FP16/BF16 `512³` test with tile `32³`, explicit
stride reduced VRF 20→16/thread with shared unchanged at 4096 B/block; this is
a resource result, not a universal speedup.

Known numerical failure: converting both FP16 load results to FP32 before dot
in the tested `32³` probe removed SME, still emitted FP16 matrix instructions,
and produced max absolute error about 31.99448. Root cause is unresolved. Reject
that candidate; do not generalize the failure to all FP32 dot operations.

## Memory, layout, and synchronization

- Ordinary global reads/writes use `ml_lsa_*`. True pointer alignment, masks,
  and per-thread layout determine packing. A 2-byte FP16 offset changed the
  measured dword loads to ushort loads and increased static data-load positions
  2→4 and VRF 6→8. Alignment hints must state true facts.
- `.ca/.cg/.cv` changed load encodings in the tested backend. Hold working set,
  reuse, offsets, and layout fixed to test cache consequences. Encoding alone
  does not establish which cache is bypassed or a bandwidth improvement.
- Vector source types do not promise matching instruction width: tested C++
  `uint4` read lowered in LLVM to one `<2 x i32>` and two scalar i32 loads.
- Uniform program-level scale reuse reduced static load positions and VRF, but
  its load still targeted vector registers. Common expressions do not prove
  SRF/Single-Lane scalarization.
- Shared is 128 KiB/SM and up to 128 KiB/block in this environment. Keep both
  read and write layouts compatible with consumers. Scalar 64-lane contiguous
  32-bit accesses were conflict-free; `word_index=lane*64` conflicted severely.
  XOR swizzling helped the matching scalar test, not every BF16/SME consumer.
- Ordinary shared accesses use `ml_slb_*`; block synchronization uses
  `sl_barrier`. A barrier is not proof that outstanding SME transfers completed.
  Preserve completion waits and buffer ownership before use/reuse.
- More `num_stages` consumes shared and can reduce residency. The tested
  stages 1/2/3/4 used 4/8/12/16 KiB and all had `g2scnt(0)` waits; neither
  stage count nor static SME-load count proves transfer/compute overlap.
- Global atomic adds were verified for INT32/FP32/FP16/BF16, plus INT32 CAS.
  FP32 atomic max used integer atomic operations. Aggregate locally when useful;
  preserve numerical requirements and verify contention costs.
- `sem="relaxed"` removed acquire/release-related waits/cache maintenance in
  the tested INT32 atomic add; release inserted pre-waits, acquire post-waits
  plus `ml_lsa_wbinv`. Do not weaken publication/locking protocols for speed.

## Warp64 and scheduling resources

V150 has 16 SMs, Warp64, 262144 32-bit VRF entries/SM, separate SRF, 8192
runtime-reported resident threads/SM, 16 MiB L2, and 32 GiB HBM on the reference
device. Verify the live device before using these figures for budgets.

Use actual participating lanes and 64-bit ballot masks. C++
`__shfl_xor(x,32,64)` crossed Warp32 halves correctly; shuffle disassembly used
`SPERMUTE.B32`. Short rows may need subgroup-aware work assignment rather than
more warps. Shared is needed when exchanging beyond a warp.

Estimate VRF demand as registers/thread × threads/block, and shared demand
including all stage buffers. Residency also depends on allocation granularity,
SRF, thread limits, and launch constraints; verify runtime prediction and ixKN.
256 registers/thread is an observed pressure point, not a proven universal cap.
Inspect spill/local instructions and allocation over the whole grid.

Small decode/head/expert grids can leave SMs idle or create tail waves. Consider
parallel decomposition and reduction costs together. Account for masked lanes,
address arithmetic, and branch divergence on irregular sequences.

Pinned/nonblocking transfers worked; runtime reported one async engine. Neither
that property nor two streams proves overlap. The measured two-stream kernel
case had no gain; a cross-stream Graph capture failed while a single-stream
Graph worked. Revalidate the exact dependency graph rather than claiming the
chip lacks all concurrent execution.

## Unsupported or insufficient evidence

`__dp2a_lo` failed compilation in the tested SDK. Header declarations of
TMA/WGMMA/SMEX/async operations span targets and are not V150 support evidence.
Historical TLE copy used ordinary LSA/SLB, not SME or hardware asynchronous
copy; its sequential pipe did not establish hardware mbarrier overlap.
`nv_mma_shared_layout=True`, WGMMA and related primitives have recorded failures.
Use only the task-enabled, project-documented TLE subset after its assessment.

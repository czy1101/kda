# PPU matrix and transfer lessons

These lessons were extracted from validated PPU-ZW810E operator work on
2026-11-01, using PyTorch 2.10, Triton 3.6 and the installed FlagTree PPU backend.
Hardware capabilities, compiler implementation details, and performance outcomes
are different kinds of evidence. Compiler conditions below describe that observed
stack, not an ABI or a promise about another FlagTree revision. Use existing
capability knowledge directly; inspect the current lowering when a candidate
depends on its exact conditions. Keep task-specific sources, results and compiler
dumps in the task workspace rather than this skill.

## 1. Chained GEMMs: make the accumulator feed operand A

A dot result in MMA accumulator layout does not automatically become an efficient
input to the next dot. On the observed PPU compiler, the C-to-A conversion path
can reuse/repack the fragment through native half-precision MMA instead of a
general Shared round trip. Operand B does not have the same conversion path.

The observed `matchMmaV1AndDotOperandLayout` requires:

- MMA encoding version 1 and dot operand index 0 (A).
- `warpsPerCTA[1] == 1` and `vecSize == 1`.
- Non-FP32 elements and `kWidth == 32 / element_bit_width`.

Inspect `third_party/ppu/lib/TritonPPUGPUToLLVM/Utility.cpp` and
`ConvertLayoutOpToLLVM.cpp` for the actual revision. Inspect
`third_party/ppu/lib/TritonPPUGPUTransforms/AcceleratePPUMatmul.cpp` (or locate the
file by name if it moved) for warp assignment. In the observed implementation,
an eligible chain with M at least `16 * num_warps` can use an M-major
`[num_warps, 1]` warp distribution. This condition alone does not prove selection.

For a measured expensive chain, consider algebraically transposing its state or
intermediate so that the persistent accumulator becomes the next dot's A operand.
For example, changing a K-by-V state into a V-by-K state can align the value rows
with the warp distribution. A V tile of 64 and four warps is one tested pattern,
not a universal optimum. Compare task count and head/batch parallelism before
reducing the number of V tiles; small grids may need the prior orientation.

Trace the exact selected TTGIR layout conversions into final ISA. Useful observed
signatures include `v.mma.f32.f16.m16n16k16` for FP32-accumulating FP16 GEMM and
`v.mma.f16.f16.m16n16k16` in the FP16 fragment-repack path. BF16 has its own
instruction variants. Attribute repack instructions to their lowering context:
an FP16 repack MMA does not mean that the numerical state accumulation was changed
to FP16. Conversely, absence of that literal FP16 mnemonic does not diagnose BF16.
The ordinary output GEMM may already use this path even when the state GEMM does
not; inspect both instead of assuming a pipeline-wide deficiency.

## 2. Control fragment lifetime, not just tile size

Transposing a persistent state can enable the native chain yet make the kernel
slower because a large operand-B fragment is replicated across M warps. The
observed compiler produced heavy spilling for a direct full-K transformation.

A useful alternative is to partition the contraction dimension into native-sized
pieces, retain separate FP32 state blocks, and stream the B tiles:

```text
prediction = dot(cast(state0), transpose(keys0))
prediction = dot(cast(state1), transpose(keys1), acc=prediction)
...
state0 = dot(weighted_values, reload(keys0), acc=state0 * decay)
state1 = dot(weighted_values, reload(keys1), acc=state1 * decay)
```

Splitting K128 into K64 pieces is a tested instance. Reloading a short-lived B
tile can be cheaper than keeping the replicated full-K operand live. Confirm
actual lifetimes and generated loads; source-level reloads may be eliminated.
Passing the decayed state as dot's accumulator also avoids an unnecessary separate
FP32 product where supported. Reassociation changes rounding, so check output and
final state against the unchanged numerical contract and independent reference.

More registers or nonzero spill can coexist with a faster operator. Fewer
registers can coexist with a slower one. Evaluate the complete tradeoff: fragment
conversion, live state, reloads, Shared buffers, synchronization, grid size and
end-to-end time. Do not optimize resource metadata as a standalone score.

## 3. Native-tile algebra can shorten serial solves

For a strictly lower triangular n-by-n matrix L, `L**n == 0`. This permits an
exact finite series in real arithmetic:

```text
(I + L)^(-1) = I - L + L^2 - ... + (-L)^(n-1)
P = -L
R = I + P
repeat log2(n)-1 times, when n is a power of two:
    P = P @ P
    R = R + R @ P
```

For n=16, three doublings cover powers through 15. Native 16-by-16 diagonal
solves and blockwise off-diagonal composition can replace a long serial row
extraction/reduction chain. Use only when strict triangularity, signs and block
dependencies are established. This is a finite polynomial, not a convergent
infinite-series approximation; its floating-point evaluation still introduces
rounding and can perform more matrix work than the serial method.

For FP32 block products, IEEE lowering may use scalar FMA while `tf32x3` can use
Tensor Cell instructions. TF32x3 is a numerical choice, not an equivalent alias
for IEEE FP32. Keep task tolerances fixed, preserve required FP32 accumulation,
compare precision modes with identical geometry, and test slow decay or other
error-amplifying inputs. Do not replace FP32 products with FP16 merely to trigger
MMA. Eliminating a full global KKT scratch is a separate dataflow opportunity;
verify both the saved allocation/traffic and the new solve cost.

## 4. AIU descriptors must describe physical pitch

In the observed TLE AIU lowering, a tensor-pointer async load forwarded base,
shape, offsets and order into AIU but did not preserve arbitrary strides.
Inspect `third_party/ppu/spec_cpp/lib/Dialect/Triton/Transforms/RewriteTensorPointer.cpp`,
especially `rewriteAIULoadOp`, before relying on strided descriptors in that stack.

A head-interleaved tensor `[B,T,H,K]` has time pitch `H*K`, even if the requested
logical tile is T-by-K. Simply marking its usual `(T,K)` block pointer async can
therefore read incorrect elements. Where the installed lowering requires packed
descriptors, describe the physical row width `(T,H*K)`, strides `(H*K,1)`, and
put the head offset in the physical column coordinate. Keep requested block
width K (or its partition). Apply the same reasoning separately to values and
inverse matrices; their physical pitches may be `Hv*V` and `Hv*BT`.

If instead retaining a head-shifted base, prove the descriptor's maximum address
and boundary checks are valid for every requested tile; a correct pitch alone
does not justify out-of-bounds descriptors. Test nonzero heads, multiple batches,
tails and masks as applicable. A contiguous single-head test cannot expose this
error. Do not modify the installed compiler to make one operator pass.

`triton.experimental.tle.language.load(..., is_async=True)` generated AIU on the
observed PPU stack. Confirm `vmem.aiu.ld.tsm...` plus matching Shared consumption
and waits in the actual binary. Compare global-to-Shared instructions separately:
ordinary async copy is not AIU, and generic Shared loads are not proof of native
swizzle consumption. PLA-style `tsm.ld.swzl...` requires a compatible layout.

AIU can increase staging space and extend the critical path even while reducing
register use. Test it at an identified transfer edge, preserving other factors.
Reject it when the whole operator regresses; do not enable AIU everywhere merely
because the ISA now contains a specialized instruction.

## 5. Explicit Shared APIs and compiler transformations are separate capabilities

The observed PPU TLE whitelist supported `gpu.alloc` and `gpu.local_ptr`, but not
`gpu.buffered_tensor.load/store`. Supported access was a `gpu.local_ptr(buffer)`
view used by normal `tl.load` and `tl.store`, with correct producer/consumer
synchronization. Check `get_supported_primitives('ppu')` or the installed
`tle_supported.py` for the active version. Do not patch the whitelist or assume
that an API's presence implies backend support.

For that API revision:

- Pass a real element dtype such as `X.dtype.element_ty` to `gpu.alloc`; a wrapped
  dtype constexpr parameter failed its frontend type check.
- Request a PPU-compatible Shared encoding, not the default NVIDIA MMA encoding.
  The observed explicit constructor was
  `swizzled_shared_layout(vectorSize, perPhase, maxPhase, order, numCTAs,
  numCTAsPerCGA, numCTASplit, numCTAOrder)`, used with
  `nv_mma_shared_layout=False`. These are version-specific API details; choose
  layout parameters for the consumer rather than copying them blindly.
- A successful small Shared round-trip proves API lowering and data access, not
  an efficient fused GEMM or legal transformation inside its persistent loop.

In the observed fusion, the compiler promoted a global-load/Shared-store pair to
async copy and then failed in `TritonTleDowngradeInvalidAsyncCopy` with
`cannot downgrade invalid async copy with non-trivial async token users`.
Inspect the triggering load and transformation, not only the final pass name.
The observed `TleOptimizeLocalPointerAsyncStores.cpp` excludes volatile loads;
an explicit synchronous `tl.load(..., volatile=True)` avoided that promotion in
the experiment. This is a narrow compiler workaround with possible performance
cost, not an optimization recipe or a replacement for barriers. Preserve the
reproducer and benchmark any workaround before retaining it.

## 6. Fusion and chunk length must be judged as a full dataflow

An internal chunk length of 32 can reduce local quadratic products relative to
64 while doubling serial chunk iterations. Fusing output with state can remove
global state snapshots and updated-value intermediates, but adds live Q/K/V
fragments, Shared scratch, conversions and synchronization. Removing bytes or a
launch does not establish a net gain. Compare the complete public invocation,
including final-state production and reference conversions in their original
timing scopes. Keep a fallback selected by real algorithmic and resource regimes,
not a whitelist of public benchmark cases.

For a fair comparison with a native reference, inspect the exact launched binary
specialization. Its context partitioning, state correction and auxiliary kernels
may differ from its visible main GEMM. Include these costs and dependencies;
comparing only the main kernel or grid dimensions can misidentify the bottleneck.

Validate every newly selected dispatch path with independent numerical cases;
small-head tests can pass while never exercising a new large-head kernel. Include
final state, multiple seeds and relevant layout/boundary regimes. Use same-run
reference timings and interleaved parent/candidate processes for final reporting.
Distinguish an aggregate win from winning every shape, and distinguish a marginal
aggregate per-shape win from winning that shape in each paired process. Keep raw
samples and static artifacts; do not infer measured DRAM bytes, occupancy, bank
conflicts or stall percentages from their absence.

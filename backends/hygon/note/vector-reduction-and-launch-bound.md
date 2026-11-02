---
id: hygon-vector-reduction-launch-bound
kind: guide
status: provisional
device: BW
arch: gfx936
scope: "Diagnose reduction-bound, memory-bound, and launch-bound Triton kernels and choose the experiment that tests each"
revalidate_after: 2027-01-31
---

# Reduction, memory, and launch-bound kernels

`mmac-fma-path-selection.md` covers matrix-shaped work. Most remaining operators
are one of three other shapes: a reduction (softmax, norms, sums, attention
normalization), a streaming/memory-bound pass (elementwise, gather/scatter,
layout conversion), or a launch-bound pipeline (many small kernels, or a grid
too small to fill the device).

This note is **provisional**: the classes and experiments below are derived from
the validated architecture facts in
[gfx936-validation.md](gfx936-validation.md) and from practice on this
toolchain, but they are not yet backed by a recorded result note. Treat each as a
hypothesis to measure, and replace the guidance with a confirmed note once one
of these experiments is recorded.

## 1. Classify before choosing

Compute arithmetic intensity from the operator's own numbers:

```text
arithmetic_intensity (FLOP/byte) = measured_FLOPs / bytes_moved
```

Compare it with the machine's balance point, derived from the validated
calibration anchors (BF16 GEMM 305 TFLOPS, D2D copy 1.35 TB/s):

```text
BF16 balance point ≈ 305e12 / 1.35e12 ≈ 226 FLOP/byte
FP32 balance point ≈  55e12 / 1.35e12 ≈  41 FLOP/byte
```

These are anchors from PyTorch GEMM and a copy, not vendor peaks — use them to
choose an experiment, not as an acceptance target. A kernel at 10 FLOP/byte is
memory-bound on this device no matter how good its math is.

Read the class off the profile, not the source:

| Observation | Class |
|---|---|
| Achieved bandwidth near the copy anchor, low FLOPs | memory-bound |
| Long serial accumulation per output, low occupancy, few waves | reduction-bound |
| Kernel duration small and many launches, or a short tail wave | launch-bound |
| All three plausible | fix the largest measured term first, then re-measure |

## 2. Reduction-bound

Symptoms in the trace: one CTA per row, a serial loop accumulating a single
lane-per-element vector, low waves per CU, or a tail kernel much shorter than the
main one but still charging full launch latency.

Experiments, in the order that usually pays:

1. **Split the row across the block.** Reduce within each lane's register
   accumulator, then across lanes, then across waves through LDS. On Wave64,
   keep the block size a multiple of 64 so no lane sits idle through the
   reduction.
2. **Two-stage reduction** when a row does not fit one block's reach: stage one
   writes partials, stage two combines. This costs a global round trip, so it
   only wins when the row is long enough that one block serializes.
3. **Atomics only for few contributors.** Per-element atomics on a long row
   serialize in L2; use them when the number of partials per output is small and
   the output is large.
4. **More rows per CTA** when the row is short: amortize the reduction tree over
   several rows, provided each row keeps its own mask and scale.

Deciding evidence: kernel time plus the generated reduction tree (how many
serial steps remain). A reduction win must be checked against the output
tolerance, because changing the tree changes the summation order.

## 3. Memory-bound

Symptoms: achieved bandwidth near the measured copy anchor, or a kernel whose
time scales with bytes and barely moves when math is removed.

Experiments:

1. **Coalesce and vectorize.** Load contiguous 128-bit chunks (`buffer_load` of
   four dwords in the generated code); misaligned or strided access turns one
   transaction into several.
2. **Do not stage through LDS when each value is used once.** Staging costs a
   global-to-VGPR load, an LDS write, a barrier, and an LDS read. If the tile has
   no reuse, the barrier is pure loss.
3. **Use the 8 MiB L2 budget deliberately.** A working set that fits L2 amortizes
   the second pass; one that exceeds it pays HBM twice. Measure the two cases.
4. **Hoist masking and page-table lookups.** A mask evaluated per element and a
   page-table load per tile are both address traffic that competes with data.
5. **Prefer a layout that makes the fast axis contiguous** at the operator
   boundary, instead of transposing inside the kernel.

Deciding evidence: bytes moved per shape (from the kernel's own tiling), kernel
time, and the memory-pipeline counters when a counter capture is available.
Without counters, a layout A/B with identical math is the evidence.

## 4. Launch-bound

Symptoms: `hipprof` shows gaps between kernels, or the trace shows a grid whose
last wave is nearly empty, or per-kernel time is small compared with end-to-end
operator time.

Experiments:

1. **Merge stages** that share data or a synchronization point, so one launch and
   one round trip replace two.
2. **Round the grid to full waves.** With 80 CUs, a grid of 100 blocks pays a
   second wave for 20 blocks. Either raise per-block work so 80 blocks suffice,
   or reshape the decomposition.
3. **Multi-row programs**: give each program several rows so the grid shrinks and
   each CTA has enough work to hide its own prologue. See
   [examples/triton-multirow-program.md](examples/triton-multirow-program.md).
4. **Raise per-thread work before raising `num_warps`.** More warps add register
   and LDS pressure; per-thread vectorization usually costs less.

Deciding evidence: a timeline showing the gap or the tail wave, plus end-to-end
operator time. A kernel-only win that leaves the same gap is not a win.

## 5. Evidence protocol for these classes

For every candidate in this note:

- correctness against the reference, including the fallback and one boundary
  shape;
- the same-shape A/B latency table with identical warmup, repeat, and device
  mask;
- one line explaining which measured term moved (bytes, serial steps, or
  launches) and why;
- the resource columns (VGPR/SGPR/LDS/waves) when the change affects occupancy.

Falsify explicitly: state what result would have rejected the hypothesis. If the
trace does not show the term you intended to reduce, the hypothesis was wrong
even if the number improved.

## 6. Open questions

- Whether an LDS-staged double-buffered reduction beats a register-only tree on
  this device, and at which row length.
- Whether `cache_modifier` variants change streaming bandwidth on this toolchain.
- The grid-size threshold at which launch overhead actually dominates operator
  time; the ~20 µs figure used above is a rule of thumb, not a measurement.
- Whether the L2 behavior for paged attention differs enough from dense attention
  to need a separate note.

# S60-TRITON-013: operand preparation before a matrix op: redundant masks and narrow strided transfers

## Generalization gate

- Reusable mechanism, interface or bounded backend limitation: two operand
  preparation patterns that cost real time on S60/GCU300 without changing any
  arithmetic result:
  1. re-applying a causal/validity mask with `tl.where` to an operand that the
     producer already stored as exact zeros, boolean-masked or masked **for the
     complete consumer tile**. If the producer writes only diagonal or
     lower-triangular sub-blocks and leaves upper off-diagonal blocks unwritten,
     the consumer mask is still required.
  2. reading a `[BK, BT]` operand through a key-major block pointer whose element
     stride is the packed head stride, instead of loading the same elements
     time-major and transposing in-kernel.
- Another operator/workload class to which it applies: chunked attention and
  linear-attention kernels that publish a masked `[BT, BT]` intra-chunk score
  matrix and re-load it in a second kernel; recurrent/chunk state kernels that
  multiply a transposed key tile by a value tile.
- Independent probe, mathematical argument or cross-workload evidence: both
  patterns were changed one at a time inside one operator and measured with
  alternating A/B rounds; the first is exact for the measured `NC=1` target
  (the producer's zeros are the mask's zeros) and conditional as stated below;
  the second moves only the transfer shape of identical elements.
- Conditions that invalidate the claim: an operand whose producer does not
  guarantee zeros outside the valid region (then the mask is required), row/column
  identity or tie semantics encoded in the masked lanes, a consumer layout where
  the in-kernel transpose cannot fold into the matrix operand, or a transposed read
  that is already contiguous because the last dimension is the strided one.

## Scope and status

- Observation date and review date: observed 2026-11-17, reviewed 2026-11-17.
- Device/backend/architecture: Enflame S60 / GCU300, GCU backend.
- Framework, compiler, SDK/driver and device-library versions: Enflame Triton 3.6.0,
  `torch_gcu`; driver/SDK not established by this card.
- Language, dtype/accumulation, M/N/K, strides/layout, launch options: Triton, BF16
  operands, FP32 accumulation, `BT=64`, `BK=128`, packed `B,T,H,D` layout with head
  stride `H*D`; one execution group, `num_stages=1`; the FP32/FP32 dot keeps the
  backend's default `ieee` precision.
- Shape class, inference/training mode and numerical semantics: dense causal
  chunked GLA forward; the masked operand is a strictly causal `[BT, BT]` block
  with `i >= j` ownership.
- Status: performance-observed; the transfer/select lowering cause is inferred;
  the transpose rewrite is numerically validated; mask-removal exactness is
  conditional on the producer writing the complete `[BT,BT]` block (`NC=1` here).

## Observed facts

For the second kernel of the pair (a `[BT, BT]` FP32 score block loaded from a
workspace buffer, masked with `i >= j`, cast to BF16 and used as a matrix operand)
removing the mask lowered that kernel's median by about 10% and the full contracted
call by about 2%, with identical numerics. The producing kernel had already stored
exact zeros above the diagonal with the same predicate, so the mask was a no-op
whose only effect was on operand materialization.

Pre-commit audit correction (2026-11-17): this exactness is conditional. The
contracted BF16 target had `BT=BC=64` (`NC=1`), so the producer wrote the complete
`[BT,BT]` diagonal block. FP16 uses `BC=16`, `BT=64` (`NC=4`), and the Enflame GDN2
native path uses `BC=16`, `BT=64`; both leave upper off-diagonal blocks unwritten.
Forcing the skip on those paths changed outputs (max abs diff 1.9055 for FP16
`T=63`, 9.6919 for a small GDN2 native case). The consumer now dispatches
`NEEDS_A_MASK` and keeps the original mask for those callers; only the `NC=1`
GLA path skips it.

For the state kernel, the two operands `k` and the cumulative gate were read
through a key-major block pointer `(K, T)` with strides `(1, H*K)`, i.e. `BK` narrow
segments per transfer, and the gate factor was broadcast along the segment axis.
Reading the same elements time-major `(T, K)` with contiguous last dimension,
broadcasting along the transposed axis, and transposing before the dot lowered that
kernel's median by about 14% and the full call by about 3.8%, with numerically
identical state and final-state results.

Related negative evidence in the same operator: deleting a duplicated staged tile
load changed nothing, and an explicit `is_async` load spelling changed nothing,
consistent with [S60-TRITON-002](triton-compute-memory-paths.md)'s guidance that
source-level load spelling and added transfers need not change generated code or
the critical path.

Single-point replay from `e001aea` (2026-11-19, four alternating pairs)
isolated both patterns: the producer-aware mask skip alone lowered the full call
by 2.30% (median; geomean 0.97755) and the output kernel by 9.38%; at a fixed
BK=128/BV=128 configuration the full-call gain was 1.86%, so about 0.4
percentage points of the deployed effect came from the autotuner switching the
output kernel from BK=128 to 256 after the mask was removed. The time-major
`k`/`gk` load plus in-kernel transpose alone lowered the full call by 3.66%
(geomean 0.96306) and the local-state kernel by 14.75%, with `shared` dropping
from 65536 to 32768.

## Mechanism and candidate hypothesis

Hypothesis for (1): the mask forces the loaded operand to be materialized as a
selected FP32 tile before the output-precision conversion, so the load -> convert
path gains an extra vector pass and a wider live value. Expected change: the
consumer kernel's median drops while every output stays bit-identical because the
masked lanes are already zero. Falsified by an unchanged kernel median or by any
differing element.

Hypothesis for (2): a key-major block pointer over a packed `[T, H, D]` tensor
issues `BK` separate narrow segments, while the time-major tile is one regular
slice and the transpose folds into the matrix operand layout. Expected change: the
state kernel's median drops with unchanged output. Falsified by an unchanged or
worse median, or by a numerical difference.

Inferred, not proven: the extra cost in (1) is attributed to tile materialization
or a blocked load->convert fusion; the cost in (2) is attributed to per-segment
transfer overhead. Neither was confirmed with generated code or counters in this
run (profiling and low-level artifacts were disabled by the task contract), so the
mechanism level is performance-observed only.

## Reproduction and validation

Before removing a consumer-side mask, prove that *every* producer stores exact
zeros in the masked region, including alternate entry points that share the
consumer kernel, and that the zeros are stored rather than masked only at load
time. Then compare outputs elementwise, not only through a tolerance gate.

For transfer orientation, keep the transposed and upright forms algebraically
identical (same element set, same broadcast axis, same accumulator and output
type), verify the dot operand order, and check tail blocks where the upright tile
and the key-major tile carry different masks. Measure the kernel and the full
contracted call with alternating A/B rounds and retain raw samples.

## Limits, failures and revalidation

- Both mechanisms are per-kernel: the same operator's other kernels did not
  benefit from analogous edits, so do not generalize to "remove masks" or
  "always transpose in-kernel".
- A transpose that cannot fold into the matrix operand would add a real layout
  conversion; that case was not measured and may regress.
- A consumer shared by several operators is only safe if every producer is
  audited. The GLA BF16 `NC=1` producer writes the complete block; FP16 `NC>1`
  and the Enflame GDN2 native producer leave upper off-diagonal blocks unwritten,
  so the final consumer keeps the mask for them. Other callers were not exercised
  in this run.
- The measured magnitudes belong to one operator and tile size; a different
  `BT/BK`, dtype or store volume needs a fresh probe.
- No utilization, bandwidth or overlap claim is made; the device's DRAM counters
  are known-unavailable per [S60-EVIDENCE-002](efsmi-gpm-monitor.md).

- See [S60-TRITON-014](triton-shared-transpose-materialization.md) for the
  shared-memory accounting of the key-major FP32 gate tile: the time-major
  form removes a 32K+32K shared transpose pair.
- Removing an operand-preparation cost can change the autotuner's selected
  configuration (here the output kernel moved from BK=128 to 256). Report the
  deployed auto-config delta and a fixed-config control; only the fixed-config
  delta isolates the semantic change. In this operator the config switch was
  worth about 0.4 percentage points, inside the four-pair batch volatility.

## Sources

- `gla-r2-benchmark-ledger`, `gla-r2-candidate-ledger`, `gla-r2-final-report`,
  `gla-r2-profile-baseline`, `gla-r2-profile-final` in sources.json.
- `gla-r2-attr-report`, `gla-r2-attr-manifest`, `gla-r2-attr-ab-summary`,
  `gla-r2-attr-profile-summary` in sources.json (2026-11-19 single-point
  attribution replay from `e001aea`).
- Prior contrary/neutral evidence: [S60-TRITON-002](triton-compute-memory-paths.md)
  (async spelling, prefetch and transfer-count neutrality).

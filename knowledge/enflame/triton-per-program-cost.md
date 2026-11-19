# S60-TRITON-012: per-program cost can dominate small-tile kernels

## Generalization gate

- Reusable mechanism, interface or bounded backend limitation: on S60/GCU300, a
  Triton kernel whose grid is one program per small output tile can be limited by
  a per-program cost that is nearly independent of the tile's data volume. In
  that regime, assigning several consecutive work items to one program
  (amortizing the prologue, schedule and transfer waits) reduces the kernel time
  even though the arithmetic and bytes are unchanged.
- Another operator/workload class to which it applies: chunked linear-attention
  and gated linear-attention forward kernels whose grid is
  `chunks x blocks x batch-heads` with 64x64 or 64x128 tiles, and any operator
  that launches one short-lived program per tile of a few tens of KB.
- Independent probe, mathematical argument or cross-workload evidence: the
  same-program-count cost regression holds for three different kernels of one
  operator at two sequence lengths, and the batching change was measured on the
  full contracted call with alternating A/B rounds (see Sources).
- Conditions that invalidate the claim: a kernel whose per-program work already
  dominates (long serial loops, large stores, or per-program traffic close to the
  device's steady-state bandwidth), a grid that must stay one-program-per-tile for
  a documented reduction/ordering reason, or a batch size large enough to inflate
  the unrolled body or staged buffers. Padding or extra guarded iterations that
  add real work also invalidate the accounting.

## Scope and status

- Observation date and review date: observed 2026-11-17, reviewed 2026-11-17.
- Device/backend/architecture: Enflame S60 / GCU300, one visible card, GCU backend.
- Framework, compiler, SDK/driver and device-library versions: Enflame Triton 3.6.0,
  `torch_gcu`; driver/SDK and device-library versions not established by this card.
  Compiler metadata for the measured specializations: `vector_length=512`,
  `default_dot_input_precision: ieee`.
- Language, dtype/accumulation, M/N/K, strides/layout, launch options: Triton, BF16
  operands with FP32 accumulation, `BT=BC=64`, `BK=128`, contiguous `B,T,H,D`
  layout, one execution group (`num_warps=1`), `num_stages=1`.
- Shape class, inference/training mode and numerical semantics: dense causal
  chunked GLA forward, inference path, FP32 intra/intermediate buffers with BF16
  publication.
- Status: performance-observed (mechanism), compiler path not attributed.

## Observed facts

Two unrelated specialization sizes of the same kernels were measured:
`(B,T,H,D)=(2,16384,16,128)` and `(4,2048,16,128)`. Program counts differ by 4x
while the per-program tile is identical. For the intra-chunk attention kernel, the
grid-region kernels, and the state kernels alike, the measured kernel time was
linear in program count with a near-zero intercept (about 0.5-0.6 microseconds per
program, intercept on the order of 0.02 ms). The per-program cost was insensitive
to the number of staged payload tiles: removing one of four identical staged
transfers changed the kernel median by 0.02%, and widening the execution group
(`num_warps` 1/2/4 in an autotune list) changed the full call by 0.3%.

Giving each program two consecutive work items (a loop over two adjacent chunk
indices, one warp, unchanged tile work) reduced that kernel's median by 7.9% with
the full-call delta matching the kernel delta. A four-item batch removed the gain
entirely. Applying the same two-item batch to a kernel with a 64 KB FP32 store per
item produced no kernel change, and to a kernel with two dots and five transfers
per item no resolvable kernel change.

Single-point replay from the fixed first-round baseline `e001aea` (2026-11-19,
BF16 `(4,2048,16,128)`, four alternating pairs): batching two chunks in the
intra-A kernel lowered the full call by 2.05% (median; geomean 0.97890) and that
kernel by 7.85%, with the grid halved from 32 to 16. The same batching applied
alone to the output kernel (two dots and five transfers per item) did not help:
the full call was 0.61% slower (geomean 1.00743) and the output kernel was 3.06%
slower, even though the grid halved from 32 to 16. Applying both output changes
(batching plus the producer-aware mask skip) gave 1.99%, statistically
indistinguishable from the mask skip alone (2.30%); the small batching penalty
disappeared but no additive batching gain appeared.

## Mechanism and candidate hypothesis

Hypothesis: for a small tile the program prologue (scalar address setup,
schedule/slot acquisition) plus the unavoidable transfer-wait latency is a fixed
cost that is serialized per program; because the device overlaps many programs,
the fixed cost still appears as roughly constant time per program, so the kernel's
total is `programs x fixed_cost + work`.

Expected change: batching `n` consecutive items halves/quarters the program count
and should reduce the kernel time by the fixed share, with no arithmetic change and
no change to per-item results. Falsified by an unchanged kernel median (fixed share
is zero or already overlapped) or by a regression at larger `n` (footprint growth).

Unverified/inferred: the physical source of the fixed cost (launch/schedule slot
acquisition versus transfer-wait serialization) was not attributed with counters;
the ISA-level cause of the batch-4 regression (unrolled body size, staging growth)
was not inspected. Batching helped only where the per-item work was small, which is
consistent with the fixed-share model but does not prove it.

## Reproduction and validation

Compare the same operator at two sequence lengths that scale the grid while holding
the tile constant; fit time against program count. If the intercept is small and
the slope is constant per program, test batching. Keep the tile arithmetic, the
causal/validity masks and the output ownership per item unchanged, gate batching on
branch-free structural conditions (dense vs ragged input, non-remapped grid), and
guard the tail item with an explicit index test in both the unrolled and the
remaining-iteration case. Validate ragged, tail and alternate-layout inputs
separately, then measure the full contracted call with alternating A/B rounds and
retain every sample; a kernel-level median change is not sufficient on its own.
Do not derive the batch size from a benchmark shape: choose it per kernel and
verify the next smaller and larger value.

## Limits, failures and revalidation

- Batching more than two items was neutral in the measured kernel; do not assume
  monotonic improvement and do not present a batch sweep as the result.
- A kernel dominated by a per-item 64 KB FP32 store, or by multiple dots, did not
  benefit in the same operator, so the mechanism is not a general rule for
  "fewer programs is faster".
- Widening the execution group is not a substitute; it was neutral here.
- Any change of tile shape, dtype, store volume or compiler version requires a
  fresh measurement; the fixed cost may already be hidden at another tile size.
- Ragged and grid-remapped paths were deliberately left unbatched, so no claim is
  made for them.

- Single-point replay changes the accounting: the historical cumulative c006
  (built on c003) showed 0.9736 vs baseline, but the same output batching built
  directly on `e001aea` showed 1.00743 (0.61% slower). A component's cumulative
  delta is not its isolated effect. Do not project a batching gain onto a kernel
  whose per-item work (two dots, five transfers) already dominates; re-measure
  from the common baseline and keep the interaction with other changes explicit.

## Sources

- `gla-r2-benchmark-ledger`, `gla-r2-candidate-ledger`, `gla-r2-final-report`,
  `gla-r2-profile-baseline`, `gla-r2-profile-final` in sources.json
  (task-side artifacts of the FlagAttention chunk_gla second S60 round).
- `gla-r2-attr-report`, `gla-r2-attr-manifest`, `gla-r2-attr-ab-summary`,
  `gla-r2-attr-profile-summary` in sources.json (2026-11-19 single-point
  attribution replay from `e001aea`).

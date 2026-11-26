# S60-TRITON-014: key-major FP32 block pointers materialize shared transpose buffers

## Generalization gate

- Reusable mechanism, interface or bounded backend limitation: on S60/GCU300
  with Triton 3.6.0, a key-major `[BK,BT]` block-pointer load of an **FP32**
  tile that is consumed by an elementwise broadcast (for example a per-key
  gate factor subtracted from a `[BK,BT]` gate tile) is lowered through a
  shared-memory transpose pair: `memref<BKxBTxf32, 2>` plus
  `memref<BTxBKxf32, 2>`. At the measured `BK=128`, `BT=64` the pair is
  128x64 FP32 elements each, i.e. **32 KiB + 32 KiB**. The equivalent
  time-major `[BT,BK]` load stays in the non-shared/local layout for the
  elementwise op; only the BF16 tile that feeds `tl.dot` needs a 16 KiB +
  16 KiB shared transpose (64x128 BF16 elements each). The sizes scale with the
  tile: at `BK=BT=128` a single FP32 buffer would be 64 KiB, so the pair must be
  recomputed for the tile actually used, not copied from this case.
- Another operator/workload class to which it applies: chunked linear-attention
  or recurrent state kernels that read an FP32 cumulative gate / prefix tile
  through a key-major block pointer and combine it with a time-major value
  tile; any kernel that chooses between a `[K,T]`-major and `[T,K]`-major
  block-pointer read for an elementwise operand.
- Independent probe, mathematical argument or cross-workload evidence: the
  two source variants differ only in block-pointer shape/stride/order and the
  explicit `tl.trans` before the dot; TTIR, GCUIR, LLIR and fatbin were captured
  from the same worktree and the same specialization. The GCUIR shared-allocation
  ledger of the time-major variant equals the reported `shared` metadata exactly
  (2 x 16 KiB = 32 KiB). For the key-major variant the ledger lists
  16K + 16K BF16 and 32K + 32K FP32, i.e. 96 KiB of allocations against
  `shared = 65536`; the ledger sum is therefore **not** a peak, and the buffers
  are not all live at once. The FP32 transpose pair alone accounts for the
  reported 64 KiB. Which buffers overlap, and where that 64 KiB physically
  lands, were not established; see
  [S60-COMPILER-001](compiler-artifact-identity.md).
- Conditions that invalidate the claim: an FP32 tile that is not consumed by a
  broadcast elementwise op; a tile small enough that the transpose pair is
  elided or folded; a layout where the key-major read is already contiguous in
  the consumer layout; a different `BT/BK` where the backend chooses another
  conversion; or a compiler version whose layout-conversion pass folds the
  transpose into the elementwise op.

## Scope and status

- Observation date and review date: observed 2026-11-20, reviewed 2026-11-20;
  revised 2026-11-26 (tile arithmetic, allocation ledger versus metadata).
- Device/backend/architecture: Enflame S60 / GCU300, GCU backend, one visible
  card.
- Framework, compiler, SDK/driver and device-library versions: Enflame Triton
  3.6.0 (FlagTree), `torch_gcu`; driver/SDK and device-library versions not
  established by this card. Compiler metadata: `vector_length=512`,
  `default_dot_input_precision=ieee`, `num_warps=1`, `num_stages=1`.
- Language, dtype/accumulation, M/N/K, strides/layout, launch options:
  Triton, BF16 `k`/`v`, FP32 `gk`/`local_state`, FP32 accumulation; `BT=64`,
  `BK=128`, `BV=128`; packed `B,T,H,D` with head stride `H*K`; one execution
  group.
- Shape class, inference/training mode and numerical semantics: dense chunked
  GLA forward, inference path, `(B,T,H,D)=(4,2048,16,128)`; state update
  `local = (k * exp2(g_last - gk))^T @ v`.
- Status: **performance-observed** for the kernel/full-call gain,
  **compiler-observed** for the shared-transpose lowering; the physical reason
  for the backend's choice is inferred, not proven with pass-level debugging.

## Observed facts

Source locations: baseline `chunk_h.py:181-215`, time-major variant
`chunk_h.py:184-218` in the task workspace.

TTIR (baseline): `k`/`gk` are `[128,64]` key-major block pointers
(`order=[0,1]`, strides `[1,H*K]`); the gate factor is broadcasts on axis 1;
`tt.dot` takes the `[128,64]` BF16 `k` directly.

TTIR (time-major): `k`/`gk` are `[64,128]` time-major block pointers
(`order=[1,0]`, strides `[H*K,1]`); the gate factor broadcasts on axis 0; a
`tt.trans` converts the BF16 `k` to `[128,64]` immediately before `tt.dot`.

GCUIR (baseline) has two `memref_ext.transpose_start` operations:

```text
k : memref<64x128xbf16, 2>  -> memref<128x64xbf16, 2>   (16 KiB + 16 KiB)
gk: memref<64x128xf32, 2>   -> memref<128x64xf32, 2>    (32 KiB + 32 KiB)
```

GCUIR (time-major) has one:

```text
k : memref<64x128xbf16, 2>  -> memref<128x64xbf16, 2>   (16 KiB + 16 KiB)
gkt: memref<64x128xf32>      (non-shared, elementwise only)
```

Counts for the local-state kernel:

| Version | shared allocs (address space 2) | metadata `shared` | transpose | dma_wait | barrier | alloc |
|---|---|---:|---:|---:|---:|---:|
| key-major | 16K + 16K bf16 + 32K + 32K f32 | 65536 | 2 | 18 | 4 | 14 |
| time-major | 16K + 16K bf16 | 32768 | 1 | 15 | 2 | 11 |

The two rows above are **allocation ledgers**, not residency. The key-major
ledger totals 96 KiB while the compiled metadata reports `shared = 65536`; the
time-major ledger totals 32 KiB, which happens to equal its metadata value.
Reading either row as a peak is an inference the artifacts do not support: the
allocation sizes alone say nothing about which buffers are live together, and
the mapping from the metadata field to the physically reserved shared region is
unresolved on this backend. The only per-variant quantities this card uses are
the ledger contents, the metadata field, and the counted transpose/wait/barrier
operations.

LLIR: 836 → 817 lines, 137 → 131 calls, 97 → 91 loads, 19 → 16 waits for the
local-state kernel. The time-major kernel median is 1.1105 → 0.9467 ms
(-14.75%) in the four-pair batch.

Counts of generated operations are static. Several of the same counters can
match while the timing differs, because a count says nothing about latency,
overlap or placement; an unchanged-code null control in this environment showed
material timing differences between identical configurations
([S60-EVIDENCE-001](evidence-and-measurement.md)). A three-way comparison of this
operator in which the primary resource counters were equal while the timings
differed was reported, but its artifacts were not available for this review, so
this card neither uses it as evidence nor states its cause.

A related control: the state-scan kernel's Triton cache hash changed
(`f847a01a…` → `776d68c5…`), but its GCUIR (`90985b64…`), LLIR
(`08c0493c…`) and fatbin (`0f121767…`) are byte-identical; the TTIR differs
only in source-line numbers (230 → 233) and SSA/`#loc` numbering. A Triton
hash change alone does not establish a device-code change.

## Mechanism and candidate hypothesis

Hypothesis: the backend's layout-conversion lowering does not fold a key-major
FP32 block-pointer tile into the row/column-broadcast elementwise op; it
materializes the tile through a shared transpose pair. The time-major load
matches the elementwise access order, so the FP32 tile stays in the local
layout; only the BF16 tile used as a dot operand needs an explicit transpose,
and that transpose is smaller (16 KiB vs 32 KiB) and does not need the FP32
pair.

Expected change: switching the FP32 gate tile to time-major removes the 32K +
32K shared transpose, lowers `shared` from 64 KiB to 32 KiB, and removes the
associated waits/barriers while keeping the dot operands and arithmetic
identical. Falsified by an unchanged GCUIR transpose count/shared size, or by
a numerical difference.

Inferred, not proven: whether the shared pair comes from a specific layout
conversion pass or from the block-pointer lowering's staging choice; the exact
liveness interval of each buffer. No pass-level dump or hardware counter was
collected.

## Reproduction and validation

Capture TTIR/TTGIR/GCUIR/LLIR/fatbin for both variants from the same worktree
(`runs/attribution/capture_ir.py` in the task workspace), then compare:

1. TTIR block-pointer shape/stride/order and `tt.trans` count.
2. GCUIR `memref_ext.transpose_start` count and the address-space-2 memref
   sizes. Record the ledger and the metadata `shared` as two separate fields;
   do not sum the ledger and call the total a peak. The two agree only when the
   buffers happen to be exactly the live set.
3. LLIR/fatbin hashes and static instruction counts.
4. The full-call A/B and the target-kernel median with alternating order.
5. If the gain depends on the autotuner's choice, a fixed-config control, so the
   semantic change and the configuration switch are reported separately.

A numerical check that compares variant A against variant B establishes
**consistency**, not accuracy: both could be equally wrong. An independent
high-precision reference, a repository regression suite and a version-to-version
comparison are three different results; do not report one as another.

Validate a second shape/dtype (different `BT/BK`, FP16 or non-square K/V)
before generalizing; keep the arithmetic, masks and output ownership
unchanged. For another workload, check whether the FP32 tile is actually used
as an elementwise operand and whether the key-major read is required by the
surrounding code.

## Limits, failures and revalidation

- Measured on one shape (`BT=64`, `BK=BV=128`, BF16/FP32) and one compiler
  version. A different tile size or compiler may fold the transpose or choose a
  different conversion.
- The exact split of the 14.75% gain among fewer transposes, 3 fewer
  `dma_wait`, 2 fewer barriers and the smaller shared footprint was not
  ablated; the allocation ledger is exact, the time attribution is aggregate.
  Do not restate the total as "the transpose removal was worth 14.75%".
- The change measured here is a **kernel source** change. It is not a compiler
  modification: no pass, rule or toolchain file was altered. If an operator
  cannot express the time-major read in source, the remaining option is a
  compiler rule change, which needs its own task, build and A/B
  ([../../docs/enflame-compiler-diagnostics.md](../../docs/enflame-compiler-diagnostics.md)).
- The four-pair batch is a measurement repetition policy, not a launch-batching
  or grid change. Per-program count, per-launch count and host submission
  batches are distinct quantities; see
  [S60-TRITON-012](triton-per-program-cost.md) for a case where the program count
  itself was the variable.
- The `.ttgir` file produced by this toolchain was byte-identical to the
  `.ttir` file; use GCUIR/LLIR/fatbin for the backend-level comparison.
- No hardware counters; no utilization, bandwidth or overlap claim.
- A Triton cache hash change is not evidence of device-code change; compare
  the compiled artifacts. The state-scan control is an example of a
  hash-only change.
- The time-major variant's benefit was positive in the measured context; do
  not project it onto a kernel whose key-major read is already coalesced or
  whose FP32 tile is not elementwise-consumed.

## Sources

- `gla-r2-attr-compiler-doc`, `gla-r2-attr-compiler-ir-baseline`,
  `gla-r2-attr-compiler-ir-c009` in sources.json (2026-11-20 compiler
  attribution; TTIR/GCUIR/LLIR/fatbin captures, shared-memory accounting).
- `gla-r2-attr-ab-summary`, `gla-r2-attr-profile-summary` (four-pair A/B and
  kernel medians).
- `gla-r2-attr-c006-direct`, `gla-r2-attr-c006-correctness` (c006 retention
  control in the same batch).
- `s60-compile-chain-v2`, `s60-compile-chain-errata` for the unresolved mapping
  from compiler-reported `shared` to the physically reserved region.

## Revision history

- 2026-11-26: the first revision stated "32 KiB + 32 KiB for `BK=BT=128`". The
  measured configuration was `BK=128`, `BT=64`, and 128x64 FP32 elements are
  32768 B. A `BK=BT=128` tile would be 64 KiB per buffer. The size is now derived
  from the tile instead of asserted for a generic one.
- 2026-11-26: the first revision described the key-major 64 KiB as "the peak" and
  treated the allocation ledger as matching the metadata. The ledger totals
  96 KiB against `shared = 65536`, so it cannot be a peak; the FP32 pair alone
  is 64 KiB and the overlap between buffers is unknown. Ledger, metadata,
  liveness and physical placement are now four separate statements.

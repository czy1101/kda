---
id: do-not-enable-unconditional-next-chunk-w-lookahead-in-gdn2-h-ker
kind: rejection
status: rejected
device: BW
arch: gfx936
operator: GDN2 H-kernel (Triton)
scope: One 64-column W tile is loaded for chunk i_t+1 after forming the current residual
  and carried through fused O, gate scaling, and current H update; tested on the 24-case
  FP16/BF16 matrix.
rejected_on: '2026-11-03'
retry_when: Retry only with a separate long-sequence kernel variant that leaves the
  default JIT kernel and its tuner key unchanged, then run paired warmed baseline/candidate
  measurements and a target H-kernel rocprofv2 trace/PMC to verify load overlap and
  no spill/occupancy regression.
---

# Do not enable unconditional next-chunk W lookahead in GDN2 H-kernel

> Status: rejected on `2026-11-03`. This is negative knowledge: it records a measured
> failure, not an optimization result. Do not cite it as a win.

## Attempted

- Device/architecture: `BW` / `gfx936`
- Operator: `GDN2 H-kernel (Triton)`
- Scope: One 64-column W tile is loaded for chunk i_t+1 after forming the current residual and carried through fused O, gate scaling, and current H update; tested on the 24-case FP16/BF16 matrix.

## Hypothesis

The independent next-chunk W load can overlap current-chunk work at the loop-carried H recurrence boundary.

## What happened

Correctness passed. Unconditional lookahead: 13/24 rows faster, 11/24 slower, 9 rows regressed >0.5%, 7 >2%; sum ratio 0.99368 but geometric ratio 1.00790. Long T=16384/H16 gained 3.72-4.20%, while short cases regressed up to 6.15%. A T>=16384 compile-time gate preserved the T16384 gain (3.79-4.24%), but the full panel still had only 6/24 faster rows, 11 regressions >0.5%, and a 1.00961 geometric ratio. Both versions were rolled back.

## Why it was rejected

The full-shape policy is inconsistent and violates the 2% per-shape regression guard; the gated false path also showed unexplained regressions, so the observed long-case gain is not sufficient to promote the current specialization.

## When to retry

Retry only with a separate long-sequence kernel variant that leaves the default JIT kernel and its tuner key unchanged, then run paired warmed baseline/candidate measurements and a target H-kernel rocprofv2 trace/PMC to verify load overlap and no spill/occupancy regression.

## Evidence

- `D:/Desktop/海光/kda-gdn2-task/runs/candidate-hlookahead-summary.md; D:/Desktop/海光/kda-gdn2-task/runs/candidate-hlookahead-gated-summary.md`

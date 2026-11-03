---
id: gdn2-hygon-triton-bv-16-tuning-trial
kind: rejection
status: rejected
device: BW
arch: gfx936
operator: GDN2
scope: H-kernel BV autotune candidates changed from [32,64] to [16,32,64]; B1T8192H96
  K128 V128 FP16/BF16 smoke timing
rejected_on: '2026-11-02'
retry_when: Retry only if a valid PMC/resource profile demonstrates VGPR-limited occupancy
  for a target shape and a paired repeat shows >2% improvement without regressions.
---

# GDN2 Hygon Triton BV=16 tuning trial

> Status: rejected on `2026-11-02`. This is negative knowledge: it records a measured
> failure, not an optimization result. Do not cite it as a win.

## Attempted

- Device/architecture: `BW` / `gfx936`
- Operator: `GDN2`
- Scope: H-kernel BV autotune candidates changed from [32,64] to [16,32,64]; B1T8192H96 K128 V128 FP16/BF16 smoke timing

## Hypothesis

A narrower V tile could reduce the H-kernel register footprint and improve occupancy on the high-register long-sequence specialization.

## What happened

FP16 12.076 ms vs warmed default about 12.06 ms; BF16 12.997 ms vs about 13.03 ms; less than 1% and not stable evidence. Broader matrix was stopped and BV=16 was reverted.

## Why it was rejected

No material repeatable speedup in the first representative shape; adding configurations increases autotune compile/search cost without demonstrated gain.

## When to retry

Retry only if a valid PMC/resource profile demonstrates VGPR-limited occupancy for a target shape and a paired repeat shows >2% improvement without regressions.

## Evidence

- `kda-gdn2-task/docs/draft.md`

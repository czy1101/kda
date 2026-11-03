---
id: gdn2-hygon-triton-akk-generic-layout
kind: rejection
status: rejected
device: BW
arch: gfx936
operator: GDN2
scope: FLAG_ATTN_GDN2_AKK_HEAD_MAJOR=0; B1T8192H96 and B4T4096H64, K128 V128, FP16/BF16
rejected_on: '2026-11-02'
retry_when: Retry after a profile demonstrates Akk layout conversion or access as
  a dominant cost, then use paired repeated same-shape measurements.
---

# GDN2 Hygon Triton Akk generic layout

> Status: rejected on `2026-11-02`. This is negative knowledge: it records a measured
> failure, not an optimization result. Do not cite it as a win.

## Attempted

- Device/architecture: `BW` / `gfx936`
- Operator: `GDN2`
- Scope: FLAG_ATTN_GDN2_AKK_HEAD_MAJOR=0; B1T8192H96 and B4T4096H64, K128 V128, FP16/BF16

## Hypothesis

Changing Akk from head-major to generic layout may reduce auxiliary-matrix layout overhead in the fused H-to-O path.

## What happened

B1T8192H96 Hygon: FP16 12.0569 ms vs default 12.0580 ms; BF16 12.9787 vs 13.0282 ms. B4T4096H64: FP16 15.6988 vs default about 15.7004 ms; BF16 16.3003 vs about 16.2983 ms. Changes are below 0.5% and not directionally consistent.

## Why it was rejected

No repeatable improvement beyond measurement variation; the candidate was env-only and did not justify adding shape dispatch.

## When to retry

Retry after a profile demonstrates Akk layout conversion or access as a dominant cost, then use paired repeated same-shape measurements.

## Evidence

- `kda-gdn2-task/docs/draft.md`

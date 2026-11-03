---
id: gdn2-hip-group-size-32-for-long-low-grid-k128
kind: rejection
status: rejected
device: BW
arch: gfx936
operator: GDN2
scope: K=128; B2T16384H16; FP16/BF16; subgroup size 32 against retained size 16
rejected_on: '2026-11-02'
retry_when: Retry only if a revised block mapping preserves CTA count while increasing
  K-side reuse, then validate on the same long low-grid K128 panel.
---

# GDN2 HIP GROUP_SIZE=32 for long low-grid K128

> Status: rejected on `2026-11-02`. This is negative knowledge: it records a measured
> failure, not an optimization result. Do not cite it as a win.

## Attempted

- Device/architecture: `BW` / `gfx936`
- Operator: `GDN2`
- Scope: K=128; B2T16384H16; FP16/BF16; subgroup size 32 against retained size 16

## Hypothesis

A 32-lane subgroup could reuse each K-side tile across twice as many V lanes compared with group 16.

## What happened

Seven paired rounds measured HIP 39.5096 ms FP16 and 40.4432 ms BF16 at group 32, compared with 31.6376 ms and 31.8600 ms for the retained group-16 path; correctness passed.

## Why it was rejected

About 25-27% latency regression, likely because the larger subgroup halves independent V groups/CTA grid and reduces available parallel work.

## When to retry

Retry only if a revised block mapping preserves CTA count while increasing K-side reuse, then validate on the same long low-grid K128 panel.

## Evidence

- `kda-gdn2-hip-task/docs/draft.md`

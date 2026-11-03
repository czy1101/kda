---
id: hy3-packing-all-eight-gqa-heads-per-cta-shows-no-gain
kind: rejection
status: rejected
device: BW
arch: gfx936
operator: HY3 BF16 contiguous prefill
scope: For GQA group size 8, pack all eight Q heads into the existing 128-row grouped
  CTA instead of two four-head subgroups
rejected_on: '2026-11-03'
retry_when: Retry only with TTGIR evidence of a better operand/accumulator mapping
  and a paired full-panel improvement above run-to-run noise, including Q=127 and
  Q=512 in both layouts.
---

# HY3 packing all eight GQA heads per CTA shows no gain

> Status: rejected on `2026-11-03`. This is negative knowledge: it records a measured
> failure, not an optimization result. Do not cite it as a win.

## Attempted

- Device/architecture: `BW` / `gfx936`
- Operator: `HY3 BF16 contiguous prefill`
- Scope: For GQA group size 8, pack all eight Q heads into the existing 128-row grouped CTA instead of two four-head subgroups

## Hypothesis

Keep the accumulator row count and CTA count while mapping one complete shared-KV head group into the CTA, potentially improving MMAC operand layout and KV reuse

## What happened

The Q=512 selection passed 12/12. Across dense and windowed 4K/32K cases in NHD/HND, HY3 gather+attention changed by approximately -0.5% to +0.8%; long dense rows were flat or slightly slower, and all window rows were slightly slower. No repeatable gain was observed.

## Why it was rejected

Changing the head-to-row mapping did not reduce the measured end-to-end latency. Keeping the additional GQA8 specialization would add dispatch complexity without a demonstrated benefit.

## When to retry

Retry only with TTGIR evidence of a better operand/accumulator mapping and a paired full-panel improvement above run-to-run noise, including Q=127 and Q=512 in both layouts.

## Evidence

- `D:/Desktop/codex/.hygon-control/runs/20261103-hy3-prefill-gqa8-pack-paired.txt; D:/Desktop/codex/.hygon-control/backups/dense_contiguous.gqa8-heads-per-cta-rejected-20261103.py`

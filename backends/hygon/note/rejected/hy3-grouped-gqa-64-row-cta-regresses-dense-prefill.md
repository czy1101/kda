---
id: hy3-grouped-gqa-64-row-cta-regresses-dense-prefill
kind: rejection
status: rejected
device: BW
arch: gfx936
operator: HY3 BF16 contiguous prefill
scope: Reduce grouped split-KV kernel BLOCK_M from 128 to 64 for every grouped-GQA
  dispatch, retaining the same four-head KV-sharing unit
rejected_on: '2026-11-03'
retry_when: Retry only with a mapping that preserves the 128-row K/V reuse footprint
  while reducing live accumulator/layout-conversion state; verify VGPR/LDS and a paired
  dense plus windowed full panel.
---

# HY3 grouped-GQA 64-row CTA regresses dense prefill

> Status: rejected on `2026-11-03`. This is negative knowledge: it records a measured
> failure, not an optimization result. Do not cite it as a win.

## Attempted

- Device/architecture: `BW` / `gfx936`
- Operator: `HY3 BF16 contiguous prefill`
- Scope: Reduce grouped split-KV kernel BLOCK_M from 128 to 64 for every grouped-GQA dispatch, retaining the same four-head KV-sharing unit

## Hypothesis

Halve per-CTA accumulator state and increase CTA count to shorten VGPR live ranges and improve occupancy

## What happened

The Q=512 correctness selection passed 12/12. In the matched 5 ms warmup / 50 ms repetition run, dense HY3 gather+attention rose from 0.9581 to 1.1306 ms at KV=4096 and from 5.0646 to 6.8300 ms at KV=32768 for NHD; HND rose from 0.9549 to 1.1270 ms and from 5.0734 to 6.8438 ms. Windowed rows were near-flat or slightly faster.

## Why it was rejected

The smaller CTA increased dense end-to-end latency by about 18% at KV=4096 and 35% at KV=32768. The results are consistent with losing too much K/V reuse per CTA, which outweighs any presumed reduction in accumulator live state.

## When to retry

Retry only with a mapping that preserves the 128-row K/V reuse footprint while reducing live accumulator/layout-conversion state; verify VGPR/LDS and a paired dense plus windowed full panel.

## Evidence

- `D:/Desktop/codex/.hygon-control/runs/20261103-hy3-prefill-window-cta-optimization.md`
- `D:/Desktop/codex/.hygon-control/runs/20261103-hy3-prefill-grouped-cta64-paired.txt`
- `D:/Desktop/codex/.hygon-control/backups/dense_contiguous.grouped-bm64-rejected-20261103.py`

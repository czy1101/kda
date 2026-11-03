---
id: hy3-window-boundary-loop-fusion-cuts-static-code-without-aggrega
kind: rejection
status: rejected
device: BW
arch: gfx936
operator: HY3 BF16 contiguous prefill
scope: Combine the left and right row-masked window-boundary loops into one dynamic
  loop while keeping the unmasked full-interior loop separate
rejected_on: '2026-11-03'
retry_when: Retry after compiler changes reduce dynamic loop control or a dedicated
  sparse-boundary kernel can share the masked body without regressing the 512/32768
  panel; require repeated paired full-panel measurements and a usable profiler trace.
---

# HY3 window-boundary loop fusion cuts static code without aggregate gain

> Status: rejected on `2026-11-03`. This is negative knowledge: it records a measured
> failure, not an optimization result. Do not cite it as a win.

## Attempted

- Device/architecture: `BW` / `gfx936`
- Operator: `HY3 BF16 contiguous prefill`
- Scope: Combine the left and right row-masked window-boundary loops into one dynamic loop while keeping the unmasked full-interior loop separate

## Hypothesis

Share one generated QK/softmax/PV boundary body to cut duplicate LDS staging and barrier code without adding runtime control overhead

## What happened

Correctness 12/12 and full benchmark 19/19 passed. Full-panel FA/HY3 geometric mean was 0.7055x versus 0.7093x for candidate A. Q=127 window E2E improved about 3%, KV=4096 was flat, and KV=32768 was about 0.5% slower. Static HSACO LDS reads fell 234 to 158 in inspected code objects, but the rocprofv2 split-kernel average was 1.3596 ms versus 1.3494 ms for A.

## Why it was rejected

Static code-object instruction counts fell, but the same-workload full-panel aggregate and traced kernel duration did not improve; the dynamic loop/index selection did not pay for itself.

## When to retry

Retry after compiler changes reduce dynamic loop control or a dedicated sparse-boundary kernel can share the masked body without regressing the 512/32768 panel; require repeated paired full-panel measurements and a usable profiler trace.

## Evidence

- `D:/Desktop/codex/.hygon-control/runs/20261103-hy3-prefill-window-cta-optimization.md; D:/Desktop/codex/.hygon-control/runs/prefill-window-A-paired.log; D:/Desktop/codex/.hygon-control/runs/prefill-window-C-paired.log`

---
id: hy3-paged-prefill-v-half-lookahead-regresses-dense-paged-attenti
kind: rejection
status: rejected
device: BW
arch: gfx936
operator: HY3 BF16/INT8 prefill attention
scope: Load the second output-D half of the 64-token PV V tile before the first output-half
  MMAC in the paged Triton kernel
rejected_on: '2026-11-03'
retry_when: Retry only if a compiler change provides asynchronous or otherwise overlapped
  V loading without raising the paged kernel's register allocation, and the paired
  full shape/layout panel improves.
---

# HY3 paged prefill V-half lookahead regresses dense paged attention

> Status: rejected on `2026-11-03`. This is negative knowledge: it records a measured
> failure, not an optimization result. Do not cite it as a win.

## Attempted

- Device/architecture: `BW` / `gfx936`
- Operator: `HY3 BF16/INT8 prefill attention`
- Scope: Load the second output-D half of the 64-token PV V tile before the first output-half MMAC in the paged Triton kernel

## Hypothesis

A single bounded V-half lookahead may overlap its global-memory load with the first PV MMAC

## What happened

BF16 19/19 and INT8 12/12 passed. Full-panel paged ratio geometric mean changed 0.5784x to 0.5701x and HY3 paged latency geometric mean rose 0.63%; dense 32K paged rows rose 2.8-3.3%. Contiguous E2E stayed within 0.12%. rocprofv2 paged kernel rose 184 to 188 VGPR and 3.4421 to 3.5154 ms average; LDS remained 16 KiB.

## Why it was rejected

The bounded prefetch increased register use and slowed the paged dense path without improving full-panel end-to-end performance. The candidate was reverted.

## When to retry

Retry only if a compiler change provides asynchronous or otherwise overlapped V loading without raising the paged kernel's register allocation, and the paired full shape/layout panel improves.

## Evidence

- `D:/Desktop/codex/.hygon-control/runs/prefill-v-prefetch-baseline.txt; D:/Desktop/codex/.hygon-control/runs/prefill-v-prefetch-candidate.txt; D:/Desktop/codex/.hygon-control/runs/prefill-v-prefetch-profile-export.txt; D:/Desktop/codex/.hygon-control/runs/prefill-v-prefetch-int8.txt`

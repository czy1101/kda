---
id: gla-intra-tile-a-v-fusion-is-slower-at-k-v-128
kind: rejection
status: rejected
device: BW
arch: gfx936
operator: flag_attn.runtime.backend._hygon.FLA.gla.chunk_gla
scope: Fusing the intra-chunk A tile with the A@V output stage when K=128 and V=128
rejected_on: '2026-10-27'
retry_when: A tile schedule covering K=V=128 fits the 64 KiB LDS budget together with
  the measured VGPR/SGPR use, or the compiler stops decomposing the wide tile.
---

# GLA intra-tile A@V fusion is slower at K=V=128

> Status: rejected on `2026-10-27`. This is negative knowledge: it records a measured
> failure, not an optimization result. Do not cite it as a win.

## Attempted

- Device/architecture: `BW` / `gfx936`
- Operator: `flag_attn.runtime.backend._hygon.FLA.gla.chunk_gla`
- Scope: Fusing the intra-chunk A tile with the A@V output stage when K=128 and V=128

## Hypothesis

Removing the FP32 A intermediate and the separate output launch should help any GLA head dimension, so the K=V=64 fusion was widened to K=V=128.

## What happened

The widened fusion candidate measured 1.5x to 1.7x slower than the baseline on the tested K=V=128 cases, while it stayed faster at K=V=64.

## Why it was rejected

At K=V=128 the fused path no longer holds one complete tile per stage: matching tile widths forces the baseline-wide tile to decompose into repeated subtiles, which adds LDS pressure and repeated staging that outweigh removing the intermediate.

## When to retry

A tile schedule covering K=V=128 fits the 64 KiB LDS budget together with the measured VGPR/SGPR use, or the compiler stops decomposing the wide tile.

## Evidence

- `backends/hygon/note/gla/c003-fuse-bf16-k-v-64-gla-intra-tile-output-on-gfx936.md`

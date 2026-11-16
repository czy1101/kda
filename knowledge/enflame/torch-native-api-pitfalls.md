# S60-TORCH-003: native APIs require complete numerical validation

## Scope and observed failures

S60/GCU300, Torch 2.10.0+cpu with `torch_gcu`, Python 3.12.3, as recorded in
the MoBA Torch run on 2026-10-15. This is version/workload-scoped negative
evidence, not a claim that SDPA or integer indexing is always broken on GCU.

| Attempt | Actual evidence | Consequence |
|---|---|---|
| Native half-precision boolean-mask SDPA for shared KV | Small primitive probe passed; complete candidate c002 failed six frozen forward subcases at original tolerances | Reject without benchmarking; restore c001 |
| Additive-mask SDPA primitive | Numerical probe failed | Do not assume changing mask type repairs the full path |
| `index_copy_` with int32 indices | Probe raised `Expected a long tensor for index, but got Int` | Respect per-operation index dtype; tested scatter alternative used |
| Extra contiguous Q copy (r304) | Correct but both end-to-end timings regressed | Reject copy despite plausible layout benefit |

## Reusable decision process

Check installed signatures and support for the exact device, dtype, shape,
strides, mask semantics, scale, causal mode, return values and grad mode.
Backend registration or a callable attribute establishes API presence only.
A successful small probe is not full operator correctness or acceptable
rounding. Preserve original tolerances, references and tests; never weaken
them to justify a faster library call.

Run numerical validation separately from performance commands. Keep failures
as candidate evidence and do not benchmark a numerically rejected candidate.
Pure Torch may dispatch internally to vendor libraries; this does not prove a
particular `tops_convgen_*` path or expose a supported direct Python API for it.
Check compiler/library evidence before naming a low-level execution path.

The successful alternative in that run was FP32 BMM + mask + softmax + BMM
with shared payloads and tested fallbacks, not a generic guarantee that FP32
always beats native SDPA. New SDK/Torch-GCU releases or different workloads
may change both numerical support and performance; repeat the probes/tests.

Sources: `torch-round1-report`, `torch-round1-candidates`,
`torch-round3-report`, `torch-sparse-report` in [sources.json](sources.json).

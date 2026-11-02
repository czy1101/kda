# Triton structural optimization example: multi-row programs

This example records a reusable Triton optimization pattern validated on a
Hygon gfx936/BW device. It is intentionally written as a workflow example,
not as a dispatch table for one benchmark suite.

## Problem pattern

The baseline program processes one logical row at a time:

```text
program_id -> one row
load query and RMS weights
for source in residual_sources:
    reduce over D
    update online softmax and output vector
store one row
```

For moderate hidden sizes, this can create a large number of small programs.
The query and RMS-weight vectors are identical for every row, while the source
loop and the D reduction are independent across rows. The launch/grid work can
therefore dominate even when the kernel is otherwise bandwidth- or VALU-bound.

## Structural candidate

Process two rows in one Triton program. The logical tile is
`[BLOCK_M, BLOCK_D]`, with `BLOCK_M=2` in the validated candidate:

```python
rows = program_id(0) * BLOCK_M + arange(0, BLOCK_M)
offsets_d = arange(0, BLOCK_D)
mask = (rows[:, None] < N) & (offsets_d[None, :] < D)

query_weight = load(query) * load(rms_weight)       # shared by both rows
output_values = zeros([BLOCK_M, BLOCK_D], fp32)

for source in static_range(0, L):
    values = load(residual[source] + rows[:, None] * D + offsets_d[None, :])
    rstd = rsqrt(sum(values * values, axis=1) / D + eps)
    score = sum(values * query_weight[None, :], axis=1) * rstd * scale
    output_values = online_softmax_update(output_values, score, values)

store(output + rows[:, None] * D + offsets_d[None, :], output_values)
```

The source loop remains online, so this change does not introduce a second
residual read or an intermediate score buffer. It only changes the program
granularity and reuses row-invariant data within a program.

## Safe applicability guard

The initial candidate was enabled only when all of the following held:

```text
number of residual sources >= 3
768 <= number of rows <= 1536
hidden size <= 2048
output RMSNorm is disabled
probability/logit output is disabled
```

An optional extension can keep output RMSNorm inside the same two-row tile,
but it needs a stricter `D` bound because it adds another per-row reduction.
It must be benchmarked separately; the no-RMSNorm result does not justify
enabling this extension automatically.

In the validation run, the extension was correct but statistically neutral:
for `L=5, N=1024, D=1024, BF16` with output RMSNorm it changed approximately
`0.0682 ms` to `0.0653 ms`, while neighboring cases were within measurement
noise. It was therefore rejected and removed from the production dispatch.

These are resource-based tuning constraints, not public-shape exceptions:

- `D <= 2048` keeps the `[2, BLOCK_D]` register tile bounded;
- the lower row bound amortizes the launch and fixed program overhead;
- the upper row bound avoids replacing a sufficiently occupied one-row grid;
- output RMSNorm and probability output have different register and store
  requirements, so they stay on the general kernel until separately validated.

The implementation should provide an emergency rollback switch. In the
validated example it is:

```bash
export FLAG_ATTN_ATTNRES_MULTIROW=0
```

The default is enabled only after the candidate passes correctness and repeated
benchmark checks.

## Evidence protocol

1. Run the original one-row path with the same input seed, device, warmup, and
   repetition count.
2. Run the two-row candidate with exactly the same conditions.
3. Compare both outputs with the reference implementation before timing.
4. Repeat timings in alternating order to detect cache, clock, or autotune
   ordering effects.
5. Include boundary cases outside the guard. A candidate that wins one shape
   but regresses its neighboring range must remain gated or be rejected.

## Representative result

Measured on Hygon gfx936/BW, BF16, no output RMSNorm, five residual sources,
with Triton warmup and repeated median timing:

| Workload | One-row program | Two-row program | Relative change |
|---|---:|---:|---:|
| `L=5, N=1024, D=512` | 0.0656 ms | 0.0189 ms | 3.47x faster |
| `L=5, N=1024, D=1024` | 0.0672 ms | 0.0235 ms | 2.86x faster |
| `L=5, N=1536, D=1024` | 0.0378 ms | 0.0310 ms | 1.22x faster |
| `L=5, N=1536, D=2048` | 0.0501 ms | 0.0491 ms | approximately neutral |

Correctness passed for the BF16/FP16 candidate cases and the full generic
AttnRes test suite. Long-D workloads (`D=4096/7168`), output-normalized cases,
and probability-output cases remained on the one-row kernel with no benchmark
regression.

## What not to do

Do not generalize this result to every row count or hidden size. In boundary
tests, the two-row kernel could be neutral or slower once the one-row grid was
already well occupied. Do not use a fixed benchmark shape as the dispatch rule,
and do not keep the candidate if correctness, resource usage, or neighboring
shape performance regresses.

## KDA decision template

Record this candidate with:

```text
hypothesis: row-invariant query/RMS work and launch/grid overhead justify a
            BLOCK_M=2 Triton program for a bounded moderate-D range
change:     one-row program -> [2, BLOCK_D] program; online source softmax kept
correctness: reference comparison before timing; full public tests afterward
benchmark:  baseline and candidate in alternating order, per-shape medians
promotion:  only guarded ranges with stable benefit and no accepted regressions
rollback:   FLAG_ATTN_ATTNRES_MULTIROW=0
```

The same loop applies to other operators, but the guard must be recalculated
from that operator's register tile, reduction axes, output requirements, and
profile evidence.

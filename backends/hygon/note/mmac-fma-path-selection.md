# Hygon gfx936 MMAC/FMA path selection and multi-row fusion

This document records a reusable optimization method for Hygon gfx936
operators. It is based on the HY3 Decode work, but the method is broader than
HY3: map the effective logical rows of a matrix-like subproblem to the native
tile shape, fuse rows only when reuse justifies it, and select MMAC or vector
FMA from measured tile utilization rather than from an instruction mnemonic.

The document is a design and validation guide. It is not a claim that every
operator or every gfx936 workload benefits from MMAC, fusion, or split-K.

## 1. What is reusable and what is HY3-specific

The reusable method is:

1. derive the effective row count and tile occupancy from the workload;
2. identify the candidate hardware matrix tile and the vector fallback;
3. fuse independent rows only when the fused kernel reuses expensive K/V or
   weight data and preserves masking and reduction semantics;
4. choose the compute path per workload class;
5. verify generated code, resources, correctness, and end-to-end latency;
6. promote only a stable candidate that beats the baseline on a representative
   workload panel.

The HY3-specific instance is MTP fusion in Decode. In the validated design,
MTP=2 with GQA=8 and MTP=4 with GQA=4 both produce 16 logical query-head
rows. This makes an M=16 matrix tile fully useful. The MTP value in this
document means the number of query tokens processed by one fused Decode
kernel; it is not a generic benchmark column or a promise that arbitrary MTP
values should be fused.

## 2. Tile completeness is the dispatch signal

For a matrix-like operation, define:

```text
M_eff = query_rows_fused * query_heads_per_group
occupancy = M_eff / M_tile
```

On the tested gfx936 toolchain, the relevant BF16 MMAC candidate uses an
`M_tile` of 16. A complete tile (`M_eff == 16`) avoids padding rows and is the
best starting point for an MMAC candidate. When `M_eff < 16`, the matrix path
may execute padded work, consume more registers, or pay a layout/conversion
cost that outweighs the arithmetic throughput. In that case vector FMA is a
valid and often preferable path.

This is a decision rule, not a hard-coded shape table. The implementation
should derive `M_eff` from the runtime workload and use a conservative fallback
when a shape is outside the measured domain.

The same rule applies to other tile dimensions. Check the actual K/N tile,
alignment, mask density, and reduction shape before selecting a matrix path.
Do not infer suitability from M alone.

## 3. Multi-row fusion

Fusion is useful when multiple logical rows use the same K/V or weight tiles.
The fused kernel can load a tile once and compute several rows, reducing global
memory traffic and launch/scheduling overhead. It must still implement each
row's own:

- sequence-length or causal mask;
- quantization scale and zero-point, if present;
- softmax or normalization state;
- output and accumulation location.

The HY3 examples are:

| Fused rows | GQA | Effective rows | Expected use |
|---:|---:|---:|---|
| 2 query tokens | 8 heads/group | 16 | Complete M=16 candidate |
| 4 query tokens | 4 heads/group | 16 | Complete M=16 candidate |

MTP=8/GQA=2 and MTP=16/GQA=1 are not automatically better. They require
separate correctness and resource checks; a larger fused program can increase
register pressure, reduce occupancy, or make ragged masking expensive.

Fusion should be rejected when rows do not share enough data, when valid rows
are sparse, or when the fused kernel causes spills or a worse end-to-end
latency. A synthetic GEMM that emits MMAC is only a lowering probe; it is not
evidence that the attention kernel benefits.

## 4. BF16 and FP8 path policy

The current HY3 evidence supports a dtype-specific starting policy:

- BF16: test the complete M=16 MMAC candidate first, then compare it with a
  vector/FMA candidate on the same shapes.
- FP8: start with the lower-register vector/FMA candidate. Test MMAC only when
  the toolchain's FP8 lowering is known and the same-shape benchmark shows a
  gain; do not assume that BF16 MMAC lowering transfers to FP8.

Implementations may expose explicit experiment controls such as
`HY3_BF16_MTP_COMPUTE_PATH=auto|mmac|vector` and
`HY3_MTP_COMPUTE_PATH=auto|mmac|vector`. These controls are for A/B validation,
not a substitute for a workload-aware default. The default must remain
correct and conservative for unsupported shapes.

## 5. Workload-aware dispatch sketch

The following pseudocode describes the intended policy. It deliberately avoids
dispatching from a list of benchmark shapes:

```python
def choose_path(dtype, query_rows, heads_per_group, mask_density, resources):
    m_eff = query_rows * heads_per_group
    complete_m_tile = (m_eff == 16)
    reuse_is_high = resources.shared_kv_reuse and mask_density > 0

    if not complete_m_tile or not reuse_is_high:
        return "vector"
    if dtype == "bf16" and resources.bf16_mmac_validated:
        return "mmac"
    if dtype == "fp8" and resources.fp8_mmac_validated:
        return "mmac"
    return "vector"
```

In production code, replace the symbolic `resources` fields with capabilities
that have been established by the local toolchain and benchmark ledger. Keep a
vector fallback for partial tiles, unsupported layouts, and failed validation.

For split-K or producer/consumer pipelines, apply the same principle to
parallelism: derive the number of tasks from available work and occupancy,
then measure combine and scheduling overhead. Do not increase split-K merely
because a long sequence exists.

## 6. Required evidence before promotion

Every candidate must have four kinds of evidence.

### Correctness

Compare the candidate with a trusted reference across:

- all supported dtypes and quantization modes;
- MTP=1, the fused cases, and at least one unsupported/partial-tile case;
- uniform and ragged sequence lengths;
- short, medium, and long sequences;
- causal and non-causal masks when applicable.

Use dtype-appropriate tolerances and test the fallback path explicitly. A
failed candidate must remain outside the default dispatch.

### Generated code

For Triton candidates, inspect the generated HSACO and disassembly for the
actual kernel. On the tested toolchain, a BF16 matrix candidate may contain an
instruction such as `v_mmac_f32_16x16x16_bf16`. The exact mnemonic is compiler
and ISA dependent. The presence of `v_mmac_*` alone is not a performance claim.

Record at least:

- matrix versus vector instruction mix;
- conversion and packing instructions;
- VGPR/SGPR usage and occupancy;
- LDS usage and spills;
- barriers and wait counts;
- whether the inspected kernel is the one timed by the benchmark.

### Profiling

Use `rocprofv2` for kernel traces and structured counters when available, and
`hipprof` for timeline questions. Use `rocminfo` and `rocm-smi` (or the native
Hygon monitor) as device-state probes, not as kernel profilers. The complete
workflow is in [profiling-workflow.md](profiling-workflow.md).

For Decode, separate producer, task-map, and combine time when possible. For
Prefill, separate QK, PV, scale loads, LDS movement, and reduction. A path
change should explain a measured bottleneck, not only change the instruction
listing.

### Same-shape benchmark

Compare baseline, vector candidate, and MMAC candidate with the same warmup,
repeat count, stream, graph mode, allocator state, and device mask. Report
median and dispersion. Include a short control group so that a faster long
case is not hiding a regression in latency-sensitive workloads.

### Promotion rule

Promote a default only when correctness passes and the candidate gives a stable
end-to-end improvement on the representative panel. Keep raw reports and the
decision in the candidate ledger. If the result is mixed, keep the candidate
behind an explicit experiment flag or retain the baseline.

## 7. Where this method applies

| Operator pattern | Relevance | Notes |
|---|---|---|
| GEMM / batched matmul | High | Derive M/N/K tile occupancy directly. |
| Attention QK/PV | High | Row fusion can reuse K/V; masks and softmax are mandatory checks. |
| GLA/KDA/MSA matrix subkernels | High | Apply the same tile and reuse analysis to each matrix stage. |
| Split-K reductions | Conditional | Useful when task parallelism exceeds one CTA; measure combine cost. |
| Softmax, RMSNorm, AttnRes | Low to conditional | Usually vector/reduction bound; tile fusion is not automatically useful. |
| Irregular indexing and scatter/gather | Conditional | Memory coalescing and masking often dominate matrix throughput. |

The MTP fusion itself is Decode Attention-specific. The tile-completeness and
vector-versus-matrix validation method is the portable part.

## 8. Anti-overfitting rules

- Do not encode a long list of exact shapes as the optimization policy.
- Do not select MMAC because a disassembly contains one MMAC instruction.
- Do not copy a BF16 path to FP8 without validating its lowering and resources.
- Do not report a kernel-only win as an operator win until launch, combine, and
  synchronization costs are included.
- Keep an explicit fallback for partial tiles and unsupported layouts.
- Record rejected candidates and the workload panel that rejected them.

## 9. KDA checklist

1. Describe the operator, dtype, layout, masks, and effective logical rows.
2. Run the correctness baseline and capture a same-shape latency baseline.
3. Probe the generated HSACO and profiler capabilities on the target gfx936.
4. Form one hypothesis: tile completion, data reuse, scale traffic, or task
   parallelism.
5. Implement one candidate with a fallback and an experiment flag.
6. Run correctness on fused, partial, ragged, and control shapes.
7. Compare vector and MMAC candidates with identical benchmark settings.
8. Inspect resources and profile counters; explain the latency change.
9. Promote only a stable result and document scope, evidence, and limitations.

This checklist is the reusable optimization reference for future Hygon
operators. It should be used together with the backend-specific profile
workflow and the operator's local benchmark/test instructions.

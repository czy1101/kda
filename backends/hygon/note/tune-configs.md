---
id: hygon-triton-tune-configs
kind: guide
status: provisional
device: BW
arch: gfx936
operator: "triton tune_configs.yaml"
scope: "Author Hygon Triton tune_configs.yaml candidates without breaking loader selection"
revalidate_after: 2027-01-31
---

# Hygon Triton `tune_configs.yaml` guide

## Purpose

`src/flag_attn/runtime/backend/_hygon/tune_configs.yaml` is a target-project configuration file for curated Triton launch candidates. Each top-level key identifies a Triton kernel name; its list contains candidate compile-time metadata and launch settings. The file does not define operator math, select a backend by itself, or contain compiled kernels.

KDA does not automatically parse or apply this YAML. KDA supplies the controlled device environment, runs correctness and benchmark commands, captures profiler evidence, and records candidate results. The target project must load the YAML and pass the resulting `triton.Config` list to the operator's autotune/decorator or launcher.

## Current Hygon status

The inspected Hygon PR tree contains the file, but its current contents are an empty mapping:

```yaml
{}
```

An empty mapping has no kernel candidates. The repository's generic `TunedConfigLoader` can resolve a backend-specific path and parse the YAML, but an operator only changes behavior if it calls the loader (directly or through a backend compatibility helper) and uses the returned configs. The local PR checkout inspected for this note had no Hygon `get_tuned_config()` call sites; the recursive remote call-site search timed out. Therefore, treat the current file as a placeholder until Hygon operators are explicitly wired to consume it.

The AttnRes implementation in the inspected PR checkout sets launch values in its Python wrapper (for example, `num_warps`, `num_stages`, and padded `BD`). Adding entries to the empty YAML alone will not change that kernel's launch configuration.

## Configuration fields

A typical entry has this shape:

```yaml
some_hygon_triton_kernel:
  - META: {BK: 32, BV: 64}
    num_warps: 4
    num_stages: 2
  - META: {BK: 64, BV: 64}
    num_warps: 4
    num_stages: 2
```

Replace `some_hygon_triton_kernel`, `BK`, and `BV` with names that exist in the target kernel. This is an illustrative schema, not a measured Hygon recommendation.

- **Top-level kernel key:** Must match the string passed to `get_tuned_config(op_name)` in the operator source. It is not necessarily the Python file name.
- **`META`:** Compile-time kernel arguments, usually tile sizes or other `tl.constexpr` values such as `BK`, `BV`, `BH`, `BS`, or `B_VCHUNK`. Names must match the kernel's constexpr parameters and must not conflict with arguments that the launch wrapper still supplies explicitly.
- **`num_warps`:** Triton warps per program/CTA. This changes lane-level parallelism and register distribution.
- **`num_stages`:** Software pipeline depth where supported and relevant to the kernel's loop/load structure.
- **Optional Triton config fields:** The current generic loader recognizes `num_ctas` and `maxnreg` when those parameters are present in the installed `triton.Config` signature. Confirm backend/compiler support before adding them.

These values control specialization and launch behavior. They are not interchangeable across architectures: an Enflame S60 candidate is not evidence that the same tile or warp count is suitable for Hygon gfx936.

## Loader and selection behavior

In the inspected FlagAttention implementation, `TunedConfigLoader(backend)` reads:

```text
src/flag_attn/runtime/backend/_<backend>/tune_configs.yaml
```

It parses the file with `yaml.safe_load`, then converts each list entry into a `triton.Config`. Config loading is cached in Python by operator name. Triton's JIT/autotune specialization and compiled-kernel caches are separate mechanisms; this YAML is not a `CompiledKernel` or launcher cache.

The Enflame compatibility helper provides a useful reference for fallback behavior:

1. If `FLAG_ATTN_USE_TUNED_CONFIGS=0`, use code-defined fallback candidates.
2. Otherwise, use the YAML candidates when the requested kernel key exists.
3. If the key is missing or has no candidates, use the fallback list.
4. If `FLAG_ATTN_FAST_AUTOTUNE=1`, restrict a non-empty YAML candidate list to its first entry.

That environment-variable behavior belongs to the compatibility helper that implements it. Do not assume these variables affect Hygon until Hygon code uses an equivalent helper. Passing one candidate still may make Triton benchmark that candidate on first use; it reduces the search set and does not necessarily disable autotune measurement.

When wiring Hygon, explicitly define the empty/missing-config behavior. Prefer returning a known code fallback instead of passing an empty list to an autotune decorator. Keep the fallback available so users can disable curated configs or handle unvalidated shapes.

## Adding Hygon candidates safely

1. **Confirm the backend and target.** In the KDA task contract and live preflight, record the Hygon device, gfx architecture, DTK/Triton versions, dtype, and device mask. Read the matching `backends/hygon/note/INDEX.md` guidance.
2. **Inspect the target kernel.** Identify its constexpr arguments, launch arguments, autotune key, existing fallback configs, and correctness requirements. Confirm that the requested YAML key matches the source call site.
3. **Wire the loader deliberately.** Add or reuse a Hygon compatibility helper that obtains the `hygon` config list, handles missing/empty entries with an explicit fallback, and feeds candidates into the Triton autotune path. Avoid loading Enflame settings for Hygon.
4. **Establish the baseline first.** Run the existing correctness and benchmark path with the fallback/current launch settings. Save output and source revision in the task workspace.
5. **Add a small candidate set.** Change one meaningful parameter family at a time (for example, tile size or warp count). Avoid a large Cartesian product; it inflates tuning time and can favor noisy measurements.
6. **Check correctness for every candidate path.** Include boundary sizes, dtypes, optional outputs, and shapes used by the operator. A faster but incorrect config is rejected.
7. **Benchmark on Hygon.** Use the task's paired/alternating benchmark protocol and repeat enough rounds to separate stable gains from device-clock or host-load noise. Keep public-API latency distinct from kernel-only timing.
8. **Profile only to answer a concrete question.** For Hygon kernel claims, use the validated profiler workflow, preferably rocprofv2 kernel traces. Do not infer occupancy or memory bottlenecks from a config value alone.
9. **Record and promote evidence.** Keep each candidate in the task workspace ledger (`runs/candidates.jsonl`, `runs/benchmark.csv`, and `profile/`). Promote only correctness-passing, repeatable candidates; record regressions and rejected settings too.

KDA does not require storing per-task candidate YAML or benchmark output in this repository. Keep those artifacts in the task workspace; this note documents the reusable workflow.

## Practical environment controls

The following controls are available only if the target project's Hygon loader/helper implements them. Verify the Hygon call path before relying on them:

```bash
# Use the curated Hygon candidates, when integrated.
export FLAG_ATTN_USE_TUNED_CONFIGS=1

# Limit tuning to the first candidate per kernel, when implemented.
export FLAG_ATTN_FAST_AUTOTUNE=1

# Return to the code-defined fallback candidates, when implemented.
export FLAG_ATTN_USE_TUNED_CONFIGS=0
```

For an initial correctness/performance study, leave fast mode off so the candidate set can be compared, then use fast mode only after confirming the first candidate is acceptable for every intended specialization key. Record the environment values in KDA benchmark evidence because they change which candidates can be selected.

## Common mistakes

- Assuming the existence of `tune_configs.yaml` means an operator consumes it.
- Treating `{}` as a set of default launch parameters; it contains no candidates.
- Copying Enflame S60 or another backend's measured parameters to gfx936 without retuning.
- Using a `META` key that the kernel does not declare as a compile-time argument.
- Supplying the same constexpr both in the `triton.Config` and explicitly in the launch wrapper.
- Assuming `FLAG_ATTN_FAST_AUTOTUNE` or other environment variables have an effect when Hygon code does not read them.
- Claiming a performance gain from the YAML change without correctness and paired benchmark evidence.

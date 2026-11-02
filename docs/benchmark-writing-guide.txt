FlagAttention Benchmark Writing Guide
====================================

Purpose
-------
This guide records conventions found in the repository's benchmark/ directory
and applies the additional requirement that the default human-readable output
is a table. Keep benchmark code and task-specific results in the operator
repository; this KDA document is reusable guidance only.

Repository examples inspected
-----------------------------
- benchmark/base.py: shared Benchmark class and default test_backend.do_bench use.
- benchmark/test_fused_attnres_benchmark.py: correctness check, public API scope,
  alternating provider order, repeated rounds, median summary, result recording.
- benchmark/test_diffkv_attention_benchmark.py: fixed-width table, repeated
  samples, median/min/max/CV reporting, and structured recording.
- benchmark/recording.py and conftest.py: structured result fields and JSON output.
- pyproject.toml: registered pytest marks and test discovery settings.

1. File and test structure
--------------------------
- Put operator benchmarks under benchmark/ and use a descriptive name such as
  test_<operator>_benchmark.py. Keep the benchmark callable small and explicit.
- Name pytest entry points test_<operator>_benchmark. Put input construction,
  shape metadata, correctness checks, timing, table output, and result recording
  in clearly separated helpers when the case matrix is large.
- Build inputs outside the timed callable. The callable should contain exactly
  the work named by the benchmark scope.
- Use @torch.inference_mode() for inference-only measurements.
- Validate candidate output against the reference before timing. Record the
  tolerance and a useful error metric such as max_abs_error when comparing two
  implementations.

2. Pytest marks and case selection
----------------------------------
- Use one operator-specific mark as the primary selection tag, for example
  pytest.mark.fused_attnres. Register every custom mark in
  [tool.pytest.ini_options].markers in pyproject.toml to avoid unknown-mark
  warnings and make selection discoverable.
- Use pytest.mark.skipif with a specific reason when a backend, device, or
  required implementation is unavailable. Prefer the repository backend
  capability helpers over ad hoc device-name checks.
- Use pytest.mark.parametrize for workload variants. Supply stable, readable
  IDs when a case has several dimensions (e.g. L5_N128_D7168_normTrue).
- Optional generic marks such as performance or benchmark are registered in
  this repository, but the operator-specific mark is the established primary
  tag. Add generic marks only when they support a real runner or collection
  policy.
- The project config sets testpaths to tests/. Pass benchmark test paths
  explicitly when running a benchmark under benchmark/.

Example invocation:
  pytest -q benchmark/test_fused_attnres_benchmark.py \
    -m fused_attnres --record=json --output=runs/fused_attnres.json

3. Timing protocol
------------------
- Prefer `from flag_attn.testing import backend as test_backend` and
  `test_backend.do_bench(fn, warmup=..., rep=..., return_mode="median")`.
  This selects the backend-appropriate timing implementation. On NVIDIA it
  delegates to Triton's do_bench; other devices use the backend event API when
  available, with synchronized wall time as a fallback.
- `warmup` and `rep` are millisecond time budgets in test_backend.do_bench;
  they are not counts of calls. The helper performs an untimed call and device
  synchronization before warmup. It synchronizes during warmup and sampling,
  then returns the requested statistic over samples. With no device event API,
  the fallback is synchronized host wall time and includes host dispatch.
- Compile and initialize the candidate before the measured section. The
  correctness call normally provides this warm-up; explicitly synchronize
  before timing if setup work might remain queued.
- For reference-versus-candidate comparisons, use the same input tensors,
  device, dtype, API scope, timing helper, warmup budget, repetition budget,
  and summary statistic. Do not compare graph replay for one implementation
  against ordinary launches for the other.
- Alternate which implementation runs first across samples or rounds to reduce
  clock and load-order bias. Measure several paired rounds and use the median
  as the primary result. Preserve per-round values in structured output; report
  min/max and sample CV when stability is relevant.
- Keep synchronization outside the callable unless synchronization is part of
  the API contract. Let the timing helper control event placement and device
  synchronization.
- State whether the timed scope is `public_api_forward` (including wrapper,
  dispatch, and output allocation) or `kernel` (a lower-level launch, often with
  preallocated buffers). The AttnRes benchmark explicitly measures the public
  API and includes allocation and dispatch. Do not label that result as pure
  kernel time.
- Print the exact timing method and units. For a custom event loop, graph
  replay, or host-wall-time timer, document its start/stop boundaries and
  synchronization behavior instead of calling it `do_bench`.

4. Default stdout format: table
--------------------------------
Print a short metadata header followed by one table row per measured case. Do
not make free-form per-case sentences the default result format.

The header should state:
- benchmark/operator and benchmark scope;
- device name and vendor/backend;
- dtype;
- timing method;
- warmup and repetition budgets with units;
- number of outer samples/paired rounds and the aggregation statistic;
- reference/baseline implementation, or `none`.

Use columns that identify the workload and show the measured result. Include
only applicable dimensions, but never hide shape information in an opaque case
name. Recommended comparison columns:

  Case | Parameters | Dtype | Reference (ms) | Candidate (ms) |
  Speedup (x) | Max abs error

Define `Speedup (x)` as `reference_latency / candidate_latency`; values above
1.0 mean the candidate is faster. Use `N/A` when no reference or accuracy
comparison exists. Use consistent precision, for example latency to 6 decimals,
ratio to 3 decimals, and error to 6 decimals. Separate warnings from data rows
and flush output so long-running jobs show progress.

Example:

  AttnRes | public_api_forward | Hygon gfx936 | bfloat16
  Timing: test_backend.do_bench; warmup=1000 ms; rep=100 ms;
          7 alternating paired rounds; median
  | Case             | Parameters                 | Reference (ms) | Candidate (ms) | Speedup (x) | Max abs error |
  |------------------|----------------------------|----------------|----------------|-------------|---------------|
  | L2_N1_D7168       | output_norm=False          | 0.038000       | 0.050000       | 0.760       | 0.000000      |
  | L5_N128_D7168     | output_norm=True           | 0.050000       | 0.060000       | 0.833       | 0.007812      |

A plain-text aligned table or a pipe-delimited Markdown table is acceptable;
use one style consistently within a benchmark. Tables are the default for
human-readable stdout. Do not rely on stdout parsing as the machine interface.

5. Structured results
---------------------
- Also record structured results through benchmark/recording.py helpers:
  benchmark_metric, BenchmarkRecorder, or record_benchmark_result. In pytest,
  pass record_property through so conftest.py can collect records.
- Include op_name, dtype, mode/scope, device/vendor, baseline name, and one
  result object per workload. Each result should retain shape_detail, latency,
  latency_base, speedup, and applicable accuracy/error and sample statistics.
- Keep latency units explicit in the field names or metadata; the project
  recorder uses milliseconds for latency values. Derive speedup from measured
  latencies (`latency_base / latency`) rather than hand-entering an unrelated
  ratio.
- Enable JSON recording with `--record=json` and use `--output=PATH` to make
  the artifact location explicit. The table remains for people; JSON is for
  automation and downstream summaries.

6. KDA task contract and evidence
---------------------------------
- In the operator workspace's .kernelpilot/task.yaml, set the exact
  benchmark.command, workload shapes, latency metric, direction, unit, and
  baseline comparator. Ensure the command's real timing controls agree with
  the contract and printed metadata.
- Distinguish millisecond budgets (warmup_ms/rep_ms) from invocation counts
  (calls_per_round) and outer repetitions (rounds/samples). Never use an
  ambiguous `repeat` label for both.
- Keep benchmark.shapes strings stable because KDA uses them to validate and
  aggregate runs/benchmark.csv. Report the same case identity in the stdout
  table and structured result.
- Put task-specific benchmark scripts and result artifacts in the operator
  workspace, not in the reusable KDA repository. Keep this guide and other
  reusable workflow material in KDA docs/.
- Record only measured values. Preserve failed, noisy, or regressed results and
  state the reason; do not infer a speedup from profiling traces alone.

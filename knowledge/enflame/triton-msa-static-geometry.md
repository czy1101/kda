# S60-TRITON-008: specialize known MSA geometry before changing hardware paths

## Scope and evidence

Observed2026-10-16, S60/GCU300, torch_gcu2.10.0 and Enflame Triton3.6.0.
BF16,D128,K16; public MSA prefill and decode. Baseline is the previously
optimized S60-TRITON-007 c2, not the original unoptimized implementation.
Retained c4: formal14-case sum ratio 0.883363894, speedup
1.1320x, every shape within the2% regression cap.

Correctness and timing run in separate processes. Each candidate has one
complete14-shape invocation, event median,warmup200/rep300 ms. No routine
three-run repetition or suppressed sample. Supplemental PR dimensions are
an independent performance boundary, not a promotion comparator.

## Mechanisms tested

1. Make host-known head/group dimensions, TopK, loop count, scale and strides
   constexpr for the single-query direct prefill kernel. This specializes real
   tensor metadata, not benchmark values or output data. c1 reduced the formal
   sum5.92%; two GQA12 full calls improved16.4% and17.2%. No dtype/API change.
2. Because the wrapper launches BLOCK_SIZE_Q=1, load/store Q/O as2D[H,D]
   tiles and form the causal mask as2D[H,K], removing the unnecessary query
   tensor axis. Keep actual strides, padded heads/dimensions and each query's
   own selected pages. c2 cumulative sum ratio0.884356032 (11.56% reduction).
   Four affected direct-prefill full calls improve9.7%-24.2%; grouped cases
   and decode remain on their existing paths.

Both kernels already called convgen and structured page-transfer routines.
This is not a first activation of a matrix unit or a demonstrated conversion
of bulk KV gather into DTE copies. The independent24.9x page-transfer probe
from S60-TRITON-005 is not an MSA performance result.

Actual direct-prefill callsites retain m8/m16_n128_k128 convgen. Static DTE
waits decrease34->20; shared metadata75776->65536 bytes for the GQA6 class,
and86016->65536 for GQA12. These are compiler facts paired with full-call
timing, not dynamic waiting-time fractions or proof of hardware utilization.

The decode denominator candidate moves log2 out of the selected-page loop.
Its static log2 callsite count remains1, so counts alone miss this change in
dynamic placement; static exp2 sites19->17. Its incremental measured gains
were modest. Source/IR placement and full-call timing remain necessary.

Actual-launch diagnostics pin source/JIT/compiled hashes, GCU/LLVM callsites,
GCU300 ELF and disassembly. The task's low-level-analysis.md and call-changes.json
show which routines changed. Static DTE waits or shared metadata do not measure
transfer time, occupancy, bandwidth or overlap. No working GCU profiler profile
was available; no tool/driver/clock was installed or changed. TLE disabled.

## Further candidate results

- c3: reject; prefill-single-contiguous-KV-page-load-and-local-split; validation-timeout600s-after-three-tests-no-performance-measurement.
- c4: keep; decode-online-denominator-instead-of-per-page-log-state; correct-lower-sum-and-all-shapes-within-cap.
- c5: reject; bounded-device-decode-TopK-replaces-Torch-fallback; decode-improved-but-sum-0.0216-percent-above-c4-under-frozen-formal-rule.

c3 is a validation timeout, not an established compilation or device-execution
root cause. No performance measurement was made. c5's device TopK reduced small
decode full calls approximately0.95->0.37ms and passed separate exact selector
guards, but its sum over all14 cases was0.0216% above c4. It was rejected by the
frozen sum rule. That tiny sum difference includes unmodified prefill variability
and is not evidence of a hardware slowdown. Weighted sums can obscure decode
gains; select a task-appropriate comparator before measurement, never afterward.
The user then changed formal workloads to PR94; the new benchmark is separately
authorized and tested. No8-row decode-floor candidate was attempted in this round.

## Compare the same operator boundary

PR94 documents BF16,Hq8,Hkv1,D128,page128,K16 and21 prefill/18 decode
workloads. Its table times Attention with real indices generated beforehand;
our regular14-case end-to-end benchmark also times scoring and TopK, with
Hq96-384 and Hkv4-16. Do not divide those original full-call values directly
by PR Attention values.

Fresh matched-dimension S60 baseline/PR GPU geometric ratios were8.699 for
prefill and2.813 for decode. Retained ratios are
8.315 and2.782;
matched S60 baseline-to-best geometric speedups are
1.046x and1.011x.
Small batch1 decode was already approximately0.114ms vs PR0.112-0.114ms.

These are approximate historical comparisons: S60's frozen GCU event helper
does not clear L2, while PR Triton do_bench does. PR provides no benchmark
script in the five changed files and leaves seed, page order and init/local
unspecified. Local assumptions are seed0,random pages,init/local1/2 and suffix
query positions. Synchronized20-call wall medians are also retained. These
differences prevent a strict hardware ranking or utilization attribution.

## Revalidation

Recheck compiler/device library, tensor geometry/strides, head/dimension tails,
ragged prefixes, selected-page masks, TopK and actual public backend dispatch.
Only flatten an axis whose singleton launch property is guaranteed. constexpr
specialization can increase compilation/cache variants; steady-state speed does
not prove lower compilation cost. A fused contiguous payload load may incur
additional split/layout cost; consult failed candidate evidence before retrying.
Use current correctness and full-call measurement; do not generalize a lower
static call count or the Hygon MFMA flag to Enflame.

## Contrary cross-operator geometry result (2026-10-21)

An independent GCU300/Triton3.6.0 attention audit confirms that blanket
specialization of host-known sequence extents and tensor strides is not a
general performance recipe. It removed masked K/V payload gather in the
tested specialization, while several already-native dense kernels gained
shared-storage attributes and additional static transfer waits and regressed
in full-call measurement. Separate correctness passed, so the candidate was
rejected on performance rather than numerical behavior.

A narrower immutable-base/scalar-loop-offset address transformation also
removed that gather, retained dense matrix/transfer call families and avoided
the broad geometry candidate's dense regression. Boolean-mask performance
improved; additive-mask behavior and near-parity dense performance were
reported separately. See [S60-TRITON-002](triton-compute-memory-paths.md).

This narrows the portability of the earlier MSA observation without changing
its scoped evidence. Inspect the generated decomposition and full workload
after specializing real metadata; fewer dynamic indices, native-unit activity
or successful compilation alone does not establish a faster operator.
Sources: `sage-attention-static-path`, `sage-attention-path-comparison`,
`sage-attention-confirmation`, and `sage-attention-report` in
[sources.json](sources.json).

## PR94 project benchmark defaults

The subsequent user-authorized benchmark migration changes defaults to exactly
21 prefill and18 decode cases, BF16,Hq8,Hkv1,D128,K16. Five-value shapes are
B,Qlen,Seq,KVH,Hq; prefill Q/index-Q use Qlen and prefix=Seq-Qlen. Legacy
four-value custom shapes remain supported. The unchanged c4 kernels completed
all39 full-call/per-step measurements once. Correctness-test and timer hashes
remain unchanged. Benchmark numerical output checks and accuracy columns were
removed; independent configuration checks run separately. The source manifest
links the new measurements/report/proof separately from old14-case comparisons.

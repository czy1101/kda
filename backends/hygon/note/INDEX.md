# Hygon optimization notes and reports

Start with the notes that match the target device, architecture, language,
operator, and workload. Confirm that the task environment matches each note's
validated scope before applying its findings.

The archived gfx936 validation uses `HIP_VISIBLE_DEVICES=6`; a current task
must use the device mask from its own contract and preflight instead of copying
that historical setting.

## Confirmed operator results

- [GLA chunk GLA: fuse BF16 K=V=64 intra-tile output](gla/c003-fuse-bf16-k-v-64-gla-intra-tile-output-on-gfx936.md) — confirmed 33.83% aggregate latency improvement on the validated panel; scope and limitations are in the report.

## Optimization guides and examples

- [HY3 prefill GQA and split-KV](hy3-prefill-gqa-split-kv.md)
- [MMAC/FMA path selection and multi-row fusion](mmac-fma-path-selection.md)
- [Triton multi-row program example](examples/triton-multirow-program.md)
- [Hygon gfx936 profiling workflow](profiling-workflow.md)
- [Using Triton tune_configs.yaml for Hygon](tune-configs.md)
- [Validated gfx936 architecture, ISA, and calibration](gfx936-validation.md)
- [Profiler skill procedure](../../../skills/hygon-profile/SKILL.md)

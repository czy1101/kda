# 证据驱动的候选与恢复

先读 [瓶颈判别](bottleneck-diagnosis.md)。为一个候选写：父版本、观察/证据、一个主要变更、预期产物与计时变化、正确性/资源风险、反证、测试命令和预算。不要无依据统一调 block/warps/stages。

矩阵与搬运优先评估 Tensor Cell/AIU，但核验 dtype、accumulator、tile、真实 stride、对齐、mask、Shared consumer 与 wait；专用路径生成不等于获益。通用访存优先必要流量和局部性；归约/不规则算子还要核验 Warp 参与掩码、地址整数范围和同步。硬件边界查询知识库相应主题，并在当前版本验证。

## Require a causal chain for each candidate

Write the chain before implementing: formal shape and timing scope → measured slow kernel → expensive computation or transfer inside it → compiled evidence → competing bottleneck hypotheses → discriminating experiment → one code change → predicted compiler and timing effects. If a link is missing, gather that evidence first. Do not call a change “PPU-specific” merely because it runs on PPU; identify the device behavior or instruction path that the change uses.

After a targeted change, inspect the exact candidate specialization and compare its kernel time, neighboring producer/consumer time, and formal end-to-end latency against the same parent. Explain disagreement between these measures before adding another change. In particular, a fused kernel may be longer even though one launch disappears; a lower register count may have no time benefit; and a faster microbenchmark may fail to improve the operator. Keep a useful, correct microbenchmark result as diagnostic evidence without promoting it to a full-operator claim.

## 复用 KDA 闭环

1. 保存原始状态；复用 baseline/best checkpoint，保留用户已有未提交内容。
2. 只编辑 `constraints.allowed_paths`；远端编辑镜像后受控 push，不能直接上传绕过 hash guard。
3. 先 `kernelpilot.py run correctness --workspace W`，完整遵循项目接口、容差、shape/stride/layout。失败立即恢复 best，不测该错误实现的“收益”。
4. 正确才执行契约 benchmark 与 generalization，串行、同设备、完整必需矩阵。
5. 把每次尝试写 `runs/candidates.jsonl`，包括失败 reason、failure_category、rollback；正式结果写既有 benchmark.csv，单位、方向、比较器、聚合完全遵循契约。
6. 只有原有晋升门禁允许，才 `checkpoint save-best --candidate <id>`。退化/不稳定/超预算恢复 best，保留反证并修订假设。
7. 最后明确修改文件、收益范围、退化行、未解问题与停止原因；不能用整体 git reset/clean 撤销用户改动。

辅助比较脚本不会晋升候选。不要重写 KDA checkpoint 机制；未达目标可以保留原始 baseline 为最佳。不要为证明 Skill 可运行而擅自修改真实算子项目，闭环验证在隔离 fixture 任务中完成。

## Retention, promotion, and fusion tradeoffs

Treat these as separate outcomes: an experiment can be archived for later work; a correct candidate can be explicitly retained despite disclosed regressions; only a candidate satisfying the current task contract can become KDA `best`. If the user changes the acceptance policy, record the new criterion, affected shapes, and decision in the task workspace before evaluating a new promotion. Never relabel an old strict-contract failure as a strict-contract pass. Preserve the earlier result and append the later decision with its evidence. Do not replace a newer, better deployed implementation with an older retained intermediate version merely to mark the latter as accepted.

When eliminating an intermediate kernel, write the before-and-after dataflow in terms of actual tensors and kernels. Measure the removed kernel, its consumer before and after fusion, and the full operator. Check whether the fused consumer repeats work, changes precision or accumulation order, increases register/shared-memory use, or reduces grid parallelism. A saved launch or global-memory round trip is a hypothesis for a net gain, not the gain itself. Keep a fallback path for unsupported device features and verify the path on the intended runtime.

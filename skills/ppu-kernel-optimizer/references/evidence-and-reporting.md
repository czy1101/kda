# 证据文件、安装与交付

## 文件与事实协议

KDA 原有 `runs/benchmark.csv`、`runs/candidates.jsonl`、`runs/analysis-capabilities.json`、`runs/analysis-strategy.json` 和 `.kernelpilot/checkpoints/` 是规范来源；功能证据为 `runs/analysis-functional.json`。不复制第二套正式台账。

每次实验在任务工作区新建 `profile/<kernel>/<run-id>/`，保存 environment、原始 timing 样本、compiler/manifest、microbench、hypotheses、对照与报告。目录名使用明确运行标识，不能用“最新文件”隐含归属。工具功能日志位于 `profile/tool-validation/<tool>/`，功能 JSON 引用它们。失败、超时和空报告同样保留。

`generate_report.py --facts <json>` 接受实际采集事实字段：`object/environment/baseline/hypotheses/experiments/optimizations/correctness/interpretation/next_experiments/commands/evidence_paths/conclusion`。缺失字段显示未采集，不补测量。假设条目记录 grade、observation、evidence_paths、alternatives、refutation、status、result、next_experiment。禁止把计划写成结果，或将 fixture 的数字注入真实 benchmark。

生成器面向完整审计，把已有 ledger 全部保留；如果用户只要求单份成功方法总结，使用 `operator-optimization-report` 的边界调整最终文档，失败仍留内部证据。摘要不能掩盖最终实现的退化行。

## 报告解释顺序

工程结论先回答当前最可能限制与证据等级、是否改善、下一动作。深入说明按观察→怀疑原因→验证命令/变量/结果→证明范围→优化原理→正确性/性能/波动/其他 shape→剩余疑问。无计数器时不生成虚构 occupancy、stall、hit-rate 或理论峰值。

完整报告覆盖对象、设备/软件、可用和失败工具、baseline、瓶颈证据、反证、优化及失败尝试、正确性、全矩阵比较、未解问题、复现命令和证据。Level 4 必须明确：未测 baseline，无当前设备加速比。

## 安装与路由

Skill 源码在 KDA `skills/ppu-kernel-optimizer`；backend `always` 确保没有 profile command 也加载诊断规则。实际 Codex 发现还需用户级链接；不自动安装或改用户配置。POSIX 可由用户执行：

```bash
ln -sfn <absolute-kda>/skills/ppu-kernel-optimizer ~/.codex/skills/ppu-kernel-optimizer
```

Windows PowerShell 可由用户在允许创建 junction 的环境执行（目标目录应尚不存在）：

```powershell
New-Item -ItemType Junction -Path "$env:USERPROFILE/.codex/skills/ppu-kernel-optimizer" -Target 'C:/Users/songchao/Desktop/JL_work/projects/jlwork/kda/skills/ppu-kernel-optimizer'
```

先确认当前安装规范、路径和用户设置，链接后重载并用 `/skills` 验证。项目路由不等于 Codex 已安装。不要自动提交或发布。

## 无设备验证与目标机复现

在仓库运行 `python -m unittest discover -s tests`；使用可用 skill-creator 的 `quick_validate.py skills/ppu-kernel-optimizer`。fixture 仅验证机制，不作为硬件通过证明。

目标机经授权后的顺序：填任务 target（ppu/ppu001/PPU-ZW810E/triton）→环境 preflight→正确性→正式 baseline→工具发现→必要时功能验证→编译归档→一项反证→允许时单因素候选→正确性/benchmark→恢复/晋升→中文报告。

微基准先计划：`run_microbenchmarks.py --output <new-dir> --modes copy --sizes 4096 65536 --blocks 256`。当前执行端经授权才加 `--execute`；GEMM 单独用 `--modes gemm --sizes 128 256 --blocks 32 --dtype bf16`。每 worker 限时，总预算/分配有界；未支持配置标失败并保留编译错误，不冒充已验证 backend 语法。受限环境不能真实运行时将设备验证列为 skipped，留下这些命令和前提。

## Evidence needed for a version comparison

For every reported step, state the exact parent and candidate source hashes, correctness status, benchmark hash, shape/dtype coverage, device and card, run order, timing scope, sample counts, per-run medians, pooled summary, and restoration state. Give separate tables for each dtype when both are part of the formal matrix. A useful row contains parent latency, candidate latency, latency reduction, optional reference latency, and reference/candidate speedup. Define the ratio so values below one are not mistaken for improvements. State whether the original baseline and final version were measured together or in different sessions.

Keep observations separate from explanations: a stage time and compiler resource count are measured facts; the hardware bottleneck inferred from them is a hypothesis until a controlled change supports it. For a candidate with a small negative row, show both the complete matrix and a focused paired retest; preserve a user-authorized aggregate tradeoff as a separate decision from the task's original no-regression gate. The report should make clear which implementation is currently deployed and which retained variants are only archived.

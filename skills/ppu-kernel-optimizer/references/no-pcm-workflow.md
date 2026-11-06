# 可降级分析路径

| Level | 必要条件 | 能回答 | 不能据此断言 |
|---|---|---|---|
| 1 | 当前 acu capture/export 验证通过且任务允许 | 请求的计数、资源、调度问题 | 未采字段、另一个 kernel 的行为 |
| 2 | 基础 asys Trace 实测有效 | launch、API→kernel、同步、copy、时间分布、可观测重叠 | 指令利用率、cache 命中率 |
| 3 | 目标 kernel 能正确执行并可靠计时 | 规模、复用、布局、并行度、依赖等受控实验 | 真实 stall 百分比、硬件 occupancy 数值 |
| 4 | 无设备或不能运行目标 kernel | 源码、已有产物、数学工作量、实验计划 | 当前设备延迟、加速比、已确定动态瓶颈 |

1. 检查已有功能证据。用户已说明本轮 acu 不可用时，记录现有错误并直接降级，不为了流程再触发 PCM。若已有失败记录，`run profile` 与验证脚本会拒绝重试。
2. 只有基础 asys 能区分当前问题、且有明确授权命令时才测试。asys 失败继续 Level 3。计时、编译与微基准可单独运行，不依赖 profile command；主 Skill 由 backend `always` 加载。
3. Level 3 先正式 baseline，再估算逻辑工作量/字节量并建立至少两种解释。例如“访存吞吐”与“并行度不足”；改变 grid/规模以及复用，比较是否符合预期。
4. 每次选择一项主要控制变量，记录反证，结果冲突则修订。无新证据不无限扫描参数；遵循候选数、时间、采集预算和停止条件。
5. Level 4 用源码分析和微基准计划生成脚本，记录 baseline 未测。已有历史产物只说明历史 specialization，不填当前性能表。

## Establish a mechanism without PCM counters

Level 3 is not limited to source-based guesses. When the question is why a measured change works, use [causal mechanism verification](causal-mechanism-verification.md): separate software factors, compare runtime mappings within a verified identical binary where useful, probe supported device timestamp/execution-ID reads, and collect actual-kernel intervals when instrumentation is valid. Then remove or weaken the hypothesized cause in a counterfactual. These experiments do not require a successful PCM capture, but still require authorized execution and instrumentation checks. They cannot produce uncollected cache, DRAM, stall, or hardware occupancy counters. If the probes or controls fail, preserve the gain as measured and leave its mechanism unresolved.

`analysis.profiling: required` 失败会让 strategy 报错，不能擅改为 auto；可另外交付替代分析，仍报告未通过阶段。`disabled` 不运行采集。显式固定 `profile.tool: acu` 时失败不会擅自改成 asys，可按既定 fallback 做无 PCM 分析，或在授权后更新任务选择。

典型分析请求：`使用 $ppu-kernel-optimizer 分析任务契约中的 GDN kernel；本轮 acu 已报 600，不改正式实现，用无 PCM 证据解释瓶颈并输出中文报告。`

授权优化请求：`使用 $ppu-kernel-optimizer，按当前契约允许路径和预算优化，先完整正确性、再 benchmark；失败恢复 best，保留所有候选和反证。`

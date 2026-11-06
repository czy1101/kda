# 可反证的瓶颈实验库

每条是候选解释，不是自动判定规则。实际记录需含观察、输入、候选/源码身份、命令、原始结果路径、固定变量、替代原因、反证与结论更新。未执行标 planned。

## Locate the bottleneck before proposing code changes

1. Reproduce one formal invocation with the same inputs, dtype, shape, selected specialization, and device. Record every launched kernel in execution order, its producer and consumer tensors, and whether host preprocessing, allocation, or synchronization lies inside the formal timer. If a vendor timeline is unavailable, instrument the existing path with device Events and mark that timing as diagnostic. Keep a separate uninstrumented end-to-end baseline.
2. Choose representative shapes that exercise each distinct execution path, not just the smallest or fastest case. For each, make a stage table with kernel name, call count, median time, share of the measured operator total, and the actual source/compiled specialization. Identify whether the same kernel remains dominant as batch, sequence length, heads, or dtype changes. Do not add instrumented stage medians and call the sum the formal score.
3. Rank the expensive stages and calculate a rough opportunity bound: if a stage occupies 15% of the total, eliminating that stage alone cannot yield a 30% end-to-end latency reduction. Account for any work moved into its consumer. Select a specific inner loop, data transfer, dependency chain, or launch boundary to investigate, then inspect its compiled path.
4. Write at least two plausible causes and a test that separates them. Examples: compare unchanged arithmetic with a different work partition to test parallelism; vary reuse at fixed arithmetic to test traffic; vary dependent versus independent chains to test latency; compare an isolated operation with its fused parent and consumer. Ensure the compiler did not remove the tested computation and that output is consumed or checked.
5. If a control experiment contradicts the proposed cause, update the diagnosis. Do not continue tweaking tiles, warps, stages, or instructions merely because a previous hypothesis sounded plausible. When evidence remains weak, report the current stage breakdown and the next discriminating measurement instead of claiming a hardware bottleneck.

Use profiler counters only after checking their device-specific definitions. When counters are absent, compiler output plus controlled timing can support a hypothesis, but cannot produce a numerical stall percentage, achieved occupancy, or HBM traffic measurement.

For a measured grid/order gain, do not stop at "the code is unchanged, so scheduling must explain it." Read [causal mechanism verification](causal-mechanism-verification.md) to distinguish compiler effects, actual task placement and completion tails, and a counterfactual that removes heterogeneous work. Equal total work does not imply equal parallel completion time, and differing task lengths do not imply inter-CTA dependencies.

| 候选与原理/症状 | 直接证据及无 PCM 判别 | 替代解释与反证 | 对应优化及风险 |
|---|---|---|---|
| Global bandwidth：大量必要流量，延迟随字节量近线性 | 有效 counters 的流量/带宽；否则计算最小逻辑 bytes/time，与同设备同范围 copy/read/write 比较，扫描规模/复用 | cache、地址开销、并行度不足也近线性；增复用无收益且扩 grid 有收益会削弱带宽解释 | 合并、减少往返/融合、复用；可能增寄存器或浪费 mask |
| Memory latency/依赖：单链 load→地址→load 暴露等待 | 有效 latency/stall；否则比较相同读数量的独立链与依赖链、增加独立实例 | 吞吐、分支/取模成本；独立化没有收益可否定链是主要限制 | 并行链、prefetch 候选、寻址复用；增加状态/资源 |
| Compute throughput：运算量大且复用高 | 匹配最终指令路径与有效管线 counters；否则扫同数据不同计算次数、矩阵尺寸并测可观察输出 | 转换、spill、launch；改变 FLOPs 不改变延迟会削弱吞吐解释 | dot 分块、消除转换、原生路径；数值和资源风险 |
| Launch：短 kernel/多次提交开销占比高 | timeline API→kernel 间隙；否则同步 host 与 Event 范围、次数/批量/合法融合对照 | 小 grid、同步或分配；设备端同样很慢则非单纯 launch | fusion/批量/已有 Graph；端到端范围、生命周期约束 |
| Register pressure：活跃值/展开使驻留降低或 spill | 编译 metadata 直接证明数量，spill 指令；再固定工作量改展开/tile/活跃区间计时 | 指令和复用也变；寄存器显著下降而延迟不变可反证主因 | 缩活跃区间、分阶段；重复 load 或计算增加 |
| Shared pressure：stage/tile 占用影响并行或访问冲突 | shared 字节、两端地址/指令，有效冲突计数；否则等数学工作量布局与 stage 对照 | bank/group、转换、同步；减 shared 无改善则不是容量单因素 | padding/swizzle/stages；增加容量、producer 好但 consumer 变差 |
| Layout conversion：数据表示不匹配引入重排 | 归属 IR/PASM 转换序列；保持数学与访存量改变布局并计时 | 访存合并、矩阵路径同时变化；移除 conversion 却无收益则不是主因 | 统一生产/消费布局；可能损伤 load/dot 映射 |
| Cache/working-set：复用距离引发敏感性 | 有效按层 counters；否则热态重复、规模/访问顺序/步长扫描 | TLB、grid/调度、对齐；改变访问模式后拐点消失则不能归为固定容量 | tile 复用/局部性；小样本只适合 cache，不泛化 HBM |
| Instruction overhead：寻址/分支/转换主导 | 最终指令与有效 mix；否则等数据量简化索引或类型转换的对照 | 编译优化删除计算、访存路径变化；指令少但时间无变则弱化 | 公共表达式、减少除模/转换；整数范围/精度风险 |
| Synchronization：barrier/wait/跨 kernel 等待 | 真实 timeline 或匹配同步 counters；否则检查依赖、合法移位等待对照 | 资源/内存流量、负载不均；移除冗余等待不改善则不是主因 | 减冗余同步/拆分；必须证明可见性，不能删必要 barrier |
| Parallelism/occupancy：grid 太小或资源限制隐藏延迟 | 真正 occupancy 指标；否则 program 数/工作分解扫描、编译资源联查 | 带宽饱和、tail mask、频率；增加独立实例无益则削弱假设 | grid 重分解/适量 warps；不允许报告推测的实际 occupancy 百分比 |
| Pipeline：传输依赖无法被计算覆盖 | 有效 timeline/counters 与静态 wait 序列；否则 stages/buffer/工作量对照 | shared/寄存器增长、编译无改变；多 stage 仅增资源无益则撤销 | 合法双缓冲/调度；正确性、驻留和容量风险 |
| 中间搬运：多 kernel 临时张量往返和 launch | timeline 分布、数据流及必要 bytes；否则 fused/unfused 全范围比较 | fusion 后 spill/计算重复；总延迟不降则融合没有净收益 | fusion/减少中间结果；全输入精度、资源/并行度损失 |

## 证据范围

- A：直接可观测的具体事实；例如 metadata 报 64 个寄存器，不代表“寄存器导致慢”也是 A。
- B：多项独立受控实验支持，并排除主要替代解释。
- C：一致现象，但干扰未排除。
- D：经验/代码提出，待执行实验。

每个主要判断至少设计一项反证。有条件时实际运行优先反证，再优化。结论与数据冲突就更新 hypotheses，不为保留原解释忽略退化。等级不换算成置信概率。

带宽特别注意：逻辑最少字节是模型，实际 HBM 流量不可知；不要据有效带宽接近某 microbench 就写“DRAM 饱和”。微基准观测峰值不是物理峰值；未知 TFLOPS 不填写 roofline 顶板常数。

# 可信计时与受控比较

正式 benchmark、比较器与晋升一律使用任务契约。辅助 adapter 只补充诊断，不改 `runs/benchmark.csv` 的格式或指标。吞吐 maximize 与延迟 minimize 不得用同一方向公式；聚合由 KDA 重算，不能手写 speedup 作为晋升证据。

## Adapter API

在任务工作区创建已授权 Python adapter，定义：

- `prepare()`：分配/JIT/autotune 初始化，不计入样本。
- `check()`：执行原有正确性检查，失败抛异常；辅助脚本通过不替代完整 correctness 门禁。
- `launch()`：准确的测量范围，禁止隐式换输入或启动新搜索。
- `synchronize()`：同步实际使用设备与 stream，不使用无关 CPU barrier。
- `describe()`：返回 JSON 字典，含 `shape/stride/dtype/device/environment_fingerprint/config/config_fixed/scope/cache_protocol/execution_kind`；另记录 runtime/compiler/driver、device mapping、源码版本、autotune 选择和其他负载线索。CPU fixture 的 execution_kind 必须是 fixture，不能写 ppu。
- Event 模式额外定义 `events()`，返回同一 launch stream 上支持 record、synchronize、elapsed_time（毫秒）的已验证 start/end 对象。

`measure()` 分离准备及 warmup 时间，在正式样本前后检查 describe 是否变化。host 模式每批前后设备同步，包含 host launch；Event 模式记录同 stream 区间，但可能包含提交间隙，不等于单条指令时间。批量结果除以 batch 表示平均每次范围，不能据此称为单次 p10。原始样本与 median/P10/P90/CV 同时保存。零时长拒绝，避免分辨率不足导致无限加速比。

CLI 子进程有总超时；每 kernel 工作量必须自身有界，超时不代表设备立即空闲。正式对比串行使用同设备，记录温度/频率/其他负载可见线索，不为测量改时钟或停其他任务。发现高波动先找干扰，不能挑最有利轮次。

## JIT/autotune/cache

预热必须覆盖实际 specialization 和 autotune 搜索。固定比较运行条件，记录每次实际配置；源码修改触发新编译只在准备阶段。无法固定时 config_fixed=false，比较器拒绝直接收益结论。合法配置优化可改变 warps/tile，但前后各自必须稳定，并记录改动变量。

默认微基准是 warm_repeated。没有验证 cache flush 就不写 cold；工作集变化只是敏感性实验。多 kernel 中间分配/拷贝、Graph、端到端范围和单 kernel 预分配计时分开，不自动相加阶段耗时。

`compare_experiments.py` 核对输入、设备环境、范围、cache、计时协议和配置稳定性；可变配置内容留作实验变量。输出诊断 ratio 与波动；默认 CV .05 仅为辅助判据，P10/P90 分离不是显著性概率，也不替代任务稳定性标准与全 shape 正式比较。

## Formal benchmark alignment and paired retests

Before comparing versions, inspect the benchmark code rather than inferring its timing scope from its name. Record which tensors are prepared outside the timer, whether the timed callable includes allocations or host preprocessing, which provider runs first, how `do_bench` warms up and samples, and whether the result is a per-call median, a graph replay, or an average over a batch. Keep the customer's benchmark unchanged. An instrumented wrapper may collect its raw samples, but must call the same workload and timing method; label any added fixed-iteration Event measurements as diagnostic.

For a full-matrix decision, identify each row by shape, dtype, source hash, benchmark hash, device mapping, and run order. Retain per-process raw samples and medians for both parent and candidate. A pooled median can hide a slow process, so report the individual runs before interpreting a small gain or regression. If the parent and candidate were measured far apart, or the decision hinges on a sub-percent difference, interleave them under the same device and cache protocol (for example A/B/B/A). Use a same-run comparison provider as a drift control when the formal benchmark already includes one; if both provider and candidate slow in one run, flag that run for investigation, but do not invent the cause or discard it silently.

An isolated kernel Event time and an end-to-end benchmark measure different scopes. Stage times need not sum to end-to-end latency because launch gaps, host work, allocation, synchronization, or profiler overhead may differ. For a fusion, compare both the removed stage plus its consumer and the full operator. Record the consumer's longer runtime as well as the eliminated launch and intermediate traffic. Confirm the actual executed path and specialization before attributing a speedup to the source edit.

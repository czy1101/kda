# 编译产物与归属

直接复用知识库 §9 已知 FlagTree 0.7/Triton 3.6 的 `source/ttir/ttgir/llir/hgbin/cubin` 与 `hgobjdump --isa` 接口，不重跑通用支持性探针；每个候选归档真实存在的 keys，遇到实际版本差异再检查安装源码。`cubin` 可以是兼容名称，不能推断为 NVIDIA binary。

## 采集协议

adapter 的 `compiled()` 返回 `(已 launch CompiledKernel, provenance)`。provenance 必须记录 `source_sha256/signature/shape/dtype/stride/grid/config/target/versions/candidate`；config 包括实际 autotune 与 warps/stages、mask、指针偏移/对齐及编译选项。需要时补 kernel 名/cache key。不能按缓存时间猜归属。

`collect_compiler_artifacts.py` 不编造 dump 开关，逐一保存 asm mapping 现有 str/bytes，metadata 原文与公开资源；不提供值为 null，真实 0 保留。目录必须新建，避免覆盖不同候选。资源字段名变化时补当前后端 adapter，不把未知转换为 0。

`analyze_compiler_artifacts.py` 检查摘要后输出静态文本特征，`--compare <manifest>` 保存逐 key 完整 unified diff；binary 保留原始摘要而不假装文本反汇编。需要 PASM 时先核验 hgobjdump help，再由契约 low-level command 使用实际参数；保存 stdout/stderr/退出码、工具版本与 binary 摘要。没有输出不等于没有指令。

## 分析顺序

1. program/thread/block/grid 与 tensor layout，lane 地址、mask、尾块与真实 stride。
2. load/store 宽度、重复寻址、global 往返、shared producer/consumer、layout conversion。
3. dot dtype/accumulator/layout 是否落到预期 Tensor Cell；AIU 是否有完整搬运、等待与匹配消费端。
4. 类型转换、重排、循环/分支、地址算术与 instruction overhead。
5. 寄存器/shared/spill、stage buffer 与 grid，fusion 是否增加资源代价。
6. 发起→计算→等待→消费的依赖；静态 async 不等于动态重叠。

编译器可证实“这个候选生成了某序列”，不能直接证实“该 stall 占 40%”或“寄存器是主瓶颈”。资源原因需变量扫描、相同工作量计时及替代解释对照。源码 AST 分析只记录调用与行号，不自动推出 memory-bound。

## Connect the slow operation to what the PPU executes

Start from the measured kernel name and its actual launch arguments. Archive the compiled object after it has launched; include the exact shape, dtype, grid, autotune choice, source hash, and cache key. An artifact from a similar source file or an unselected configuration cannot explain the measured kernel.

Trace the suspected operation across source → TTIR → TTGIR → LLVM IR → final ISA where available. For each transition, note whether a dot remains a matrix instruction, which precision and conversion path it uses, whether loads become repeated or scalarized, where shared-memory layout changes and waits occur, and whether a loop is unrolled or serialized. Record register, shared-memory, and spill metadata for the same compiled object. If an artifact is unavailable, say which layer is missing rather than filling in an assumed instruction sequence.

Before editing production code, write a prediction tied to the artifact: for example, “removing this repeated conversion should shorten kernel X while leaving its input traffic unchanged.” After the edit, diff the same specialization at each available compiler layer and remeasure kernel X and the full operator. A source-level simplification that compiles to the same instruction sequence is not a demonstrated hardware optimization. A changed instruction sequence without a stable timing effect is a compiler observation, not a measured speedup.

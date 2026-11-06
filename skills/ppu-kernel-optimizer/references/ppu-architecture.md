# 直接复用的真武 810E 知识

本页将 [真武810E_知识库](../../THead/真武810E_知识库.md) 中会改变优化决策的内容嵌入 Skill。来源为该库 2026-10-17 的官方资料整理和已验证记录，保留其证据范围；无需为使用这些知识重新运行硬件探针。完整细节和上游出处按原知识库章节查询，不另建一套硬件验证清单。

## 架构与资源预算（原库 §2–3）

| 内容 | 直接采用的知识 | 使用边界 |
|---|---|---|
| 命名 | 硬件 `ppu001`；HGCC `ppu_10`；宏 `__HGGC_ARCH__=100`；ACTLIZE `PPU0010` | 不把 `ppu0015/ppu_15/tc02` 能力归入本代 |
| 执行 | Warp32；原机 64 CU；每 CU 2048 线程、每 block 1024 线程 | 是预算上限，不是当前 kernel 实际驻留 |
| 显存 | HBM2e 标称 96 GB，原机可见 97920 MiB；板级 2765 GB/s | 不当作有效 kernel 带宽；没有已知完整 TFLOPS 表则不填 |
| Shared | 原机上报 262144 B/CU 与 B/block；32 banks、4 B/bank；2 groups、每组连续跨度 128 B | 属性不是满容量 launch 保证；读端和写端都需考虑 |
| VREG | 原机原生/PyTorch 属性均为 131072 条目/CU | 文档另有 64K 冲突；不推断物理字节数和未知分配粒度 |
| Cache | L1 属 CU、L2 属 CE、LLC 跨 CE；兼容属性上报 L2 67108864 B | 不据此确定物理层容量归属；拐点不是命中率 |

block 整体分到一个 CU；常规 block 的执行次序和同时驻留无保证，不用跨 block 自旋构造普通全局 barrier。VREG 是 per-thread 状态，SREG 是 Warp 公共状态；uniform 地址/循环量可能减少向量寻址，但不能把不同 lane 值强转 uniform。驻留预算同时看线程、寄存器、Shared 与分配约束；增加 occupancy 不是独立优化目标。

## 计算路径（原库 §4）

- Tensor Cell `tc01` 是 Warp 协作矩阵乘加。FP16/BF16 原生 TIX shape 包括 `(16,16,16)/(16,16,8)/(8,16,16)/(8,16,8)`，可有 FP32 累加；FP32/TF32 为 K=8/4 的对应形状；INT8 为 K=32/16、INT32 累加。前端支持是子集，不能把原生 fragment 等同任意 Triton tile。
- `tc01.ex` 是软件扩展 shape，不能视作与一条原生矩阵指令同代价。`tl.dot` 的实际 lowering 用具体候选产物解释；不为了“再次确认平台有 Tensor Cell”运行探针。
- 2:4 稀疏需要压缩 B、位置 metadata、psel/ssel 与正确数值语义；收益必须包含准备成本。
- Packed FP16/BF16 可用两路 16-bit 算术/FMA。原库已对相关小型路径做完整对拍与反汇编；这证明能力存在，不代表普通表达式必然自动 packed 或任何算子更快。
- DP4A/DP2A、字节排列、位操作按真实 signed/unsigned 与打包格式选择。SFU `ex2/lg2/rcp/rsqrt/sqrt/tanh` 接口存在不代表高层函数只需一条物理指令。
- 不改任务累加精度、舍入或误差要求；快速数学与 FP32/TF32 不能默默互换。FP8/MXFP4 等另一代特性不能从 SDK 类型存在推导出来。

## AIU、Shared 与流水（原库 §5–6）

AIU bulk 将 Global tile 搬入 Shared，减少逐 lane 地址/寄存器中转；原库已验证 `hggc_aiu.h` 的 `aiu::cube_load_params/tsm_load_cube` linear 路径，产生 `vmem.aiu.ld.tsm...3d.b32...` 与 `s.wait commit_group(0)`，完整对拍通过。无需重测该能力是否存在；当前候选是否触发、是否获益另看其产物与 benchmark。

- 本代 TIX bulk 语法 `.2d/.3d`，元素 `.b8/.b16/.b32`；不另造原生 1D 接口。
- swizzle 目的起点 128 B 对齐，linear 文档要求 4 B；C 按 32 B slice 补齐，一条最多 4 slices/128 B，超过需分片。
- swizzle 行批量 `cube_n*cube_h*cube_w` 按 8 的倍数预留；consumer 应使用匹配 `tc01.ldmatrix.swzl`，不能当普通线性布局。
- H/W padding 与 C 方向 padding 约束不同；转置支持也随元素宽度不同。不要复制隐藏操作数或其他厂商 API。
- 异步 copy、AIU、commit/wait、barrier 的完成/可见性语义不能互代。多 stages 必须有独立 buffer、合法等待与生命周期；更多 stages 可能增加 Shared/VREG、降低并行度。
- 对普通标量访问可用 `(byte_address/4)%32` 初步分析 bank；还看跨 Warp group 冲突及向量/矩阵 consumer，不从生产端良好推断消费端无冲突。

缓存策略使用 PPU 语义：`.ca` 缓存各级，`.cr` 绕 L1，`.cg` 绕 L1/L2 而缓存 LLC；不从 CUDA 同名参数继承含义。本代支持 L1 独立 prefetch；另一代 load/AIU 到 LLC 的额外 prefetch-size 提示不归入 810E。cache hint 不替代内存一致性。

## 通信、原子与运行时（原库 §7–8）

Warp32 shuffle/vote/redux 使用合法参与 mask；尾 Warp、分支后的活跃 lane 和数据来源必须满足语义。跨 Warp 交换通常需 Shared 与正确块级同步。

Global native Add/Min/Max 包含 U32/I32/U64/I64/F32/F64；Shared Add/Min/Max 为 U32/I32。Shared Exchange/CAS 为 B32/B64，原库特别提示本代 Shared CAS 代价较差。高层 Shared F32 Add 不等于 native 单指令；不把另一代 native F16x2/BF16x2 原子归入本代。优先减热点竞争与 Warp/Block 聚合，保证排序、scope 与发布协议。

HGGC 有 stream/Event/Graph、异步 copy 等接口，Graph 可作为重复提交开销候选，不能由接口存在宣称有并发收益。ICN 标称 700 GB/s 不代表任意卡对实测吞吐。MIG/vGPU 可能改变可见资源；只记录任务实例状态，不改变切分配置。

## 已有软件与工具知识（原库 §9–11）

历史同容器栈：PyTorch 2.10.0、FlagTree `0.7.0+ppu.git47970467`、Triton 3.6.0。兼容 target 可为 `cuda/80`，真实 metadata `backend_name=ppu`，最终 binary HGBIN。已验证产物入口是 `source/ttir/ttgir/llir/hgbin/cubin`，`hgobjdump --isa <hgbin>` 可反汇编；不要重复确认这些已知接口，直接用于任务归档。新候选仍保存其真实键与归属，缺失则如实报告。

`hgcc --help` 可用，`--version/-V` 在该安装报 unknown argument；不要把版本查询失败当编译器不可用。兼容 nvcc/HGGC 13.0 不等于 SDK 发布版本。acu 2.1 的 version/list-sections 成功不代表 PCM 可采，原库已有采集 not-ready/空报告失败记录；用户明确本轮不可用时直接降级，不重复调用 PCM。

本次扩展验证补充（2026-10-19）：同版本 asys 的 `profile -t hggc --sample=none --event-sample=none` 与 `stats --report hggc_ppu_kern_sum --format csv` 已成功，含匹配 vector 记录。工具输出可能有进度前缀，CSV 真实字段为 `Name`、`Med (ns)` 等，不能套用 `hggc_gpu_kern_sum`。这支持该环境基础 Trace 路径，不保证其他采样功能不需要 PCM。

知识使用规则：已记录事实直接采用；待验证、文档冲突和优化推导保留原标记。只验证新增实现/实际候选、未知接口和会影响本次结论的环境变化，不重跑通用硬件能力套件。

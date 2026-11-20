# FlagTree TLE 开发查阅手册

> 面向日常算子开发与后端适配的 TLE（Triton Language Extensions）分层 API 参考。
> 覆盖 TLE-Lite / TLE-Struct / TLE-Raw 三层，并单列**平头哥 PPU**与**摩尔线程 MThreads** 的详细能力矩阵。

| 项目 | 内容 |
|:--|:--|
| 文档性质 | 内部开发查阅手册（非官方文档），依据 FlagTree 公开源码与 Wiki 整理 |
| 代码基线 | `github.com/flagos-ai/FlagTree`，`main` 分支（Triton 3.6），整理时间 2026-11-18 |
| 依据来源 | `tle.md` / `tle_cn.md`、Wiki（`TLE`、`TLE-Raw`、`User-manual-*`）、`python/triton/experimental/tle/**`、`third_party/<backend>/backend/tle_supported.py`、各后端 TLE 单测 |
| 适用对象 | 算子开发、性能调优、芯片后端适配、CI 与回归排查 |
| 维护约定 | 上游 API 仍在演进，**以本文“源码签名”栏为准**；Wiki 示例存在滞后，差异见第 9 章 |

---

## 目录

- [0. 阅读约定与版本基线](#0-阅读约定与版本基线)
- [1. TLE 总览](#1-tle-总览)
- [2. TLE-Lite](#2-tle-lite)
- [3. TLE-Struct](#3-tle-struct)
- [4. TLE-Raw](#4-tle-raw)
- [5. 平头哥 PPU 能力矩阵](#5-平头哥-ppu-能力矩阵)
- [6. 摩尔线程 MThreads 能力矩阵](#6-摩尔线程-mthreads-能力矩阵)
- [7. 跨后端能力总表](#7-跨后端能力总表)
- [8. 排错与调试清单](#8-排错与调试清单)
- [9. 已知不一致与版本差异](#9-已知不一致与版本差异)
- [10. 参考资料](#10-参考资料)

---

## 0. 阅读约定与版本基线

### 0.1 记号

| 记号 | 含义 |
|:--|:--|
| ✅ | 该后端声明支持且已接入 lowering |
| ⚠️ | 有实现但存在明确限制，使用前必须读约束 |
| ❌ | 该后端不支持；调用会在前端或 `make_ttir` 阶段报错 |
| — | 该后端未声明，行为取决于具体 wheel 版本 |
| *声明* | 出现在 `tle_supported.py` 白名单中，仅表示前端允许构造 |
| *验证* | 出现在该后端的 TLE 单测/教程中，具备编译或数值证据 |

> **重要**：`tle_supported.py` 只是**前端可用性白名单**，不代表能成功 lowering。最终是否可用由 `backend/compiler.py` 的 pass 管线与被拒列表决定，见 [5.3](#53-关键约束与拒绝列表)。判断某原语是否真的可用，必须同时看“白名单 + 单测 + pass 管线”三处。

### 0.2 文件与代码位置

| 内容         | 路径                                                                                     |
| :--------- | :------------------------------------------------------------------------------------- |
| 语言入口       | `python/triton/experimental/tle/__init__.py`                                           |
| Lite 原语    | `python/triton/experimental/tle/language/core.py`                                      |
| Lite 流水    | `python/triton/experimental/tle/language/pipe.py`                                      |
| Lite 分布式   | `python/triton/experimental/tle/language/distributed.py`                               |
| Lite 通信上下文 | `python/triton/experimental/tle/language/communication.py`                             |
| Struct GPU | `python/triton/experimental/tle/language/gpu/core.py`、`gpu/types.py`、`gpu/semantic.py` |
| Struct DSA | `python/triton/experimental/tle/language/dsa/`                                         |
| Raw 语言层    | `python/triton/experimental/tle/language/raw/core.py`                                  |
| Raw 运行时    | `python/triton/experimental/tle/raw/runtime.py` 及各后端 `raw/<backend>/`                  |
| 原语总表       | `python/triton/experimental/tle/language/primitives.py`（`TLE_PRIMITIVES`）              |
| 后端白名单      | `third_party/<backend>/backend/tle_supported.py`                                       |
| 后端编译管线     | `third_party/<backend>/backend/compiler.py`                                            |
| 单测         | `python/test/tle/{unit,integration}`、`third_party/<backend>/python/test/unit/tle`      |

### 0.3 导入方式

```python
import triton
import triton.language as tl

# 分层导入（推荐，命名清晰）
import triton.experimental.tle.language as tle          # Lite：tle.load / tle.pipe / tle.device_mesh ...
import triton.experimental.tle.language.gpu as tle_gpu  # Struct：tle_gpu.alloc / tle_gpu.copy ...
import triton.experimental.tle.language.raw as tle_raw  # Raw：tle_raw.call / tle_raw.call_smem
```

常见等价写法：

- `tle.gpu.alloc(...)`、`tle.gpu.copy(...)` —— 通过 `tle` 包访问 `gpu` 子模块（教程与后端单测的主流写法）。
- `tle.alloc(...)`、`tle.local_ptr(...)` —— `third_party/tle/README.md` 中的旧式写法，等同 `tle.gpu.*`，**新代码建议统一用 `tle.gpu.*`**。

### 0.4 30 秒自检

```python
import triton
print(triton.__version__)                      # FlagTree 版本形如 0.7.x+<backend>3.6

import triton.experimental.tle.language as tle
print(hasattr(tle, "pipe"), hasattr(tle, "cumsum"))          # Lite 是否可用
print([m for m in dir(tle.gpu) if not m.startswith("_")][:20])  # Struct 可用符号

from triton._flagtree_backend import FLAGTREE_BACKEND
print("backend =", FLAGTREE_BACKEND)           # nvidia / ppu / mthreads / iluvatar ...

# 后端声明支持的原语白名单（安装后位于 triton/backends/<backend>/tle_supported.py）
try:
    from triton.backends.ppu import tle_supported          # 换成你的后端名
    print(tle_supported.TLE_SUPPORTED_PRIMITIVES)
except Exception as e:
    print("no whitelist:", e)
```

---

## 1. TLE 总览

### 1.1 设计动因

Triton 以 block 为粒度抽象硬件，隐藏内存层级、布局、流水与同步，难以覆盖两类场景：

1. **DSA / 新架构适配慢**：不能表达异步搬运、专用存储层级、跨 block 通信。
2. **深层调优表达力不足**：与 TileLang 等相比，缺少对内存层级与并行粒度的细粒度控制。

TLE 的取舍是**分层扩权**：不破坏 Triton 编程范式，按用户能力与调优深度逐级放权。

### 1.2 三层结构

| 层 | 设计哲学 | 面向用户 | 代码侵入 | 可移植性 |
|:--|:--|:--|:--|:--|
| **TLE-Lite** | write once, run anywhere；用语义 hint 引导编译器启发式 | 算法工程师、快速优化 | 极小（在 `tl.*` 基础上加参数或替换同名原语） | 最高，跨后端 |
| **TLE-Struct** | architecture-aware；按硬件族（GPGPU / DSA）暴露结构化并行与存储 | 熟悉硬件的性能工程师 | 中等（显式 SMEM/barrier/warp 分工） | 中，需按后端族选 API |
| **TLE-Raw** | native passthrough；内联厂商原生代码 | 专家 | 大（直接写 MLIR / CUDA / 厂商 DSL） | 最低，绑定单一后端 |

### 1.3 编译路径

```
              ┌────────────────────────────┐
TLE-Lite  ───▶│                            │
              │  Triton frontend → TTIR    │──▶ FLIR / TLE dialect ──▶ LLVM IR ──▶ 目标二进制
TLE-Struct ──▶│      (CommonIR 可选)       │
              └────────────────────────────┘
TLE-Raw ─────▶ 独立编译（MLIR / 厂商工具链）──▶ LLVM 模块文本 ──▶ DSLRegionOp 注入主 IR ──▶ LLVM IR
```

要点：

- **Lite / Struct** 走 TLE dialect → `TleToLLVM` 转换；NVIDIA 还接入可选 **CommonIR**（`triton._common_ir.ENABLED`）路径，此时 Tile 级 buffer 用 `tile_*` 系列 op 表达，**仅支持 smem**，且不支持 `alloc(alias=...)`。
- **Raw** 先由厂商流水线独立编译成 LLVM 文本，再以 `tle::DSLRegionOp` 注入到 Triton IR 中，因此可以与 Lite/Struct 混写。
- 三者最终链接进同一个 kernel，由运行时加载执行。

### 1.4 典型 TLE pass 管线（GPGPU 族）

NVIDIA / MThreads 的 TTGIR 阶段大体为：

```
add_early_assign_memory_space
add_select_encodings
add_insert_local_pointer_barriers
add_optimize_local_pointer_loads
add_optimize_local_pointer_stores
```

> 这五个 pass 的**顺序是语义不变量**（TLE 官方 skill 明确要求）。写后端适配或改 lowering 时不要随意重排；确实需要时必须补回归测试。

PPU 在此基础上额外有 `add_tle_promote_async_load_to_aiu`、`add_promote_local_store_staging`；MThreads 额外有 `add_tle_lower_sqmma`、`add_tle_lower_pipe`、`add_tle_lower_tme_transactions`、`add_tle_prepare/finalize/lower_warp_specialize`。

---

## 2. TLE-Lite

TLE-Lite 的核心主张：**用 hint 而非约束**。所有原语在不支持的后端上应退化为等价 Triton 语义，不应改变数值结果。

### 2.1 `tle.load`

**源码签名**

```python
tle.load(pointer, mask=None, other=None, boundary_check=(), padding_option="",
         cache_modifier="", eviction_policy="", volatile=False, is_async=False)
```

| 参数 | 说明 |
|:--|:--|
| `pointer` | 指针张量、标量指针或 `tl.make_block_ptr` 得到的 block pointer |
| `mask` / `other` | 边界掩码与掩码外填充值；block pointer 模式下必须为 `None` |
| `boundary_check` / `padding_option` | 仅 block pointer 有效；`padding_option ∈ {"", "zero", "nan"}` |
| `cache_modifier` | `""`/`.ca`/`.cg`/`.cv`（NVIDIA PTX 语义） |
| `eviction_policy` | L2 驱逐策略提示 |
| `volatile` | 关闭缓存复用 |
| **`is_async`** | TLE 扩展：给 `tt.load` 打 `tt.load.async` 属性，交由后端决定如何兑现 |

**语义**：`tle.load` 先原样调用 `tl.load`，再附加 `tt.load.async` 布尔属性。属性本身不承诺异步，**是否真正异步由后端**决定：

| 后端 | 兑现方式 |
|:--|:--|
| NVIDIA | `add_lower_async_load` 走 TMA / `cp.async` 路径 |
| MThreads | `add_tle_lower_async_load` |
| PPU | `add_tle_promote_async_load_to_aiu`：**仅 block pointer + `is_async=True` + 非 int64** 才提升为 `aiu_load`（AIU 1.x，`.2d.b8/b16/b32`） |
| 其它 | 属性被忽略，退化为同步 load |

**用法**

```python
# 尾块带掩码的异步读
offs = base + tl.arange(0, BLOCK)
x = tle.load(x_ptr + offs, mask=offs < n_elements, other=0.0, is_async=True)

# 与计算重叠（依赖编译器软件流水）
for k in tl.range(0, K, BK, num_stages=2):
    a = tle.load(a_ptr + k, is_async=True)
    b = tle.load(b_ptr + k, is_async=True)
    acc = tl.dot(a, b, acc)
```

**实践建议**

- 只对“之后会被反复消费”的全局读加 `is_async=True`；纯一次性读取加了也无害但无收益。
- 边界 tile 必须显式给 `mask` / `other`，异步路径不会替你补零。
- 想看是否真的生效：dump `ttir`，PPU 查 `aiu_load`，NVIDIA 查 `ttng.async_tma_copy` / `cp.async`。

### 2.2 `tle.cumsum`

**源码签名**

```python
tle.cumsum(input, axis=0, reverse=False, dtype=None) -> (exclusive_sum, total_sum)
```

- 返回**二元组**：`exclusive_sum[i] = sum(input[:i])`（`reverse=True` 时为反向前缀），`total_sum = sum(input)`。
- **标量（非 block）输入**不会报错，而是返回 `(zero_scalar, input)` —— 前缀和恒为 0、总和就是输入本身。
- dtype 提升规则（源码 `_tle_pick_sum_dtype`）：
  - 显式给 `dtype` → 直接用它；
  - 有符号整数且位宽 < 32 → `int32`；
  - 无符号整数且位宽 < 32 → `uint32`；
  - 位宽 = 32 或浮点 → 不提升（`None`，保持原 dtype）；
  - `bfloat16` 在任何情况下都先经 `_promote_bfloat16_to_float32` 升到 `float32`。
- 常见误用：把返回值当单个 tensor 使用，会得到 tuple 而非张量。

```python
x = tl.load(x_ptr + offs)
excl, total = tle.cumsum(x)
np2 = tl.where(x != 0, 1, 0)
idx = tle.cumsum(np2)[0]          # 取 exclusive 前缀，做流式压缩/去重定位
```

后端兑现：PPU 由 `ExclusiveCumsumOp` 转换 + Allocation 处理器接入（有单测）；NVIDIA/MThreads/AMD/Sunrise 均声明支持。

### 2.3 `tle.extract_tile` / `tle.insert_tile`

**源码签名（以此为准）**

```python
tle.extract_tile(x, index, tile_shape) -> tl.tensor
tle.insert_tile(x, tile, index)        -> tl.tensor
```

| 项 | 说明 |
|:--|:--|
| `x` | 源张量（`tl.tensor`），形状必须是编译期常量 |
| `tile_shape` | 子 tile 各维大小，必须是编译期正整数；`insert_tile` 从 `tile` 自身形状推断，无此参数 |
| `index` | 见下表四种形态 |
| 约束 | `tile_shape` 的**秩必须等于源张量秩**；每维为正；`src_dim % tile_dim == 0`；索引秩也要等于源秩（标量索引除外）；静态索引需在 `[0, src_dim/tile_dim - 1]` 内。`insert_tile` 还要求源与 tile 秩相同、dtype 相同 |

**`index` 的四种形态**

| 形态 | 示例 | 路径 |
|:--|:--|:--|
| 多维静态 | `[1, 0]`（`int`/`constexpr`） | 编译期线性化，按对齐情况选 register shuffle 或 SMEM 中继 |
| 标量静态 | `2` | 视为已线性化的 tile 编号 |
| 标量动态 | `tl.tensor`（i32/i64 标量） | 运行期线性化，**总是走 SMEM 中继** |
| 多维动态 | `[i, j]`（可混 `int` 与 tensor） | 前端运行期线性化后按标量动态路径处理 |

**用法**

```python
# 静态切片：registers / smem 原地处理
sub = tle.extract_tile(x, index=[1, 0], tile_shape=[2, 2])   # x[2:4, 0:2]
sub = tl.maximum(sub, 0.0)
x = tle.insert_tile(x, sub, index=[1, 0])

# 动态索引：按 batch / expert 选 tile
sel = tle.extract_tile(big, index=[expert_id, 0], tile_shape=[BM, BK])
```

**性能提示**

- 静态多维索引优先：可命中 register shuffle 路径，不额外占用 SMEM。
- 动态索引一定走 SMEM 中继，多一次 store+load 与 barrier，循环内高频使用要评估净收益。
- 非 2 的幂 tile 在 `alloc` 上受限（见 3.1.2），但 `extract_tile` 的 `tile_shape` 只需满足整除约束。

### 2.4 `tle.pipe` —— 显式数据流边

`tle.pipe` 把“数据在第几个 buffer stage”和“producer/consumer 何时可见”统一成一个带类型的 pipe 描述，替代手工维护 `empty/full` 两套 barrier。

**源码签名**

```python
tle.pipe(*, capacity, scope="cta", name=None, readers=None, one_shot=False, **fields) -> pipe_value
```

| 参数 | 约束 |
|:--|:--|
| `capacity` | 编译期正整数，stage 数；**每个 field 的首维必须等于 capacity** |
| `scope` | 当前 MVP 只支持 `"cta"` |
| `name` | 可选，用于 IR 与诊断。必须是 `str`，且**必须满足 `str.isidentifier()`、不能以 `_` 开头、不能是 `fields`/`readers` 保留字**（`_validate_public_name`）；违反时报 `tle.pipe name must be a Python identifier` 等 |
| `readers` | 省略为默认 SPSC 单 reader；`("qk", "value")` 形式声明 SPMC 具名 reader。必须是**非空** tuple/list，逐个满足与 `name` 相同的标识符规则，且**不得重复**；不满足时报 `tle.pipe readers must be unique, got duplicate ...` |
| `one_shot` | 单一 ready/full 边，用于一次性广播；**`one_shot=True` 不支持 `close`** |
| `**fields` | 一个或多个 payload buffer，必须是 `tle.gpu.alloc(..., scope=tle.gpu.smem)` 返回的 `buffered_tensor`，**秩 ≥ 2** |

**端点 API**

```python
pipe.writer()                      -> pipe_writer     # 不接受 reader_name / field_names
pipe.reader(name=None, fields=None) -> pipe_reader
writer.acquire(iter)               -> pipe_slot       # slot.<field> 取字段，返回该 stage 的槽视图
writer.commit(iter)                -> None
writer.close(iter)                 -> None            # one_shot=True 时调用会报错
reader.wait(iter)                  -> pipe_wait_result  # {slot, is_closed}
reader.release(iter)               -> None
```

- `pipe.writer()` 若传了 `reader_name` 或 `field_names` 会报 `tle.pipe.writer does not accept reader endpoint metadata`。
- `reader.wait(iter)` 的返回对象带两个字段：`slot`（`pipe_slot`）与 `is_closed`（`tl.int1` 张量）。只看 `slot` 的写法不需要关心后者。

**语义要点**

- `iter` 映射为 `stage = iter % capacity`，另有一位 phase 区分环形复用的轮次。
- 正常读取用 `wait(iter).slot`；需要感知生产者提前关闭时读 `wait(iter).is_closed`。
- `reader(..., fields=("kv",))` 只订阅部分字段，减少不必要的依赖。
- CTA 作用域 SMEM pipe 在 NVIDIA 上 lower 为 NVWS token/mbarrier；MThreads 有自己的 `add_tle_lower_pipe`。

**示例 1：SPSC 双缓冲 load/compute**

```python
smem = tle.gpu.alloc([2, BLOCK], dtype=tl.float32, scope=tle.gpu.smem)
pipe = tle.pipe(capacity=2, scope="cta", name="tile_pipe", tile=smem)
writer = pipe.writer()
reader = pipe.reader()
offs = tl.arange(0, BLOCK)

# producer
for i in tl.range(0, n_tiles):
    slot = writer.acquire(i)
    tl.store(tle.gpu.local_ptr(slot.tile), tl.load(gmem_ptr + i * BLOCK + offs))
    writer.commit(i)

# consumer
for i in tl.range(0, n_tiles):
    ready = reader.wait(i)
    tile = tl.load(tle.gpu.local_ptr(ready.slot.tile))
    acc += tile
    reader.release(i)
```

**示例 2：SPMC + 部分字段订阅**

```python
kv   = tle.gpu.alloc([CAP, BK, D], dtype=tl.float16, scope=tle.gpu.smem)
meta = tle.gpu.alloc([CAP, BK],    dtype=tl.int32,   scope=tle.gpu.smem,
                     nv_mma_shared_layout=False)
pipe = tle.pipe(capacity=CAP, scope="cta", name="kv_pipe",
                readers=("qk", "value"), kv=kv, meta=meta)

writer      = pipe.writer()
qk_reader   = pipe.reader("qk")
value_reader = pipe.reader("value", fields=("kv",))   # 只订阅 kv

slot = writer.acquire(k)
tl.store(tle.gpu.local_ptr(slot.kv), kv_tile)
tl.store(tle.gpu.local_ptr(slot.meta), valid_mask.to(tl.int32))
writer.commit(k)

qk = tl.dot(q, tl.trans(tl.load(tle.gpu.local_ptr(qk_reader.wait(k).slot.kv))))
qk_reader.release(k)
```

**示例 3：混合传输 pipe（编译器推断每条字段的搬运方式）**

```python
q     = tle.gpu.alloc([CAP, BM, BK], dtype=tl.float16, scope=tle.gpu.smem)
scale = tle.gpu.alloc([CAP, BM],     dtype=tl.float32, scope=tle.gpu.smem,
                      nv_mma_shared_layout=False)

pipe = tle.pipe(capacity=CAP, scope="cta", name="mixed_inputs",
                readers=("mma", "epilogue"), q=q, scale=scale)
writer = pipe.writer()
mma_reader      = pipe.reader("mma", fields=("q",))
epilogue_reader = pipe.reader("epilogue", fields=("q", "scale"))

slot = writer.acquire(k)
tle.gpu.copy(q_desc, slot.q, [BM, BK], [q_block, k_block])          # 判定为 TMA 字段
tl.store(tle.gpu.local_ptr(slot.scale, (tl.arange(0, BM),)), sv)     # 判定为 local-store 字段
writer.commit(k)
```

**混合 pipe 使用规则**

1. 按**逻辑生命周期与 reader 协议**切分 pipe，不要按传输方式切分。
2. 每个 payload 分量单独一个 SMEM 字段，首维统一为 `capacity`。
3. 所有字段必须在同一个 `acquire(iter)` 与 `commit(iter)` 之间写完。
4. 传输方式由编译器从 producer 侧 IR 推断，**不要写进 pipe 名/字段名/自定义属性**。
5. 字段来源必须可证明：不透明的 SMEM 指针外泄、未追踪的 shared store、无法证明不重叠的写，会被**拒绝**而不是静默退化。

### 2.5 `tle.range`（`tl.range` + `reorder`）

```python
tle.range(arg1, arg2=None, step=None, num_stages=None, loop_unroll_factor=None,
          disallow_acc_multi_buffer=False, flatten=False, warp_specialize=False,
          disable_licm=False, reorder=False)
```

- 行为与 `tl.range` 完全一致，仅多一个 **`reorder`** hint。
- `reorder=True` 且设置了 `loop_unroll_factor` 时，提示 FlagTree 的 `reorder-loop-loads` 优化把展开后循环体里的 load 聚集到前面，改善访存延迟隐藏。
- 需要 FlagTree 以 `__FLAGTREE_REORDER_LOOP_LOADS__` 编译；运行时 `FLAGTREE_REORDER_LOOP_LOADS=0`（`knobs.language.reorder_loop_loads`）可关闭该 hint。
- 自动软件流水仍用 `num_stages`：`for k in tle.range(0, K, BK, num_stages=3)`。

### 2.6 分布式原语

#### 2.6.1 `tle.device_mesh`

```python
tle.device_mesh(topology: Mapping | MeshConfig | None = None, *,
                _shape=None, _dim_names=None, _physical_ids=None,
                _launch_shape=None, _launch_dim_names=None) -> device_mesh
```

- `topology` 的 key 是层级名，value 是 `int`（1D）或 `[(子名, 尺寸), ...]`（多维）。
- 也接受 `MeshConfig`（内部 `to_dict()` 后再解析）。
- **不能为空**（报 `topology cannot be empty`）；所有维度名必须**全局唯一**（报 `dimension names must be unique`）。
- `_shape` / `_dim_names` / `_physical_ids` / `_launch_shape` / `_launch_dim_names` 是**内部重建用**的关键字参数，`topology=None` 时才走该路径，且前三者必填；用户代码不需要碰。

```python
topology = {
    "node": [("node_x", 2), ("node_y", 2)],   # 跨节点 2x2 = 4
    "device": 4,                              # 节点内 4 卡
    "block_cluster": [("cluster_x", 2), ("cluster_y", 2)],
    "block": 4,
}
mesh = tle.device_mesh(topology=topology)
# mesh.shape -> (2, 2, 4, 2, 2, 4)，size = 256
```

常用成员：`shape`、`ndim`、`dim_names`、`physical_ids`、`size`、`launch_shape`、`launch_dim_names`、`flatten()`、`reshape(*shape)`、`mesh[...]`（切片/整数索引取子 mesh）。

> `device_mesh` 是**不可变**拓扑描述符，`__deepcopy__` 直接返回自身（`return self`），以便 JIT 全局变更检查稳定。不要试图在 kernel 内修改它的字段。

#### 2.6.2 `tle.sharding` 与分片记号

```python
tle.sharding(mesh, split=None, partial=None) -> ShardingSpec
tle.make_sharded_tensor(handle, sharding, shape=None) -> ShardedTensor
```

- `split`：每个张量维上有序的 mesh 轴组；`split` 轴必须唯一、不得与 `partial` 重叠。
- `partial`：该张量是部分和，需要在对应轴上 reduce 后才能当完整值使用。
- 未出现在 `split`/`partial` 中的 mesh 轴自动视为 **broadcast**。
- 记号：`tle.S(axis)` 切分、`tle.B` 广播/复制、`tle.P(axis)` 部分和。

```python
spec = tle.sharding(mesh, split=[["cluster_x", "cluster_y"], "device"], partial=["block"])
x = tle.make_sharded_tensor(x_ptr, sharding=spec, shape=[M, K])
```

> **注意**：`tle.sharding` 目前是**纯注解**，只影响编译器检查与后续通信插入，不产生直接代码。

#### 2.6.3 `tle.shard_id` / `tle.n_pes`

```python
tle.shard_id(mesh, axis, device_dptr=None) -> tl.tensor(int32)
tle.n_pes(dev_mem_ptr) -> tl.tensor(int32)
```

- `axis` 可为名字（`"node"`/`"device"`/`"cluster_x"`/`"chiplet"`）或索引（支持负数）。
- `axis == "device"`：返回节点内 rank；缺 `device_dptr` 时显式报 `ValueError`。
- `axis == "chiplet"`：同样走节点内 rank 的取值路径（与 `device` 同一实现）。
- `axis == "node"`：需要 `device_dptr`，返回 `world_rank // local_world_size`。
- 其它轴：`mesh` 必须是 `device_mesh`，由 `program_id(0)` 与 `launch_shape` 推导坐标。

```python
mesh = tle.device_mesh({"node": 2, "device": 4})
node_rank   = tle.shard_id(mesh, "node", device_dptr=ctx.device_dptr)
device_rank = tle.shard_id(mesh, "device", device_dptr=ctx.device_dptr)
```

#### 2.6.4 `tle.remote` / `tle.distributed_barrier`

```python
tle.remote(tensor=None, shard_id=None, scope=None, space="cluster",
           dtype=None, offset=None, coopkind=None, context_id=0)
tle.distributed_barrier(mesh=None, device_dptr=None, space=None,
                        group_kind=GroupKind.BLOCK, barrier_kind=BarrierKind.SYNC,
                        order=MemoryOrder.ACQ_REL, index=0, context_id=0,
                        memory_scope=MemoryScope.SYSTEM)
```

`distributed_barrier` 的分发顺序：

1. 切片 cluster mesh → cluster 子 mesh 同步；
2. 显式 `space`（`intra`/`intra_node`/`device`、`inter`/`inter_node`/`node`、`world`）→ FlagCX 团队同步；
3. 纯 block mesh → 协作式 grid 同步；
4. 其余 → 全 cluster 同步。

枚举：`BarrierKind{ARRIVE,WAIT,SYNC}`、`MemoryOrder{RELAXED,ACQUIRE,RELEASE,ACQ_REL}`、`MemoryScope{SYSTEM,DEVICE,BLOCK,THREAD}`、`GroupKind{THREAD,WARP,BLOCK,TILE_SPAN,LANES,GRID}`。

```python
next_device = (device_rank + 1) % mesh.shape[1]
remote_x = tle.remote(x, shard_id=(node_rank, next_device), scope=mesh)
tle.distributed_barrier(mesh)
neighbor = tl.load(remote_x)
```

远端访问的三条硬约束（来自 `tle_cn.md` 的负例说明）：

- **只支持连续 1D 区间**：node 路径把 `load/store` 合并为一次 RDMA put/get，无法表达 2D tile 的离散地址；需要传二维数据时先在本地 buffer 排成连续布局再整体传输。
- **偏移不加在 `shard_id` 上**：`tle.remote` 只解析目标 peer 并返回对端窗口基址，源/目的偏移由后续 `load/store` 的 `src_offset`/`dst_offset` 表达。
- **两侧区间与 mask 必须一致**：传输长度由指针表达式的 `arange` 区间推导，两侧 range/mask 不一致会在编译期被拒绝。

#### 2.6.5 `tle.signal` / `tle.signal_wait`

```python
tle.signal(device_dptr, peer, slot_id, value=None, op="inc",
           space="intra_node", group_kind=GroupKind.BLOCK,
           context_id=0, scope=MemoryScope.SYSTEM)

tle.signal_wait(device_dptr, slot_id, wait_kind, target=None,
                group_kind=GroupKind.BLOCK, context_id=0,
                order=MemoryOrder.ACQUIRE)
```

**`tle.signal`**

- 只发送信号，**不搬运数据、不等待对端完成**。
- `op ∈ {"inc", "add"}`：`inc` 对目标槽加一（`value` 被忽略），`add` 加上 `value`。
- `peer` 是 `space` 所选 team 内的 rank；`slot_id` 选择信号槽；`context_id` 选择预分配的 FlagCX 网络上下文。
- `group_kind="block"`（默认）要求 CTA 内**收敛执行**，整组发出一次远端更新；`thread`/`warp` 组也可用。
- `space` 取值见 2.6.4 的 team 别名表。

**`tle.signal_wait`**

| 参数 | 规则 |
|:--|:--|
| `wait_kind` | `"signal"` / `"counter"` / `"shadow"`（也接受 `attr.SignalWaitKind`） |
| `target` | **`signal`/`counter` 必填，`shadow` 必须省略**（`shadow` 从 FlagCX 本地维护的 shadow buffer 读目标值） |
| `order` | `"relaxed"` 或 `"acquire"`（默认 `acquire`）；传入其它值报 `order must be 'relaxed' or 'acquire'` |
| `group_kind` | `thread`/`warp`/`block` |
| `slot_id` | 必须是 int32 标量，按 signal 槽命名空间解释；`target` 为 int64 标量 |
| `context_id` | 编译期 int，必须在 int32 范围内 |

> 源码注释明确：`target is required for wait_kind="signal" and wait_kind="counter". wait_kind="shadow" instead reads the target from FlagCX's locally maintained shadow buffer, so target must be omitted.` 这条约束由 `utils.verify_signal_wait` 强制。

#### 2.6.6 通信上下文（host 侧）

```python
from triton.experimental.tle.language import (
    init_communicator, get_mem_pool, create_dist_tensor, cleanup_communicator,
)

init_communicator()                 # 建立 torch.distributed + FlagCX comm
ctx = create_dist_tensor(buf)       # 绑定一块 buffer 为分布式张量
# kernel 内以编译期参数传入 ctx 与 mesh
cleanup_communicator()
```

- `init_communicator` 使用 `nccl` backend 初始化 `torch.distributed`，再 `flagcxCommInitRank`。
- Ascend 后端走独立分支（SHMEM/HCCL + `symm_at`/`get_rank`），由 `language/dsa/ascend/communication.py` 提供。

#### 2.6.7 当前实现状态（重要）

| API | 状态 |
|:--|:--|
| `device_mesh` / `sharding` / `make_sharded_tensor` / `shard_id` / `n_pes` | 可用 |
| `remote` / `distributed_barrier` / `signal` / `signal_wait` | 可用（需后端支持 FlagCX 或 cluster 路径） |
| **`reshard`** | `raise NotImplementedError("reshard is deferred to M4")` |
| **`distributed_dot`** | `raise NotImplementedError("distributed_dot is deferred to M5")` |

> Wiki 的 `3.2.4.5 Resharding`、`3.2.4.6 Distributed GEMM` 是设计描述，**当前源码未实现**。写代码时不要按 Wiki 直接调用，会得到 `NotImplementedError`。

---

## 3. TLE-Struct

TLE-Struct 按**硬件拓扑族**分类：`GPU`（GPGPU）与 `DSA`（领域专用架构）。同一族内 API 尽量统一，跨族不通用。

### 3.1 GPU

#### 3.1.1 类型体系

**存储作用域 `scope`**

```python
tle.gpu.smem   # scope('share_memory')：共享内存
tle.gpu.tmem   # scope('tensor_memory')：张量内存（NVIDIA TMEM）
```

**布局 `layout`**

| 类型 | 用途 |
|:--|:--|
| `tle.gpu.BlockEncoding` | 显式 CTASPerCGA / CTASplitNum / CTAOrder 的 block 编码 |
| `tle.gpu.MmaEncoding` | MMA 结果/累加器编码 |
| `tle.gpu.DotOperandEncoding` | dot 操作数寄存器编码 |
| `tle.gpu.SlicedEncoding` | 切片编码 |
| `tle.gpu.shared_layout` | 共享内存布局基类 |
| `tle.gpu.swizzled_shared_layout` | 共享内存 swizzle 布局（`make_default(rank)`） |
| `tle.gpu.nv_mma_shared_layout` | NVIDIA MMA 消费布局（`make_default(shape, elemType)`） |
| `tle.gpu.tensor_memory_layout` | TMEM 布局 |

**值类型**

| 类型 | 说明 |
|:--|:--|
| `buffered_tensor` | 显式管理的本地 buffer 句柄；`dtype`、`type.shape`、`type.storage`、`type.layout`、`type.alloc_shape` |
| `barrier` / `barrier_type` | mbarrier / named barrier 句柄；数组用 `bars[i]` 索引 |
| `pipe_value` / `pipe_writer` / `pipe_reader` / `pipe_slot` / `pipe_wait_result` | 2.4 节 pipe 端点 |
| `PENDING` / `READY` | barrier 初值 |

**`buffered_tensor` 的视图方法**

```python
buf.load(writable=True, target_shape=None) -> tl.tensor      # 整块读入
buf.store(value)                                             # 整块写回
buf.slot(stage)                                              # 环形缓冲取第 stage 级（rank>=2，仅 smem）
buf.reshape(shape)                                           # 别名视图，元素总数必须一致
buf.subslice(start, length, dim=-1)                          # 沿维连续子视图
buf.subslice((o0, o1), (s0, s1))                             # 多维形式
buf.make_permute(handle, dims)                               # 维度置换视图
```

`slot` / `subslice` 的取舍：

- `slot(stage)`：把首维当 stage 维取一级。要求 `rank >= 2`、`storage is smem`、`stage` 必须是 **int32 标量**（其它整型也会被拒）。
- `subslice`：保留原秩与布局，参数必须是编译期整数。**两种形式规则不同**：
  - **一维形式** `subslice(start, length, dim=-1)`：只需 `start >= 0`、`length > 0`、`start + length <= shape[dim]`，**无对齐/2 的幂要求**。
  - **多维形式** `subslice((o0, o1), (s0, s1))`：两个序列长度都必须等于 buffer 秩；除上述范围检查外，若 `size != extent` 还额外要求 **`size` 是 2 的幂且 `offset % size == 0`**（报 `requires an aligned power-of-two tile`）。

#### 3.1.2 `tle.gpu.alloc`

```python
tle.gpu.alloc(shape, dtype, layout=None, scope=tle.gpu.smem,
              init_value=None, alias=None, alias_offset_bytes=0,
              nv_mma_shared_layout=True, alignment_bytes=None) -> buffered_tensor
```

| 参数 | 说明 |
|:--|:--|
| `shape` | 完整逻辑形状。**2D = 一个矩阵**；**3D = 带显式 stage 维的流水缓冲** |
| `dtype` | `tl.dtype` |
| `layout` | 显式 shared 编码；`None` 时走自动逻辑存储规划（推荐） |
| `scope` | 目前实现以 `tle.gpu.smem` 为主 |
| `init_value` | 可选寄存器初值；3D 分配时可给带 stage 维的张量逐 stage 初始化（多余 stage 忽略） |
| `alias` / `alias_offset_bytes` | 复用另一块 smem 的静态字节偏移视图（**不能与 `init_value` 同用**，仅 smem） |
| `nv_mma_shared_layout` | `layout=None` 时是否选择 MMA 消费方定义的布局 |
| `alignment_bytes` | 新建 smem 的额外 2 的幂对齐；**不能用于 alias 视图** |

**形状规则**

- 最多允许**一个**非 2 的幂的矩阵维度，且必须是 16 的倍数；超出会直接报错。
- **非 2 的幂路径是 NVIDIA 专有的**，触发时会强制以下全部条件，任一不满足即报错（`gpu/core.py` 约 470–495 行）：
  - `FLAGTREE_BACKEND` 必须为空（即 NVIDIA）；
  - 构建需带 `__TLE_TILED_SMEM__`（`mark_logical_alloc_candidate` 存在）；
  - `storage` 必须是 `smem`；
  - **不能与 `alias` 同用**；
  - payload 必须是 **rank-2**；
  - `layout` 必须是 `None`；
  - `nv_mma_shared_layout` 必须为 `True`；
  - dtype 必须是 Hopper WGMMA 兼容类型：`float16`/`bfloat16`/`float32`(TF32)/`float8e4nv`/`float8e5`/`int8`。
- 显式 `layout` 且尾部维度全为 2 的幂时，按物理 arena 处理，不受上述限制。
- 3D 的首维是 stage 数，**永不 padding**（非 2 的幂时只 padding 该矩阵维，见 `storage_shape[logical_non_power_axis] = 1 << (extent-1).bit_length()`）。
- 显式 `nv_mma_shared_layout` 对象在部分后端（如 Iluvatar）会被拒绝，只能给 `True`/`False`。

> **实务影响**：天数、平头哥、摩尔线程等非 NVIDIA 后端上，只要 tile 维度不是 2 的幂就直接报错，没有"逻辑域规划"兜底。写跨后端 kernel 时不要依赖非 2 的幂形状。

**后端差异**

| 后端 | `nv_mma_shared_layout=True` 的实际含义 |
|:--|:--|
| NVIDIA | 选 `#ttg.nvmma_shared` 编码 |
| MThreads | 不产生 NVIDIA 编码，由 SQMMA lowering 在 lowering 期物化成 MUSA 布局 |
| Iluvatar | 若 tile 是 TCU 可用的 2D 操作数（dtype ∈ int8/fp16/bf16/fp32 且连续维字节数 ≥ 64），生成 `#ttg.swizzled_shared` + `useTcu=true`、`vec=512/bitwidth`；否则用通用 swizzled |
| PPU | 走通用路径；社区示例普遍显式写 `nv_mma_shared_layout=False` |

**CommonIR 限制**：开启 `triton._common_ir.ENABLED` 时，GPU 侧**仅支持 smem**，且 `alloc(alias=...)` 直接报错。

#### 3.1.3 `tle.gpu.local_ptr`

```python
tle.gpu.local_ptr(buffer, indices=None, shape=None) -> tl.tensor | tl.ptr
```

| 项 | 规则 |
|:--|:--|
| `buffer` | `tle.gpu.alloc` 返回的 `buffered_tensor`（也接受 `tle.remote(...)` 的远端 buffer） |
| `indices` | `tuple`/`list`/`tl.tuple`，长度必须等于 `rank(buffer)`；元素必须是**整数**索引；要么全标量、要么全张量，**不能混用**；张量索引之间形状必须完全一致 |
| `indices=None` | 生成覆盖整个 buffer 形状的全视图；rank>0 返回指针张量，rank=0 返回标量指针 |
| `shape` | 可选，把索引广播到指定结果形状（不改变索引取值） |
| 返回 | 共享内存地址空间（LLVM `addrspace=3`）的指针；按行主序线性化 |

**支持的下游操作**：`tl.load`、`tl.store`、`tl.atomic_{add,and,cas,max,min,or,xchg,xor}`。

```python
# 1D 切片
smem = tle.gpu.alloc([BLOCK], dtype=tl.float32, scope=tle.gpu.smem)
idx = offset + tl.arange(0, SLICE)
vals = tl.load(tle.gpu.local_ptr(smem, (idx,)))

# 2D 分块
smem_a = tle.gpu.alloc([BM, BK], dtype=tl.float16, scope=tle.gpu.smem)
rows = tl.broadcast_to(tl.arange(0, BM)[:, None], (BM, KW))
cols = tl.broadcast_to(tl.arange(0, KW)[None, :] + k_start, (BM, KW))
a_slice = tl.load(tle.gpu.local_ptr(smem_a, (rows, cols)))

# 任意 gather 视图：每行取不同列
rows = tl.broadcast_to(tl.arange(0, H)[:, None], (H, SLICE))
cols = tl.broadcast_to(1 + tl.arange(0, SLICE)[None, :], (H, SLICE))
out  = tl.load(tle.gpu.local_ptr(smem, (rows, cols)))

# 同一指针上 store / atomic / load
ptr = tle.gpu.local_ptr(smem_i32, (tl.arange(0, BLOCK),))
tl.store(ptr, tl.zeros([BLOCK], dtype=tl.int32))
tl.atomic_add(ptr, 1)
vals = tl.load(ptr)
```

**同步**：`TleInsertLocalPointerBarriers` 会自动为 local-pointer 的 load-after-store 危险插入 barrier；只有在 pass 覆盖范围外的自定义同步模式才需要手写 barrier。

**远端变体**：`tle.gpu.local_ptr(remote_buffer, indices)` 语义相同，但解引用路由到 `shard_id` 指定的远端 shard，跨 shard 顺序需配 `tle.distributed_barrier`。

#### 3.1.4 `tle.gpu.copy`

```python
tle.gpu.copy(src, dst, shape, offsets=None, barrier=None, mask=None,
             cache_policy=None, other=None, is_async=False, eviction_policy="")
```

**方向由操作数类型推断**

| src | dst | 方向 |
|:--|:--|:--|
| `tl.tensor` / `tl.tensor_descriptor` | `buffered_tensor` | GM → LOCAL |
| `buffered_tensor` | `tl.tensor` / `tl.tensor_descriptor` | LOCAL → GM |

**参数**

| 参数 | 说明 |
|:--|:--|
| `shape` | 本次搬运的 tile 形状 |
| `offsets` | **descriptor 路径必填**，给出描述符坐标 |
| `barrier` | 可选 TMA 完成 barrier；**仅** NVIDIA GM→SMEM TMA 且 src 是 descriptor 时有效；必须来自 `alloc_barrier(s)(expect_bytes=...)` |
| `mask` | 指针张量路径逐元素掩码。GM→LOCAL 时被掩码位置写 `other`（默认 0）；LOCAL→GM 时不写。**TMA descriptor 路径不接受 mask** |
| `other` | 被掩码位置的填充值（`-inf`/`nan`/整数均可），**必须与 `mask` 同用**，仅 GM→LOCAL |
| `cache_policy` | **仅 MThreads 且仅 descriptor→shared 方向**：`copy` 主体先查 `cache_policy is not None and (is_normcopy or direction != GM_TO_LOCAL)` → 报 `copy cache_policy is only supported for descriptor-to-shared copies`；其它后端一进来就报 `copy cache_policy is not supported by this backend` |
| `is_async` | 仅支持“全局指针张量 → smem `buffered_tensor`”；发送后**不会自动 commit/wait**，需配 `async_commit_group()` / `async_wait_group()`。**后端限制：NVIDIA / Iluvatar 可用，MThreads 显式拒绝**（`copy(is_async=True) is currently supported only by the NVIDIA and Iluvatar TLE backends`）。同时不接受 `barrier`、`offsets`；要求 `dst.type.storage is smem`、src 指针位于 global 地址空间 1、元素类型一致、`shape` 全为正值静态整数且与两侧完全匹配 |
| `eviction_policy` | L2 驱逐策略：`""` / `"evict_first"` / `"evict_last"` |

```python
# 普通指针搬运
tle.gpu.copy(a_ptrs + ystride_a * yoffs[None, :], a_smem, [XBLOCK, YBLOCK])

# TMA + 显式完成 barrier
a_bar = tle.gpu.alloc_barrier(expect_bytes=BLOCK_M * BLOCK_K * 2)
tle.gpu.copy(a_desc, a_smem, [BLOCK_M, BLOCK_K], [pid_m * BLOCK_M, pid_k * BLOCK_K], barrier=a_bar)
tle.gpu.barrier_wait(a_bar, phaseIdx=0)

# 短行确定性 -inf 尾巴
tle.gpu.copy(x_ptr + off, buf.slot(stage), [N], mask=off < row_len, other=float("-inf"))

# 异步组
tle.gpu.copy(gmem_ptrs, smem, [BLOCK])
tle.gpu.async_commit_group()
tle.gpu.async_wait_group(0)
```

`copy(..., barrier=...)` **不支持**：普通 copy、SMEM→GM 的 TMA store、named barrier、无 `expect_bytes` 的 barrier。

#### 3.1.5 `tle.gpu.async_commit_group` / `async_wait_group`

```python
tle.gpu.async_commit_group()                    # 提交上一次 commit 之后发出的所有异步 copy
tle.gpu.async_wait_group(max_pending=0)         # 等待至多 max_pending 组未完成；取值 [0, 7]
```

#### 3.1.6 Barrier API

```python
tle.gpu.alloc_barriers(num_barriers, arrive_count=1, init=tle.gpu.PENDING, expect_bytes=None) -> barrier
tle.gpu.alloc_barrier(arrive_count=1, init=tle.gpu.PENDING, expect_bytes=None) -> barrier
tle.gpu.barrier_wait(barr, phaseIdx=None)
tle.gpu.barrier_arrive(barr, arrive_count=1, phaseIdx=None)
```

| 项 | 说明 |
|:--|:--|
| `num_barriers` | 正整数，槽位数；数组用 `bars[i]` |
| `arrive_count` | mbarrier 为逻辑到达数；named barrier 为参与线程数（通常 `num_warps * 32`） |
| `init` | `PENDING`（初始不可通过）/ `READY`（初始可通过，**仅 mbarrier 路径**） |
| `expect_bytes` | TMA GM→SMEM 完成字节数，正整数；要求 mbarrier 路径 |

**路径选择（关键）**

- 传 `phaseIdx` → **mbarrier** 路径。
- 省略 `phaseIdx` → **named barrier** 路径。
- 同一槽位不能混用两种路径（`_record_barrier_backend` 会记录）。
- `expect_bytes` 与 `init=READY` 强制 mbarrier。
- named barrier 要求**静态槽位索引**（如 `sync[0]`），动态索引被拒绝。
- **named 路径下 `barrier_arrive` 的 `arrive_count` 必须为 1**，否则报 `named barrier backend requires barrier_arrive arrive_count = 1`。
- 环形复用：把“该槽第几次被用到”的 `use_id` 当 `phaseIdx` 传，lowering 内部映射到硬件 phase。

`alloc_barriers` 内部的选路判定是 `can_use_named_backend = (expect_bytes is None and init == PENDING)`——只有这个组合才会预留 named barrier id（`_reserve_named_barrier_ids`），其余一律按 mbarrier 类型身份处理。

```python
bars = tle.gpu.alloc_barriers(num_barriers=NUM_SLOTS, arrive_count=1, init=tle.gpu.READY)
for k in range(NUM_ITERS):
    slot, use_id = k % NUM_SLOTS, k // NUM_SLOTS
    tle.gpu.barrier_wait(bars[slot], phaseIdx=use_id)
    ...
    tle.gpu.barrier_arrive(bars[slot], phaseIdx=use_id)
```

**named barrier 示例（省略 `phaseIdx`）**

```python
@triton.jit
def consumer(sync, CID: tl.constexpr):
    if CID == 0:
        tle.gpu.barrier_arrive(sync[0])      # arrive_count 必须是 1
    else:
        tle.gpu.barrier_wait(sync[0])        # init 必须是 PENDING

sync = tle.gpu.alloc_barriers(num_barriers=1, arrive_count=256)
```

> 注意 named 路径下 `arrive_count` 的语义分裂：**分配时**给的是参与线程总数（如 `256`），**调用 `barrier_arrive` 时**必须传 `1`。这两处不要混。

```python
bars = tle.gpu.alloc_barriers(num_barriers=NUM_SLOTS, arrive_count=1, init=tle.gpu.READY)
for k in range(NUM_ITERS):
    slot, use_id = k % NUM_SLOTS, k // NUM_SLOTS
    tle.gpu.barrier_wait(bars[slot], phaseIdx=use_id)
    ...
    tle.gpu.barrier_arrive(bars[slot], phaseIdx=use_id)
```

#### 3.1.7 `tle.gpu.warp_specialize`

```python
tle.gpu.warp_specialize(functions_and_args, worker_num_warps, worker_num_regs)
```

| 项 | 规则 |
|:--|:--|
| `functions_and_args` | `[(fn0, args0), (fn1, args1), ...]`；**第 0 项进默认 partition**，其余进 worker。至少要有 1 项（默认 partition） |
| `worker_num_warps` | 长度必须等于 `len(functions_and_args) - 1`，否则报 `got N worker functions but M warp counts` |
| `worker_num_regs` | 同上 |
| `args` | 必须是 tuple；纯 `int/float/bool/tl.dtype` 按 constexpr 传参 |
| 返回值 | 只来自默认 partition；无返回值时返回 `None`，单值返回值解包，多值返回 tuple；worker 只能产生副作用 |
| 同步 | **本身不提供可见性顺序**，必须配 `tle.pipe` 的 commit/wait/release、barrier 或其它同步原语 |

> 该函数**没有** `@tl.builtin` 装饰器（对比 `gpu.copy`/`local_ptr` 等都有），它依赖内部传入的 `_generator`；缺失时直接报 `warp_specialize requires a Triton code generator`。因此只能在 `@triton.jit` kernel 内调用，不能当普通 Python 函数调。

```python
@triton.jit
def producer(writer, x_ptr, n_tiles: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    for i in tl.range(0, n_tiles):
        slot = writer.acquire(i)
        tl.store(tle.gpu.local_ptr(slot.tile), tl.load(x_ptr + i * BLOCK + offs))
        writer.commit(i)

@triton.jit
def consumer(reader, out_ptr, n_tiles: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in tl.range(0, n_tiles):
        acc += tl.load(tle.gpu.local_ptr(reader.wait(i).slot.tile))
        reader.release(i)
    tl.store(out_ptr + offs, acc)

@triton.jit
def kernel(x_ptr, out_ptr, n_tiles: tl.constexpr, BLOCK: tl.constexpr):
    smem = tle.gpu.alloc([2, BLOCK], dtype=tl.float32, scope=tle.gpu.smem)
    pipe = tle.pipe(capacity=2, scope="cta", name="x_pipe", tile=smem)
    tle.gpu.warp_specialize(
        [(producer, (pipe.writer(), x_ptr, n_tiles, BLOCK)),
         (consumer, (pipe.reader(), out_ptr, n_tiles, BLOCK))],
        [4],      # worker warps
        [168],    # worker registers
    )
```

**使用要点**

- worker 里用 barrier 时，**在 `warp_specialize` 之前分配**并通过 `functions_and_args` 传入。
- 多个 worker 复用同一个 JIT 函数时，用 constexpr 参数（如 `CID`）区分行为。
- 捕获的 worker 参数在 IR 中会去重，多个 worker 可共享同一 pipe 端点或 buffer 句柄。
- 若 `alloc_barrier(s)` / `wgmma_wait` 标记存在于 `warp_specialize` 之前，前端会把 partition 函数**内联**而非保留 `tt.call`，以便 barrier 与 WGMMA 依赖对 lowering 可见（WGMMA user-promise 模式）。
- NVIDIA 还支持受限的**多 producer**：每个 producer 只能用 TMA copy，字段集合互不重叠且并集覆盖完整 stage，各自 `acquire/commit` 同一 stage 序列；overlapping 或无法证明不重叠的别名会被拒绝。

#### 3.1.8 `tle.gpu.wgmma` / `wgmma_wait`

```python
tle.gpu.wgmma(a, b, acc=None, input_precision=None, max_num_imprecise_acc=None,
              out_dtype=tl.float32, trans_a=False, trans_b=False) -> tl.tensor
tle.gpu.wgmma_wait(pendings, acc=None) -> tl.tensor
```

**约束（NVIDIA 语义，逐一对应 `gpu/core.py` 校验）**

| 项 | 要求 | 源码位置 |
|:--|:--|:--|
| `a` | SMEM `buffered_tensor` 或寄存器 `tl.tensor` | `_WGMMA_ALLOWED_OPERAND_TYPE_PAIRS` |
| `b` | **只能是** SMEM `buffered_tensor` | `_require_wgmma_smem_operand` |
| SMEM 操作数 | 必须 rank-2、`storage is smem`、**layout 必须是 `nv_mma_shared_layout`** | 同上 |
| `M` | `M >= 64` 且 `M % 64 == 0` | `wgmma` 主体 |
| `N` | `N >= 8` 且 `N % 8 == 0` | 同上 |
| `K` | `k >= 256 // a.dtype.primitive_bitwidth`，且 `a.K == b.K` | 同上 |
| dtype | A/B 相同（除非**两者都是 FP8**）；非 FP8 时只能是 `int8`/`float16`/`bfloat16`/`float32`；`int` 操作数仅 `int8` | `_wgmma_ret_scalar_ty` |
| `out_dtype` | 必须是 `tl.dtype`；**`bfloat16` 直接报错**；`int` 输入强制返回 `int32` | 同上 |
| `acc` | 若传入，形状必须是 `(M, N)`、dtype 必须等于推导出的返回类型 | `wgmma` 主体 |
| `max_num_imprecise_acc` | 非负；`None` 时 FP8 取后端默认值，非 FP8 取 0 | 同上 |
| `trans_a`/`trans_b` | 编译期布尔；只支持 rank-2 操作数 | `_require_rank2_wgmma_operand` |

**关于 `K` 的重要更正**：K 下限**不是**固定 16，而是按位宽推算的 `256 / bitwidth`：

| 输入 dtype | 位宽 | 最小 K |
|:--|:--:|:--:|
| `int8` / FP8 | 8 | **32** |
| `float16` / `bfloat16` | 16 | **16** |
| `float32`（TF32 路径） | 32 | **8** |

源码注释明确说明 TF32 允许 `K=8` 的载体。按 fp16 的 16 去写 int8 kernel 会通不过校验。

**关键差异：SMEM 操作数必须用 MMA 布局**

`_require_wgmma_smem_operand` 会检查 `isinstance(value.type.layout, tle.nv_mma_shared_layout)`，因此：

```python
# ✅ 正确：默认 alloc 会走 MMA 布局
a_smem = tle.gpu.alloc([64, 16], dtype=tl.float16, scope=tle.gpu.smem)

# ❌ 错误：显式关了 MMA 布局，wgmma 会报
#    "wgmma a must use nv_mma_shared_layout; allocate it with the default tle.gpu.alloc layout"
a_smem = tle.gpu.alloc([64, 16], dtype=tl.float16, scope=tle.gpu.smem, nv_mma_shared_layout=False)
```

这一点与 `tle.gpu.copy`/`local_ptr` 的示例相反：那些路径常写 `nv_mma_shared_layout=False` 走通用 swizzled，但**同一个 buffer 若要被 `wgmma` 消费就必须保持 MMA 布局**。混用需要两块 buffer 或显式 `set_layout`。

```python
acc = tle.gpu.wgmma(a_smem, b_smem, out_dtype=tl.float32)
acc = tle.gpu.wgmma_wait(0, acc)

# 手工流水：保留 1 组在飞
acc = tle.gpu.wgmma(a0, b0, acc)
acc = tle.gpu.wgmma(a1, b1, acc)
acc = tle.gpu.wgmma_wait(1, acc)   # 仍有 1 组未完成，可插入独立工作
acc = tle.gpu.wgmma_wait(0, acc)
```

**运行时循环中的等待要求**

源码文档字符串明确：NVIDIA 要求**从运行时循环里带出累加器前必须先 wait**，未 wait 的 loop yield 会在 lowering 阶段被拒绝。MThreads 反而可以把依赖穿过嵌套运行时 `for`，且单条 SQMMA 的循环结果支持隐式完成等待。

> MThreads 上 `wgmma` 映射为 **SQMMA**，操作数、dtype、`out_dtype`、`pendings` 约束都不同（见 [6.3](#63-sqmma对于-tlegpuwgmma的约束)）。

#### 3.1.9 `tle.gpu.memory_space` / `set_layout`

```python
tle.gpu.memory_space(input, space)      # space: "shared_memory" / "tensor_memory" / "register" / 目标自定义
tle.gpu.set_layout(value, layout)       # 给 block tensor 指定编码；layout 必须有 to_ir()
```

- `memory_space` 通过给 SSA 值打 `tt.memory_space` 字符串属性实现，用于引导后续编码选择。
- `set_layout` 要求构建期具备 `ensure_ttg_layout_attrs`（`__TLE__` 显式布局支持）。使用自研 TLE 变体的后端（如 `__ILUVATAR_TLE__`、`__MCTLE__`）可能未实现该 op，会抛带解释的 `RuntimeError`。

#### 3.1.10 GPU Struct Cookbook

**A. SMEM 暂存（数据多次复用）**

```python
a_smem = tle.gpu.alloc([BM, BK], dtype=tl.float16, scope=tle.gpu.smem)
tle.gpu.copy(a_ptrs, a_smem, [BM, BK])

rows = tl.broadcast_to(tl.arange(0, BM)[:, None], (BM, BK))
cols = tl.broadcast_to(tl.arange(0, BK)[None, :], (BM, BK))
a_tile = tl.load(tle.gpu.local_ptr(a_smem, (rows, cols)))
```

**B. SMEM 原子（直方图 / 分桶 / radix 计数）**

```python
bins = 256
counts = tle.gpu.alloc([bins], dtype=tl.int32, scope=tle.gpu.smem)
idx = tl.arange(0, BLOCK) % bins
tl.atomic_add(tle.gpu.local_ptr(counts, (idx,)), 1)
```

**C. 生产者/消费者 + warp 分工（Hopper 风格）**

```python
a_empty = tle.gpu.alloc_barrier(init=tle.gpu.READY)
a_full  = tle.gpu.alloc_barrier(expect_bytes=A_TILE_BYTES)
tle.gpu.warp_specialize(
    [(producer, (a_desc, a_smem, a_empty, a_full)),
     (consumer, (a_smem, a_empty, a_full))],
    [4], [232],
)
```

**D. 显式布局（仅在需要驱动物理地址时）**

```python
layout = tle.gpu.swizzled_shared_layout.make_default(rank=2)
buf = tle.gpu.alloc([BM, BK], dtype=tl.float16, layout=layout, scope=tle.gpu.smem)
```

### 3.2 DSA

DSA 层按后端族分命名空间：通用 `tle.dsa.*`，后端专属 `tle.dsa.ascend.*` / `tle.dsa.thrive.*` / `tle.dsa.tsingmicro.*`。当前 `main` 上 `language/dsa/` 下有 `ascend`、`thrive`、`tsingmicro` 三个实现。

#### 3.2.1 内存与搬运

```python
tle.dsa.alloc(shape, dtype, mem_addr_space) -> buffered_tensor
tle.dsa.copy(src, dst, shape, inter_no_alias=False)
tle.dsa.local_ptr(buffer, indices=None) -> tl.tensor | tl.ptr
tle.dsa.to_tensor(buffer, writable=True) -> tl.tensor
tle.dsa.to_buffer(tensor, space) -> buffered_tensor
```

Ascend 暴露的地址空间：`tle.dsa.ascend.UB`、`L1`、`L0A`、`L0B`、`L0C`。

```python
a_ub = tle.dsa.alloc([BM, BK], dtype=tl.float16, mem_addr_space=tle.dsa.ascend.UB)
b_l1 = tle.dsa.alloc([BM, BK], dtype=tl.float16, mem_addr_space=tle.dsa.ascend.L1)

tle.dsa.copy(a_ptrs, a_ub, [BM, BK])      # GMEM -> local
tle.dsa.copy(a_ub, out_ptrs, [BM, BK])    # local -> GMEM
```

`local_ptr` 与 GPU 版同构（`indices` 规则一致），`tle.remote` 得到的远端 DSA buffer 也可用 `tle.dsa.local_ptr` 物化指针。

#### 3.2.2 逐元素运算（buffer 形式）

```python
tle.dsa.add(lhs, rhs, out)   # out = lhs + rhs
tle.dsa.sub(lhs, rhs, out)
tle.dsa.mul(lhs, rhs, out)
tle.dsa.div(lhs, rhs, out)
tle.dsa.max(lhs, rhs, out)
tle.dsa.min(lhs, rhs, out)
```

- 三个操作数**必须同秩同形**，本层不做隐式广播。
- 三操作数 dtype 保持一致；可原地复用输出（如 `tle.dsa.mul(tmp, b, tmp)`），但除非后端明确允许，不要输入输出互为别名。

```python
tle.dsa.copy(a_ptrs, a_ub, [BM, BK])
tle.dsa.copy(b_ptrs, b_ub, [BM, BK])
tle.dsa.copy(s_ptrs, s_ub, [BM, BK])
tle.dsa.sub(a_ub, b_ub, tmp_ub)        # tmp = a - b
tle.dsa.mul(tmp_ub, b_ub, tmp_ub)      # tmp = tmp * b
tle.dsa.div(tmp_ub, s_ub, out_ub)      # out = tmp / s
tle.dsa.copy(out_ub, out_ptrs, [BM, BK])
```

#### 3.2.3 循环与提示

```python
tle.dsa.pipeline(...)      # 流水迭代
tle.dsa.parallel(...)      # 显式并行域
with tle.dsa.hint(inter_no_alias=True):
    tle.dsa.copy(x_ptr + offs, a_ub, [tail_size], inter_no_alias=True)
```

#### 3.2.4 切片与视图

```python
sub  = tle.dsa.extract_slice(full, offsets=(0, k0), sizes=(BM, BK), strides=(1, 1))
full = tle.dsa.insert_slice(full, sub, offsets=(0, k0), sizes=(BM, BK), strides=(1, 1))
elem = tle.dsa.extract_element(sub, indice=(i, j))
view = tle.dsa.subview(...)
```

#### 3.2.5 DSA 专属扩展

| 后端 | 专属 API |
|:--|:--|
| Ascend | `dsa.tile_alloc` / `tile_copy` / `tile_subview` / `tile_to_tensor` / `tile_set_flag` / `tile_wait_flag` / `tile_pipe_barrier` / `tile_gm_offset` / `tile_cube_launch` / `tile_cube_wait` / `dsa.workspace` |
| 清微 tsingmicro | `dsa.tsingmicro.randgen` / `rand` / `randn` |
| 辉羲 thrive | `dsa.thrive.putmem` / `getmem` / `wait` / `notify` / `fence` / `sync`（die 间 RMA） |

---

## 4. TLE-Raw

### 4.1 定位

TLE-Raw 打破 DSL 抽象边界，**内联厂商原生代码**。目标指令由厂商私有流水线生成，绕过通用编译中间层，专家可完全控制指令调度、寄存器分配与低层同步。

代价：与单一后端强绑定，不可移植。

### 4.2 编译与注入流程

```
@dialect(name=...)  ──▶ RawJITFunction / MLIRJITFunction / PPUJITFunction ...
                                │
                                ▼
                     独立编译得到 LLVM 模块文本
                                │
        tle_raw.call(fn, args) ─┼─▶ tle::DSLRegionOp（create_tle_raw_region_by_llvm_func）
                                │
                                ▼
         tensor 操作数 ──LocalAlloc/LocalStore──▶ SMEM MemDesc ──▶ LLVM CallOp
                                │
                     tensor 结果 ◀──LocalLoad── SMEM
```

关键 lowering 点：

| 环节 | 实现 |
|:--|:--|
| 张量进/出 region | `ConvertArgToMemDesc.cpp`：`LocalAllocOp` + `LocalStoreOp` 入、`LocalLoadOp` 出，并在 region 前插入 block 级 `NVVM::Barrier0Op` |
| SMEM 布局 | 禁用 swizzle（`(1,1,1)`），保留 Triton 编码的 order |
| `DSLRegionOp` 消除 | `DSLRegionInline.cpp` 改写为显式 `LLVM::BrOp` |
| `ExtractOp` lowering | `ExtractAllocatedPtrOp`/`ExtractAlignedPtrOp` → SMEM 指针；`ExtractOffsetOp` → 0；`ExtractSizesOp`/`ExtractStridesOp` 由 memdesc 形状与 order 推导 |

### 4.3 Raw 层原语总览

TLE-Raw 与 Lite/Struct 有**结构性差异**：Lite/Struct 是往 Triton 语言里加操作，Raw 是**把另一种语言编译进来**。因此语言层原语极少，全部集中在两个模块：

| 原语 | 导入 | 性质 |
|:--|:--|:--|
| `dialect(name=..., **kwargs)` | `triton.experimental.tle.raw` | 装饰器：声明一个外部 kernel 对象 |
| `Input[T]` / `InOut[T]` | `triton.experimental.tle.raw` | 参数标注（编译期类型描述） |
| `call(func, args, output_indices=None, hint="")` | `triton.experimental.tle.language.raw` | 在 Triton kernel 内调用 raw kernel，返回 `tl.tensor` |
| `call_smem(func, args, output_indices=None, hint="")` | 同上 | 同 `call`，但返回值包装为 `buffered_tensor`（SMEM 语义） |
| `vprintf(*args)` | `triton.experimental.tle.raw.mlir` | **仅 MLIR 方言**：device 侧格式化打印 |
| `vassert(cond, fmt, *args)` | 同上 | **仅 MLIR 方言**：device 侧断言 |

> `raw/__init__.py` 的 `__all__` 只有 `["dialect", "Input", "InOut"]`；`language/raw/__init__.py` 只有 `["call", "call_smem"]`。`vprintf` / `vassert` 由 `raw/mlir/__init__.py` 单独导出。
>
> **不存在** `tle.raw.language()` 或 `tle.hardware_constraint()`——Wiki 的 3.4.1.1 示例用了这两个名字，但当前源码无此符号，详见第 9 章。

### 4.4 `@dialect` —— 原语①

**源码签名**

```python
def dialect(*, name: str, **kwargs):
    def decorator(fn): ...
    return decorator
```

`name` **只能按关键字传**，且决定走哪个 JIT 类；`kwargs` 原样转发给该类的构造函数，同时参与缓存键计算。

```python
from triton.experimental.tle.raw import dialect

@dialect(name="cuda", file=Path(__file__).parent / "vec_add.cu")
def edsl(*args, **kwargs):
    ...                       # 函数体不参与编译，仅作为占位
```

**可用的 `name` 值（注册表）**

`raw/runtime.py` 的 `registry` 是**惰性 + 条件**构建的：`import` 失败即不注册。

| name | 实现类 | 可用条件 | region dialect |
|:--|:--|:--|:--|
| `"mlir"` | `MLIRJITFunction` | 装好 MLIR Python 绑定（见 4.12） | `mlir` |
| `"cuda"` | `CUDAJITFunction` | NVIDIA 构建 | `cuda` |
| `"ppu"` | `PPUJITFunction` | PPU 构建 | `ppu` |
| `"corex"` | `CorexJITFunction` | Iluvatar 构建 | `corex` |
| `"hcu"` / `"hip"` | `HCUJITFunction` | HCU 构建 | `hcu` |
| `"tops"` / `"tops_mlir"` | `TOPSJITFunction` / `TOPSMLIRJITFunction` | 燧原构建 | `tops` |

**别名重定向**（容易踩坑）

| 后端 | 别名 | 行为 |
|:--|:--|:--|
| ppu | `"cuda"` → `PPUJITFunction` | 复用 CUDA 方言源码，走 PPU 工具链 |
| iluvatar | `"cuda"` → `CorexCudaAliasJITFunction` | 由 corex clang 编译，**首次使用告警一次**；region dialect 仍是 `corex`，不是 `cuda` |
| hcu | `"cuda"` / `"hip"` → `HCUJITFunction` | `.cu` 由 DTK Clang 以 HIP device 模式编译 |

> 别名只是为了**源码复用**，不代表二进制兼容。Iluvatar 上官方建议新代码直接写 `name="corex"`。
>
> 若 `name` 无效且不是已注册项，`dialect` 内部对 `"cuda"`/`"mlir"` 有兜底 import 尝试；其余名字会得到 `KeyError`。

**通用 kwargs**（`RawJITFunction.__init__` 统一消费）

| 参数 | 默认 | 说明 |
|:--|:--|:--|
| `extern_func_name` | `""` | 外部源文件里的 **device 函数符号名**。CUDA/Corex/PPU 的 deferred 路径**强制要求**；HCU 可由源码自动推断 |
| `deferred` | `False` | 延迟编译：先登记源码，真正需要时再编（见 4.8）。TOPS 例外，默认 `True` |
| `library` | `""` | 引入设备位码库。`"nvshmem"`（CUDA）或 `"efshmem"`/`"nvshmem"`（TOPS） |

**各后端专属 kwargs**

| 后端 | 额外参数 | 说明 |
|:--|:--|:--|
| `mlir` | `pipeline: list[str]`、`context: ir.Context` | 覆盖 lowering pass 列表；默认见 4.11 |
| `cuda` | `file: Path`（**必填**）、`opt_level` | `opt_level` 默认 `-O3` |
| `ppu` | `file: Path`（**必填**） | 优化级别在 `make_llvm` 里**硬编码 `-O2`**，不接受 `opt_level` |
| `corex` | `file: Path`（**必填**） | **`library` 非空直接 `RuntimeError`**：Iluvatar 支�� plain corex device source |
| `hcu` / `hip` | `file: Path`（**必填**）、`opt_level` | 默认 `-O3` |
| `tops` / `tops_mlir` | `file: Optional[Path]`、`arch`、`extra_flags` | `arch` 缺省从环境推断（如 `gcu400`）；`deferred` 默认 **True** |

> **`file` 传参必须用关键字**：`dialect` 内部是 `registry[name](fn, **kwargs)`，位置参数只会落到 `fn`。

### 4.5 `Input` / `InOut` —— 原语②

```python
from triton.experimental.tle.raw import Input, InOut

@dialect(name="mlir")
def kernel(out: InOut["!llvm.ptr<1>"], x: Input["!llvm.ptr<1>"],
           n: Input["i32"]):
    ...
```

| 标注 | 语义 | 前端行为 |
|:--|:--|:--|
| `Input[T]` | 只读，函数内**不得修改** | 仅作为 operands 进入 `DSLRegionOp` |
| `InOut[T]` | 可能被修改 | **自动追加到返回值**，便于 Triton 侧做 SSA 值变化分析 |

- 方括号里的字符串是 **MLIR 类型文本**（如 `"!llvm.ptr<1>"`、`"i32"`、`"tensor<1024xf32>"`），会被 `ir.Type.parse` 解析并用于构造函数签名。
- 目前**必须标注**，未标注参数无法建立签名。
- `Input` / `InOut` 本身只是 `__class_getitem__` 返回字符串，无运行时开销。

### 4.6 `call` / `call_smem` —— 原语③④

**源码签名**

```python
tle_raw.call(func, args, output_indices=None, hint="")
tle_raw.call_smem(func, args, output_indices=None, hint="")
```

| 参数 | 说明 |
|:--|:--|
| `func` | `@dialect(...)` 装饰后的对象（带 `__triton_builtin__ = True`） |
| `args` | 操作数列表，**顺序与外部 kernel 的函数签名一致**（输出操作数通常在最前）。纯 `tl.constexpr` 会自动转成 tensor 句柄 |
| `output_indices` | 哪些 `args` 位置是输出（用于建立别名关系）。**`deferred=True` 时必填**；eager 模式缺省时由 `compute_alias_operand_indices(llvm, handles, extern_func_name)` 从 LLVM 签名推断 |
| `hint` | 字符串提示，写入 region 属性；支持传 `tl.constexpr`（内部 `_normalize_hint` 会解包并 `str()`） |
| 返回 | 无输出时返回 `None`；单输出返回标量对象；多输出返回 `tl.tuple` |

**`call` 与 `call_smem` 的唯一区别**

| | 返回值包装 |
|:--|:--|
| `call` | `tensor(handle, aliased.type)` —— 普通寄存器/全局语义 |
| `call_smem` | `buffered_tensor(handle, dtype, shape, storage, layout, semantic)` —— **SMEM 语义**，可直接接 `tle.gpu.local_ptr` / `tle.gpu.copy` |

**Ascend 特化**：`FLAGTREE_BACKEND == "ascend"` 时，`call` 被替换为接受 `ascend_<op>` 字符串的版本，前缀会被剥离后分派到已注册的 custom op；`call_smem` 在该后端 `raise NotImplementedError`。

```python
# eager：由 LLVM 签名推断输出
tle_raw.call(edsl_eager, [output_ptr, x_ptr, y_ptr, n_elements])

# deferred：必须显式给 output_indices
tle_raw.call(edsl_deferred, [output_ptr, x_ptr, y_ptr, n_elements],
             output_indices=[0])
```

### 4.7 `vprintf` / `vassert` —— 原语⑤⑥（仅 MLIR）

```python
from triton.experimental.tle.raw.mlir import vprintf, vassert

vprintf("Hello from bidx %d, tidx %d\n", bidx, tidx)
vassert(cond, "value %d out of range", value)
```

| 原语 | 签名 | 说明 |
|:--|:--|:--|
| `vprintf` | `vprintf(*args)` | 运行时按目标自动选实现：HIP 目标用 `AMDGPUPrintf`，其它用 `VPrintf`；格式串作为 `llvm.mlir_global` 常量落地，函数声明去重 |
| `vassert` | `vassert(cond, fmt, *args)` | 通过 **调用栈帧** 自动抓取 `文件名`/`函数名`/`行号`，底层声明 `__assertfail`；`cond` 为假时触发 |

- 二者都只在 **`name="mlir"`** 方言下可用，CUDA/HIP 源里请直接用原生 `printf`/`assert`。
- 参数类型（`i32`/`i64`/指针）由 codegen 侧推导，混用宽度时先显式 cast。

### 4.8 `deferred` 模式与源码存储

`deferred=True` 把编译推迟到真正需要时，配合 `source_store` 做跨进程/跨编译单元复用。

**登记流程**（`raw/source_store.py`）

```python
register_source(*, region_dialect, extern_func_name, source, hint="", extra=None) -> source_id
get_source(source_id) -> dict | None
list_pending_sources() -> dict
clear_pending_sources() -> None
```

- `source_id = sha256(f"{region_dialect}\0{extern_func_name or ''}\0{source}")`，因此**源码、方言名、符号名任一变化都会换 id**。
- 之后用 `builder.create_tle_raw_region_deferred(source_id, region_dialect, arg_dialect, handles, alias_indices, hint)` 建立 region；真正的 LLVM 生成发生在后续的 `compile_deferred_pending_source(entry, context=...)`。

**约束**

| 后端 | deferred 要求 |
|:--|:--|
| `cuda` | `extern_func_name` 必填，否则 `RuntimeError: deferred tle_raw CUDA source requires extern_func_name=` |
| `ppu` | 同上 |
| `corex` | 同上 |
| `mlir` | 缺省用函数名（`self.extern_func_name or self.fnname`），但 `call` 侧仍需传 `output_indices` |
| `tops` | 默认就是 deferred |

**选择建议**：调参迭代期用 eager（错误更直接、少一层间接）；需要把 `.cu`/`.tops` 源码与 Triton kernel 解耦、或走离线编译时用 deferred。

### 4.9 编译缓存键

`raw/cache_key.py` 定义了 `@dialect` 的缓存协议，属性名：

```python
__triton_tle_raw_source_cache_key__      # TLE_RAW_SOURCE_CACHE_KEY_ATTR
```

哈希输入 = **规范化后的 dialect kwargs** + **引用源文件内容**：

| 类别 | 键 |
|:--|:--|
| kwargs | `name`、`compiler`、`target`、`extern_func_name`、`deferred` |
| 路径 | `file`、`extern_file`（**解析为绝对路径**后参与哈希，且文件内容也进哈希） |
| 附加 | `extra_source_files`（列表中每个路径及其内容） |
| 回退 | 若没有任何源文件路径，则哈希**函数自身源码**（`inspect.getsource`） |

实践含义：

- 改了 `.cu` / `.tops` 内容会自动失效缓存，不需要清 `~/.triton/cache`。
- 内联 EDSL（不传 `file`）改函数体也会失效，因为走的是 `inspect.getsource` 回退。
- `cores`/`opt_level` 这类后端私有参数**不在** `_DIALECT_KWARG_KEYS` 里，改它们**不会**自动失效缓存——手动 `clear_pending_sources()` 或删缓存目录。

### 4.10 完整示例：CUDA 方言

**`vec_add.cu`**

```cuda
extern "C" __global__ void vec_add(const float* x, const float* y,
                                   float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] + y[i];
}
```

**Triton 侧**

```python
from pathlib import Path
import torch, triton
import triton.language as tl
from triton.experimental.tle.raw import dialect
import triton.experimental.tle.language.raw as tle_raw

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@dialect(name="cuda", file=Path(__file__).parent / "vec_add.cu")
def edsl(*args, **kwargs):
    ...


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    tle_raw.call(edsl, [output_ptr, x_ptr, y_ptr, n_elements])


def add(x, y):
    output = torch.empty_like(x)
    n = output.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, output, n, BLOCK_SIZE=1024)
    return output
```

要点：

- 函数体写 `...` 即可，`file=` 才是真正的源码来源。
- 目标架构优先取 `TLE_CUDA_ARCH` 环境变量，否则自动探测；clang 路径可用 `knobs.nvidia.tle_raw_clang` 覆盖，附加 flags 用 `knobs.nvidia.tle_raw_clang_flags`。
- `library="nvshmem"` 时会额外启用 nvshmem device bitcode，并注册 kernel 初始化钩子（`nvshmemx_cumodule_init`）。

### 4.11 完整示例：MLIR 方言

```python
from typing_extensions import Literal as L

from mlir import ir
from mlir.dialects import arith, llvm, nvvm, scf
import triton
import triton.language as tl
from triton.experimental.tle.raw import dialect, Input
import triton.experimental.tle.language.raw as tle_raw


@dialect(name="mlir")
def edsl(
    output: Input[L["!llvm.ptr<1>"]],
    x: Input[L["!llvm.ptr<1>"]],
    y: Input[L["!llvm.ptr<1>"]],
    n_elements: Input[L["i32"]],
):
    tidx  = nvvm.read_ptx_sreg_tid_x(ir.IntegerType.get_signless(32))
    bdimx = nvvm.read_ptx_sreg_ntid_x(ir.IntegerType.get_signless(32))
    gdimx = nvvm.read_ptx_sreg_nctaid_x(ir.IntegerType.get_signless(32))
    bidx  = nvvm.read_ptx_sreg_ctaid_x(ir.IntegerType.get_signless(32))
    idx   = arith.addi(arith.muli(bidx, bdimx), tidx)
    step  = arith.muli(bdimx, gdimx)
    n_elements = arith.index_cast(ir.IndexType.get(), n_elements)
    for i in scf.for_(idx, n_elements, step):
        i = arith.index_cast(ir.IntegerType.get_signless(32), i)
        ptrty, f32ty = ir.Type.parse("!llvm.ptr<1>"), ir.Type.parse("f32")
        xptr = llvm.getelementptr(ptrty, x, [i], [-2147483648], f32ty, 0)
        yptr = llvm.getelementptr(ptrty, y, [i], [-2147483648], f32ty, 0)
        out  = arith.addf(llvm.load(f32ty, xptr), llvm.load(f32ty, yptr))
        llvm.store(out, llvm.getelementptr(ptrty, output, [i], [-2147483648], f32ty, 0))
        scf.yield_([])


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    tle_raw.call(edsl, [output_ptr, x_ptr, y_ptr, n_elements])
```

**默认 lowering pipeline**（`pipeline=None` 时）

```
convert-scf-to-cf → finalize-memref-to-llvm → convert-arith-to-llvm
→ convert-cf-to-llvm → convert-func-to-llvm → convert-index-to-llvm
→ convert-nvvm-to-llvm → cse
```

- `pipeline` 可整体覆盖，顺序即执行顺序，`PassManager` 开启了 verifier。
- 生成的 MLIR 以 `f"{module}"` 转成文本，再交给 `create_tle_raw_region_by_llvm_func` 注入。
- `context` 可传入外部 `ir.Context` 以复用类型/属性池。

### 4.12 环境要求（MLIR 路径）

两条路：

**A. 源码构建 llvm-project**

```bash
git clone https://github.com/flagos-ai/llvm-project.git
cd llvm-project && git checkout triton_3.6.x

cmake -G Ninja -B build -S llvm \
  -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld" \
  -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
  -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_ENABLE_LLD=ON -DMLIR_ENABLE_BINDINGS_PYTHON=ON
cmake --build build

export PYTHONPATH=<LLVM_PROJECT_PREFIX>/build/tools/mlir/python_packages/mlir_core/mlir/:${PYTHONPATH}
```

**B. 使用 LLVM wheel（推荐）**

```bash
git clone --recursive https://github.com/flagos-ai/flagtree_mlir.git
cd llvm-wheel && python -m build -w
pip install ./dist/llvm_wheel-0.1.0-cp{}-cp{}-linux_x86_64.whl --force-reinstall
```

随后按常规方式构建 FlagTree。NVIDIA 上也可直接用 `LLVM_SYSPATH` 指向预编译 LLVM（见本工作区 H100 环境笔记）。

### 4.13 限制与注意事项

- MLIR 后端仍标注为**实验性**，轮子版本与 `triton_v3.6.x` 分支需匹配。
- Raw region 前后的 SMEM 中转**是强制的**，会引入额外 store/load 与一次 block barrier；数据量大时这笔开销必须计入 ROI。
- 张量操作数需要能放进 SMEM；超大 tile 会直接撞共享内存上限。
- 后端注册不到时不会给出好错误，先确认 `@dialect(name=...)` 的 name 在当前构建里存在。

---

## 5. 平头哥 PPU 能力矩阵

### 5.1 后端概况

| 项 | 内容 |
|:--|:--|
| 后端名 | `ppu`（`FLAGTREE_BACKEND=ppu`） |
| 芯片 | 平头哥真武 810E（T-Head PPU） |
| Triton 基线 | 3.6（`main` 分支） |
| 推荐镜像 | `harbor.baai.ac.cn/flagtree/flagtree-ppu-py312-torch2.10.0-sdk2.1.0-cu130-ubuntu24.04:202607-3.6-vllm0.24.0` |
| 源码标签 | `0.7.0+ppu3.6` |
| 工具链 | PPU SDK（`ppu-llc` 等），产物阶段 `ttir → ttgir → llir → hgbin` |
| 设备节点 | `/dev/alixpu_ppu0`、`/dev/alixpu_ppu1` … |
| README 声明 | TLE-Lite ✅ / TLE-Struct ✅ / TLE-Raw ✅ / CommonIR ❌ |

### 5.2 分层能力矩阵

**TLE-Lite**

| 原语 | 声明 | 验证 | 说明 |
|:--|:--:|:--:|:--|
| `tle.load(is_async=True)` | ✅ | ✅ | **仅 block pointer 会提升为 `aiu_load`**（AIU v1，`.2d.b8/b16/b32`）；普通指针张量、`is_async=False`、`int64` 均不提升 |
| `tle.cumsum` | ✅ | ✅ | `ExclusiveCumsumOp` 转换 + Allocation 处理器已接入，可产出 `ttir/ttgir/llir/hgbin` |
| `tle.extract_tile` | ✅ | ✅ | 静态/动态索引均验证（`test_tle_tile_ops.py`） |
| `tle.insert_tile` | ✅ | ✅ | 同上 |
| `tle.pipe` | ❌ | — | 白名单无 `pipe*`；PPU 未接入 NVWS |
| `tle.range` | ❌ | — | 白名单无 `range`；可用 `tl.range` |
| `tle.device_mesh` / `sharding` / `shard_id` | ❌ | — | 白名单无 |
| `tle.distributed_barrier` | ⚠️ | ✅（拒绝） | **白名单存在但编译期被拒**，见 5.3 |
| `tle.remote` | ❌ | ✅（拒绝） | `tle.remote_pointers` 命中拒绝列表 |
| `tle.signal` / `signal_wait` | ❌ | — | 白名单无 |

> `distributed_barrier` 的“声明 ✅ / 实际 ❌”是白名单与拒绝列表不一致的典型，**以拒绝列表为准**。

**TLE-Struct（GPU）**

| 原语 | 声明 | 验证 | 说明 |
|:--|:--:|:--:|:--|
| `gpu.alloc` | ✅ | ✅ | 示例普遍显式 `layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False` |
| `gpu.local_ptr` | ✅ | ✅ | 支持完整索引规则；`test_tle_gpu_local_ptr.py` 覆盖 |
| `gpu.copy` | ⚠️ | ✅ | **仅指针张量形式**；descriptor/TMA 形式被拒 |
| `gpu.warp_specialize` | ⚠️ | ✅（拒绝） | 白名单存在但 `ttg.warp_specialize` 在拒绝列表 |
| `gpu.buffered_tensor.slot` | ✅ | ✅ | `test_tle_gpu_slot.py`：lower 为 `ttg.memdesc_index` |
| `gpu.async_commit_group` / `async_wait_group` | — | — | 白名单未列。PPU 编译管线里有 `add_downgrade_invalid_async_copy` 与 `add_lower_async_load`，说明异步 copy 有部分接通，但这两个组操作**未在 PPU 白名单声明**；`copy(is_async=True)` 前端不会拒绝 PPU，依赖组语义时需在本机实测 |
| `gpu.alloc_barrier(s)` / `barrier_wait` / `barrier_arrive` | ❌ | — | 白名单无 |
| `gpu.wgmma` / `wgmma_wait` | ❌ | — | 白名单无（PPU 无 WGMMA 语义） |
| `gpu.memory_space` | ❌ | — | 白名单无 |
| `gpu.set_layout` | ❌ | — | 白名单无 |
| `gpu.buffered_tensor.subslice` / `reshape` | ❌ | — | 白名单无 |
| `gpu.buffered_tensor.load` / `store`（CommonIR） | ❌ | — | 白名单无 |

**TLE-Struct（DSA）**：`tle.dsa.*` 在 PPU 上**全部不可用**（PPU 走 GPU 族，不是 DSA 族）。

**TLE-Raw**

| 项 | 状态 | 说明 |
|:--|:--:|:--|
| `raw.call` / `raw.call_smem` | ✅ | 声明 + `PPUJITFunction` 已注册 |
| `@dialect(name="ppu")` | ✅ | 直接可用 |
| `@dialect(name="cuda")` | ✅ | 在 PPU 构建上**自动路由到 PPU 实现**，可复用 CUDA 方言源码 |
| `@dialect(name="mlir")` | ⚠️ | 取决于是否装了 MLIR 绑定 |

### 5.3 关键约束与拒绝列表

`third_party/ppu/backend/compiler.py`：

```python
PPU_UNSUPPORTED_TLE_OPS = {
    "ttg.warp_specialize":  "tle.warp_specialize",
    "ttg.tma_copy":         "tle.gpu.copy.tensor_descriptor",
    "tle.distributed_barrier": "tle.distributed_barrier",
    "tle.remote_pointers":  "tle.remote",
}
```

行为特征：

- 检查发生在 **frontend 产出的 TTIR** 上，一次性收集**所有**命中项后抛出：
  `NotImplementedError: unsupported TLE feature on the 'ppu' backend: <features>`
- 目的是让错误直接指向用户调用的 API，而不是 MLIR legalization 阶段的含糊失败。
- 排除规律：`tl.range(warp_specialize=True)` 只设循环属性、不产生 op，因此不会被误判；只有 TLE 内建会真的产生 `ttg.warp_specialize` / `ttg.tma_copy`。

**AIU 提升的三个必要条件**（缺一不可）

1. 指针来源是 `tl.make_block_ptr` 的 block pointer；
2. `is_async=True`；
3. 元素类型不是 int64（int8→`.b8`，int16/fp16/bf16→`.b16`，int32/fp32→`.b32`）。

smoke 层还会校验：`ttgir` 含 `versionMajor = 1`，`llir` 中所有 `ppu.cp.async.aiu` 指令的宽度一致（不允许同一 kernel 混用 `.b8/.b16/.b32`）。

### 5.4 已验证的可运行写法

**A. `alloc` + `local_ptr` 最小闭环**

```python
@triton.jit
def _vector_add_tle(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    smem = tle.gpu.alloc([BLOCK], dtype=tl.float32, scope=tle.gpu.smem)
    ptrs = tle.gpu.local_ptr(smem, (tl.arange(0, BLOCK),))
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    tl.store(ptrs, x + y, mask=mask)
    tl.store(out_ptr + offs, tl.load(ptrs, mask=mask, other=0.0), mask=mask)
```

降级断言（可作为 CI 基线）：`ttgir` 中保留 `tle.local_pointers`，`llir` 中**不得**残留任何 `tle.*` op。

**B. GEMM 骨架（SMEM 暂存 + 双向 copy）**

```python
a_smem = tle.gpu.alloc([BLOCK_M, BLOCK_N], dtype=tl.float32, layout=None,
                       scope=tle.gpu.smem, nv_mma_shared_layout=False)
b_smem = tle.gpu.alloc([BLOCK_M, BLOCK_N], dtype=tl.float32, layout=None,
                       scope=tle.gpu.smem, nv_mma_shared_layout=False)

rows = tl.broadcast_to(tl.arange(0, BLOCK_M)[:, None], (BLOCK_M, BLOCK_N))
cols = tl.broadcast_to(tl.arange(0, BLOCK_N)[None, :], (BLOCK_M, BLOCK_N))
a_ptrs_smem = tle.gpu.local_ptr(a_smem, (rows, cols))
b_ptrs_smem = tle.gpu.local_ptr(b_smem, (rows, cols))

for k_start in range(0, K, BLOCK_K):
    k_offs = k_start + tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
    b_ptrs = b_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn

    tle.gpu.copy(a_ptrs, a_smem, [BLOCK_M, BLOCK_N])
    tle.gpu.copy(b_ptrs, b_smem, [BLOCK_M, BLOCK_N])
    acc += tl.dot(tl.load(a_ptrs_smem), tl.load(b_ptrs_smem), input_precision="ieee")
```

要点：`nv_mma_shared_layout=False` 走通用 swizzled 路径，避开 Hopper 专属编码；`copy` 的 `shape` 必须与 buffer 形状严格一致。

**C. `slot` 环形缓冲**

```python
smem = tle.gpu.alloc([2, BLOCK], dtype=tl.int32, layout=None,
                     scope=tle.gpu.smem, nv_mma_shared_layout=False)
slot = smem.slot(0)                       # → ttg.memdesc_index
val  = tl.load(tle.gpu.local_ptr(slot, (tl.arange(0, BLOCK),)))
```

`ttgir` 应出现 `!ttg.memdesc<2x64xi32>`（外层）与 `!ttg.memdesc<64xi32>`（单槽）。

**D. 异步 block pointer 读**

```python
a_bp = tl.make_block_ptr(a_ptr, shape=(M, K), strides=(K, 1),
                         offsets=(pid * BLOCK_M, 0),
                         block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
a = tle.load(a_bp, is_async=True)        # → aiu_load
```

### 5.5 PPU 排错表

| 现象 | 原因 | 处置 |
|:--|:--|:--|
| `unsupported TLE feature on the 'ppu' backend: tle.warp_specialize` | 用了 `gpu.warp_specialize` | 改用 `tl.range(num_stages=...)` 自动流水 |
| `... tle.gpu.copy.tensor_descriptor` | 用了 `tl.make_tensor_descriptor` + `copy` | 改成指针张量 + `copy` |
| `... tle.distributed_barrier` / `tle.remote` | 白名单存在但被拒 | PPU 当前不做跨卡 TLE 通信，改用 FlagCX 层方案 |
| `is_async=True` 但 TTIR 无 `aiu_load` | 不是 block pointer / int64 / `is_async=False` | 改用 `tl.make_block_ptr`；int64 拆成 int32 |
| 同一 kernel 多个 AIU 指令 `.b8`/`.b32` 混用 | 不同 dtype 的异步读 | 统一成同一宽度，或拆分 kernel |
| `tle.gpu.copy` 报形状不符 | PPU 要求 `shape == buffer.shape` | 让 buffer 与 copy 形状一致 |
| `distributed_barrier` 白名单里有却报错 | 白名单 ≠ 可用 | 以 `PPU_UNSUPPORTED_TLE_OPS` 为准 |

---

## 6. 摩尔线程 MThreads 能力矩阵

### 6.1 后端概况

| 项 | 内容 |
|:--|:--|
| 后端名 | `mthreads`（`FLAGTREE_BACKEND=mthreads`） |
| 架构 | MUSA（PH1），矩阵指令为 **SQMMA**，异步搬运为 **TME** |
| Triton 基线 | 3.6（`main` 分支） |
| TLE 方言目录 | `third_party/mthreads/tle/`（`dialect/`、`frontend/`） |
| README 声明 | TLE-Lite ✅ / TLE-Struct ✅ / TLE-Raw ❌ / CommonIR ❌ |

MThreads 是当前 TLE 支持**最完整**的非 NVIDIA 后端之一：pipe 全套、warp_specialize、SQMMA、TME cache policy 都有实现与单测。

### 6.2 分层能力矩阵

**TLE-Lite**

| 原语 | 声明 | 验证 | 说明 |
|:--|:--:|:--:|:--|
| `tle.load`（含 `is_async`） | ✅ | ✅ | `add_tle_lower_async_load`；`test_tle_async_load.py` |
| `tle.cumsum` | ✅ | ✅ | `add_tle_lower_exclusive_cumsum` |
| `tle.extract_tile` | ✅ | ✅ | `test_tle_tile_ops.py` |
| `tle.insert_tile` | ✅ | ✅ | 同上 |
| `tle.pipe` 全套 | ✅ | ✅ | 见 6.5，`test_tle_pipe.py`、`test_tle_pipe_persistent.py` |
| `tle.range` | ❌ | — | 白名单无；用 `tl.range` |
| `tle.device_mesh` / `sharding` / `shard_id` / `remote` | ❌ | — | 白名单无跨卡 TLE 原语 |
| `tle.distributed_barrier` | ❌ | — | 白名单无 |
| `tle.signal` / `signal_wait` | ❌ | — | 白名单无 |

**TLE-Struct（GPU）**

| 原语 | 声明 | 验证 | 说明 |
|:--|:--:|:--:|:--|
| `gpu.alloc` | ✅ | ✅ | 支持非 2 的幂前导维（`mthreads.buffer` 专项校验） |
| `gpu.copy` | ⚠️ | ✅ | 严格校验：buffer 必须 smem、`shape` 必须等于 buffer 形状、dtype 必须一致；descriptor 路径即 **TME** |
| `gpu.local_ptr` | ✅ | ✅ | `test_tle_local_ptr.py` |
| `gpu.memory_space` | ✅ | ✅ | `test_tle_memory_space.py` |
| `gpu.set_layout` | ✅ | ✅ | 配合 `MusaWmmaEncoding` / `MusaSqmmaEncoding` |
| `gpu.warp_specialize` | ✅ | ✅ | `test_tle_warp_specialize*.py` |
| `gpu.alloc_barrier(s)` | ✅ | ✅ | `test_tle_barrier*.py` |
| `gpu.barrier_wait` / `barrier_arrive` | ✅ | ✅ | 同上 |
| `gpu.wgmma` / `wgmma_wait` | ⚠️ | ✅ | **映射为 SQMMA**，约束见 6.3 |
| `gpu.buffered_tensor.slot` | ✅ | ✅ | `test_tle_slot.py` |
| `gpu.buffered_tensor.subslice` | ✅ | ✅ | `test_tle_subslice.py` |
| `gpu.buffered_tensor.reshape` | ❌ | — | 白名单无 |
| `gpu.async_commit_group` / `async_wait_group` | ⚠️ | — | 白名单未列。**关键**：MThreads 的 `gpu.copy(is_async=True)` 会在前端直接 `ValueError`（只支持 NVIDIA/Iluvatar），因此这两个 API 在 MThreads 上**没有配套的异步 copy 路径可用**，形同死代码 |
| `gpu.buffered_tensor.load` / `store`（CommonIR） | ❌ | — | 白名单无 |

**TLE-Raw**：`raw.call` / `raw.call_smem` 均**未声明**，README 亦标 ❌。
（`third_party/mthreads/tle/` 是 TLE 方言与 lowering 实现，不等于 TLE-Raw 的 `@dialect` 内联能力。）

**MThreads 专属扩展**

| 名称 | 说明 |
|:--|:--|
| `tle.gpu.mthreads.TMECachePolicy(inner="normal", outer="normal")` | 单次 TME copy 的内/外域缓存持久化提示，取值 `none`/`once`/`normal`/`persist`；作为 `copy(cache_policy=...)` 传入；不可变值，可作 `tl.constexpr` 参数；**不改变完成与同步语义** |
| `tle.gpu.mthreads.MusaWmmaEncoding(version, warps_per_cta, instr_shape, cga_layout)` | `#ttg.musa_wmma`，WMMA 结果/累加器 |
| `tle.gpu.mthreads.MusaSqmmaEncoding(...)` | `#ttg.musa_sqmma`，SQMMA 结果/累加器 |
| `tle.gpu.mthreads.MusaDotOperandEncoding(operand_index, parent, k_width=0)` | `#ttg.dot_op`，WMMA 寄存器操作数 |

### 6.3 SQMMA（对于 `tle.gpu.wgmma`）的约束

源码校验（`gpu/mthreads/wgmma.py`）与 NVIDIA 差异很大，**照抄 NVIDIA 示例会直接报错**：

| 项 | MThreads 要求 |
|:--|:--|
| 操作数 | `a`、`b` **都必须是 SMEM `buffered_tensor`**，rank-2 |
| dtype | `a`、`b` 必须相同，且 ∈ `{float16, bfloat16, float8e4nv}`（**不支持 float32 操作数**） |
| 累加器 | 必须是 `tl.float32` |
| `out_dtype` | 必须为 `tl.float32` |
| `max_num_imprecise_acc` | 必须为 `0`（非 0 直接拒绝） |
| 结果 M/N | `M >= 16 且 M % 16 == 0`；`N >= 16 且 N % 16 == 0` |
| `wgmma_wait` | **只支持 `pendings = 0`**，非零在飞组未实现 |
| `trans_a` / `trans_b` | 通过 `create_memdesc_trans` 构造转置视图，会同步置换 `alloc_shape` 与 layout |

```python
a = tle.gpu.alloc((128, 64),  dtype=tl.float16, layout=None)
b = tle.gpu.alloc((64, 128),  dtype=tl.float16, layout=None)
acc = tle.gpu.wgmma(a, b, acc)
acc = tle.gpu.wgmma_wait(0, acc)          # 只能等 0
```

**编码约束**

- `MusaSqmmaEncoding` / `MusaWmmaEncoding` 只支持 **MUSA PH1 = `[3, 1]`**。
- `warps_per_cta` 秩为 2 或 3，元素必须为正的 2 的幂；秩为 3 时第三维必须为 `1`。
- `MusaSqmmaEncoding` 额外要求 `warps_per_cta[0] % 4 == 0`，`instr_shape M >= 16 且 % 8 == 0`，`N % 8 == 0`，`K % 8 == 0`。
- `MusaWmmaEncoding`：`M % 8 == 0`，`N % 8 == 0`，`K == 4 || K % 8 == 0`。
- `MusaDotOperandEncoding` 的 `parent` **只能是 `MusaWmmaEncoding`**（PH1 SQMMA 操作数由后端经 SMEM 暂存），`k_width` 必须为 0。

### 6.4 TME 与 `TMECachePolicy`

MThreads 的 descriptor 搬运即 TME（对标 NVIDIA TMA）：

```python
bar = tle.gpu.alloc_barrier(expect_bytes=BLOCK_M * BLOCK_K * 2)
tle.gpu.copy(a_desc, a_smem, (BLOCK_M, BLOCK_K), (0, k_off),
             barrier=bar,
             cache_policy=tle.gpu.mthreads.TMECachePolicy(inner="normal", outer="normal"))
tle.gpu.barrier_wait(bar, phaseIdx=0)
```

约束与建议：

- `barrier` 仅对 **GM→SMEM** 的 descriptor copy 有效；必须是**带索引的 barrier 槽**且 `expect_bytes > 0`。
- `cache_policy` 只在 MThreads 生效，其它后端显式传入即 `ValueError`。
- `inner="none"` 对纯流式访问看起来“省缓存”，实测常**严重拖慢 GEMM**，非默认策略必须在目标负载上实测。
- `cache_policy` 只影响 hint，不改变完成/同步时序。

### 6.5 pipe 与 warp_specialize 的版本标记

`language/gpu/mthreads/__init__.py` 暴露一组能力标记，用于在**老 wheel** 上安全降级（`getattr(libtriton.ir.builder, ..., 0)`）：

| 标记 | 覆盖能力 |
|:--|:--|
| `MTHREADS_GEMM_COMPILER_OPTIONS_VERSION`（当前 2） | GEMM 编译选项 |
| `MTHREADS_TLE_CONSUMER_SYNC_VERSION`（1） | 物理 CTA barrier 初始化、周期性多 epoch 具名 reader 会合、混合 warp 矩形 SMEM 切片操作数 |
| `MTHREADS_TLE_PIPE_SQMMA_VERSION` | pipe 与 SQMMA 联动 |
| `MTHREADS_TLE_MULTIFIELD_PIPE_VERSION` | 多字段 pipe |
| `MTHREADS_TLE_ONE_SHOT_PIPE_VERSION` | `one_shot=True` |
| `MTHREADS_TLE_SPMC_PIPE_VERSION` | SPMC 具名 reader |
| `MTHREADS_TLE_DYNAMIC_LOOPS_VERSION` | 动态循环下的 pipe |
| `MTHREADS_TLE_SPLIT_M_SQMMA_VERSION` / `SPLIT128_SQMMA_VERSION` | SQMMA M 维切分 |
| `MTHREADS_TLE_LOCAL_BARRIER_VERSION` | 局部 barrier |
| `MTHREADS_TLE_PERSISTENT_ORDERED_SQMMA_VERSION` | persistent + 有序 SQMMA |
| `MTHREADS_TLE_DYNAMIC_PARTITION_SYNC_VERSION` | 动态 partition 同步 |
| `MTHREADS_TLE_GROUPED_COMPLETION_VERSION` | 分组完成 |
| `MTHREADS_TLE_16_WARP_PERSISTENT_VERSION` | 16 warp persistent |
| `MTHREADS_TLE_FUSED_PIPE_CONSUMER_VERSION` | 融合 pipe consumer |
| `MTHREADS_TLE_DISABLE_MAX_ILP_SCHEDULER_VERSION`（1） | 关闭 max-ILP 调度器（persistent kernel 保 SQMMA 优化但避开不安全调度） |

用法：

```python
from triton.experimental.tle.language.gpu import mthreads as tle_mt

if tle_mt.MTHREADS_TLE_SPMC_PIPE_VERSION >= 1:
    pipe = tle.pipe(capacity=CAP, readers=("qk", "value"), kv=kv, meta=meta)
else:
    pipe = tle.pipe(capacity=CAP, kv=kv)     # 退回 SPSC
```

`warp_specialize` 的 MThreads 差异：

- `worker_num_warps` 必须是**正的编译期整数序列**，`worker_num_regs` 必须是编译期整数序列。
- partition 函数**强制内联**（`partition_function_caller` 返回 `generator.inline_JitFunction`），因为 MUSA TLE region 是 isolated-from-above 且用直接 SSA 捕获重映射。
- `create_warp_specialize` 保持两参调用以兼容“老 libtriton + 新 Python 源码”的组合。

### 6.6 共享内存布局约束（SQMMA root MemDesc）

来自 `third_party/mthreads/tle/README.md`：

1. 多个 SQMMA consumer 只有在推断出**完全相同**的 root `MemDescType` 时才能共享同一块分配；这包括 shape、元素类型、shared 编码、memory space、mutability 与 allocation shape。编码不同**不能**以“地址映射可能等价”为由放行。
2. 逻辑操作数方向属于**最终视图**，不对 allocation root 施加额外的行/列契约。
3. 例：FP8 的 Q `[32, 256]`、KV `[64, 256]`、P `[32, 64]` 可用同一块物理 KV 分配同时支持 `wgmma(Q, KV, trans_b=True)` 与 `wgmma(P, KV)`；**对应 BF16 配置因物理 swizzle 不兼容而被拒绝**。这不是禁止 BF16 共享：两个非转置 BF16 consumer 共享 `[64, 64]` tile 是可以的，但这不能推出 BF16 的 K-转置/V 复用成立。
4. 带 stage 的分配在 consumer 使用子切片时**仍保留完整 stage 形状**；无论该 partition 是否使用捕获，捕获的 allocation 或 stage 视图都必须在所有 `warp_specialize` partition 中与对应 block 参数**匹配**。
5. 共享物理布局**不免除同步义务**：producer 必须在两个 consumer 读取前完成写；两个 SQMMA consumer 都完成后 producer 才能覆盖该 stage。`tle.pipe` 提供该 stage 的发布与复用协议。

回归用例：`third_party/mthreads/python/test/unit/tle/test_tle_sqmma_kv_root.py`。

### 6.7 MThreads 排错表

| 现象 | 原因 | 处置 |
|:--|:--|:--|
| `mthreads TLE wgmma requires f16, bf16, or fp8e4nv` | 传了 fp32/fp64 操作数 | 操作数改 fp16/bf16/fp8 |
| `requires matching A/B dtypes` | A/B dtype 不一致 | 统一 dtype |
| `requires out_dtype=tl.float32` / `requires max_num_imprecise_acc=0` | 参数不合法 | 按 6.3 默认值传 |
| `result M/N dimension must be divisible by 16` | tile 太小或非 16 对齐 | 调整 BLOCK 尺寸 |
| `wgmma_wait currently requires pendings=0` | 用了手工在飞组 | 只等 0，或改用 pipe 表达流水 |
| `copy shape ... must match buffer shape` | copy 与 buffer 形状不一致 | 对齐两者 |
| `MUSA TLE copy only supports tle.gpu.smem buffers` | 传了 tmem/其它 scope | 用 smem |
| `copy cache_policy is only supported by the MUSA TLE backend` | 在非 MThreads 上用了 policy | 仅在 mthreads 分支传 |
| `MusaSqmmaEncoding warps_per_cta[0] must be a multiple of 4` | 编码参数非法 | 依 6.3 修正 |
| `MusaDotOperandEncoding does not support MusaSqmmaEncoding` | parent 传错 | parent 用 `MusaWmmaEncoding` |

---

## 7. 跨后端能力总表

数据来自各 `third_party/<backend>/backend/tle_supported.py` 与 README 声明表。

### 7.1 分层声明（README）

| 后端 | Triton | Lite | Struct | Raw | CommonIR |
|:--|:--:|:--:|:--:|:--:|:--:|
| nvidia | 3.6 | ✅ | ✅ | ✅ | ✅ |
| amd | 3.6 | ✅ | ✅ | | |
| enflame（燧原） | 3.6 | ✅ | ✅ | ✅ | |
| iluvatar（天数） | 3.6 | ✅ | ✅ | ✅ | |
| hcu（海光） | 3.6 | ✅ | ✅ | ✅ | |
| mthreads（摩尔线程） | 3.6 | ✅ | ✅ | | |
| thrive（达摩院） | 3.6 | ✅ | ✅ | | |
| metax（沐曦） | 3.6 | ✅ | ✅ | | |
| sunrise（曦望） | 3.6 | ✅ | ✅ | | |
| xpu（昆仑芯） | 3.6 | | ✅ | ✅ | |
| ppu（平头哥） | 3.6 | ✅ | ✅ | ✅ | |
| spacemit（进迭） | 3.6 | ✅ | | | |
| tsingmicro（清微） | 3.6 | ✅ | ✅ | | |
| ascend（昇腾） | 3.5 | | ✅ | ✅ | ✅ |
| intellifusion（云天励飞） | 3.8 | — | — | — | — |
| cambricon（寒武纪） | 3.2 | — | — | — | — |
| aipu / cpu（安谋） | 3.3 | — | — | — | — |

### 7.2 关键原语可用性（按白名单）

| 原语 | nvidia | ppu | mthreads | iluvatar | amd | hcu | metax | sunrise | tsingmicro | ascend | xpu |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| `load`（含 is_async） | ✅ | ✅ | ✅ | ✅ | | | | ✅ | | ✅ | |
| `cumsum` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | | ✅ | ✅ | ✅ | |
| `extract_tile` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | |
| `insert_tile` | ✅ | ✅ | ✅ | ✅ | ✅ | | ✅ | ✅ | ✅ | ✅ | |
| `range`（reorder） | ✅ | | | | | | | | | ✅ | |
| `pipe` 全套 | ✅ | | ✅ | ✅ | | | | | | ✅ | |
| `device_mesh` / `sharding` / `shard_id` | ✅ | | | ✅ | | | | | ✅ | ✅ | |
| `distributed_barrier` | ✅ | ⚠️* | | ✅ | | | | | ✅ | ✅ | |
| `remote` | ✅ | | | ✅ | | | | | ✅ | ✅ | |
| `signal` / `signal_wait` | ✅ | | | | | | | | | | |
| `gpu.alloc` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | | | ✅ |
| `gpu.copy` | ✅ | ✅ | ✅ | ✅ | | ✅ | ✅ | ✅ | | | ✅ |
| `gpu.local_ptr` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | | | ✅ |
| `gpu.memory_space` | ✅ | | ✅ | ✅ | | | | ✅ | | | ✅ |
| `gpu.set_layout` | ✅ | | ✅ | ✅ | | ✅ | | | | | |
| `gpu.warp_specialize` | ✅ | ⚠️* | ✅ | ✅ | | | | | | | |
| `gpu.alloc_barrier(s)` / `barrier_*` | ✅ | | ✅ | | | | | | | | |
| `gpu.wgmma` / `wgmma_wait` | ✅ | | ✅ | | | | | | | | |
| `gpu.buffered_tensor.slot` | ✅ | ✅ | ✅ | ✅ | ✅ | | | | | | |
| `gpu.buffered_tensor.subslice` | ✅ | | ✅ | | | | | | | | |
| `gpu.buffered_tensor.reshape` | ✅ | | | | | | | | | | |
| `raw.call` / `call_smem` | ✅ | ✅ | | ✅ | | ✅ | | | | ✅ | ✅ |
| `dsa.*` | | | | | | | | | ✅ | ✅ | ⚠️ |

\* `⚠️*`：出现在 PPU 白名单，但命中 `PPU_UNSUPPORTED_TLE_OPS`，实际编译期报错。

> 表中空白表示**未在白名单声明**。未声明不等于绝对不可用（某些后端把 TLE 能力编译进 `libtriton` 而不走白名单），但**不可依赖**——需要该能力时必须在本机验证。

---

## 8. 排错与调试清单

### 8.1 快速定位顺序

1. **缩到最小可复现 shape**，保留出错的单一原语。
2. 对照 Torch 参考确认是**数值问题**还是**编译问题**。
3. `compiled = kernel.warmup(...)`，逐阶段看 `compiled.asm`：

```python
compiled = kernel.warmup(x, y, out, n, alpha, BLOCK=block, grid=grid)
for stage in ("ttir", "ttgir", "llir", "ptx", "hgbin"):
    asm = compiled.asm.get(stage, "")
    print(stage, len(asm))
print("has local_pointers:", "tle.local_pointers" in compiled.asm["ttgir"])
print("tle residue in llir:", [l for l in compiled.asm["llir"].splitlines() if "tle." in l][:5])
```

4. 判断层次：**API 契约**（形状/索引/dtype）、**verifier**、**lowering**、**运行时**。

### 8.2 分症状对照

| 症状 | 可能层次 | 动作 |
|:--|:--|:--|
| `local_ptr indices must provide N tensors` / dtype 报错 | API / verifier | 核对索引数量 = buffer 秩、全整数、全标量或全张量、形状一致 |
| 编译通过但结果错 | kernel 逻辑 或 lowering | 缩小到一个 tile，打印中间 load/store |
| local store 后偶发不一致 | 顺序 / barrier | 检查 `TleInsertLocalPointerBarriers` 覆盖范围，简化控制流；手写 barrier 只在 pass 之外 |
| 加了 SMEM 暂存却没提速 | 布局转换 / 流水 | 对比 TTGIR 中布局转换数量与访存指令数，确认全局流量真的下降 |
| `tle.*` 残留在 llir | lowering 缺失 | 该原语在此后端没有转换 pattern |
| `unsupported TLE feature on the '<backend>' backend` | 后端拒绝列表 | 读该后端 `compiler.py` 的拒绝表，换等价写法 |
| `requires a Triton build with __TLE__ ...` | wheel 能力不足 | 该 wheel 用的是自研 TLE 变体（`__ILUVATAR_TLE__` / `__MCTLE__`），无此 op |
| `KeyError` / 无 `@dialect` 可用 | Raw 注册表 | 确认 `name` 在当前构建里被注册 |

### 8.3 定位代码位置的常用检索

```bash
rg -n "def local_ptr\(|analyze_local_pointer_operation" python/triton/experimental/tle/language/gpu
rg -n "LocalPointersOp::verify|kSharedMemoryAddressSpace" third_party/tle/dialect/lib/IR/Ops.cpp
rg -n "TleSelectEncodings|TleInsertLocalPointerBarriers" third_party/tle/dialect/lib/Transforms
rg -n "add_early_assign_memory_space|add_select_encodings" third_party/<backend>/backend/compiler.py
rg -n "UNSUPPORTED_TLE|reject_unsupported_tle" third_party/<backend>/backend/compiler.py
```

### 8.4 回归测试入口

```bash
# 公共 TLE 单测
python -m pytest python/test/tle/unit -vv
python -m pytest python/test/tle/integration -vv

# 单文件（定位问题最快）
python -m pytest python/test/tle/unit/test_tle_gpu_local_ptr.py -vv -s
python -m pytest python/test/tle/integration/test_tle_distributed.py -vv -s

# 后端专属
python -m pytest third_party/ppu/python/test/unit/tle -vv
python -m pytest third_party/mthreads/python/test/unit/tle -vv
python -m pytest third_party/iluvatar/python/test/unit/tle -vv

# 全量脚本
cd python/test/tle && python run_tests.py
```

### 8.5 调参优先级（Struct 层）

1. tile 尺寸（`BLOCK_M/N/K` 或 1D `BLOCK`）；
2. `num_warps`；
3. `num_stages`；
4. 访存路径（`gpu.copy` vs `local_ptr + tl.load/store`）；
5. 布局（`nv_mma_shared_layout`、swizzle 选择）。

单参数扫描法：固定 shape/seed/grid，**每次只改一个参数** → 跑正确性 → 跑计时 → 留 TTGIR/PTX 证据。连续 3 次单参数无收益即停。

### 8.6 编写新 TLE 功能时的文件地图

| 层 | 文件 |
|:--|:--|
| Python API | `language/__init__.py`、`language/core.py`、`language/gpu/core.py`、`language/gpu/semantic.py` |
| IR / verifier | `third_party/tle/dialect/include/IR/TleOps.td`、`.../lib/IR/Ops.cpp` |
| Lowering | `third_party/tle/dialect/lib/Conversion/TleToLLVM/*` |
| Pass 接线 | `third_party/tle/dialect/lib/Transforms/TleSelectEncodings.cpp`、`TleInsertLocalPointerBarriers.cpp`、`third_party/<backend>/backend/compiler.py` |
| 测试 | `python/test/tle/unit`、`python/test/tle/integration`、`third_party/<backend>/python/test` |

语义变更的最小测试集：**1 个正例 + 1 个反例契约 + 1 个回归例**（去掉修复就会失败）。

改上游原生 Triton 文件以承载 TLE 行为时，使用编译期开关 `#ifdef __TLE__` / `#endif`，不要用注释标记块做策略开关。

---

## 9. 已知不一致与版本差异

整理过程中确认的**文档 vs 源码**差异，写代码时以源码为准。

| # | 文档/教程写法 | 当前源码 | 处置 |
|:--:|:--|:--|:--|
| 1 | `x.extract_tile(index=[0,0], shape=[2,2])` | `tle.extract_tile(x, index, tile_shape)`，**无 `shape` 关键字，也不是张量方法** | 用 `tle.extract_tile(x, index=[...], tile_shape=[...])` |
| 2 | `z = x.insert_tile(y, index=[0,0])` | `tle.insert_tile(x, tile, index)` | 用模块级函数，`x` 在前 |
| 3 | `tle.reshard(x, spec=...)` 可用 | `reshard` 抛 `NotImplementedError("deferred to M4")` | 暂不使用 |
| 4 | `tle.distributed_dot(a, b, c)` 可用 | 抛 `NotImplementedError("deferred to M5")` | 暂不使用 |
| 5 | `tle.pipeline(...)`（Struct 旧文档） | GPU 侧为 `tle.gpu.*` 体系；`tle.dsa.pipeline` 属 DSA | 用 `tl.range(num_stages=)` 或 `tle.pipe` |
| 6 | `tle.alloc` / `tle.copy` / `tle.local_ptr` 顶层 | 实际由 `tle.gpu.*` 提供，`language/__init__.py` 只导出 `load`/`cumsum`/`extract_tile`/`insert_tile`/`pipe*`/分布式 | 新代码统一 `tle.gpu.*` |
| 7 | `scope=tle.gpu.storage_kind.smem` | `storage_kind` 只是 `memory_space` 函数的兼容别名，**没有 `.smem` 属性** | 用 `tle.gpu.smem` / `tle.gpu.tmem` |
| 8 | `tle.raw.call(fn, outputs, inputs)` 三参 | 二参 + 关键字：`call(func, args, output_indices=None, hint="")` | 用 `call(fn, [out, in, ...])` |
| 9 | `raw.call` 说明中的 `tle_raw.call` | 模块路径为 `triton.experimental.tle.language.raw`，也可 `from triton.experimental.tle.language import raw` | 按实际 import |
| 10 | PPU 白名单含 `distributed_barrier` | `PPU_UNSUPPORTED_TLE_OPS` 编译期拒绝 | 以拒绝列表为准 |
| 11 | `tle_supported.py` 视为能力清单 | 只是**前端白名单** | 结合单测与 pass 管线判断 |
| 12 | `wgmma_wait(1, acc)` 通用 | MThreads 只支持 `pendings=0` | 分支处理 |
| 13 | Wiki 3.4.1.1 用 `@tle.raw.language(name="mlir")` + `@tle.hardware_constraint(threads_dim=1, sync_scope="block")` | 全仓**无** `hardware_constraint` 符号，`tle.raw` 命名空间也不存在（正确路径是 `triton.experimental.tle.raw` 的 `dialect`） | 用 `@dialect(name="mlir")`，删掉 `hardware_constraint` |
| 14 | `tle.raw.call` / `tle_raw.call` 混用 | 实际模块是 `triton.experimental.tle.language.raw`（导出 `call`/`call_smem`），装饰器在 `triton.experimental.tle.raw` | 按 4.3 的原语表导入 |
| 15 | Wiki 3.4.1.1 示例 `from triton.experimental.flagtree.edsl import dialect` / `triton.experimental.flagtree.language as fl` | 该命名空间在 `main` 上不存在 | 用 `triton.experimental.tle.raw` |
| 16 | 认为 `@dialect` 的 `file=` 可位置传参 | `dialect` 为纯关键字签名，`registry[name](fn, **kwargs)` | 必须 `dialect(name=..., file=...)` |
| 17 | 认为改 `opt_level` 会自动重编 | `opt_level` 不在缓存键的 `_DIALECT_KWARG_KEYS` 中 | 改后手动清缓存或 `clear_pending_sources()` |
| 18 | 直觉认为 `wgmma` 的 `K >= 16` | 实际 `k >= 256 // bitwidth`：int8/FP8 要 **32**，fp16/bf16 要 16，fp32/TF32 只需 8 | 按 3.1.8 的表核对；int8 kernel 用 K=16 会失败 |
| 19 | 用 `nv_mma_shared_layout=False` 的 buffer 喂 `wgmma` | `_require_wgmma_smem_operand` 强制 `nv_mma_shared_layout` | 供 compute 的 SMEM 保持默认布局；要同时给 `copy` 用就分两块 buffer |
| 20 | 以为非 2 的幂 tile 全后端可用 | 非 2 的幂路径是 **NVIDIA 专有**，且要求 `FLAGTREE_BACKEND` 为空、`__TLE_TILED_SMEM__`、smem、无 alias、rank-2、`layout=None`、`nv_mma_shared_layout=True`、WGMMA 兼容 dtype | 非 NVIDIA 上坚持用 2 的幂 |
| 21 | named barrier 里 `barrier_arrive` 沿用分配时的线程数 | named 路径下 `barrier_arrive` 的 `arrive_count` **必须为 1** | 分配给线程总数，arrive 传 1 |
| 22 | `subslice` 一律要求对齐的 2 的幂 | 只有**多维元组形式**在 `size != extent` 时才要求；一维形式无此限制 | 一维切片可任意长度 |
| 23 | 以为 `copy(is_async=True)` 全后端可用 | MThreads 前端直接拒绝；只有 NVIDIA / Iluvatar 支持 | MThreads 上改用 `copy` 同步形式或 TME + barrier |
| 24 | 以为 `cache_policy` 只要后端是 MThreads 就能传 | 还必须**方向是 descriptor→shared** | 指针张量 copy 上传 policy 会报错 |

**其他版本敏感点**

- 各后端镜像标签与 Triton 版本强绑定（如 Iluvatar `202606-3.6-base` 对应 Triton 3.6，`202604-base` 对应 3.1），**不可混用**。
- `ascend` 走独立分支 `triton_v3.5.x`，其 `tle.pipe` / 分布式由 `language/dsa/ascend/` 提供，与 GPU 族语义不同（`device_mesh`/`remote`/`shard_id` 被覆盖）。
- `third_party/tle/README.md` 里的 `tle.pipeline()`、`tle.shared_layout` 等仍保留旧名，属历史文档。
- CommonIR 开关会改变 `alloc`/`copy` 的可用参数（仅 smem、无 alias），排查“同一份代码在不同 wheel 上行为不一致”时先查它。

---

## 10. 参考资料

**仓库内**

- `tle.md` / `tle_cn.md` —— TLE 架构设计（已含 API 参考与评估章节）
- `third_party/tle/README.md` —— TLE 方言与早期 API 说明
- `third_party/mthreads/tle/README.md` —— SQMMA 共享内存视图兼容规则
- `python/tutorials/tle/` —— 教程：`01-fft.py`、`02-moe_align_block_size.py`、`03-topk.py`、`04-cluster-gemm.py`、`05~09` 分布式集合通信、`deepseek_v32/`、`dsa/`、`raw/{cuda,mlir,nvshmem}/`
- `third_party/tle/tutorials/` —— Hopper warp-specialized GEMM / FA persistent kernel
- `skills/tle-developer/` —— 官方 TLE 开发 skill（`SKILL.md`、`references/tle-sources.md`）

**Wiki**

- `TLE`：`https://github.com/flagos-ai/FlagTree/wiki/TLE`
- `TLE-Raw`：`https://github.com/flagos-ai/FlagTree/wiki/TLE-Raw`
- `Hints` / `Architecture` / `FlagTree-Backend-Specialization`
- 后端用户手册：`User-manual-for-{nvidia,amd,iluvatar,hcu,mthreads,ppu,enflame,ascend,metax,sunrise,xpu,spacemit,tsingmicro,rpu,tileir}`

**站点**

- FlagOS 官网：`https://flagos.io`
- FlagOS 文档：`https://docs.flagos.io/zh-cn/latest/overview.html`
- FlagTree：`https://github.com/flagos-ai/FlagTree`
- FLIR：`https://github.com/flagos-ai/flir`
- 定制 LLVM：`https://github.com/flagos-ai/llvm-project`（`triton_3.6.x` 分支）
- LLVM wheel 构建器：`https://github.com/flagos-ai/flagtree_mlir`

---

*本文档由公开源码与 Wiki 整理，随上游演进需定期复核；结论以对应 FlagTree 版本的源码为准。*

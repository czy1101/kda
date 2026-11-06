# 环境与功能能力

## 区分当前证据与历史知识

先检查任务契约和当前进程所在主机。Windows 控制端没有 PPU 不代表远端没有；历史容器中有 acu 也不代表当前目标可采。`probe_environment.py` 用 `shutil.which` 和 argv 子进程，避免在 PowerShell 中执行 `command -v`；torch/triton 导入分别隔离并超时。输出 OS、Python、设备可见性、公开状态、工具 stdout/stderr 与目标。SDK 发布号未知时保持 null。

KDA profile `thead-zw810e` 使用 `backend: ppu`、`arch: ppu001`、`device: PPU-ZW810E`、`language: triton`；这是任务硬件分类。知识库已核验过 CUDA 兼容 target `cuda/80` 与 metadata `backend_name=ppu` 共存，不能用兼容 target 选择 NVIDIA profile。`ppu_10` 是 HGCC 编译目标，不作为默认 Triton arch 别名。

## 功能矩阵

| 状态 | 证据要求 | 后续 |
|---|---|---|
| not_installed | 找不到路径 | 继续其他工具 |
| not_runnable | version/help/导入失败 | 留原始错误 |
| untested | 路径/版本存在，未 capture | 不能声称可采集 |
| failed/restricted | capture/export 超时、退出失败、权限错误或空报告 | 同环境停止自动重试 |
| verified no-PCM | 当前基础 Trace 与导出确有匹配 kernel | Level 2，可解释时间线 |
| verified counters | 当前请求 sections 的采集/导出成功 | Level 1，仅解释实际字段 |

`runs/analysis-capabilities.json` 保留原 schema。补充 `runs/analysis-functional.json` 按工具保存环境指纹、状态、时间、命令、退出码、匹配规则与证据。KernelPilot 加载时合并状态到内存，不改原 discovery 文件；`strategy` 排除 failed。重复 discover-tools 不清除同环境失败。契约环境变化后旧功能证据不适用；相同 setup 字符串下系统实际变化不能自动感知，需用户明确重试，不擅自重试。

## 实测方法

先读取现有功能证据和嵌入知识，已知同环境接口直接采用，不重复读取帮助或运行支持性探针。必要的当前任务发现由 KernelPilot 执行 backend allowlist。capture/export 命令由任务 `profile.tools.acu` / `profile.tools.asys` 提供；保持任务环境、工作目录、容器与 transport 一致。只有尚无功能证据或本次请求范围不同才做最小验证；不能附着服务。验证也消耗任务 profile 预算，开始前查看已有记录和 `stop.max_profile_runs`。

`probe_capabilities.py` 不接收 shell 命令，只执行契约已有 candidate/export；每步限时并留日志。导出默认需 JSON `{"kernels":[{"name":"...","duration_us":1.0}]}`；这是验证器输入协议，不是 acu/asys 原生格式承诺。任务 export_command 必须使用版本验证过的解析器转换，且保留原生报告、字段/单位；也可指定 `--format csv --name-column "Kernel Name" --value-column <actual-field>`。列的语义和单位需人工确认，正数校验不证明所有 metric 正确。`--kernel` 必须是实际目标名，避免宽泛正则匹配无关 kernel。

某工具的一个请求成功不表示所有功能可用。asys 命令必须明确为无 counters 的基础 Trace；如果无法从当前 help/输出验证，标未验证，留在 Level 3。PyTorch Profiler 的 CPU 活动可独立记录，设备活动另做功能测试。Event 接口存在不代表计时已有效，须最小已知工作量、同步与 host 对照。

不主动运行系统监控变更、驱动升级、PCM 占用者清理。not-ready 常见解释不等于已确定根因；容器看不到占用者不证明设备独占。

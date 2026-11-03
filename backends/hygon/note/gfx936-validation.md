---
id: hygon-gfx936-validation
kind: guide
status: confirmed
device: BW
arch: gfx936
scope: "Validated architecture, ISA, and calibration anchors for the BW/gfx936 DTK image"
last_validated: 2026-11-03
revalidate_after: 2027-02-28
---

# Validated Hygon gfx936 reference

This document records a validated reference for the BW/gfx936 DTK image used
by KDA. It is a calibration aid for kernel optimization, not a vendor peak
specification and not an acceptance threshold. Re-run the probes after a DTK,
driver, Triton, or device change.

## Scope and environment

The validation was performed in the `xiaodan-dev` container on one BW card
selected with `HIP_VISIBLE_DEVICES=6`:

```bash
source /opt/dtk-26.04-DCC2602-0317/env.sh
export KDA_DTK_ROOT=/opt/dtk-26.04-DCC2602-0317
export PATH="$KDA_DTK_ROOT/rocprofiler/bin:$KDA_DTK_ROOT/hip/bin:$KDA_DTK_ROOT/bin:$PATH"
export HIP_VISIBLE_DEVICES=6
unset ROCR_VISIBLE_DEVICES DCU_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
```

The observed software versions were PyTorch 2.9.0, HIP 6.3.26045, and Triton
3.6.0 targeting `gfx936` with `warp_size=64`.

## Verified architecture facts

The following values came from `rocminfo`, `rocm-smi`, PyTorch device
properties, and `lspci`; they are safe to use as initial launch constraints for
this BW/gfx936 image:

| Property | Validated value |
|---|---|
| ISA target | `amdgcn-amd-amdhsa--gfx936:sramecc+:xnack-` |
| Compute units | 80 CU |
| SIMD per CU | 4 |
| Wavefront | 64 threads (Wave64) |
| Maximum waves per CU | 40 |
| Maximum workgroup | 1024 work-items |
| Maximum work-items per CU | 2560 |
| Group/LDS pool | 64 KiB |
| L1 cache | 32 KiB |
| L2 cache | 8192 KiB (8 MiB) |
| VRAM reported by `rocm-smi` | 65520 MiB |
| PCIe link | 32 GT/s, x16 |

These facts do not provide a direct H100-to-BW speed ratio. In particular,
`num_warps` is interpreted through Wave64: four warps corresponds to 256
threads and eight warps to 512 threads. More warps can increase register and
LDS pressure without improving throughput.

## Calibration measurements

The measurements below are reproducibility anchors for the validated image.
They are not theoretical HBM or vendor peak values.

### Device-to-device bandwidth

`BandwidthTest -t 2` reported:

| Transfer size | Average effective bandwidth | Peak effective bandwidth |
|---|---:|---:|
| 512 MiB | 1348.992 GB/s | 1350.277 GB/s |
| 1 GiB | 1354.101 GB/s | 1355.599 GB/s |

This is a large device-to-device copy measurement. It should be used as an
effective streaming-copy baseline, not as the HBM theoretical peak. Do not
derive HBM bandwidth from `mclk` alone.

### PyTorch GEMM calibration

For square `torch.mm` with `N=8192`, five warmup iterations, ten timed
iterations, and CUDA event timing, the median results were:

| Input/output dtype | Median time | Effective throughput |
|---|---:|---:|
| FP16 | 3.607198 ms | 304.810 TFLOPS |
| BF16 | 3.601438 ms | 305.298 TFLOPS |
| FP32 | 19.874309 ms | 55.323 TFLOPS |
| FP64 | 46.629332 ms | 23.580 TFLOPS |

These numbers characterize the PyTorch GEMM path in this environment. They do
not replace a standalone peak benchmark and should not be copied into a task
acceptance contract without a fresh run.

PyTorch `torch.mm` for `float8_e4m3fn`, `float8_e5m2`, `float8_e4m3fnuz`, and
`float8_e5m2fnuz` raised `NotImplementedError` (`addmm_cuda` is not
implemented). This is a framework capability result, not proof that the
hardware lacks native FP8 matrix instructions.

## ISA evidence for the current Triton path

For the current hy3 prefill kernel, the HSACO generated in the same run was
disassembled with the DTK LLVM objdump. The observed instruction mix included:

- `v_mmac_f32_16x16x16_f16` matrix instructions;
- `buffer_load` instructions;
- `ds_write` and `ds_read` LDS instructions;
- `s_waitcnt` and `s_barrier` synchronization.

The disassembly contained no `buffer_load.*lds` pattern. The observed data
movement is therefore ordinary global load to VGPR, LDS store, synchronization,
and LDS read. Source-level names such as `async`, `commit`, or `wait` are not
enough to claim a hardware direct global-to-LDS asynchronous path.

## LDS tiling and software double buffering as a gfx936 candidate

Treat software-managed LDS staging and double buffering as a testable Hygon
optimization candidate. It is analogous to TMA only in its goal of staging
tiles and overlapping data movement with computation; it is not a TMA engine
or an asynchronous bulk-copy engine.

### Current toolchain probe

On 2026-10-19, a compile-only probe was run in `xiaodan-dev` on physical BW
device 5 (`ROCR_VISIBLE_DEVICES=5`), targeting `gfx936` with DTK
`dtk-26.04-DCC2602-0317`, LLVM/clang 18, and Triton 3.6.0:

- `global_load_lds_dword` was rejected by the assembler as unsupported for
  `gfx936`; the same opcode form was accepted for `gfx942`.
- `__builtin_amdgcn_global_load_lds` was unknown to the installed clang.
- The tested `buffer_load_* ... lds` forms were rejected for `gfx936`.
- A TLE descriptor copy with a completion barrier failed during compilation
  with `TMA completion barrier is only supported on NVIDIA backend`. Removing
  the barrier did not yield an HSACO: HCU lowering failed on the tensor
  descriptor conversion.
- An ordinary TLE global-to-local copy compiled to a sequence including
  `buffer_load_dwordx4`, `ds_write_b128`, `s_barrier`, and `ds_read_u16`.
  This confirms the ordinary VMEM/VGPR/LDS route, not a direct transfer engine.
- The probe's HCU pipeline reported `use_async_copy=false` and
  `use_pingpong=false`; the backend helper also returns disabled ping-pong by
  default for `gfx936`.

These are compiler/toolchain observations, not proof that no private or future
Hygon ISA has any related capability. Re-run after changing the DTK, LLVM,
Triton, driver, or device. Do not emit or execute unrecognized inline ISA as a
substitute for a supported path.

### When to try the software-managed path

Consider it when profiling indicates repeated tile reads, meaningful reuse of
K/V or matrix operands within a CTA, and enough independent computation to
overlap the next tile's load with the current tile's compute. A candidate may:

1. Load a coalesced tile using ordinary global/buffer loads.
2. Stage the tile in CTA-local LDS and synchronize before consumers read it.
3. Use two LDS buffers so the next tile can be prepared while the current tile
   is consumed, with explicit producer/consumer ordering.
4. Compare against the non-staged kernel using the same correctness and
   benchmark workload.

This is less promising when each value is consumed once, access is highly
irregular, the kernel is already launch-bound, or LDS/register pressure lowers
occupancy. Two stages consume roughly twice the staged-tile LDS capacity, so
check the 64 KiB per-workgroup limit together with VGPR/SGPR, workgroup size,
barrier cost, and tail handling. Do not assume the compiler will create
ping-pong overlap automatically on gfx936.

Promote only if correctness passes and alternating candidate/reference
benchmarks show a stable gain across representative shapes. Record the HSACO
instruction path and resource columns. Describe the result as software-managed
LDS staging/overlap, never as TMA or direct global-to-LDS transfer unless a
future task-local disassembly proves that path.

For a new candidate, always disassemble the HSACO generated by that candidate:

```bash
llvm-objdump -d path/to/current-run.hsaco > profile/current-isa.txt
grep -nEi 'buffer_load|ds_(read|write)|s_waitcnt|s_barrier|mmac|mfma|wmma' \
  profile/current-isa.txt
```

## Profile evidence and limitations

`rocprofv2 --kernel-trace` produced a usable `results_kernels.csv`. In a
representative hy3 prefill run (`Q=512`, `KV=4096`, quant1, NHD, no mask), the
main kernel had 112 launches with these stable resource fields:

```text
Kernel:     _bsa_fp8_prefill_kernel.kd
Grid:       32768
Workgroup:  256
Wave size:  64
LDS:        32768 bytes
Arch_VGPR:  256
SGPR:       112
Median one-kernel duration: 6.191676 ms
```

The current DTK image listed counters such as `SQ_INSTS_VALU`,
`SQ_INSTS_VMEM_RD`, `SQ_INSTS_VMEM_WR`, `SQ_INSTS_LDS`, `SQ_WAVES`, and
`SQ_BUSY_CYCLES`. However, the attempted `rocprofv2 -i` capture returned zero
without producing a counter result file.

Re-checked on 2026-11-03 in `jiangrenkai-dev` (DTK `26.04-DCC2602-0317`) with a
1024×1024 BF16 `torch.mm` workload:

- `rocprofv2 -i pmc.txt -d out -o counters <app>` exited 0, printed nothing, and
  created no file — the v2 counter path is still broken on this image;
- `rocprof -i pmc.txt -o counters.csv <app>` (v1 syntax `pmc: SQ_INSTS_VALU`)
  produced `counters.csv` with the counter column populated.

Treat numeric PMC data as unavailable from v2; route counter work through
`rocprof` v1 and still smoke-test it, because the path is image-dependent. Never
treat missing counter output as zero. Command details:
[profiling-tools.md](profiling-tools.md).

Use `hipprof --hip-trace --stats` for launch gaps and synchronization questions.
Use `rocm-smi` and `rocminfo` only as device-state and architecture probes, not
as kernel profilers.

## KDA optimization rules for this target

1. Start with Wave64-aware `num_warps` and `num_stages` sweeps. Always record
   workgroup size, LDS, VGPR, SGPR, and end-to-end latency together.
2. Treat 64 KiB LDS and the observed register usage as joint constraints. A
   larger tile, extra stage, or scale staging buffer needs a resource check and
   a multi-shape benchmark.
3. For paged or sparse attention, profile indirect page-table loads, K/V
   loads, scale loads, and matrix instructions separately where possible.
4. Do not promote an FP8 optimization as a native FP8 matrix optimization when
   the generated ISA uses an FP16 MMAC path.
5. Keep HBM peak, effective D2D bandwidth, GEMM throughput, and operator
   latency as separate metrics with separate units.
6. If PMC capture is unavailable, use kernel trace plus HSACO disassembly and
   explicitly mark VMEM/VALU/LDS/occupancy conclusions as qualitative.
7. Retain failed candidates. A candidate that passes correctness but regresses
   q0 or long-sequence workloads must be rejected and the best correct
   checkpoint restored.

## Still unverified

The following remain intentionally open rather than being filled with inferred
values:

- vendor/theoretical HBM peak bandwidth;
- cross-card P2P bandwidth and interconnect topology performance;
- native FP8 MMAC peak;
- numeric VMEM/VALU/LDS/occupancy counters from rocprofv2;
- performance benefit of software-managed LDS double buffering on a new
  operator/workload;
- whether a private or future Hygon toolchain exposes a dedicated direct
  global-to-LDS asynchronous path.

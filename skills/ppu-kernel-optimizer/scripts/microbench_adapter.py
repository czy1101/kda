"""Bounded Triton probes. Loaded only in an authorized executor worker."""
import hashlib
import json
import os
from pathlib import Path


def build(config):
    import torch
    import triton
    import triton.language as tl
    from common import run
    import importlib.metadata
    if not torch.cuda.is_available():
        raise RuntimeError("没有可用的兼容设备接口")
    name = torch.cuda.get_device_name()
    if not any(word in name.lower() for word in ("ppu", "810")):
        raise RuntimeError("设备未识别为 PPU；拒绝用其他设备冒充 PPU 验证")
    versions = {"torch": torch.__version__, "triton": triton.__version__,
                "triton_path": triton.__file__,
                "runtime_target": str(triton.runtime.driver.active.get_current_target()),
                "driver_status": run(["ppu-smi", "-v"], timeout=5)}
    for package in ("flagtree", "triton-for-sail"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    # The command timestamp is provenance, not a comparison dimension.
    versions["driver_status"].pop("tested_at", None)

    @triton.jit
    def vector(X, Y, N: tl.constexpr, STRIDE: tl.constexpr, BLOCK: tl.constexpr,
               ITERS: tl.constexpr, MODE: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        if MODE == "write":
            v = i.to(tl.float32)
        elif MODE == "dependency":
            idx = i
            for _ in range(ITERS):
                idx = tl.load(X + idx, i < N, 0).to(tl.int32)
            v = idx
        else:
            v = tl.load(X + i * STRIDE, i < N, 0)
            if MODE == "fma":
                for _ in range(ITERS):
                    v = tl.fma(v, 1.00001, 0.001)
            if MODE == "reuse":
                # A recurrence keeps reused values observable; not a DRAM traffic count.
                for _ in range(ITERS):
                    v = tl.fma(v, 0.99999, 0.001)
        if MODE == "read":
            total = tl.sum(v, 0)
            tl.store(Y + tl.program_id(0), total)
        else:
            tl.store(Y + i, v, i < N)

    @triton.jit
    def matmul(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        m = tl.program_id(0) * BM + tl.arange(0, BM)
        n = tl.program_id(1) * BN + tl.arange(0, BN)
        k = tl.arange(0, BK)
        acc = tl.full((BM, BN), 0, tl.float32)
        for start in range(tl.cdiv(K, BK)):
            kk = start * BK + k
            a = tl.load(A + m[:, None] * K + kk[None, :],
                        (m[:, None] < M) & (kk[None, :] < K), 0)
            b = tl.load(B + kk[:, None] * N + n[None, :],
                        (kk[:, None] < K) & (n[None, :] < N), 0)
            acc += tl.dot(a, b)
        tl.store(C + m[:, None] * N + n[None, :], acc,
                 (m[:, None] < M) & (n[None, :] < N))

    class Adapter:
        def prepare(self):
            n, stride, mode = config["n"], config["stride"], config["mode"]
            self.mode = mode
            if mode == "gemm":
                dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[config["dtype"]]
                self.x = torch.full((n, n), .125, dtype=dtype, device="cuda")
                self.b = torch.full((n, n), .25, dtype=dtype, device="cuda")
                self.y = torch.empty((n, n), dtype=torch.float32, device="cuda")
            else:
                self.x = (torch.arange(n, device="cuda", dtype=torch.int32) + stride) % n if mode == "dependency" else torch.ones(n * stride, device="cuda")
                size = triton.cdiv(n, config["block"]) if mode == "read" else n
                self.y = torch.empty(size, device="cuda", dtype=torch.int32 if mode == "dependency" else torch.float32)
            self.launch()

        def launch(self):
            n = config["n"]
            if self.mode == "gemm":
                tile = config["block"]
                self.last = matmul[(triton.cdiv(n, tile), triton.cdiv(n, tile))](
                    self.x, self.b, self.y, n, n, n, tile, tile, 32,
                    num_warps=config["warps"], num_stages=config["stages"])
            else:
                self.last = vector[(triton.cdiv(n, config["block"]),)](
                    self.x, self.y, n, config["stride"], config["block"],
                    config["iters"], self.mode, num_warps=config["warps"],
                    num_stages=config["stages"])

        def check(self):
            n = config["n"]
            if self.mode == "gemm":
                expected = torch.full_like(self.y, n * .125 * .25)
            elif self.mode == "read":
                expected = torch.full_like(self.y, config["block"])
                expected[-1] = n - (len(self.y) - 1) * config["block"]
            elif self.mode == "write":
                expected = torch.arange(n, dtype=torch.float32, device="cuda")
            elif self.mode == "dependency":
                expected = (torch.arange(n, device="cuda", dtype=torch.int32) + config["stride"] * config["iters"]) % n
            else:
                expected = torch.ones_like(self.y)
                if self.mode in ("fma", "reuse"):
                    for _ in range(config["iters"]):
                        expected = expected * (1.00001 if self.mode == "fma" else .99999) + .001
            torch.testing.assert_close(self.y, expected, rtol=1e-4, atol=1e-4)

        def synchronize(self):
            torch.cuda.synchronize()

        def events(self):
            return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        def describe(self):
            identity = {"device": name, **versions,
                        "visibility": os.environ.get("CUDA_VISIBLE_DEVICES"),
                        "properties": str(torch.cuda.get_device_properties(0))}
            return {"shape": list(self.x.shape), "stride": list(self.x.stride()),
                    "dtype": str(self.x.dtype), "device": name,
                    "environment_fingerprint": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
                    "versions": identity, "config": config, "config_fixed": True,
                    "scope": "single_kernel_preallocated", "cache_protocol": "warm_repeated",
                    "execution_kind": "ppu"}

        def compiled(self):
            self.prepare()
            self.check()
            self.synchronize()
            desc = self.describe()
            return self.last, {"candidate": "microbench", "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                               "signature": str(self.x.dtype), "shape": desc["shape"],
                               "dtype": desc["dtype"], "stride": desc["stride"],
                               "grid": ([(config["n"] + config["block"] - 1) // config["block"]] if self.mode != "gemm" else [(config["n"] + config["block"] - 1) // config["block"]] * 2),
                               "config": config, "target": versions["runtime_target"],
                               "versions": desc["versions"]}

    return Adapter()

# Candidate/reference comparison checklist

## Establish comparability

- Same device and relevant software versions.
- Same input shapes, dtypes, layouts, and semantic workload.
- Same warmup, synchronization, repetition, and capture policy.
- Same clock or power policy when those conditions are controlled.
- Complete kernel sequence for both implementations.

## Compare in layers

1. End-to-end latency and kernel count.
2. Kernel decomposition, launch dimensions, and time distribution.
3. Memory traffic, cache behavior, and achieved bandwidth.
4. Occupancy, registers, shared memory, and active warps.
5. Warp stalls, scheduler issue behavior, and synchronization.
6. Instruction mix and use of specialized compute paths.
7. Available Triton IR, PTX, and final machine instructions.

Normalize interpretations by useful work. A faster reference may fuse work,
remove an intermediate, or use a different algorithm; raw per-kernel counter
differences are not automatically actionable.

## Required conclusion

Identify the largest defensible performance gap, cite the supporting artifacts,
name uncertainties, and propose one change whose expected counter and benchmark
effects can be tested.

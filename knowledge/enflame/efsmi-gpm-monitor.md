# S60-EVIDENCE-002: read-only efsmi GPM monitoring

Observed2026-10-16, S60/GCU300, efsmi1.10.6 at /usr/bin/efsmi inside the
contracted container. This is device-level monitoring, not a per-kernel
profiler or an automatically available KDA backend profile. Revalidate the
installed executable, help, device mapping and actual samples before reuse.

## Interface

After read-only help/version/device inventory, this command returned samples:

```bash
efsmi dmon -i 0 --gpm-metrics 2,3,18,19 -c 8 -w 1 -o
```

Metric2 reports SIP utilization;3 SIP occupancy (SIPUsed column);18/19
DRAM read/write MiB/s. Installed help recommends one-second sampling; too
short intervals can yield invalid deltas/N/A. Device0 was matched to the
GCU ordinal and PCI inventory with no visibility remapping in this run.
Check device mapping again when visibility variables or containers differ.
No privileged enable, counter configuration or clock change was required.

## Active-load validation and limits

Idle samples returned zero. Separate sustained real MSA Attention loops
reported roughly98-99% SIP busy. The predeclared summary excludes first/last
of eight samples and retains every raw row. Indices and warmup precede
monitoring. The synchronized loop's calls/s includes dispatch and sync and
does not replace performance-only event benchmarks.

DRAM read/write stayed zero under these active loads: treat those fields as
unavailable, not evidence of no traffic, cache residency, or spare bandwidth.
SIP utilization/occupancy do not measure convgen matrix-pipeline utilization,
DTE utilization, stalls, overlap or arithmetic efficiency. A faster kernel
need not show a higher already-saturated busy percentage. Establish actual
matrix/copy lowering from runtime-linked compiler/ISA evidence separately.

The top-level VPU section in this installed efsmi describes video encoder and
decoder activity. Do not mistake it for vector-compute utilization.

KDA discovery still had no matching declared GCU profiler profile. This
read-only monitor was separately inventoried for the explicit utilization
request; it was not presented as a discovery-selected profiler. Raw logs,
helper, sample policy and source pointers remain in the task workspace.

# Troubleshooting AMD GPUs

XXX: this is very early - collecting various tools/notes

As most of us are well familiar with NVIDIA tools, I will try to provide the mapping where possible to the familiar tools.

The overall NVIDIA-to-AMD mapping of the software stack (ROCm, HIP, `amdgpu`, `rocm-smi`, `rocminfo`) is in [AMD / ROCm equivalents](../../gpu-software-stack.md#8-amd--rocm-equivalents), and the hardware terms (CU, wavefront, XCD, LDS) in [AMD differences](../execution-model.md#amd-differences).

## Tools

### HIP_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES

ROCm has more than one “visible devices” knob because isolation can happen at different layers of the stack. From the top down: PyTorch calls **HIP**, AMD's CUDA-like runtime API, and HIP sits on **ROCr** (the ROCm runtime, AMD's implementation of the HSA, Heterogeneous System Architecture, runtime API), which talks to the kernel driver. For PyTorch on ROCm the usual CUDA-style choice is HIP-level:

```bash
HIP_VISIBLE_DEVICES=0,1 python my-program.py
```

`CUDA_VISIBLE_DEVICES` is honored the same way (CUDA-compat alias for HIP on AMD).

That only filters what the **HIP** runtime exposes. ROCr can still initialize every GPU, and non-HIP clients sitting on ROCr (OpenCL, a vendor-neutral GPU compute API; AOMP, AMD's OpenMP offloading compiler; UCX, a communication library that MPI and others use; …) are unaffected.

To hide devices from the whole user-mode ROCm stack, filter at **ROCr**:

```bash
ROCR_VISIBLE_DEVICES=0,1 python my-program.py
```

`ROCR_VISIBLE_DEVICES` also accepts UUID strings (HIP’s list is indices only). If both are set, HIP sees only the devices ROCr left visible, so HIP indices are relative to that filtered set.

On Linux, AMD’s isolation docs lean on `ROCR_VISIBLE_DEVICES` when you want stack-wide isolation; for ordinary single-framework PyTorch work, `HIP_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES` is enough. See [GPU isolation techniques](https://rocm.docs.amd.com/en/latest/reference/system-optimization/gpu-isolation.html).

### rocm-smi

`rocm-smi` (`nvidia-smi` equivalent) shows a condensed state of all the ROCm accelerators.

For example here is an 8xMI300X node:
```bash
$ rocm-smi
========================================= ROCm System Management Interface =========================================
=================================================== Concise Info ===================================================
Device  [Model : Revision]    Temp        Power     Partitions      SCLK    MCLK    Fan  Perf  PwrCap  VRAM%  GPU%
        Name (20 chars)       (Junction)  (Socket)  (Mem, Compute)
====================================================================================================================
0       [0x74a1 : 0x00]       45.0°C      173.0W    NPS1, SPX       132Mhz  900Mhz  0%   auto  750.0W    0%   0%
        AMD Instinct MI300X
1       [0x74a1 : 0x00]       41.0°C      179.0W    NPS1, SPX       132Mhz  900Mhz  0%   auto  750.0W    0%   0%
        AMD Instinct MI300X
2       [0x74a1 : 0x00]       47.0°C      180.0W    NPS1, SPX       131Mhz  900Mhz  0%   auto  750.0W    0%   0%
        AMD Instinct MI300X
3       [0x74a1 : 0x00]       45.0°C      178.0W    NPS1, SPX       131Mhz  900Mhz  0%   auto  750.0W   17%   0%
        AMD Instinct MI300X
4       [0x74a1 : 0x00]       45.0°C      175.0W    NPS1, SPX       132Mhz  900Mhz  0%   auto  750.0W    0%   0%
        AMD Instinct MI300X
5       [0x74a1 : 0x00]       43.0°C      175.0W    NPS1, SPX       132Mhz  900Mhz  0%   auto  750.0W    0%   0%
        AMD Instinct MI300X
6       [0x74a1 : 0x00]       45.0°C      175.0W    NPS1, SPX       132Mhz  900Mhz  0%   auto  750.0W    0%   0%
        AMD Instinct MI300X
7       [0x74a1 : 0x00]       43.0°C      176.0W    NPS1, SPX       132Mhz  900Mhz  0%   auto  750.0W    0%   0%
        AMD Instinct MI300X
====================================================================================================================
=============================================== End of ROCm SMI Log ================================================
```

How to read the columns: `Temp (Junction)` is the hottest on-die sensor, `Power (Socket)` the whole package's draw, `Partitions (Mem, Compute)` the memory and compute partitioning modes (`NPS1` = all HBM as one NUMA node, `SPX` = all XCDs as one GPU; the MI300X can also be split into several smaller GPUs), `SCLK` the compute (shader) clock, `MCLK` the memory clock, `Perf` the performance level policy, `PwrCap` the power limit, `VRAM%` the share of HBM in use and `GPU%` the busy percentage - which, like `nvidia-smi`'s `GPU-Util`, is the share of time something ran, not how much of the GPU it used (see [How to get the real GPU utilization metrics](../nvidia/debug.md#how-to-get-the-real-gpu-utilization-metrics)). The 132 MHz `SCLK` here is an idle GPU's clock.

Oddly it shows no real memory usage - only the percentage, which isn't very practical. (`rocm-smi --showmeminfo vram` prints the used and total bytes per GPU.)

A handy alias to watch updates in real time:
```bash
alias wr='watch -n 1 rocm-smi'
```

(`watch -n 1` re-runs the command every second; put the alias in your `~/.bashrc` to keep it.)

### rocminfo

`rocminfo` (`nvidia-smi -q` equivalent) shows the detailed information about each accelerator.

This one shows both the CPU and the GPU information

Here is a snippet for cpu0 and gpu0 (note it starts counting the cpus as nodes 0..1, and then GPUs as nodes 2..9). HSA calls every processor that can run work an *agent*, CPU or GPU. The fields that matter most for a GPU are `Name` (`gfx942` is the ISA name of the MI300X), `Compute Unit` (304 CUs), `Wavefront Size` (64), `Max Clock Freq.`, the cache sizes, `BDFID` (the PCI bus:device.function address packed into one number) and the `Pool Info` sizes, which are the HBM (here 192GiB per GPU). *Coarse grained* memory is coherent only at kernel boundaries and is the fast default for device memory; *fine grained* memory can be shared coherently with the host or other GPUs while a kernel runs. In the ISA string, `sramecc+` means ECC on the on-chip SRAM is enabled and `xnack-` that GPU page-fault retry is off:
```bash
$ rocminfo
ROCk module is loaded
=====================
HSA System Attributes
=====================
Runtime Version:         1.1
System Timestamp Freq.:  1000.000000MHz
Sig. Max Wait Duration:  18446744073709551615 (0xFFFFFFFFFFFFFFFF) (timestamp count)
Machine Model:           LARGE
System Endianness:       LITTLE
Mwaitx:                  DISABLED
DMAbuf Support:          YES

==========
HSA Agents
==========
*******
Agent 1
*******
  Name:                    AMD EPYC 9534 64-Core Processor
  Uuid:                    CPU-XX
  Marketing Name:          AMD EPYC 9534 64-Core Processor
  Vendor Name:             CPU
  Feature:                 None specified
  Profile:                 FULL_PROFILE
  Float Round Mode:        NEAR
  Max Queue Number:        0(0x0)
  Queue Min Size:          0(0x0)
  Queue Max Size:          0(0x0)
  Queue Type:              MULTI
  Node:                    0
  Device Type:             CPU
  Cache Info:
    L1:                      32768(0x8000) KB
  Chip ID:                 0(0x0)
  ASIC Revision:           0(0x0)
  Cacheline Size:          64(0x40)
  Max Clock Freq. (MHz):   2450
  BDFID:                   0
  Internal Node ID:        0
  Compute Unit:            128
  SIMDs per CU:            0
  Shader Engines:          0
  Shader Arrs. per Eng.:   0
  WatchPts on Addr. Ranges:1
  Features:                None
  Pool Info:
    Pool 1
      Segment:                 GLOBAL; FLAGS: FINE GRAINED
      Size:                    792303268(0x2f3996a4) KB
      Allocatable:             TRUE
      Alloc Granule:           4KB
      Alloc Alignment:         4KB
      Accessible by all:       TRUE
    Pool 2
      Segment:                 GLOBAL; FLAGS: KERNARG, FINE GRAINED
      Size:                    792303268(0x2f3996a4) KB
      Allocatable:             TRUE
      Alloc Granule:           4KB
      Alloc Alignment:         4KB
      Accessible by all:       TRUE
    Pool 3
      Segment:                 GLOBAL; FLAGS: COARSE GRAINED
      Size:                    792303268(0x2f3996a4) KB
      Allocatable:             TRUE
      Alloc Granule:           4KB
      Alloc Alignment:         4KB
      Accessible by all:       TRUE
  ISA Info:
[...]

  Name:                    gfx942
  Uuid:                    GPU-ababaeeffecddc50
  Marketing Name:          AMD Instinct MI300X
  Vendor Name:             AMD
  Feature:                 KERNEL_DISPATCH
  Profile:                 BASE_PROFILE
  Float Round Mode:        NEAR
  Max Queue Number:        128(0x80)
  Queue Min Size:          64(0x40)
  Queue Max Size:          131072(0x20000)
  Queue Type:              MULTI
  Node:                    2
  Device Type:             GPU
  Cache Info:
    L1:                      16(0x10) KB
    L2:                      8192(0x2000) KB
  Chip ID:                 29857(0x74a1)
  ASIC Revision:           1(0x1)
  Cacheline Size:          64(0x40)
  Max Clock Freq. (MHz):   2100
  BDFID:                   50688
  Internal Node ID:        7
  Compute Unit:            304
  SIMDs per CU:            4
  Shader Engines:          32
  Shader Arrs. per Eng.:   1
  WatchPts on Addr. Ranges:4
  Coherent Host Access:    FALSE
  Features:                KERNEL_DISPATCH
  Fast F16 Operation:      TRUE
  Wavefront Size:          64(0x40)
  Workgroup Max Size:      1024(0x400)
  Workgroup Max Size per Dimension:
    x                        1024(0x400)
    y                        1024(0x400)
    z                        1024(0x400)
  Max Waves Per CU:        32(0x20)
  Max Work-item Per CU:    2048(0x800)
  Grid Max Size:           4294967295(0xffffffff)
  Grid Max Size per Dimension:
    x                        4294967295(0xffffffff)
    y                        4294967295(0xffffffff)
    z                        4294967295(0xffffffff)
  Max fbarriers/Workgrp:   32
  Packet Processor uCode:: 132
  SDMA engine uCode::      19
  IOMMU Support::          None
  Pool Info:
    Pool 1
      Segment:                 GLOBAL; FLAGS: COARSE GRAINED
      Size:                    201310208(0xbffc000) KB
      Allocatable:             TRUE
      Alloc Granule:           4KB
      Alloc Alignment:         4KB
      Accessible by all:       FALSE
    Pool 2
      Segment:                 GLOBAL; FLAGS: EXTENDED FINE GRAINED
      Size:                    201310208(0xbffc000) KB
      Allocatable:             TRUE
      Alloc Granule:           4KB
      Alloc Alignment:         4KB
      Accessible by all:       FALSE
    Pool 3
      Segment:                 GLOBAL; FLAGS: FINE GRAINED
      Size:                    201310208(0xbffc000) KB
      Allocatable:             TRUE
      Alloc Granule:           4KB
      Alloc Alignment:         4KB
      Accessible by all:       FALSE
    Pool 4
      Segment:                 GROUP
      Size:                    64(0x40) KB
      Allocatable:             FALSE
      Alloc Granule:           0KB
      Alloc Alignment:         0KB
      Accessible by all:       FALSE
  ISA Info:
    ISA 1
      Name:                    amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-
      Machine Models:          HSA_MACHINE_MODEL_LARGE
      Profiles:                HSA_PROFILE_BASE
      Default Rounding Mode:   NEAR
      Default Rounding Mode:   NEAR
      Fast f16:                TRUE
      Workgroup Max Size:      1024(0x400)
      Workgroup Max Size per Dimension:
        x                        1024(0x400)
        y                        1024(0x400)
        z                        1024(0x400)
      Grid Max Size:           4294967295(0xffffffff)
      Grid Max Size per Dimension:
        x                        4294967295(0xffffffff)
        y                        4294967295(0xffffffff)
        z                        4294967295(0xffffffff)
      FBarrier Max Size:       32
```

## Hangs or slow multi-GPU runs and IOMMU

If a multi-GPU ROCm run hangs or crawls, the host's IOMMU configuration is a common culprit. The IOMMU translates the addresses that devices use for DMA, and pass-through mode (`iommu=pt`) skips that translation for devices the host owns; both are introduced in [IOMMU, VFIO and PCIe passthrough](../../../foundations/containers-and-virtualization.md#iommu-vfio-and-pcie-passthrough). The correct setting is platform-specific - match your hardware instead of blindly disabling it:

- **Modern Instinct (MI300X, MI325X, MI350X, MI355X)**: AMD's system-acceptance guides want the IOMMU **enabled** in BIOS and running in pass-through mode on the kernel command line - `iommu=pt` together with `amd_iommu=on` (or `intel_iommu=on` on Intel hosts). MI350X/MI355X (ROCm 7.0.1+) explicitly list `IOMMU enabled` + `iommu=pt`.
- **MI100 / MI200 (MI210/MI250/MI250X)**: AMD's qualified BIOS tables typically set the NBIO **IOMMU to Disable**, with `iommu=pt` called out only for hosts with 256+ logical CPUs (so X2APIC, the interrupt-controller mode that can address more than 255 CPUs and needs the IOMMU for interrupt remapping, can address every core). NBIO is the CPU's north-bridge I/O block, where the BIOS's IOMMU switch lives.
- **Legacy last resort**: on older systems where neither pass-through nor the BIOS default helped, fully disabling it via `amd_iommu=off` (or the softer `iommu=soft`) was the historical workaround - see AMD's [IOMMU advisory for multi-GPU environments](https://community.amd.com/t5/knowledge-base/iommu-advisory-for-multi-gpu-environments/ta-p/477468) and this [issue report](https://github.com/stas00/ml-engineering/issues/1#issuecomment-1076830400) (that particular case was actually on NVIDIA A6000s). This is an invasive, system-wide boot change, and current AMD guidance prefers pass-through over a full disable.

These are boot-time kernel parameters set in `/etc/default/grub` (e.g. `GRUB_CMDLINE_LINUX="... iommu=pt"`; the grub config path varies by OS), followed by regenerating grub and rebooting - so they need admin access. GRUB is the bootloader that passes these parameters to the kernel; on Ubuntu/Debian regenerate with `sudo update-grub`, on RHEL-family systems with `sudo grub2-mkconfig -o <path of grub.cfg>`, and after the reboot `cat /proc/cmdline` shows the parameters the running kernel actually got.

The per-GPU `IOMMU Support` line in the `rocminfo` output above reflects what the runtime actually sees.

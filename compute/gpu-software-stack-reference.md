# GPU Nodes: Driver Install, Acceptance Testing and Troubleshooting

Reference material split out of [The GPU Node Software Stack From Zero: Drivers, CUDA, NCCL and Fabric Bring-up](gpu-software-stack.md) so the main chapter reads faster. The sections stand alone: read them when you need them. Earlier chapters link straight to the section they use.

## Installing and upgrading the NVIDIA driver on Ubuntu

### Where the driver can come from

On Ubuntu 22.04/24.04/26.04 there are three sources. Pick one per fleet and never mix them on the same node.

| Source | Package names (as of Nov 2026) | Pros | Cons |
|---|---|---|---|
| Ubuntu archive ("restricted") | `nvidia-driver-570-server-open`, `nvidia-fabricmanager-570`, `nvidia-utils-570-server` | Canonical signs the modules for Secure Boot. `ubuntu-drivers` picks one for you. | lags NVIDIA by weeks |
| NVIDIA CUDA network repo | `nvidia-open` (latest), `nvidia-open-580` (one branch, up to 580), `nvidia-driver-pinning-<branch>`, `cuda-drivers` | newest drivers, NVIDIA's documented path, matching FM/DCGM/NCCL in the same repo | you sign the modules for Secure Boot (or Secure Boot is off) |
| `.run` installer | `NVIDIA-Linux-x86_64-<ver>.run` | no repo needed | invisible to apt; the next `apt upgrade` can half-overwrite it. Avoid on fleets. |

All the names above were checked against the repos **[apt, verified]**. For example, the Ubuntu archive had `nvidia-driver-570-server-open 570.211.01-0ubuntu1.24.04.1` and the CUDA repo had `nvidia-open 620.70.06-1ubuntu1`.

The `-server` packages in Ubuntu's archive are the datacenter-branch builds with a longer support lifecycle. Prefer them over the desktop `nvidia-driver-NNN` on servers.

NVIDIA ships several driver *branches* at once: production branches with long support and short-lived "new feature" branches. Pick a long-lived production branch for a fleet and move between branches deliberately. Check [NVIDIA's driver lifecycle documentation](https://docs.nvidia.com/datacenter/tesla/drivers/index.html) for which branches are long-term support right now. I have not verified the current LTS list.

### Open vs proprietary kernel modules

Since 2022 NVIDIA has published the source of its GPU kernel modules ([open-gpu-kernel-modules](https://github.com/NVIDIA/open-gpu-kernel-modules), dual MIT/GPLv2). The open modules are possible because Turing and later GPUs run most of the driver logic on an on-GPU microcontroller, the **GSP** (GPU System Processor). The kernel module is then mostly a thin shim, and NVIDIA could open-source it without exposing much.

The practical rules (as of driver 620):

- Turing (T4), Ampere (A100), Hopper (H100/H200) and Blackwell (B200/B300) can all use the open modules. Grace Hopper and Blackwell *require* them.
- Pre-Turing GPUs such as Volta (V100) need the proprietary modules, which are only available on older branches.
- NVIDIA's own install guide now documents only the open flavor. Since R560 it has been the default.

Check which one is loaded: **[untested]**

```bash
$ grep -o 'Open Kernel Module' /proc/driver/nvidia/version && echo open || echo proprietary
Open Kernel Module
open
$ modinfo nvidia | grep -E '^(version|license)'
version:        580.95.05
license:        Dual MIT/GPL          # proprietary builds say "NVIDIA"
```

### DKMS: why the driver gets rebuilt when the kernel changes

A kernel module is compiled against one specific kernel version's internal interfaces, which are not stable. When Ubuntu installs a new kernel, every out-of-tree module (a module not shipped inside the Linux source tree: NVIDIA's, and sometimes DOCA-OFED, the RDMA NIC stack from [section 5 of the main chapter](gpu-software-stack.md#the-software-inbox-rdma-core-vs-doca-ofed)) has to be rebuilt for it. **DKMS** (Dynamic Kernel Module Support) automates that. The driver package installs the module *source* under `/usr/src/nvidia-<ver>/`, and an apt hook recompiles it for every kernel that gets installed. The `nvidia-dkms-*-open` package is the DKMS piece. The precompiled alternative (`linux-modules-nvidia-*` in Ubuntu) ships modules already built for Canonical's kernels.

This only works if the kernel headers for the target kernel are installed, so put `linux-headers-$(uname -r)` (or the `linux-headers-generic` metapackage) in your base image. A missing header package is the most common reason a node comes back from a kernel update with no GPU. **[untested]**

```bash
$ dkms status
nvidia/580.95.05, 6.8.0-85-generic, x86_64: installed
nvidia/580.95.05, 6.8.0-87-generic, x86_64: installed      # one line per kernel it is built for

# after installing a kernel without headers, the module silently isn't there for it:
$ ls /lib/modules/6.8.0-88-generic/updates/dkms/ 2>&1
ls: cannot access '/lib/modules/6.8.0-88-generic/updates/dkms/': No such file or directory
$ sudo apt install linux-headers-6.8.0-88-generic && sudo dkms autoinstall -k 6.8.0-88-generic
```

**Secure Boot** is a feature of UEFI (the firmware interface that replaced the classic BIOS; people still say "BIOS" for both) that refuses to load unsigned kernel code. Canonical's precompiled modules are signed. DKMS-built modules are signed with a local key (a **MOK**, machine owner key), which someone must enroll at the console on the next boot (`sudo mokutil --import /var/lib/shim-signed/mok/MOK.der`). Fleets usually either use signed modules or turn Secure Boot off in the BIOS. If `modprobe nvidia` fails with `Key was rejected by service`, this is the cause.

### A clean install from the NVIDIA repo

Compute-only (headless), open modules, pinned to the 580 branch. These follow [NVIDIA's Ubuntu install guide](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/ubuntu.html). **[untested]**

Reading the block: `echo -e "..." | sudo tee FILE` writes a file that only root may write (`sudo echo > FILE` would not work, because the redirection is done by your unprivileged shell). Files in `/etc/modprobe.d/` are options for `modprobe`; `blacklist nouveau` stops the kernel loading **nouveau**, the community-written open-source NVIDIA driver, which would otherwise grab the GPU first. `update-initramfs -u` rebuilds the **initramfs**, the small boot-time filesystem that loads early drivers, so the blacklist also applies at boot. `dpkg -i` installs a downloaded `.deb` package file directly (`apt` is the higher-level tool that downloads and resolves dependencies, see [package management](../foundations/linux-and-os.md#package-management-apt-vs-conda-vs-pip-vs-uv)); `-y` answers yes to prompts and `-V` shows the versions it picks.

```bash
# 0. nothing else may own the GPU: blacklist nouveau (the reverse-engineered open driver)
echo -e "blacklist nouveau\noptions nouveau modeset=0" | sudo tee /etc/modprobe.d/blacklist-nouveau.conf
sudo update-initramfs -u

# 1. headers for DKMS, and the NVIDIA repo
sudo apt install -y linux-headers-$(uname -r)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt update

# 2. pin the branch BEFORE installing (NVIDIA's recommendation), then the compute-only driver
sudo apt install -y nvidia-driver-pinning-580
sudo apt install -y -V libnvidia-compute nvidia-dkms-open      # compute-only; 'nvidia-open' = full driver
sudo apt install -y nvidia-persistenced                        # if not pulled in already

# 3. reboot and verify
sudo reboot
nvidia-smi -L
```

Starting with branch 590, NVIDIA's Ubuntu packages dropped the branch number from their names (`nvidia-open` rather than `nvidia-open-590`). Choosing a branch or an exact version is now done only with the `nvidia-driver-pinning-*` packages. The repo has both `nvidia-driver-pinning-580` (branch) and `nvidia-driver-pinning-580.126.20` (exact version) **[apt, verified]**.

### Pinning: making `apt upgrade` safe

An unattended `apt upgrade` that moves a node from 580 to 620 while jobs are running is a classic outage. The new user-space libraries no longer match the loaded kernel module, and every new process fails with the NVML mismatch error until someone reboots. The pinning package is simply an apt preferences file (files in `/etc/apt/preferences.d/` tell apt which version of a package to prefer when several are available). This is the real content of `nvidia-driver-pinning-580`, extracted from the .deb **[apt, verified]**:

```
# /etc/apt/preferences.d/nvidia-driver-pin  (from nvidia-driver-pinning-580)
Package: src:nvidia-graphics-drivers-580:any src:nvidia-kmod-open:any
Pin: version 580*
Pin-Priority: 1000

Package: src:nvidia-modprobe src:nvidia-persistenced src:nvidia-settings src:nvidia-xconfig
Pin: version 580*
Pin-Priority: 1000

Package: src:cuda-drivers:any src:nvidia-open:any
Pin: version 580*
Pin-Priority: 1000

Package: src:libnvidia-nscq src:libnvsdm src:nvidia-fabricmanager src:nvidia-imex src:nvlink5 src:fabricmanager src:imex
Pin: version 580*
Pin-Priority: 1000
```

`Pin-Priority: 1000` means "prefer this version even if it means a downgrade". The pin covers every source package of the driver, including Fabric Manager and IMEX (a daemon for sharing GPU memory across nodes on rack-scale systems, [below](gpu-software-stack.md#fabric-manager-required-on-nvswitch-systems)), so they all stay on the same branch. That matters because before driver 620, Fabric Manager must match the driver exactly (see below).

Belt and braces for a fleet: also keep the driver out of unattended upgrades, and pin the kernel too, since a kernel change triggers a DKMS rebuild. **[untested]**

In the block below, `dpkg-query -W -f='${Package}\n'` prints the name of every installed package, one per line, and `apt-mark hold` freezes each one at its current version. `cat <<'EOF' ... EOF` is a **heredoc**: the lines between the two `EOF` markers become the command's standard input (the quotes around `'EOF'` stop the shell from expanding `$` inside), and `| sudo tee FILE` writes them to a root-owned file.

```bash
# hold exact versions (stricter than the branch pin)
sudo apt-mark hold $(dpkg-query -W -f='${Package}\n' | grep -E '^(nvidia|libnvidia|linux-(image|headers|modules))')
# keep unattended-upgrades away from them
cat <<'EOF' | sudo tee /etc/apt/apt.conf.d/51nvidia-no-unattended
Unattended-Upgrade::Package-Blacklist { "nvidia-"; "libnvidia-"; "linux-"; };
EOF
apt-mark showhold | head
```

### Upgrading without breaking the node

A driver upgrade means "nothing may hold `/dev/nvidia*` open, swap the modules, verify". The order matters. **[untested]**

Step 1 is a preview of the cluster schedulers taught later. A **scheduler** (Slurm or Kubernetes) is the service that decides which jobs run on which nodes. **Draining** a node tells it "let running jobs finish, but place nothing new here"; **resuming** puts the node back in service. `scontrol update nodename=... state=drain reason=...` is Slurm's admin command for that ([Slurm: draining and resuming](../orchestration/slurm/from-zero.md#day-to-day-draining-resuming-adding-nodes)), and `kubectl drain` is the Kubernetes equivalent ([Kubernetes from zero](../orchestration/kubernetes-from-zero.md)). `hostname -s` prints this machine's short name. `lsof /dev/nvidia*` lists processes that have the GPU device files open ([lsof](../foundations/linux-and-os.md#lsof-and-ss-open-files-and-network-ports)).

```bash
# 1. drain: tell the scheduler to stop placing jobs, wait for running ones
sudo scontrol update nodename=$(hostname -s) state=drain reason="driver 580->620"   # Slurm
# kubectl drain $(hostname) --ignore-daemonsets                                     # Kubernetes

# 2. stop everything that holds the GPU open
sudo systemctl stop nvidia-dcgm nvidia-fabricmanager nvidia-persistenced
sudo lsof /dev/nvidia* 2>/dev/null          # must print nothing; kill or stop what it shows

# 3. move the pin, upgrade, reboot (a reboot is the reliable way to get matching modules loaded)
sudo apt install -y nvidia-driver-pinning-620 && sudo apt purge -y nvidia-driver-pinning-580
sudo apt update && sudo apt install -y libnvidia-compute nvidia-dkms-open nvidia-fabricmanager
sudo reboot

# 3b. (no-reboot variant, when nothing else depends on the modules)
#     sudo modprobe -r nvidia_peermem nvidia_uvm nvidia_drm nvidia_modeset nvidia && sudo modprobe nvidia

# 4. verify, then run the acceptance script (below) before resuming
cat /proc/driver/nvidia/version && nvidia-smi -L
sudo python3 node_accept.py --gpus 8 --driver 620.70.06 && sudo scontrol update nodename=$(hostname -s) state=resume
```

Roll the upgrade one node at a time and keep a canary node on the new driver for a few days of real jobs before rolling out to the rest. Driver regressions do happen, and a fleet-wide rollback is far more expensive than a slow rollout.


## AMD / ROCm equivalents

The architecture is the same and the names differ. On AMD the kernel driver is `amdgpu`, which is in the mainline kernel. AMD also ships a newer copy as `amdgpu-dkms`. ROCm is the user-space stack, and RCCL is AMD's NCCL fork. It keeps NCCL's API and reads the same `NCCL_*` environment variables.

| Job | NVIDIA | AMD / ROCm |
|---|---|---|
| kernel driver | `nvidia.ko` (open) via DKMS | `amdgpu` (inbox or `amdgpu-dkms`) |
| install | `nvidia-open` / `libnvidia-compute` + pinning | `amdgpu-install --usecase=rocm` or the ROCm apt repo |
| inventory/health CLI | `nvidia-smi` | `amd-smi` (newer) / `rocm-smi` |
| GPU programming API | CUDA | HIP (CUDA-like C++ API; `hipcc` compiles it) |
| device info | `nvidia-smi -q`, `deviceQuery` | `rocminfo` (look for `gfx942` = MI300X) |
| intra-node fabric | NVLink/NVSwitch + Fabric Manager | Infinity Fabric / xGMI (no FM daemon) |
| fleet daemon, diagnostics | DCGM, `dcgmi diag` | RDC (ROCm Data Center), RVS (ROCm Validation Suite), AMD's field health check tools |
| collectives + tests | NCCL, nccl-tests | RCCL, [rccl-tests](https://github.com/ROCm/rccl-tests) (same CLI: `all_reduce_perf -b 8 -e 8G -f 2 -g 8`) |
| containers | `--gpus all` / CDI | `--device=/dev/kfd --device=/dev/dri --group-add video` |
| framework check | `torch.version.cuda` | `torch.version.hip` (and `torch.cuda.*` still works) |
| device selection | `CUDA_VISIBLE_DEVICES` | `HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES` ([details](#amd-hiding-devices)) |

The equivalent of the inventory step. **[untested]**

```bash
amd-smi version                       # amdgpu driver + ROCm version
amd-smi list                          # one line per GPU with BDF (PCI bus:device.function address) and UUID
rocminfo | grep -E '^\s+Name:\s+gfx' | sort | uniq -c    # "8  Name: gfx942"
amd-smi topology                      # xGMI link matrix, like nvidia-smi topo -m
python -c 'import torch; print(torch.__version__, torch.version.hip, torch.cuda.device_count())'
```

The same version rule applies: the kernel driver is on the host and ROCm is in the container. AMD publishes a compatibility matrix between ROCm releases and `amdgpu` driver versions on the [ROCm docs site](https://rocm.docs.amd.com/en/latest/). On AMD multi-GPU nodes, IOMMU settings are a frequent cause of hangs ([AMD: hangs and the IOMMU](#amd-hangs-or-slow-multi-gpu-runs-and-the-iommu)); `rocm-smi` and `rocminfo` output is read line by line in [the troubleshooting section](#amd-rocm-smi-and-amd-smi).


## A node acceptance script

### What it checks and why

One Python file, standard library only, so it runs before any venv exists. Each check prints a `PASS`, `FAIL` or `SKIP` line, and the exit code is 1 if anything failed. A scheduler can therefore use it directly as a health check. The checks follow the layers of this chapter:

| Check | Layer | FAIL when |
|---|---|---|
| kernel module | driver | `nvidia.ko` not loaded, or version != `--driver` |
| nvidia-smi / gpu count / gpu health | driver, hardware | NVML mismatch, wrong GPU count, persistence off, ECC off, uncorrected ECC, row remap pending/failed, PCIe width degraded, HW slowdown |
| xid errors | hardware | any `NVRM: Xid` in this boot's kernel log |
| nvswitch fabric | FM | any GPU's fabric state not `Completed/Success` |
| nvlink | NVLink | an inactive link, or GPUs with different link counts |
| cuda + torch | runtime | a matmul on any GPU disagrees with an fp64 CPU reference, or P2P is missing |
| rdma ports | RDMA | a port not `ACTIVE`, wrong rate, non-zero error counters |
| gpudirect rdma | RDMA ↔ GPU | neither `nvidia_peermem` nor the DMA-BUF prerequisites |
| dcgm diag | everything | any `Fail` at the chosen level |
| nccl all_reduce | NCCL + NVLink | `#wrong` != 0 or busbw below `--min-busbw` |
| container gpus | container toolkit | the container sees a different number of GPUs |

The parsers are pure functions, separate from the code that runs the commands. `--selftest` feeds them canned tool output (and a fake `/sys/class/infiniband` tree) and asserts on the results. That way the parsing is tested even on a laptop. Tool output formats do change between versions. When one does, add the new output as a fixture and rerun the selftest.

### The script

```python
#!/usr/bin/env python3
"""GPU node acceptance test. Prints one PASS/FAIL/SKIP line per check, exits 1 on any FAIL.

  sudo python3 node_accept.py --gpus 8 --driver 580.95.05 --ib-rate 400 \
      --nccl-tests ~/nccl-tests/build --min-busbw 350 --dcgm-level 2
  python3 node_accept.py --selftest      # runs the parsers against canned outputs, no GPU needed
"""
import argparse, glob, os, platform, re, shutil, subprocess, sys, tempfile

SYS = os.environ.get("ACCEPT_SYSFS", "/sys")  # point at a fake tree to test the RDMA checks
results = []


def sh(cmd, timeout=900):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, f"timeout after {timeout}s"
    return p.returncode, p.stdout + p.stderr


def report(name, ok, detail=""):
    status = {True: "PASS", False: "FAIL", None: "SKIP"}[ok]
    results.append(status)
    print(f"[{status}] {name:<24} {detail}", flush=True)


def read(path, default=""):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return default


# ---------------- pure parsers: everything below here is exercised by --selftest ----------------

GPU_FIELDS = ("index,pci.bus_id,persistence_mode,ecc.mode.current,ecc.errors.uncorrected.volatile.total,"
              "remapped_rows.pending,remapped_rows.failure,pcie.link.width.current,pcie.link.width.max,"
              "clocks_event_reasons.hw_slowdown")


def parse_csv(text, fields=GPU_FIELDS):
    keys = fields.split(",")
    return [dict(zip(keys, (c.strip() for c in line.split(","))))
            for line in text.strip().splitlines() if line.strip()]


def gpu_problems(g):
    p = []
    if g["persistence_mode"] != "Enabled": p.append("persistence off")
    if g["ecc.mode.current"] != "Enabled": p.append("ECC off")
    if g["ecc.errors.uncorrected.volatile.total"] not in ("0", "[N/A]"):
        p.append(f"uncorrected ECC={g['ecc.errors.uncorrected.volatile.total']}")
    if g["remapped_rows.pending"] == "Yes": p.append("row remap pending: reset GPU")
    if g["remapped_rows.failure"] == "Yes": p.append("row remap failed: RMA")
    if g["pcie.link.width.current"] != g["pcie.link.width.max"]:
        p.append(f"PCIe x{g['pcie.link.width.current']} of x{g['pcie.link.width.max']}")
    if g["clocks_event_reasons.hw_slowdown"] == "Active": p.append("HW slowdown")
    return p


def parse_kmod(proc_version):
    """/proc/driver/nvidia/version -> (version, is_open)."""
    m = re.search(r"Kernel Module(?: for \S+)?\s+(\d+\.\d+(?:\.\d+)?)", proc_version)
    return (m.group(1) if m else None), ("Open Kernel Module" in proc_version)


def parse_fabric(smi_q):
    """nvidia-smi -q -> list of (state, status), one per GPU that reports a fabric."""
    pairs = re.findall(r"Fabric\s*\n\s*State\s*:\s*(.+?)\s*\n\s*Status\s*:\s*(.+?)\s*\n", smi_q)
    return [p for p in pairs if p[0] != "N/A"]


def parse_nvlink(status_text):
    """nvidia-smi nvlink -s -> {gpu: (active_links, inactive_links)}."""
    out, gpu = {}, None
    for line in status_text.splitlines():
        if m := re.match(r"GPU (\d+):", line):
            gpu = int(m.group(1)); out[gpu] = [0, 0]
        elif gpu is not None and re.match(r"\s+Link \d+:", line):
            out[gpu][1 if "inactive" in line.lower() else 0] += 1
    return {g: tuple(v) for g, v in out.items()}


def parse_xids(kernel_log):
    """dmesg / journalctl -k text -> sorted set of (pci_addr, xid)."""
    return sorted(set(re.findall(r"NVRM: Xid \(PCI:([0-9a-fA-F:.]+)\): (\d+)", kernel_log)))


def parse_busbw(nccl_out):
    m = re.search(r"Avg bus bandwidth\s*:\s*([\d.]+)", nccl_out)
    wrong = re.search(r"Out of bounds values\s*:\s*(\d+)", nccl_out)
    return (float(m.group(1)) if m else None), (int(wrong.group(1)) if wrong else None)


IB_ERR_COUNTERS = ("symbol_error", "link_downed", "port_rcv_errors", "local_link_integrity_errors",
                   "excessive_buffer_overrun_errors")


def rdma_ports(sysfs=None):
    """Walk /sys/class/infiniband -> list of dicts, one per (device, port)."""
    sysfs = sysfs or SYS
    ports = []
    for pdir in sorted(glob.glob(f"{sysfs}/class/infiniband/*/ports/*")):
        dev, port = pdir.split("/")[-3], pdir.split("/")[-1]
        errs = {c: int(read(f"{pdir}/counters/{c}", "0") or 0) for c in IB_ERR_COUNTERS}
        ports.append(dict(name=f"{dev}/{port}", state=read(f"{pdir}/state"), rate=read(f"{pdir}/rate"),
                          link_layer=read(f"{pdir}/link_layer"), errors={k: v for k, v in errs.items() if v}))
    return ports


def port_problems(p, want_rate):
    probs = []
    if "ACTIVE" not in p["state"]: probs.append(f"state={p['state'] or '?'}")
    rate = float(p["rate"].split()[0]) if p["rate"] else 0.0
    if want_rate and rate != want_rate: probs.append(f"rate={p['rate']}")
    if p["errors"]: probs.append(f"errors={p['errors']}")
    return probs


# ---------------- checks: each one calls report() exactly once ----------------

def check_driver(args):
    kmod = read("/proc/driver/nvidia/version")
    if not kmod:
        return report("kernel module", False, "nvidia.ko not loaded (/proc/driver/nvidia/version missing)")
    ver, is_open = parse_kmod(kmod)
    ok = args.driver is None or ver == args.driver
    report("kernel module", ok, f"{ver} ({'open' if is_open else 'proprietary'})"
           + ("" if ok else f", expected {args.driver}"))


def check_gpus(args):
    if not shutil.which("nvidia-smi"):
        return report("nvidia-smi", False, "not on PATH")
    rc, out = sh(f"nvidia-smi --query-gpu={GPU_FIELDS} --format=csv,noheader,nounits")
    if rc != 0:  # 'Driver/library version mismatch' lands here: user-space libs != loaded nvidia.ko
        return report("nvidia-smi", False, out.strip().splitlines()[-1] if out.strip() else f"rc={rc}")
    gpus = parse_csv(out)
    report("gpu count", len(gpus) == args.gpus, f"{len(gpus)} found, expected {args.gpus}")
    bad = {g["pci.bus_id"]: gpu_problems(g) for g in gpus if gpu_problems(g)}
    report("gpu health", not bad, "; ".join(f"{k}: {', '.join(v)}" for k, v in bad.items()) or
           "persistence, ECC, row remap, PCIe width, HW slowdown ok")


def check_xids(args):
    rc, out = sh("journalctl -k -b --no-pager 2>/dev/null || dmesg 2>/dev/null")
    if not out.strip():
        return report("xid errors", None, "no kernel log access (run as root)")
    xids = parse_xids(out)
    report("xid errors", not xids, ", ".join(f"Xid {x} on {pci}" for pci, x in xids) or "none since boot")


def check_fabric(args):
    rc, out = sh("nvidia-smi -q")
    pairs = parse_fabric(out) if rc == 0 else []
    if not pairs:
        return report("nvswitch fabric", None, "no GPU reports a fabric state (no NVSwitch or pre-Hopper)")
    bad = [p for p in pairs if p != ("Completed", "Success")]
    _, svc = sh("systemctl is-active nvidia-fabricmanager")
    report("nvswitch fabric", not bad, f"{len(pairs)} GPUs Completed/Success, fabricmanager {svc.strip()}"
           if not bad else f"{bad[:2]}... fabricmanager={svc.strip()}")


def check_nvlink(args):
    rc, out = sh("nvidia-smi nvlink -s")
    links = parse_nvlink(out) if rc == 0 else {}
    if not any(a for a, _ in links.values()):
        return report("nvlink", None, "no active NVLink links reported")
    counts = {a for a, _ in links.values()}
    inactive = {g: i for g, (_, i) in links.items() if i}
    report("nvlink", len(counts) == 1 and not inactive,
           f"active links per GPU: {sorted(counts)}" + (f", inactive: {inactive}" if inactive else ""))


def check_cuda(args):
    code = r"""
import torch
n = torch.cuda.device_count()
assert n > 0, "torch sees no GPUs"
for d in range(n):
    a = torch.randn(2048, 2048, device=f"cuda:{d}")            # (N, N)
    ref = (a.double().cpu() @ a.double().cpu())                # (N, N) fp64 on CPU
    err = ((a @ a).double().cpu() - ref).abs().max().item() / ref.abs().max().item()
    assert err < 1e-3, f"cuda:{d} matmul rel err {err:.2e}"
    torch.cuda.synchronize(d)
p2p = all(torch.cuda.can_device_access_peer(i, j) for i in range(n) for j in range(n) if i != j)
print(f"torch {torch.__version__} cuda {torch.version.cuda}, {n} GPUs matmul ok, p2p={'all' if p2p else 'MISSING'}")
"""
    try:
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
        rc, out = p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        rc, out = 124, "timeout after 300s"
    if "No module named 'torch'" in out:
        return report("cuda + torch", None, "torch not installed in this python")
    last = out.strip().splitlines()[-1] if out.strip() else f"rc={rc}"
    report("cuda + torch", rc == 0 and "p2p=all" in last, last)


def check_rdma(args):
    ports = [p for p in rdma_ports() if not args.hcas or p["name"].split("/")[0] in args.hcas]
    if not ports:
        return report("rdma ports", None if args.ib_rate is None else False, "no RDMA devices in /sys/class/infiniband")
    bad = {p["name"]: port_problems(p, args.ib_rate) for p in ports if port_problems(p, args.ib_rate)}
    report("rdma ports", not bad, "; ".join(f"{k}: {', '.join(v)}" for k, v in bad.items()) or
           f"{len(ports)} ports ACTIVE, {ports[0]['rate']}, {ports[0]['link_layer']}, no error counters")


def check_gdr(args):
    if not glob.glob(f"{SYS}/class/infiniband/*"):
        return report("gpudirect rdma", None, "no RDMA devices")
    _, lsmod = sh("lsmod")
    if re.search(r"^nvidia_peermem\s", lsmod, re.M):
        return report("gpudirect rdma", True, "nvidia_peermem loaded")
    _, is_open = parse_kmod(read("/proc/driver/nvidia/version"))
    kernel = tuple(int(x) for x in re.findall(r"\d+", platform.release())[:2])
    ok = is_open and kernel >= (5, 12)
    report("gpudirect rdma", ok, "dma-buf path possible (open kmod, kernel >= 5.12): confirm GDRDMA in NCCL log"
           if ok else "neither nvidia_peermem nor dma-buf (open kmod + kernel >= 5.12) available")


def check_dcgm(args):
    if args.dcgm_level == 0 or not shutil.which("dcgmi"):
        return report("dcgm diag", None, "disabled or dcgmi not installed")
    rc, out = sh(f"dcgmi diag -r {args.dcgm_level}", timeout=3600)
    fails = [l.strip() for l in out.splitlines() if re.search(r"\bFail\b", l)]
    report("dcgm diag", rc == 0 and not fails, f"-r {args.dcgm_level}: " + ("; ".join(fails[:3]) or "all Pass"))


def check_nccl(args):
    binary = os.path.join(args.nccl_tests or "", "all_reduce_perf")
    if not args.nccl_tests or not os.access(binary, os.X_OK):
        return report("nccl all_reduce", None, "pass --nccl-tests <dir with all_reduce_perf>")
    rc, out = sh(f"{binary} -b 4G -e 4G -g {args.gpus} -w 5 -n 20")
    busbw, wrong = parse_busbw(out)
    ok = rc == 0 and wrong == 0 and busbw is not None and busbw >= args.min_busbw
    report("nccl all_reduce", ok, f"busbw {busbw} GB/s (min {args.min_busbw}), wrong={wrong}")


def check_container(args):
    if not args.container:
        return report("container gpus", None, "pass --container <image> to test the container runtime")
    rc, out = sh(f"docker run --rm --gpus all {args.container} nvidia-smi -L", timeout=600)
    n = len(re.findall(r"^GPU \d+:", out, re.M))
    report("container gpus", rc == 0 and n == args.gpus, f"{n} GPUs visible inside {args.container}")


CHECKS = [check_driver, check_gpus, check_xids, check_fabric, check_nvlink, check_cuda,
          check_rdma, check_gdr, check_dcgm, check_nccl, check_container]


# ---------------- selftest: canned outputs from the tools, asserts on the parsers ----------------

def selftest():
    csv = ("0, 00000000:18:00.0, Enabled, Enabled, 0, No, No, 16, 16, Not Active\n"
           "1, 00000000:2A:00.0, Disabled, Enabled, 2, Yes, No, 8, 16, Active\n")
    g = parse_csv(csv)
    assert gpu_problems(g[0]) == []
    assert gpu_problems(g[1]) == ["persistence off", "uncorrected ECC=2", "row remap pending: reset GPU",
                                  "PCIe x8 of x16", "HW slowdown"], gpu_problems(g[1])
    assert parse_kmod("NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  580.95.05  Release Build") \
        == ("580.95.05", True)
    assert parse_kmod("NVRM version: NVIDIA UNIX x86_64 Kernel Module  550.54.15  Tue Mar  5") == ("550.54.15", False)
    q = ("GPU 00000000:18:00.0\n    Fabric\n        State                     : Completed\n"
         "        Status                    : Success\n        CliqueId : 0\n"
         "GPU 00000000:2A:00.0\n    Fabric\n        State                     : In Progress\n"
         "        Status                    : N/A\n")
    assert parse_fabric(q) == [("Completed", "Success"), ("In Progress", "N/A")]
    nvl = ("GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-a)\n\t Link 0: 26.562 GB/s\n\t Link 1: 26.562 GB/s\n"
           "GPU 1: NVIDIA H100 80GB HBM3 (UUID: GPU-b)\n\t Link 0: 26.562 GB/s\n\t Link 1: <inactive>\n")
    assert parse_nvlink(nvl) == {0: (2, 0), 1: (1, 1)}
    log = ("[  812.1] NVRM: Xid (PCI:0000:2a:00): 79, pid=1234, GPU has fallen off the bus.\n"
           "[  812.2] NVRM: Xid (PCI:0000:2a:00): 79, pid=1234, GPU has fallen off the bus.\n")
    assert parse_xids(log) == [("0000:2a:00", "79")]
    assert parse_busbw("#  Out of bounds values : 0 OK\n# Avg bus bandwidth    : 361.48 \n") == (361.48, 0)
    with tempfile.TemporaryDirectory() as root:  # fake /sys with one healthy and one sick port
        for dev, state, rate, sym in [("mlx5_0", "4: ACTIVE", "400 Gb/sec (4X NDR)", "0"),
                                      ("mlx5_1", "1: DOWN", "10 Gb/sec (4X SDR)", "17")]:
            d = f"{root}/class/infiniband/{dev}/ports/1"
            os.makedirs(f"{d}/counters")
            for name, val in [("state", state), ("rate", rate), ("link_layer", "InfiniBand"),
                              ("counters/symbol_error", sym), ("counters/link_downed", "0")]:
                with open(f"{d}/{name}", "w") as f:
                    f.write(val + "\n")
        ports = rdma_ports(root)
        assert port_problems(ports[0], 400) == []
        assert port_problems(ports[1], 400) == ["state=1: DOWN", "rate=10 Gb/sec (4X SDR)",
                                                "errors={'symbol_error': 17}"], port_problems(ports[1], 400)
    print("selftest: all parser assertions passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--driver", help="exact expected driver version, e.g. 580.95.05")
    ap.add_argument("--ib-rate", type=float, help="expected per-port rate in Gb/s, e.g. 400")
    ap.add_argument("--hcas", nargs="*", help="only check these RDMA devices, e.g. mlx5_0 mlx5_1")
    ap.add_argument("--dcgm-level", type=int, default=1, choices=[0, 1, 2, 3, 4])
    ap.add_argument("--nccl-tests", help="directory containing all_reduce_perf")
    # Note: busbw floor is a calibration knob; set it from a known-good node of the same SKU
    ap.add_argument("--min-busbw", type=float, default=0.0)
    ap.add_argument("--container", help="image to test GPU visibility inside, e.g. nvidia/cuda:13.0.2-base-ubuntu24.04")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    print(f"node acceptance on {platform.node()} ({platform.system()} {platform.release()})")
    for check in CHECKS:
        try:
            check(args)
        except Exception as e:  # a crashing check is a failing check, keep going
            report(check.__name__.removeprefix("check_"), False, f"check crashed: {e!r}")
    fails = results.count("FAIL")
    print(f"\n{'NODE FAIL' if fails else 'NODE PASS'}: {results.count('PASS')} pass, {fails} fail, "
          f"{results.count('SKIP')} skip")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
```

### Running it

The selftest and a real run on this Mac. There is no NVIDIA stack, so the two driver checks fail and everything else skips. The exit code is 1, which is what a scheduler must see on a non-GPU host. **[ran]**

```
$ python3 node_accept.py --selftest
selftest: all parser assertions passed

$ python3 node_accept.py --gpus 8; echo rc=$?
node acceptance on my-laptop.local (Darwin 24.6.0)
[FAIL] kernel module            nvidia.ko not loaded (/proc/driver/nvidia/version missing)
[FAIL] nvidia-smi               not on PATH
[SKIP] xid errors               no kernel log access (run as root)
[SKIP] nvswitch fabric          no GPU reports a fabric state (no NVSwitch or pre-Hopper)
[SKIP] nvlink                   no active NVLink links reported
[SKIP] cuda + torch             torch not installed in this python
[SKIP] rdma ports               no RDMA devices in /sys/class/infiniband
[SKIP] gpudirect rdma           no RDMA devices
[SKIP] dcgm diag                disabled or dcgmi not installed
[SKIP] nccl all_reduce          pass --nccl-tests <dir with all_reduce_perf>
[SKIP] container gpus           pass --container <image> to test the container runtime

NODE FAIL: 0 pass, 2 fail, 9 skip
rc=1
```

With a python that has torch (the Mac's MPS-only build), the CUDA check gets further and fails for the right reason **[ran]**:

```
[FAIL] cuda + torch             AssertionError: torch sees no GPUs
```

On a healthy 8x H100 InfiniBand node it should look like this. **[untested]** This output is a sketch, not a capture.

```
$ sudo python3 node_accept.py --gpus 8 --driver 580.95.05 --ib-rate 400 \
    --hcas mlx5_0 mlx5_1 mlx5_2 mlx5_3 mlx5_4 mlx5_5 mlx5_6 mlx5_7 \
    --nccl-tests /opt/nccl-tests/build --min-busbw 350 --dcgm-level 2 \
    --container nvidia/cuda:13.0.2-base-ubuntu24.04
node acceptance on gpu-017 (Linux 6.8.0-87-generic)
[PASS] kernel module            580.95.05 (open)
[PASS] gpu count                8 found, expected 8
[PASS] gpu health               persistence, ECC, row remap, PCIe width, HW slowdown ok
[PASS] xid errors               none since boot
[PASS] nvswitch fabric          8 GPUs Completed/Success, fabricmanager active
[PASS] nvlink                   active links per GPU: [18]
[PASS] cuda + torch             torch 2.15.0 cuda 13.2, 8 GPUs matmul ok, p2p=all
[PASS] rdma ports               8 ports ACTIVE, 400 Gb/sec (4X NDR), InfiniBand, no error counters
[PASS] gpudirect rdma           nvidia_peermem loaded
[PASS] dcgm diag                -r 2: all Pass
[PASS] nccl all_reduce          busbw ... GB/s (min 350.0), wrong=0
[PASS] container gpus           8 GPUs visible inside nvidia/cuda:13.0.2-base-ubuntu24.04

NODE PASS: 12 pass, 0 fail, 0 skip
```

`--min-busbw 350` is a placeholder. Set it to around 90% of the value you measure on a known-good node of the same SKU, driver and NCCL version.

### Wiring it into the scheduler

Preview: this uses Slurm and Kubernetes vocabulary taught in [Slurm from zero](../orchestration/slurm/from-zero.md) and [Kubernetes from zero](../orchestration/kubernetes-from-zero.md); skim it now and come back. Just enough to follow: `slurm.conf` is Slurm's cluster-wide config file, and its `HealthCheckProgram` setting names a script every node runs periodically. A Kubernetes **DaemonSet** runs one copy of a container on every node ([DaemonSet](../orchestration/kubernetes-from-zero.md#daemonset)); an **init container** runs to completion before a pod's main container starts; **labels** are key/value tags on a node, and a **taint** marks a node so that ordinary workloads are not scheduled on it ([taints and tolerations](../orchestration/kubernetes-from-zero.md#taints-and-tolerations)).

- **Slurm**: run a fast subset as the node health check (`HealthCheckProgram` in `slurm.conf`, typically every few minutes, which drains the node on a non-zero exit). Run `--dcgm-level 1` in the job prolog. Run the full version, with DCGM level 2 and nccl-tests, after reboots and repairs, before `scontrol update state=resume`. See [Slurm](../orchestration/slurm/from-zero.md).
- **Kubernetes**: run it as a DaemonSet or as an init container of a "node validator" pod that labels or taints the node on failure. The NVIDIA GPU Operator's own validator does a subset of this ([Kubernetes from zero](../orchestration/kubernetes-from-zero.md)).
- Log every run, with the node's GPU UUIDs, to persistent storage. When a "new" cloud node fails, that history tells you whether you got [the same broken GPUs again](#detecting-that-you-got-the-same-broken-node-again).

Not covered here: multi-node acceptance (bisecting a slow all-reduce across a whole cluster), and burn-in under thermal load (hours of `dcgmi diag -r 4` or a training job). Both build on this script. [Profiling and optimization](../training/profiling-and-optimization.md) covers what to do once the node passes and the job is still slow.

## Troubleshooting GPUs: errors, diagnostics and health checks

[Section 3 of the main chapter](gpu-software-stack.md#3-driver-level-services-and-gpu-health) and the acceptance script above set up the health checks and the acceptance gate. This section is what you need when one of them fails, or when a job crashes and you suspect a GPU. It covers how to read the errors the driver reports, how to run deeper diagnostics, how to tell a GPU that needs a reset from one that has to go back to the vendor, and the AMD equivalents.

No hardware is perfect. Manufacturing defects and wear (especially from long exposure to heat) mean that GPUs hit hardware faults. Many of them are corrected by the hardware without you ever noticing, and if the application keeps running there is usually nothing to do. When an application crashes because of a hardware fault, you need to know why and what to do about it. With a handful of GPUs you may never see one. At hundreds or thousands of GPUs you will see them every week.

All GPU-side shell sessions in this section are **[untested]** here (no GPU hardware); their output is real output quoted from production nodes or vendor documentation, as noted. The Python and shell examples marked **[ran]** were run on the Mac.

### Glossary

| Term | Meaning |
|---|---|
| DBE | Double Bit ECC Error: more than one bit flipped in a word, which ECC can detect but not fix ("uncorrectable") |
| DCGM | (NVIDIA) Data Center GPU Manager, see [DCGM](gpu-software-stack.md#dcgm-the-health-daemon-and-diagnostics) |
| ECC | Error-Correcting Code, see [ECC, row remapping, clocks and GPU reset](gpu-software-stack.md#ecc-row-remapping-clocks-and-gpu-reset) |
| FB | Frame Buffer: NVIDIA's name for the GPU's own memory, i.e. its HBM |
| HBM | High Bandwidth Memory, the GPU's main DRAM |
| RMA | Return Merchandise Authorization: sending a faulty part back to the vendor for a replacement |
| SBE | Single Bit ECC Error: one flipped bit, which ECC fixes ("correctable") |
| SDC | Silent Data Corruption: a wrong result that no error counter caught |
| VBIOS | the GPU's firmware, stored on the card itself ([the layers](gpu-software-stack.md#the-layers-bottom-to-top)) |
| Xid | a numbered error report that the NVIDIA kernel driver writes to the kernel log |

For diagnosing crashes from the software side (stack traces, hangs, `CUDA_LAUNCH_BLOCKING`, compute-sanitizer), see [Debugging PyTorch](../development/debugging-pytorch.md); for multi-GPU hangs see [Debugging distributed training](../development/debugging-distributed.md).

### Xid errors: reading, finding and looking them up

In the system logs you will occasionally see lines like this one:

```
NVRM: Xid (PCI:0000:10:1c): 63, pid=1896, Row Remapper: New row marked for remapping, reset gpu to activate.
```

How to read it, left to right: `NVRM` is the NVIDIA kernel driver; `PCI:0000:10:1c` is the PCI bus id of the GPU that reported it (the same id `nvidia-smi` shows in its `Bus-Id` column); `63` is the Xid number you look up; `pid` is the process that was using the GPU; the rest is the driver's description.

To get these lines out of the kernel log (the kernel log and `dmesg` are covered in [Logs: journalctl and dmesg](../foundations/linux-and-os.md#logs-journalctl-and-dmesg)), one of these should work:

```bash
sudo grep Xid /var/log/syslog
sudo dmesg -T | grep Xid
```

As long as training doesn't crash, these errors usually mean the hardware corrected the problem itself.

Where to look an Xid up:

- NVIDIA's [full list of Xid errors](https://docs.nvidia.com/deploy/xid-errors/latest/index.html) gives each number's meaning.
- NVIDIA's [Xid Catalog](https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html) goes further for Ampere and newer GPUs. For each Xid it lists which GPUs can log it, what triggers it, and what to do: an immediate action, such as `RESTART_APP` or `CONTACT_SUPPORT`, and a follow-up action, such as `CHECK_APP/CUDA`. It can be downloaded as a spreadsheet, which helps when you want to map the Xids in your logs to actions automatically. For Xid 13 and 31, which it says are typically application bugs, its first recommendation is Compute Sanitizer's `memcheck` (how to run it on a PyTorch program is in [Debugging PyTorch](../development/debugging-pytorch.md)).
- Many Xid lines carry far more fields than the one above, and they are cryptic. [Xid 31 MMU Faults: What Causes Them and How to Fix Production GPU Crashes](https://www.abhik.ai/articles/gpu-xid31-mmu-faults) is a good worked example of decoding one. It takes an Xid 31 line from `dmesg` and decodes every field: the faulting engine and hardware unit, the virtual address, `FAULT_PDE` vs `FAULT_PTE` (whether the GPU's page-table walk failed at a page directory entry or at a page table entry; page tables are in [Pages and address translation](../foundations/computer-architecture.md#pages-and-address-translation)), and the access type. It follows them to the cause, a use-after-free between the PyTorch, CuPy (a NumPy-like GPU array library) and NVDEC (the GPU's hardware video decoder) memory allocators. It also shows that not every Xid is a hardware problem: this one was caused by software.
- NVIDIA's [GPU debug guidelines](https://docs.nvidia.com/deploy/gpu-debug-guidelines/index.html) go deeper still. They include a triage chart for deciding when to RMA a GPU and more on Xid 63-like errors.

The Xids you will meet most in this section:

| Xid | Meaning | Typical action |
|---|---|---|
| 13, 31 | graphics engine exception / GPU memory page fault (MMU fault) | usually an application bug: run compute-sanitizer |
| 48 | DBE (uncorrectable ECC error); the application crashes | reset the GPU; the page is blacklisted |
| 63 | ECC page retirement or row remapping recording event | reset when convenient so the remap takes effect |
| 64 | ECC page retirement or row remapping recorder failure | if it recurs after a reset: RMA |
| 94 | contained uncorrectable ECC error | restart the affected application, reset later |

A log scraper only needs the bus id and the number. A minimal parser for the line format above, with its own check: **[ran]**

```python
# xid_triage.py - parse NVRM Xid lines and the row-remapper CSV, decide reset vs RMA.
import re

XID = re.compile(r"NVRM: Xid \(PCI:(?P<bus>[0-9a-fA-F:.]+)\): (?P<xid>\d+), (?:pid=(?P<pid>\d+|'<unknown>'), )?(?P<msg>.*)")

def parse_xid(line):
    m = XID.search(line)
    return None if m is None else (m["bus"], int(m["xid"]), m["msg"])

def remap_verdict(row):
    # row: "gpu_name, bus_id, failure, pending, correctable, uncorrectable"
    name, bus, failure, pending, corr, uncorr = [f.strip() for f in row.split(",")]
    if failure == "Yes":
        return bus, "RMA: remapping failed"
    if pending == "Yes":
        return bus, "reset GPU when convenient (remap pending, a page may be offlined)"
    return bus, f"ok ({corr} correctable, {uncorr} uncorrectable rows already remapped)"

log = [
    "[Wed Jun  7 19:32:16 2023] NVRM: Xid (PCI:0000:10:1c): 63, pid=1896, Row Remapper: New row marked for remapping, reset gpu to activate.",
    "[Wed Jun  7 19:40:02 2023] nvidia-peermem nv_get_p2p_free_callback:127 ERROR detected invalid context",
]
for line in log:
    print(parse_xid(line))

csv = """NVIDIA H200, 00000000:59:00.0, No, No, 0, 0
NVIDIA H200, 00000000:5A:00.0, No, Yes, 1, 0
NVIDIA H200, 00000000:A5:00.0, Yes, No, 0, 0"""
for row in csv.splitlines():
    print(remap_verdict(row))

assert parse_xid(log[0]) == ("0000:10:1c", 63, "Row Remapper: New row marked for remapping, reset gpu to activate.")
assert parse_xid(log[1]) is None
```

```
('0000:10:1c', 63, 'Row Remapper: New row marked for remapping, reset gpu to activate.')
None
('00000000:59:00.0', 'ok (0 correctable, 0 uncorrectable rows already remapped)')
('00000000:5A:00.0', 'reset GPU when convenient (remap pending, a page may be offlined)')
('00000000:A5:00.0', 'RMA: remapping failed')
```

The first line is the Xid split into its three useful fields; the second log line isn't an Xid, so it returns `None`. The remap verdicts are explained in [Row-remapping health](#checking-row-remapping-health-before-a-job-dies) below; the CSV rows are the format of that section's query.

### ECC errors and row remapping in `nvidia-smi -q`

After an Xid, `nvidia-smi -q` shows the error counters. In its output, `SRAM` counts errors in the on-chip memories (register files, caches, shared memory) and `DRAM` errors in HBM. For the Xid 63 above you'd see something like this (a real A100 node):

```
Timestamp                                 : Wed Jun  7 19:32:16 2023
Driver Version                            : 510.73.08
CUDA Version                              : 11.6

Attached GPUs                             : 8
GPU 00000000:10:1C.0
    Product Name                          : NVIDIA A100-SXM4-80GB
    [...]
    ECC Errors
        Volatile
            SRAM Correctable              : 0
            SRAM Uncorrectable            : 0
            DRAM Correctable              : 177
            DRAM Uncorrectable            : 0
        Aggregate
            SRAM Correctable              : 0
            SRAM Uncorrectable            : 0
            DRAM Correctable              : 177
            DRAM Uncorrectable            : 0
    Retired Pages
        Single Bit ECC                    : N/A
        Double Bit ECC                    : N/A
        Pending Page Blacklist            : N/A
    Remapped Rows
        Correctable Error                 : 1
        Uncorrectable Error               : 0
        Pending                           : Yes
        Remapping Failure Occurred        : No
        Bank Remap Availability Histogram
            Max                           : 639 bank(s)
            High                          : 1 bank(s)
            Partial                       : 0 bank(s)
            Low                           : 0 bank(s)
            None                          : 0 bank(s)
[...]
```

How to read it:

- **Volatile vs Aggregate.** `Volatile` counts errors since the last reboot or GPU reset (driver reload). `Aggregate` counts them over the GPU's whole life.
- **Correctable vs Uncorrectable.** A correctable error is an SBE: the memory was faulty but ECC recovered the right value, and the application is unaffected. An uncorrectable error is a DBE: more than one bit was wrong, and the application using it crashes. The memory page with the uncorrectable error is blacklisted and stays inaccessible until the GPU is reset.
- **Xid 63** here means "ECC page retirement or row remapping recording event". It has three possible causes: hardware error, driver error, or frame buffer (FB) corruption. One memory row is malfunctioning, and at the next reboot or GPU reset one of the spare rows (640 on the A100) replaces it. `Pending: Yes` says that replacement is waiting for the reset.
- **Bank Remap Availability Histogram.** This sorts the memory banks by how many of their spare rows are still unused. `Max` means none used yet; `High`, `Partial` and `Low` mean progressively more used; `None` means the bank has no spare rows left. Here 639 banks are untouched and one has started using its spares, which is the remap just recorded.
- **Retired Pages** is the older mechanism, used before Ampere (on the A100 above it reads `N/A`). The driver retires a whole memory page after 1 DBE or 2 SBEs at the same address ([dynamic page retirement](https://docs.nvidia.com/deploy/dynamic-page-retirement/index.html)). If pages are scheduled for retirement you'll see:

```
    Retired pages
        Single Bit ECC             : 2
        Double Bit ECC             : 0
        Pending Page Blacklist    : Yes
```

Each retired page reduces the memory available to applications, but retirement is capped at 4 MiB in total, so it barely dents an 80 GB GPU.

The debug guidelines linked above suggest, for example:

> If associated with XID 94, the application that encountered the error needs to be restarted. All other applications on the system can keep running as is until there is a convenient time to reboot for row remapping to activate.

If after a reboot the same condition occurs at the same memory address, remapping has failed and the driver emits **Xid 64**. If that keeps happening after reboots, the GPU needs to be RMA'ed. In the `Remapped Rows` block it looks like this. This is [Crusoe Cloud's published signature](https://docs.crusoecloud.com/resources/troubleshooting) for a GPU that qualifies for replacement:

```
Remapped Rows
        Correctable Error                 : 0
        Uncorrectable Error               : 0
        Pending                           : No
        Remapping Failure Occurred        : Yes
        Bank Remap Availability Histogram
            Max                           : 639 bank(s)
            High                          : 0 bank(s)
            Partial                       : 0 bank(s)
            Low                           : 0 bank(s)
            None                          : 1 bank(s)
```

Not a single row was remapped successfully, yet a failure was recorded, and one bank sits at `None`: it has no spare rows left to remap into. That combination is what distinguishes a GPU that needs replacing from one that just needs a reset. To print only this block instead of the whole report, use `nvidia-smi -q -d ROW_REMAPPER`.

At other times Xid 63 or 64 comes with a crash and further Xids. Most of the time that means the error was uncorrectable, a DBE, which is reported as Xid 48.

To reset a GPU, reboot the machine or run:

```bash
nvidia-smi -r -i gpu_id
```

where `gpu_id` is the GPU's index, e.g. `0` for the first one. Without `-i`, all GPUs are reset. As [section 3](gpu-software-stack.md#ecc-row-remapping-clocks-and-gpu-reset) says, a reset needs root and fails while any process still holds the GPU.

### "CUDA error: uncorrectable ECC error encountered"

If a job dies with:

```
CUDA error: uncorrectable ECC error encountered
```

the `ECC Errors` entries of `nvidia-smi -q` tell you which GPU it was. If you just need a quick yes/no to decide whether to recycle a node:

```bash
$ nvidia-smi -q | grep -i correctable | grep -v 0
            SRAM Uncorrectable            : 1
            SRAM Uncorrectable            : 5
```

On a good node this prints nothing, because every counter is 0. Here one GPU is broken. There are two lines because the full record was:

```
    ECC Errors
        Volatile
            SRAM Correctable              : 0
            SRAM Uncorrectable            : 1
            DRAM Correctable              : 0
            DRAM Uncorrectable            : 0
        Aggregate
            SRAM Correctable              : 0
            SRAM Uncorrectable            : 5
            DRAM Correctable              : 0
            DRAM Uncorrectable            : 0
```

The first line is the `Volatile` counter (1 since the last driver reload) and the second the `Aggregate` one (5 over the GPU's life), so this is not the first time this GPU has had the problem. It typically corresponds to Xid 94, most likely without Xid 48.

One trap in that one-liner: `grep -v 0` drops every line containing the digit 0 *anywhere*, so a count of 10 or 105 is hidden too. Drop only the zero counts with `grep -v ': 0$'`. On a canned sample: **[ran]**

```bash
$ printf '  SRAM Correctable : 0\n  SRAM Uncorrectable : 1\n  DRAM Correctable : 10\n  DRAM Uncorrectable : 0\n' > ecc.txt
$ grep -i correctable ecc.txt | grep -v 0
  SRAM Uncorrectable : 1
$ grep -i correctable ecc.txt | grep -v ': 0$'
  SRAM Uncorrectable : 1
  DRAM Correctable : 10
```

The fix is the same as before: reset the GPU (`nvidia-smi -r -i gpu_id`) or reboot the machine. If the Aggregate SRAM Uncorrectable count is above 4, that is usually reason enough to RMA the GPU.

### Running diagnostics with `dcgmi diag`

[DCGM](gpu-software-stack.md#dcgm-the-health-daemon-and-diagnostics) was installed in section 3 of the main chapter. Its `dcgmi diag` is the quickest way to find a bad GPU when you suspect one on a node. Reference: the [DCGM user guide](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/index.html); downloads and quickstart are on [GitHub](https://github.com/NVIDIA/DCGM#quickstart).

The levels, from fast to thorough:

| Level | Time on an 8-GPU node | Use |
|---|---|---|
| `-r 1` | seconds | deployment and software checks; job prolog |
| `-r 2` | a few minutes | short stress, PCIe, memory; in practice it already catches most faulty GPUs |
| `-r 3` | about 10 minutes | in-depth; when you suspect a GPU and `-r 2` passed |
| `-r 4` | more than an hour | the most thorough; the only level that can catch some silent data corruption |

A Slurm script that runs level 3 on one whole node (Slurm is taught in [Slurm from zero](../orchestration/slurm/from-zero.md); the minimum to read it is below):

```bash
$ cat dcgmi-1n.slurm
#!/bin/bash
#SBATCH --job-name=dcgmi-1n
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --output=%x-%j.out

set -x -e
echo "START TIME: $(date)"
srun --cpus-per-task=$SLURM_CPUS_PER_TASK --output=%x-%j-%N.out dcgmi diag -r 3
echo "END TIME: $(date)"
```

How to read it: `--gres=gpu:8` asks for all 8 GPUs; `--exclusive` keeps other jobs off the node; in the output file names, `%x`, `%j` and `%N` expand to the job name, the job id and the node name; `set -x -e` echoes each command and stops at the first failure; and `srun` launches `dcgmi diag` inside the allocation. `sbatch` submits the script, and `--nodelist` pins it to the named node, so to test specific nodes:

```bash
sbatch --nodelist=node-115 dcgmi-1n.slurm
sbatch --nodelist=node-151 dcgmi-1n.slurm
sbatch --nodelist=node-170 dcgmi-1n.slurm
```

If the node is drained or down and Slurm won't run jobs on it, `ssh` into it and run the command directly:

```bash
dcgmi diag -r 3
```

If that finds nothing but the application still fails, run level 4 (more than an hour):

```bash
dcgmi diag -r 4
```

On a node that keeps throwing Xid 64, the report is likely to include:

```
+---------------------------+------------------------------------------------+
| Diagnostic                | Result                                         |
+===========================+================================================+
|-----  Deployment  --------+------------------------------------------------|
| Error                     | GPU 3 has uncorrectable memory errors and row  |
|                           |  remappings are pending                        |
```

so you know which GPU to RMA if remapping fails.

In practice, `-r 2` already detects most faulty GPUs and takes only a few minutes. Its output on a faulty node:

```
| GPU Memory                | Pass - GPUs: 1, 2, 3, 4, 5, 6, 7               |
|                           | Fail - GPU: 0                                  |
| Warning                   | GPU 0 Thermal violations totaling 13.3 second  |
|                           | s started at 9.7 seconds into the test for GP  |
|                           | U 0 Verify that the cooling on this machine i  |
|                           | s functional, including external, thermal mat  |
|                           | erial interface, fans, and any other componen  |
|                           | ts.
```

Read it per row: `GPU Memory` passed on GPUs 1-7 and failed on GPU 0, and the warning says GPU 0 also spent 13.3 s throttled for heat, so check its cooling as well as its memory.

**Silent data corruption.** SDC can apparently only be detected with `dcgmi diag -r 4`, and even then some is missed. It happens occasionally, and you may not know that a GPU is getting a `matmul` wrong now and then. Stas Bekman describes a case that was most likely this: training showed weird glitches, many days with the NVIDIA team failed to diagnose it, and the problem eventually disappeared, probably because the bad GPUs were replaced after other reported failures. The [acceptance script](#a-node-acceptance-script)'s matmul-vs-fp64 check is a cheap partial guard.

The fast levels fit in a Slurm **epilog** (a script the cluster admins configure to run on each node right after a job ends; see [Slurm from zero](../orchestration/slurm/from-zero.md#what-happens-from-sbatch-to-processes-on-nodes)). That checks the node is fit for the next job, instead of the next user discovering it when their job crashes.

When filing an RMA, the vendor will ask for the output of `nvidia-bug-report.sh`. It ships with the driver, needs root, and collects the driver state and logs into `nvidia-bug-report.log.gz` in the current directory. Keep the diagnostics logs too, one per node:

```bash
dcgmi diag -r 2 | tee -a dcgmi-r2-`hostname`.txt
dcgmi diag -r 3 | tee -a dcgmi-r3-`hostname`.txt
dcgmi diag -r 4 | tee -a dcgmi-r4-`hostname`.txt
```

`` `hostname` `` is the older backtick form of `$(hostname)`, so each node's log gets its own name, and `tee -a` shows the output while appending it to the file.

### VBIOS version

The VBIOS version can matter when researching an issue (vendors fix bugs in firmware, and a mixed fleet behaves inconsistently). Query it together with the name and bus id:

```bash
$ nvidia-smi --query-gpu=gpu_name,gpu_bus_id,vbios_version --format=csv
name, pci.bus_id, vbios_version
NVIDIA H100 80GB HBM3, 00000000:04:00.0, 96.00.89.00.01
[...]
NVIDIA H100 80GB HBM3, 00000000:8B:00.0, 96.00.89.00.01
```

Every GPU on a node, and on every node of a fleet, should show the same version. To list dozens of other queryable fields:

```bash
nvidia-smi --help-query-gpu
```

### Is the GPU's PCIe generation actually used?

The kernel reports at boot when a device's PCIe link is slower than the device could do:

```bash
$ sudo dmesg | grep -i 'limited by'
[   10.735323] pci 0000:04:00.0: 252.048 Gb/s available PCIe bandwidth, limited by 16.0 GT/s PCIe x16 link at 0000:01:00.0 (capable of 504.112 Gb/s with 32.0 GT/s PCIe x16 link)
[...]
[   13.301989] pci 0000:8b:00.0: 252.048 Gb/s available PCIe bandwidth, limited by 16.0 GT/s PCIe x16 link at 0000:87:00.0 (capable of 504.112 Gb/s with 32.0 GT/s PCIe x16 link)
```

How to read it: each line names a device by its PCI bus id, the bandwidth it actually gets, the link that limits it (`16.0 GT/s PCIe x16` is a gen4 link of 16 lanes, `32.0 GT/s` would be gen5, and `at 0000:01:00.0` is the upstream device that link belongs to) and, in parentheses, what the device itself could do. PCIe generations and lanes are covered in [PCIe](../foundations/computer-architecture.md#pcie), and the spec tables are in [PCIe](../network/interconnects.md#pcie).

The PCIe 5 x16 spec is about 504 Gbps, so on this node only half of the possible bandwidth is usable, because the PCIe switch is gen4. Since the GPUs most likely talk to each other over [NVLink](../network/interconnects.md#nvlink), this doesn't matter for GPU-to-GPU traffic. It does slow every transfer between GPU and host, because a path is only as fast as its slowest link. (Remember the trap from [section 3](gpu-software-stack.md#ecc-row-remapping-clocks-and-gpu-reset): `nvidia-smi` shows the *current* generation, which drops at idle. The boot message above is about what the link *can* negotiate.)

### NVLink error counters

If you suspect an NVLink is malfunctioning, check its error counters:

```bash
$ nvidia-smi nvlink -e
GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-abcdefab-cdef-abdc-abcd-abababababab)
         Link 0: Replay Errors: 0
         Link 0: Recovery Errors: 0
         Link 0: CRC Errors: 0

         Link 1: Replay Errors: 0
         Link 1: Recovery Errors: 0
         Link 1: CRC Errors: 0

         [...]

         Link 17: Replay Errors: 0
         Link 17: Recovery Errors: 0
         Link 17: CRC Errors: 0
```

How to read it: `CRC Errors` count data that arrived corrupted (its checksum didn't match); `Replay Errors` count retransmissions triggered by such corruption; `Recovery Errors` count times the link had to be retrained. Healthy links read 0 or grow very slowly. Counters that keep rising under load point at a bad link.

`nvidia-smi nvlink --status` (short form `-s`, used in [NVLink and topology checks](gpu-software-stack.md#nvlink-and-topology-checks)) shows each link's current speed:

```bash
$ nvidia-smi nvlink --status
GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-abcdefab-cdef-abdc-abcd-abababababab)
         Link 0: 26.562 GB/s
         [...]
         Link 17: 26.562 GB/s
```

`nvidia-smi nvlink -h` lists the rest (reporting, resetting counters, and so on). Whether two GPUs can do peer-to-peer at all is checked in [How to check if P2P is supported](../network/measuring-and-debugging.md#how-to-check-if-p2p-is-supported).

### Checking row-remapping health before a job dies

Row remapping is how Ampere and later GPUs handle memory that has gone bad: a degrading row in an HBM bank is replaced by one of the spare rows each bank reserves for this. It is [NVIDIA's replacement for page retirement](https://docs.nvidia.com/deploy/a100-gpu-mem-error-mgmt/latest/row-remapping.html), with a much larger budget: up to 512 remappings for the frame buffer, against 64 page retirements before. The [ECC section above](#ecc-errors-and-row-remapping-in-nvidia-smi--q) read it after an Xid 63 or 64. This one checks it deliberately, before a job dies.

`nvidia-smi -q` prints a long block per GPU. For a whole node, a CSV query gives one line per GPU:

```bash
$ nvidia-smi --query-remapped-rows=gpu_name,gpu_bus_id,remapped_rows.failure,remapped_rows.pending,remapped_rows.correctable,remapped_rows.uncorrectable --format=csv
gpu_name, gpu_bus_id, remapped_rows.failure, remapped_rows.pending, remapped_rows.correctable, remapped_rows.uncorrectable
NVIDIA H200, 00000000:59:00.0, No, No, 0, 0
[...]
NVIDIA H200, 00000000:A5:00.0, No, No, 0, 0
```

That is a healthy node: no failures, nothing pending, no rows remapped. When it isn't all `No` and `0` (this is what `remap_verdict` in `xid_triage.py` above encodes):

- `correctable` and `uncorrectable` count the rows already remapped, after repeated SBEs and after a DBE respectively. A non-zero count alone is not a reason to pull the GPU: it means the sparing did its job.
- `pending: Yes` means a remap has been decided but takes effect only at the next GPU reset. It doesn't put the running job at risk of touching the bad cell. It does change how much memory the GPU has: when the trigger was an uncorrectable error, the driver offlines the page containing it immediately, and that page stays out of the allocatable pool until the reset remaps the row in hardware and returns the address space. So a job sized to fill HBM can start hitting OOM on this GPU while still fitting on a healthy one.
- `failure: Yes` means a remap was attempted and failed. This one means RMA rather than reset-and-return.

The same fields are available in `--query-gpu`, alongside other per-GPU columns (the acceptance script uses this form), e.g. `nvidia-smi --query-gpu=gpu_bus_id,remapped_rows.pending,remapped_rows.failure --format=csv`. For the full field list, including the `remapped_rows.sbe`/`.dbe` aliases:

```bash
nvidia-smi --help-query-remapped-rows
```

### Detecting a node with missing GPUs

A new VM occasionally comes up with fewer GPUs than it should have. A quick check that there are exactly 8:

```bash
#!/bin/bash
# test-gpu-count.sh
expected=8
n=$(nvidia-smi -q | grep -c UUID)
if [ "$n" -ne "$expected" ]; then echo "broken node: $n GPUs, expected $expected"; exit 1; fi
echo "ok: $n GPUs"
```

Run it with `bash test-gpu-count.sh`; the exit code is 1 on a broken node and 0 on a good one, so a scheduler health check can use it directly. With a fake `nvidia-smi` on `PATH` that reports 7 GPUs: **[ran]**

```
$ PATH=$PWD/fakebin:$PATH bash test-gpu-count.sh; echo rc=$?
broken node: 7 GPUs, expected 8
rc=1
```

Why the `if`: the tempting one-liner `test $(...) != 8 && echo "broken node" && false` reports correctly, but its exit code is wrong. On a good node the `test` fails, and since it's the last command that ran, the script exits 1 on a good node too. The acceptance script's `gpu count` check does the same job inside the full suite.

### Detecting that you got the same broken node again

This matters mostly to cloud users who rent GPU VMs. You launch a VM, find a broken GPU, discard it, launch a new one, and the GPUs are broken again. Chances are you got the same physical machine with the same broken GPUs. Each GPU has a unique UUID, so log them before discarding the node:

```bash
$ nvidia-smi -q | grep UUID
    GPU UUID                              : GPU-2b416d09-4537-ecc1-54fd-c6c83a764be9
    GPU UUID                              : GPU-0309d0d1-8620-43a3-83d2-95074e75ec9e
    GPU UUID                              : GPU-4fa60d47-b408-6119-cf63-a1f12c6f7673
    GPU UUID                              : GPU-fc069a82-26d4-4b9b-d826-018bc040c5a2
    GPU UUID                              : GPU-187e8e75-34d1-f8c7-1708-4feb35482ae0
    GPU UUID                              : GPU-43bfd251-aad8-6e5e-ee31-308e4292bef3
    GPU UUID                              : GPU-213fa750-652a-6cf6-5295-26b38cb139fb
    GPU UUID                              : GPU-52c408aa-3982-baa3-f83d-27d047dd7653
```

On the new VM, run it again. If the UUIDs match, you have the same broken GPUs. Once a VM is gone it's too late to collect them, so add this to the node's startup:

```bash
nvidia-smi -q | grep UUID > nvidia-uuids.$(hostname).$(date '+%Y-%m-%d-%H:%M').txt
```

Save it on a persistent filesystem that survives the VM; if there is none, write it locally and copy it to cloud storage immediately.

What to do with that knowledge:

- Sometimes a reboot gets you new hardware. With some providers that happens almost every reboot, with others never, and the behavior can change.
- If you keep getting the same broken node, allocate a new VM while still holding the broken one, and discard the broken one only once the new one is running. That guarantees different GPUs (though not working ones). If the workload allows, a static cluster makes keeping good hardware much easier.
- This matters most when GPUs fail only after some use, so the problem isn't obvious at first. Even after you report a node, the technician may not reproduce the failure and may put it back into circulation. Keep a log of bad UUIDs so you recognize a lemon immediately instead of 10 hours in.
- Report bad nodes through the provider's mechanism, not just discard them. Most users just discard, so the next user gets the bad node; in some situations users get a very high percentage of bad nodes.

### Real GPU utilization: dcgm-exporter

The `GPU-Util` column of `nvidia-smi` is not GPU utilization. As [this article](https://arthurchiao.art/blog/understanding-gpu-performance/) explains, it is the percentage of time during which one or more kernels were executing. It says nothing about whether one SM or all of them were busy, so a tiny `matmul` running continuously shows a very high `GPU-Util` while most of the GPU idles. (Stas Bekman noticed this when `GPU-Util` read 100% on every GPU of a node while one GPU had stopped responding and all the others were blocked waiting for it.)

What you want is how much of the available capacity is used, sometimes called saturation. `nvidia-smi` doesn't report it. DCGM's profiling metrics do, and [dcgm-exporter](https://github.com/NVIDIA/dcgm-exporter) serves them over HTTP in the text format of [Prometheus](https://prometheus.io/) (the time-series database from [section 3](gpu-software-stack.md#dcgm-the-health-daemon-and-diagnostics), which scrapes such endpoints periodically and stores the values for dashboards and alerts). It needs DCGM (`datacenter-gpu-manager`), root access, and (to build it) a recent Go toolchain. It works only on datacenter NVIDIA GPUs, not consumer ones. **[untested]**

Build it:

```bash
git clone https://github.com/NVIDIA/dcgm-exporter.git
cd dcgm-exporter
make binary
```

Write a config file listing the metrics to export:

```bash
$ cat << EOT > dcp-metrics-custom.csv
DCGM_FI_PROF_SM_OCCUPANCY,       gauge, The ratio of number of warps resident on an SM.
DCGM_FI_PROF_PIPE_TENSOR_ACTIVE, gauge, Ratio of cycles the tensor (HMMA) pipe is active.
DCGM_FI_PROF_PIPE_FP16_ACTIVE,   gauge, Ratio of cycles the fp16 pipes are active.
DCGM_FI_PROF_PIPE_FP32_ACTIVE,   gauge, Ratio of cycles the fp32 pipes are active.
EOT
```

Each line is a DCGM field name, its Prometheus metric type (`gauge` is a value that can go up and down) and a description. Warps, SM occupancy and the tensor pipe (HMMA is the matrix-multiply-accumulate instruction) are covered in [SMs, warps and schedulers](accelerator/execution-model.md#sms-warps-and-schedulers) and [Tensor cores](accelerator/execution-model.md#tensor-cores). The repo's `etc/dcp-metrics-included.csv` lists all available metrics if you want more.

Launch the daemon (root is required); `-c 500` refreshes every 0.5 s:

```bash
$ sudo cmd/dcgm-exporter/dcgm-exporter -c 500 -f dcp-metrics-custom.csv
[...]
INFO[0000] Starting webserver
INFO[0000] Listening on                                  address="[::]:9400"
```

Then, with a GPU workload running in another console, poll it:

```bash
watch -n 0.5 "curl http://localhost:9400/metrics"
```

(`watch -n 0.5` re-runs the command every 0.5 seconds and redraws the screen.) The last column of each metric line is the value, where `1.0` means 100%.

That's the quick way. The intended use is to have Prometheus scrape it and chart it. The article's dashboard shows SM occupancy and tensor core, FP16 and FP32 pipe utilization in its second row:

![dcgm-metrics](accelerator/nvidia/images/dcgm-metrics.png)

([source](https://arthurchiao.art/blog/understanding-gpu-performance/))

The same article demonstrates the problem with a kernel that does nothing except occupy one SM:

```bash
$ cat << EOT > 1_sm_kernel.cu
__global__ void simple_kernel() {
    while (true) {}
}

int main() {
    simple_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}
EOT
```

`<<<1, 1>>>` launches a single block of a single thread, so the kernel occupies one SM and leaves the rest idle (kernel launches and `nvcc` are covered in [CUDA C++ from zero](accelerator/writing-kernels.md#cuda-c-from-zero)). Compile it, run it in console A:

```bash
nvcc 1_sm_kernel.cu -o 1_sm_kernel
./1_sm_kernel
```

and in console B:

```bash
$ nvidia-smi
Tue Oct  8 09:49:34 2024
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.90.12              Driver Version: 550.90.12      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA A100 80GB PCIe          Off |   00000000:01:00.0 Off |                    0 |
| N/A   32C    P0             69W /  300W |     437MiB /  81920MiB |    100%      Default |
|                                         |                        |             Disabled |
```

`GPU-Util` reads `100%`, yet 1 of the A100's 108 SMs is in use, and it isn't computing anything, just spinning in an infinite loop. The honest number for "is my training using the GPU" is MFU, covered in [Profiling and optimization](../training/profiling-and-optimization.md).

### AMD: hiding devices

[The AMD section](#amd--rocm-equivalents) mapped the AMD stack onto NVIDIA's; the hardware terms (CU, wavefront, XCD, LDS) are in [AMD differences](accelerator/execution-model.md#amd-differences). This and the next subsections are AMD's troubleshooting tools. AMD's tooling story is younger than NVIDIA's, so expect fewer tools and rougher edges.

ROCm has more than one "visible devices" variable because devices can be hidden at different layers of the stack. From the top down: PyTorch calls **HIP**, AMD's CUDA-like runtime API; HIP sits on **ROCr**, the ROCm runtime (AMD's implementation of the HSA, Heterogeneous System Architecture, runtime API), which talks to the kernel driver.

For PyTorch on ROCm, the usual CUDA-style choice is at the HIP level:

```bash
HIP_VISIBLE_DEVICES=0,1 python my-program.py
```

`CUDA_VISIBLE_DEVICES` is honored the same way on AMD (a CUDA-compatible alias for HIP's variable). Both only filter what the **HIP** runtime exposes. ROCr can still initialize every GPU, and non-HIP clients that sit on ROCr are unaffected: OpenCL (a vendor-neutral GPU compute API), AOMP (AMD's OpenMP offloading compiler), UCX (a communication library that MPI and others use), and so on. To hide devices from the whole user-mode ROCm stack, filter at the **ROCr** level:

```bash
ROCR_VISIBLE_DEVICES=0,1 python my-program.py
```

`ROCR_VISIBLE_DEVICES` also accepts UUID strings; HIP's list takes indices only. If both are set, HIP sees only the devices ROCr left visible, so HIP's indices are relative to that filtered set: with `ROCR_VISIBLE_DEVICES=4,5,6,7 HIP_VISIBLE_DEVICES=1`, the process gets physical GPU 5.

On Linux, AMD's isolation docs use `ROCR_VISIBLE_DEVICES` when you want stack-wide isolation; for ordinary single-framework PyTorch work, `HIP_VISIBLE_DEVICES` or `CUDA_VISIBLE_DEVICES` is enough. See AMD's [GPU isolation techniques](https://rocm.docs.amd.com/en/latest/reference/system-optimization/gpu-isolation.html).

### AMD: `rocm-smi` and `amd-smi`

`rocm-smi` is the `nvidia-smi` equivalent: a condensed state of all ROCm accelerators. `amd-smi` is its newer replacement with the same role ([the AMD section](#amd--rocm-equivalents) shows its inventory commands). An 8x MI300X node:

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

How to read the columns:

- `[Model : Revision]` is the PCI device id (`0x74a1` = MI300X) and revision.
- `Temp (Junction)` is the hottest on-die sensor; `Power (Socket)` is the whole package's draw; `PwrCap` is the power limit.
- `Partitions (Mem, Compute)` are the memory and compute partitioning modes. `NPS1` means all HBM is one NUMA node, and `SPX` means all XCDs (the MI300X's compute dies) act as one GPU. The MI300X can also be split into several smaller GPUs.
- `SCLK` is the compute (shader) clock and `MCLK` the memory clock. The 132 MHz `SCLK` here is an idle GPU's clock.
- `Perf` is the performance-level policy.
- `VRAM%` is the share of HBM in use, and `GPU%` the busy percentage. Like `nvidia-smi`'s `GPU-Util`, `GPU%` is the share of time something ran, not how much of the GPU it used ([Real GPU utilization](#real-gpu-utilization-dcgm-exporter)).

Oddly it shows memory usage only as a percentage, which isn't very practical. `rocm-smi --showmeminfo vram` prints used and total bytes per GPU.

A handy alias to watch it update in real time (put it in your `~/.bashrc` to keep it; `watch -n 1` re-runs the command every second):

```bash
alias wr='watch -n 1 rocm-smi'
```

### AMD: `rocminfo`

`rocminfo` is the `nvidia-smi -q` equivalent: detailed information about each device. It lists both CPUs and GPUs. HSA calls every processor that can run work an **agent**, CPU or GPU, and numbers them as nodes: here the two CPUs are nodes 0-1 and the eight GPUs nodes 2-9. A snippet for cpu0 and gpu0:

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

How to read it. The fields that matter most for a GPU:

- `Name` is the ISA name: `gfx942` is the MI300X. `Compute Unit: 304` is the number of CUs, and `Wavefront Size: 64` the AMD equivalent of the warp size.
- `Max Clock Freq.` and the cache sizes (`L1` 16 KB per CU, `L2` 8 MB).
- `BDFID` is the PCI bus:device.function address packed into one number. `50688` is `0xC600`, i.e. bus `0xC6`, device 0, function 0, which you can match against `lspci`.
- `Pool Info` sizes are the HBM: `201310208 KB` is 192 GiB per GPU. **Coarse grained** memory is coherent only at kernel boundaries and is the fast default for device memory; **fine grained** memory can be shared coherently with the host or other GPUs while a kernel runs. The `GROUP` pool (64 KB) is the per-workgroup LDS, AMD's shared memory.
- In the ISA string, `sramecc+` means ECC on the on-chip SRAM is enabled and `xnack-` that GPU page-fault retry is off.
- `IOMMU Support` is what the runtime actually sees for the IOMMU, the subject of the next subsection.

For the CPU agent, `Compute Unit: 128` is the 64-core EPYC's hardware threads (with SMT), and its `Pool Info` (about 756 GiB) is that socket's system memory.

### AMD: hangs or slow multi-GPU runs and the IOMMU

If a multi-GPU ROCm run hangs or crawls, the host's IOMMU configuration is a common culprit. The IOMMU translates the addresses that devices use for DMA, and pass-through mode (`iommu=pt`) skips that translation for devices the host owns; both are introduced in [IOMMU, VFIO and PCIe passthrough](../foundations/containers-and-virtualization.md#iommu-vfio-and-pcie-passthrough). The correct setting is platform-specific, so match your hardware instead of blindly disabling it:

| Platform | AMD's guidance |
|---|---|
| MI300X, MI325X, MI350X, MI355X | IOMMU **enabled** in BIOS and in pass-through mode on the kernel command line: `iommu=pt` together with `amd_iommu=on` (or `intel_iommu=on` on Intel hosts). MI350X/MI355X (ROCm 7.0.1+) explicitly list `IOMMU enabled` + `iommu=pt`. |
| MI100, MI200 (MI210/MI250/MI250X) | AMD's qualified BIOS tables typically set the NBIO **IOMMU to Disable** (NBIO is the CPU's north-bridge I/O block, where the BIOS's IOMMU switch lives). `iommu=pt` is called out only for hosts with 256+ logical CPUs, so that X2APIC (the interrupt-controller mode that can address more than 255 CPUs, and needs the IOMMU for interrupt remapping) can address every core. |
| Older systems, last resort | where neither pass-through nor the BIOS default helped, fully disabling it with `amd_iommu=off` (or the softer `iommu=soft`) was the historical workaround. This is an invasive, system-wide boot change, and current AMD guidance prefers pass-through. See AMD's [IOMMU advisory for multi-GPU environments](https://community.amd.com/t5/knowledge-base/iommu-advisory-for-multi-gpu-environments/ta-p/477468) and this [issue report](https://github.com/stas00/ml-engineering/issues/1#issuecomment-1076830400) (that case was actually on NVIDIA A6000s, so this is not AMD-only). |

These are boot-time kernel parameters, so they need admin access. GRUB is the bootloader that passes them to the kernel; they go in `/etc/default/grub` (the path varies by OS), then you regenerate the GRUB config and reboot. **[untested]**

```bash
# /etc/default/grub
GRUB_CMDLINE_LINUX="... amd_iommu=on iommu=pt"

sudo update-grub                                  # Ubuntu/Debian
# sudo grub2-mkconfig -o <path of grub.cfg>       # RHEL family
sudo reboot
cat /proc/cmdline                                 # after reboot: the parameters the running kernel actually got
```

Then check the `IOMMU Support` line of `rocminfo` above, which shows what the runtime sees.

Parts of this chapter are adapted from Stas Bekman's [Machine Learning Engineering Open Book](https://github.com/stas00/ml-engineering) (CC BY-SA 4.0).

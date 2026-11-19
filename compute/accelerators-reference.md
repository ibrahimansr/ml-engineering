# Accelerators: Reference

Reference material split out of [Accelerators: Specs, Vendors and Achievable Performance](accelerators.md) so the main chapter reads faster. The sections stand alone: read them when you need them. Earlier chapters link straight to the section they use.

## Caches

A cache is fast on-chip memory (SRAM) that holds recently or frequently used data and instructions so they don't have to be fetched from HBM again. L1 is usually the smallest and fastest, L2 larger and a bit slower, and some accelerators add an L3 that is bigger and slower still. Together they massively reduce trips to HBM. ([The memory hierarchy](../foundations/computer-architecture.md#the-memory-hierarchy) covers caches in general; the GPU's hierarchy in detail is in the next chapter's [Memory hierarchy](accelerator/execution-model.md#memory-hierarchy).)

Cache size matters when running benchmarks, because you need to flush the cache between measurements - otherwise the second iteration reads its inputs from cache and runs faster than a real training step would. The MAMF finder [does exactly that](#how-mamf-and-msmf-are-defined), with a buffer larger than any cache below.

Cache specs are hard to normalize across vendors, because they describe different resources at different scopes. NVIDIA reports a combined, configurable L1/texture/shared-memory capacity per SM; AMD reports L1 per CU and L2 per XCD; Intel Gaudi has software-managed SRAM and configurable cache modes that don't map onto a conventional GPU cache hierarchy. So the tables below keep the scope used in each vendor's documentation, and don't derive per-accelerator totals by multiplying private local caches. Accelerators without a directly documented comparable value are left out rather than filled with an estimate.

**Table 1: broadly shared on-chip caches.** Sorted by shared capacity descending, then announcement date descending; for a grouped row the date is that of the newest product in the row. It answers which accelerator publishes more broadly shared cache, but it isn't a cache-performance ranking - cache type, bandwidth, latency, sharing domain and workload hit rate still differ. Vendor unit labels are preserved, so the ordering of `MB` and `MiB` values is approximate.

| Shared capacity | Accelerator / architecture       | Cache type                     | Scope / qualification | Approx.<br>announced | Ref. |
| :-------------- | :------------------------------- | :----------------------------- | :-------------------- | :------------------- | :--- |
| 256MiB          | AMD MI350X / MI355X              | Infinity Cache (LLC)           | per accelerator       | 2025-06              | 1    |
| 256MiB          | AMD MI300X / MI325X              | Infinity Cache (LLC)           | per accelerator       | 2024-10              | 2    |
| 192MB           | AMD MI455X                       | global L2                      | per accelerator       | 2026-06              | 3    |
| 126MB           | NVIDIA Blackwell (GB200 example) | L2                             | per GPU               | 2024-03              | 4    |
| 96MiB           | Intel Gaudi3                     | configurable L3 or 4 L2 slices | per accelerator       | 2024-04              | 5    |
| 50MiB           | NVIDIA Hopper (H100 example)     | L2                             | per accelerator       | 2022-03              | 6    |
| 40MiB           | NVIDIA Ampere (A100 example)     | L2                             | per accelerator       | 2020-05              | 7    |

Sources:

1. AMD MI350X / MI355X [cache specification](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/workload-optimization.html) and [announcement](https://newsroom.amd.com/news/amd-unveils-vision-for-an-open-ai-ecosystem-detai/)
2. AMD MI300X / MI325X [cache specification](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/workload-optimization.html) and [announcement](https://www.amd.com/en/newsroom/press-releases/2024-10-10-amd-delivers-leadership-ai-performance-with-amd-in.html)
3. [AMD Instinct MI455X specifications and announcement](https://www.amd.com/en/products/accelerators/instinct/mi400/mi455x.html)
4. NVIDIA Blackwell [cache specification](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html) and [announcement](https://nvidianews.nvidia.com/news/nvidia-blackwell-platform-arrives-to-power-a-new-era-of-computing)
5. Intel Gaudi3 [cache specification](https://www.intel.com/content/www/us/en/content-details/817486/intel-gaudi-3-ai-accelerator-white-paper.html) and [announcement](https://newsroom.intel.com/artificial-intelligence/vision-2024-enterprise-ai-gaudi-3-open-systems-strategy)
6. NVIDIA Hopper [cache specification](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html) and [announcement](https://nvidianews.nvidia.com/news/nvidia-announces-dgx-h100-systems-worlds-most-advanced-enterprise-ai-infrastructure)
7. NVIDIA Ampere [cache specification](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html) and [announcement](https://nvidianews.nvidia.com/news/nvidias-new-ampere-data-center-gpu-in-full-production)

**Table 2: vendor-native local resources.** Sorted by vendor, then announcement date descending.

| Vendor | Accelerator / architecture | Approx.<br>announced | Vendor-reported local resource                  | Scope               | Ref. |
| :----- | :------------------------- | :------------------- | :---------------------------------------------- | :------------------ | :--- |
| AMD    | MI455X                     | 2026-06              | not disclosed in the cited public specification |                     | 1    |
| AMD    | MI350X / MI355X            | 2025-06              | 32KiB L1 data cache; 4MiB L2 cache              | per CU; per XCD     | 2    |
| AMD    | MI300X / MI325X            | 2024-10              | 32KiB L1 data cache; 4MiB L2 cache              | per CU; per XCD     | 3    |
| Intel  | Gaudi3                     | 2024-04              | 24MiB configurable cache                        | per DCORE; 4 DCOREs | 4    |
| Intel  | Gaudi2                     | 2022-05              | 48MiB software-managed SRAM                     | per accelerator     | 5    |
| NVIDIA | Blackwell (GB200 example)  | 2024-03              | 256KB combined L1/texture/shared memory         | per SM              | 6    |
| NVIDIA | Hopper (H100 example)      | 2022-03              | 256KiB combined L1/texture/shared memory        | per SM              | 7    |
| NVIDIA | Ampere (A100 example)      | 2020-05              | 192KiB combined L1/texture/shared memory        | per SM              | 8    |

Sources:

1. [AMD Instinct MI455X specifications and announcement](https://www.amd.com/en/products/accelerators/instinct/mi400/mi455x.html)
2. AMD MI350X / MI355X [cache specification](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/workload-optimization.html) and [announcement](https://newsroom.amd.com/news/amd-unveils-vision-for-an-open-ai-ecosystem-detai/)
3. AMD MI300X / MI325X [cache specification](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/workload-optimization.html) and [announcement](https://www.amd.com/en/newsroom/press-releases/2024-10-10-amd-delivers-leadership-ai-performance-with-amd-in.html)
4. Intel Gaudi3 [cache specification](https://www.intel.com/content/www/us/en/content-details/817486/intel-gaudi-3-ai-accelerator-white-paper.html) and [announcement](https://newsroom.intel.com/artificial-intelligence/vision-2024-enterprise-ai-gaudi-3-open-systems-strategy)
5. Intel Gaudi2 [architecture specification](https://docs.habana.ai/en/latest/Gaudi_Overview/Gaudi_Architecture.html) and [announcement](https://www.intel.com/content/www/us/en/developer/articles/technical/habana-gaudi2-processor-for-deep-learning.html)
6. NVIDIA Blackwell [cache specification](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html) and [announcement](https://nvidianews.nvidia.com/news/nvidia-blackwell-platform-arrives-to-power-a-new-era-of-computing)
7. NVIDIA Hopper [cache specification](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html) and [announcement](https://nvidianews.nvidia.com/news/nvidia-announces-dgx-h100-systems-worlds-most-advanced-enterprise-ai-infrastructure)
8. NVIDIA Ampere [cache specification](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html) and [announcement](https://nvidianews.nvidia.com/news/nvidias-new-ampere-data-center-gpu-in-full-production)

**Table 3: per-product detail.** This restores the per-product detail of an earlier, simpler comparison that mixed scopes, while separating vendor-native scope from derived distributed capacity. Sorted by vendor, then product. `Distributed capacity` is arithmetic across private units and doesn't mean one operation can access it as a single shared cache. The `Qualification` column records what was wrong or ambiguous in that earlier entry.

| Vendor | Accelerator / product             | Vendor-native detail                                 | Distributed capacity       | Shared cache / LLC        | Qualification of the earlier entry                                        | Ref. |
| :----- | :-------------------------------- | :--------------------------------------------------- | :------------------------- | :------------------------ | :------------------------------------------------------------------------ | :--- |
| AMD    | MI300X                            | 32KiB L1/CU; 4MiB L2/XCD; 8 XCDs                     | 32MiB L2; L1 not derived   | 256MiB Infinity Cache     | Earlier 0.25MiB L1 total multiplied a per-CU value by the eight-XCD count | 1    |
| AMD    | MI325X                            | 32KiB L1/CU; 4MiB L2/XCD; 8 XCDs                     | 32MiB L2; L1 not derived   | 256MiB Infinity Cache     | Earlier 0.25MiB L1 total multiplied a per-CU value by the eight-XCD count | 1    |
| AMD    | MI350X / MI355X                   | 32KiB L1/CU; 4MiB L2/XCD; 8 XCDs                     | 32MiB L2; L1 not derived   | 256MiB Infinity Cache     | Earlier 0.25MiB L1 total multiplied a per-CU value by the eight-XCD count | 1    |
| Intel  | Gaudi2                            | 48MiB software-managed SRAM per accelerator          | not applicable             | no conventional GPU LLC   | Earlier 48MiB L2 label is retained here with its software-managed meaning | 2    |
| Intel  | Gaudi3                            | 24MiB configurable cache/DCORE; 4 DCOREs             | 96MiB                      | configurable L3 or 4 L2s  | Earlier 24MiB x 4 = 96MiB arithmetic is retained with its cache mode      | 3    |
| NVIDIA | A100 PCIe                         | 192KiB combined L1/texture/shared memory/SM; 108 SMs | 20.25MiB                   | 40MiB L2                  | Earlier row said 128KiB/SM, but its 20.25MiB total used 192KiB/SM         | 4    |
| NVIDIA | A100 SXM                          | 192KiB combined L1/texture/shared memory/SM; 108 SMs | 20.25MiB                   | 40MiB L2                  | Earlier row said 128KiB/SM, but its 20.25MiB total used 192KiB/SM         | 4    |
| NVIDIA | B200 SXM                          | not disclosed in the cited product-specific form     | not derived                | not disclosed             | Earlier `???` placeholder is retained as `not disclosed`                  | 5    |
| NVIDIA | B300 SXM                          | not disclosed in the cited product-specific form     | not derived                | not disclosed             | Earlier `???` placeholder is retained as `not disclosed`                  | 5    |
| NVIDIA | GH100 full implementation         | 256KiB combined L1/texture/shared memory/SM; 144 SMs | 36MiB                      | up to 60MB L2             | Earlier `GH100 SXM` mixed 132 enabled SMs with the full-die 60MB L2       | 6, 7 |
| NVIDIA | GH200 SXM (original label)        | uses its installed Hopper GPU cache hierarchy        | not derived                | use the GPU configuration | Earlier 256KiB x 132 = 33MiB and 60MiB L2 mixed product/full-die scopes   | 6, 7 |
| NVIDIA | H100 SXM                          | 256KiB combined L1/texture/shared memory/SM; 132 SMs | 33MiB                      | 50MiB L2                  | Earlier 192KiB x 132 = 24.75MiB distributed-capacity value is preserved   | 6, 7 |
| NVIDIA | H200 NVL                          | 256KiB combined L1/texture/shared memory/SM; 132 SMs | 33MiB                      | 60MiB L2                  | New row                                                                   | 8    |
| NVIDIA | H200 SXM                          | 256KiB combined L1/texture/shared memory/SM; 132 SMs | 33MiB                      | 50MiB L2                  | Earlier 192KiB x 132 = 24.75MiB distributed-capacity value is preserved   | 6, 7 |

Sources:

1. [AMD Instinct MI300 Series / MI350 Series workload optimization](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/workload-optimization.html)
2. [Intel Gaudi architecture](https://docs.habana.ai/en/latest/Gaudi_Overview/Gaudi_Architecture.html)
3. [Intel Gaudi3 white paper](https://www.intel.com/content/www/us/en/content-details/817486/intel-gaudi-3-ai-accelerator-white-paper.html)
4. [NVIDIA Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)
5. [NVIDIA Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)
6. [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
7. [NVIDIA Hopper architecture white paper](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c)
8. [NVIDIA H200 specifications](https://www.nvidia.com/en-us/data-center/h200/)

The distributed-capacity column is simple multiplication, e.g. H100 SXM: $256\ \text{KiB} \times 132 = 33{,}792\ \text{KiB} = 33\ \text{MiB}$; A100: $192\ \text{KiB} \times 108 = 20.25\ \text{MiB}$; MI300X L2: $4\ \text{MiB} \times 8 = 32\ \text{MiB}$.

When comparing cache specs:

- **Compare the scope first.** A per-SM or per-CU cache is private to one compute unit; an L2, L3, LLC or Infinity Cache may be shared by a much larger part of the accelerator.
- **Check the semantics.** A hardware-managed cache, configurable shared memory and software-managed SRAM can have similar capacities while needing very different programming and access patterns.
- **Don't treat a sum of private caches as one big shared cache.** The total describes physical capacity, not how much cache any one operation can access.
- **Capacity is only one factor.** Bandwidth, latency, associativity, sharing domain and the workload's hit rate can matter more, so compare performance with workload-representative benchmarks.

Vendor-specific notes:

1. AMD's L3 is the AMD Infinity Cache, which AMD also calls the Last Level Cache (LLC) in its specs.
2. Gaudi's architecture differs from a GPU's. Its MME and TPC have private buffers, called Suspension Buffers, that do some of an L1 cache's job: they let data be reused from the buffer instead of being read repeatedly from L2/L3/HBM. Gaudi2 and Gaudi3 have the same Suspension Buffers for the TPC and MME.
3. Gaudi2 has no conventional GPU cache hierarchy. It has scratchpad SRAM: software decides what goes into or out of it, by programming dedicated DMA engines to move data between SRAM and HBM.
4. Gaudi3's 96MiB cache can be configured by software as either a single L3 cache or 4 slices of 24MiB L2 (at tensor-level granularity). The L2 configuration is 2x faster than L3.



## Clock speed

The [clock rate](https://en.wikipedia.org/wiki/Clock_rate) is the frequency the chip runs at, and it's one of the three factors of the [TFLOPS formula](accelerators.md#how-to-calculate-theoretical-tflops). Newer generations typically raise it.

Spec sheets may publish a base clock, a boost (peak, maximum) clock, or no compute clock at all:

- A **base clock** is a vendor-defined nominal operating point, not the minimum idle clock.
- A **boost**, peak or maximum clock is a vendor-defined upper operating target. It isn't guaranteed under every workload; the actual frequency depends on the SKU, the power limit, temperature and the workload itself (see [Cooling](#cooling) for how the chip picks it).

The clock is useful for calculating TFLOPS *within* one architecture, but MHz alone doesn't rank performance *across* architectures - an AMD MI355X at 2400MHz isn't faster than a B200 at 1965MHz because of the clock. The same family can also have different clocks across SKUs and system configurations, so always check the clock of your specific accelerator.

The table is sorted by compute clock, highest first; products whose vendors don't disclose a compute clock follow alphabetically. `not disclosed` means the linked product specification didn't publish a compute clock when checked on 2026-07-28.

**Generally available:**

| Accelerator      | Compute Clock (MHz) | Notes                                        |
| :--------------- | ------------------: | :------------------------------------------- |
| AMD MI355X       |                2400 | 1; Peak Engine Clock                         |
| AMD MI350X       |                2200 | 14; Peak Engine Clock                        |
| AMD MI300X       |                2100 | 3; Peak Engine Clock                         |
| AMD MI325X       |                2100 | 4; Peak Engine Clock                         |
| NVIDIA GB200 SXM |                2062 | 5; device-reported maximum SM clock          |
| NVIDIA B200 SXM  |                1965 | 5; device-reported maximum SM clock          |
| NVIDIA H200 SXM  |                1830 | 12                                           |
| NVIDIA H100 SXM  |                1830 | 12                                           |
| Intel Gaudi2     |                1650 | 6,10,11; device-reported MME clock; TPC=1800 |
| Intel Gaudi3     |                1600 | 6,10,11; device-reported MME clock; TPC=1600 |
| NVIDIA A100 SXM  |                1410 | 7; GPU Boost Clock                           |
| NVIDIA A100 PCIe |                1410 | 7; GPU Boost Clock                           |
| NVIDIA B300 SXM  |       not disclosed | 8                                            |
| NVIDIA GB300 SXM |       not disclosed | 8                                            |
| NVIDIA H200 NVL  |       not disclosed | 13                                           |

**Announced, availability not confirmed:**

| Accelerator      | Compute Clock (MHz) | Notes                                        |
| :--------------- | ------------------: | :------------------------------------------- |
| AMD MI455X       |                2400 | 2; Peak Engine Clock                         |
| NVIDIA Rubin GPU |       not disclosed | 9                                            |

Notes:

1. [AMD Instinct MI355X specifications](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html)
2. [AMD Instinct MI455X specifications](https://www.amd.com/en/products/accelerators/instinct/mi400/mi455x.html)
3. [AMD Instinct MI300X specifications](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)
4. [AMD Instinct MI325X specifications](https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x.html)
5. [NVIDIA System Management Interface documentation](https://docs.nvidia.com/deploy/nvidia-smi/)
6. [Intel Gaudi `hl-smi` documentation](https://docs.habana.ai/en/latest/Management_and_Monitoring/Embedded_System_Tools_Guide/System_Management_Interface_Tool.html)
7. [NVIDIA A100 architecture white paper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)
8. [NVIDIA GB300 NVL72 specifications](https://www.nvidia.com/en-us/data-center/gb300-nvl72/)
9. [NVIDIA Vera Rubin NVL72 specifications](https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/)
10. [Intel Gaudi architecture documentation](https://docs.habana.ai/en/latest/Gaudi_Overview/Gaudi_Architecture.html)
11. Intel Gaudi exposes separate clocks for its Matrix Multiplication Engine (MME) and Tensor Processing Core (TPC).
12. 1830MHz is NVIDIA's official compute clock for H200 and H100, superseding the 1980MHz that was widely published earlier. It's also self-consistent: it reproduces the 989 TFLOPS bf16 spec exactly through the [TFLOPS calculation](accelerators.md#how-to-calculate-theoretical-tflops), where 1980MHz would give 1070.
13. [NVIDIA H200 specifications](https://www.nvidia.com/en-us/data-center/h200/) publish no clock for H200 NVL. Its 835 TFLOPS bf16 spec works back to about 1545MHz through the [TFLOPS calculation](accelerators.md#how-to-calculate-theoretical-tflops) (the last line of `tflops.py`), if it has the same 132 SMs as the H200 SXM - which NVIDIA doesn't publish for the NVL either.
14. [AMD Instinct MI350X specifications](https://www.amd.com/en/products/accelerators/instinct/mi350/mi350x.html)

### Reading the actual clock

The clock that matters is the one under load. How to read it on each vendor **[GPU, untested]**. `rocm-smi`/`amd-smi` and Intel's `hl-smi` are AMD's and Intel's counterparts of [`nvidia-smi`](../foundations/linux-and-os.md#nvidia-smi):

```bash
# NVIDIA, headless: current SM clock of GPU 0 (drop -i 0 for all GPUs; drop the grep for the full report,
# whose "Max Clocks" section shows the theoretical clock)
nvidia-smi -q -d CLOCK -i 0 | grep -B2 " SM " | head -3
# NVIDIA, a continuous log with timestamps, one row per second
nvidia-smi --query-gpu=index,timestamp,power.draw,clocks.sm,clocks.mem,clocks.gr --format=csv -l 1 -i 0
# NVIDIA with an X server running
nvidia-settings -q GPUCurrentClockFreqs

# AMD: actual, then theoretical
rocm-smi -g
amd-smi metric --clock

# Intel Gaudi: actual, then theoretical (either of the last two)
hl-smi -Q clocks.current.soc --format=csv
hl-smi -Q clocks.max.soc --format=csv
hl-smi -Q clocks.limit.soc --format=csv
```

In the continuous log, `clocks.sm` is the compute clock and `power.draw` the board power. Run it in a second terminal while your training runs: if `clocks.sm` sits well below the spec clock while `power.draw` sits at the power limit, the GPU is power-capped (normal for a heavy matmul, see [Why MSMF is below the spec](accelerators.md#why-msmf-is-well-below-the-spec-and-mamf-above-msmf)); if it drops while the power is below the limit, look at temperature (see [Cooling](#cooling)).



## Cooling

The one practical thing to understand about cooling: an accelerator that isn't kept cool throttles its compute clock and slows everything down, and can sometimes crash, although throttling is supposed to prevent that.

What temperature to aim for under heavy load is hard to state exactly. Probably anything under 80°C is good, and lower is better - 70-75°C is an excellent range. Throttling is likely to start somewhere around 84-90°C. Beyond throttling, a prolonged very high temperature is likely to shorten a GPU's life.

On modern GPUs throttling isn't a single cliff at the thermal limit: the boost clock is a continuous function of temperature, power and current. NVIDIA's GPU Boost (and AMD's equivalent on Instinct) continuously samples die temperature, board power and per-rail current, and picks the highest stable point on the chip's [V-F-T curve](accelerators.md#silicon-lottery) that fits inside all of those budgets. Cross any one of them and the clock steps down; come back under and it steps back up. Which budget binds first depends on the workload. A heavy matmul usually hits the power cap first, and then a cooler GPU gains little until it gets close to its slowdown temperature (see the measurement below). A GPU that does run hot - a poorly seated heatsink, blocked airflow - hits its temperature budget first and becomes an obvious straggler.

**A measured example: what temperature does below the throttle point.** On an 8x H200 node and an 8x B200 node, the same matmul ran on all 8 GPUs at once for 10 minutes, starting from a cool idle, while each GPU's temperature, power, clock and the driver's clock-limit reasons were logged every second:

- Every GPU was held back by its power cap (700W on H200, 1000W on B200) throughout, never by its temperature. The thermal slowdown flags never fired, and the GPUs peaked at 65-75°C, 11-26°C below their slowdown point (the hottest H200 came within 6°C).
- While each GPU warmed up by 10-13°C its clock barely moved: by a median of 0.3MHz per °C on H200 and 1MHz per °C on B200. That cost a median of under 1% of throughput over the whole warm-up, and about 2% on the most affected B200.
- GPUs in the hot slots ran 8-11°C hotter than those in the cool ones, yet that accounted for only 2 of the 40MHz between the H200 GPUs' clocks and 11 of the 142MHz between the B200 ones. The rest is the chip - the [silicon lottery](accelerators.md#silicon-lottery).

These are two nodes' numbers and a different set of GPUs may differ, but the pattern should hold: on a power-capped workload like a large matmul, a GPU comfortably below its slowdown temperature gains almost nothing from running cooler. Cooling pays off by keeping a GPU away from its slowdown point, where temperature takes over the clock. (The MI350X node in note 8 of the [MAMF table](accelerators.md#mamf-and-msmf-comparison-table) is the counter-example: its GPUs at 95-100°C sustained up to 25% less than those at 74-77°C, one of them throttling below its power cap.)

To see which limit is holding your GPU back, run this under load **[GPU, untested]**:

```bash
nvidia-smi --query-gpu=temperature.gpu,temperature.gpu.tlimit,clocks_event_reasons.sw_power_cap,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown --format=csv -l 1
```

How to read it: `-l 1` prints a new CSV row every second; each `clocks_event_reasons.*` column reads `Active` while that limit is currently holding the clock down; and `temperature.gpu.tlimit` is how many °C the GPU is below its maximum operating temperature. `sw_power_cap` `Active` with both thermal columns `Not Active` is the healthy power-capped state described above.

For a one-shot check whether an NVIDIA GPU is being throttled, run `nvidia-smi -q -d PERFORMANCE`: if `SW Thermal Slowdown` or other entries read `Active`, you aren't getting the GPU's full performance, and you should look into better cooling. GPU health checks more broadly, and the AMD equivalents, are in [The GPU node software stack](gpu-software-stack.md).



## The MAMF finder: `mamf-finder.py`

<a id="maximum-achievable-and-sustainable-matmul-flops-finder"></a>
<a id="maximum-achievable-matmul-flops-finder"></a>

### What it is and why

[`mamf-finder.py`](accelerator/benchmarks/mamf-finder.py) is the Maximum Achievable and Sustainable Matmul FLOPS benchmark. It lives at `compute/accelerator/benchmarks/mamf-finder.py` in this repository, next to its node-wide wrapper [`mamf-finder-all-gpus.py`](accelerator/benchmarks/mamf-finder-all-gpus.py). It was derived from the research in [The Case for Co-Designing Model Architectures with Hardware](https://arxiv.org/abs/2401.14489) and from the [EleutherAI cookbook's sizing benchmarks](https://github.com/EleutherAI/cookbook/tree/main/benchmarks/sizing).

The motivation is the one from [MAMF and MSMF](accelerators.md#mamf-and-msmf): manufacturers publish theoretical TFLOPS that can't be reached, so when we optimize software we have no realistic bar to compare against. This benchmark scans many large matmul shapes and reports both the highest *achievable* TFLOPS (MAMF, boost burst) and the highest *sustainable* TFLOPS (MSMF, held steady under full load). Since ML workloads are dominated by large matmuls, MSMF is a realistic bar for sustained training throughput, and MAMF the short-burst ceiling. It produced every row of the [MAMF and MSMF table](accelerators.md#mamf-and-msmf-comparison-table).

Supported accelerators:

- NVIDIA: Volta, Ampere, Hopper, Blackwell, Rubin.
- AMD: MI250 and higher.
- Intel Gaudi2/3.
- Intel GPUs (A770, A750, B580, etc.).
- Apple silicon (MPS) - which is how it was run for this chapter.

`--search auto` (the default) derives its shapes from the GPU's compute-unit layout, which `mamf-finder.py` models only for NVIDIA and AMD; on the others, give it a range with `--search grid`. Power and clock telemetry, whose SM clock certifies that a MAMF iteration ran at boost, exists for NVIDIA, AMD and Gaudi (see [Telemetry](#telemetry)). `mamf-finder-all-gpus.py` runs on NVIDIA and AMD only.

Some important notes:

- If you find a better and more efficient way to detect the best matmul TFLOPS by treating each new accelerator as a black box, the tool's author welcomes the improvement, with the log of the run.
- Likewise if you know special conditions under which the benchmark shows better results - kernel settings or similar. The MI300X `numa_balancing` advice in [Architecture-specific setup](#architecture-specific-setup) came about that way.
- A big part of each matmul's cost is HBM I/O. A fused kernel that chains 2 or more matmuls without their intermediate results leaving the accelerator's registers will be faster than what this benchmark reports.
- Sample your accelerator's actual clock. If it runs slower than the clock used in the spec, there's no chance of reaching the theoretical TFLOPS (see [How to calculate theoretical TFLOPS](accelerators.md#how-to-calculate-theoretical-tflops)).

### How MAMF and MSMF are defined

Both numbers are measured on the same candidate shapes with the same per-iteration work: zero a 256MB buffer to flush the L2 cache, rewrite the output matrix, then run the matmul. The L2 flush makes every iteration read its inputs from HBM, as a real training step does, rather than from a cache still warm from the previous iteration. Only the matmul is timed, with device events (CUDA events, see [Events](../foundations/concurrency.md#events)), and TFLOPS is `2*M*N*K` divided by that time. That's `mini_mamf.py` [above](accelerators.md#the-core-of-a-matmul-benchmark).

**MAMF** (Maximum *Achievable* Matmul FLOPS) is the rate a short burst reaches at the boost clock:

- Each shape gets 10 bursts of 20 back-to-back iterations, and each burst starts after 0.25s of idle so the SM clock climbs back to boost.
- A background thread samples the SM clock every ~0.5ms, and each iteration is paired with the lowest clock sampled while it ran.
- An iteration counts as boost if that clock is at least 0.97x the highest clock seen anywhere in the run.
- A shape's MAMF is the median of its 5 fastest boost iterations, so a single lucky iteration can't set it; on H200 and B200 the median of 5 and the single fastest differ by at most 0.7%.
- The MAMF headline is the highest shape MAMF among the shapes that reached boost.

**MSMF** (Maximum *Sustainable* Matmul FLOPS) is the highest rate a shape holds without falling, over a timed window, on a GPU that has run at full load long enough for its clock to settle:

1. **Settle clock.** Up to 20s of 8192x8192x8192 BF16 matmuls, stopping early once the SM clock stops dropping. An idle GPU boosts and reads high; this brings it down to the clock its power limit allows under load. It doesn't wait for the temperature to settle: on one H200, a GPU went on warming from 63 to 72°C over the minutes of measurement that followed, but below the throttle point warming costs under 1% on most GPUs (see [Cooling](#cooling)).
2. **Rank.** Every candidate runs 0.2s untimed, then 1s timed.
3. **Window.** The 10 fastest get the full window: 1s untimed, then 2s timed, split into 8 consecutive sub-windows. The window's TFLOPS is total FLOPs over total matmul time. A shape is **steady** if its last 4 sub-windows are at most 1% slower than its first 4. The sub-windows may also spread by at most 10%: at the power cap a 0.25s sub-window jitters by up to ~4% while the window's mean holds.
4. **K walk** (`--search auto` only, explained below). Starting from the 4 fastest steady layouts and the most-square 1-wave layouts, K is stepped down, each step measured over the same window.
5. **Lock-in.** The fastest shape that was steady, or that was faster than every steady one, is re-measured over a 4s window and published if it's steady there; otherwise the next fastest is tried, up to 4 of them. If none holds steady over 4s, the fastest shape that was steady over its 2s window is published. If no shape is steady at all, `mamf-finder.py` warns and publishes the fastest unsteady one.

The window is a *time*, not an iteration count, because a fixed count of 100 iterations lasts 13ms on a fast B200 shape and about 1s on a slow one, and a GPU keeps some of its boost for a fraction of a second after a shape starts. On one H200 a 0.3s window read `2816x1536x20480` at 745-752 TFLOPS, its 4s lock-in window at 728-735, and a 10s run at 736-739. Power doesn't enter the definition: a shape whose throughput doesn't fall is sustainable, whatever lets it hold that rate. The `W` and `MHz` printed next to MSMF are the mean instantaneous power and SM clock over the timed window. On NVIDIA the instantaneous power comes from NVML's `NVML_FI_DEV_POWER_INSTANT` field (NVML, the NVIDIA Management Library, is the driver API that `nvidia-smi` itself reads), since its default power reading is a 1s average.

**The K walk.** At the power cap, the K that a short burst favors isn't the K that sustains best. A shorter K lets the same `(M, N)` hold a higher clock at the same power, up to where the clock reaches its ceiling. On H200 BF16, `2816x1536` sustains:

| K     | TFLOPS | Clock / power                         |
| ----: | -----: | :------------------------------------ |
| 20480 |    727 |                                       |
| 10240 |    741 |                                       |
|  6144 |    749 | 1879MHz, 685W                         |
|  4096 |    722 | 1980MHz ceiling, below the 700W cap   |

At K=4096 the clock is at its ceiling and the power falls below the cap, so shortening K further no longer helps. The search's candidates sit at long K, so the walk takes the 4 fastest steady `(M, N)` layouts and steps each one's K down by K/8 rounded down to a power of two (2048 from K=20480), for up to 10 steps, stopping once a step is more than 1% below the best in that walk. A walk never starts above the search's largest K (20480): the shapes past it are there for MAMF, and a longer K only sustains less.

Why several layouts, and why these rules - each was learned from a miss:

- **Two layouts**, because two can look alike at long K and part at short K: on H200 FP8, `1280x16896` and `1536x2816` both sustain about 1290-1305 TFLOPS at K=16384-20480, but only `1536x2816` climbs to 1335-1372 at K=10240. Walks from one layout found it in 5 of 8 runs, against 10 of 11 with two.
- **Four**, because three can tie: in one more H200 FP8 run, three layouts sat within 0.2% of each other at long K, `1536x2816` was not among the two walked, and MSMF came out 1281 instead of about 1330.
- **Plus the 1-wave layouts**, because even four can miss: in one of 32 H200 FP8 runs, six layouts sat at 1276-1291 TFLOPS at long K, `1536x2816` read the lowest and wasn't walked, and MSMF came out 1289 instead of about 1337. So the most-square 1-wave layouts (`msmf_kwalk_waves`=1; one wave is one thread-block tile per SM, see [What makes a fast shape](#what-makes-a-fast-shape)) - `1536x2816` among them on H200 - are walked in both orientations no matter how they rank. In 32 more H200 FP8 runs none missed; the extra walks cost about 10s on H200 and nothing measurable on B200. On an A100 PCIe, walking 4 layouts instead of 2 plus the 1-wave ones took a run from 5:13-5:32 to 6:06 minutes, with the same MAMF and MSMF shapes.
- **1s ranking and the top 10**, because in the 11th of the two-layout runs a 0.3s ranking window read `1536x2816` 2.5% below its usual rate, which left it out of the 6 shapes that then got a full window, so no walk started from it and MSMF came out 4% low. With 0.3s and the top 6, 2 of 32 more H200 FP8 runs missed the same way; with 1s and the top 10 none of 32 did, at the cost of about a minute per run.

**How close it gets.** On an H200 and a B200, five `--search auto` runs each gave MSMF of 743-746 and 1421-1428 TFLOPS, and MAMF of 843-850 and 1682-1705. Each published MSMF shape, run back to back for 10s in two passes, sustained within 0.6% of its MSMF; the best 10s rate among them was 750 on the H200 and 1433 on the B200.

### Architecture-specific setup

<a id="architecture-specific-notes"></a>

Follow these before running the benchmark to get the best results.

**AMD MI300X, MI325X and similar:**

1. Turn `numa_balancing` off. Automatic NUMA balancing is a Linux kernel feature that periodically unmaps a process's pages to see which NUMA node (see [NUMA and numactl](../foundations/linux-and-os.md#numa-and-numactl)) touches them, and migrates them there; the extra page faults and migrations add jitter. The setting lasts until reboot **[GPU host, untested]**:

   ```bash
   sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'
   cat /proc/sys/kernel/numa_balancing     # should now print 0
   ```

2. Enable PyTorch TunableOp:

   ```bash
   export PYTORCH_TUNABLEOP_ENABLED=1
   ```

   This makes the first iteration of each new `matmul` shape very slow, while PyTorch searches the BLAS libraries for the best GEMM algorithm for it, but subsequent calls are likely to be significantly faster than the default. See [Accelerating models on ROCm using PyTorch TunableOp](https://rocm.blogs.amd.com/artificial-intelligence/pytorch-tunableop/README.html) and the [TunableOp documentation](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/tunable/README.md). On ROCm 10, tuning a new shape takes ~2 minutes, which would turn a ~2,000-shape search into days, so `mamf-finder.py` pauses tuning while it searches (the default kernel is within ~1-2% on good shapes, close enough to rank them) and tunes only the shapes it confirms, before timing them. `--tune tunableop_confirm_max=N` (default 8) caps how many shapes that is; a BF16 `--search auto` run on MI300X then needs about 5 minutes of search plus ~16 minutes of tuning.

On ROCm, `--search auto` prints a note that it was checked against an exhaustive grid only on one MI300X in BF16, where it came within 0.1% of a 16,000-shape grid. A newer software stack can move the winning *shape* rather than the peak: `10240x15360x8192` dropped from 663.6 TFLOPS (ROCm 6.3, torch 2.5.1) to 625.5 (ROCm 10, torch 2.12), while `12288x9728x8192` held ~660 on both. So after a stack upgrade, re-search instead of re-timing the old winner. Pick the GPU with `--cuda_device N`, not `HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES` (ROCm's equivalents of `CUDA_VISIBLE_DEVICES`): on some virtualized MI300X hosts, hiding GPU0 left torch with no usable device.

**Intel dGPUs (A770, A750, B580, etc.):** follow the Intel Extension for PyTorch [installation steps](https://pytorch-extension.intel.com/installation?platform=gpu).

**Intel Gaudi:** the table's Gaudi rows were measured with `PT_HPU_LAZY_MODE=1` (lazy mode: ops are collected into a graph and compiled before running).

#### Telemetry

On NVIDIA, AMD and Gaudi, the SM clock certifies that a MAMF iteration ran at boost, power ranks the scouted shapes, and both are printed next to each number; MSMF is judged on throughput alone. `mamf-finder.py` refuses to start if the telemetry package is missing - `nvidia-ml-py`, `amdsmi` or `habana-pyhlml` respectively, the Python bindings to each vendor's management library (what `nvidia-smi`, `amd-smi` and `hl-smi` are built on) - or can't read power and clock. `--telemetry off` runs anyway, with MAMF not boost-validated. The `amdsmi` reads are confirmed on MI300X; as of this writing the `pyhlml` reads are untested on hardware. On MI300X the card hits its 750W cap within one BF16 kernel and bursts after an idle start *slower*, not faster, so on that GPU MAMF is less reliable than MSMF.

On Apple MPS there's no telemetry at all; the run below prints `-` in the `W` and `MHz` columns and warns that MAMF isn't boost-validated.

### Data types

`K` is the reduction dimension: $(M \times K) \cdot (K \times N) = (M \times N)$. The default dtype is `bfloat16`; `--dtype` also accepts:

- `float16`, `float32`;
- `float8_e4m3fn` - the fp8 of NVIDIA and AMD MI350X/MI355X;
- `float8_e4m3fnuz` - the fp8 of AMD MI300X/MI325X;
- three block-scaled formats with a `bfloat16` output: `mxfp8` (`float8_e4m3fn` operands with one `float8_e8m0fnu` scale per 32 elements), `mxfp4` (fp4 `e2m1` operands with one `float8_e8m0fnu` scale per 32 elements) and `nvfp4` (fp4 `e2m1` operands with one `float8_e4m3fn` scale per 16 elements).

`mxfp8` and `mxfp4` need hardware MX support, such as NVIDIA Blackwell or AMD MI350X/MI355X; `nvfp4` needs NVIDIA Blackwell; and both fp4 formats need `K` to be a multiple of 32. In these names `eXmY` is the number of exponent and mantissa bits, `fn` means finite only (no infinities) and `uz` unsigned zero (no negative zero); the formats are decoded in [Floating point: IEEE 754](../foundations/computer-architecture.md#floating-point-ieee-754) and [FP8 training](../training/arithmetic.md#fp8-training). Before searching, `mamf-finder.py` runs a tiny matmul in the requested dtype and exits with the vendor library's error if the device or the software stack can't run it - and for an fp8 dtype it names the other fp8 variant if that one runs.

### Auto search: the best the GPU can do anywhere

```bash
cd compute/accelerator/benchmarks
./mamf-finder.py --output_file=$(date +'%Y-%m-%d-%H:%M:%S').txt
# equivalent: ./mamf-finder.py --search auto --output_file=$(date +'%Y-%m-%d-%H:%M:%S').txt
```

`$(date +'%Y-%m-%d-%H:%M:%S')` is shell command substitution: it's replaced by the current timestamp, so each run writes its log to a new file. (Without `--output_file` the log goes to `results/mm.out` next to the script.)

Auto finds near-peak shapes via hardware heuristics and reports **two** headlines:

- **MAMF** - the boost-clock burst ceiling.
- **MSMF** - the steady rate under full load, which matches real training throughput. **For picking shapes a real model will use, MSMF is the number that matters.**

It's great for a spec-sheet number and isn't tied to any particular model. On H200 and B200, in BF16 and FP8, every `--search auto` run came within 0.6% of a 128,000-shape exhaustive scout followed by the same confirm, and most runs beat it, by up to 3.6%. A run takes about 4 minutes on one H200, 4.5 on a B200 and 6 on an A100 PCIe - including adaptive warmup, the recall screen, the settle-clock step and both confirms - against 17-29 minutes on 8 GPUs for the exhaustive scout. On AMD with TunableOp, tuning the confirm shapes adds ~2 minutes per shape; a run took 20 minutes on MI300X.

How `--search auto` chooses candidate shapes (then both modes share the confirm phase below):

1. **Wave candidates at a short K set.** Enumerate the `(M, N)` shapes that fill an integer number of full waves of thread-block tiles across the SMs (most-square, widest and tallest per wave count 1..16), and measure *every* one at K = 1024, 2048, 3072, 4096, 8192, 12288, 14336, 16384 and 20480. That covers the wave-perfect basin at both ends of K; measuring all wave candidates (not only a ranked top-N) at all those Ks is what keeps both H200's high-K winners (`1536x2816x20480`) and B200's low-K ones (`3072x18944x2048`) in the confirm set. Without K=2048-4096, B200 runs reached the best burst family only when a step-3 seed happened to land near it, and their MAMF varied by 1-2% on the same GPU.
2. **Coarse M×N planes at `Kmin` and K=3072.** Cheap grids at the smallest K and in the low-K boost basin. The latter is required for B200's non-wave BF16 MAMF family, found by exhaustive search.
3. **Tight local grid** around both power-ranked and raw-peak scout seeds (±4 steps of 256 on M/N, ±2 steps of 1024 on K): endgame polish for off-axis peaks, without letting MSMF-oriented ranking hide a MAMF basin.
4. **MAMF recall screen.** Cheaply measure the raw scout leaders plus the top several K choices from every wave `(M, N)` layout in the actual idle-then-burst regime (2 iterations after 50ms idle), and promote its leaders into a dedicated MAMF shortlist. The GPU is still hot from scouting, so a burst that never reaches the boost clock is retried after the confirm's 0.25s idle: on H200 the first screened shapes otherwise burst at 1635MHz against a 1980MHz boost. Then the screen looks past the search's largest K (`max_size`, 20480): for up to 8 layouts (`k_edge_layouts`) whose burst at that K beats every shorter K they were screened at, steepest gain first, K at 1.5x and 2x is screened too (up to `k_edge_max`=65536), stopping once the burst stops rising. H200 BF16's `2816x1536` bursts 828 TFLOPS at K=20480 and 842 at K=32768. The scouts can't show this: they run hot, and a long-K call throttles more, so they read a falling curve.

The shared confirm phase (identical for `--search auto` and `--search grid`, except that grid measures only shapes in your range, so it skips the K-edge screen and the MSMF K walk):

5. **Candidate union and settle clock.** Combine the independently ranked MAMF and MSMF shortlists, then run the chip at full load until its clock stops dropping before any sustainable measurement (`msmf_settle_clock_s`). Without it an idle card boosts and reads high, so MSMF would depend on what the card was doing before the run.
6. **MSMF confirm** (sustainable): the ranking, windows, K walk and lock-in of [How MAMF and MSMF are defined](#how-mamf-and-msmf-are-defined), over every shape in the union. The union always includes (a) the best scouted K for each most-square low-wave shape (w=1..4) in **both** `(M, N)` orientations - e.g. H200's `1536x2816` family - and (b) the **fattest** scouts (largest `min(M, N, K)`, then volume), which reliably reach the power cap.
7. **MAMF confirm** (achievable): re-measure the **same union** as MSMF - not MSMF's candidate list, and not a scout-clock pre-filter. A fat shape that saturates while scouting still recovers boost after a short idle (measured on B300). Each burst is **queued and synchronized once**: synchronizing before each start event would leave the GPU idle at the moment the event is taken, charging kernel-launch latency and the idle DVFS re-ramp (dynamic voltage and frequency scaling, the mechanism that raises and lowers the clock with load) to the timed kernel - 1.6-19% depending on shape, worst for large footprints, enough to re-rank candidates. Per-iteration clocks are recovered by projecting each iteration's GPU-time window onto the background sampler's host timeline and taking the minimum clock inside it. After both headlines are picked, `mamf-finder.py` prints a **same-shape cross-check**: each winning shape measured in both regimes, so the boost-to-saturated penalty is comparable on the same GEMM.

Each shape is scouted only once, so a candidate that several phases propose costs nothing extra.

`mamf-finder.py -h` lists the options. The algorithm's own knobs - the names in backticks above (`confirm_reps`, `msmf_settle_clock_s`, `msmf_kwalk_waves`, `k_edge_layouts`, `k_edge_max`, `tunableop_confirm_max`, ...) - live in the `Tuning` class near the end of the script, each with its default and a line on what it does. To try another value pass e.g. `--tune msmf_settle_clock_s=30 --tune confirm_reps=7`.

#### Reproducibility

A published figure must be reproducible on the same GPU with the same setup, so:

- **Measure while the other GPUs compute.** A real workload runs every GPU of the node at once, and they share the board's power and cooling budget; measured with idle siblings, MSMF gets headroom a full node never has. See [Measuring a node and reporting results](#measuring-a-node-and-reporting-results).
- **Single GPU: run `mamf-finder.py` on its own.** With idle siblings, MAMF is a best-case boost burst and MSMF a single-GPU upper bound. `mamf-finder.py` samples its same-board siblings throughout the MSMF measurement and, if any sat idle, says so under the results with each idle GPU's share of the time.
- **Neither headline is a lucky maximum.** MSMF is the mean over a 4s window that held steady, and MAMF is the median of the 5 fastest boost iterations; the power and clock they ran at are printed in the headline (`… 985W 1360MHz`).
- **To reproduce a published number, re-run its exact shape in grid mode**, e.g. `--m 9472 --n 6144 --k 12288`. Auto's job is to *find* a near-peak shape anywhere; grid *reproduces* (or searches within) a known range and still runs the same MAMF/MSMF confirm. Re-running `auto` re-searches and may land on a different, equivalent shape.

How much a result moves between runs, shown on two GPUs as examples rather than measured for every GPU: with the default settings, over 4 runs on every GPU of an 8x H200 node (FP8) and an 8x B200 node (BF16), all 8 measured at once, a GPU's MSMF moved by 0.5-0.6% at the median and 2.2% at most, its MAMF by 0.3-0.9% and 1.5%, and none of the 64 runs missed the best shape. The 2.2% was one H200 run whose 4s lock-in read 1.6% below the same shape's window in the K walk, as the GPU warmed from 63 to 72°C. The GPUs of one node differ by more than a GPU does between runs, as the next measurement shows.

**Alone vs all-8 concurrent.** The same protocol on both chips: five `--search auto` runs on GPU0 with its siblings idle, then five rounds with the same script on all 8 GPUs at once.

*B200 bf16 (torch 2.14.0+cu130):*

| Setup                        | MSMF mean      | MSMF range | MSMF clock    | MAMF mean |
| :--------------------------- | -------------: | ---------: | ------------: | --------: |
| GPU0 alone (n=5)             |         1423.8 |       0.5% | 1396–1411 MHz |    1697.6 |
| GPU0 while all 8 busy (n=5)  | 1417.2 (−0.5%) |       0.5% | 1387–1396 MHz |    1688.8 |
| All 8 GPUs × 5 rounds (n=40) | 1437.5 (+1.0%) |   **4.9%** | 1368–1527 MHz |    1746.7 |

*H200 bf16 (torch 2.14.0+cu130):*

| Setup                        | MSMF mean     | MSMF range | MSMF clock    | MAMF mean |
| :--------------------------- | ------------: | ---------: | ------------: | --------: |
| GPU0 alone (n=5)             |         744.8 |       0.4% | 1835–1873 MHz |     848.4 |
| GPU0 while all 8 busy (n=5)  | 743.4 (−0.2%) |       1.1% | 1850–1872 MHz |     847.2 |
| All 8 GPUs × 5 rounds (n=40) | 755.5 (+1.4%) |       3.4% | 1850–1938 MHz |     852.1 |

How to read the tables: the first two rows compare the *same* GPU with its siblings idle and busy; the third pools every GPU of the node over 5 rounds. The lesson: on these two nodes, the other GPUs computing at the same time barely moved a GPU's own numbers. GPU0's MSMF fell 0.5% on B200 and 0.2% on H200, and its MAMF stayed within run-to-run noise, since boost bursts draw only ~150-330W. What sets the GPUs apart is the chips themselves: across the 40 runs, MSMF spans **4.9%** on the B200 node and 3.4% on the H200 node, and each GPU stays in its own part of that range from round to round (on B200, GPU0 averaged 1417 and GPU5 1472). GPU0 happened to be the slowest chip on both nodes, which is why the 8-GPU mean sits above it. So a single GPU's MSMF describes that chip, not the GPU model, and the published node figure is the median GPU. A full-node synchronous training run still waits on its slowest GPU, so the bottom of the 8-GPU range is what it gets. How much the concurrency itself costs depends on the board's shared power and cooling, so measure the whole node.

### Reading the output

The console shows each phase as a title, a column header and one row that is updated in place as shapes are measured; when the phase ends, its winning row stays on screen. On an H200, `mamf-finder.py --dtype bfloat16 --m_range 1024 4097 1024 --n 1536 2816 --k 4096 8192 20480` prints:

```
Grid search: sweeping 24 shapes (scout 20 iters / 8 warmup), then confirm ...
     #  MxNxK                mean median    max   best     W   MHz
    18  3072x2816x20480     828.4  828.6  831.6  828.4   506  1980  <- best mean

MAMF recall screen: 8 shapes ...
     #  MxNxK                peak   MHz
     1  3072x2816x20480     837.7  1965  <- best peak

MSMF (sustainable) confirm: 10 shapes ...
  MxNxK              TFLOPS     W   MHz
  3072x2816x8192      722.5   681  1612 drop   0.7% spread  4.2%  locked in  <- MSMF

MAMF (achievable) confirm: 10 shapes ...
  MxNxK              TFLOPS     W   MHz  top peaks
  3072x2816x20480     842.3   172  1980  842.8 842.3 842.3 841.9 841.2  <- MAMF

--------------------------------------------------------------------------------

** Results:

Tried 24 shapes => the best outcomes were:
MAMF (max achievable,  boost burst): 842 TFLOPS @ 3072x2816x20480 (MxNxK)  172W 1980MHz
MSMF (max sustainable, saturated):   722 TFLOPS @ 3072x2816x8192 (MxNxK)  681W 1612MHz
note: 7 same-board sibling GPU(s) sat idle during the MSMF measurement, so MSMF reads high.
      Share of the time each was idle: GPU1 100%, GPU2 100%, GPU3 98%, GPU4 94%, GPU5 94%, GPU6 94%, GPU7 94%
      With all siblings idle it is a single-GPU upper bound; for full-node MSMF use mamf-finder-all-gpus.py.
```

How to read it:

- In the grid-search (scout) row, `#` is how many shapes have been measured so far, `mean`/`median`/`max` are that shape's TFLOPS over its scout iterations, and `best` is the highest `mean` seen so far; the row left on screen is the shape with the best mean.
- The recall-screen `peak` is the fastest burst iteration.
- The MSMF row is the 4s lock-in window, which is what the headline uses: `drop` is how much slower its second half ran than its first, and `spread` how far its 8 sub-windows differ.
- The MAMF row's `top peaks` are the shape's 5 fastest boost iterations, whose median is the headline.
- The two headlines landed on different K here: K=20480 bursts fastest, while K=8192 holds a higher clock at the power cap. `--search auto` on the same GPU found MSMF 745 at `1536x2816x6144`, a K this grid doesn't include.
- `W`, `MHz` and, in the MSMF rows, `C` (the GPU temperature) are live NVML (or amdsmi/hlml) samples taken while the shape ran. On a short MAMF burst NVML's power sample can lag the kernel: on A100 PCIe a ~80ms burst after an idle reads near-idle watts, so that `W` isn't the burst's draw.
- The MAMF and MSMF headlines are integer TFLOPS; a half rounds up.
- Every row, plus the details behind each decision (the MSMF ranking, K walk and lock-in rows, the boost reference, the same-shape cross-check), is written to the `--output_file` log.

And here is a real run of the unmodified tool on the M4 laptop GPU used throughout this chapter, a 4-shape grid **[torch]**:

```bash
python compute/accelerator/benchmarks/mamf-finder.py --m 2048 4096 --n 2048 4096 --k 4096 --output_file /tmp/mps.txt
```

```
** Dtype: bfloat16
** Platform/Device info:
- Apple Metal Performance Shaders (MPS)
** Critical software versions:
- torch=2.15.0
** Additional notes:
- benchmark version: 4
- search mode: grid
--------------------------------------------------------------------------------
Warming up (adaptive: until matmul throughput plateaus) ...
adaptive warmup: throughput plateaued after 5.1s / 350 iters
Grid search: sweeping 4 shapes (scout 20 iters / 8 warmup), then confirm ...
     #  MxNxK                mean median    max   best
     3  4096x2048x4096        3.6    3.6    3.6    3.6  <- best mean

MAMF recall screen: 4 shapes ...
     #  MxNxK                peak   MHz
     1  4096x2048x4096        3.3     -  <- best peak

MSMF (sustainable) confirm: 4 shapes ...
  settle clock 41.3s
  MxNxK              TFLOPS     W   MHz   C
  4096x4096x4096        3.1     -     -   - drop   1.0% spread  4.0%  not steady
  2048x2048x4096        3.1     -     -   - drop  -3.3% spread 14.3%  not steady
  4096x2048x4096        3.3     -     -   - drop  -1.1% spread  2.8%  locked in  <- MSMF

MAMF (achievable) confirm: 4 shapes ...
  MxNxK              TFLOPS     W   MHz  top peaks
  2048x4096x4096        3.6     -     -  3.6 3.6 3.6 3.6 3.6  <- MAMF
WARNING: no SM clock readings, so MAMF 4 TFLOPS is not boost-clock validated

** Results:

Tried 4 shapes => the best outcomes were:
MAMF (max achievable,  boost burst): 4 TFLOPS @ 2048x4096x4096 (MxNxK)
MSMF (max sustainable, saturated):   3 TFLOPS @ 4096x2048x4096 (MxNxK)

Across 4 shapes in range: m=2048,4096 | n=2048,4096 | k=4096 in this run:
arithmetic mean: 3.3 TFLOPS
geometric mean:  3.3 TFLOPS

Legend: TFLOPS = 10**12 FLOPS
Elapsed time: 0:01:46
```

(The output is trimmed: the full header also prints the command line, OS and environment variables, and the in-place rows of the MSMF ranking are omitted.) How to read it: the M4 does about 3.6 TFLOPS in a burst and 3.3 sustained, a ~9% gap; two shapes were rejected as `not steady` (their sub-windows spread 4% and 14%, over the limits in [How MAMF and MSMF are defined](#how-mamf-and-msmf-are-defined)); there's no telemetry on MPS, hence the `-` columns and the warning; and because the headlines are integers, "4" and "3" are coarse for a chip this slow - read the confirm rows for the decimals. The run took 1:46, of which 41s was the settle-clock step.

### What makes a fast shape

Peak GEMM shapes aren't mysterious; they follow the recipe of the paper the tool is based on ([arXiv:2401.14489](https://arxiv.org/abs/2401.14489)):

1. **Tensor-core alignment**: M, N, K multiples of `128 B / sizeof(dtype)` elements (bf16: 64, fp8: 128).
2. **Tile quantization**: the output divides cleanly into the kernel's tile (the canonical efficient tile is 128×256), so M and N multiples of 256 are safe.
3. **Wave quantization**: the thread-block count is close to an exact multiple of the SM count, so there's no partial tail wave. Winners typically have a wave efficiency (`wave_eff`, the fraction of the last wave that does useful work) of at least ~0.99. Tile and wave quantization are worked through in [Why dims want multiples of 8, 16, 64, 128](accelerator/execution-model.md#why-dims-want-multiples-of-8-16-64-128).
4. **K only sets arithmetic intensity** (AI, FLOPs per byte moved, see [Arithmetic intensity and the roofline model](../foundations/computer-architecture.md#arithmetic-intensity-and-the-roofline-model)); it doesn't enter the tile/wave math. Larger K means more compute-bound. Empirically there are *two* high-TFLOPS basins: high AI (large K) and many waves (minimal K, huge M×N), which is why `auto` covers both.

The H200's favorite `1536x2816` is an example: with the 128×256 tile it's $12 \times 11 = 132$ output tiles - exactly one per SM, one perfect wave:

```python
# wave.py - why 1536x2816 is a favorite on a 132-SM H100/H200
import math
sms, tm, tn = 132, 128, 256
for m, n in [(1536, 2816), (2816, 1536), (2048, 2816), (4096, 4096)]:
    for a, b in [(tm, tn), (tn, tm)]:
        tiles = math.ceil(m / a) * math.ceil(n / b)
        waves = tiles / sms
        print(f"{m}x{n} tile {a}x{b}: {tiles:4d} tiles = {waves:5.2f} waves, last-wave efficiency {waves/math.ceil(waves):.2f}")
```

```
1536x2816 tile 128x256:  132 tiles =  1.00 waves, last-wave efficiency 1.00
1536x2816 tile 256x128:  132 tiles =  1.00 waves, last-wave efficiency 1.00
2816x1536 tile 128x256:  132 tiles =  1.00 waves, last-wave efficiency 1.00
2816x1536 tile 256x128:  132 tiles =  1.00 waves, last-wave efficiency 1.00
2048x2816 tile 128x256:  176 tiles =  1.33 waves, last-wave efficiency 0.67
2048x2816 tile 256x128:  176 tiles =  1.33 waves, last-wave efficiency 0.67
4096x4096 tile 128x256:  512 tiles =  3.88 waves, last-wave efficiency 0.97
4096x4096 tile 256x128:  512 tiles =  3.88 waves, last-wave efficiency 0.97
```

How to read it: `1536x2816` fills the 132 SMs exactly in either orientation, whereas `2048x2816` - a "rounder" number - needs a second wave that's only one-third full, so a third of the machine idles for the last part of the kernel. (Here "last-wave efficiency" is total waves over rounded-up waves, the same idea as `wave_eff`.) For model-design work, picking the shapes your layers will emit, see also [Profiling and optimization](../training/profiling-and-optimization.md).

### Grid search: the best shape in your range

Use `--search grid` - or just pass any `--m`/`--n`/`--k`/`--*_range` argument, which implies grid - when you care about a **specific subspace**: the shapes your new model will actually emit, a single training shape, or an accelerator-specific band you want to map exhaustively. This is the practical case.

Grid reports the **same MAMF and MSMF headlines** as auto. The only difference is how candidates are chosen: auto derives them from hardware heuristics, grid sweeps the M/N/K range *you* give, then runs the shared confirm phase. For model work, **MSMF in your range** is the useful answer, not the GPU's absolute peak.

```bash
# shapes your model will use (example: M in 2k..8k, N=K=4096)
./mamf-finder.py --m_range 2048 8193 256 --n 4096 --k 4096 --output_file=$(date +'%Y-%m-%d-%H:%M:%S').txt

# one exact shape (reproduce a published headline)
./mamf-finder.py --m 1024 --n 1024 --k 1024 --output_file=$(date +'%Y-%m-%d-%H:%M:%S').txt

# fp8 sweep over a coarse lattice
./mamf-finder.py --m_range 0 20480 1024 --n_range 0 20480 1024 --k_range 0 20480 1024 \
    --dtype float8_e4m3fn --output_file=$(date +'%Y-%m-%d-%H:%M:%S').txt

# an explicit list of shapes, one MxNxK per line, instead of the product of --m/--n/--k
./mamf-finder.py --shapes_file my_model_shapes.txt
```

How the arguments work:

- `--m`/`--n`/`--k` take one or more exact values; `--m_range`/`--n_range`/`--k_range` take `START STOP STEP` with `STOP` excluded, like Python's `range` - which is why the first example ends at 8193, to include 8192. A `START` of 0 is treated as `STEP`, since a dimension can't be 0. The grid is every combination of the M, N and K values.
- `--shapes_file` takes exact shapes, one per line (`MxNxK`, or separated by commas or spaces) - the natural input if you've logged the GEMM shapes your model actually runs.
- `--cuda_device N` picks which device to measure; `--notes "..."` adds text to the log header.

**Give the range enough room to saturate.** MSMF needs at least one shape that holds a steady rate over its window. A narrow or small grid can leave every candidate unsteady; `mamf-finder.py` then warns and publishes the fastest unsteady one. Widen M/N/K (especially $\min(M, N, K)$) so a shape that reaches the power cap is in the set. Grid doesn't run the K walk, so the best sustainable K is found only if your K range includes it.

You can Ctrl-C a long grid run and still get the best result so far. Finer steps cost a lot: 512 or 256 instead of 1024 multiplies wall time by 8x or 64x, because each of the three dimensions gets 2x or 4x more values.

### Measuring a node and reporting results

<a id="how-to-report-mamfmsmf-entries"></a>

On a node with more than one GPU, measure with [`mamf-finder-all-gpus.py`](accelerator/benchmarks/mamf-finder-all-gpus.py) rather than `mamf-finder.py` alone. In training every GPU of the node computes at once and they share the board's power and cooling budget, so a GPU measured while its siblings sit idle gets headroom a busy node never has, and its MSMF is only a single-GPU upper bound. And the GPUs of one node differ from chip to chip - by up to 4% in MSMF on the 8x H200 and 8x B200 nodes measured [above](#reproducibility) - so one GPU's numbers describe that chip, not the GPU model.

What the wrapper does:

1. Runs the full search on GPU0, which gives MAMF, MSMF and their shapes.
2. Runs those shapes pinned on each other GPU in turn, while all the rest - GPU0 included - run a continuous matmul that pins them at their power limit.
3. Reports the **median** GPU as the node's MAMF and MSMF: the median describes the GPU model rather than the luck of one chip, and one broken card barely moves it. It also reports the **slowest** GPU, which synchronous training runs at, and the spread.
4. Warns about GPUs whose MSMF window wasn't steady, ran 15°C or more hotter than the coolest GPU, or drew under 90% of the highest power: those pull the median down because of the node's cooling, not the GPU model.

GPU0's search gets all your arguments, and the pinned runs get them minus the shape selection. The script shows GPU0's console, writes each GPU's log to its own file, and runs everything with the Python you start it with - so use the one whose `torch` you want measured. With a single GPU it runs `mamf-finder.py` on its own, which gives a `Siblings` = `no` row. Pass it the same arguments as `mamf-finder.py` **[GPU, untested here]**:

```bash
python mamf-finder-all-gpus.py                        # bf16
python mamf-finder-all-gpus.py --dtype float8_e4m3fn  # fp8
```

On 8 H200s the default `--search auto` took about 9 minutes and gave a node MSMF of 755 TFLOPS, with the slowest GPU at 745; on 8 B200s it also took about 9 minutes and gave 1439, the slowest GPU 1421. A single GPU's MSMF moved by up to 1.3% between runs there, while the GPUs of one node differed by up to 4%, so rerun before calling one GPU slow.

**Contributing a result.** The numbers in the [MAMF table](accelerators.md#mamf-and-msmf-comparison-table) were gathered by Stas Bekman where he had access to the hardware, and otherwise by [contributors](https://github.com/stas00/ml-engineering/blob/master/contributors.md) who invested their time to measure them. If you get a better MAMF or MSMF than the table shows - e.g. with a more recent `torch`, CUDA or ROCm, BLAS library or driver - or you measured an accelerator or dtype the table doesn't have yet, you can contribute it upstream via a [pull request](https://github.com/stas00/ml-engineering/pulls) that updates the table or an [issue](https://github.com/stas00/ml-engineering/issues). Share the full log files of the run: the wrapper's output directory (`results/all-gpus-<timestamp>/` next to the script, or `OUT_DIR`) with `summary.txt` and every GPU's `gpu<N>.txt`, or the `--output_file` log of a single-GPU run. The logs record the software stack, the shapes, the power and the clocks, which is what makes an entry checkable and reproducible. Also mention any system settings the logs don't show, such as a disabled `numa_balancing`. A re-run of the same GPU can move its numbers by up to 2.2%, so a gain smaller than that is noise, not an improvement.



## Vendors and products

<a id="high-end-accelerators-for-ml-workloads"></a>

### Where you can get what

<a id="cloud-accelerators"></a>

The most common accelerators that can be either rented from compute clouds or purchased, as of 2026-07-31 unless stated otherwise. The "~=" comparisons are rough positioning by spec, not measured equivalence - for that, use the [MAMF table](accelerators.md#mamf-and-msmf-comparison-table).

**NVIDIA:**

- [Vera Rubin NVL72](https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/) - 72 GPUs connected at NVLink speed: 36 blocks of 2 Rubin GPUs + 1 Vera CPU.
- [GB300 NVL72](https://www.nvidia.com/en-us/data-center/gb300-nvl72/) - 72 B300-class GPUs at NVLink speed (Grace Blackwell: 36 blocks of 2 B300 + 1 Grace CPU, NVIDIA's own Arm server CPU).
- [GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) - 72 B200-class GPUs at NVLink speed (36 blocks of 2 B200 + 1 Grace CPU).
- [B300](https://www.nvidia.com/en-us/data-center/hgx/) - NVIDIA now publishes official HGX B300 platform specs and lists it as shipping. Those are 8-GPU board figures (144 PFLOPS FP4 sparse, 2.1TB total memory, 14.4TBps total NVLink), so per-GPU values have to be divided out, and HBM bandwidth still isn't published there; the [DGX B300 datasheet](https://resources.nvidia.com/en-us-dgx-systems/dgx-b300-datasheet) remains the practical source for it.
- [B200](https://www.nvidia.com/en-us/data-center/hgx/) - same story: official HGX B200 platform specs published and shipping (144 PFLOPS FP4 sparse, 1.4TB total memory, 14.4TBps total NVLink), with the [DGX B200 datasheet](https://resources.nvidia.com/en-us-dgx-systems/dgx-b200-datasheet) filling in what the platform page omits. Worth knowing: B300 cut FP64 hard - 10 TFLOPS against B200's 296 per the same table - so it isn't a strict upgrade for double-precision work.
- [H200](https://www.nvidia.com/en-us/data-center/h200/) - mainly the same as H100, but with more and faster memory. Widely available.
- [H100](https://www.nvidia.com/en-us/data-center/h100) - 2-3x faster than A100 at half precision, 6x faster with fp8; available on all Tier 1 clouds since Q4-2023.
- [GH200](https://www.nvidia.com/en-us/data-center/grace-hopper-superchip/) - 2 chips on one module: an H100-class GPU with 96GiB HBM3 or 144GiB HBM3e, plus a Grace CPU with 624GiB RAM. Available. Don't confuse it with the H200, a different card.
- [L40S](https://www.nvidia.com/en-us/data-center/l40s/) - a powerful card that is supposed to be more than 2x cheaper than H100, and more powerful than A100.
- [A100](https://www.nvidia.com/en-us/data-center/a100/#specifications) - huge availability but already dated. Given its much lower cost than H100 it's still a good GPU.

**AMD:**

- [MI455X](https://www.amd.com/en/products/accelerators/instinct/mi400/mi455x.html) ~= Rubin, a little above it on the published numbers - but those are AMD Performance Labs projections carrying "Results subject to change when products are released in market", and it isn't purchasable as of 2026-07-31. It sits in the "announced" halves of the [TFLOPS](accelerators.md#tflops-comparison-table) and [memory](accelerators.md#memory-comparison-table) tables.
- [MI355X](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html) ~= B200 - just starting to appear, mainly on Tier 2 clouds.
- [MI350X](https://www.amd.com/en/products/accelerators/instinct/mi350/mi350x.html) ~= B200 - the same silicon as MI355X, air-cooled at 1000W and 2200MHz; rentable on demand on Tier 2 clouds as of 2026-10-07.
- [MI325X](https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x.html) ~= H200 - available mainly on Tier 2 clouds.
- [MI300X](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html) ~= H100 - available mainly on Tier 2 clouds (lots of new startups).
- [MI250](https://www.amd.com/en/products/accelerators/instinct/mi200/mi250.html) ~= A100 - very few clouds have them.

**Intel:**

- Falcon Shores - canceled.
- [Gaudi3](https://habana.ai/products/gaudi3/) - somewhat below B200 in theoretical TFLOPS; available on Intel's cloud. [Spec](https://www.intel.com/content/www/us/en/content-details/817486/intel-gaudi-3-ai-accelerator-white-paper.html).
- GPU D-series / D270 (Crescent Island) - an inference GPU, launched at OCP 2026, sampling H2-2026: 160GB LPDDR5X (ODM up to 480GB) at 1.5TBps, 350W, FP8 1.3 PFLOPS (the only published compute figure). See [Intel GPU D-series](#intel-gpu-d-series-crescent-island).
- [Gaudi2](https://habana.ai/products/gaudi2/) - between A100 and H100 in theoretical TFLOPS. [Spec](https://docs.habana.ai/en/latest/Gaudi_Overview/Gaudi_Architecture.html). Available on Intel's cloud and on premises via Supermicro and WiWynn. AWS has the older Gaudi1 as [DL1 instances](https://aws.amazon.com/ec2/instance-types/dl1/).

**Amazon:**

- [Trainium2](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium2.html) < H100 - available on AWS; programmed via PyTorch XLA.

**Google:**

- [TPU7x](https://docs.cloud.google.com/tpu/docs/tpu7x) (Ironwood) - 2307 TFLOPS bf16 and 192GB HBM per chip, in 4-chip VMs, up to a 9216-chip pod. Rent-only on GCP, like every TPU generation.
- [TPU v6e](https://docs.cloud.google.com/tpu/docs/v6e) (Trillium) and [TPU v5p](https://docs.cloud.google.com/tpu/docs/v5p) - the previous two generations, both still offered.

**SambaNova:**

- [SN40L](https://sambanova.ai/products/rdu-ai-chips) - fourth-generation RDU, 16 per SambaRack; inference-oriented rather than a training part.
- [SN50](https://sambanova.ai/blog/introducing-the-sn50-rdu-purpose-built-for-agentic-inference) - fifth generation, announced Feb-2026, "will start shipping to customers in the second half of 2026".

**Cerebras on-premises clusters:** [Cerebras](https://www.cerebras.ai/) sells clusters and systems based on its Wafer Scale Engine (WSE).

<a id="cloud-only-solutions"></a>

**Cloud-only:**

- Google [TPUs](https://cloud.google.com/tpu) ([system architecture](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm)) - lock-in: you can't switch vendors the way you can go from NVIDIA to AMD. Recent architecture specs: [v7x](https://docs.cloud.google.com/tpu/docs/tpu7x), [v6e](https://docs.cloud.google.com/tpu/docs/v6e), [v5p](https://docs.cloud.google.com/tpu/docs/v5p).
- [Cerebras Cloud](https://www.cerebras.ai/cloud).

<a id="new-hardware-startups"></a>

**New hardware startups.** Possible future competitors to the big vendors; they typically target inference:

- [Tenstorrent](https://tenstorrent.com) - [n150s/n300s specs](https://docs.tenstorrent.com/aibs/wormhole/index.html#specificationsrequirements).
- [d-Matrix](https://www.d-matrix.ai) - [specs](https://www.d-matrix.ai/product/).

<a id="accelerators-in-detail"></a>

### NVIDIA

NVIDIA-specific vocabulary (CUDA, Compute Unified Device Architecture, is NVIDIA's proprietary programming platform):

- **CUDA cores** are like CPU cores, but where a CPU typically has 10-100 powerful cores, a GPU has thousands of weaker ones for massively parallel general-purpose computation. Like CPU cores, each performs a single operation per clock cycle.
- **Tensor cores** are special units designed specifically for fast multiply-and-add over small matrix tiles, i.e. matrix multiplication. Each performs many operations per clock cycle, on low- or mixed-precision data types (fp16, bf16, tf32, fp8, etc.) with some loss of precision. They're designed for ML workloads, and they're where almost all of the TFLOPS in the tables come from.
- **Streaming Multiprocessors (SMs)** are clusters of CUDA cores, tensor cores and other components (registers, shared memory, schedulers).

For example:

| GPU         | CUDA cores       | Tensor cores      | SMs |
| :---------- | ---------------: | ----------------: | --: |
| A100-80GB   | 6912             | 432 (3rd gen)     | 108 |
| H100        | 16896 (FP32)     | 528 (4th gen)     | 132 |

Both are 4 tensor cores per SM; A100 has 64 FP32 CUDA cores per SM and H100 has 128. [GPU Execution Model and Kernels](accelerator/execution-model.md) takes the SM apart.

### AMD

AMD-specific vocabulary:

- **Stream Processors** are similar in function to CUDA cores - the parallel computation units. But they aren't the same, so you can't compare two GPUs by comparing the number of CUDA cores with the number of Stream Processors.
- **Compute Units (CUs)** are clusters of Stream Processors and other components, the counterpart of an SM. Each contains Matrix Cores, AMD's counterpart of tensor cores.

For example, the AMD MI250 has 13,312 Stream Processors in 208 Compute Units (64 per CU). AMD keeps a [table comparing its high-end GPUs](https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html).

### Intel Gaudi

[Gaudi architecture](https://docs.habana.ai/en/latest/Gaudi_Overview/Gaudi_Architecture.html). Gaudi2 in numbers:

- 24x 100 Gigabit Ethernet (RoCEv2, RDMA over Ethernet) ports integrated on the chip. 21 of them connect to the other 7 chips in the node and 3 go out of the node.
- 96GiB of HBM2E per chip at 2.45TBps.
- A server/node is built from 8 Gaudi accelerators, and can be expanded with racks of those servers.

```python
# vendor_math.py - dividing out vendor totals into per-chip numbers
# Gaudi2: 24 x 100GbE per chip; 21 go to the 7 peers in the node, 3 leave the node
intra_links, inter_links, gbps = 21, 3, 100
print(f"Gaudi2 intra-node per chip: {intra_links*gbps} Gbps = {intra_links*gbps/8:.1f} GBps")
print(f"Gaudi2 inter-node per node: 8 chips x {inter_links} links x {gbps} Gbps = {8*inter_links*gbps/1000:.1f} Tbps")
print(f"Gaudi2 node HBM: 8 x 96 GiB = {8*96} GiB")

# Huawei Atlas 950 SuperPoD
print("64-card product table, per card:",
      " / ".join(f"{p*1000/64:.0f}" for p in (114.1, 58.8, 31.1)), "TFLOPS mxFP4 / mxFP8 / bf16")
print("1024-card totals, per card:", f"{2e6/1024:.0f} / {1e6/1024:.0f}", "TFLOPS mxFP4 / mxFP8")
print(f"cross-checks: 1024 / 16 cabinets = {1024//16} per cabinet;  1024 x 1.68 TBps = {1024*1.68/1000:.2f} PBps;"
      f"  1.68/2 = {1.68/2*1000:.0f} GBps per direction")

# TPU7x ICI: 3D torus -> 6 neighbours
print(f"TPU7x ICI: 6 neighbours x 200 GBps = {6*200} GBps per chip")
```

```
Gaudi2 intra-node per chip: 2100 Gbps = 262.5 GBps
Gaudi2 inter-node per node: 8 chips x 3 links x 100 Gbps = 2.4 Tbps
Gaudi2 node HBM: 8 x 96 GiB = 768 GiB
64-card product table, per card: 1783 / 919 / 486 TFLOPS mxFP4 / mxFP8 / bf16
1024-card totals, per card: 1953 / 977 TFLOPS mxFP4 / mxFP8
cross-checks: 1024 / 16 cabinets = 64 per cabinet;  1024 x 1.68 TBps = 1.72 PBps;  1.68/2 = 840 GBps per direction
TPU7x ICI: 6 neighbours x 200 GBps = 1200 GBps per chip
```

How to read the Gaudi lines: each chip has 262.5GBps of intra-node bandwidth (21 links × 100Gbps, bits divided by 8), the node has 2.4Tbps of inter-node bandwidth (8 chips × 3 links), and 768GiB of HBM in total. The Huawei and TPU lines are used in their sections below.

On TFLOPS: Intel publishes only partial figures (see note 8 of the [TFLOPS table](accelerators.md#tflops-comparison-table)), and Stas Bekman was told by an Intel representative that they had no intention of publishing more. Intel publishes [model performance benchmarks](https://www.intel.com/content/www/us/en/developer/platform/gaudi/model-performance.html) instead, but it isn't obvious how to use those to compare against other vendors - which is one more reason to measure with [`mamf-finder.py`](#the-mamf-finder-mamf-finderpy), which supports Gaudi. Gaudi2 was positioned as an H100 competitor; on the MAMF table it reaches 97% of its own (lower) peak, the highest efficiency of any accelerator measured.

On what comes after Gaudi: Falcon Shores was to replace it and was canceled. Gaudi3 remains Intel's current training-class accelerator, and the announced next data-center GPU is the [GPU D-series (Crescent Island)](#intel-gpu-d-series-crescent-island).

### Intel GPU D-series (Crescent Island)

The Intel GPU D-series, code-named Crescent Island, is Intel's inference-oriented Xe3P data-center accelerator. The first product is the **Intel GPU D270**, launched at OCP 2026 (2026-10-12) with customer sampling in H2-2026. It's an inference-only part - Intel removed the 3D-graphics and ray-tracing hardware - built from 32 Xe3P cores, 256 XMX matrix engines and 32MB of unified L2.

Its defining choice is LPDDR5X instead of HBM. Intel's own PCIe card ships with 160GB, and ODM partners can build cards with up to 480GB, at 1.5TBps (a 1280-bit bus running LPDDR5X at 9.6Gbps: $1280/8 \times 9.6 = 1536$ GBps). That trades HBM-class bandwidth for large capacity and lower cost, aimed at memory-bound inference: Intel cites up to ~300B parameters on one card and over 1T across an 8-GPU server. It's a 350W air-cooled PCIe 5.0 x16 card.

The Xe3P XMX engines natively support the full precision ladder - FP64, FP32, TF32, FP16, BF16, dual-format FP8, FP4, the OCP MX formats (MXFP4/MXFP8) and int8/int4 - but as of launch the only compute figure Intel has published is **1.3 PFLOPS of dense FP8**; there are no official rates for the other formats, which stay unquantified in the tables (see note 30 of the [TFLOPS table](accelerators.md#tflops-comparison-table)). Sources: [Intel newsroom](https://www.intel.com/content/www/us/en/newsroom/news/artificial-intelligence/intel-accelerates-open-infrastructure-agentic-ai-era.html), [Intel GPU D270 product page](https://www.intel.com/content/www/us/en/products/details/discrete-gpus/data-center-gpu/d-series.html), [OCP announcement](https://newsroom.intel.com/artificial-intelligence/intel-to-expand-ai-accelerator-portfolio-with-new-gpu), [Hot Chips 2026](https://newsroom.intel.com/client-computing/intel-outlines-architectures-for-agentic-ai-at-hot-chips-2026).

### AWS Trainium

AWS-specific vocabulary:

- **NeuronCore** - the compute unit. A Trainium2 chip holds "eight NeuronCore-V3 cores".
- **NeuronLink** - the chip-to-chip interconnect. Trainium2 uses "NeuronLink-v3 for chip-to-chip interconnect provides 1.28 TB/sec bandwidth per chip".
- **CC-Cores** - dedicated collective-communication engines; "16 CC-Cores orchestrate collective communication among Trainium2 chips within and across instances".

A `trn2.48xlarge` instance holds 16 Trainium2 chips, and an UltraServer extends NeuronLink across "64 Trainium2 chips across four Trn2 instances". Memory pooling spans "Up to 64 chips", against 16 for the previous generation.

You program it through the Neuron SDK and PyTorch XLA rather than CUDA, which is the main porting cost. AWS publishes no per-link rate or link count, only the 1.28TBps aggregate, so peer-to-peer bandwidth can't be derived (see [Interconnects](../network/interconnects.md) for the intra-node bandwidth tables).

Specs: [Trainium2](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium2.html), [Trn2 instances](https://aws.amazon.com/ec2/instance-types/trn2/).

### Google TPU

TPU-specific vocabulary:

- **TensorCore and SparseCore** - the compute units. A TPU7x chip has "2 TensorCores and 4 SparseCores", split across two chiplets joined by a die-to-die interface.
- **ICI** (inter-chip interconnect) - the scale-up fabric, a **3D torus**: chips laid out on a 3D grid, each wired to its neighbors along every axis, with the edges wrapping around to the opposite side, so each chip has "a direct connection to the nearest neighboring chips in 3 dimensions".
- **Slice, cube, pod** - the units of allocation. A slice is "a collection of chips all located inside the same TPU Pod connected by high-speed inter-chip interconnects (ICI)".

TPU7x (Ironwood) per chip: 192GB HBM at "approximately 7.37 TB/s", 2307 TFLOPS bf16 and 4614 TFLOPS fp8, and a "Bidirectional inter-chip interconnect (ICI) bandwidth per chip (GBps)" of 1200. The torus carries "bi-directional bandwidth of 200 GBps per axis", which is consistent with 1200 in total across six neighbors (the last line of `vendor_math.py`). A VM is always 4 chips, and a pod reaches "a 9,216-chip footprint".

The catch isn't performance, it's lock-in: TPUs are rent-only on GCP and the software path is XLA, so a codebase that relies on custom CUDA kernels doesn't move over for free.

Specs: [TPU7x](https://docs.cloud.google.com/tpu/docs/tpu7x), [TPU system architecture](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm).

### Huawei Ascend

The one vendor here that most readers can't buy from, and the one with the least accessible documentation - but it ships at scale, so it belongs here for context if nothing else.

Huawei-specific vocabulary:

- **Ascend** - the accelerator line; **Atlas** - the servers and rack-scale SuperPoDs built from them.
- **UB Link** (UnifiedBus) - the scale-up interconnect, Huawei's answer to NVLink and UALink (all covered in [Interconnects](../network/interconnects.md)).
- **HiF8** - Huawei's own 8-bit float format, quoted alongside the OCP `mxFP8` standard rather than instead of it.
- **CANN** - the compute architecture and toolchain, with MindSpore as the first-party framework and PyTorch/TensorFlow adaptation layers alongside it.

As of 2026-08 Huawei's Chinese pages carry the full specification table where the English ones carry almost nothing, so use those. For the Atlas 950 SuperPoD the Chinese page gives up to 1024 Ascend 950DT accelerators, on-chip memory of up to 1024 × 96GB at a bandwidth of up to 4.0TBps, AI compute of up to 1 EFLOPS mxFP8/FP8/HiF8 and 2 EFLOPS mxFP4, and a per-cabinet total interconnect bandwidth of up to 64 × 1.68TBps bidirectional.

Divided over the 1024 accelerators (the Huawei lines of `vendor_math.py` [above](#intel-gaudi)), that's roughly 977 TFLOPS mxFP8 and 1953 TFLOPS mxFP4 per Ascend 950DT, with 96GB at 4.0TBps and 840GBps of unidirectional UB Link scale-up bandwidth each - which would place it just below NVLink 5. This division is safe rather than inferred: the accelerator count is stated, and it cross-checks twice - 1024 accelerators across the stated 16 compute cabinets is 64 per cabinet, exactly as the bandwidth line says, and 1024 × 1.68TBps is the 1.72PBps total the same page claims.

Two things to be careful about:

- The TFLOPS and memory comparison-table rows for Ascend 950DT use the **64-card** Atlas 950 SuperPoD product-spec table on the English Ascend Community page (`114.1 / 58.8 / 31.1` PFLOPS at mxFP4 / mxFP8-HiF8 / FP16-BF16, and `64 × 96GB` on-chip memory at 4.0TBps), divided per accelerator - not the 1024-card SuperPoD totals on the Chinese page (~1953 / 977 mxFP4/mxFP8, no bf16). The two configurations aren't simply the same part scaled; don't mix figures between them. The memory type is `?` in the table because Huawei publishes "On-chip Memory" (片上内存) without an HBM generation.
- `HiF8` is quoted alongside `mxFP8` as one published rate, so the `fp8` column carries that pair. HiF8 itself is tapered-precision ([arXiv:2409.16626](https://arxiv.org/abs/2409.16626)), not OCP `E4M3`/`E5M2`, so the usual fp8 exponent/mantissa reading doesn't apply to it.

Specs: [Ascend Community](https://www.hiascend.com/en). Be aware that both Huawei sites are JavaScript applications that serve only a page shell to a plain downloader: `e.huawei.com` returns 255KB of HTML containing not one occurrence of `PFLOPS`, `TB/s` or even `Ascend`. The Ascend Community site happens to inline enough text to read the Atlas 950 spec off; the product pages don't. So these figures are harder to confirm than any other vendor's here, and a second-hand UB Link or Ascend number should be treated as unverified until traced back to Huawei.

### Cerebras

The outlier in approach: rather than connecting many chips, put the whole cluster on one wafer. The WSE-3 measures "46,225 mm2 and containing 4 trillion transistors", delivers compute through "900,000 AI-optimized cores", and is rated at "125 petaflops of AI compute". For scale, an H100 die is about 814 mm² - the WSE-3 is about 57 of them.

Wafer scale trades a familiar problem for an unfamiliar one. There's no intra-node fabric to benchmark and no model-parallel partitioning across accelerators, because it's one device - but you also can't buy 8 of them in a node the way the rest of this chapter assumes, and the software stack is entirely its own. Manufacturing yield (a wafer that big always has defects) is handled architecturally rather than avoided, "with redundant compute cores, redundant routing, and a fail-in-place architecture".

On-chip SRAM capacity and memory bandwidth aren't published on the chip page, despite a heading promising them, so the two figures that matter most for a training workload have to come from the datasheet.

Specs: [Cerebras WSE](https://www.cerebras.ai/chip).

### SambaNova

Inference-first, and explicitly so - both the vocabulary and the product line are organized around serving rather than training.

SambaNova-specific vocabulary:

- **RDU** (Reconfigurable Dataflow Unit) - the accelerator. The "fourth-generation RDU SN40" and "fifth-generation SN50" are the current parts.
- **Dataflow architecture** - the execution model: the chip is reconfigured around a model's graph instead of dispatching kernels one after another.
- **SambaRack** - the rack; "The combination of 16 SN40L RDUs creates a single, high-performance rack".
- **Three-tier memory** - HBM plus large-capacity memory plus SRAM, so that "Models residing in HBM and SRAM can be hot swapped in milliseconds".

SN50 was announced in February 2026 as the fifth generation and "will start shipping to customers in the second half of 2026", so SN40L is what you can actually deploy today. A SambaRack SN50 combines 16 chips for "five times more compute per accelerator and four times more network bandwidth", scales "up to 256 chips across multiple racks", and targets models "up to 10 trillion parameters in size" at context lengths "up to 10 million tokens".

The three-tier memory and millisecond model swapping are the genuinely different idea here: they're aimed at serving many models from one machine, a different problem from the one the rest of this chapter optimizes for. Don't read SambaNova's throughput comparisons against B200 as training numbers; they're agentic-inference numbers.

Specs: [RDU AI chips](https://sambanova.ai/products/rdu-ai-chips), [SambaRack](https://sambanova.ai/products/sambarack).



## Software stacks

<a id="api"></a>

Which software do you need to use each accelerator? This matters as much as the hardware: a chip whose software you can't run your model on is worth nothing to you.

- **NVIDIA** GPUs run on [CUDA](https://developer.nvidia.com/cuda/toolkit). Installing and versioning the driver, toolkit and NCCL is covered in [The GPU node software stack](gpu-software-stack.md).
- **AMD** GPUs run on [ROCm](https://www.amd.com/en/products/software/rocm.html). PyTorch intentionally reuses the same `torch.cuda` Python API on ROCm, so many programs that use only PyTorch APIs run unchanged on supported AMD GPUs such as MI250 and MI300X. Software that includes custom CUDA or PTX code, NVIDIA-specific libraries, or unsupported operations may still need porting or replacement, and validation. See [AMD / ROCm equivalents](gpu-software-stack-reference.md#amd--rocm-equivalents).
- **Intel Gaudi** is programmed via the [Habana SynapseAI SDK](https://habana.ai/intel-gaudi-software/), which supports PyTorch and TensorFlow. A useful integration is [HF Optimum Habana](https://github.com/huggingface/optimum-habana), which also includes a [DeepSpeed](https://github.com/deepspeedai/DeepSpeed) integration.
- **Google TPU** and **AWS Trainium** go through XLA: JAX or PyTorch/XLA on TPUs, the Neuron SDK with PyTorch XLA on Trainium (see their sections above).
- **Huawei Ascend** uses CANN with MindSpore, plus PyTorch/TensorFlow adaptation layers.

Here's how the "same API" works in practice on ROCm. Code written against `torch.cuda` runs as-is, and you can tell which backend you're on from the version fields **[GPU, untested]**:

```python
import torch
print(torch.cuda.is_available())            # True on both NVIDIA and AMD builds
print(torch.version.cuda, torch.version.hip) # e.g. '13.0' None on NVIDIA, None '7.0...' on ROCm
x = torch.randn(4096, 4096, device="cuda")   # "cuda" means "the GPU" on a ROCm build too
```



## AMD performance tuning and profiling

AMD's own guides are the starting point for getting performance out of MI300-series systems:

- [AMD Instinct MI300X system optimization](https://instinct.docs.amd.com/projects/system-acceptance/en/latest/gpus/mi300x.html) - BIOS, OS and kernel settings, including the `numa_balancing` advice used in [Architecture-specific setup](#architecture-specific-setup).
- [AMD Instinct MI300X workload optimization](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/workload-optimization.html) - framework and library tuning, including PyTorch TunableOp; it's also the cache-specification source cited in [Caches](#caches).

The software point from the [MAMF table](accelerators.md#mamf-and-msmf-comparison-table) is worth repeating here: on MI300X the same matmul went from 450 to roughly 750 TFLOPS between ROCm 6.1 and 6.2, and a newer stack can move the best shape. On AMD more than anywhere, re-measure after every stack upgrade.

### Profilers

[omniperf](https://github.com/ROCm/rocprofiler-compute) - "Advanced Profiling and Analytics for AMD Hardware" - can, among many other things, plot a roofline of your AMD accelerator with your kernels placed on it. It has since been renamed the **ROCm Compute Profiler** (`rocprof-compute`), which is the name the link now lands on. It plays the role [Nsight Compute](accelerator/execution-model.md#nsight-compute-why-is-this-kernel-slow) plays on NVIDIA: per-kernel analysis of why a kernel is slow. For how to read a roofline plot, see [Arithmetic intensity and the roofline model](../foundations/computer-architecture.md#arithmetic-intensity-and-the-roofline-model) and the next chapter's [Roofline per op](accelerator/execution-model.md#roofline-per-op).

A typical session looks like this **[AMD GPU, untested; check `rocprof-compute --help` for your ROCm version, as the subcommands have changed across releases]**:

```bash
rocprof-compute profile -n my_run -- python train_step.py   # collect counters for every kernel
rocprof-compute analyze -p workloads/my_run/<gpu-name>      # per-kernel report, incl. the roofline
```



## Apples-to-apples comparison

It's very hard to compare the specs of different offerings, because pretty much all competitors use marketing tricks that make two spec sheets impossible to compare directly: sparse vs dense TFLOPS, rounded or unachievable clocks, per-package vs per-chip numbers, GB vs GiB, bidirectional vs unidirectional bandwidth. This chapter's defenses, in one list:

- Use dense TFLOPS, per chip, and check them with the [formula](accelerators.md#how-to-calculate-theoretical-tflops).
- Compare measured MSMF, not spec TFLOPS, and measure it on the node you'd actually get, with all GPUs busy.
- Measure with the shapes and dtypes your model uses, using the same software stack you'll run.
- Check memory capacity and bandwidth against your model's needs, not against each other.

[MLPerf](https://mlcommons.org/), run by MLCommons, publishes hardware benchmarks for training, inference, storage and other tasks. The round numbers advance a few times a year, so go to the suite pages rather than a pinned version: [training](https://mlcommons.org/benchmarks/training/) and [inference: datacenter](https://mlcommons.org/benchmarks/inference-datacenter/) each show the latest results.

Stas Bekman's verdict on MLPerf is blunt: he has no idea how to make use of it. It's close to impossible to make sense of, or control, the view - a great intention lost in over-engineering, without thinking about how the user will benefit. For example, if you don't care about computer-vision results and only want to quickly see the LLM rows, you can't. And the comparisons still aren't apples to apples, so it's hard to tell which hardware is better. Treat MLPerf as a source of "this system can do at least X on this benchmark with an expert team tuning it", not as a buying guide.



## How to get the best price

- **Advertised prices are almost always negotiable**, if you're willing to buy or rent in bulk, or to rent for 1-3 years. The price you end up paying can be many times lower than the advertised "public" one. Some clouds already show the discount for a longer commitment on their website, but it's always best to negotiate directly with their sales team. Instead of, or on top of, a \$\$ discount you may be offered useful features or upgrades for free.
- **If your company has venture-capital investors, mention it.** The provider then knows you're likely to buy more compute later and is more likely to discount.
- **Tier 2 clouds are likely to be cheaper than Tier 1** (as of 2026-08, Tier 1 is AWS, OCI, Azure and GCP).
- **For baseline prices**, several sites publish up-to-date public price comparisons across clouds - search for something like [cloud gpu pricing comparison](https://www.google.com/search?q=cloud+gpu+pricing+comparison). Good starting points are [vast.ai](https://cloud.vast.ai/create/), and specifically for clusters [gpulist.ai](https://gpulist.ai). Lantern Works tracks [on-demand GPU cloud rental prices](https://thelanternworks.com/gpu/price-comparison/) across multiple providers.
- **Price per useful FLOP, not per GPU-hour.** Dividing the hourly price by the measured MSMF (not the spec TFLOPS) gives a comparable cost per sustained PFLOP-hour. A GPU that's 30% cheaper but sustains 40% less is the more expensive one.

And remember [the most important thing](accelerators.md#the-most-important-thing-to-understand): renting the most powerful accelerator isn't enough. You also need fast intra-node and inter-node connectivity (see [Interconnects](../network/interconnects.md)) and sufficiently fast [storage](../foundations/storage-and-data.md); without them the expensive accelerators idle waiting for data, and you waste a lot of money and time. [Getting compute](../platform/getting-compute.md) covers choosing a provider in depth.

Parts of this chapter are adapted from Stas Bekman's [Machine Learning Engineering Open Book](https://github.com/stas00/ml-engineering) (CC BY-SA 4.0).

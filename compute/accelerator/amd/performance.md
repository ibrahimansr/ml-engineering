# AMD GPUs Performance

As I haven't had a chance to do any serious work with AMD GPUs, just sharing links for now.

- [AMD Instinct MI300X system optimization](https://instinct.docs.amd.com/projects/system-acceptance/en/latest/gpus/mi300x.html)
- [AMD Instinct MI300X workload optimization](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/workload-optimization.html)

## Profilers

[omniperf](https://github.com/ROCm/rocprofiler-compute) - Advanced Profiling and Analytics for AMD Hardware - e.g. can plot a roofline performance of your AMD accelerator and many other things. It has since been renamed ROCm Compute Profiler (`rocprof-compute`), which is the name the link now lands on; it plays the role [Nsight Compute](../execution-model.md#nsight-compute-why-is-this-kernel-slow) plays on NVIDIA. For how to read a roofline plot see [Arithmetic intensity and the roofline model](../../../foundations/computer-architecture.md#arithmetic-intensity-and-the-roofline-model).

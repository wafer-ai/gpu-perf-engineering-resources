<p align="center">
  <img src="cover.avif" alt="AI Performance Engineering" width="100%">
</p>

# AI Performance Engineering

A curated resource list for learning GPU performance engineering and production inference.

The list is ordered from a single inference request to a single GPU, optimized kernels, inference engines, and distributed systems. Read **Start here** first. After that, use it as a reference.

The core list uses original papers, official specifications and documentation, creator repositories, and direct implementation work.

If you work on these problems, [Wafer is hiring](https://wafer.ai).

## Contents

- [Start here: the minimum mental model](#start-here-the-minimum-mental-model)
- [1. GPU fundamentals](#1-gpu-fundamentals)
  - [Programming model](#programming-model)
  - [Compilation and machine code](#compilation-and-machine-code)
- [2. Kernel optimization](#2-kernel-optimization)
  - [Foundational kernel exercises](#foundational-kernel-exercises)
  - [Matrix multiplication](#matrix-multiplication)
  - [Direct implementation work](#direct-implementation-work)
  - [Tensor cores and low precision](#tensor-cores-and-low-precision)
  - [Attention](#attention)
- [3. Programming models and profiling](#3-programming-models-and-profiling)
  - [Triton](#triton)
  - [CUTLASS, CuTe, and CUDA Tile](#cutlass-cute-and-cuda-tile)
  - [Other hardware stacks](#other-hardware-stacks)
  - [Profiling, benchmarking, and correctness](#profiling-benchmarking-and-correctness)
- [4. Inference engines](#4-inference-engines)
  - [Scheduling and continuous batching](#scheduling-and-continuous-batching)
  - [KV cache systems](#kv-cache-systems)
  - [Quantization](#quantization)
  - [Speculative decoding](#speculative-decoding)
  - [Structured decoding and fairness](#structured-decoding-and-fairness)
  - [Long context and multimodal inference](#long-context-and-multimodal-inference)
- [5. Distributed inference](#5-distributed-inference)
  - [Parallelism, collectives, and topology](#parallelism-collectives-and-topology)
  - [Mixture-of-experts serving](#mixture-of-experts-serving)
  - [Prefill and decode disaggregation](#prefill-and-decode-disaggregation)
  - [Production systems](#production-systems)
  - [Serving benchmarks](#serving-benchmarks)
- [6. Current hardware](#6-current-hardware)
  - [NVIDIA](#nvidia)
  - [AMD](#amd)
  - [Google TPU](#google-tpu)
  - [AWS Trainium](#aws-trainium)
- [Frontier](#frontier)
  - [AI-generated kernels](#ai-generated-kernels)
  - [Watchlist](#watchlist)
- [Source policy](#source-policy)

## Start here: the minimum mental model

Read these in order if you are new to the field.

1. [How to Scale Your Model: Inference](https://jax-ml.github.io/scaling-book/inference/) - One request from prefill through decode, with batching, KV memory, and parallelism.
2. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - The transformer computation that the rest of the list optimizes.
3. [CUDA C++ basics](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html) - The shortest official introduction to the CUDA execution model.
4. [Programming Massively Parallel Processors](https://www.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0) - The main textbook for GPU programming, memory, and kernel design.
5. [Roofline: An Insightful Visual Performance Model](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html) - The compute, memory-bandwidth, and arithmetic-intensity model.
6. [Transformer Inference Arithmetic](https://kipply.github.io/blog/transformer-inference-arithmetic/) - FLOPs, parameter bytes, KV bytes, and communication for transformer inference.
7. [Efficiently Scaling Transformer Inference](https://proceedings.mlsys.org/paper_files/paper/2023/file/c4be71ab8d24cdfb45e3d06dbfca2780-Paper-mlsys2023.pdf) - Latency, memory, and parallelism costs for large-model inference.
8. [Etalon](https://arxiv.org/html/2407.07000) - TTFT, TPOT, goodput, and latency SLOs for generative-model serving.

For a practical companion, use the [GPU Mode lectures](https://github.com/gpu-mode/lectures).

## 1. GPU fundamentals

### Programming model

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) - The normative CUDA reference.
- [CUDA programming model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html) - Threads, warps, blocks, grids, and the memory hierarchy.
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) - Coalescing, shared memory, occupancy, synchronization, and optimization workflow.
- [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/) - TMA, thread-block clusters, asynchronous execution, and Hopper-specific limits.
- [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/) - Tensor memory, Blackwell execution features, and architecture limits.

### Compilation and machine code

- [NVCC Compiler Driver](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/) - The CUDA compilation trajectory and artifact controls.
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) - NVIDIA's virtual instruction set and memory model.
- [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/) - `cuobjdump` and `nvdisasm` for inspecting GPU binaries.
- [Understanding PTX](https://developer.nvidia.com/blog/understanding-ptx-the-assembly-language-of-cuda-gpu-computing/) - NVIDIA's introduction to the role of PTX between CUDA and machine code.

## 2. Kernel optimization

### Foundational kernel exercises

- [Efficient Matrix Transpose in CUDA C/C++](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/) - Coalescing, shared-memory tiling, and bank conflicts.
- [Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf) - Synchronization, divergence, occupancy, and instruction cost.
- [Single-pass Parallel Prefix Scan with Decoupled Look-back](https://research.nvidia.com/sites/default/files/pubs/2016-03_Single-pass-Parallel-Prefix/nvr-2016-002.pdf) - A work-efficient scan with one pass over memory.
- [Online Normalizer Calculation for Softmax](https://arxiv.org/abs/1805.02867) - Numerically stable online softmax without materialized intermediates.

### Matrix multiplication

- [Benchmarking GPUs to Tune Dense Linear Algebra](https://mc.stanford.edu/cgi-bin/images/6/65/SC08_Volkov_GPU.pdf) - The canonical case for reasoning from measured hardware behavior instead of occupancy alone.
- [CuTe GEMM tutorial](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html) - Tiling, layouts, copies, and matrix-multiply atoms.
- [CUTLASS 3.x design](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cutlass_3x_design.html) - The collective and kernel structure used by modern CUTLASS.
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) - A compact production FP8 GEMM implementation for Hopper.

### Direct implementation work

- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM) - A matrix multiplication built from naive CUDA through shared-memory and register tiling.
- [Inside NVIDIA GPUs: Anatomy of High-Performance Matmul Kernels](https://www.aleksagordic.com/blog/matmul) - Layouts, tiling, PTX, machine code, and roofline analysis.
- [Outperforming cuBLAS on H100: A Worklog](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog) - A direct Hopper optimization worklog using tensor cores and asynchronous movement.
- [CUTLASS Tutorial: Mastering TMA](https://research.colfax-intl.com/tutorial-hopper-tma/) - Working kernels built around the Tensor Memory Accelerator.

### Tensor cores and low precision

- [OCP 8-bit Floating Point Specification](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-1-final-pdf) - E4M3 and E5M2 formats.
- [OCP Microscaling Formats Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) - Shared-scale MX formats.
- [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine) - FP8 and FP4 transformer execution with scaling controls.
- [Blackwell matrix multiply instructions](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/mma_docs/tcgen05_programming.html) - `tcgen05`, tensor memory, and Blackwell MMA programming.

### Attention

- [FlashAttention](https://arxiv.org/abs/2205.14135) - IO-aware exact attention.
- [FlashAttention-2](https://arxiv.org/abs/2307.08691) - Better work partitioning and parallelism.
- [FlashAttention-3](https://arxiv.org/abs/2407.08608) - Asynchronous movement and tensor-core overlap on Hopper.
- [FlashAttention-4](https://proceedings.mlsys.org/paper_files/paper/2026/file/ae8b0b5838ba510daff1198474e7b984-Paper-Conference.pdf) - The Blackwell attention schedule.
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) - Attention and related kernels for serving workloads.

## 3. Programming models and profiling

### Triton

- [Triton paper](https://eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf) - The original blocked-program language and compiler design.
- [Triton programming guide](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html) - The official programming model.
- [Triton repository](https://github.com/triton-lang/triton) - Compiler, examples, tests, and backend implementation.

### CUTLASS, CuTe, and CUDA Tile

- [CuTe layout algebra](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html) - Layouts and layout composition.
- [CUTLASS GEMM tutorial](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html) - A GEMM expressed through CuTe layouts and atoms.
- [CUTLASS pipeline documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/pipeline.html) - Producer-consumer pipelines and asynchronous stages.
- [CUDA Tile IR programming model](https://docs.nvidia.com/cuda/tile-ir/latest/sections/prog_model.html) - NVIDIA's compiler-owned tile abstraction.
- [CUDA Tile repository](https://github.com/NVIDIA/cuda-tile) - The current implementation and examples.

### Other hardware stacks

- [ROCm Composable Kernel](https://github.com/ROCm/composable_kernel) - AMD tiling, layout, and operator primitives.
- [ROCm AITER](https://github.com/ROCm/aiter) - AMD inference and transformer operator implementations.
- [HipKittens](https://github.com/HazyResearch/HipKittens) - A tile abstraction for AMD GPUs.
- [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html) - The JAX kernel model for GPU and TPU backends.
- [NKI programming model](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/programming_model.html) - The tile-level programming model for AWS NeuronCore hardware.

### Profiling, benchmarking, and correctness

- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/) - System timelines, CPU-GPU interaction, and distributed traces.
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/) - Kernel metrics, sections, replay, and roofline analysis.
- [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/) - Memory, race, initialization, and synchronization checks.
- [CUTLASS GEMM measurement methodology](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_performance_measurement_methodology_guidelines.html) - Reproducible GEMM benchmarking.
- [ROCm Compute Profiler](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/) - AMD performance counters and roofline analysis.

## 4. Inference engines

### Scheduling and continuous batching

- [Orca](https://www.usenix.org/conference/osdi22/presentation/yu) - Iteration-level scheduling for autoregressive serving.
- [PagedAttention and vLLM](https://arxiv.org/html/2309.06180) - Paged KV allocation and continuous batching.
- [Sarathi-Serve](https://www.usenix.org/system/files/osdi24-agrawal.pdf) - Chunked prefills that reduce interference with decode.
- [SGLang](https://arxiv.org/html/2312.07104) - Prefix reuse, structured programs, and a serving runtime.
- [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang), and [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) - The main production engine implementations.

### KV cache systems

- [Grouped-Query Attention](https://arxiv.org/abs/2305.13245) - Fewer key-value heads and a smaller KV cache.
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434) - Multi-head latent attention and compressed KV state.
- [KIVI](https://proceedings.mlr.press/v235/liu24bz.html) - KV quantization with separate treatment for keys and values.
- [CacheGen](https://cs.stanford.edu/~keithw/sigcomm2024/sigcomm24-final1571-acmpaginated.pdf) - KV compression for transfer.
- [Mooncake](https://www.usenix.org/conference/fast25/presentation/qin) - A distributed KV cache and data plane.

### Quantization

- [GPTQ](https://arxiv.org/abs/2210.17323) - One-shot second-order weight quantization.
- [SmoothQuant](https://proceedings.mlr.press/v202/xiao23c.html) - W8A8 execution by moving quantization difficulty from activations into weights.
- [AWQ](https://proceedings.mlsys.org/paper_files/paper/2024/file/42a452cbafa9dd64e9ba4aa95cc1ef21-Paper-Conference.pdf) - Low-bit weight-only inference with salient-weight protection.

### Speculative decoding

- [Fast Inference from Transformers via Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html) - Exact sampling with a draft model.
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) - The parallel formulation and analysis.
- [Medusa](https://arxiv.org/html/2401.10774) - Multiple prediction heads on the target model.
- [EAGLE](https://proceedings.mlr.press/v235/li24bt.html) - Feature-level drafting.

### Structured decoding and fairness

- [Guiding LLMs the Right Way](https://proceedings.mlr.press/v235/beurer-kellner24a.html) - Constrained decoding without changing the intended token distribution.
- [XGrammar](https://proceedings.mlsys.org/paper_files/paper/2025/file/5c20ca4b0b20b0bd2f1d839dc605e70f-Paper-Conference.pdf) - A fast grammar engine for structured generation.
- [Fairness in Serving Large Language Models](https://arxiv.org/html/2401.00588) - Fair scheduling when request sizes are different and unknown.

### Long context and multimodal inference

- [Ring Attention](https://arxiv.org/abs/2310.01889) - Exact distributed attention by circulating KV blocks around a device ring.
- [MInference 1.0](https://arxiv.org/abs/2407.02490) - Dynamic sparse patterns for long-context prefill on existing models.
- [Native Sparse Attention](https://arxiv.org/abs/2502.11089) - A model trained with a hardware-aligned sparse attention hierarchy.
- [vLLM multimodal inputs](https://docs.vllm.ai/en/latest/features/multimodal_inputs.html) - Current engine support for text, image, audio, and video inputs.

## 5. Distributed inference

### Parallelism, collectives, and topology

- [Megatron-LM](https://arxiv.org/abs/1909.08053) - Tensor and pipeline parallelism for transformer models.
- [NCCL](https://github.com/NVIDIA/nccl) - NVIDIA's collective communication implementation.
- [Multi-node NVLink Systems Tuning Guide](https://docs.nvidia.com/multi-node-nvlink-systems/multi-node-tuning-guide/) - NVLink and InfiniBand topology in GB200 NVL systems.
- [UALink 1.0 Specification](https://ualinkconsortium.org/wp-content/uploads/2025/04/UALink200_Specification_v1.0_Evaluation_Copy.pdf) - An open scale-up interconnect.
- [Ultra Ethernet 1.0.3 Specification](https://ultraethernet.org/wp-content/uploads/sites/20/2026/08/UE-Specification-1.0.3.pdf) - The scale-out transport specification.

### Mixture-of-experts serving

- [DeepSeek-V3](https://arxiv.org/html/2412.19437) - Routed experts, shared experts, and the model-system design.
- [DeepEP](https://github.com/deepseek-ai/DeepEP) - Expert dispatch and combine kernels.
- [EPLB](https://github.com/deepseek-ai/EPLB) - Expert placement and replication from measured load.
- [MegaScale-Infer](https://arxiv.org/abs/2504.02263) - Large-scale MoE inference and communication overlap.

### Prefill and decode disaggregation

- [DistServe](https://arxiv.org/html/2401.09670) - Separate prefill and decode workers optimized for goodput under latency constraints.
- [Splitwise](https://www.microsoft.com/en-us/research/publication/splitwise-efficient-generative-llm-inference-using-phase-splitting/) - Phase-specific allocation and scheduling.
- [Mooncake](https://www.usenix.org/conference/fast25/presentation/qin) - KV-centric disaggregated inference.
- [NIXL](https://github.com/ai-dynamo/nixl) - A transport layer for moving inference state across memory and network backends.
- [Dynamo disaggregated serving](https://docs.nvidia.com/dynamo/design-docs/disaggregated-serving.md) - A current production implementation.

### Production systems

- [Clockwork](https://www.usenix.org/conference/osdi20/presentation/gujarati) - Predictable model serving through centralized scheduling.
- [ServerlessLLM](https://www.usenix.org/conference/osdi24/presentation/fu) - Faster model startup and live migration.
- [Gateway API Inference Extension](https://gateway-api-inference-extension.sigs.k8s.io/) - Model, accelerator, and KV-aware request routing.
- [llm-d](https://github.com/llm-d/llm-d) - Distributed routing, scheduling, and disaggregated serving on Kubernetes.

### Serving benchmarks

- [MLPerf Inference](https://www.cs.toronto.edu/ecosystem/papers/ISCA_20/MLPerf%20Inference.pdf) - Reproducible benchmark scenarios and load generation.
- [Etalon](https://arxiv.org/html/2407.07000) - Goodput under per-request latency SLOs.
- [ServeGen](https://www.usenix.org/system/files/nsdi26-xiang-servegen.pdf) - Workload generation that preserves important production-trace properties.
- [BurstGPT](https://github.com/HPMLL/BurstGPT) - A public trace for bursty LLM workloads.
- [MLPerf Endpoints](https://mlcommons.org/benchmarks/endpoints/) - An endpoint-level benchmark for interactive generative AI.

## 6. Current hardware

Read each architecture with its ISA or tuning guide. Vendor peak numbers are not performance measurements.

### NVIDIA

- [Blackwell architecture brief](https://resources.nvidia.com/en-us-blackwell-architecture/blackwell-architecture-technical-brief) - Blackwell and Blackwell Ultra system architecture.
- [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/) - Programming and optimization guidance.
- [CUTLASS Blackwell documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell.html) - Blackwell matrix multiply and data-movement support.

### AMD

- [CDNA 4 architecture whitepaper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf) - MI350 compute, memory, and chiplet architecture.
- [CDNA 4 instruction set](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf) - The native machine instruction reference.
- [MI350 performance counters](https://rocm.docs.amd.com/en/latest/reference/gpu-arch/mi350-performance-counters.html) - Counter definitions and measurement guidance.

### Google TPU

- [TPU v1 analysis](https://research.google/pubs/in-datacenter-performance-analysis-of-a-tensor-processing-unit/) - The original datacenter TPU paper.
- [TPU v4](https://arxiv.org/abs/2304.01433) - The TPU v4 chip, interconnect, and system.
- [Ironwood documentation](https://docs.cloud.google.com/tpu/docs/tpu7x) - Current TPU v7 architecture and configuration.
- [Pallas TPU hardware model](https://docs.jax.dev/en/latest/pallas/tpu/hardware.html) - The TPU execution and memory model for kernel authors.

### AWS Trainium

- [Trainium and Inferentia2 architecture](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/architecture/trainium_inferentia2_arch.html) - NeuronCore v2 compute and memory architecture.
- [Trainium3 architecture](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/architecture/trainium3_arch.html) - The current NeuronCore architecture.
- [NKI performance guide](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.24.0/general/nki/nki_perf_guide.html) - Kernel optimization for Trainium and Inferentia.

## Frontier

Verified on **2026-08-23**. This section is kept separate from the core list because the evidence changes quickly.

### AI-generated kernels

- [KernelBench](https://proceedings.mlr.press/v267/ouyang25a.html) - The original benchmark for converting PyTorch operators into faster GPU kernels.
- [KernelBench-Verified](https://arxiv.org/html/2607.16241) - Stronger correctness tests and baseline parity.
- [SOL-ExecBench](https://github.com/nvidia/sol-execbench) - Correctness and performance measured against a hardware speed-of-light model.

### Watchlist

- [NVIDIA Rubin](https://developer.nvidia.com/blog/inside-nvidia-rubin-gpu-architecture-powering-the-era-of-agentic-ai/) and Rubin CPX, pending shipped systems and reproducible measurements.
- AMD MI400, CDNA 5, and Helios, pending architecture and ISA documents.
- Session-aware and agentic scheduling against public production traces.
- Real-time voice and video serving with complete quality and latency metrics.
- Inference ASICs, processing in memory, analog compute, and photonic compute with reproducible deployments.
- Individual AI kernel agents that have not been rerun on a hardened evaluator.

## Source policy

A core source must be one of the following:

- the paper that introduced the mechanism;
- the specification or official documentation that defines it;
- the repository that implements it;
- a direct implementer report with code, measurements, and enough detail to reproduce the result.

Performance claims need the hardware, workload, precision, baseline, and correctness method. Otherwise the number is omitted.

See [CONTRIBUTING.md](CONTRIBUTING.md) before proposing a resource.

## License

MIT

## Maintainer

emilio@wafer.ai

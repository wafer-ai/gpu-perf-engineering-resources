<p align="center">
  <img src="cover.avif" alt="Performance Engineering for AI Infra" width="100%">
</p>

# Learning Guide: Performance Engineering for AI Infra

## Purpose

The purpose of this guide is to help engineers learn GPU kernel programming and optimization, with a focus on high-performance AI systems. It covers the full journey from fundamentals to production deployment, balancing foundational concepts with cutting-edge techniques.

If you're interested in GPU performance engineering - [we're hiring at Wafer](https://wafer.ai).

## How to read

Recommended reading order:

1. Read "Tier 1" for all topics
2. Read "Tier 2" for all topics
3. Etc

## Table of contents

- [Fundamentals](#fundamentals)
  - [Introduction to GPU programming](#introduction-to-gpu-programming)
  - [Architecture deep dives](#architecture-deep-dives)
  - [Low-level details](#low-level-details)
- [Matrix Multiplication](#matrix-multiplication)
  - [Essential tutorials](#essential-tutorials)
  - [Advanced implementations](#advanced-implementations)
  - [cuBLAS internals](#cublas-internals)
- [Tensor Cores & Mixed Precision](#tensor-cores--mixed-precision)
  - [Tensor core fundamentals](#tensor-core-fundamentals)
  - [Precision formats](#precision-formats)
  - [Blackwell-specific](#blackwell-specific)
- [Attention & Memory-Bound Kernels](#attention--memory-bound-kernels)
  - [FlashAttention](#flashattention)
  - [PagedAttention & serving](#pagedattention--serving)
  - [KV cache optimization](#kv-cache-optimization)
- [Compiler & DSL Approaches](#compiler--dsl-approaches)
  - [Triton](#triton)
  - [CUTLASS & CuTe](#cutlass--cute)
  - [Other DSLs](#other-dsls)
- [Profiling & Optimization](#profiling--optimization)
  - [NVIDIA tools](#nvidia-tools)
  - [Optimization techniques](#optimization-techniques)
  - [Advanced topics](#advanced-topics)
- [AMD & Alternative Hardware](#amd--alternative-hardware)
  - [ROCm fundamentals](#rocm-fundamentals)
  - [CDNA architecture](#cdna-architecture)
  - [TPU & others](#tpu--others)
- [Production Inference Systems](#production-inference-systems)
  - [Core systems](#core-systems)
  - [Continuous batching](#continuous-batching)
  - [Speculative decoding](#speculative-decoding)
- [LLM-Generated Kernels](#llm-generated-kernels)
  - [Benchmarks & models](#benchmarks--models)
  - [Agentic approaches](#agentic-approaches)
  - [Research papers](#research-papers)
- [Distributed & Multi-GPU](#distributed--multi-gpu)
  - [Communication primitives](#communication-primitives)
  - [Parallelism strategies](#parallelism-strategies)
  - [Kernel fusion](#kernel-fusion)
- [The Big Picture](#the-big-picture)
  - [Industry analysis](#industry-analysis)
  - [Practitioner blogs](#practitioner-blogs)
  - [Communities](#communities)
- [Maintainer](#maintainer)

## Fundamentals

### Introduction to GPU programming

#### Tier 1

- [Programming Massively Parallel Processors (PMPP)](https://www.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0) - Hwu, Kirk, El Hajj. The canonical textbook, 5th edition covers Ampere/Hopper/Blackwell
- [GPU Mode Lectures](https://github.com/gpu-mode/lectures) - Community-driven lecture series: profiling → kernels → CUTLASS → SASS. Active Discord (23k+ members): [discord.gg/gpumode](https://discord.gg/gpumode)
- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) - Official documentation, essential reference for programming model

### Architecture deep dives

#### Tier 2

- [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) - TMA, Thread Block Clusters, Distributed Shared Memory, WGMMA
- [Chips and Cheese: Blackwell](https://chipsandcheese.com/p/blackwell-nvidias-massive-gpu) - Microbenchmarking analysis of GB202, memory latency comparisons
- [Dissecting the NVIDIA Hopper GPU Architecture](https://arxiv.org/abs/2402.13499) - Academic microbenchmarking of H100
- [Dissecting the NVIDIA Blackwell Architecture](https://arxiv.org/abs/2507.10789) - Microbenchmarks covering tcgen05, TMEM, 2SM MMA

### Low-level details

#### Tier 3

- [PTX ISA Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/) - Official PTX instruction set reference
- [Understanding PTX](https://developer.nvidia.com/blog/understanding-ptx-the-assembly-language-of-cuda-gpu-computing) - Introduction to CUDA's virtual assembly language
- [DocumentSASS](https://github.com/0xD0GF00D/DocumentSASS) - Unofficial SASS instruction documentation extracted from nvdisasm
- [JEB SASS Disassembler](https://www.pnfsoftware.com/blog/reversing-nvidia-cuda-sass-code/) - Reverse engineering GPU binaries (Volta → Blackwell)

## Matrix Multiplication

### Essential tutorials

#### Tier 1

- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM) - siboehm. The canonical starting tutorial. Covers tiling, shared memory, vectorized loads
- [Inside NVIDIA GPUs: Anatomy of High-Performance Matmul Kernels](https://www.aleksagordic.com/blog/matmul) - Aleksa Gordić. 47 figures. Covers PTX/SASS, wave quantization, ILP, roofline model, warp tiling
- [Outperforming cuBLAS on H100: A Worklog](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog) - cudaforfun. Real optimization journey using WGMMA and TMA
- [Fast CUDA GEMM with Tensor Cores](https://github.com/lezcano/gemm) - lezcano. Practical tensor core implementation

### Advanced implementations

#### Tier 2

- [Advanced Matrix Multiplication Optimization](https://salykova.github.io/sgemm-gpu) - salykova. Detailed optimization techniques following CUTLASS approach
- [CUDA Matrix Multiplication Optimization](https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/) - Lei Mao. Systematic optimization progression
- [Optimizing SGEMV for cuBLAS-like Performance](https://maharshi.bearblog.dev/optimizing-sgemv-cuda/) - Maharshi. Matrix-vector multiplication optimization worklog
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) - DeepSeek. Clean FP8 GEMM implementation for Hopper, ~300 lines

### cuBLAS internals

#### Tier 3

- [New cuBLAS 12.0 Features](https://developer.nvidia.com/blog/new-cublas-12-0-features-and-matrix-multiplication-performance-on-nvidia-hopper-gpus/) - Hopper-specific optimizations and performance
- [cuBLAS 12.9 Floating Point Emulation](https://developer.nvidia.com/blog/boosting-matrix-multiplication-speed-and-flexibility-with-nvidia-cublas-12-9/) - FP32 emulation with BF16 tensor cores

## Tensor Cores & Mixed Precision

### Tensor core fundamentals

#### Tier 1

- [NVIDIA Tensor Core Evolution: Volta to Blackwell](https://semianalysis.com/2025/06/23/nvidia-tensor-core-evolution-from-volta-to-blackwell/) - SemiAnalysis. Comprehensive evolution: WMMA → MMA → WGMMA → tcgen05
- [Deep Dive on Hopper TMA Unit for FP8 GEMMs](https://pytorch.org/blog/hopper-tma-unit/) - PyTorch. TMA programming model and FP8 integration
- [CUTLASS Tutorial: Mastering TMA](https://research.colfax-intl.com/tutorial-hopper-tma/) - Colfax Research. Tensor Memory Accelerator programming

### Precision formats

#### Tier 2

- [Introducing FP8 for Efficient AI Training](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/) - NVIDIA. E4M3 vs E5M2 formats, scaling strategies
- [Introducing NVFP4 for Low-Precision Inference](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) - NVIDIA. Blackwell FP4 with microscaling (MXFP4)
- [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine) - Library for FP8/FP4 training and inference
- [Per-Tensor and Per-Block Scaling for FP8](https://developer.nvidia.com/blog/per-tensor-and-per-block-scaling-strategies-for-effective-fp8-training/) - NVIDIA. Scaling strategies for quantization

### Blackwell-specific

#### Tier 3

- [Matrix Multiplication on Blackwell: Part 1](https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-1-introduction) - Modular. tcgen05, TMEM, 2SM MMA programming
- [Blackwell Pipelining with CuTeDSL](https://www.linkedin.com/posts/simon-veitner-174a681b6_blackwell-pipelining-with-cutedsl-activity-7409301467171328000-bTPv) - Simon Veitner. Blog post on advanced Blackwell kernel patterns

## Attention & Memory-Bound Kernels

### FlashAttention

#### Tier 1

- [FlashAttention: Fast and Memory-Efficient Attention](https://arxiv.org/abs/2205.14135) - Dao et al. Original paper: IO-aware exact attention
- [FlashAttention-2](https://arxiv.org/abs/2307.08691) - Dao. Better parallelization, work partitioning
- [FlashAttention-3: Fast and Accurate Attention with Asynchrony](https://arxiv.org/abs/2407.08608) - Dao et al. Hopper-specific: warp specialization, WGMMA pipelining
- [A Case Study in CUDA Kernel Fusion: FlashAttention-2 on Hopper](https://arxiv.org/abs/2312.11918) - Jay Shah et al. CUTLASS implementation details

### PagedAttention & serving

#### Tier 2

- [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) - vLLM team. Virtual memory for KV cache
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) - Kernel library for LLM serving (MLSys 2025 Best Paper). PagedAttention, FlashAttention-3, MLA support
- [Accelerating Self-Attentions with FlashInfer](https://flashinfer.ai/2024/02/02/introduce-flashinfer.html) - Architecture and design decisions

### KV cache optimization

#### Tier 3

- [Mastering LLM Techniques: Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) - NVIDIA. Comprehensive guide: GQA, MQA, KV cache compression
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) - Google. Grouped Query Attention for memory efficiency
- [Multi-Head Latent Attention (MLA)](https://arxiv.org/abs/2405.04434) - DeepSeek. Low-rank KV compression, 8x cache reduction
- [A Survey on LLM Acceleration based on KV Cache Management](https://arxiv.org/abs/2412.19442) - Comprehensive taxonomy of KV cache techniques

## Compiler & DSL Approaches

### Triton

#### Tier 1

- [Introducing Triton](https://openai.com/index/triton) - OpenAI. Original announcement and motivation
- [Triton Language](https://github.com/triton-lang/triton) - Development repository
- [Deep Dive into Triton Internals (Parts 1-3)](https://www.kapilsharma.dev/posts/deep-dive-into-triton-internals/) - Kapil Sharma. Compiler pipeline: Python → MLIR → PTX → CUBIN
- [GPU Mode: Triton Internals Talk](https://www.kapilsharma.dev/posts/gpu-mode-triton-internals-talk/) - Kapil Sharma. Video + slides from the lecture

### CUTLASS & CuTe

#### Tier 2

- [Learn CUTLASS the Hard Way](https://leimao.github.io/article/Learn-CUTLASS-The-Hard-Way/) - Lei Mao. Naive GEMM → real CUTLASS progression
- [CUTLASS Tutorial: GEMM Kernel Design with Pipelining](https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/) - Colfax Research. Warp specialization, producer-consumer patterns
- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass) - CUDA Templates for Linear Algebra Subroutines
- [cuTile (CUDA Tile)](https://github.com/NVIDIA/cutile-python) - New tile-level programming model in CUDA 13.1

### Other DSLs

#### Tier 3

- [TileLang](https://github.com/tile-ai/tilelang) - Composable tiled programming, 1075x speedup over PyTorch on H100
- [ThunderKittens](https://github.com/HazyResearch/ThunderKittens) - Stanford Hazy Research. DSL for writing fast GPU kernels
- [Apache TVM](https://tvm.apache.org/) - End-to-end ML compiler with auto-tuning (Ansor)
- [MLIR GPU Dialect](https://mlir.llvm.org/) - Compiler infrastructure for heterogeneous compute
- [Mojo](https://www.modular.com/mojo) - MLIR-based language targeting GPU/CPU, SIMD-first design

## Profiling & Optimization

### NVIDIA tools

#### Tier 1

- [Nsight Compute Roofline Analysis](https://developer.nvidia.com/blog/accelerating-hpc-applications-with-nsight-compute-roofline-analysis/) - NVIDIA. Roofline modeling for bottleneck analysis
- [CUDA Occupancy Calculator](https://developer.nvidia.com/blog/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/) - NVIDIA. `cudaOccupancyMaxActiveBlocksPerMultiprocessor` API
- [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/) - Official optimization guide for H100

### Optimization techniques

#### Tier 2

- [Memory Coalescing and Bank Conflicts](https://medium.com/@dhanushg295/mastering-cuda-matrix-multiplication-an-introduction-to-shared-memory-tile-memory-coalescing-and-d7979499b9c5) - Shared memory optimization, padding tricks
- [Understanding CUDA Occupancy](https://medium.com/@manisharadwad/unlocking-gpu-potential-understanding-and-optimizing-cuda-occupancy-2f43ee01ad7e) - Thread block configuration
- [The Roofline Model](https://docs.nersc.gov/tools/performance/roofline/) - NERSC. Arithmetic intensity, compute vs memory bound
- [Understanding the Top-K CUDA Kernel with PTX](https://blog.alpindale.net/posts/top_k_cuda/) - alpindale. 10x speedup over torch.topk, PTX-level optimization

### Advanced topics

#### Tier 3

- [CUDA Graphs for Reduced Launch Overhead](https://developer.nvidia.com/blog/cuda-graphs/) - NVIDIA. Batch kernel launches, 5x speedup for small kernels
- [Kernel Batching with CUDA Graphs](https://arxiv.org/abs/2501.09398) - Optimal batch sizes (50-100 nodes), 1.4x improvement
- [Warp Specialization in PyTorch](https://pytorch.org/blog/warp-specialization/) - Producer-consumer patterns, async execution
- [Tawa: Automatic Warp Specialization](https://arxiv.org/abs/2510.14719) - Matches FlashAttention-3 performance with less effort

## AMD & Alternative Hardware

### ROCm fundamentals

#### Tier 1

- [Developing Triton Kernels on AMD GPUs](https://rocm.blogs.amd.com/artificial-intelligence/triton/README.html) - AMD ROCm Blog. Triton for MI300X
- [Triton Kernel Optimizations on AMD](https://rocm.blogs.amd.com/software-tools-optimization/kernel-development-optimizations-with-triton-on-/README.html) - AMD ROCm Blog. Performance tuning for CDNA
- [HipKittens](https://github.com/HazyResearch/HipKittens) - ThunderKittens for AMD. Tile programming abstraction for MI300X

### CDNA architecture

#### Tier 2

- [Chips and Cheese: AMD CDNA 3](https://chipsandcheese.com) - MI300X architecture analysis, chiplet design
- [Chips and Cheese: RDNA 4](https://chipsandcheese.com/p/amds-rdna4-gpu-architecture-at-hot) - Dynamic register allocation, cache strategies
- [AMD RDNA 3 Microbenchmarking](https://chipsandcheese.com/p/microbenchmarking-amds-rdna-3-graphics-architecture) - Chips and Cheese

### TPU & others

#### Tier 3

- [The Rise of Pallas: Custom TPU Kernels](https://towardsdatascience.com/the-rise-of-pallas-unlocking-tpu-potential-with-custom-kernels-67be10ab846a/) - Towards Data Science. JAX Pallas for TPU programming
- [vLLM TPU: Unified JAX Backend](https://blog.vllm.ai/2025/10/16/vllm-tpu.html) - vLLM Blog. 20% throughput improvement via JAX primitives
- [Building Production AI on Cloud TPUs with JAX](https://docs.cloud.google.com/tpu/docs/jax-ai-stack) - Google

## Production Inference Systems

### Core systems

#### Tier 1

- [vLLM](https://github.com/vllm-project/vllm) - PagedAttention, continuous batching, high throughput
- [SGLang](https://github.com/sgl-project/sglang) - RadixAttention, structured generation, prefix caching
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) - NVIDIA's optimized inference library
- [Accelerating Transformers with cuDNN 9](https://developer.nvidia.com/blog/accelerating-transformers-with-nvidia-cudnn-9/) - NVIDIA. Fused attention, Graph API

### Continuous batching

#### Tier 2

- [Orca: Distributed Serving with Iteration-Level Scheduling](https://www.usenix.org/conference/osdi22/presentation/yu) - OSDI 2022. Original continuous batching paper, 36.9x throughput
- [Continuous Batching from First Principles](https://huggingface.co/blog/continuous_batching) - Hugging Face. Clear explanation of dynamic batching
- [Achieve 23x LLM Inference Throughput](https://www.anyscale.com/blog/continuous-batching-llm-inference) - Anyscale. vLLM performance analysis

### Speculative decoding

#### Tier 3

- [Medusa: Simple Framework for Accelerating LLM Generation](https://github.com/FasterDecoding/Medusa) - Multiple heads for parallel draft tokens
- [EAGLE: Speculative Sampling with Draft Model](https://github.com/SafeAILab/EAGLE) - Autoregressive draft prediction
- [Speculative Decoding Overview](https://docs.vllm.ai) - vLLM Docs. Implementation in vLLM

## LLM-Generated Kernels

### Benchmarks & models

#### Tier 1

- [KernelBench: Can LLMs Write Efficient GPU Kernels?](https://arxiv.org/abs/2502.10517) - Stanford. 250 PyTorch workloads, fast_p metric
- [KernelLLM](https://huggingface.co/facebook/KernelLLM) - Meta. 8B model trained on 25k PyTorch→Triton pairs, beats GPT-4o
- [TritonBench](https://arxiv.org/abs/2502.14752) - 184 real-world Triton operators from GitHub

### Agentic approaches

#### Tier 2

- [The AI CUDA Engineer](https://sakana.ai/ai-cuda-engineer/) - Sakana AI. Evolutionary optimization, 10-100x speedups (with caveats about benchmark gaming)
- [AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) - Google DeepMind. 32.5% FlashAttention speedup, 23% GEMM speedup
- [Kevin: Multi-Turn RL for CUDA Kernels](https://arxiv.org/abs/2507.11948) - First multi-turn RL model, 82% correctness (vs 56% base)
- [CUDA-L1: Contrastive RL for CUDA Optimization](https://arxiv.org/abs/2507.14111) - 3.12x average speedup on KernelBench

### Research papers

#### Tier 3

- [EvoEngineer: Automated CUDA Kernel Evolution](https://arxiv.org/abs/2510.03760)
- [QiMeng-Kernel: Macro-Thinking Micro-Coding for GPU Kernels](https://arxiv.org/abs/2511.20100)
- [CUDA-LLM: LLMs Can Write Efficient CUDA Kernels](https://arxiv.org/abs/2506.09092)
- [GEAK: Triton Kernel AI Agent](https://rocm.blogs.amd.com/software-tools-optimization/triton-kernel-ai/README.html) - AMD ROCm. 51% accuracy, 1.81x speedup on MI300X

## Distributed & Multi-GPU

### Communication primitives

#### Tier 1

- [NVIDIA NCCL](https://github.com/NVIDIA/nccl) - Collective communication: all-reduce, all-gather, broadcast
- [Fast Multi-GPU Collectives with NCCL](https://developer.nvidia.com/blog/fast-multi-gpu-collectives-nccl/) - NVIDIA. Ring, tree algorithms, topology-aware optimization
- [Demystifying NCCL](https://arxiv.org/abs/2507.04786) - In-depth analysis of GPU communication protocols
- [Collective Communication for 100k+ GPUs](https://arxiv.org/abs/2510.20171) - Meta NCCLX. Scaling to massive clusters

### Parallelism strategies

#### Tier 2

- [Megatron-LM: Training Multi-Billion Parameter Models](https://arxiv.org/abs/1909.08053) - NVIDIA. Tensor parallelism, pipeline parallelism
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) - Up to 47% MFU on H100 clusters
- [Large Scale Tensor Parallel Training](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html) - PyTorch Tutorial. Native TP support in PyTorch
- [Horovod](https://github.com/horovod/horovod) - Ring-allreduce distributed training, 90% scaling efficiency

### Kernel fusion

#### Tier 3

- [Kernel Fusion in CUDA](https://www.vrushankdes.ai/diffusion-policy-inference-optimization/part-vi---kernel-fusion-in-cuda) - vrushankdes.ai. Vertical vs horizontal fusion, U-Net optimization
- [Automatic Horizontal Fusion for GPU Kernels](https://www.cs.toronto.edu/ecosystem/papers/CGO_22/Horizontal_Fusion.pdf) - CMU. 12-55% speedup via parallel kernel execution

## The Big Picture

### Practitioner blogs

- [Michal Pitr - From Scratch](https://michalpitr.substack.com) - GPU programming, inference optimization
- [cudaforfun Substack](https://cudaforfun.substack.com) - cuBLAS-level kernel development
- [Lei Mao's Log Book](https://leimao.github.io) - CUTLASS, CUDA optimization deep dives
- [Aleksa Gordić's Blog](https://www.aleksagordic.com/blog) - Ex-DeepMind, GPU architecture and matmul

### Communities

- [GPU Mode Discord](https://discord.gg/gpumode) - 23k+ members, weekly lectures, kernel leaderboard
- [GPU Mode Resource Stream](https://github.com/gpu-mode/resource-stream) - Curated CUDA/GPU learning materials

## Contributing

Contributions welcome! Please ensure resources meet our quality criteria:
- Primary sources (papers, official docs)
- Practitioner blogs with real implementation insights
- Active maintenance or timeless fundamentals
- No surface-level tutorials
- No AI-generated content without human verification

## License

MIT

## Maintainer

emilio@wafer.ai

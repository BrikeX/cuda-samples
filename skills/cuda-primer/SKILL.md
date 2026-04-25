---
name: cuda-primer
description: >-
  A progressive CUDA learning path built from 147 samples (118 buildable +
  29 that need fixes) in the current container (CUDA 12.2).  Use when the
  user asks for a study plan, tutorial order, or "where to start" with CUDA.
  For samples needing OpenGL/FreeImage/Vulkan/EGL/MPI, see cuda-advanced.
---

# CUDA Primer — Learning Path

A structured curriculum using the samples in this repository.
Samples marked with a wrench need a fix before they build (see the fix
guide at the bottom).  Even when broken, read the source to learn the concept.

Legend:

- (no mark) — builds successfully
- :wrench: **API mismatch** — needs `Common/nvrtc_helper.h` patch (see Fix Guide)
- :wrench: **CUFFT codes** — needs `Common/helper_cuda.h` patch (see Fix Guide)
- :wrench: **Thrust/CUB** — needs CUDA toolkit upgrade
- :wrench: **CUDA 12.4+** — requires CUDA 12.4 or newer
- :wrench: **nvJitLink** — requires newer CUDA toolkit
- :wrench: **NPP sig** — needs NPP API patch or toolkit upgrade

## How to use

```bash
./scripts/build_samples --update-arch 2>&1 | tee /tmp/build_output.log
python3 scripts/analyze_build.py /tmp/build_output.log --show-success
```

Run any built sample:

```bash
./build/Samples/<category>/<sample>/<sample>
```

---

## Stage 1 — First Contact

Goal: compile, launch, and query a GPU.

| # | Sample | Category | What you learn |
| --- | ------ | -------- | -------------- |
| 1 | `deviceQuery` | 1_Utilities | Runtime API: enumerate GPUs and read device properties |
| 2 | `vectorAdd` | 0_Introduction | Runtime API "hello world": allocate, copy, launch, verify |
| 3 | `simplePrintf` | 0_Introduction | `printf` inside device code for quick debugging |
| 4 | `template` | 0_Introduction | Minimal project skeleton — copy this to start new work |
| 5 | `deviceQueryDrv` | 1_Utilities | Driver API version of device enumeration (compare with #1) |

## Stage 2 — Execution Model

Goal: understand threads, blocks, grids, timing, and occupancy.

| # | Sample | Category | What you learn |
| --- | ------ | -------- | -------------- |
| 6 | `clock` | 0_Introduction | Per-block `clock()` for fine-grained kernel timing |
| 7 | `simpleAssert` | 0_Introduction | `assert()` in device code; how device-side errors surface |
| 8 | `simpleTemplates` | 0_Introduction | Kernel templates and dynamic shared memory with templates |
| 9 | `simpleOccupancy` | 0_Introduction | Occupancy API and launch configurator vs manual tuning |
| 10 | `asyncAPI` | 0_Introduction | CUDA events for timing; overlap CPU and GPU work |
| 11 | `LargeKernelParameter` | 6_Performance | Passing large parameters to kernels |

## Stage 3 — Memory Hierarchy

Goal: master global, shared, texture, surface, pinned, zero-copy, and UM.

| # | Sample | Category | What you learn |
| --- | ------ | -------- | -------------- |
| 12 | `simpleTexture` | 0_Introduction | Basic 1D/2D texture fetches in a kernel |
| 13 | `simplePitchLinearTexture` | 0_Introduction | 2D pitched memory bound as a texture |
| 14 | `simpleCubemapTexture` | 0_Introduction | Cubemap texture fetches |
| 15 | `simpleLayeredTexture` | 0_Introduction | Layered (array) textures |
| 16 | `simpleSurfaceWrite` | 0_Introduction | Read/write surfaces vs read-only textures |
| 17 | `simpleZeroCopy` | 0_Introduction | Mapped pinned memory: GPU reads/writes host memory directly |
| 18 | :wrench: `simpleTextureDrv` | 0_Introduction | Driver API texture usage (**API mismatch**) |
| 19 | `UnifiedMemoryPerf` | 6_Performance | UM with advise/prefetch vs pageable/pinned/zero-copy |
| 20 | `UnifiedMemoryStreams` | 0_Introduction | UM with concurrent streams and prefetch |
| 21 | `alignedTypes` | 6_Performance | Memory alignment effects on coalescing and throughput |
| 22 | `cudaCompressibleMemory` | 3_CUDA_Features | `cuMemMap` for compressible memory to save DRAM footprint |
| 23 | `simpleAttributes` | 0_Introduction | Setting cache attributes on global memory windows |

## Stage 4 — Streams, Concurrency, and Multi-GPU

Goal: overlap compute and data transfer; use multiple GPUs.

| # | Sample | Category | What you learn |
| --- | ------ | -------- | -------------- |
| 24 | `simpleMultiCopy` | 0_Introduction | Streams to overlap kernels with H2D/D2H copies |
| 25 | `simpleCallback` | 0_Introduction | Stream/event CPU callbacks for heterogeneous pipelines |
| 26 | `simpleHyperQ` | 0_Introduction | Multiple streams and concurrent kernels (Hyper-Q) |
| 27 | `simpleStreams` | 0_Introduction | Practical stream overlap patterns |
| 28 | `StreamPriorities` | 3_CUDA_Features | Stream priorities to bias GPU scheduling |
| 29 | `simpleMultiGPU` | 0_Introduction | Multi-GPU context and threading model |
| 30 | `simpleP2P` | 0_Introduction | Peer-to-peer memory access between GPUs |
| 31 | `p2pBandwidthLatencyTest` | 5_Domain_Specific | Measure P2P bandwidth and latency between GPU pairs |
| 32 | `topologyQuery` | 1_Utilities | Multi-GPU and P2P topology queries |
| 33 | `cudaOpenMP` | 0_Introduction | OpenMP on the host driving multiple GPUs |

## Stage 5 — Atomics, Warp Intrinsics, and Cooperative Groups

Goal: fine-grained synchronization and communication.

| # | Sample | Category | What you learn |
| --- | ------ | -------- | -------------- |
| 34 | `simpleAtomicIntrinsics` | 0_Introduction | Global memory atomics (add, min, max, CAS) |
| 35 | `systemWideAtomics` | 0_Introduction | System-scope atomics (across GPU + CPU) |
| 36 | `simpleVoteIntrinsics` | 0_Introduction | Warp-vote: `__all_sync`, `__any_sync` |
| 37 | `warpAggregatedAtomicsCG` | 3_CUDA_Features | Warp-level aggregated atomics with cooperative groups |
| 38 | `simpleCooperativeGroups` | 0_Introduction | Cooperative groups: partition, sync within a block |
| 39 | `binaryPartitionCG` | 3_CUDA_Features | Binary partition + reduce-style ops in cooperative groups |
| 40 | `simpleAWBarrier` | 0_Introduction | Arrive-wait barriers for producer/consumer patterns |
| 41 | `globalToShmemAsyncCopy` | 3_CUDA_Features | Async copy from global to shared (`cp.async` on Ampere+) |
| 42 | `newdelete` | 3_CUDA_Features | Device-side C++ `new`/`delete` and dynamic global heap |

## Stage 6 — Classic Parallel Algorithms

Goal: implement fundamental parallel patterns on the GPU.

| # | Sample | Category | What you learn |
| --- | ------ | -------- | -------------- |
| 43 | `scalarProd` | 2_Concepts_and_Techniques | Scalar product (dot product) across large vectors |
| 44 | `reduction` | 2_Concepts_and_Techniques | Sum reduction: shared memory, shuffles, coop_group variants |
| 45 | `threadFenceReduction` | 2_Concepts_and_Techniques | Single-kernel reduction with `__threadfence` + global atomics |
| 46 | `reductionMultiBlockCG` | 2_Concepts_and_Techniques | One-pass multi-block reduction with cooperative launch |
| 47 | `scan` | 2_Concepts_and_Techniques | Parallel prefix sum (inclusive/exclusive scan) |
| 48 | `shfl_scan` | 2_Concepts_and_Techniques | Warp-shuffle-based scan (`__shfl_up_sync`) |
| 49 | `histogram` | 2_Concepts_and_Techniques | Parallel histogram (64/256 bins) with efficient bin reduction |
| 50 | `transpose` | 6_Performance | Matrix transpose kernels and tiling/coalescing tradeoffs |
| 51 | `sortingNetworks` | 2_Concepts_and_Techniques | Bitonic and odd-even sorting networks |
| 52 | `mergeSort` | 0_Introduction | Merge-style sorting for small/medium batches |
| 53 | `radixSortThrust` | 2_Concepts_and_Techniques | Parallel radix sort using Thrust (keys and key-value) |
| 54 | `fastWalshTransform` | 5_Domain_Specific | Fast Walsh-Hadamard transform in parallel |
| 55 | `dwtHaar1D` | 5_Domain_Specific | 1D Haar wavelet discrete wavelet transform |
| 56 | :wrench: `segmentationTreeThrust` | 2_Concepts_and_Techniques | Segmentation tree with Thrust (**Thrust/CUB**) |

## Stage 7 — Matrix Multiply and Tensor Cores

Goal: from naive tiled matmul to Tensor Core WMMA.

| # | Sample | Category | What you learn |
| --- | ------ | -------- | -------------- |
| 57 | `matrixMul` | 0_Introduction | Walk-through tiled matrix multiply |
| 58 | `matrixMulCUBLAS` | 4_CUDA_Libraries | High-performance GEMM via cuBLAS |
| 59 | :wrench: `matrixMulDrv` | 0_Introduction | Driver API matrix multiply (**API mismatch**) |
| 60 | :wrench: `matrixMul_nvrtc` | 0_Introduction | NVRTC-compiled matrix multiply (**API mismatch**) |
| 61 | `fp16ScalarProduct` | 0_Introduction | FP16 (`__half`) scalar product on the GPU |
| 62 | `cudaTensorCoreGemm` | 3_CUDA_Features | FP16 WMMA GEMM on Tensor Cores |
| 63 | `bf16TensorCoreGemm` | 3_CUDA_Features | BF16 GEMM via WMMA + async global-to-shared pipeline |
| 64 | `tf32TensorCoreGemm` | 3_CUDA_Features | TF32 Tensor Core GEMM with async loads to shared memory |
| 65 | `dmmaTensorCoreGemm` | 3_CUDA_Features | Double-precision WMMA (DMMA) GEMM |
| 66 | `immaTensorCoreGemm` | 3_CUDA_Features | Integer IMMA GEMM on Tensor Cores |

## Stage 8 — CUDA Graphs

Goal: capture, instantiate, and update work graphs for low-overhead replay.

| # | Sample | Category | What you learn |
| --- | ------ | -------- | -------------- |
| 67 | `simpleCudaGraphs` | 3_CUDA_Features | Capture a stream into a graph, instantiate, and launch |
| 68 | `jacobiCudaGraphs` | 3_CUDA_Features | Update an executed graph (`cudaGraphExecUpdate`) |
| 69 | `graphMemoryNodes` | 3_CUDA_Features | Alloc/free memory inside CUDA graphs |
| 70 | `graphMemoryFootprint` | 3_CUDA_Features | How graph memory nodes reuse VA and physical memory |
| 71 | `cudaGraphsPerfScaling` | 6_Performance | Graph launch overhead at scale |
| 72 | `conjugateGradientCudaGraphs` | 4_CUDA_Libraries | Graph-captured cuBLAS/cuSPARSE calls for lower overhead |
| 73 | :wrench: `graphConditionalNodes` | 3_CUDA_Features | Conditional execution in graphs (**CUDA 12.4+**) |

## Stage 9 — CUDA Dynamic Parallelism

Goal: launch kernels from the device.

| # | Sample | Category | What you learn |
| --- | ------ | -------- | -------------- |
| 74 | `cdpSimplePrint` | 3_CUDA_Features | Minimal CDP: device launches a child kernel |
| 75 | `cdpSimpleQuicksort` | 3_CUDA_Features | Recursive quicksort with device-side launches |
| 76 | `cdpAdvancedQuicksort` | 3_CUDA_Features | Advanced CDP quicksort with selection sort fallback |
| 77 | `cdpBezierTessellation` | 3_CUDA_Features | CDP for adaptive Bezier curve tessellation |
| 78 | :wrench: `cdpQuadtree` | 3_CUDA_Features | CDP quadtree with Thrust (**Thrust/CUB**) |

## Stage 10 — NVRTC and Driver API

Goal: runtime compilation and low-level GPU control.

| # | Sample | Category | What you learn |
| --- | ------ | -------- | -------------- |
| 79 | :wrench: `vectorAdd_nvrtc` | 0_Introduction | NVRTC "hello world": compile kernel at runtime (**API mismatch**) |
| 80 | :wrench: `vectorAddDrv` | 0_Introduction | Driver API vectorAdd with module loading (**API mismatch**) |
| 81 | :wrench: `clock_nvrtc` | 0_Introduction | NVRTC variant of the clock sample (**API mismatch**) |
| 82 | :wrench: `simpleAssert_nvrtc` | 0_Introduction | NVRTC variant of simpleAssert (**API mismatch**) |
| 83 | :wrench: `simpleAtomicIntrinsics_nvrtc` | 0_Introduction | NVRTC variant of simpleAtomicIntrinsics (**API mismatch**) |
| 84 | :wrench: `inlinePTX_nvrtc` | 2_Concepts_and_Techniques | NVRTC with inline PTX (**API mismatch**) |
| 85 | :wrench: `simpleDrvRuntime` | 0_Introduction | Driver + Runtime APIs together (**API mismatch**) |
| 86 | :wrench: `threadMigration` | 2_Concepts_and_Techniques | Driver API thread and context migration (**API mismatch**) |
| 87 | :wrench: `vectorAddMMAP` | 0_Introduction | Memory-mapped vector add with cuMemMap (**API mismatch**) |
| 88 | :wrench: `memMapIPCDrv` | 3_CUDA_Features | Driver API cuMemMap for IPC (**API mismatch**) |

## Stage 11 — Libraries (cuBLAS, cuSOLVER, cuRAND, cuFFT, NPP, nvJPEG)

Goal: leverage GPU-accelerated libraries.

| # | Sample | Category | What you learn |
| --- | ------ | -------- | -------------- |
| 89 | `simpleCUBLAS` | 4_CUDA_Libraries | Basic cuBLAS SGEMM/DGEMM API usage |
| 90 | `simpleCUBLASXT` | 4_CUDA_Libraries | cuBLAS-XT: out-of-core / multi-GPU GEMM |
| 91 | `simpleCUBLAS_LU` | 4_CUDA_Libraries | Batched LU factorization via cuBLAS |
| 92 | `batchCUBLAS` | 4_CUDA_Libraries | Batched cuBLAS calls (many small GEMMs) |
| 93 | `conjugateGradient` | 4_CUDA_Libraries | Iterative CG solver with cuBLAS + cuSPARSE |
| 94 | `conjugateGradientPrecond` | 4_CUDA_Libraries | Preconditioned CG (sparse) |
| 95 | `conjugateGradientUM` | 4_CUDA_Libraries | CG using Unified Memory |
| 96 | `conjugateGradientMultiBlockCG` | 4_CUDA_Libraries | Multi-block cooperative groups inside an iterative solver |
| 97 | `conjugateGradientMultiDeviceCG` | 4_CUDA_Libraries | Multi-GPU CG with UM and prefetch hints |
| 98 | `cuSolverDn_LinearSolver` | 4_CUDA_Libraries | cuSOLVER dense: LU, QR, Cholesky factorization |
| 99 | `cuSolverSp_LinearSolver` | 4_CUDA_Libraries | cuSOLVER sparse: direct/iterative solvers |
| 100 | `cuSolverSp_LowlevelCholesky` | 4_CUDA_Libraries | cuSOLVER low-level sparse Cholesky |
| 101 | `cuSolverSp_LowlevelQR` | 4_CUDA_Libraries | cuSOLVER low-level sparse QR factorization |
| 102 | `cuSolverRf` | 4_CUDA_Libraries | cuSolverRF: refactorization for repeated sparse solves |
| 103 | `MersenneTwisterGP11213` | 4_CUDA_Libraries | cuRAND Mersenne Twister PRNG setup and use |
| 104 | :wrench: `simpleCUFFT` | 4_CUDA_Libraries | Basic cuFFT 1D transform (**CUFFT codes**) |
| 105 | :wrench: `simpleCUFFT_MGPU` | 4_CUDA_Libraries | Multi-GPU cuFFT (**CUFFT codes**) |
| 106 | :wrench: `simpleCUFFT_2d_MGPU` | 4_CUDA_Libraries | 2D multi-GPU cuFFT (**CUFFT codes**) |
| 107 | :wrench: `simpleCUFFT_callback` | 4_CUDA_Libraries | cuFFT callback routines (**CUFFT codes**) |
| 108 | `nvJPEG` | 4_CUDA_Libraries | Decode JPEGs (single and batched) with nvJPEG |
| 109 | `nvJPEG_encoder` | 4_CUDA_Libraries | Encode to JPEG with nvJPEG |
| 110 | `lineOfSight` | 4_CUDA_Libraries | Thrust-based line-of-sight on a heightfield |
| 111 | :wrench: `watershedSegmentationNPP` | 4_CUDA_Libraries | NPP watershed segmentation (**NPP sig**) |

## Stage 12 — Domain Applications

Goal: see CUDA applied to finance, PDE, image processing.

| # | Sample | Category | What you learn |
| --- | ------ | -------- | -------------- |
| 112 | `BlackScholes` | 5_Domain_Specific | Closed-form Black-Scholes pricing in parallel |
| 113 | :wrench: `BlackScholes_nvrtc` | 5_Domain_Specific | NVRTC variant of Black-Scholes (**API mismatch**) |
| 114 | `binomialOptions` | 5_Domain_Specific | Binomial lattice pricing of European calls |
| 115 | :wrench: `binomialOptions_nvrtc` | 5_Domain_Specific | NVRTC variant of binomialOptions (**API mismatch**) |
| 116 | `MonteCarloMultiGPU` | 5_Domain_Specific | Multi-GPU Monte Carlo option pricing |
| 117 | `MC_EstimatePiInlineP` | 2_Concepts_and_Techniques | Monte Carlo pi estimation with cuRAND inline (different PRNGs) |
| 118 | `MC_EstimatePiInlineQ` | 2_Concepts_and_Techniques | Same with quasi-random sequences |
| 119 | `MC_EstimatePiP` | 2_Concepts_and_Techniques | Monte Carlo pi estimation (pseudo-random) |
| 120 | `MC_EstimatePiQ` | 2_Concepts_and_Techniques | Same with quasi-random sequences |
| 121 | `MC_SingleAsianOptionP` | 2_Concepts_and_Techniques | MC pricing of an Asian option |
| 122 | `quasirandomGenerator` | 5_Domain_Specific | Niederreiter quasirandom sequences for Quasi-MC |
| 123 | :wrench: `quasirandomGenerator_nvrtc` | 5_Domain_Specific | NVRTC variant (**API mismatch**) |
| 124 | `SobolQRNG` | 5_Domain_Specific | Sobol quasirandom generator |
| 125 | `FDTD3d` | 5_Domain_Specific | 3D FDTD stencil time-marching (PDE) |
| 126 | `convolutionSeparable` | 2_Concepts_and_Techniques | Separable 2D convolution (e.g., Gaussian filter) |
| 127 | `convolutionTexture` | 2_Concepts_and_Techniques | Texture-cache-friendly separable convolution |
| 128 | :wrench: `convolutionFFT2D` | 5_Domain_Specific | Large 2D convolution via FFT (**CUFFT codes**) |
| 129 | `dct8x8` | 2_Concepts_and_Techniques | 8x8 DCT for blocks — naive vs optimized |
| 130 | `HSOpticalFlow` | 5_Domain_Specific | Horn-Schunck optical flow with textures and iterative PDE |
| 131 | `stereoDisparity` | 5_Domain_Specific | Disparity map with SAD using SIMD intrinsics |
| 132 | `dxtc` | 5_Domain_Specific | DXT block texture compression on the GPU |
| 133 | `NV12toBGRandResize` | 5_Domain_Specific | YUV NV12 to BGR conversion and resize for inference prep |
| 134 | `eigenvalues` | 2_Concepts_and_Techniques | Bisection to find eigenvalues of tridiagonal matrices |
| 135 | `interval` | 2_Concepts_and_Techniques | Interval arithmetic with C++ templates |

## Stage 13 — IPC, Stream-Ordered Allocation, and Advanced Memory

Goal: multi-process GPU sharing and memory pool APIs.

| # | Sample | Category | What you learn |
| --- | ------ | -------- | -------------- |
| 136 | `simpleIPC` | 0_Introduction | Inter-process CUDA: share buffers across processes |
| 137 | `streamOrderedAllocation` | 2_Concepts_and_Techniques | `cudaMallocAsync` / memory pool (cudaMemPool) use |
| 138 | `streamOrderedAllocationIPC` | 2_Concepts_and_Techniques | IPC-exported mempools + stream-ordered alloc |
| 139 | `streamOrderedAllocationP2P` | 2_Concepts_and_Techniques | P2P access to stream-ordered allocations (multi-GPU) |

## Stage 14 — PTX, JIT, libNVVM, and Low-Level Compilation

Goal: work with PTX, runtime compilation, and NVVM IR.

| # | Sample | Category | What you learn |
| --- | ------ | -------- | -------------- |
| 140 | `inlinePTX` | 2_Concepts_and_Techniques | Inline PTX assembly in CUDA kernels |
| 141 | `ptxjit` | 3_CUDA_Features | Driver API JIT of PTX to SASS via `cuLink` |
| 142 | `matrixMulDynlinkJIT` | 0_Introduction | Dynamic linking and JIT of a matrix multiply module |
| 143 | `ptxgen` | 7_libNVVM | PTX generation from NVVM IR |
| 144 | :wrench: `simple` | 7_libNVVM | libNVVM basic usage (**API mismatch**) |
| 145 | :wrench: `uvmlite` | 7_libNVVM | libNVVM with unified virtual memory (**API mismatch**) |
| 146 | :wrench: `device-side-launch` | 7_libNVVM | libNVVM device-side kernel launch (**API mismatch**) |
| 147 | :wrench: `jitLto` | 4_CUDA_Libraries | JIT link-time optimization (**nvJitLink** + **API mismatch**) |

---

## Fix Guide — Making broken samples build

### Fix 1: cuCtxCreate API mismatch (20 samples)

The samples source targets CUDA 13.2 which changed `cuCtxCreate` to accept a
`CUctxCreateParams` struct.  In CUDA 12.2 the old signature is still used.

**Option A — Patch `Common/nvrtc_helper.h`** (quick, local fix):

Find the `cuCtxCreate` calls that pass a params struct and replace them with the
old two-argument form:

```c
// New API (CUDA 13.2):
// cuCtxCreate(&context, &ctxCreateParams, device);
// Old API (CUDA 12.2):
cuCtxCreate(&context, 0, device);
```

Affected files: `Common/nvrtc_helper.h` and individual sample `.c`/`.cpp` files
under `7_libNVVM/`.

**Option B — Upgrade the base image** to `nvidia/cuda:13.2` or newer.

### Fix 2: Newer CUFFT error codes (5 samples)

`Common/helper_cuda.h` references `CUFFT_MISSING_DEPENDENCY` and other enums
added after CUDA 12.2.

**Option A — Patch `Common/helper_cuda.h`**: wrap the new enum cases in
`#if CUFFT_VERSION >= ...` guards.

**Option B — Upgrade** the base image.

### Fix 3: Thrust/CUB API incompatibility (2 samples)

`cdpQuadtree` and `segmentationTreeThrust` use `thrust::minimum` and other APIs
whose signatures changed between the Thrust version in CUDA 12.2 and 13.2.

**Fix**: Upgrade the CUDA toolkit.  No simple local patch.

### Fix 4: CUDA 12.4+ APIs (1 sample)

`graphConditionalNodes` uses `cudaGraphConditionalHandle` which was added in
CUDA 12.4.

**Fix**: Upgrade to CUDA 12.4+.

### Fix 5: nvJitLink (1 sample)

`jitLto` uses the `nvJitLink` API not present in CUDA 12.2.

**Fix**: Upgrade to a CUDA toolkit that ships `libnvJitLink`.

### Fix 6: NPP API signature (1 sample)

`watershedSegmentationNPP` passes `int*` where CUDA 13.2 NPP expects `size_t*`.

**Fix**: Cast the arguments or upgrade the toolkit.

---

## Container context

- Base image: `nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04`
- CUDA 12.2, max architecture `compute_90` (Hopper)
- 118 build successfully, 29 need fixes (above), 38 need extra libs (see cuda-advanced)
- Run `python3 scripts/analyze_build.py /tmp/build_output.log` for current status

## Tips

- Start at Stage 1 even if you know C/C++ — the GPU execution model is different
- Read the sample's own `README.md` (under `Samples/<category>/<sample>/`) before running
- Stages 1-6 are sequential prerequisites; Stages 7-14 can be explored in any order
- Wrench-marked samples are still worth reading for the concepts; just skip running them

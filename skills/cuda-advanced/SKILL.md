---
name: cuda-advanced
description: >-
  CUDA samples that require additional system libraries (OpenGL, FreeImage,
  Vulkan, EGL, MPI, NvSCI) or Windows.  Use when the user wants to go beyond
  the primer and explore graphics interop, image I/O, distributed computing,
  or platform-specific features.
---

# CUDA Advanced — Samples Requiring Extra Dependencies

These 38 samples are skipped at cmake configure time because the container
lacks certain system libraries.  Install the dependencies below to unlock them.

Prerequisite: complete at least Stages 1-6 of the **cuda-primer** skill first.

---

## Group 1 — OpenGL Interop (22 samples)

**Install**:

```bash
apt-get install -y libgl-dev libglu1-mesa-dev freeglut3-dev libglew-dev libglfw3-dev
```

Note: a display server (X11/EGL) or virtual framebuffer (`xvfb`) is needed to
run most of these.

| # | Sample | Category | What you learn |
| --- | ------ | -------- | -------------- |
| 1 | `simpleGL` | 5_Domain_Specific | Minimal CUDA-OpenGL interop: write to a PBO from a kernel |
| 2 | `simpleCUDA2GL` | 0_Introduction | Map a CUDA surface to an OpenGL texture |
| 3 | `simpleTexture3D` | 0_Introduction | 3D texture with OpenGL volume rendering |
| 4 | `postProcessGL` | 5_Domain_Specific | Post-processing effects in CUDA with GL display |
| 5 | `Mandelbrot` | 5_Domain_Specific | Mandelbrot / Julia set with interactive GL zoom |
| 6 | `boxFilter` | 2_Concepts_and_Techniques | Box filter on images with OpenGL display path |
| 7 | `SobelFilter` | 5_Domain_Specific | Sobel edge detection with CUDA and GL display |
| 8 | `recursiveGaussian` | 5_Domain_Specific | Recursive Gaussian filter with GL display |
| 9 | `bicubicTexture` | 5_Domain_Specific | Bicubic texture filtering with OpenGL display |
| 10 | `bilateralFilter` | 5_Domain_Specific | Edge-preserving bilateral filter with GL display |
| 11 | `imageDenoising` | 2_Concepts_and_Techniques | Image denoising with NLM filter and GL display |
| 12 | `bindlessTexture` | 3_CUDA_Features | Bindless textures with CUDA-GL interop |
| 13 | `FunctionPointers` | 2_Concepts_and_Techniques | Device function pointers with GL display |
| 14 | `volumeRender` | 5_Domain_Specific | Ray-cast volume rendering with GL display |
| 15 | `volumeFiltering` | 5_Domain_Specific | 3D volume filtering with GL volume rendering |
| 16 | `particles` | 2_Concepts_and_Techniques | Particle system simulation with GL rendering |
| 17 | `smokeParticles` | 5_Domain_Specific | Smoke particle system with 3D GL rendering |
| 18 | `nbody` | 5_Domain_Specific | N-body gravitational simulation with GL visualization |
| 19 | `marchingCubes` | 5_Domain_Specific | Marching cubes isosurface extraction with GL rendering |
| 20 | `fluidsGL` | 5_Domain_Specific | 2D fluid simulation (Navier-Stokes) with GL rendering |
| 21 | `oceanFFT` | 4_CUDA_Libraries | FFT-based ocean surface simulation with GL rendering |
| 22 | `randomFog` | 4_CUDA_Libraries | Quasi/pseudo-random fog with cuRAND and GL |

## Group 2 — FreeImage (Image I/O) (5 samples)

**Install**:

```bash
apt-get install -y libfreeimage-dev
```

| # | Sample | Category | What you learn |
| --- | ------ | -------- | -------------- |
| 23 | `boxFilterNPP` | 4_CUDA_Libraries | NPP box filter on an image loaded via FreeImage |
| 24 | `cannyEdgeDetectorNPP` | 4_CUDA_Libraries | NPP Canny edge detection pipeline |
| 25 | `histEqualizationNPP` | 4_CUDA_Libraries | NPP histogram equalization for contrast enhancement |
| 26 | `freeImageInteropNPP` | 4_CUDA_Libraries | FreeImage + NPP interop pattern |
| 27 | `FilterBorderControlNPP` | 4_CUDA_Libraries | NPP filter with border control modes |

## Group 3 — Vulkan Interop (3 samples)

**Install**:

```bash
apt-get install -y libvulkan-dev vulkan-tools
```

| # | Sample | Category | What you learn |
| --- | ------ | -------- | -------------- |
| 28 | `simpleVulkan` | 5_Domain_Specific | Minimal CUDA-Vulkan interop with external memory |
| 29 | `simpleVulkanMMAP` | 5_Domain_Specific | CUDA-Vulkan interop using memory-mapped buffers |
| 30 | `vulkanImageCUDA` | 5_Domain_Specific | Share Vulkan images with CUDA kernels |

## Group 4 — EGL (2 samples)

**Install**:

```bash
apt-get install -y libegl1-mesa-dev
```

| # | Sample | Category | What you learn |
| --- | ------ | -------- | -------------- |
| 31 | `EGLStream_CUDA_CrossGPU` | 2_Concepts_and_Techniques | EGL streams for cross-GPU data passing |
| 32 | `EGLStream_CUDA_Interop` | 2_Concepts_and_Techniques | CUDA-EGL stream interop for producer/consumer |

## Group 5 — MPI (1 sample)

**Install**:

```bash
apt-get install -y libopenmpi-dev openmpi-bin
```

| # | Sample | Category | What you learn |
| --- | ------ | -------- | -------------- |
| 33 | `simpleMPI` | 0_Introduction | MPI + CUDA: distributed GPU computing across nodes |

## Group 6 — NvSCI (1 sample)

Requires NVIDIA NvSCI libraries (Tegra/DRIVE platforms only).

| # | Sample | Category | What you learn |
| --- | ------ | -------- | -------------- |
| 34 | `cudaNvSci` | 4_CUDA_Libraries | NvSciBuf/NvSciSync interop with CUDA |

## Group 7 — Windows Only (3 samples)

These require DirectX and will not build on Linux.

| # | Sample | Category | What you learn |
| --- | ------ | -------- | -------------- |
| 35 | `simpleD3D11` | 5_Domain_Specific | CUDA-Direct3D 11 interop |
| 36 | `simpleD3D11Texture` | 5_Domain_Specific | CUDA-D3D11 texture sharing |
| 37 | `simpleD3D12` | 5_Domain_Specific | CUDA-Direct3D 12 interop |

## Group 8 — CMake Flag Required (1 sample)

| # | Sample | Category | What you learn |
| --- | ------ | -------- | -------------- |
| 38 | `cuda-c-linking` | 7_libNVVM | Linking CUDA C with libNVVM (needs ENABLE flag) |

---

## Suggested learning order

1. **FreeImage** samples (Group 2) — simplest to install, extends NPP knowledge
2. **MPI** sample (Group 5) — introduces multi-node GPU computing
3. **OpenGL** samples (Group 1) — large set; start with `simpleGL` then `nbody`
4. **Vulkan** samples (Group 3) — modern graphics API interop
5. **EGL** samples (Group 4) — headless rendering / stream interop
6. Groups 6-8 are platform-specific; skip unless you target those platforms

## Container context

- Base image: `nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04`
- The `apt-get install` commands above work in this Ubuntu 22.04 container
- After installing deps, rebuild: `./scripts/build_samples --update-arch`

---
name: analyze-build
description: >-
  Analyze CUDA samples cmake build results and summarize success/failure status.
  Use when the user asks to build samples, check build status, diagnose build
  failures, or investigate why samples fail to compile in the current container.
---

# Analyze CUDA Samples Build

## Quick Start

- Build the samples (captures log for analysis):

```bash
./scripts/build_samples 2>&1 | tee /tmp/build_output.log
```

Use `--update-arch` to auto-detect the GPU and update `CMAKE_CUDA_ARCHITECTURES`:

```bash
./scripts/build_samples --update-arch 2>&1 | tee /tmp/build_output.log
```

- Run the analyzer (failures only by default):

```bash
python3 scripts/analyze_build.py /tmp/build_output.log
```

Show successful samples too:

```bash
python3 scripts/analyze_build.py /tmp/build_output.log --show-success
```

JSON output (for programmatic use):

```bash
python3 scripts/analyze_build.py /tmp/build_output.log --json
```

## What the Analyzer Reports

The tool parses cmake + make output and classifies every sample into three statuses,
then outputs two separate sections for failures:

| Status             | Meaning                                                         |
| ------------------ | --------------------------------------------------------------- |
| **success**        | Built successfully (hidden by default, use `--show-success`)    |
| **cmake_skipped**  | Excluded at configure time (missing dependency, wrong platform) |
| **compile_failed** | Attempted to build but failed                                   |

### CMAKE-SKIPPED reasons

| Reason                                  | Typical Fix                                  |
| --------------------------------------- | -------------------------------------------- |
| Missing OpenGL/FreeImage/Vulkan/EGL/MPI | Install library in Dockerfile                |
| Platform-only (Windows)                 | Cannot build on Linux                        |
| ENABLE flag not set                     | Set cmake flag (e.g. LLVM for cuda-c-linking)|

### COMPILE-FAILED reasons

| Reason                                   | Typical Fix                                            |
| ---------------------------------------- | ------------------------------------------------------ |
| Unsupported GPU arch                     | Use `--update-arch` or edit `CMAKE_CUDA_ARCHITECTURES` |
| cuCtxCreate API requires newer CUDA      | Upgrade toolkit or patch `Common/nvrtc_helper.h`       |
| Thrust/CUB API incompatibility           | Upgrade toolkit (Thrust version mismatch)              |
| Newer CUFFT error codes                  | Upgrade toolkit or patch `Common/helper_cuda.h`        |
| cudaGraphConditional* not available      | Requires CUDA 12.4+                                    |
| nvJitLink API not available              | Requires newer CUDA toolkit                            |
| NPP API signature change (size_t vs int) | Upgrade toolkit or patch sample source                 |

## Container Context

- Base image: `nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04` (see `docker/Dockerfile`)
- CUDA 12.2 supports architectures up to `compute_90` (Hopper)
- Architectures 100+ (Blackwell) require CUDA 12.8+
- Intermediate cmake targets (`generate_*`) are filtered from results

## After Analysis

Use `AskQuestion` to let the user decide next steps:

- Fix arch issues (`--update-arch`, or edit CMakeLists.txt)
- Install missing system dependencies in Dockerfile
- Patch headers for CUDA API compatibility
- Exclude unfixable samples from the build
- Upgrade the CUDA base image

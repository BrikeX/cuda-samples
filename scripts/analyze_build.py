#!/usr/bin/env python3
"""Analyze cmake build output and summarize success/failure status.

Usage:
    # Analyze from a build log file:
    python3 scripts/analyze_build.py build.log

    # Also show successfully built samples:
    python3 scripts/analyze_build.py build.log --show-success

    # Output as JSON:
    python3 scripts/analyze_build.py build.log --json

    # Pipe from build_samples:
    ./scripts/build_samples 2>&1 | tee build.log
    python3 scripts/analyze_build.py build.log
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SampleResult:
    name: str
    category: str  # e.g. "0_Introduction"
    status: str  # "success", "cmake_skipped", "compile_failed"
    reasons: list[str] = field(default_factory=list)


def parse_build_log(log_text: str) -> dict:
    lines = log_text.splitlines()

    cmake_skipped: dict[str, str] = {}
    built_targets: set[str] = set()
    compile_errors: dict[str, list[str]] = defaultdict(list)

    # --- Phase 1: CMake configure messages ---

    skip_pattern = re.compile(r"-- (\S.*?) - will not build sample '(\S+)'")
    skip_platform_pattern = re.compile(r"-- Sample '(\S+)' is (\S+)-only - skipping")
    skip_generic_pattern = re.compile(r"-- Skipping the build of the (\S+) sample")
    skip_generic_reason_pattern = re.compile(
        r"-- Skipping the build of the (\S+) sample[.:]\s*(.*)"
    )

    for line in lines:
        m = skip_pattern.search(line)
        if m:
            reason, sample = m.group(1), m.group(2)
            cmake_skipped[sample] = f"Missing dependency: {reason}"
            continue

        m = skip_platform_pattern.search(line)
        if m:
            sample, platform = m.group(1), m.group(2)
            cmake_skipped[sample] = f"Platform-only: {platform}"
            continue

        m = skip_generic_reason_pattern.search(line)
        if m:
            sample = m.group(1)
            detail = m.group(2).strip() if m.group(2).strip() else None
            cmake_skipped[sample] = (
                f"Skipped by cmake: {detail}"
                if detail
                else "Skipped by cmake (ENABLE flag not set or missing dependency)"
            )
            continue

        m = skip_generic_pattern.search(line)
        if m:
            sample = m.group(1)
            cmake_skipped[sample] = (
                "Skipped by cmake (ENABLE flag not set or missing dependency)"
            )
            continue

    # --- Phase 2: Build results ---

    built_pattern = re.compile(r"Built target (\S+)")
    arch_err_pattern = re.compile(
        r"nvcc fatal\s*:\s*Unsupported gpu architecture '(compute_\d+)'"
    )
    src_err_pattern = re.compile(r"^(/workspace/\S+):(\d+):\d+: error: (.+)")
    nvcc_err_pattern = re.compile(r"^(/\S+\.(?:cu|cpp|c|h|cuh))\((\d+)\): error: (.+)")
    gmake_err_pattern = re.compile(r"gmake\[2\].*?(Samples/[^/]+/[^/]+)/.*Error \d+")

    building_context: Optional[str] = None
    building_pattern = re.compile(
        r"Building (?:CUDA|CXX|C) object (Samples/[^/]+/[^/]+)/"
    )

    for i, line in enumerate(lines):
        m = built_pattern.search(line)
        if m:
            built_targets.add(m.group(1))
            continue

        m = building_pattern.search(line)
        if m:
            building_context = m.group(1)

        m = arch_err_pattern.search(line)
        if m:
            arch = m.group(1)
            if building_context:
                sample_name = building_context.split("/")[-1]
                msg = f"Unsupported GPU arch '{arch}' (CUDA toolkit too old)"
                if msg not in compile_errors.get(sample_name, []):
                    compile_errors[sample_name].append(msg)

        m = src_err_pattern.search(line)
        if not m:
            m = nvcc_err_pattern.search(line)
        if m:
            filepath, lineno, error_msg = m.group(1), m.group(2), m.group(3)
            sample_match = re.search(r"Samples/[^/]+/([^/]+)", filepath)
            if not sample_match:
                # Error from a project header (Common/) -- use path to find sample.
                # Skip system headers here; they'll be handled by the
                # nvcc summary backward-lookup in the second pass.
                common_match = re.search(r"Samples/[^/]+/([^/]+)/", filepath)
                if common_match:
                    sample_name = common_match.group(1)
                    msg = _classify_source_error(error_msg, filepath)
                    if msg and msg not in compile_errors.get(sample_name, []):
                        compile_errors[sample_name].append(msg)
            else:
                sample_name = sample_match.group(1)
                msg = _classify_source_error(error_msg, filepath)
                if msg and msg not in compile_errors.get(sample_name, []):
                    compile_errors[sample_name].append(msg)

        m = gmake_err_pattern.search(line)
        if m:
            sample_path = m.group(1)
            sample_name = sample_path.split("/")[-1]
            if sample_name not in compile_errors:
                compile_errors[sample_name] = ["Build failed (see log for details)"]

    # Second pass: use nvcc "N errors detected" summary + backward context
    nvcc_summary_pattern = re.compile(
        r'\d+ errors? detected in the compilation of "(/\S+)"'
    )
    for i, line in enumerate(lines):
        if "Unsupported gpu architecture" in line:
            for j in range(max(0, i - 15), i):
                m = building_pattern.search(lines[j])
                if m:
                    sample_path = m.group(1)
                    sample_name = sample_path.split("/")[-1]
                    arch_m = arch_err_pattern.search(line)
                    arch = arch_m.group(1) if arch_m else "unknown"
                    msg = f"Unsupported GPU arch '{arch}' (CUDA toolkit too old)"
                    if msg not in compile_errors.get(sample_name, []):
                        compile_errors[sample_name].append(msg)

        if "too many arguments to function" in line and "cuCtxCreate" in line:
            for j in range(max(0, i - 20), i):
                m = building_pattern.search(lines[j])
                if m:
                    sample_name = m.group(1).split("/")[-1]
                    msg = (
                        "CUDA API version mismatch: cuCtxCreate API requires newer CUDA"
                    )
                    if msg not in compile_errors.get(sample_name, []):
                        compile_errors[sample_name].append(msg)

        # nvcc prints "N errors detected in compilation of <file>" at the end
        # of a failed compilation. Use this to associate earlier error messages
        # (which may be from system headers) back to the correct sample.
        m = nvcc_summary_pattern.search(line)
        if m:
            cu_path = m.group(1)
            sample_match = re.search(r"Samples/[^/]+/([^/]+)", cu_path)
            if sample_match:
                sample_name = sample_match.group(1)
                # Find the "Building CUDA object" line for this exact .cu
                # file. In parallel builds, output is interleaved, so we
                # search backwards for the matching .cu.o compile line.
                cu_basename = Path(cu_path).stem  # e.g. "cdpQuadtree"
                scan_start = max(0, i - 500)
                for j in range(i - 1, scan_start - 1, -1):
                    if cu_basename in lines[j] and "Building CUDA object" in lines[j]:
                        scan_start = j
                        break

                for j in range(scan_start, i):
                    err_m = nvcc_err_pattern.search(lines[j])
                    if err_m:
                        err_path = err_m.group(1)
                        err_msg = err_m.group(3)
                        if not re.search(r"Samples/", err_path):
                            # Verify: the instantiation trace for THIS error
                            # must reference our .cu file. The trace runs
                            # from the error to the next error/blank line.
                            trace_end = j + 1
                            for k in range(j + 1, i):
                                if nvcc_err_pattern.search(lines[k]):
                                    break
                                if lines[k].strip() == "":
                                    break
                                trace_end = k + 1
                            owns_error = False
                            for k in range(j, trace_end):
                                if cu_basename in lines[k]:
                                    owns_error = True
                                    break
                            if not owns_error:
                                continue
                            msg = _classify_source_error(err_msg, err_path)
                            if msg and msg not in compile_errors.get(sample_name, []):
                                compile_errors[sample_name].append(msg)

    # Remove successfully built targets from error lists
    for t in built_targets:
        compile_errors.pop(t, None)

    return {
        "cmake_skipped": cmake_skipped,
        "built_targets": sorted(built_targets),
        "compile_errors": dict(compile_errors),
    }


def _classify_source_error(error_msg: str, context: str) -> Optional[str]:
    """Classify a source-level error into a human-readable category."""
    error_msg = error_msg.strip()

    if "CUctxCreateParams" in error_msg:
        return "CUDA API version mismatch: cuCtxCreate API requires newer CUDA"
    if "ctxCreateParams" in error_msg or "ctx_params" in error_msg:
        return "CUDA API version mismatch: cuCtxCreate API requires newer CUDA"
    if "too many arguments to function 'cuCtxCreate" in error_msg:
        return "CUDA API version mismatch: cuCtxCreate API requires newer CUDA"
    if "CUFFT_" in error_msg and "not declared" in error_msg:
        return "Newer CUFFT error codes not in this CUDA version"
    if "nvJitLink" in error_msg:
        return "Newer nvJitLink API not available in this CUDA version"
    if "cannot convert" in error_msg and "size_t" in error_msg:
        return "NPP API signature change (size_t* vs int*)"
    if "cudaGraphConditional" in error_msg and "undefined" in error_msg:
        return "CUDA 12.4+ API: cudaGraphConditional* not available"
    if "CUFFT_" in error_msg and "undefined" in error_msg:
        return "Newer CUFFT error codes not in this CUDA version"
    if "thrust::" in error_msg and "cannot be called" in error_msg:
        return "Thrust/CUB API incompatibility with this CUDA version"
    if "no operator" in error_msg and "matches these operands" in error_msg:
        return "Thrust/CUB API incompatibility with this CUDA version"
    if "cannot be called with the given argument list" in error_msg:
        return "Thrust/CUB API incompatibility with this CUDA version"

    return None


_HELPER_TARGET_PATTERN = re.compile(r"^generate_")


def build_report(parsed: dict) -> list[SampleResult]:
    results = []

    for sample, reason in parsed["cmake_skipped"].items():
        results.append(
            SampleResult(
                name=sample, category="", status="cmake_skipped", reasons=[reason]
            )
        )

    for target in parsed["built_targets"]:
        if _HELPER_TARGET_PATTERN.match(target):
            continue
        results.append(
            SampleResult(name=target, category="", status="success", reasons=[])
        )

    for sample, errors in parsed["compile_errors"].items():
        if sample in parsed["built_targets"]:
            continue
        specific = [e for e in errors if e != "Build failed (see log for details)"]
        reasons = specific if specific else errors
        results.append(
            SampleResult(
                name=sample, category="", status="compile_failed", reasons=reasons
            )
        )

    return results


def group_by_reason(results: list[SampleResult]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for r in results:
        if r.status == "success":
            groups["SUCCESS"].append(r.name)
        else:
            for reason in r.reasons:
                groups[reason].append(r.name)
    for key in groups:
        groups[key] = sorted(set(groups[key]))
    return dict(groups)


def print_report(results: list[SampleResult], show_success: bool = False) -> None:
    success = [r for r in results if r.status == "success"]
    skipped = [r for r in results if r.status == "cmake_skipped"]
    failed = [r for r in results if r.status == "compile_failed"]

    total = len(success) + len(skipped) + len(failed)

    print("=" * 60)
    print("  CUDA Samples Build Analysis")
    print("=" * 60)
    print(f"  Total samples  : {total}")
    print(f"  Successful     : {len(success)}")
    print(f"  CMake-skipped  : {len(skipped)}")
    print(f"  Compile-failed : {len(failed)}")
    print()

    if show_success and success:
        print("-" * 60)
        print(f"  SUCCESSFUL ({len(success)})")
        print("-" * 60)
        for r in sorted(success, key=lambda x: x.name):
            print(f"    {r.name}")
        print()

    skip_groups = group_by_reason(skipped)
    fail_groups = group_by_reason(failed)
    skip_groups.pop("SUCCESS", None)
    fail_groups.pop("SUCCESS", None)

    if skip_groups:
        print("-" * 60)
        print(f"  CMAKE-SKIPPED ({len(skipped)}) - excluded at configure time")
        print("-" * 60)
        for reason, samples in sorted(skip_groups.items(), key=lambda x: -len(x[1])):
            print(f"\n  [{len(samples)}] {reason}")
            for s in samples:
                print(f"    - {s}")
        print()

    if fail_groups:
        print("-" * 60)
        print(f"  COMPILE-FAILED ({len(failed)}) - attempted but failed")
        print("-" * 60)
        for reason, samples in sorted(fail_groups.items(), key=lambda x: -len(x[1])):
            print(f"\n  [{len(samples)}] {reason}")
            for s in samples:
                print(f"    - {s}")

    print()
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze cmake build output for CUDA samples"
    )
    parser.add_argument(
        "logfile",
        help="Path to the build log file",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of text",
    )
    parser.add_argument(
        "--show-success",
        action="store_true",
        help="List successfully built samples (hidden by default)",
    )
    args = parser.parse_args()

    log_path = Path(args.logfile)
    if not log_path.exists():
        print(f"Error: {log_path} not found", file=sys.stderr)
        sys.exit(1)

    log_text = log_path.read_text()
    parsed = parse_build_log(log_text)
    results = build_report(parsed)

    if args.json:
        groups = group_by_reason(results)
        if not args.show_success:
            groups.pop("SUCCESS", None)
        output = {
            "summary": {
                "total": len(results),
                "success": len([r for r in results if r.status == "success"]),
                "cmake_skipped": len(
                    [r for r in results if r.status == "cmake_skipped"]
                ),
                "compile_failed": len(
                    [r for r in results if r.status == "compile_failed"]
                ),
            },
            "groups": groups,
            "samples": [asdict(r) for r in results],
        }
        json.dump(output, sys.stdout, indent=2)
        print()
    else:
        print_report(results, show_success=args.show_success)


if __name__ == "__main__":
    main()

"""
Run Enhanced Benchmarks with Summary
Shows performance for parser-optimized operations
"""

import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

# Get the tests directory and benchmark script
tests_dir: Path = Path(__file__).parent
benchmark_script: Path = tests_dir / "benchmark_enhanced.py"

print("Running Enhanced Ekilang Benchmarks...")
print("Testing: Async/Await, Augmented Assignments, Operators, and More")
print("=" * 80)

# Run the benchmark and capture output
result: subprocess.CompletedProcess[str] = subprocess.run(
    [sys.executable, str(benchmark_script)], capture_output=True, text=True
)
output: str = result.stdout

# If process crashed (segfault), show error
if result.returncode != 0 and result.returncode != 1:
    print(f"\nNote: benchmark_enhanced.py crashed with code {result.returncode}")
    print("This may indicate parser issues with certain code patterns.")
    print("Try running: python .\\tests\\benchmark_enhanced.py\n")

# Parse results
benchmarks: list[dict[str, Any]] = []
current_name: str | None = None
current_eki: float | None = None
current_py: float | None = None
current_ratio: float | None = None

for line in output.split("\n"):
    if "Benchmark:" in line and "============" not in line:
        # Extract benchmark name from line like "Benchmark: Simple Integer Arithmetic (100K iterations)"
        match = re.search(r"Benchmark: (.+)$", line)
        if match:
            current_name = match.group(1).strip()
    elif (
        "Ekilang:" in line
        and "compiled code only" not in line
        and "error" not in line.lower()
    ):
        # Extract time from line like "  Ekilang: 27.553ms (avg of 3 runs)"
        match = re.search(r"(\d+\.\d+)ms", line)
        if match:
            current_eki = float(match.group(1))
    elif "Python:" in line and "execution" not in line.lower():
        # Extract time from line like "  Python:  18.555ms (avg of 3 runs)"
        match = re.search(r"(\d+\.\d+)ms", line)
        if match:
            current_py = float(match.group(1))
    elif "Ratio:" in line:
        # Extract ratio and direction from line like "  Ratio:   1.48x slower"
        ratio_match = re.search(r"(\d+\.\d+)x", line)
        is_faster: bool = "faster" in line.lower()
        if ratio_match:
            current_ratio = float(ratio_match.group(1))
            if (
                current_name
                and current_eki is not None
                and current_py is not None
                and current_ratio
            ):
                benchmarks.append(
                    {
                        "name": current_name,
                        "eki": current_eki,
                        "py": current_py,
                        "ratio": current_ratio,
                        "faster": is_faster,
                    }
                )
                # Reset for next benchmark
                current_name = None
                current_eki = None
                current_py = None
                current_ratio = None

# Print summary
print("\n" + "=" * 100)
print("EKILANG BENCHMARK RESULTS - EXECUTION TIME")
print("=" * 100)
print(f"{'Benchmark':<45} {'Ekilang':<15} {'Python':<15} {'Performance':<20}")
print("-" * 100)

total_eki: float = 0
total_py: float = 0
count: int = 0

# Categorize benchmarks
async_benchmarks: List[Dict[str, Any]] = []
operator_benchmarks: List[Dict[str, Any]] = []
feature_benchmarks: List[Dict[str, Any]] = []

for b in benchmarks:
    if any(x in b["name"] for x in ["Async"]):
        async_benchmarks.append(b)
    elif any(
        x in b["name"]
        for x in [
            "Augmented",
            "Bitwise",
            "Comparison",
            "Operator",
            "Unary",
            "Walrus",
            "Pipeline",
        ]
    ):
        operator_benchmarks.append(b)
    else:
        feature_benchmarks.append(b)

# Print by category
if async_benchmarks:
    print("\n  ASYNC/AWAIT OPERATIONS:")
    print("-" * 100)
    for b in async_benchmarks:
        total_eki += b["eki"]
        total_py += b["py"]
        count += 1
        perf_str: str = (
            f"{'↑ ' if b['faster'] else '↓ '}{b['ratio']:.2f}x {'faster' if b['faster'] else 'slower'}"
        )
        print(f"{b['name']:<45} {b['eki']:<15.3f}ms {b['py']:<15.3f}ms {perf_str:<20}")

if operator_benchmarks:
    print("\n  OPERATOR OPTIMIZATIONS:")
    print("-" * 100)
    for b in operator_benchmarks:
        total_eki += b["eki"]
        total_py += b["py"]
        count += 1
        perf_str: str = (
            f"{'↑ ' if b['faster'] else '↓ '}{b['ratio']:.2f}x {'faster' if b['faster'] else 'slower'}"
        )
        print(f"{b['name']:<45} {b['eki']:<15.3f}ms {b['py']:<15.3f}ms {perf_str:<20}")

if feature_benchmarks:
    print("\n  LANGUAGE FEATURES & OPERATIONS:")
    print("-" * 100)
    for b in feature_benchmarks:
        total_eki += b["eki"]
        total_py += b["py"]
        count += 1
        perf_str: str = (
            f"{'↑ ' if b['faster'] else '↓ '}{b['ratio']:.2f}x {'faster' if b['faster'] else 'slower'}"
        )
        print(f"{b['name']:<45} {b['eki']:<15.3f}ms {b['py']:<15.3f}ms {perf_str:<20}")

avg_eki: float = 0
avg_py: float = 0
avg_ratio: float = 0

if count > 0:
    print("-" * 100)
    avg_eki = total_eki / count
    avg_py = total_py / count
    avg_ratio = avg_eki / avg_py
    faster: str = "FASTER" if avg_ratio < 1 else "SLOWER"
    perf_str: str = f"{'↑ ' if avg_ratio < 1 else '↓ '}{avg_ratio:.2f}x {faster}"
    print(f"{'AVERAGE':<45} {avg_eki:<15.3f}ms {avg_py:<15.3f}ms {perf_str:<20}")

print("=" * 100)

# Show statistics
faster_count: int = sum(1 for b in benchmarks if b["faster"])
slower_count: int = len(benchmarks) - faster_count

print(f"\n  Statistics:")
print(f"  Total benchmarks: {len(benchmarks)}")
print(f"  Ekilang faster: {faster_count}")
print(f"  Python faster: {slower_count}")

if count > 0:
    overhead_pct = (avg_ratio - 1) * 100
    if avg_ratio > 1:
        print(f"  Average overhead: {overhead_pct:.1f}% slower")
    else:
        print(f"  Average speedup: {abs(overhead_pct):.1f}% faster")

print(f"\n  Optimization Focus:")
print(f"  - Rust parser helpers active: {len(operator_benchmarks)} operator tests")
print(f"  - Async/await performance: {len(async_benchmarks)} tests")
print(f"  - Language features: {len(feature_benchmarks)} tests")
print()

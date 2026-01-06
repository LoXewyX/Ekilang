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

# Parse results
benchmarks: list[dict[str, Any]] = []
current_name: str | None = None
current_eki: float | None = None
current_py: float | None = None
current_ratio: float | None = None

for line in output.split("\n"):
    if "Benchmark:" in line:
        current_name = line.split("Benchmark: ")[1]
    elif "Ekilang:" in line and "error" not in line.lower():
        match = re.search(r"(\d+\.\d+)ms", line)
        if match:
            current_eki = float(match.group(1))
    elif "Python:" in line:
        match = re.search(r"(\d+\.\d+)ms", line)
        if match:
            current_py = float(match.group(1))
    elif "Ratio:" in line:
        match = re.search(r"(\d+\.\d+)x", line)
        if match:
            current_ratio = float(match.group(1))
            if current_name and current_eki and current_py and current_ratio:
                benchmarks.append(
                    {
                        "name": current_name,
                        "eki": current_eki,
                        "py": current_py,
                        "ratio": current_ratio,
                        "faster": "faster" in line.lower(),
                    }
                )

# Print summary
print("\n" + "=" * 80)
print("ENHANCED BENCHMARK SUMMARY - PARSER & RUNTIME OPTIMIZATIONS")
print("=" * 80)
print(f"\n{'Benchmark':<45} {'Ekilang':<12} {'Python':<12} {'Result':<15}")
print("-" * 80)

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
    elif any(x in b["name"] for x in ["Augmented", "Bitwise", "Comparison", "Operator", "Unary", "Walrus", "Pipeline"]):
        operator_benchmarks.append(b)
    else:
        feature_benchmarks.append(b)

# Print by category
if async_benchmarks:
    print("\nASYNC/AWAIT OPERATIONS:")
    print("-" * 80)
    for b in async_benchmarks:
        total_eki += b["eki"]
        total_py += b["py"]
        count += 1
        result_str: str = f"{b['ratio']:.2f}x {'FASTER' if b['faster'] else 'SLOWER'}"
        print(f"{b['name']:<45} {b['eki']:<12.2f}ms {b['py']:<12.2f}ms {result_str:<15}")

if operator_benchmarks:
    print("\nOPERATOR OPTIMIZATIONS:")
    print("-" * 80)
    for b in operator_benchmarks:
        total_eki += b["eki"]
        total_py += b["py"]
        count += 1
        result_str: str = f"{b['ratio']:.2f}x {'FASTER' if b['faster'] else 'SLOWER'}"
        print(f"{b['name']:<45} {b['eki']:<12.2f}ms {b['py']:<12.2f}ms {result_str:<15}")

if feature_benchmarks:
    print("\nLANGUAGE FEATURES:")
    print("-" * 80)
    for b in feature_benchmarks:
        total_eki += b["eki"]
        total_py += b["py"]
        count += 1
        result_str: str = f"{b['ratio']:.2f}x {'FASTER' if b['faster'] else 'SLOWER'}"
        print(f"{b['name']:<45} {b['eki']:<12.2f}ms {b['py']:<12.2f}ms {result_str:<15}")

avg_eki: float = 0
avg_py: float = 0
avg_ratio: float = 0

if count > 0:
    print("-" * 80)
    avg_eki = total_eki / count
    avg_py = total_py / count
    avg_ratio = avg_eki / avg_py
    faster: str = "FASTER" if avg_ratio < 1 else "SLOWER"
    print(
        f"{'AVERAGE':<45} {avg_eki:<12.2f}ms {avg_py:<12.2f}ms {avg_ratio:.2f}x {faster:<10}"
    )

print("=" * 80)

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

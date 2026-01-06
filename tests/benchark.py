"""
Ekilang vs Python Performance Benchmark Summary
Shows performance comparison for common operations
"""

import re
import subprocess
import sys
from pathlib import Path
from typing import Any

# Get the tests directory and benchmark script
tests_dir: Path = Path(__file__).parent
benchmark_script: Path = tests_dir / "benchmark_vs_python.py"

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
print("EKILANG vs PYTHON PERFORMANCE BENCHMARK SUMMARY")
print("=" * 80)
print(f"\n{'Benchmark':<40} {'Ekilang':<12} {'Python':<12} {'Result':<15}")
print("-" * 80)

total_eki: float = 0
total_py: float = 0
count: int = 0

for b in benchmarks:
    total_eki += b["eki"]
    total_py += b["py"]
    count += 1

    result_str: str = f"{b['ratio']:.2f}x {'FASTER' if b['faster'] else 'SLOWER'}"
    print(f"{b['name']:<40} {b['eki']:<12.2f}ms {b['py']:<12.2f}ms {result_str:<15}")

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
        f"{'AVERAGE':<40} {avg_eki:<12.2f}ms {avg_py:<12.2f}ms {avg_ratio:.2f}x {faster:<10}"
    )

print("=" * 80)

# Show statistics
faster_count: int = sum(1 for b in benchmarks if b["faster"])
slower_count: int = len(benchmarks) - faster_count

print(f"\nSummary:")
print(f"  Total benchmarks: {len(benchmarks)}")
print(f"  Ekilang faster: {faster_count}")
print(f"  Python faster: {slower_count}")

if count > 0:
    print(
        f"  Average overhead: {((avg_ratio - 1) * 100):.1f}% {'slower' if avg_ratio > 1 else 'faster'}"
    )
print()

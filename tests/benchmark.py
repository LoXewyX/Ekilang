"""
Ekilang Benchmark Suite
Measures: Execution time, memory consumption
"""

import argparse
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import os
import sys
import time
import psutil
from ekilang.fast_executor import Compiler, FastProgram, run_fast
from ekilang.lexer import Lexer
from ekilang.parser import Parser, Use
from ekilang.runtime import compile_module, execute
from ekilang.builtins import BUILTINS

# Add current dir to path for ekilang package
ROOT = os.path.abspath(os.path.dirname(__file__))
PARENT = os.path.dirname(ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)


def compile_simple_loop(mod: Any) -> Optional[FastProgram]:
    """Compile module to fast bytecode if supported; returns None on fallback."""
    compiler = Compiler()
    return compiler.compile(mod)


def run_repeated(
    fn: Callable[[], Any], runs: int = 5, warmup: int = 0
) -> Dict[str, float]:
    """Run a function multiple times and collect timing, memory, and CPU usage stats"""
    times: List[float] = []
    mem_deltas: List[float] = []
    cpu_usages: List[float] = []
    process = psutil.Process()
    total_iters: int = warmup + runs
    for i in range(total_iters):
        t0: float = time.perf_counter()
        mem_before: float = psutil.Process().memory_info().rss / 1024 / 1024
        fn()
        elapsed: float = time.perf_counter() - t0
        mem_after: float = psutil.Process().memory_info().rss / 1024 / 1024
        cpu_usage = process.cpu_percent(interval=elapsed)
        if i >= warmup:
            times.append(elapsed)
            mem_deltas.append(mem_after - mem_before)
            cpu_usages.append(cpu_usage)
    times_sorted: List[float] = sorted(times)
    p95: float = times_sorted[int(0.95 * (len(times_sorted) - 1))]
    return {
        "min": min(times),
        "mean": sum(times) / len(times),
        "p95": p95,
        "mem": sum(mem_deltas) / len(mem_deltas),
        "cpu": sum(cpu_usages) / len(cpu_usages),
    }


def run_ekilang_benchmark(
    name: str, filepath: str | Path, runs: int = 5, warmup: int = 0, fast: bool = False
) -> Dict[str, Any]:
    """Run a Ekilang benchmark file multiple times and report aggregated metrics"""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    file_path = Path(filepath)

    with open(file_path, "r", encoding="utf-8") as f:
        code = f.read()

    parse_start = time.perf_counter()
    tokens = Lexer(code).tokenize()
    mod = Parser(tokens).parse()
    parse_time = time.perf_counter() - parse_start

    has_use = any(isinstance(stmt, Use) for stmt in getattr(mod, "body", []))
    code_obj = None
    compile_time = None
    shared_ns = None
    fast_prog: Optional[FastProgram] = None
    if not has_use:
        compile_start = time.perf_counter()
        code_obj = compile_module(mod)
        compile_time = time.perf_counter() - compile_start
        shared_ns = dict(BUILTINS)
        shared_ns["__ekilang_builtins_loaded__"] = True
        if fast:
            fast_prog = compile_simple_loop(mod)

    def exec_cached() -> None:
        if fast and fast_prog is not None and shared_ns is not None:
            run_fast(fast_prog, shared_ns)
        elif code_obj:
            execute(mod, globals_ns=shared_ns, code_obj=code_obj)
        else:
            execute(mod)

    agg = run_repeated(exec_cached, runs=runs, warmup=warmup)

    print(
        f"  Exec Time   min/mean/p95: {agg['min']:.3f}s / {agg['mean']:.3f}s / {agg['p95']:.3f}s"
    )
    print(f"  Memory Delta (avg): {agg['mem']:.2f} MB")
    print(f"  CPU Usage (avg during exec): {agg['cpu']:.1f}%")
    print(f"{'='*60}")

    return {
        "name": name,
        "parse_time": parse_time,
        "compile_time": compile_time,
        "time_min": agg["min"],
        "time_mean": agg["mean"],
        "time_p95": agg["p95"],
        "memory": agg["mem"],
        "cpu": agg["cpu"],
        "fast": bool(fast and fast_prog is not None),
    }


def count_primes_py(limit: int) -> int:
    """Count number of primes up to limit in pure Python"""
    count: int = 0
    n: int = 2
    while n <= limit:
        is_p: bool = True
        d: int = 2
        while d * d <= n:
            if n % d == 0:
                is_p = False
            d += 1
        if is_p:
            count += 1
        n += 1
    return count


def run_python_workload() -> Dict[str, int]:
    """Run equivalent workload in pure Python for comparison"""
    prime_count: int = count_primes_py(1000)

    total: int = 0
    i: int = 0
    while i < 10000:
        total += i
        i += 1

    nested_sum: int = 0
    m: int = 0
    while m < 100:
        n: int = 0
        while n < 100:
            nested_sum += m + n
            n += 1
        m += 1

    text: str = "Benchmark"
    k: int = 0
    while k < 200:
        text = f"{text}!"
        k += 1

    acc: int = 1
    idx: int = 0
    while idx < 200000:
        acc = (acc * 3 + idx) % 10000019
        idx += 1

    deep_total: int = 0
    a: int = 0
    while a < 150:
        b: int = 0
        while b < 150:
            deep_total += (a * b) % 17
            b += 1
        a += 1

    return {
        "prime_count": prime_count,
        "total": total,
        "nested_sum": nested_sum,
        "text_length": len(text),
        "acc": acc,
        "deep_total": deep_total,
    }


def run_python_benchmark(name: str, runs: int = 5, warmup: int = 0) -> Dict[str, Any]:
    """Run equivalent workload in pure Python for comparison"""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    agg = run_repeated(run_python_workload, runs=runs, warmup=warmup)

    print(
        f"  Exec Time   min/mean/p95: {agg['min']:.3f}s / {agg['mean']:.3f}s / {agg['p95']:.3f}s"
    )
    print(f"  Memory Delta (avg): {agg['mem']:.2f} MB")
    print(f"  CPU Usage (avg during exec): {agg['cpu']:.1f}%")
    print("  Results shown from last run above")
    print(f"{'='*60}")

    return {
        "name": name,
        "time_min": agg["min"],
        "time_mean": agg["mean"],
        "time_p95": agg["p95"],
        "memory": agg["mem"],
        "cpu": agg["cpu"],
    }


# --- N-Queens Python version for fair comparison ---
def is_safe_py(queens: List[int], row: int, col: int) -> bool:
    """Check if placing a queen at (row, col) is safe given current queens' positions."""
    for i in range(row):
        qcol: int = queens[i]
        if qcol == col or abs(qcol - col) == row - i:
            return False
    return True


def solve_nqueens_py(n: int, row: int, queens: List[int], count: List[int]) -> None:
    """Recursive backtracking solver for N-Queens problem."""
    if row == n:
        count[0] += 1
        return
    for col in range(n):
        if is_safe_py(queens, row, col):
            queens[row] = col
            solve_nqueens_py(n, row + 1, queens, count)


def nqueens_bench_py(n: int) -> int:
    """Count number of solutions to N-Queens problem in pure Python."""
    queens: List[int] = [0] * n
    count: List[int] = [0]
    solve_nqueens_py(n, 0, queens, count)
    return count[0]


def run_python_nqueens_benchmark(
    name: str, N: int = 12, runs: int = 3, warmup: int = 1
) -> Dict[str, Any]:
    """Run N-Queens benchmark in pure Python for comparison"""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    def run() -> None:
        result: int = nqueens_bench_py(N)
        # Print only once for clarity
        if runs == 1:
            print(f"N-Queens solutions for N={N}: {result}")

    agg = run_repeated(run, runs=runs, warmup=warmup)
    print(
        f"  Exec Time   min/mean/p95: {agg['min']:.3f}s / {agg['mean']:.3f}s / {agg['p95']:.3f}s"
    )
    print(f"  Memory Delta (avg): {agg['mem']:.2f} MB")
    print(f"  CPU Usage (avg during exec): {agg['cpu']:.1f}%")
    print(f"{'='*60}")

    return {
        "name": name,
        "time_min": agg["min"],
        "time_mean": agg["mean"],
        "time_p95": agg["p95"],
        "memory": agg["mem"],
        "cpu": agg["cpu"],
    }


def main() -> None:
    """Main function to run benchmarks based on command-line arguments."""
    parser = argparse.ArgumentParser(description="Ekilang vs Python benchmark")
    parser.add_argument(
        "--runs", type=int, default=5, help="Measured runs per language"
    )
    parser.add_argument(
        "--warmup", type=int, default=0, help="Warmup runs (not measured)"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fast-path executor when available (no use imports)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  EKILANG BENCHMARK SUITE")
    print("=" * 60)
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  RAM: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.2f} GB")
    print(f"  CPUs: {psutil.cpu_count()}")

    results: List[Dict[str, Any]] = []

    bench_file = Path(__file__).parent.parent / "examples" / "benchmark_intense.eki"
    if bench_file.exists():
        mod_result = run_ekilang_benchmark(
            "Ekilang: Intense Benchmark (CPU + Memory)",
            bench_file,
            runs=args.runs,
            warmup=args.warmup,
            fast=args.fast,
        )
        if mod_result:
            results.append(mod_result)

    py_result = run_python_benchmark(
        "Python: Intense Benchmark (CPU + Memory)", runs=args.runs, warmup=args.warmup
    )
    if py_result:
        results.append(py_result)

    # --- N-Queens Benchmarks ---
    nqueens_file = Path(__file__).parent.parent / "examples" / "benchmark_nqueens.eki"
    if nqueens_file.exists():
        ny_result = run_ekilang_benchmark(
            "Ekilang: N-Queens Benchmark (N=12)",
            nqueens_file,
            runs=args.runs,
            warmup=args.warmup,
            fast=args.fast,
        )
        if ny_result:
            results.append(ny_result)

    py_result = run_python_nqueens_benchmark(
        "Python: N-Queens Benchmark (N=12)", N=12, runs=args.runs, warmup=args.warmup
    )
    if py_result:
        results.append(py_result)

    if results:
        print("\n" + "=" * 60)
        print("  SUMMARY (min / mean / p95; avg mem)")
        print("=" * 60)
        for r in results:
            if "parse_time" in r:
                compile_str = (
                    f" comp {r['compile_time']:.3f}s |"
                    if r.get("compile_time") is not None
                    else ""
                )
                fast_str = " fast" if r.get("fast") else ""
                print(
                    f"  {r['name']:<32} parse {r['parse_time']:.3f}s |{compile_str} exec {r['time_min']:.3f}/{r['time_mean']:.3f}/{r['time_p95']:.3f}s | mem {r['memory']:.2f}MB{fast_str}"
                )

                # Calculate overhead vs Python equivalent
                py_match = None
                for pr in results:
                    if (
                        not "parse_time" in pr
                        and r["name"].replace("Ekilang: ", "Python: ") == pr["name"]
                    ):
                        py_match = pr
                        break
                if py_match:
                    time_overhead = (
                        (r["time_mean"] - py_match["time_mean"]) / py_match["time_mean"]
                    ) * 100
                    mem_overhead = (
                        ((r["memory"] - py_match["memory"]) / py_match["memory"]) * 100
                        if py_match["memory"] != 0
                        else 0
                    )
                    cpu_overhead = (
                        ((r["cpu"] - py_match["cpu"]) / py_match["cpu"]) * 100
                        if py_match["cpu"] != 0
                        else 0
                    )
                    print(
                        f"    Overhead vs Python: {time_overhead:+.1f}% time, {mem_overhead:+.1f}% mem, {cpu_overhead:+.1f}% cpu"
                    )
            else:
                print(
                    f"  {r['name']:<32} exec {r['time_min']:.3f}/{r['time_mean']:.3f}/{r['time_p95']:.3f}s | mem {r['memory']:.2f}MB"
                )
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

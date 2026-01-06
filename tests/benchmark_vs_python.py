"""
Ekilang vs Python Performance Benchmark
Compares execution time for common tasks
"""

import time
import sys
from pathlib import Path
from typing import List
from ekilang.lexer import tokenize
from ekilang.parser import Parser
from ekilang.runtime import compile_module

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_ekilang(code: str) -> None:
    """Execute Ekilang code"""
    tokens = tokenize(code)
    ast = Parser(tokens).parse()
    code_obj = compile_module(ast)
    exec(code_obj)


def benchmark(name: str, eki_code: str, py_code: str, runs: int = 3) -> None:
    """Benchmark Ekilang vs Python code"""
    print(f"\n{'='*60}")
    print(f"Benchmark: {name}")
    print("=" * 60)

    # Warmup
    try:
        for _ in range(1):
            run_ekilang(eki_code)
    except Exception as e:
        print(f"Ekilang warmup error: {e}")
        return

    try:
        for _ in range(1):
            exec(py_code)
    except Exception as e:
        print(f"Python warmup error: {e}")
        return

    # Ekilang benchmark
    eki_times: list[float] = []
    for _ in range(runs):
        try:
            t0: float = time.perf_counter()
            run_ekilang(eki_code)
            eki_times.append(time.perf_counter() - t0)
        except Exception as e:
            print(f"Ekilang error: {e}")
            return

    # Python benchmark
    py_times: List[float] = []
    for _ in range(runs):
        try:
            t0 = time.perf_counter()
            exec(py_code)
            py_times.append(time.perf_counter() - t0)
        except Exception as e:
            print(f"Python error: {e}")
            return

    eki_avg: float = sum(eki_times) / len(eki_times)
    py_avg: float = sum(py_times) / len(py_times)
    ratio: float = eki_avg / py_avg if py_avg > 0 else float("inf")

    print(f"Ekilang: {eki_avg*1000:.3f}ms (avg of {runs} runs)")
    print(f"Python:  {py_avg*1000:.3f}ms (avg of {runs} runs)")
    print(f"Ratio:   {ratio:.2f}x {'slower' if ratio > 1 else 'faster'}")


# ============================================================================
# 1. ARITHMETIC & LOOPS
# ============================================================================

benchmark(
    "Simple Loop (1M iterations)",
    """x = 0
for i in range(1_000_000) { x = x + i }
print(x)""",
    """x = 0
for i in range(1_000_000):
    x = x + i
print(x)""",
    runs=2,
)


# ============================================================================
# 2. LIST COMPREHENSION
# ============================================================================

benchmark(
    "List Comprehension (100K items)",
    """result = [x * 2 for x in range(100_000)]
print(len(result))""",
    """result = [x * 2 for x in range(100_000)]
print(len(result))""",
)


# ============================================================================
# 3. FUNCTION CALLS (using block syntax)
# ============================================================================

benchmark(
    "Function Call (sum of range)",
    """fn sum_range(n) {
    s = 0
    for i in range(n) { s = s + i }
    s
}
result = sum_range(100_000)
print(result)""",
    """def sum_range(n):
    s = 0
    for i in range(n):
        s = s + i
    return s

result = sum_range(100_000)
print(result)""",
)


# ============================================================================
# 4. DICT OPERATIONS
# ============================================================================

benchmark(
    "Dictionary Creation & Access (50K items)",
    """d = {}
for i in range(50_000) { d[i] = i * 2 }
result = sum(d.values())
print(result)""",
    """d = {}
for i in range(50_000):
    d[i] = i * 2
result = sum(d.values())
print(result)""",
)


# ============================================================================
# 5. FILTER & MAP OPERATIONS
# ============================================================================

benchmark(
    "Filter & Map (50K items)",
    """nums = range(50_000)
filtered = [x for x in nums if x % 2 == 0]
mapped = [x * 3 for x in filtered]
print(len(mapped))""",
    """nums = range(50_000)
filtered = [x for x in nums if x % 2 == 0]
mapped = [x * 3 for x in filtered]
print(len(mapped))""",
)


# ============================================================================
# 6. LAMBDA & HIGHER-ORDER FUNCTIONS
# ============================================================================

benchmark(
    "Lambda Functions (100K items)",
    """square = (x) => x * x
nums = [1, 2, 3, 4, 5] * 20_000
result = sum([square(x) for x in nums])
print(result)""",
    """square = lambda x: x * x
nums = [1, 2, 3, 4, 5] * 20_000
result = sum([square(x) for x in nums])
print(result)""",
)


# ============================================================================
# 7. TRY/EXCEPT HANDLING
# ============================================================================

benchmark(
    "Try/Except Blocks (5K iterations)",
    """count = 0
for i in range(5_000) {
    try { if i % 3 == 0 { raise ValueError("test") } else { count = count + 1 } }
    except ValueError { }
}
print(count)""",
    """count = 0
for i in range(5_000):
    try:
        if i % 3 == 0:
            raise ValueError("test")
        count = count + 1
    except ValueError:
        pass
print(count)""",
)


# ============================================================================
# 8. NESTED LOOPS
# ============================================================================

benchmark(
    "Nested Loops (1000x1000)",
    """total = 0
for i in range(1_000) {
    for j in range(1_000) { total = total + i + j }
}
print(total)""",
    """total = 0
for i in range(1_000):
    for j in range(1_000):
        total = total + i + j
print(total)""",
    runs=2,
)


# ============================================================================
# 9. TERNARY OPERATIONS
# ============================================================================

benchmark(
    "Ternary Operator (100K iterations)",
    """result = [i if i % 2 == 0 else (-i) for i in range(100_000)]
print(len(result))""",
    """result = [i if i % 2 == 0 else -i for i in range(100_000)]
print(len(result))""",
)


# ============================================================================
# 10. SET OPERATIONS
# ============================================================================

benchmark(
    "Set Comprehension (50K items)",
    """s = {x * 2 for x in range(50_000)}
print(len(s))""",
    """s = {x * 2 for x in range(50_000)}
print(len(s))""",
)


print("\n" + "=" * 60)
print("Benchmark Complete!")
print("=" * 60)

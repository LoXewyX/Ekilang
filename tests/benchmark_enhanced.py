"""
Enhanced Ekilang Benchmarks - Focused on Parser and Runtime Performance
Tests areas where Rust optimizations can make a difference
"""

import time
import sys
import traceback
from pathlib import Path
from typing import Any, List
from ekilang.lexer import tokenize
from ekilang.parser import Parser
from ekilang.runtime import compile_module
from ekilang.executor import execute
from ekilang.types import Module

sys.path.insert(0, str(Path(__file__).parent.parent))


def compile_ekilang(code: str) -> tuple[Module, Any]:
    """Compile Ekilang code and return (ast, code_obj)."""
    # Allow printing of very large integers in benchmarks (Python 3.11+)
    if hasattr(sys, "set_int_max_str_digits"):
        sys.set_int_max_str_digits(0)
    tokens = tokenize(code)
    ast = Parser(tokens).parse()
    code_obj = compile_module(ast)
    return ast, code_obj


def run_ekilang_compiled(ast: Module, code_obj: Any) -> None:
    """Execute pre-compiled Ekilang code."""
    try:
        execute(ast, code_obj=code_obj)
    except Exception as e:
        print(f"Error in run_ekilang_compiled: {type(e).__name__}: {e}")
        traceback.print_exc()
        raise


def benchmark(name: str, eki_code: str, py_code: str, runs: int = 3) -> None:
    """Benchmark Ekilang vs Python code, separating compilation and execution."""
    print(f"\n{'='*60}")
    print(f"Benchmark: {name}")
    print("=" * 60)

    # Compile Ekilang code (outside timing)
    try:
        eki_ast, eki_code_obj = compile_ekilang(eki_code)
    except Exception as e:
        print(f"Ekilang compilation error: {type(e).__name__}: {e}")
        return

    # Warmup
    try:
        run_ekilang_compiled(eki_ast, eki_code_obj)
    except Exception as e:
        print(f"Ekilang warmup error: {type(e).__name__}: {e}")
        return

    try:
        exec(py_code, {})
    except Exception as e:
        print(f"Python warmup error: {type(e).__name__}: {e}")
        return

    # Time Ekilang execution only (code already compiled)
    eki_exec_times: list[float] = []
    for _ in range(runs):
        try:
            t0: float = time.perf_counter()
            run_ekilang_compiled(eki_ast, eki_code_obj)
            eki_exec_times.append(time.perf_counter() - t0)
        except Exception as e:
            print(f"Ekilang execution error: {type(e).__name__}: {e}")
            return

    # Time Python execution
    py_exec_times: List[float] = []
    for _ in range(runs):
        try:
            t0 = time.perf_counter()
            exec(py_code, {})
            py_exec_times.append(time.perf_counter() - t0)
        except Exception as e:
            print(f"Python execution error: {type(e).__name__}: {e}")
            return

    eki_exec_avg: float = sum(eki_exec_times) / len(eki_exec_times)
    py_exec_avg: float = sum(py_exec_times) / len(py_exec_times)
    exec_ratio: float = eki_exec_avg / py_exec_avg if py_exec_avg > 0 else float("inf")

    print(f"Execution Time (compiled code only):")
    print(f"  Ekilang: {eki_exec_avg*1000:.3f}ms (avg of {runs} runs)")
    print(f"  Python:  {py_exec_avg*1000:.3f}ms (avg of {runs} runs)")
    
    # Display ratio: when Ekilang is faster, show inverse ratio for clarity
    if exec_ratio < 1:
        display_ratio = 1 / exec_ratio
        print(f"  Ratio:   {display_ratio:.2f}x faster")
    else:
        print(f"  Ratio:   {exec_ratio:.2f}x slower")


# ============================================================================
# SIMPLE OPERATIONS (BASELINE)
# ============================================================================

benchmark(
    "Simple Integer Arithmetic (100K iterations)",
    """result = 0
for i in range(100_000) {
    result = result + i
    result = result - (i // 2)
}
print(result)""",
    """result = 0
for i in range(100_000):
    result = result + i
    result = result - (i // 2)

print(result)""",
    runs=3,
)

# ============================================================================
# BINARY OPERATORS (100K iterations)
# ============================================================================

benchmark(
    "Binary Operations (100K iterations)",
    """result = 0
for i in range(100_000) {
    result = result * 2 + i % 10
    result = result >> 1
    result = result & 255
}
print(result)""",
    """result = 0
for i in range(100_000):
    result = result * 2 + i % 10
    result = result >> 1
    result = result & 255

print(result)""",
    runs=3,
)

# ============================================================================
# LIST OPERATIONS
# ============================================================================

benchmark(
    "List Operations (50K iterations)",
    """items = []
for i in range(50_000) {
    items.append(i)
    if i % 100 == 0 {
        items.pop()
    }
}
print(len(items))""",
    """items = []
for i in range(50_000):
    items.append(i)
    if i % 100 == 0:
        items.pop()

print(len(items))""",
    runs=3,
)

# ============================================================================
# ASYNC/AWAIT OPERATIONS
# ============================================================================

benchmark(
    "Async Function Declaration (5K iterations)",
    """async fn fetch(x) {
    x * 2
}

result = 0
for i in range(5_000) {
    result = result + i
}
print(result)""",
    """async def fetch(x):
    x * 2

result = 0
for i in range(5_000):
    result = result + i

print(result)""",
    runs=3,
)

benchmark(
    "Async Multiple Functions (2K iterations)",
    """async fn task1(x) {
    x + 10
}

async fn task2(x) {
    x * 3
}

result = 0
for i in range(2_000) {
    result = result + 1
}
print(result)""",
    """async def task1(x):
    x + 10

async def task2(x):
    x * 3

result = 0
for i in range(2_000):
    result = result + 1

print(result)""",
    runs=3,
)

# ============================================================================
# ASYNC CONTEXT MANAGERS / ASYNC ITERATION
# ============================================================================

benchmark(
    "Async With + Async For (streaming 500 items)",
    """use asyncio

class ChunkStream {
    fn __init__(self, chunks) {
        self.chunks = chunks
    }
    fn __aiter__(self) {
        async fn iter_chunks(chunks) {
            for chunk in chunks {
                yield chunk
            }
        }
        iter_chunks(self.chunks)
    }
}

class Response {
    fn __init__(self, chunks) {
        self.content = ChunkStream(chunks)
    }
    async fn __aenter__(self) { self }
    async fn __aexit__(self, exc_type, exc, tb) { none }
}

class Session {
    async fn get(self, n) {
        # simulate n small chunks
        chunks = [f"c{i}" for i in range(n)]
        Response(chunks)
    }
}

async fn main() {
    session = Session()
    total = 0
    async with await session.get(500) as resp {
        async for chunk in resp.content {
            total = total + len(chunk)
        }
    }
    total
}

result = asyncio.run(main())
print(result)""",
    """import asyncio

class ChunkStream:
    def __init__(self, chunks):
        self.chunks = chunks
    def __aiter__(self):
        async def iter_chunks(chunks):
            for chunk in chunks:
                yield chunk
        return iter_chunks(self.chunks)

class Response:
    def __init__(self, chunks):
        self.content = ChunkStream(chunks)
    async def __aenter__(self):
        return self
    async def __aexit__(self, exc_type, exc, tb):
        return None

class Session:
    async def get(self, n):
        chunks = [f"c{i}" for i in range(n)]
        return Response(chunks)

async def main():
    session = Session()
    total = 0
    async with await session.get(500) as resp:
        async for chunk in resp.content:
            total = total + len(chunk)
    return total

result = asyncio.run(main())
print(result)""",
    runs=3,
)

# ============================================================================
# CONTEXT MANAGER OPERATIONS
# ============================================================================

benchmark(
    "With Statement (1000 iterations)",
    """use contextlib

@contextlib.contextmanager
fn simple_context() {
    yield
}

count = 0
for i in range(1_000) {
    with simple_context() {
        count = count + 1
    }
}
print(count)""",
    """import contextlib

@contextlib.contextmanager
def simple_context():
    yield

count = 0
for i in range(1_000):
    with simple_context():
        count = count + 1

print(count)""",
    runs=3,
)

benchmark(
    "Multiple Context Managers (500 iterations)",
    """use contextlib

@contextlib.contextmanager
fn ctx1() {
    yield "a"
}

@contextlib.contextmanager
fn ctx2() {
    yield "b"
}

result = ""
for i in range(500) {
    with ctx1() as a, ctx2() as b {
        result = a + b
    }
}
print(result)""",
    """import contextlib

@contextlib.contextmanager
def ctx1():
    yield "a"

@contextlib.contextmanager
def ctx2():
    yield "b"

result = ""
for i in range(500):
    with ctx1() as a, ctx2() as b:
        result = a + b

print(result)""",
    runs=3,
)

benchmark(
    "Nested Context Managers (500 iterations)",
    """use contextlib

@contextlib.contextmanager
fn outer() {
    yield 1
}

@contextlib.contextmanager
fn inner() {
    yield 2
}

total = 0
for i in range(500) {
    with outer() as x {
        with inner() as y {
            total = total + x + y
        }
    }
}
print(total)""",
    """import contextlib

@contextlib.contextmanager
def outer():
    yield 1

@contextlib.contextmanager
def inner():
    yield 2

total = 0
for i in range(500):
    with outer() as x:
        with inner() as y:
            total = total + x + y

print(total)""",
    runs=3,
)

# ============================================================================
# OPERATOR OPTIMIZATIONS
# ============================================================================

benchmark(
    "Bitwise Operations (100K iterations)",
    """MASK = 0xFFFFFFFFFFFFFFFF
result = 0
for i in range(100_000) {
    result = ((result << 1) | (i & 3)) & MASK
    result = (result ^ 0xFF) & MASK
}
# Print low 16 hex digits for readability
print(hex(result & 0xFFFF_FFFF_FFFF_FFFF)[-16:])""",
    """MASK = 0xFFFFFFFFFFFFFFFF
result = 0
for i in range(100_000):
    result = ((result << 1) | (i & 3)) & MASK
    result = (result ^ 0xFF) & MASK

# Print low 16 hex digits for readability
print(hex(result & 0xFFFF_FFFF_FFFF_FFFF)[-16:])""",
    runs=3,
)

benchmark(
    "Power Operations (50K iterations)",
    """result = 1
for i in range(50_000) {
    result = result * 2
    if result > 1_000_000 {
        result = result // 100
    }
}
print(result)""",
    """result = 1
for i in range(50_000):
    result = result * 2
    if result > 1_000_000:
        result = result // 100

print(result)""",
    runs=3,
)

benchmark(
    "Boolean Logic Operations (100K iterations)",
    """count = 0
for i in range(100_000) {
    if (i % 2 == 0) and (i % 3 == 0) {
        count = count + 1
    }
    if (i % 5 == 0) or (i % 7 == 0) {
        count = count + 1
    }
}
print(count)""",
    """count = 0
for i in range(100_000):
    if (i % 2 == 0) and (i % 3 == 0):
        count = count + 1
    if (i % 5 == 0) or (i % 7 == 0):
        count = count + 1

print(count)""",
    runs=3,
)

# ============================================================================
# DICTIONARY OPERATIONS
# ============================================================================

benchmark(
    "Comparison Operations (100K iterations)",
    """result = 0
for i in range(100_000) {
    if i > 50 {
        result = result + i
    }
    if i < 30_000 {
        result = result - 1
    }
}
print(result)""",
    """result = 0
for i in range(100_000):
    if i > 50:
        result = result + i
    if i < 30_000:
        result = result - 1

print(result)""",
    runs=3,
)

# ============================================================================
# FUNCTION CALLS
# ============================================================================

benchmark(
    "Function Calls (10K iterations)",
    """fn add(a, b) {
    return a + b
}

result = 0
for i in range(10_000) {
    result = add(result, i)
}
print(result)""",
    """def add(a, b):
    return a + b

result = 0
for i in range(10_000):
    result = add(result, i)

print(result)""",
    runs=3,
)

# ============================================================================
# STRING OPERATIONS
# ============================================================================

benchmark(
    "String Concatenation (10K iterations)",
    """result = ""
for i in range(10_000) {
    result = result + str(i)
}
print(len(result))""",
    """result = ""
for i in range(10_000):
    result = result + str(i)

print(len(result))""",
    runs=3,
)

# ============================================================================
# DICTIONARY OPERATIONS
# ============================================================================

benchmark(
    "Dictionary Operations (5K iterations)",
    """data = dict()
for i in range(5_000) {
    key = f"key_{i}"
    data[key] = i * 2
}
print(len(data))""",
    """data = {}
for i in range(5_000):
    key = f"key_{i}"
    data[key] = i * 2

print(len(data))""",
    runs=3,
)

# ============================================================================
# LIST COMPREHENSION
# ============================================================================

benchmark(
    "List Comprehension (10K iterations)",
    """items = [x * x for x in range(10_000)]
print(len(items))""",
    """items = [x * x for x in range(10_000)]
print(len(items))""",
    runs=3,
)

# ============================================================================
# LAMBDA/CLOSURE
# ============================================================================

benchmark(
    "Lambda Functions (5K iterations)",
    """double = (x) => { x * 2 }
result = 0
for i in range(5_000) {
    result = result + double(i)
}
print(result)""",
    """double = lambda x: x * 2
result = 0
for i in range(5_000):
    result = result + double(i)

print(result)""",
    runs=3,
)

# ============================================================================
# NESTED LOOPS
# ============================================================================

benchmark(
    "Nested Loops (100x100)",
    """total = 0
for i in range(100) {
    for j in range(100) {
        total = total + i * j
    }
}
print(total)""",
    """total = 0
for i in range(100):
    for j in range(100):
        total = total + i * j

print(total)""",
    runs=3,
)

# ============================================================================
# TUPLE OPERATIONS
# ============================================================================

benchmark(
    "Tuple Operations (5K iterations)",
    """tuples = []
for i in range(5_000) {
    t = (i, i+1, i+2)
    tuples.append(t)
}
print(len(tuples))""",
    """tuples = []
for i in range(5_000):
    t = (i, i+1, i+2)
    tuples.append(t)

print(len(tuples))""",
    runs=3,
)

# ============================================================================
# RECURSION
# ============================================================================

benchmark(
    "Recursion (fibonacci up to 25)",
    """fn fib(n) {
    if n <= 1 {
        return n
    }
    return fib(n - 1) + fib(n - 2)
}

result = fib(25)
print(result)""",
    """def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)

result = fib(25)
print(result)""",
    runs=1,
)

# ============================================================================
# TYPE CONVERSION
# ============================================================================

benchmark(
    "Type Conversions (10K iterations)",
    """result = 0
for i in range(10_000) {
    s = str(i)
    n = int(s)
    result = result + n
}
print(result)""",
    """result = 0
for i in range(10_000):
    s = str(i)
    n = int(s)
    result = result + n

print(result)""",
    runs=3,
)

# ============================================================================
# F-STRING FORMATTING
# ============================================================================

benchmark(
    "F-String Formatting (5K iterations)",
    """result = ""
for i in range(5_000) {
    s = f"Value: {i}, Double: {i*2}"
    result = result + s
}
print(len(result))""",
    """result = ""
for i in range(5_000):
    s = f"Value: {i}, Double: {i*2}"
    result = result + s

print(len(result))""",
    runs=3,
)

print("\n" + "=" * 60)
print("Enhanced Benchmark Complete!")
print("=" * 60)

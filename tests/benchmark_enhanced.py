"""
Enhanced Ekilang Benchmarks - Focused on Parser and Runtime Performance
Tests areas where Rust optimizations can make a difference
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
# ASYNC/AWAIT OPERATIONS (PARSING PERFORMANCE)
# ============================================================================

benchmark(
    "Async String Operations (1K iterations)",
    """fn process_strings() {
  code = \"async fn compute(n) { n * 2 }\"
  results = []
  
  for i in range(1000) {
    modified = code + str(i)
    if \"async\" in modified {
      results.append(len(modified))
    }
  }
  
  print(len(results))
}

process_strings()""",
    """def process_strings():
  code = "async def compute(n): return n * 2"
  results = []
  
  for i in range(1000):
    modified = code + str(i)
    if "async" in modified:
      results.append(len(modified))
  
  print(len(results))

process_strings()""",
    runs=3
)

benchmark(
    "Async Filtering (50K patterns)",
    """fn filter_async() {
  patterns = [
    \"async fn\",
    \"await \",
    \"fn \",
    \"def \"
  ]
  
  count = 0
  for i in range(50000) {
    p = patterns[i % 4]
    if p.startswith(\"async\") {
      count = count + 1
    }
  }
  
  print(count)
}

filter_async()""",
    """def filter_async():
  patterns = [
    "async def",
    "await ",
    "def ",
    "def "
  ]
  
  count = 0
  for i in range(50000):
    p = patterns[i % 4]
    if p.startswith("async"):
      count = count + 1
  
  print(count)

filter_async()""",
    runs=3
)

benchmark(
    "Await Calls (25K operations)",
    """fn simulate_await() {
  results = []
  
  for i in range(25000) {
    x = i * 2 + 1
    y = x % 7
    z = y + i
    results.append(z)
  }
  
  print(len(results))
}

simulate_await()""",
    """def simulate_await():
  results = []
  
  for i in range(25000):
    x = i * 2 + 1
    y = x % 7
    z = y + i
    results.append(z)
  
  print(len(results))

simulate_await()""",
    runs=3
)

benchmark(
    "Keyword Matching (100K iterations)",
    """fn match_keywords() {
  async_keywords = [
    \"async\",
    \"await\",
    \"fn\",
    \"yield\",
    \"with\",
    \"try\",
    \"except\",
    \"finally\"
  ]
  
  count = 0
  for i in range(100000) {
    kw = async_keywords[i % 8]
    if kw == \"async\" {
      count = count + 1
    }
  }
  
  print(count)
}

match_keywords()""",
    """def match_keywords():
  async_keywords = [
    "async",
    "await",
    "def",
    "yield",
    "with",
    "try",
    "except",
    "finally"
  ]
  
  count = 0
  for i in range(100000):
    kw = async_keywords[i % 8]
    if kw == "async":
      count = count + 1
  
  print(count)

match_keywords()""",
    runs=3
)

# ============================================================================
# AUGMENTED ASSIGNMENT OPERATORS (Parser Optimization Target)
# ============================================================================

benchmark(
    "Augmented Assignment Ops (100K iterations)",
    """x = 0
y = 10
z = 5

for i in range(100_000) {
    x += i
    y *= 2
    z -= 1
    y //= 3
}
print(x + y + z)""",
    """x = 0
y = 10
z = 5

for i in range(100_000):
    x += i
    y *= 2
    z -= 1
    y //= 3

print(x + y + z)""",
    runs=2,
)


# ============================================================================
# BITWISE OPERATIONS (Parser Classification)
# ============================================================================

benchmark(
    "Bitwise Operations (50K iterations)",
    """result = 0
for i in range(50_000) {
    result = result | (i & 0xFF)
    result = result ^ (i << 2)
    result = result & (i >> 1)
}
print(result)""",
    """result = 0
for i in range(50_000):
    result = result | (i & 0xFF)
    result = result ^ (i << 2)
    result = result & (i >> 1)

print(result)""",
)


# ============================================================================
# COMPARISON CHAINS (Parser Optimization)
# ============================================================================

benchmark(
    "Chained Comparisons (50K iterations)",
    """count = 0
for i in range(50_000) {
    if 0 < i < 25_000 { count = count + 1 }
    if i >= 10 and i <= 30_000 { count = count + 1 }
}
print(count)""",
    """count = 0
for i in range(50_000):
    if 0 < i < 25_000:
        count = count + 1
    if i >= 10 and i <= 30_000:
        count = count + 1

print(count)""",
)


# ============================================================================
# OPERATOR PRECEDENCE (Parser Hot Path)
# ============================================================================

benchmark(
    "Complex Expressions (25K iterations)",
    """result = 0
for i in range(25_000) {
    result = result + i * 2 ** 3 - i // 2 + i % 3
}
print(result)""",
    """result = 0
for i in range(25_000):
    result = result + i * 2 ** 3 - i // 2 + i % 3

print(result)""",
)


# ============================================================================
# PIPELINE OPERATORS (Unique to Ekilang)
# ============================================================================

benchmark(
    "Pipeline Operators (10K iterations)",
    """fn double(x) { x * 2 }
fn add_ten(x) { x + 10 }
fn square(x) { x * x }

results = []
for i in range(10_000) {
    result = i |> double |> add_ten |> square
    results.append(result)
}
print(sum(results))""",
    """def double(x):
    return x * 2

def add_ten(x):
    return x + 10

def square(x):
    return x * x

results = []
for i in range(10_000):
    result = square(add_ten(double(i)))
    results.append(result)

print(sum(results))""",
)


# ============================================================================
# LAMBDA EXPRESSIONS & CLOSURES
# ============================================================================

benchmark(
    "Lambda with Closures (20K iterations)",
    """fn make_adder(n) {
    (x) => x + n
}

add5 = make_adder(5)
add10 = make_adder(10)

result = 0
for i in range(20_000) {
    result = result + add5(i) + add10(i)
}
print(result)""",
    """def make_adder(n):
    return lambda x: x + n

add5 = make_adder(5)
add10 = make_adder(10)

result = 0
for i in range(20_000):
    result = result + add5(i) + add10(i)

print(result)""",
)


# ============================================================================
# MATCH STATEMENTS (Pattern Matching)
# ============================================================================

benchmark(
    "Match Pattern Matching (10K iterations)",
    """fn classify(n) {
    match n % 4 {
        0 => {
            "divisible by 4"
        }
        1 => {
            "remainder 1"
        }
        2 => {
            "remainder 2"
        }
        _ => {
            "remainder 3"
        }
    }
}

results = []
for i in range(10_000) {
    results.append(classify(i))
}
print(len(results))""",
    """def classify(n):
    match n % 4:
        case 0:
            return "divisible by 4"
        case 1:
            return "remainder 1"
        case 2:
            return "remainder 2"
        case _:
            return "remainder 3"

results = []
for i in range(10_000):
    results.append(classify(i))

print(len(results))""",
)


# ============================================================================
# CLASS INSTANTIATION & METHOD CALLS
# ============================================================================

benchmark(
    "Class Methods (5K objects)",
    """class Counter {
    fn __init__(self, start) {
        self.value = start
    }
    
    fn increment(self) {
        self.value += 1
        self.value
    }
    
    fn get(self) {
        self.value
    }
}

counters = [Counter(i) for i in range(5_000)]
for c in counters {
    c.increment()
    c.increment()
}
total = sum([c.get() for c in counters])
print(total)""",
    """class Counter:
    def __init__(self, start):
        self.value = start
    
    def increment(self):
        self.value += 1
        return self.value
    
    def get(self):
        return self.value

counters = [Counter(i) for i in range(5_000)]
for c in counters:
    c.increment()
    c.increment()

total = sum([c.get() for c in counters])
print(total)""",
)


# ============================================================================
# UNARY OPERATORS
# ============================================================================

benchmark(
    "Unary Operations (50K iterations)",
    """result = 0
for i in range(1, 50_000) {
    result = result + (-i)
    result = result + (~i & 0xFFFF)
}
print(result)""",
    """result = 0
for i in range(1, 50_000):
    result = result + (-i)
    result = result + (~i & 0xFFFF)

print(result)""",
)


# ============================================================================
# WALRUS OPERATOR (Named Expressions)
# ============================================================================

benchmark(
    "Walrus Operator (25K iterations)",
    """results = []
for i in range(25_000) {
    if (x := i * 2) > 10_000 {
        results.append(x)
    }
}
print(len(results))""",
    """results = []
for i in range(25_000):
    if (x := i * 2) > 10_000:
        results.append(x)

print(len(results))""",
)


# ============================================================================
# DECORATOR PERFORMANCE
# ============================================================================

benchmark(
    "Decorated Functions (10K calls)",
    """fn timing_decorator(func) {
    fn wrapper(*args, **kwargs) {
        func(*args, **kwargs)
    }
    wrapper
}

@timing_decorator
fn compute(x) {
    x * x
}

for i in range(10_000) {
    compute(i)
}
print("done")""",
    """def timing_decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@timing_decorator
def compute(x):
    return x * x

for i in range(10_000):
    compute(i)

print("done")""",
)


print("\n" + "=" * 60)
print("Enhanced Benchmark Complete!")
print("=" * 60)

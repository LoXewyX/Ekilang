<div align="center">

<img src=".github/splash.png" alt="Ekilang Splash" />

**A modern, high-performance interpreted language built on Python**

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-262%20passing-brightgreen.svg)](tests/)

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Performance](#-performance) • [Documentation](#-documentation)

</div>

---

## Overview

Ekilang is a custom programming language with modern syntax that transpiles to Python AST and executes via Python's runtime. It combines the best features from Rust, Python, and modern functional languages while maintaining competitive performance with Python.

**Key Highlights:**
- Rust-powered lexer for fast tokenization
- Zero-overhead type annotations
- Pipeline operators for functional composition
- Rust-style import system with relative paths
- Performance competitive with Python, faster on numeric operations

## Features

### Modern Syntax
- **Pipeline Operators**: Chain operations elegantly with `|>` (forward) and `<|` (backward)
- **Block Lambdas**: Multi-line anonymous functions with implicit returns
- **F-Strings**: String interpolation with `f"value: {x}"`
- **Range Operators**: Rust-style `..` (exclusive) and `..=` (inclusive)

### Advanced Language Features
- **Async/Await**: Native Python async support with `async fn` and `await`
- **Pattern Matching**: Match expressions for complex control flow
- **Destructuring**: Unpack tuples, lists, and iterables in assignments
- **Generators**: Use `yield` for lazy evaluation and iteration
- **Comprehensions**: List, set, and dict comprehensions

### Type System
- **Optional Typing**: Add type hints without runtime overhead
- **Type Casting**: Explicit conversions with `expr as type`
- **Generic Support**: Leverage Python's type system

### Import System
- **Rust-Style Syntax**: Clean `use module::item` imports
- **Relative Imports**: `.::` (current), `..::` (parent)
- **Grouped Imports**: `use module { a, b, c }`
- **Aliasing**: Import with custom names

### Advanced Parameters
- **Default Values**: `fn greet(name, greeting = "Hello")`
- **Variadic Args**: `*args` and `**kwargs` support
- **Keyword Arguments**: Named parameters in function calls

## Installation

### Prerequisites
- Python 3.9 or higher
- Rust toolchain (for building the lexer)

### Install from source

```bash
# Clone the repository
git clone https://github.com/LoXewyX/Ekilang.git
cd Ekilang

# Install with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

## Quick Start

### Hello World

```rust
fn greet(name) {
    f"Hello, {name}!"
}

print(greet("World"))
```

### Pipeline Operations

```rust
# Transform data elegantly
result = [1, 2, 3, 4, 5]
    |> map(fn(x) { x * 2 })
    |> filter(fn(x) { x > 5 })
    |> list

print(result)  # [6, 8, 10]
```

### Async/Await

```rust
use asyncio

async fn fetch_data(url) {
    # Async operation
    await asyncio.sleep(1)
    return f"Data from {url}"
}

async fn main() {
    result = await fetch_data("example.com")
    print(result)
}

asyncio.run(main())
```

### Pattern Matching

```rust
fn classify(value) {
    match value {
        0 -> "zero"
        1 -> "one"
        _ -> "many"
    }
}
```

### Relative Imports

```rust
# Import from current directory
use .::utils::helpers { greet }

# Import from parent directory
use ..::config { settings }

# Import from subdirectory
use .::math::operations { add, multiply }
```

## Performance

Ekilang performance varies by operation type. **Results may vary** between runs - sometimes Ekilang is faster, sometimes slower than Python.

### Latest Benchmark Results

**Async/Await Operations:**

| Benchmark | Ekilang | Python | Result |
|-----------|---------|--------|--------|
| Async Function Declaration (5K iterations) | 0.88ms | 0.51ms | 1.72x slower |
| Async Multiple Functions (2K iterations) | 0.57ms | 0.23ms | 2.53x slower |

**Operator Optimizations:**

| Benchmark | Ekilang | Python | Result |
|-----------|---------|--------|--------|
| Comparison Operations (100K iterations) | 28.52ms | 24.11ms | 1.18x slower |

**Language Features:**

| Benchmark | Ekilang | Python | Result |
|-----------|---------|--------|--------|
| Simple Integer Arithmetic (100K iterations) | 14.54ms | 14.76ms | **0.98x faster** ✨ |
| Boolean Logic Operations (100K iterations) | 20.72ms | 29.25ms | **0.71x faster** ✨ |
| Function Calls (10K iterations) | 2.38ms | 14.01ms | **0.17x faster** ✨ |
| String Concatenation (10K iterations) | 20.91ms | 28.31ms | **0.74x faster** ✨ |
| Binary Operations (100K iterations) | 20.81ms | 15.58ms | 1.34x slower |
| List Operations (50K iterations) | 5.17ms | 4.85ms | 1.07x slower |
| Power Operations (50K iterations) | 6.88ms | 5.64ms | 1.22x slower |
| Dictionary Operations (5K iterations) | 18.30ms | 3.09ms | 5.92x slower |
| List Comprehension (10K iterations) | 1.31ms | 0.88ms | 1.48x slower |
| Nested Loops (100x100) | 2.63ms | 1.93ms | 1.36x slower |
| Tuple Operations (5K iterations) | 2.58ms | 2.04ms | 1.26x slower |
| Type Conversions (10K iterations) | 15.22ms | 7.72ms | 1.97x slower |
| F-String Formatting (5K iterations) | 15.41ms | 14.59ms | 1.06x slower |

**Summary:**
- Total benchmarks: 16
- Ekilang faster: 4 benchmarks
- Python faster: 12 benchmarks
- Average overhead: 5.6% slower

**Notes:**
- Performance varies significantly by operation type and cache state
- Results may differ between runs and environments
- Ekilang excels at boolean logic, function calls, and string operations
- Some overhead on dictionary/tuple operations and async/await
- Cold vs hot cache states can significantly affect results

*See [tests/benchmark.py](tests/benchmark.py) for full benchmark suite*

## Documentation

### Running Code

```bash
# Execute a .eki file
python -m ekilang script.eki

# Show generated Python code
python -m ekilang script.eki --dump-py

# Save transpiled Python
python -m ekilang script.eki --dump-py > output.py
```

### Command Line Options

```bash
ekilang <file.eki>           # Run Ekilang script
ekilang <file.eki> --dump-py # Show transpiled Python code
```

### Testing

```bash
# Run all tests (262 tests)
python -m pytest

# Run benchmarks
python tests/benchmark.py
```

### Examples

Check out the [examples/](examples/) directory for comprehensive code samples:
- `advanced_parameters.eki` - Function parameters showcase
- `async.eki` - Async/await patterns
- `classes.eki` - Object-oriented programming
- `comprehensions.eki` - List/dict/set comprehensions
- `decorators.eki` - Function decorators
- `generators.eki` - Generator functions
- `imports.eki` - Import system examples
- `pipeline_operators.eki` - Pipeline operator usage
- And many more!

## Architecture

```
Ekilang Source (.eki)
         ↓
   Rust Lexer (PyO3)
         ↓
    Token Stream
         ↓
   Python Parser
         ↓
   Ekilang AST
         ↓
   Python AST
         ↓
 Python Bytecode
         ↓
    Execution
```

- **Tokenizer**: Rust-powered lexer via PyO3 for maximum performance
- **Parser**: Pure Python recursive descent parser
- **AST**: Maps to Python AST nodes for seamless execution
- **Runtime**: Leverages Python's `compile()` and `exec()` for native speed

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [PyO3](https://github.com/PyO3/pyo3) for Rust-Python interop
- Inspired by Rust, Python, and modern functional languages
- Powered by Python's robust AST and runtime

---

<div align="center">

**[⬆ back to top](#ekilang)**

Made with ❤️ by [LoXewyX](https://github.com/LoXewyX)

</div>

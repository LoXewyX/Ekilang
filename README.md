<div align="center">

<img src=".github/splash.png" alt="Ekilang Splash" />

**A modern, high-performance interpreted language built on Python**

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-262%20passing-brightgreen.svg)](tests/)

[Features](#features) • [Installation](#installation) • [Quick Start](#quick-start) • [Performance](#performance) • [Documentation](#documentation)

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
    |> map((x) => { x * 2 })
    |> filter((x) => { x > 5 })
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

Ekilang execution performance is measured on **compiled code only** (no tokenization/parsing overhead). This gives a fair comparison of runtime performance between Ekilang and Python after both are ready to execute.

### Latest Benchmark Results (January 2026)

**Execution Time Comparison** (lower is better, compiled code only)

#### Async/Await Operations

| Benchmark | Ekilang | Python | Performance |
|-----------|---------|--------|-------------|
| Async Function Declaration (5K) | 0.446ms | 0.673ms | ↑ **0.66x faster |
| Async Multiple Functions (2K) | 0.159ms | 0.205ms | ↑ **0.78x faster |

#### Operator Optimizations

| Benchmark | Ekilang | Python | Performance |
|-----------|---------|--------|-------------|
| Bitwise Operations (100K) | 42.820ms | 68.944ms | ↑ **0.62x faster |
| Comparison Operations (100K) | 26.500ms | 25.182ms | ↓ 1.05x slower |

#### Language Features & Operations

| Benchmark | Ekilang | Python | Performance |
|-----------|---------|--------|-------------|
| Simple Integer Arithmetic (100K) | 16.863ms | 20.316ms | ↑ 0.83x faster |
| Binary Operations (100K) | 23.531ms | 20.370ms | ↓ 1.16x slower |
| List Operations (50K) | 7.785ms | 8.072ms | ↑ 0.96x faster |
| Power Operations (50K) | 22.420ms | 18.730ms | ↓ 1.20x slower |
| Boolean Logic Operations (100K) | 25.282ms | 29.788ms | ↑ 0.85x faster |
| Function Calls (10K) | 2.002ms | 1.419ms | ↓ 1.41x slower |
| String Concatenation (10K) | 10.544ms | 13.312ms | ↑ 0.79x faster |
| Dictionary Operations (5K) | 1.436ms | 2.025ms | ↑ 0.71x faster |
| List Comprehension (10K) | 0.554ms | 0.602ms | ↑ 0.92x faster |
| Lambda Functions (5K) | 0.783ms | 0.691ms | ↓ 1.13x slower |
| Nested Loops (100x100) | 1.464ms | 0.992ms | ↓ 1.48x slower |
| Tuple Operations (5K) | 0.951ms | 2.271ms | ↑ 0.42x faster |
| Recursion (fibonacci up to 25) | 13.971ms | 14.083ms | ↑ 0.99x faster |
| Type Conversions (10K) | 2.975ms | 3.726ms | ↑ 0.80x faster |
| F-String Formatting (5K) | 14.496ms | 14.012ms | ↓ 1.03x slower |

#### Summary

- **Total Benchmarks:** 19
- **Ekilang Faster:** 12 operations (63%)
- **Python Faster:** 7 operations (37%)
- **Average:** 0.88x (12.4% faster)

**Notes:**
- As an interpreted language, Ekilang is expected to run slower than standard Python.
- Despite optimizations, it has higher overhead than Python.
- Results may vary between runs and across different environments.
- Performance can differ significantly depending on the type of operation and cache state.
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

Made with ❤️ by [LoXewyX](https://github.com/LoXewyX)

</div>

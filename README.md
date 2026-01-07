<div align="center">

<img src=".github/splash.png" alt="Ekilang Splash" />

**A modern, high-performance interpreted language built on Python**

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-283%20passing-brightgreen.svg)](tests/)

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

### Building from source (custom Rust lexer)

If you want to build a custom version or modify the Rust lexer:

```bash
# Install maturin (Rust-Python build tool)
pip install maturin

# Build and install directly (recommended for development)
# Note: Currently using Python 3.14, which requires forward compatibility flag
$env:PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1  # Windows PowerShell
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1  # Linux/Mac

maturin develop --release

# Alternative: Build wheel for distribution (creates wheel in target/wheels/)
maturin build --release

# The compiled module will be placed at:
# ekilang/_rust_lexer.cp314-win_amd64.pyd (Windows)
# ekilang/_rust_lexer.cpython-314-x86_64-linux-gnu.so (Linux)
```

**Note**: 
- Use `maturin develop` to install directly into your package (for development)
- Use `maturin build` to create a wheel file (for distribution)
- PyO3 0.21.2 officially supports up to Python 3.12. The forward compatibility flag allows building with Python 3.14, but you may want to upgrade PyO3 in `Cargo.toml` for better support.

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
| Async Function Declaration (5K) | 0.315ms | 0.354ms | ↑ 1.12x faster |
| Async Multiple Functions (2K) | 0.109ms | 0.159ms | ↑ 1.46x faster |
| Async With + Async For (500 items) | 0.809ms | 1.354ms | ↑ 1.67x faster |

#### Operator Optimizations

| Benchmark | Ekilang | Python | Performance |
|-----------|---------|--------|-------------|
| Bitwise Operations (100K) | 29.293ms | 24.671ms | ↓ 1.19x slower |
| Comparison Operations (100K) | 18.343ms | 20.326ms | ↑ 1.11x faster |

#### Language Features & Operations

| Benchmark | Ekilang | Python | Performance |
|-----------|---------|--------|-------------|
| Simple Integer Arithmetic (100K) | 13.445ms | 12.847ms | ↓ 1.05x slower |
| Binary Operations (100K) | 18.544ms | 15.291ms | ↓ 1.21x slower |
| List Operations (50K) | 5.690ms | 5.001ms | ↓ 1.14x slower |
| With Statement (1000 iterations) | 0.990ms | 1.019ms | ↑ 1.03x faster |
| Multiple Context Managers (500) | 0.967ms | 1.050ms | ↑ 1.09x faster |
| Nested Context Managers (500) | 0.971ms | 1.303ms | ↑ 1.34x faster |
| Power Operations (50K) | 4.445ms | 4.109ms | ↓ 1.08x slower |
| Boolean Logic Operations (100K) | 15.700ms | 14.979ms | ↓ 1.05x slower |
| Function Calls (10K) | 1.909ms | 1.918ms | ↑ 1.00x faster |
| String Concatenation (10K) | 16.906ms | 25.957ms | ↑ 1.54x faster |
| Dictionary Operations (5K) | 2.925ms | 2.462ms | ↓ 1.19x slower |
| List Comprehension (10K) | 0.678ms | 1.387ms | ↑ 2.04x faster |
| Lambda Functions (5K) | 1.102ms | 3.906ms | ↑ 3.55x faster |
| Nested Loops (100x100) | 1.565ms | 4.803ms | ↑ 3.07x faster |
| Tuple Operations (5K) | 2.189ms | 11.052ms | ↑ 5.05x faster |
| Recursion (fibonacci up to 25) | 25.522ms | 9.820ms | ↓ 2.60x slower |
| Type Conversions (10K) | 2.467ms | 3.104ms | ↑ 1.26x faster |
| F-String Formatting (5K) | 10.514ms | 10.047ms | ↓ 1.05x slower |

#### Summary

- **Total Benchmarks:** 23
- **Ekilang Faster:** 14 operations (61%)
- **Python Faster:** 9 operations (39%)
- **Average:** 0.99x (0.9% faster)

**Notes:**
- Results may vary between runs and across different environments
- Performance can differ significantly depending on the type of operation and cache state
- Benchmark shows Ekilang matches Python performance overall

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
# Run all tests (283 tests)
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

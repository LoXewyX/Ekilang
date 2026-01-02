Ekilang - An optimized Python interpreted language

Overview
- Custom syntax that transpiles to Python AST and executes via Python for performance similar to native Python.
- Modern features: block-style lambdas, f-strings, Rust-style imports, async/await, optional typing, pipeline operators, and more.

Syntax Preview
```
x = 2
fn add(a, b) {
  return a + b
}
if x > 1 {
  print(add(x, 3))
}

# Pipeline operators for elegant data transformation
result = [1, 2, 3, 4, 5] |> sum |> double
```

Features
- **Pipeline Operators**: Forward `|>` and backward `<|` pipes for functional composition
- **String/Sequence Slicing**: Full Python-style slicing with `[start:stop:step]` syntax for substrings and subsequences
- **Destructuring Assignments**: Unpack tuples, lists, and iterables with `x, y = getValue()`
- **Advanced Parameters**: Default values, `*args`, `**kwargs` with proper Python ordering
- **Keyword Arguments**: Call functions with `func(a=1, b=2)` syntax
- **Rust-Style Imports**: `use module::item` with grouped imports and aliasing
  - **Absolute imports**: `use examples::math_utils { square, cube }`
  - **Relative imports**: `.::` (current dir), `..::` (parent dir)
  - **Examples**: 
    - `use .::utils::helpers { greet }` imports from same directory
    - `use ..::utils { add }` imports from parent directory
    - `use .::subdir::module { func }` imports from subdirectory
- **Async/Await**: Native Python async with `async fn` and `await` keyword
- **Optional Typing**: Zero-overhead type annotations for parameters and returns
- **Collections**: Tuples, sets, dicts with proper syntax
- **Type Casting**: `expr as type` for explicit conversions
- **Block Lambdas**: Multi-line lambdas with implicit returns
- **F-Strings**: String interpolation with `f"value: {x}"`
- **Range Operators**: Rust-style `..` and `..=` for ranges
- **List Comprehensions**: `[x * 2 for x in nums if x > 0]`
- **Modern Operators**: Ternary, compound assignments, bitwise ops
- **Generators**: Use `yield` inside functions; any function containing `yield` becomes a generator. Iterate with `for` to consume values. See `examples/generators.eki`.
- `ekilang --dump-py path/to/file.eki` - show generated Python source
- `ekilang .\examples\sample.eki --dump-py | Out-File test.py -Encoding utf8` - transforms Ekilang into Python (PowerShell)

Design Notes
- Tokenizer and parser produce a Ekilang AST that maps to Python `ast` nodes.
- Execution uses Python `compile` and `exec` for performance.
- Builtins namespace includes `print`, `len`, and can be extended.

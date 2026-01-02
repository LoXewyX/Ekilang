mod token;
mod lexer;
mod error;
mod builtins;

use lexer::Lexer;
use pyo3::{Bound, PyErr, PyResult, Python, pyfunction, pymodule, types::PyModule, wrap_pyfunction};
use token::Token;

/// Tokenize Ekilang source code using Rust lexer
/// Returns a list of Token objects compatible with Python
#[pyfunction]
fn tokenize(source: &str) -> PyResult<Vec<Token>> {
    let mut lexer = Lexer::new(source);
    
    match lexer.tokenize() {
        Ok(tokens) => Ok(tokens),
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PySyntaxError, _>(
            e.to_string()
        )),
    }
}

#[pymodule]
#[pyo3(name = "_rust_lexer")]
fn rust_lexer(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register Token class
    m.add_class::<Token>()?;
    
    // Register lexer functions
    m.add_function(wrap_pyfunction!(tokenize, m)?)?;
    
    // Register optimization functions from builtins module
    m.add_function(wrap_pyfunction!(builtins::apply_binop, m)?)?;
    m.add_function(wrap_pyfunction!(builtins::apply_compare, m)?)?;
    
    Ok(())
}
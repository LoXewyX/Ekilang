mod token;
mod lexer;
mod error;
mod builtins;
mod parser_helpers;

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
    
    // Register parser helper functions
    m.add_function(wrap_pyfunction!(parser_helpers::get_operator_precedence, m)?)?;
    m.add_function(wrap_pyfunction!(parser_helpers::is_aug_assign_op, m)?)?;
    m.add_function(wrap_pyfunction!(parser_helpers::is_comparison_op, m)?)?;
    m.add_function(wrap_pyfunction!(parser_helpers::is_binary_op, m)?)?;
    m.add_function(wrap_pyfunction!(parser_helpers::is_unary_op, m)?)?;
    m.add_function(wrap_pyfunction!(parser_helpers::is_statement_keyword, m)?)?;
    m.add_function(wrap_pyfunction!(parser_helpers::is_right_associative, m)?)?;
    m.add_function(wrap_pyfunction!(parser_helpers::is_valid_token_type, m)?)?;
    m.add_function(wrap_pyfunction!(parser_helpers::canonicalize_operator, m)?)?;
    m.add_function(wrap_pyfunction!(parser_helpers::validate_interpolation_braces, m)?)?;
    m.add_function(wrap_pyfunction!(parser_helpers::is_valid_id_start, m)?)?;
    m.add_function(wrap_pyfunction!(parser_helpers::is_valid_id_continue, m)?)?;
    m.add_function(wrap_pyfunction!(parser_helpers::validate_operators, m)?)?;
    m.add_function(wrap_pyfunction!(parser_helpers::classify_operator, m)?)?;
    m.add_function(wrap_pyfunction!(parser_helpers::classify_keyword, m)?)?;
    
    Ok(())
}
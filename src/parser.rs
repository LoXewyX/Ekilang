use pyo3::exceptions::PyValueError;
use pyo3::{pyfunction, PyErr, PyResult};

/// Fast operator precedence check - returns precedence level
/// Higher numbers = higher precedence
#[pyfunction]
pub fn get_operator_precedence(op: &str) -> PyResult<u8> {
    let precedence = match op {
        // Assignment (lowest)
        "=" | "+=" | "-=" | "*=" | "**=" | "/=" | "//=" | "%=" | "&=" | "|=" | "^=" | "<<="
        | ">>=" => 0,

        // Named expression (walrus)
        ":=" => 1,

        // Pipe operators
        "|>" | "<|" => 2,

        // Ternary
        "if" => 3,

        // Logical OR
        "or" => 4,

        // Logical AND
        "and" => 5,

        // Logical NOT
        "not" => 6,

        // Comparisons
        "<" | ">" | "<=" | ">=" | "==" | "!=" | "in" | "is" => 7,

        // Bitwise OR
        "|" => 8,

        // Bitwise XOR
        "^" => 9,

        // Bitwise AND
        "&" => 10,

        // Bit shifts
        "<<" | ">>" => 11,

        // Range
        ".." | "..=" => 12,

        // Addition/Subtraction
        "+" | "-" => 13,

        // Multiplication/Division/Modulo
        "*" | "/" | "//" | "%" => 14,

        // Power (highest precedence)
        "**" => 15,

        _ => {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "Unknown operator: {}",
                op
            )))
        }
    };

    Ok(precedence)
}

/// Fast check if operator is augmented assignment
#[pyfunction]
pub fn is_aug_assign_op(op: &str) -> bool {
    matches!(
        op,
        "+=" | "-=" | "*=" | "**=" | "/=" | "//=" | "%=" | "&=" | "|=" | "^=" | "<<=" | ">>="
    )
}

/// Fast check if operator is comparison
#[pyfunction]
pub fn is_comparison_op(op: &str) -> bool {
    matches!(op, "<" | ">" | "<=" | ">=" | "==" | "!=" | "in" | "is")
}

/// Fast check if operator is binary arithmetic
#[pyfunction]
pub fn is_binary_op(op: &str) -> bool {
    matches!(
        op,
        "+" | "-" | "*" | "/" | "//" | "%" | "**" | "&" | "|" | "^" | "<<" | ">>"
    )
}

/// Fast check if operator is unary
#[pyfunction]
pub fn is_unary_op(op: &str) -> bool {
    matches!(op, "-" | "~" | "not")
}

/// Fast check if token is a keyword statement starter
#[pyfunction]
pub fn is_statement_keyword(kw: &str) -> bool {
    matches!(
        kw,
        "class"
            | "use"
            | "for"
            | "with"
            | "break"
            | "continue"
            | "yield"
            | "return"
            | "async"
            | "fn"
            | "if"
            | "match"
            | "while"
            | "try"
            | "let"
    )
}

/// Fast check if operator is right-associative
#[pyfunction]
pub fn is_right_associative(op: &str) -> bool {
    matches!(op, "**" | "<|")
}

/// Fast token type validation - checks if type string is valid
#[pyfunction]
pub fn is_valid_token_type(type_str: &str) -> bool {
    matches!(
        type_str,
        "INT"
            | "FLOAT"
            | "STR"
            | "FSTR"
            | "TSTR"
            | "BSTR"
            | "ID"
            | "KW"
            | "OP"
            | "("
            | ")"
            | "{"
            | "}"
            | "["
            | "]"
            | "NL"
            | ":"
            | ","
            | "@"
            | "."
            | "EOF"
    )
}

/// Optimize operator string by returning a canonical form
/// This helps with faster matching in hot loops
#[pyfunction]
pub fn canonicalize_operator(op: &str) -> String {
    // Most operators are already canonical, but we can intern common ones
    match op {
        "+" | "-" | "*" | "/" | "//" | "%" | "**" => op.to_string(),
        "==" | "!=" | "<" | ">" | "<=" | ">=" => op.to_string(),
        "&" | "|" | "^" | "~" | "<<" | ">>" => op.to_string(),
        "+=" | "-=" | "*=" | "/=" | "//=" | "%=" | "**=" => op.to_string(),
        "&=" | "|=" | "^=" | "<<=" | ">>=" => op.to_string(),
        "|>" | "<|" => op.to_string(),
        ".." | "..=" => op.to_string(),
        ":=" => op.to_string(),
        _ => op.to_string(),
    }
}

/// Fast string interpolation validation - checks if braces are balanced
#[pyfunction]
pub fn validate_interpolation_braces(content: &str) -> PyResult<bool> {
    let mut depth = 0;
    let mut in_escape = false;

    for ch in content.chars() {
        if in_escape {
            in_escape = false;
            continue;
        }

        match ch {
            '\\' => in_escape = true,
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth < 0 {
                    return Ok(false);
                }
            }
            _ => {}
        }
    }

    Ok(depth == 0)
}

/// Fast check for valid identifier start character
#[pyfunction]
pub fn is_valid_id_start(ch: char) -> bool {
    ch.is_ascii_alphabetic() || ch == '_'
}

/// Fast check for valid identifier continuation character
#[pyfunction]
pub fn is_valid_id_continue(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '_'
}

/// Batch operator validation - validates multiple operators at once
/// Returns Vec of bools indicating which operators are valid
#[pyfunction]
pub fn validate_operators(operators: Vec<String>) -> Vec<bool> {
    operators
        .iter()
        .map(|op| {
            matches!(
                op.as_str(),
                "+" | "-"
                    | "*"
                    | "/"
                    | "//"
                    | "%"
                    | "**"
                    | "=="
                    | "!="
                    | "<"
                    | ">"
                    | "<="
                    | ">="
                    | "&"
                    | "|"
                    | "^"
                    | "~"
                    | "<<"
                    | ">>"
                    | "+="
                    | "-="
                    | "*="
                    | "/="
                    | "//="
                    | "%="
                    | "**="
                    | "&="
                    | "|="
                    | "^="
                    | "<<="
                    | ">>="
                    | "|>"
                    | "<|"
                    | ".."
                    | "..="
                    | ":="
                    | "="
                    | ":"
            )
        })
        .collect()
}

/// Optimize common pattern matching for parser lookahead
/// Returns (is_assignment, is_aug_assign, is_comparison, is_binary_op)
#[pyfunction]
pub fn classify_operator(op: &str) -> (bool, bool, bool, bool) {
    (
        op == "=",
        is_aug_assign_op(op),
        is_comparison_op(op),
        is_binary_op(op),
    )
}

/// Fast keyword classification for statement parsing
/// Returns category: 0=not_keyword, 1=definition, 2=control_flow, 3=simple_stmt
#[pyfunction]
pub fn classify_keyword(kw: &str) -> u8 {
    match kw {
        // Definition keywords
        "class" | "fn" | "async" | "let" => 1,

        // Control flow
        "if" | "elif" | "else" | "while" | "for" | "match" | "try" | "except" | "finally" => 2,

        // Simple statements
        "break" | "continue" | "return" | "yield" | "use" => 3,

        // Expression keywords
        "and" | "or" | "not" | "in" | "is" | "as" | "await" => 4,

        _ => 0,
    }
}

use pyo3::pyfunction;
use pyo3::{PyErr, PyResult};

/// Fast binary operator for numeric operations
#[pyfunction]
pub fn apply_binop(left: f64, op: &str, right: f64) -> PyResult<f64> {
    let result = match op {
        "+" => left + right,
        "-" => left - right,
        "*" => left * right,
        "/" => left / right,
        "//" => (left / right).floor(),
        "%" => left % right,
        "**" => left.powf(right),
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unsupported operator: {}",
                op
            )))
        }
    };
    Ok(result)
}

/// Fast comparison operations
#[pyfunction]
pub fn apply_compare(left: f64, op: &str, right: f64) -> PyResult<bool> {
    let result = match op {
        "<" => left < right,
        ">" => left > right,
        "<=" => left <= right,
        ">=" => left >= right,
        "==" => (left - right).abs() < f64::EPSILON,
        "!=" => (left - right).abs() >= f64::EPSILON,
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unsupported comparison: {}",
                op
            )))
        }
    };
    Ok(result)
}

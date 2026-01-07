use std::fmt;

#[derive(Debug, Clone)]
pub enum LexError {
    UnterminatedString { line: u32, col: u32, kind: String },
    InvalidCharacter { line: u32, col: u32, ch: char },
}

impl fmt::Display for LexError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            LexError::UnterminatedString { line, col, kind } => {
                write!(f, "Unterminated {} at {}:{}", kind, line, col)
            }
            LexError::InvalidCharacter { line, col, ch } => {
                write!(f, "Unexpected character '{}' at {}:{}", ch, line, col)
            }
        }
    }
}

impl std::error::Error for LexError {}

impl LexError {
    pub fn to_string(&self) -> String {
        format!("{}", self)
    }
}

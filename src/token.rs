use pyo3::{pyclass, pymethods};

/// Token type enumeration
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenType {
    // Literals
    Int,
    Float,
    Str,
    FStr,
    TStr,
    
    // Identifiers and keywords
    Id,
    Kw,
    
    // Operators
    Op,
    
    // Delimiters
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    
    // Structural
    Newline,
    Semicolon,
    Colon,
    Comma,
    At,
    Dot,
    
    // Special
    Eof,
}

impl TokenType {
    pub fn as_str(&self) -> &'static str {
        match self {
            TokenType::Int => "INT",
            TokenType::Float => "FLOAT",
            TokenType::Str => "STR",
            TokenType::FStr => "FSTR",
            TokenType::TStr => "TSTR",
            TokenType::Id => "ID",
            TokenType::Kw => "KW",
            TokenType::Op => "OP",
            TokenType::LParen => "(",
            TokenType::RParen => ")",
            TokenType::LBrace => "{",
            TokenType::RBrace => "}",
            TokenType::LBracket => "[",
            TokenType::RBracket => "]",
            TokenType::Newline => "NL",
            TokenType::Semicolon => "NL",
            TokenType::Colon => ":",
            TokenType::Comma => ",",
            TokenType::At => "@",
            TokenType::Dot => ".",
            TokenType::Eof => "EOF",
        }
    }
}

/// Python-compatible Token class
#[pyclass]
#[derive(Debug, Clone)]
pub struct Token {
    #[pyo3(get)]
    pub r#type: String,
    
    #[pyo3(get)]
    pub value: String,
    
    #[pyo3(get)]
    pub line: u32,
    
    #[pyo3(get)]
    pub col: u32,
}

impl Token {
    pub fn new(token_type: TokenType, value: String, line: u32, col: u32) -> Self {
        Token {
            r#type: token_type.as_str().to_string(),
            value,
            line,
            col,
        }
    }

    pub fn new_with_type(type_str: &str, value: String, line: u32, col: u32) -> Self {
        Token {
            r#type: type_str.to_string(),
            value,
            line,
            col,
        }
    }

    pub fn eof(line: u32, col: u32) -> Self {
        Token {
            r#type: "EOF".to_string(),
            value: String::new(),
            line,
            col,
        }
    }

    pub fn newline(line: u32, col: u32) -> Self {
        Token {
            r#type: "NL".to_string(),
            value: "\n".to_string(),
            line,
            col,
        }
    }
}

#[pymethods]
impl Token {
    #[new]
    fn py_new(r#type: String, value: String, line: u32, col: u32) -> Self {
        Token {
            r#type,
            value,
            line,
            col,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Token(type='{}', value='{}', line={}, col={})",
            self.r#type, self.value, self.line, self.col
        )
    }

    fn __eq__(&self, other: &Token) -> bool {
        self.r#type == other.r#type
            && self.value == other.value
            && self.line == other.line
            && self.col == other.col
    }
}

// Keywords set
pub static KEYWORDS: &[&str] = &[
    "fn", "async", "await", "use", "as", "if", "elif", "else", "match", "while",
    "return", "yield", "for", "in", "true", "false", "none", "and", "or", "not",
    "is", "break", "continue", "class",
];

pub fn is_keyword(s: &str) -> bool {
    KEYWORDS.contains(&s)
}

use crate::token::{Token, TokenType, is_keyword};
use crate::error::LexError;

pub struct Lexer {
    source: Vec<char>,
    pos: usize,
    line: u32,
    col: u32,
}

impl Lexer {
    pub fn new(source: &str) -> Self {
        Lexer {
            source: source.chars().collect(),
            pos: 0,
            line: 1,
            col: 1,
        }
    }

    fn current(&self) -> Option<char> {
        if self.pos < self.source.len() {
            Some(self.source[self.pos])
        } else {
            None
        }
    }

    fn peek(&self, offset: usize) -> Option<char> {
        let pos = self.pos + offset;
        if pos < self.source.len() {
            Some(self.source[pos])
        } else {
            None
        }
    }

    fn advance(&mut self) -> Option<char> {
        if let Some(ch) = self.current() {
            self.pos += 1;
            if ch == '\n' {
                self.line += 1;
                self.col = 1;
            } else {
                self.col += 1;
            }
            Some(ch)
        } else {
            None
        }
    }

    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.current() {
            if ch.is_whitespace() && ch != '\n' && ch != '\r' {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn skip_comment(&mut self) {
        // Skip '#' and everything until newline
        while let Some(ch) = self.current() {
            if ch == '\n' || ch == '\r' {
                break;
            }
            self.advance();
        }
    }

    fn read_identifier(&mut self) -> String {
        let mut result = String::new();
        while let Some(ch) = self.current() {
            if ch.is_alphanumeric() || ch == '_' {
                result.push(ch);
                self.advance();
            } else {
                break;
            }
        }
        result
    }

    fn read_number(&mut self) -> Result<Token, LexError> {
        let start_line = self.line;
        let start_col = self.col;
        let mut num_str = String::new();
        let mut is_float = false;

        while let Some(ch) = self.current() {
            if ch.is_numeric() {
                num_str.push(ch);
                self.advance();
            } else if ch == '.' && !is_float && self.peek(1).map_or(false, |c| c.is_numeric()) {
                is_float = true;
                num_str.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        let token_type = if is_float {
            TokenType::Float
        } else {
            TokenType::Int
        };

        Ok(Token::new(token_type, num_str, start_line, start_col))
    }

    fn read_string(&mut self, quote: char, is_fstring: bool, is_tstring: bool) -> Result<Token, LexError> {
        let start_line = self.line;
        let start_col = self.col;

        // Check for triple quotes
        let triple = self.peek(1) == Some(quote) && self.peek(2) == Some(quote);
        
        // Skip opening quote(s)
        self.advance();
        if triple {
            self.advance();
            self.advance();
        }

        let end_sequence = if triple {
            vec![quote, quote, quote]
        } else {
            vec![quote]
        };

        let mut result = String::new();
        let mut found_end = false;

        while let Some(ch) = self.current() {
            // Check for end sequence
            if self.matches_sequence(&end_sequence) {
                // Skip end sequence
                for _ in 0..end_sequence.len() {
                    self.advance();
                }
                found_end = true;
                break;
            }

            if ch == '\\' && self.peek(1).is_some() {
                // Handle escape sequences
                self.advance();
                if let Some(escaped) = self.current() {
                    match escaped {
                        'n' => result.push('\n'),
                        't' => result.push('\t'),
                        'r' => result.push('\r'),
                        '\\' => result.push('\\'),
                        '\'' => result.push('\''),
                        '"' => result.push('"'),
                        _ => {
                            result.push('\\');
                            result.push(escaped);
                        }
                    }
                    self.advance();
                }
            } else {
                result.push(ch);
                self.advance();
            }
        }

        if !found_end {
            let kind = if is_fstring {
                "f-string".to_string()
            } else if is_tstring {
                "t-string".to_string()
            } else {
                "string".to_string()
            };
            return Err(LexError::UnterminatedString {
                line: start_line,
                col: start_col,
                kind,
            });
        }

        let token_type = if is_fstring {
            TokenType::FStr
        } else if is_tstring {
            TokenType::TStr
        } else {
            TokenType::Str
        };

        Ok(Token::new(token_type, result, start_line, start_col))
    }

    fn matches_sequence(&self, seq: &[char]) -> bool {
        for (i, &ch) in seq.iter().enumerate() {
            if self.peek(i) != Some(ch) {
                return false;
            }
        }
        true
    }

    fn read_operator(&mut self) -> Token {
        let start_line = self.line;
        let start_col = self.col;
        let mut op = String::new();

        // Try 3-character operators first
        if let Some(three) = self.peek_string(3) {
            if matches!(three.as_str(), "..=" | "<<=" | ">>=" | "**=" | "//=") {
                op = three;
                for _ in 0..3 {
                    self.advance();
                }
                return Token::new_with_type("OP", op, start_line, start_col);
            }
        }

        // Try 2-character operators
        if let Some(two) = self.peek_string(2) {
            if matches!(
                two.as_str(),
                "=>" | "==" | "!=" | ">=" | "<=" | "+=" | "-=" | "*=" | "/=" | "%=" | "//" | ".." |
                "<<" | ">>" | "&=" | "|=" | "^=" | "::" | "->" | "|>" | "<|" | "**"
            ) {
                op = two;
                self.advance();
                self.advance();
                return Token::new_with_type("OP", op, start_line, start_col);
            }
        }

        // Single character operators
        if let Some(ch) = self.current() {
            if "+-*/%=<>:|&^.@".contains(ch) {
                op.push(ch);
                self.advance();
                return Token::new_with_type("OP", op, start_line, start_col);
            }
        }

        // Shouldn't reach here, but return something safe
        Token::new_with_type("OP", "?".to_string(), start_line, start_col)
    }

    fn peek_string(&self, len: usize) -> Option<String> {
        if self.pos + len <= self.source.len() {
            Some(self.source[self.pos..self.pos + len].iter().collect())
        } else {
            None
        }
    }

    pub fn tokenize(&mut self) -> Result<Vec<Token>, LexError> {
        let mut tokens = Vec::new();

        loop {
            self.skip_whitespace();

            match self.current() {
                None => {
                    tokens.push(Token::eof(self.line, self.col));
                    break;
                }
                Some('\n') => {
                    tokens.push(Token::newline(self.line, self.col));
                    self.advance();
                }
                Some('\r') => {
                    self.advance();
                    if self.current() == Some('\n') {
                        self.advance();
                    }
                    tokens.push(Token::newline(self.line, self.col));
                }
                Some('#') => {
                    self.skip_comment();
                }
                Some(ch) if ch.is_alphabetic() || ch == '_' => {
                    let start_line = self.line;
                    let start_col = self.col;
                    let ident = self.read_identifier();

                    // Check for f/t string prefix
                    if (ident == "f" || ident == "t") && matches!(self.current(), Some('"') | Some('\'')) {
                        let is_fstring = ident == "f";
                        let is_tstring = ident == "t";
                        let quote = self.current().unwrap();
                        let token = self.read_string(quote, is_fstring, is_tstring)?;
                        tokens.push(token);
                    } else {
                        // Regular identifier or keyword
                        let token_type = if is_keyword(&ident) {
                            "KW"
                        } else {
                            "ID"
                        };
                        tokens.push(Token::new_with_type(token_type, ident, start_line, start_col));
                    }
                }
                Some(ch) if ch.is_numeric() => {
                    tokens.push(self.read_number()?);
                }
                Some('"') | Some('\'') => {
                    let quote = self.current().unwrap();
                    tokens.push(self.read_string(quote, false, false)?);
                }
                Some('(') => {
                    tokens.push(Token::new(TokenType::LParen, "(".to_string(), self.line, self.col));
                    self.advance();
                }
                Some(')') => {
                    tokens.push(Token::new(TokenType::RParen, ")".to_string(), self.line, self.col));
                    self.advance();
                }
                Some('{') => {
                    tokens.push(Token::new(TokenType::LBrace, "{".to_string(), self.line, self.col));
                    self.advance();
                }
                Some('}') => {
                    tokens.push(Token::new(TokenType::RBrace, "}".to_string(), self.line, self.col));
                    self.advance();
                }
                Some('[') => {
                    tokens.push(Token::new(TokenType::LBracket, "[".to_string(), self.line, self.col));
                    self.advance();
                }
                Some(']') => {
                    tokens.push(Token::new(TokenType::RBracket, "]".to_string(), self.line, self.col));
                    self.advance();
                }
                Some(':') => {
                    // Check for :: operator
                    if self.peek(1) == Some(':') {
                        tokens.push(Token::new_with_type("OP", "::".to_string(), self.line, self.col));
                        self.advance();
                        self.advance();
                    } else {
                        tokens.push(Token::new(TokenType::Colon, ":".to_string(), self.line, self.col));
                        self.advance();
                    }
                }
                Some(',') => {
                    tokens.push(Token::new(TokenType::Comma, ",".to_string(), self.line, self.col));
                    self.advance();
                }
                Some('@') => {
                    tokens.push(Token::new(TokenType::At, "@".to_string(), self.line, self.col));
                    self.advance();
                }
                Some(';') => {
                    tokens.push(Token::new_with_type("NL", ";".to_string(), self.line, self.col));
                    self.advance();
                }
                Some('.') => {
                    // Check for .. or ..=
                    if self.peek(1) == Some('.') {
                        if self.peek(2) == Some('=') {
                            tokens.push(Token::new_with_type("OP", "..=".to_string(), self.line, self.col));
                            self.advance();
                            self.advance();
                            self.advance();
                        } else {
                            tokens.push(Token::new_with_type("OP", "..".to_string(), self.line, self.col));
                            self.advance();
                            self.advance();
                        }
                    } else {
                        tokens.push(Token::new(TokenType::Dot, ".".to_string(), self.line, self.col));
                        self.advance();
                    }
                }
                Some(ch) if "+-*/%=<>:|&^.@!".contains(ch) => {
                    tokens.push(self.read_operator());
                }
                Some(ch) => {
                    return Err(LexError::InvalidCharacter {
                        line: self.line,
                        col: self.col,
                        ch,
                    });
                }
            }
        }

        Ok(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_identifiers() {
        let mut lexer = Lexer::new("x y _var");
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(tokens[0].r#type, "ID");
        assert_eq!(tokens[0].value, "x");
        assert_eq!(tokens[2].r#type, "ID");
        assert_eq!(tokens[2].value, "y");
    }

    #[test]
    fn test_keywords() {
        let mut lexer = Lexer::new("fn if else");
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(tokens[0].r#type, "KW");
        assert_eq!(tokens[0].value, "fn");
    }

    #[test]
    fn test_numbers() {
        let mut lexer = Lexer::new("42 3.14");
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(tokens[0].r#type, "INT");
        assert_eq!(tokens[0].value, "42");
        assert_eq!(tokens[2].r#type, "FLOAT");
        assert_eq!(tokens[2].value, "3.14");
    }

    #[test]
    fn test_strings() {
        let mut lexer = Lexer::new(r#""hello" 'world'"#);
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(tokens[0].r#type, "STR");
        assert_eq!(tokens[0].value, "hello");
        assert_eq!(tokens[2].r#type, "STR");
        assert_eq!(tokens[2].value, "world");
    }

    #[test]
    fn test_fstrings() {
        let mut lexer = Lexer::new(r#"f"x = {x}""#);
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(tokens[0].r#type, "FSTR");
        assert_eq!(tokens[0].value, "x = {x}");
    }

    #[test]
    fn test_operators() {
        let mut lexer = Lexer::new("+ - * / == != |> <|");
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(tokens[0].r#type, "OP");
        assert_eq!(tokens[0].value, "+");
        assert_eq!(tokens[6].r#type, "OP");
        assert_eq!(tokens[6].value, "|>");
    }

    #[test]
    fn test_comments() {
        let mut lexer = Lexer::new("x = 1 # comment\ny = 2");
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(tokens[0].value, "x");
        assert_eq!(tokens[4].r#type, "NL");
        assert_eq!(tokens[5].value, "y");
    }

    #[test]
    fn test_line_col_tracking() {
        let mut lexer = Lexer::new("x\ny");
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(tokens[0].line, 1);
        assert_eq!(tokens[2].line, 2);
    }
}

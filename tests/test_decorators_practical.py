"""Test practical decorator patterns"""

from ekilang.lexer import Lexer
from ekilang.parser import Parser
from ekilang.runtime import execute


def test_call_counter_decorator():
    """Test decorator that counts function calls"""
    src = '''
fn count_calls(func) {
    state = [0]
    fn wrapper(x) {
        state[0] = state[0] + 1
        return state[0]
    }
    return wrapper
}

@count_calls
fn noop(x) {
    return x
}

first = noop(1)
second = noop(2)
third = noop(3)
'''
    
    tokens = Lexer(src).tokenize()
    mod = Parser(tokens).parse()
    ns = execute(mod)
    
    assert ns["first"] == 1
    assert ns["second"] == 2
    assert ns["third"] == 3


def test_type_validation_decorator():
    """Test decorator that validates input types"""
    src = '''
fn validate_positive(func) {
    fn wrapper(x) {
        if x <= 0 {
            return none
        }
        return func(x)
    }
    return wrapper
}

@validate_positive
fn square(x) {
    return x * x
}

valid = square(5)
invalid = square(-3)
'''
    
    tokens = Lexer(src).tokenize()
    mod = Parser(tokens).parse()
    ns = execute(mod)
    
    assert ns["valid"] == 25
    assert ns["invalid"] is None


def test_safe_default_decorator():
    """Test decorator that provides default values"""
    src = '''
fn safe_default(default_val) {
    fn decorator(func) {
        fn wrapper(x) {
            if x is none {
                return default_val
            }
            result = func(x)
            if result is none {
                return default_val
            }
            return result
        }
        return wrapper
    }
    return decorator
}

@safe_default(0)
fn safe_divide(x) {
    if x == 0 {
        return none
    }
    return 100 / x
}

good = safe_divide(10)
bad = safe_divide(0)
'''
    
    tokens = Lexer(src).tokenize()
    mod = Parser(tokens).parse()
    ns = execute(mod)
    
    assert ns["good"] == 10.0
    assert ns["bad"] == 0


def test_timing_decorator():
    """Test decorator that logs execution"""
    src = '''
fn timing(func) {
    fn wrapper(x) {
        return func(x)
    }
    return wrapper
}

@timing
fn operation(n) {
    sum = 0
    i = 0
    while i < n {
        sum = sum + i
        i = i + 1
    }
    return sum
}

result = operation(10)
'''
    
    tokens = Lexer(src).tokenize()
    mod = Parser(tokens).parse()
    ns = execute(mod)
    
    # 0+1+2+...+9 = 45
    assert ns["result"] == 45


def test_retry_decorator():
    """Test decorator that retries on failure"""
    src = '''
fn retry_once(func) {
    fn wrapper(x) {
        result = func(x)
        if result is none {
            result = func(x)
        }
        return result
    }
    return wrapper
}

@retry_once
fn maybe_fails(x) {
    if x < 5 {
        return none
    }
    return x * 2
}

single_attempt = maybe_fails(10)
double_attempt = maybe_fails(3)
'''
    
    tokens = Lexer(src).tokenize()
    mod = Parser(tokens).parse()
    ns = execute(mod)
    
    # First call succeeds (no retry needed): 10 * 2 = 20
    assert ns["single_attempt"] == 20
    # Second call fails first, then succeeds (retry returns None second time too)
    assert ns["double_attempt"] is None


def test_nested_decorator_factories():
    """Test decorator factories with nested levels"""
    src = '''
fn apply_times(n) {
    fn decorator(func) {
        fn wrapper(x) {
            result = x
            i = 0
            while i < n {
                result = func(result)
                i = i + 1
            }
            return result
        }
        return wrapper
    }
    return decorator
}

@apply_times(3)
fn increment(x) {
    return x + 1
}

result = increment(10)
'''
    
    tokens = Lexer(src).tokenize()
    mod = Parser(tokens).parse()
    ns = execute(mod)
    
    # Apply increment 3 times: 10 + 1 + 1 + 1 = 13
    assert ns["result"] == 13


def test_decorator_with_state():
    """Test decorator that maintains internal state"""
    src = '''
fn stateful_decorator(func) {
    state = [0]
    fn wrapper(x) {
        state[0] = state[0] + 1
        result = func(x)
        return state[0]
    }
    return wrapper
}

@stateful_decorator
fn add_ten(x) {
    return x + 10
}

c1 = add_ten(1)
c2 = add_ten(2)
c3 = add_ten(3)
count = add_ten(0)
'''
    
    tokens = Lexer(src).tokenize()
    mod = Parser(tokens).parse()
    ns = execute(mod)
    
    assert ns["c1"] == 1
    assert ns["c2"] == 2
    assert ns["c3"] == 3
    assert ns["count"] == 4


def test_chaining_decorator_transformations():
    """Test decorators that chain transformations"""
    src = '''
fn add_suffix(text) {
    fn decorator(func) {
        fn wrapper(x) {
            result = func(x)
            return f"{result}{text}"
        }
        return wrapper
    }
    return decorator
}

fn add_prefix(text) {
    fn decorator(func) {
        fn wrapper(x) {
            result = func(x)
            return f"{text}{result}"
        }
        return wrapper
    }
    return decorator
}

@add_suffix("!")
@add_prefix(">> ")
fn message(text) {
    return text
}

result = message("hello")
'''
    
    tokens = Lexer(src).tokenize()
    mod = Parser(tokens).parse()
    ns = execute(mod)
    
    # Decorators applied bottom-up: message("hello") -> ">> hello" -> ">> hello!"
    assert ns["result"] == ">> hello!"


def test_conditional_decorator():
    """Test decorator that conditionally modifies behavior"""
    src = '''
fn conditional_logging(should_log) {
    fn decorator(func) {
        fn wrapper(x) {
            if should_log {
                print("calling function")
            }
            return func(x)
        }
        return wrapper
    }
    return decorator
}

@conditional_logging(false)
fn quiet_func(x) {
    return x * 2
}

result = quiet_func(5)
'''
    
    tokens = Lexer(src).tokenize()
    mod = Parser(tokens).parse()
    ns = execute(mod)
    
    assert ns["result"] == 10

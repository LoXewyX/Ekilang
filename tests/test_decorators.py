"""Test decorator functionality"""

from ekilang.lexer import Lexer
from ekilang.parser import Parser
from ekilang.runtime import execute


def test_simple_function_decorator():
    """Test basic function decorator"""
    src = '''
fn uppercase(func) {
    fn wrapper(text) {
        result = func(text)
        return result.upper()
    }
    return wrapper
}

@uppercase
fn greet(name) {
    return f"hello, {name}"
}

msg = greet("bob")
'''
    
    tokens = Lexer(src).tokenize()
    mod = Parser(tokens).parse()
    ns = execute(mod)
    
    assert ns["msg"] == "HELLO, BOB"


def test_decorator_with_arguments():
    """Test decorator that takes arguments"""
    src = '''
fn repeat(times) {
    fn decorator(func) {
        fn wrapper() {
            results = []
            i = 0
            while i < times {
                results.append(func())
                i = i + 1
            }
            return results
        }
        return wrapper
    }
    return decorator
}

@repeat(3)
fn say_hi() {
    return "Hi"
}

output = say_hi()
count = len(output)
'''
    
    tokens = Lexer(src).tokenize()
    mod = Parser(tokens).parse()
    ns = execute(mod)
    
    assert ns["count"] == 3
    assert ns["output"] == ["Hi", "Hi", "Hi"]


def test_multiple_decorators():
    """Test stacking multiple decorators"""
    src = '''
fn add_one(func) {
    fn wrapper(x) {
        return func(x) + 1
    }
    return wrapper
}

fn double(func) {
    fn wrapper(x) {
        return func(x) * 2
    }
    return wrapper
}

@add_one
@double
fn get_value(x) {
    return x
}

result = get_value(5)
'''
    
    tokens = Lexer(src).tokenize()
    mod = Parser(tokens).parse()
    ns = execute(mod)
    
    # Decorators are applied bottom-up: double(5) = 10, then add_one(10) = 11
    assert ns["result"] == 11


def test_class_decorator():
    """Test decorator on a class"""
    src = '''
fn add_method(cls) {
    fn new_method(self) {
        return "decorated!"
    }
    cls.new_method = new_method
    return cls
}

@add_method
class MyClass {
    fn __init__(self) {
        self.value = 42
    }
}

obj = MyClass()
msg = obj.new_method()
'''
    
    tokens = Lexer(src).tokenize()
    mod = Parser(tokens).parse()
    ns = execute(mod)
    
    assert ns["msg"] == "decorated!"


def test_decorator_preserves_function():
    """Test that decorator can call original function"""
    src = '''
fn logger(func) {
    fn wrapper(x) {
        result = func(x)
        return result * 10
    }
    return wrapper
}

@logger
fn add_five(x) {
    return x + 5
}

result = add_five(3)
'''
    
    tokens = Lexer(src).tokenize()
    mod = Parser(tokens).parse()
    ns = execute(mod)
    
    # add_five(3) = 8, then logger multiplies by 10 = 80
    assert ns["result"] == 80


def test_decorator_with_call_syntax():
    """Test decorator using call syntax @dec()"""
    src = '''
fn make_multiplier(factor) {
    fn decorator(func) {
        fn wrapper(x) {
            return func(x) * factor
        }
        return wrapper
    }
    return decorator
}

@make_multiplier(3)
fn get_number(n) {
    return n
}

result = get_number(7)
'''
    
    tokens = Lexer(src).tokenize()
    mod = Parser(tokens).parse()
    ns = execute(mod)
    
    assert ns["result"] == 21  # 7 * 3


def test_multiple_class_decorators():
    """Test multiple decorators on a class"""
    src = '''
fn add_x(cls) {
    cls.x = 10
    return cls
}

fn add_y(cls) {
    cls.y = 20
    return cls
}

@add_y
@add_x
class Point {
    fn __init__(self) {
        self.z = 0
    }
}

p = Point()
total = Point.x + Point.y
'''
    
    tokens = Lexer(src).tokenize()
    mod = Parser(tokens).parse()
    ns = execute(mod)
    
    assert ns["total"] == 30

"""Tests for global and nonlocal statement functionality in Ekilang."""

from pathlib import Path
import sys
from ekilang.lexer import Lexer
from ekilang.parser import Parser
from ekilang.executor import execute

sys.path.insert(0, str(Path(__file__).parent.parent))


def run(code: str):
    """Helper to run code snippets and return the namespace"""
    tokens = Lexer(code).tokenize()
    mod = Parser(tokens).parse()
    return execute(mod)


def test_global_basic():
    """Test basic global statement"""
    code = """
x = 10

fn modify_global() {
    global x
    x = 20
}

modify_global()
"""
    ns = run(code)
    assert ns["x"] == 20


def test_global_multiple_names():
    """Test global statement with multiple variables"""
    code = """
a = 1
b = 2
c = 3

fn modify_all() {
    global a, b, c
    a = 10
    b = 20
    c = 30
}

modify_all()
"""
    ns = run(code)
    assert ns["a"] == 10
    assert ns["b"] == 20
    assert ns["c"] == 30


def test_global_create_new():
    """Test creating a new global variable with global statement"""
    code = """
fn create_global() {
    global new_var
    new_var = 42
}

create_global()
"""
    ns = run(code)
    assert ns["new_var"] == 42


def test_global_in_nested_function():
    """Test global in nested function"""
    code = """
x = 5

fn outer() {
    fn inner() {
        global x
        x = 100
    }
    inner()
}

outer()
"""
    ns = run(code)
    assert ns["x"] == 100


def test_nonlocal_basic():
    """Test basic nonlocal statement"""
    code = """
fn outer() {
    x = 10
    fn inner() {
        nonlocal x
        x = 20
    }
    inner()
    return x
}

result = outer()
"""
    ns = run(code)
    assert ns["result"] == 20


def test_nonlocal_multiple_names():
    """Test nonlocal statement with multiple variables"""
    code = """
fn outer() {
    a = 1
    b = 2
    fn inner() {
        nonlocal a, b
        a = 10
        b = 20
    }
    inner()
    return (a, b)
}

result = outer()
"""
    ns = run(code)
    assert ns["result"] == (10, 20)


def test_nonlocal_in_deeply_nested_function():
    """Test nonlocal in deeply nested functions"""
    code = """
fn level1() {
    x = 1
    fn level2() {
        fn level3() {
            nonlocal x
            x = 3
        }
        level3()
    }
    level2()
    return x
}

result = level1()
"""
    ns = run(code)
    assert ns["result"] == 3


def test_global_vs_nonlocal():
    """Test interaction between global and nonlocal"""
    code = """
global_var = 100

fn outer() {
    local_var = 10
    fn inner() {
        global global_var
        nonlocal local_var
        global_var = 200
        local_var = 20
    }
    inner()
    return local_var
}

result = outer()
"""
    ns = run(code)
    assert ns["global_var"] == 200
    assert ns["result"] == 20


def test_global_with_loop():
    """Test global statement in a loop context"""
    code = """
counter = 0

fn increment() {
    global counter
    counter = counter + 1
}

loop {
    increment()
    if counter >= 5 {
        break
    }
}
"""
    ns = run(code)
    assert ns["counter"] == 5


def test_nonlocal_with_closure():
    """Test nonlocal in closure for state preservation"""
    code = """
fn make_counter() {
    count = 0
    fn increment() {
        nonlocal count
        count = count + 1
        return count
    }
    return increment
}

counter = make_counter()
r1 = counter()
r2 = counter()
r3 = counter()
"""
    ns = run(code)
    assert ns["r1"] == 1
    assert ns["r2"] == 2
    assert ns["r3"] == 3


def test_global_in_lambda():
    """Test global statement in lambda (indirectly through function)"""
    code = """
x = 10

fn make_modifier() {
    global x
    return (v) => {
        global x
        x = v
    }
}

fn modify(val) {
    global x
    x = val
}

modify(50)
"""
    ns = run(code)
    assert ns["x"] == 50


def test_nonlocal_without_enclosing_scope():
    """Test that nonlocal in module scope doesn't break (just uses local)"""
    code = """
x = 5

fn outer() {
    x = 10
    fn inner() {
        nonlocal x
        x = 20
    }
    inner()
    return x
}

result = outer()
"""
    ns = run(code)
    assert ns["result"] == 20
    assert ns["x"] == 5


def test_global_in_class_method():
    """Test global statement in class method"""
    code = """
counter = 0

class Counter {
    fn increment(self) {
        global counter
        counter = counter + 1
    }
}

c = Counter()
c.increment()
c.increment()
"""
    ns = run(code)
    assert ns["counter"] == 2


def test_nonlocal_multiple_levels():
    """Test nonlocal skipping levels to find binding"""
    code = """
fn level1() {
    x = 1
    fn level2() {
        x = 2
        fn level3() {
            nonlocal x
            x = 3
        }
        level3()
        return x
    }
    return level2()
}

result = level1()
"""
    ns = run(code)
    assert ns["result"] == 3


def test_global_modification_across_functions():
    """Test modifying global variable across multiple functions"""
    code = """
shared = []

fn add_item(item) {
    global shared
    shared = shared + [item]
}

fn get_shared() {
    global shared
    return shared
}

add_item(1)
add_item(2)
add_item(3)
result = get_shared()
"""
    ns = run(code)
    assert ns["result"] == [1, 2, 3]


def test_nonlocal_with_reassignment():
    """Test reassigning nonlocal variable"""
    code = """
fn outer() {
    x = [1, 2, 3]
    fn inner() {
        nonlocal x
        x = [4, 5, 6]
    }
    inner()
    return x
}

result = outer()
"""
    ns = run(code)
    assert ns["result"] == [4, 5, 6]


def test_global_declaration_only():
    """Test global declaration without assignment"""
    code = """
x = 100

fn read_global() {
    global x
    return x
}

val = read_global()
"""
    ns = run(code)
    assert ns["val"] == 100


def test_nonlocal_declaration_only():
    """Test nonlocal declaration without assignment"""
    code = """
fn outer() {
    y = 50
    fn inner() {
        nonlocal y
        return y
    }
    return inner()
}

result = outer()
"""
    ns = run(code)
    assert ns["result"] == 50


def test_global_with_for_loop():
    """Test global inside a for loop"""
    code = """
total = 0

fn sum_range(n) {
    global total
    for i in range(n) {
        total = total + i
    }
}

sum_range(5)
"""
    ns = run(code)
    assert ns["total"] == 10  # 0 + 1 + 2 + 3 + 4


def test_nonlocal_with_for_loop():
    """Test nonlocal inside a for loop"""
    code = """
fn outer() {
    total = 0
    fn inner(n) {
        nonlocal total
        for i in range(n) {
            total = total + i
        }
    }
    inner(5)
    return total
}

result = outer()
"""
    ns = run(code)
    assert ns["result"] == 10  # 0 + 1 + 2 + 3 + 4

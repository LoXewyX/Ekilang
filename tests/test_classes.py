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


def test_simple_class():
    """Test defining and instantiating a simple class"""
    code = """
    class Person {
        fn __init__(self, name) {
            self.name = name
        }
    }
    p = Person("Alice")
    """
    ns = run(code)
    assert ns["p"].name == "Alice"


def test_class_with_method():
    """Test class with methods"""
    code = """
    class Counter {
        fn __init__(self) {
            self.count = 0
        }
        
        fn increment(self) {
            self.count = self.count + 1
        }
        
        fn get_count(self) {
            return self.count
        }
    }
    c = Counter()
    c.increment()
    c.increment()
    result = c.get_count()
    """
    ns = run(code)
    assert ns["result"] == 2


def test_class_with_properties():
    """Test class with properties and area calculation"""
    code = """
    class Rectangle {
        fn __init__(self, width, height) {
            self.width = width
            self.height = height
        }
        
        fn area(self) {
            return self.width * self.height
        }
    }
    rect = Rectangle(5, 3)
    a = rect.area()
    """
    ns = run(code)
    assert ns["a"] == 15


def test_class_inheritance():
    """Test class inheritance and method overriding"""
    code = """
    class Animal {
        fn __init__(self, name) {
            self.name = name
        }
        
        fn speak(self) {
            return "Some sound"
        }
    }
    
    class Dog(Animal) {
        fn speak(self) {
            return "Woof!"
        }
    }
    
    d = Dog("Buddy")
    sound = d.speak()
    """
    ns = run(code)
    assert ns["d"].name == "Buddy"
    assert ns["sound"] == "Woof!"


def test_class_multiple_instances():
    """Test creating multiple instances of a class"""
    code = """
    class Point {
        fn __init__(self, x, y) {
            self.x = x
            self.y = y
        }
    }
    p1 = Point(1, 2)
    p2 = Point(3, 4)
    """
    ns = run(code)
    assert ns["p1"].x == 1
    assert ns["p1"].y == 2
    assert ns["p2"].x == 3
    assert ns["p2"].y == 4


def test_class_with_default_params():
    """Test class with default parameters in __init__"""
    code = """
    class Config {
        fn __init__(self, timeout = 30) {
            self.timeout = timeout
        }
    }
    c1 = Config()
    c2 = Config(60)
    """
    ns = run(code)
    assert ns["c1"].timeout == 30
    assert ns["c2"].timeout == 60


def test_class_str_method():
    """Test class with __str__ method"""
    code = """
    class Book {
        fn __init__(self, title) {
            self.title = title
        }
        
        fn __str__(self) {
            return f"Book: {self.title}"
        }
    }
    b = Book("Python Guide")
    s = str(b)
    """
    ns = run(code)
    assert ns["s"] == "Book: Python Guide"


def test_empty_class():
    """Test defining and instantiating an empty class"""
    code = """
    class Empty {
    }
    e = Empty()
    """
    ns = run(code)
    assert ns["e"] is not None


def test_staticmethod_basic():
    """Test basic @staticmethod decorator"""
    code = """
    class MathUtils {
        @staticmethod
        fn add(a, b) {
            return a + b
        }
        
        @staticmethod
        fn multiply(x, y) {
            return x * y
        }
    }
    
    result1 = MathUtils.add(5, 3)
    result2 = MathUtils.multiply(4, 7)
    """
    ns = run(code)
    assert ns["result1"] == 8
    assert ns["result2"] == 28


def test_staticmethod_call_from_instance():
    """Test calling staticmethod from instance"""
    code = """
    class Calculator {
        @staticmethod
        fn square(n) {
            return n * n
        }
    }
    
    calc = Calculator()
    result_class = Calculator.square(5)
    result_instance = calc.square(5)
    """
    ns = run(code)
    assert ns["result_class"] == 25
    assert ns["result_instance"] == 25


def test_staticmethod_with_regular_method():
    """Test mixing staticmethod with regular methods"""
    code = """
    class Temperature {
        fn __init__(self, celsius) {
            self.celsius = celsius
        }
        
        @staticmethod
        fn celsius_to_fahrenheit(c) {
            return c * 9 / 5 + 32
        }
        
        fn to_fahrenheit(self) {
            return Temperature.celsius_to_fahrenheit(self.celsius)
        }
    }
    
    temp = Temperature(100)
    result1 = Temperature.celsius_to_fahrenheit(0)
    result2 = temp.to_fahrenheit()
    """
    ns = run(code)
    assert ns["result1"] == 32
    assert ns["result2"] == 212


def test_staticmethod_factory_pattern():
    """Test staticmethod for factory pattern"""
    code = """
    class Point {
        fn __init__(self, x, y) {
            self.x = x
            self.y = y
        }
        
        @staticmethod
        fn origin() {
            return Point(0, 0)
        }
        
        @staticmethod
        fn from_tuple(t) {
            return Point(t[0], t[1])
        }
    }
    
    p1 = Point.origin()
    p2 = Point.from_tuple((3, 4))
    """
    ns = run(code)
    assert ns["p1"].x == 0
    assert ns["p1"].y == 0
    assert ns["p2"].x == 3
    assert ns["p2"].y == 4

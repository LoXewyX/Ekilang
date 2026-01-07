"""Tests for with statement (context manager) functionality in Ekilang."""

from pathlib import Path
import sys
from ekilang.lexer import Lexer
from ekilang.parser import Parser
from ekilang.executor import execute
import tempfile
import os

sys.path.insert(0, str(Path(__file__).parent.parent))


def run(code: str):
    """Helper to run code snippets and return the namespace"""
    tokens = Lexer(code).tokenize()
    mod = Parser(tokens).parse()
    return execute(mod)


def test_with_file_basic():
    """Test basic with statement for file operations"""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        temp_path = f.name
        f.write("Hello, Ekilang!")

    try:
        ns = run(
            f"""
result = ""
with open("{temp_path.replace(chr(92), chr(92)*2)}", "r") as f {{
    result = f.read()
}}
"""
        )
        assert ns["result"] == "Hello, Ekilang!"
    finally:
        os.unlink(temp_path)


def test_with_file_write():
    """Test with statement for file writing"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp:
        temp_path = tmp.name

    try:
        run(
            f"""
with open("{temp_path.replace(chr(92), chr(92)*2)}", "w") as f {{
    f.write("Test content")
}}
"""
        )

        # Verify the file was written
        with open(temp_path, "r") as f:
            content = f.read()
        assert content == "Test content"
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_with_multiple_context_managers():
    """Test with statement with multiple context managers"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp1:
        temp_path1 = tmp1.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp2:
        temp_path2 = tmp2.name

    # Create first file
    with open(temp_path1, "w") as f:
        f.write("Content 1")

    try:
        run(
            f"""
with open("{temp_path1.replace(chr(92), chr(92)*2)}", "r") as f1, open("{temp_path2.replace(chr(92), chr(92)*2)}", "w") as f2 {{
    content = f1.read()
    f2.write(content + " modified")
}}
"""
        )

        # Verify second file
        with open(temp_path2, "r") as f:
            content = f.read()
        assert content == "Content 1 modified"
    finally:
        for path in [temp_path1, temp_path2]:
            if os.path.exists(path):
                os.unlink(path)


def test_with_nested():
    """Test nested with statements"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp:
        temp_path = tmp.name

    with open(temp_path, "w") as f:
        f.write("Nested test")

    try:
        ns = run(
            f"""
result = []
with open("{temp_path.replace(chr(92), chr(92)*2)}", "r") as f1 {{
    line1 = f1.read()
    result.append(line1)
    with open("{temp_path.replace(chr(92), chr(92)*2)}", "r") as f2 {{
        line2 = f2.read()
        result.append(line2)
    }}
}}
"""
        )
        assert ns["result"] == ["Nested test", "Nested test"]
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_with_variable_available():
    """Test that with variable is accessible in the block"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp:
        temp_path = tmp.name

    with open(temp_path, "w") as f:
        f.write("Variable test")

    try:
        ns = run(
            f"""
file_obj = none
with open("{temp_path.replace(chr(92), chr(92)*2)}", "r") as f {{
    file_obj = f
    content = f.read()
}}
result = content
"""
        )
        assert ns["result"] == "Variable test"
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_with_in_function():
    """Test with statement inside a function"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp:
        temp_path = tmp.name

    with open(temp_path, "w") as f:
        f.write("Function test")

    try:
        ns = run(
            f"""
fn read_file(path) {{
    with open(path, "r") as f {{
        return f.read()
    }}
}}

result = read_file("{temp_path.replace(chr(92), chr(92)*2)}")
"""
        )
        assert ns["result"] == "Function test"
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_with_exception_handling():
    """Test that with statement properly handles exceptions"""
    ns = run(
        """
result = "not_set"
try {
    with open("/nonexistent/file.txt", "r") as f {
        result = f.read()
    }
} except FileNotFoundError {
    result = "file_not_found"
}
"""
    )
    assert ns["result"] == "file_not_found"


def test_with_custom_context_manager():
    """Test with statement with custom context manager using contextlib"""
    ns = run(
        """
use contextlib

@contextlib.contextmanager
fn my_context() {
    print("Enter")
    yield "custom_value"
    print("Exit")
}

result = []
with my_context() as value {
    result.append("inside")
    result.append(value)
}
result.append("after")
"""
    )
    assert ns["result"] == ["inside", "custom_value", "after"]


def test_with_custom_context_no_yield_value():
    """Test custom context manager without yielding a value"""
    ns = run(
        """
use contextlib

@contextlib.contextmanager
fn simple_context() {
    yield
}

executed = false
with simple_context() {
    executed = true
}
"""
    )
    assert ns["executed"] == True


def test_with_custom_context_state_tracking():
    """Test that custom context manager tracks entry and exit"""
    ns = run(
        """
use contextlib

state = []

@contextlib.contextmanager
fn tracking_context() {
    state.append("enter")
    yield
    state.append("exit")
}

with tracking_context() {
    state.append("inside")
}
state.append("after")
"""
    )
    assert ns["state"] == ["enter", "inside", "exit", "after"]


def test_async_with_custom_context_manager():
    """Test async with statement with async context manager"""
    ns = run(
        """
use asyncio

class AsyncCtx {
    async fn __aenter__(self) {
        "entered"
    }
    async fn __aexit__(self, exc_type, exc, tb) {
        none
    }
}

async fn main() {
    result = []
    async with AsyncCtx() as ctx {
        result.append(ctx)
    }
    result
}

result = asyncio.run(main())
"""
    )
    assert ns["result"] == ["entered"]

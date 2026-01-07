"""Tests for pipe operators |> and <| in the Ekilang language."""

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


def test_forward_pipe_basic():
    """Test basic forward pipe operator |>"""
    ns = run(
        """
fn double(x) { x * 2 }
fn add_ten(x) { x + 10 }

result = 5 |> double
result2 = 5 |> double |> add_ten
        """
    )
    assert ns["result"] == 10
    assert ns["result2"] == 20


def test_backward_pipe_basic():
    """Test basic backward pipe operator <|"""
    ns = run(
        """
fn triple(x) { x * 3 }
fn subtract_five(x) { x - 5 }

result = triple <| 4
result2 = subtract_five <| triple <| 4
        """
    )
    assert ns["result"] == 12
    assert ns["result2"] == 7


def test_pipe_with_builtins():
    """Test pipeline with builtin functions"""
    ns = run(
        """
data = [1, 2, 3, 4, 5]
total = data |> sum
str_len = "hello world" |> len
        """
    )
    assert ns["total"] == 15
    assert ns["str_len"] == 11


def test_pipe_data_transformation():
    """Test complex data transformation pipeline"""
    ns = run(
        """
fn square_all(nums) {
    [x * x for x in nums]
}

fn filter_gt_ten(nums) {
    [x for x in nums if x > 10]
}

fn sum_list(nums) {
    sum(nums)
}

nums = [1, 2, 3, 4, 5]
result = nums |> square_all |> filter_gt_ten |> sum_list
        """
    )
    # [1,4,9,16,25] -> [16,25] -> 41
    assert ns["result"] == 41


def test_pipe_mixed_operators():
    """Test mixing both pipe operators"""
    ns = run(
        """
fn add_one(x) { x + 1 }
fn double(x) { x * 2 }

# 10 |> add_one = 11, then double <| 11 = 22
result = double <| (10 |> add_one)
        """
    )
    assert ns["result"] == 22

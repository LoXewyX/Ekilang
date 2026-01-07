"""Tests for async for loops in various contexts."""

from ekilang.executor import execute
from ekilang.lexer import Lexer
from ekilang.parser import Parser


def run(code: str):
    tokens = Lexer(code).tokenize()
    mod = Parser(tokens).parse()
    return execute(mod)


def test_async_with_and_async_for_streaming():
    """Async with + async for over streamed chunks"""
    ns = run(
        """
use asyncio

class ChunkStream {
    fn __init__(self, chunks) {
        self.chunks = chunks
    }
    fn __aiter__(self) {
        async fn iter_chunks(chunks) {
            for chunk in chunks {
                yield chunk
            }
        }
        iter_chunks(self.chunks)
    }
}

class Response {
    fn __init__(self, chunks) {
        self.content = ChunkStream(chunks)
    }
    async fn __aenter__(self) { self }
    async fn __aexit__(self, exc_type, exc, tb) { none }
}

class Session {
    async fn get(self, url) {
        Response([f"{url}-chunk1", f"{url}-chunk2", f"{url}-chunk3"])
    }
}

async fn fetch(url) {
    session = Session()
    results = []
    async with await session.get(url) as response {
        async for chunk in response.content {
            results.append(chunk)
        }
    }
    results
}

result = asyncio.run(fetch("http://example.com"))
"""
    )
    assert ns["result"] == [
        "http://example.com-chunk1",
        "http://example.com-chunk2",
        "http://example.com-chunk3",
    ]

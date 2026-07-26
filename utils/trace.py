"""
Per-request tracing for the RAG pipeline.

Logging here is `print()` with a level prefix, matching the rest of the codebase.
What was missing was not a logging library but the *data*: nothing recorded what
retrieval actually scored on each request. Diagnosing the stale-embedding bug -
where stored vectors came from a different model than the query vectors - took
five throwaway scripts, because a chunk scoring 0.350 when it should have scored
0.606 left no trace anywhere. Similarity is now logged on every request, so a
whole corpus drifting downward is visible by reading the log.

One request produces exactly one [TRACE] line:

    [TRACE] q="who won the 2024 election" lang=en intent=question mode=default
            max_sim=0.606 threshold=0.35 gate=pass chunks=15/3src
            retrieval=0.41s generate=11.2s verify=1.8s total=13.6s
"""
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

# Queries are truncated in the log line; the full text is in the request body.
MAX_QUERY_CHARS = 60


class RequestTrace:
    """
    Collects fields and stage timings for one request, then emits a single line.

    Deliberately not a logging.Logger: this project's convention is prefixed
    print(), and one structured line per request is easier to grep and to paste
    into a bug report than a scattering of records.
    """

    def __init__(self, query: str, session_id: str = "default"):
        self.query = query
        self.session_id = session_id
        self.started = time.perf_counter()
        self.fields: Dict[str, Any] = {}
        self.timings: Dict[str, float] = {}
        self._emitted = False

    def set(self, **fields: Any) -> "RequestTrace":
        """Record fields to appear in the trace line. Returns self for chaining."""
        self.fields.update({k: v for k, v in fields.items() if v is not None})
        return self

    @contextmanager
    def stage(self, name: str):
        """Time a pipeline stage. Timing is recorded even if the stage raises."""
        start = time.perf_counter()
        try:
            yield self
        finally:
            self.timings[name] = time.perf_counter() - start

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def _format(self) -> str:
        query = self.query.replace('"', "'").replace("\n", " ")
        if len(query) > MAX_QUERY_CHARS:
            query = query[: MAX_QUERY_CHARS - 3] + "..."

        parts = [f'q="{query}"']
        parts += [f"{k}={self._render(v)}" for k, v in self.fields.items()]
        parts += [f"{k}={v:.2f}s" for k, v in self.timings.items()]
        parts.append(f"total={self.elapsed:.2f}s")
        return " ".join(parts)

    @staticmethod
    def _render(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    def emit(self, level: str = "TRACE") -> None:
        """
        Print the trace line. Safe to call more than once; only the first prints.

        Flushed explicitly: a trace that is still sitting in a block buffer when
        the request that produced it hangs or crashes the process is worthless,
        and those are the requests worth tracing.
        """
        if self._emitted:
            return
        self._emitted = True
        print(f"[{level}] {self._format()}", flush=True)


@contextmanager
def trace_request(query: str, session_id: str = "default"):
    """
    Trace one request, emitting the line on the way out even if it raises.

    A request that blows up mid-pipeline is exactly the one worth a trace line,
    so the emit happens in a finally block and failures are tagged.
    """
    trace = RequestTrace(query, session_id)
    try:
        yield trace
    except Exception as exc:
        trace.set(outcome="error", error=type(exc).__name__)
        trace.emit(level="ERROR")
        raise
    else:
        trace.emit()


def format_similarity_spread(similarities: Optional[list]) -> str:
    """
    Compact "max/median/min" for a similarity list, for the trace line.

    The spread matters more than the max on its own: a high max with a collapsed
    spread means one lucky chunk, while every score sliding down together is the
    signature of an embedding-space mismatch rather than a hard query.
    """
    if not similarities:
        return "none"
    ordered = sorted(similarities, reverse=True)
    median = ordered[len(ordered) // 2]
    return f"{ordered[0]:.3f}/{median:.3f}/{ordered[-1]:.3f}"

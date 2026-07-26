"""
Unit tests for per-request tracing.

The trace line is the only place a similarity collapse becomes visible without
attaching a debugger, so its content is worth guarding.
"""
import pytest

from utils.trace import RequestTrace, format_similarity_spread, trace_request


class TestSimilaritySpread:
    def test_reports_max_median_min(self):
        assert format_similarity_spread([0.2, 0.6, 0.4]) == "0.600/0.400/0.200"

    def test_empty_list_is_reported_not_crashed(self):
        assert format_similarity_spread([]) == "none"

    def test_none_is_reported_not_crashed(self):
        assert format_similarity_spread(None) == "none"

    def test_single_value_repeats(self):
        assert format_similarity_spread([0.5]) == "0.500/0.500/0.500"


class TestRequestTrace:
    def test_fields_appear_in_the_line(self, capsys):
        RequestTrace("who won?", "s1").set(lang="en", gate="pass").emit()

        line = capsys.readouterr().out
        assert "lang=en" in line
        assert "gate=pass" in line
        assert 'q="who won?"' in line

    def test_floats_are_rounded(self, capsys):
        RequestTrace("q").set(conf=0.8123456).emit()

        assert "conf=0.812" in capsys.readouterr().out

    def test_none_values_are_dropped(self, capsys):
        RequestTrace("q").set(present="yes", absent=None).emit()

        out = capsys.readouterr().out
        assert "present=yes" in out
        assert "absent" not in out

    def test_long_queries_are_truncated(self, capsys):
        RequestTrace("x" * 200).emit()

        line = capsys.readouterr().out
        assert "..." in line
        assert len(line) < 150

    def test_quotes_and_newlines_do_not_break_the_line(self, capsys):
        RequestTrace('who said "hello"\nand more').emit()

        out = capsys.readouterr().out
        assert out.count("\n") == 1
        assert '"' in out  # the wrapping quotes survive

    def test_stage_timings_are_recorded(self, capsys):
        trace = RequestTrace("q")
        with trace.stage("retrieval"):
            pass
        trace.emit()

        assert "retrieval=" in capsys.readouterr().out

    def test_stage_is_timed_even_when_it_raises(self):
        trace = RequestTrace("q")

        with pytest.raises(ValueError):
            with trace.stage("generate"):
                raise ValueError("boom")

        assert "generate" in trace.timings

    def test_total_is_always_present(self, capsys):
        RequestTrace("q").emit()
        assert "total=" in capsys.readouterr().out

    def test_emitting_twice_prints_once(self, capsys):
        trace = RequestTrace("q")
        trace.emit()
        trace.emit()

        assert capsys.readouterr().out.count("[TRACE]") == 1

    def test_set_returns_self_for_chaining(self):
        trace = RequestTrace("q")
        assert trace.set(a=1).set(b=2) is trace


class TestTraceRequest:
    def test_successful_request_emits_a_trace_line(self, capsys):
        with trace_request("who won?") as trace:
            trace.set(outcome="answered")

        assert "[TRACE]" in capsys.readouterr().out

    def test_failure_emits_an_error_line_and_re_raises(self, capsys):
        with pytest.raises(RuntimeError):
            with trace_request("who won?") as trace:
                trace.set(lang="en")
                raise RuntimeError("llm down")

        out = capsys.readouterr().out
        assert "[ERROR]" in out
        assert "outcome=error" in out
        assert "error=RuntimeError" in out
        # Whatever was recorded before the failure is still reported - that is
        # the point of tracing a request that died mid-pipeline.
        assert "lang=en" in out

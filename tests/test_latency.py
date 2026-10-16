import numpy as np
import pytest

from policyjev.evaluation import latency as lt
from policyjev.jev.cache import JevCache


class FakeClock:
    """Advances by a scripted amount on each call of work()."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_sample_is_fixed_and_bounded():
    texts = [f"comment {i}" for i in range(100)]
    first = lt.sample_texts(texts, 20, seed=42)
    again = lt.sample_texts(texts, 20, seed=42)
    other = lt.sample_texts(texts, 20, seed=43)

    assert first == again
    assert first != other
    assert len(set(first)) == 20
    assert len(lt.sample_texts(texts, 500, seed=42)) == 100


def test_single_latency_percentiles_use_the_clock_and_skip_warmup():
    clock = FakeClock()
    calls = []

    def predict_one(text: str) -> None:
        calls.append(text)
        if len(calls) > 3:
            clock.now += 0.001 * (len(calls) - 3)

    texts = [f"t{i}" for i in range(10)]
    p50, p95 = lt.single_latency_ms(predict_one, texts, warmup=3, clock=clock)

    durations = [1.0 * k for k in range(1, 11)]
    assert len(calls) == 3 + 10
    assert p50 == pytest.approx(np.percentile(durations, 50))
    assert p95 == pytest.approx(np.percentile(durations, 95))


def test_throughput_counts_comments_over_elapsed_time():
    clock = FakeClock()
    batches = []

    def predict_batch(batch: list[str]) -> None:
        batches.append(len(batch))
        if len(batches) > 1:
            clock.now += 0.5

    texts = [f"t{i}" for i in range(64)]
    rate = lt.throughput_per_second(predict_batch, texts, batch_size=32, clock=clock)

    assert batches == [32, 32, 32]
    assert rate == pytest.approx(64 / 1.0)


def test_sync_is_called_around_timed_work():
    events = []
    lt.single_latency_ms(lambda t: events.append("work"), ["a", "b"], warmup=1, sync=lambda: events.append("sync"))
    assert events.count("sync") == 3


def test_jev_latency_comes_from_recorded_calls(tmp_path):
    with JevCache(tmp_path / "jev.sqlite") as cache:
        cache.log_call("a", 1, "ok", 200, 1000, 10, 0.002, 0.2, None)
        cache.log_call("b", 1, "ok", 200, 1000, 10, 0.002, 0.4, None)
        cache.log_call("c", 1, "retryable_error", 429, 0, 0, 0.0, 0.1, "slow down")
        summary = lt.jev_latency_cost(cache)

    assert summary["n_requests"] == 2
    assert summary["latency_p50_ms"] == pytest.approx(300.0)
    assert "cost_per_1k_comments_usd" not in summary


def test_jev_summary_with_no_calls_is_nan(tmp_path):
    with JevCache(tmp_path / "jev.sqlite") as cache:
        summary = lt.jev_latency_cost(cache)
    assert summary["n_requests"] == 0
    assert np.isnan(summary["latency_p50_ms"]) or summary["latency_p50_ms"] == 0


def test_machine_info_describes_the_environment():
    info = lt.machine_info("cpu")
    assert info["device"] == "cpu"
    for key in ("device_name", "python_version", "torch_version", "sklearn_version"):
        assert info[key]

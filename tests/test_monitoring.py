"""Tests for /stats, the stats middleware and the JSON access log (src.monitoring, src.app)."""

import json
import logging
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.app import app
from src.model_loader import POINTER_ENV_VAR
from src.monitoring import (
    ACCESS_HANDLER_NAME,
    ACCESS_LOGGER_NAME,
    StatsCollector,
    configure_access_logger,
    status_class,
)

# ---------------------------------------------------------------------------
# StatsCollector
# ---------------------------------------------------------------------------


def test_status_class() -> None:
    assert [status_class(code) for code in (200, 307, 404, 422, 503)] == [
        "2xx",
        "3xx",
        "4xx",
        "4xx",
        "5xx",
    ]


def test_empty_collector_snapshot() -> None:
    snapshot = StatsCollector().snapshot()
    assert snapshot["requests"] == {
        "total": 0,
        "by_status_class": {},
        "avg_latency_ms": None,
        "by_endpoint": {},
    }
    assert snapshot["predictions"] == {
        "count": 0,
        "by_label": {"benign": 0, "dga": 0},
        "mean_probability_dga": None,
        "out_of_scope": 0,
    }
    assert snapshot["uptime_seconds"] >= 0


def test_collector_averages() -> None:
    stats = StatsCollector()
    stats.record_request("POST /predict", 200, 10.0)
    stats.record_request("POST /predict", 422, 20.0)
    stats.record_request("GET /health", 200, 3.0)
    stats.record_prediction("dga", 0.9)
    stats.record_prediction("benign", 0.1)
    stats.record_prediction("benign", 0.2)

    snapshot = stats.snapshot()
    assert snapshot["requests"]["total"] == 3
    assert snapshot["requests"]["by_status_class"] == {"2xx": 2, "4xx": 1}
    assert snapshot["requests"]["avg_latency_ms"] == pytest.approx(11.0)
    assert snapshot["requests"]["by_endpoint"]["POST /predict"] == {
        "count": 2,
        "by_status_class": {"2xx": 1, "4xx": 1},
        "avg_latency_ms": 15.0,
    }
    assert snapshot["predictions"] == {
        "count": 3,
        "by_label": {"benign": 2, "dga": 1},
        "mean_probability_dga": 0.4,
        "out_of_scope": 0,
    }


def test_collector_is_thread_safe() -> None:
    stats = StatsCollector()
    threads, per_thread = 8, 1000

    def work() -> None:
        for _ in range(per_thread):
            stats.record_request("POST /predict", 200, 1.0)
            stats.record_prediction("dga", 1.0)

    workers = [threading.Thread(target=work) for _ in range(threads)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    snapshot = stats.snapshot()
    assert snapshot["requests"]["total"] == threads * per_thread
    assert snapshot["requests"]["by_endpoint"]["POST /predict"]["count"] == threads * per_thread
    assert snapshot["predictions"]["by_label"]["dga"] == threads * per_thread


# ---------------------------------------------------------------------------
# GET /stats through the app
# ---------------------------------------------------------------------------


def endpoint(body: dict, key: str) -> dict:
    return body["requests"]["by_endpoint"][key]


def test_stats_on_a_fresh_app(client: TestClient) -> None:
    response = client.get("/stats")
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "test_fixture"
    assert body["requests"]["total"] == 0  # the /stats call itself is counted afterwards
    assert body["predictions"]["count"] == 0


def test_request_counts_rise(client: TestClient) -> None:
    client.get("/health")
    client.get("/health")
    client.post("/predict", json={"domain": "google.com"})
    client.post("/predict", json={"domain": "-bad.com"})  # 422

    body = client.get("/stats").json()
    assert body["requests"]["total"] == 4
    assert body["requests"]["by_status_class"] == {"2xx": 3, "4xx": 1}
    assert endpoint(body, "GET /health")["count"] == 2
    assert endpoint(body, "POST /predict")["by_status_class"] == {"2xx": 1, "4xx": 1}

    after = client.get("/stats").json()
    assert after["requests"]["total"] == 5
    assert endpoint(after, "GET /stats")["count"] == 1


def test_average_latency_is_positive(client: TestClient) -> None:
    client.post("/predict", json={"domain": "google.com"})
    body = client.get("/stats").json()
    assert body["requests"]["avg_latency_ms"] > 0
    assert endpoint(body, "POST /predict")["avg_latency_ms"] > 0
    assert body["uptime_seconds"] > 0


def test_label_counts_match_predictions(client: TestClient) -> None:
    domains = ["google.com", "moiqidfvjjhjmndqp.in", "teco.com.tw", "qxxsoxcwrcplo.net"]
    results = [client.post("/predict", json={"domain": d}).json() for d in domains]
    client.post("/predict", json={"domain": "-bad.com"})  # rejected: not a prediction
    client.post("/explain", json={"domain": "github.com"})  # /explain is not counted

    predictions = client.get("/stats").json()["predictions"]
    labels = [r["label"] for r in results]
    assert predictions["count"] == len(results)
    assert predictions["by_label"] == {"benign": labels.count("benign"), "dga": labels.count("dga")}
    expected_mean = sum(r["probability"] for r in results) / len(results)
    assert predictions["mean_probability_dga"] == pytest.approx(expected_mean, abs=1e-4)


def test_unknown_paths_share_one_key(client: TestClient) -> None:
    client.get("/wp-admin")
    client.get("/.env")
    client.get("/docs")

    by_endpoint = client.get("/stats").json()["requests"]["by_endpoint"]
    assert by_endpoint["unmatched"]["count"] == 2
    assert by_endpoint["unmatched"]["by_status_class"] == {"4xx": 2}
    assert by_endpoint["GET /docs"]["count"] == 1
    assert not any("wp-admin" in key or ".env" in key for key in by_endpoint)


def test_counters_reset_when_the_app_restarts(model_pointer: Path) -> None:
    with TestClient(app) as first:
        first.post("/predict", json={"domain": "google.com"})
        assert first.get("/stats").json()["predictions"]["count"] == 1
    with TestClient(app) as second:
        body = second.get("/stats").json()
    assert body["requests"]["total"] == 0
    assert body["predictions"]["count"] == 0


def test_stats_without_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(POINTER_ENV_VAR, str(tmp_path / "missing.json"))
    with TestClient(app) as no_model:
        no_model.post("/predict", json={"domain": "google.com"})  # 503
        response = no_model.get("/stats")
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] is None
    assert endpoint(body, "POST /predict")["by_status_class"] == {"5xx": 1}
    assert body["predictions"]["count"] == 0


def test_openapi_has_stats_example(client: TestClient) -> None:
    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    assert schemas["StatsResponse"]["examples"]


# ---------------------------------------------------------------------------
# JSON access log
# ---------------------------------------------------------------------------


class ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def access_records() -> Iterator[list[logging.LogRecord]]:
    """Access log records. The logger does not propagate (see configure_access_logger), so the
    test attaches its own handler to it; caplog's handler is swapped per test phase."""
    handler = ListHandler()
    access_logger = logging.getLogger(ACCESS_LOGGER_NAME)
    access_logger.addHandler(handler)
    try:
        yield handler.records
    finally:
        access_logger.removeHandler(handler)


def access_lines(records: list[logging.LogRecord]) -> list[dict]:
    return [json.loads(r.getMessage()) for r in records if r.name == ACCESS_LOGGER_NAME]


def test_one_json_log_line_per_request(
    client: TestClient, access_records: list[logging.LogRecord]
) -> None:
    client.get("/health")
    response = client.post("/predict", json={"domain": "moiqidfvjjhjmndqp.in"}).json()
    client.post("/predict", json={"domain": "-bad.com"})

    health, predict, rejected = access_lines(access_records)
    assert set(health) == {"timestamp", "method", "path", "status", "latency_ms"}
    assert health["method"] == "GET"
    assert health["path"] == "/health"
    assert health["status"] == 200
    assert health["latency_ms"] > 0
    assert predict["label"] == response["label"]
    assert predict["probability"] == response["probability"]
    assert rejected["status"] == 422
    assert "label" not in rejected


def test_log_never_contains_the_request_body(
    client: TestClient, access_records: list[logging.LogRecord]
) -> None:
    client.post("/predict", json={"domain": "moiqidfvjjhjmndqp.in"})
    client.post("/explain", json={"domain": "wedding-muscle.com"})
    client.post("/predict", json={"domain": "-bad.com"})

    text = "\n".join(r.getMessage() for r in access_records if r.name == ACCESS_LOGGER_NAME)
    assert text.count("\n") == 2  # three lines
    for secret in ("moiqidfvjjhjmndqp", "wedding-muscle", "-bad.com", "domain"):
        assert secret not in text


def test_access_logger_setup_is_idempotent() -> None:
    first = configure_access_logger()
    second = configure_access_logger()
    assert first is second
    assert first.propagate is False
    assert [h.name for h in first.handlers].count(ACCESS_HANDLER_NAME) == 1

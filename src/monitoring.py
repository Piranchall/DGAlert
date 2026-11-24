"""In-memory request and prediction statistics for GET /stats, and the JSON access log.

Counters live in process memory: they start at zero whenever the service starts, so they reset on
every restart (including when Render puts an idle free-tier service to sleep).

FastAPI runs sync endpoints in a thread pool, so StatsCollector guards its counters with a lock.
"""

import json
import logging
import sys
import threading
import time
from collections import Counter
from datetime import UTC, datetime
from typing import Any

ACCESS_LOGGER_NAME = "dgalert.access"
ACCESS_HANDLER_NAME = "dgalert-access-stdout"

# Key for requests that matched no route (e.g. 404 scans), so the endpoint keys stay bounded.
UNMATCHED_ENDPOINT = "unmatched"
LABELS: tuple[str, ...] = ("benign", "dga")


def status_class(status_code: int) -> str:
    """200 -> "2xx", 404 -> "4xx", ..."""
    return f"{status_code // 100}xx"


def average(total: float, count: int, digits: int) -> float | None:
    return round(total / count, digits) if count else None


class _EndpointStats:
    """Counters for one endpoint (not thread-safe on its own; StatsCollector holds the lock)."""

    def __init__(self) -> None:
        self.count = 0
        self.latency_ms_total = 0.0
        self.by_status_class: Counter[str] = Counter()

    def add(self, status_code: int, latency_ms: float) -> None:
        self.count += 1
        self.latency_ms_total += latency_ms
        self.by_status_class[status_class(status_code)] += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "by_status_class": dict(sorted(self.by_status_class.items())),
            "avg_latency_ms": average(self.latency_ms_total, self.count, 3),
        }


class StatsCollector:
    """Thread-safe request counts and latency (overall and per endpoint), and /predict labels."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.started_at = datetime.now(UTC)
        self._started_monotonic = time.monotonic()
        self._overall = _EndpointStats()
        self._endpoints: dict[str, _EndpointStats] = {}
        self._label_counts: Counter[str] = Counter({label: 0 for label in LABELS})
        self._probability_total = 0.0

    def record_request(self, endpoint: str, status_code: int, latency_ms: float) -> None:
        with self._lock:
            self._overall.add(status_code, latency_ms)
            self._endpoints.setdefault(endpoint, _EndpointStats()).add(status_code, latency_ms)

    def record_prediction(self, label: str, probability: float) -> None:
        with self._lock:
            self._label_counts[label] += 1
            self._probability_total += probability

    def snapshot(self) -> dict[str, Any]:
        """All counters as plain JSON-ready values (model_version is added by the API)."""
        with self._lock:
            overall = self._overall.as_dict()
            predictions = sum(self._label_counts.values())
            return {
                "started_at": self.started_at.isoformat(timespec="milliseconds"),
                "uptime_seconds": round(time.monotonic() - self._started_monotonic, 3),
                "requests": {
                    "total": overall["count"],
                    "by_status_class": overall["by_status_class"],
                    "avg_latency_ms": overall["avg_latency_ms"],
                    "by_endpoint": {
                        key: stats.as_dict() for key, stats in sorted(self._endpoints.items())
                    },
                },
                "predictions": {
                    "count": predictions,
                    "by_label": dict(self._label_counts),
                    "mean_probability_dga": average(self._probability_total, predictions, 4),
                },
            }


def configure_access_logger() -> logging.Logger:
    """The dgalert.access logger, writing bare JSON lines to stdout at INFO.

    Idempotent: the stdout handler is found by name, so calling this again (e.g. importing or
    starting the app several times in tests) never adds a second handler or duplicate lines.

    propagate is False on purpose: uvicorn's default logging config does not handle our logger,
    and a root handler added elsewhere (basicConfig, a custom --log-config) would otherwise print
    every line twice. Tests capture lines by attaching a handler to this logger directly.
    """
    access_logger = logging.getLogger(ACCESS_LOGGER_NAME)
    access_logger.setLevel(logging.INFO)
    access_logger.propagate = False
    if not any(handler.name == ACCESS_HANDLER_NAME for handler in access_logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.name = ACCESS_HANDLER_NAME
        handler.setFormatter(logging.Formatter("%(message)s"))
        access_logger.addHandler(handler)
    return access_logger


def access_log_line(
    method: str,
    path: str,
    status_code: int,
    latency_ms: float,
    prediction: dict[str, Any] | None = None,
) -> str:
    """One JSON log line. Never includes the request body: for /predict only label and P(dga)."""
    record: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
        "method": method,
        "path": path,
        "status": status_code,
        "latency_ms": round(latency_ms, 3),
    }
    if prediction is not None:
        record["label"] = prediction["label"]
        record["probability"] = prediction["probability"]
    return json.dumps(record)

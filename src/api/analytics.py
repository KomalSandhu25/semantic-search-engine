"""In-memory query analytics for the semantic search API.

Tracks every search request (query text, timestamp, result count, latency)
and exposes aggregated statistics — top queries by frequency and latency
percentiles — via a thread-safe in-memory store.

Typical usage::

    from src.api.analytics import QueryAnalytics

    analytics = QueryAnalytics()
    analytics.log(query="cats", num_results=5, latency_ms=23.4)
    report = analytics.report(top_n=10)
"""

from __future__ import annotations

import math
import statistics
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

from src.api.schemas import AnalyticsResponse, LatencyPercentiles, QueryRecord, TopQuery


@dataclass
class QueryAnalytics:
    """Thread-safe, in-memory store for query-level telemetry.

    All public methods are safe to call from concurrent request handlers.

    Args:
        max_records: Maximum number of query records to retain.  Oldest
            entries are dropped once the limit is reached (FIFO eviction).
            Defaults to 10 000.

    Example::

        store = QueryAnalytics(max_records=1000)
        store.log(query="neural IR", num_results=10, latency_ms=42.0)
        report = store.report(top_n=5)
        assert report.total_queries == 1
    """

    max_records: int = 10_000
    _records: list[QueryRecord] = field(default_factory=list, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def log(self, *, query: str, num_results: int, latency_ms: float) -> None:
        """Record a single query event.

        Args:
            query: Raw query string submitted by the user.
            num_results: Number of results returned for this query.
            latency_ms: End-to-end request latency in milliseconds.
        """
        record = QueryRecord(
            query=query.strip(),
            timestamp=time.time(),
            num_results=num_results,
            latency_ms=latency_ms,
        )
        with self._lock:
            self._records.append(record)
            if len(self._records) > self.max_records:
                overflow = len(self._records) - self.max_records
                self._records = self._records[overflow:]

    def clear(self) -> None:
        """Remove all stored records (useful in tests)."""
        with self._lock:
            self._records.clear()

    def report(self, *, top_n: int = 10) -> AnalyticsResponse:
        """Build an aggregated analytics report from current records.

        Args:
            top_n: How many top queries (by frequency) to include.

        Returns:
            :class:`~src.api.schemas.AnalyticsResponse` with total query
            count, top queries, and latency percentiles.

        Raises:
            ValueError: If *top_n* is less than 1.
        """
        if top_n < 1:
            raise ValueError(f"top_n must be >= 1, got {top_n}")

        with self._lock:
            snapshot = list(self._records)

        if not snapshot:
            return AnalyticsResponse(
                total_queries=0,
                top_queries=[],
                latency_percentiles=LatencyPercentiles(
                    p50_ms=0.0, p90_ms=0.0, p99_ms=0.0, mean_ms=0.0
                ),
            )

        counter: Counter[str] = Counter(r.query for r in snapshot)
        top_queries = [
            TopQuery(query=q, count=c) for q, c in counter.most_common(top_n)
        ]

        latencies = sorted(r.latency_ms for r in snapshot)
        percentiles = LatencyPercentiles(
            p50_ms=_percentile(latencies, 50),
            p90_ms=_percentile(latencies, 90),
            p99_ms=_percentile(latencies, 99),
            mean_ms=statistics.mean(latencies),
        )

        return AnalyticsResponse(
            total_queries=len(snapshot),
            top_queries=top_queries,
            latency_percentiles=percentiles,
        )

    @property
    def record_count(self) -> int:
        """Current number of stored records."""
        with self._lock:
            return len(self._records)


def _percentile(sorted_values: Sequence[float], pct: int) -> float:
    """Return the *pct*-th percentile of a pre-sorted sequence (nearest-rank).

    Args:
        sorted_values: A **sorted** (ascending) sequence of floats.
        pct: Percentile to compute, in the range [0, 100].

    Returns:
        The percentile value as a float.
    """
    if not sorted_values:
        return 0.0
    idx = max(0, math.ceil(len(sorted_values) * pct / 100) - 1)
    return sorted_values[idx]

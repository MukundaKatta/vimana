"""Detect infrastructure anomalies using statistical methods.

Z-score based detection with sliding windows, plus pattern matching
for known failure signatures.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from vimana.models import Anomaly, ResourceMetrics, SeverityLevel


# ---------------------------------------------------------------------------
# Known anomaly patterns
# ---------------------------------------------------------------------------

_FAILURE_PATTERNS: list[dict] = [
    {
        "name": "memory_leak",
        "check": lambda m: m.memory_percent > 90 and m.cpu_percent < 50,
        "description": "High memory with low CPU suggests a memory leak.",
        "severity": SeverityLevel.HIGH,
    },
    {
        "name": "network_partition",
        "check": lambda m: m.network_in_mbps == 0 and m.network_out_mbps == 0,
        "description": "Zero network throughput indicates a network partition.",
        "severity": SeverityLevel.CRITICAL,
    },
    {
        "name": "disk_exhaustion",
        "check": lambda m: m.disk_percent > 95,
        "description": "Disk nearly full; writes will fail imminently.",
        "severity": SeverityLevel.CRITICAL,
    },
    {
        "name": "error_storm",
        "check": lambda m: m.error_rate > 0.1,
        "description": "Error rate exceeds 10%; service degradation likely.",
        "severity": SeverityLevel.HIGH,
    },
    {
        "name": "latency_spike",
        "check": lambda m: m.latency_ms > 3000,
        "description": "Extreme latency (>3s) indicates severe contention or failure.",
        "severity": SeverityLevel.HIGH,
    },
]


# ---------------------------------------------------------------------------
# AnomalyDetector
# ---------------------------------------------------------------------------

@dataclass
class AnomalyDetector:
    """Detects infrastructure anomalies via z-score analysis and pattern matching.

    Maintains a sliding window of metric history per resource and flags
    values that deviate significantly from the running statistics.
    """

    z_threshold: float = 3.0
    window_size: int = 50
    _history: dict[str, list[dict[str, float]]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def observe(self, resource_id: str, metrics: ResourceMetrics) -> None:
        """Record a metrics observation for a resource."""
        entry = {
            "cpu": metrics.cpu_percent,
            "memory": metrics.memory_percent,
            "disk": metrics.disk_percent,
            "latency": metrics.latency_ms,
            "error_rate": metrics.error_rate,
            "net_in": metrics.network_in_mbps,
            "net_out": metrics.network_out_mbps,
        }
        hist = self._history[resource_id]
        hist.append(entry)
        if len(hist) > self.window_size:
            hist.pop(0)

    def detect(self, resource_id: str, metrics: ResourceMetrics) -> list[Anomaly]:
        """Detect anomalies for a single resource given its latest metrics.

        Combines z-score detection on every tracked metric with pattern
        matching against known failure signatures.
        """
        self.observe(resource_id, metrics)
        anomalies: list[Anomaly] = []

        # --- Z-score detection ---
        hist = self._history[resource_id]
        if len(hist) >= 5:
            anomalies.extend(self._zscore_detect(resource_id, hist))

        # --- Pattern matching ---
        anomalies.extend(self._pattern_detect(resource_id, metrics))

        return anomalies

    def detect_all(
        self,
        metrics_stream: dict[str, ResourceMetrics],
    ) -> list[Anomaly]:
        """Run detection across multiple resources."""
        all_anomalies: list[Anomaly] = []
        for rid, metrics in metrics_stream.items():
            all_anomalies.extend(self.detect(rid, metrics))
        return all_anomalies

    def reset(self, resource_id: str | None = None) -> None:
        if resource_id:
            self._history.pop(resource_id, None)
        else:
            self._history.clear()

    # -- internals ----------------------------------------------------------

    def _zscore_detect(
        self,
        resource_id: str,
        hist: list[dict[str, float]],
    ) -> list[Anomaly]:
        anomalies: list[Anomaly] = []
        latest = hist[-1]
        metric_names = list(latest.keys())

        for name in metric_names:
            values = np.array([h[name] for h in hist])
            mean = float(np.mean(values))
            std = float(np.std(values))
            if std < 1e-6:
                continue

            z = abs(latest[name] - mean) / std
            if z >= self.z_threshold:
                severity = self._z_to_severity(z)
                anomalies.append(
                    Anomaly(
                        resource_id=resource_id,
                        metric_name=name,
                        value=latest[name],
                        expected_value=round(mean, 2),
                        z_score=round(z, 2),
                        severity=severity,
                        description=f"{name} = {latest[name]:.2f} deviates {z:.1f} "
                                    f"sigma from mean {mean:.2f}",
                    )
                )
        return anomalies

    def _pattern_detect(
        self,
        resource_id: str,
        metrics: ResourceMetrics,
    ) -> list[Anomaly]:
        anomalies: list[Anomaly] = []
        for pattern in _FAILURE_PATTERNS:
            try:
                if pattern["check"](metrics):
                    anomalies.append(
                        Anomaly(
                            resource_id=resource_id,
                            metric_name=pattern["name"],
                            value=0.0,
                            expected_value=0.0,
                            z_score=0.0,
                            severity=pattern["severity"],
                            description=pattern["description"],
                        )
                    )
            except Exception:
                continue
        return anomalies

    @staticmethod
    def _z_to_severity(z: float) -> SeverityLevel:
        if z >= 5.0:
            return SeverityLevel.CRITICAL
        elif z >= 4.0:
            return SeverityLevel.HIGH
        elif z >= 3.0:
            return SeverityLevel.MEDIUM
        return SeverityLevel.LOW

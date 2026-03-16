"""Predict compute demand using time-series analysis.

Implements exponential smoothing, trend detection, and simple
forecasting to anticipate future resource needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vimana.models import DemandForecast


@dataclass
class DemandPredictor:
    """Predicts future compute demand from historical metrics.

    Uses double exponential smoothing (Holt's method) for trend-aware
    forecasting, with z-score-based trend classification.
    """

    alpha: float = 0.3       # level smoothing
    beta: float = 0.1        # trend smoothing
    min_history: int = 10    # minimum data points before forecasting
    _cpu_history: list[float] = field(default_factory=list)
    _memory_history: list[float] = field(default_factory=list)
    _request_history: list[float] = field(default_factory=list)

    def observe(self, cpu: float, memory: float, requests: float) -> None:
        """Record a new observation."""
        self._cpu_history.append(cpu)
        self._memory_history.append(memory)
        self._request_history.append(requests)

    def predict(self, horizon: int = 10) -> DemandForecast:
        """Forecast demand for the next *horizon* ticks.

        Returns a DemandForecast with predicted CPU, memory, request rates,
        confidence bands, and a trend label.
        """
        if len(self._cpu_history) < self.min_history:
            # Not enough data -- return a flat forecast at the latest values
            last_cpu = self._cpu_history[-1] if self._cpu_history else 50.0
            last_mem = self._memory_history[-1] if self._memory_history else 50.0
            last_req = self._request_history[-1] if self._request_history else 100.0
            return DemandForecast(
                timestamps=list(range(1, horizon + 1)),
                predicted_cpu=[last_cpu] * horizon,
                predicted_memory=[last_mem] * horizon,
                predicted_requests=[last_req] * horizon,
                confidence_lower=[last_cpu * 0.7] * horizon,
                confidence_upper=[last_cpu * 1.3] * horizon,
                trend="stable",
            )

        cpu_forecast = self._holt_forecast(self._cpu_history, horizon)
        mem_forecast = self._holt_forecast(self._memory_history, horizon)
        req_forecast = self._holt_forecast(self._request_history, horizon)

        # Confidence bands based on historical variance
        cpu_arr = np.array(self._cpu_history[-30:])
        std = float(np.std(cpu_arr)) if len(cpu_arr) > 1 else 5.0
        lower = [max(0.0, v - 1.96 * std) for v in cpu_forecast]
        upper = [min(100.0, v + 1.96 * std) for v in cpu_forecast]

        trend = self._detect_trend(self._cpu_history)

        return DemandForecast(
            timestamps=list(range(1, horizon + 1)),
            predicted_cpu=cpu_forecast,
            predicted_memory=mem_forecast,
            predicted_requests=req_forecast,
            confidence_lower=lower,
            confidence_upper=upper,
            trend=trend,
        )

    def reset(self) -> None:
        self._cpu_history.clear()
        self._memory_history.clear()
        self._request_history.clear()

    # -- internals ----------------------------------------------------------

    def _holt_forecast(self, series: list[float], horizon: int) -> list[float]:
        """Double exponential smoothing (Holt's linear trend method)."""
        if len(series) < 2:
            return [series[-1] if series else 0.0] * horizon

        # Initialize
        level = series[0]
        trend = series[1] - series[0]

        for val in series[1:]:
            prev_level = level
            level = self.alpha * val + (1 - self.alpha) * (level + trend)
            trend = self.beta * (level - prev_level) + (1 - self.beta) * trend

        # Forecast
        forecasts: list[float] = []
        for h in range(1, horizon + 1):
            forecasts.append(round(level + h * trend, 2))
        return forecasts

    @staticmethod
    def _detect_trend(series: list[float], window: int = 20) -> str:
        """Classify the recent trend of a time series."""
        if len(series) < 5:
            return "stable"

        recent = np.array(series[-window:])
        # Linear regression slope
        x = np.arange(len(recent), dtype=float)
        slope = float(np.polyfit(x, recent, 1)[0])

        # Check for cyclicality using autocorrelation
        if len(recent) >= 10:
            centered = recent - np.mean(recent)
            autocorr = np.correlate(centered, centered, mode="full")
            autocorr = autocorr[len(autocorr) // 2:]
            if len(autocorr) > 3:
                normalized = autocorr / (autocorr[0] + 1e-10)
                # If there is a secondary peak, it is cyclic
                if len(normalized) > 5 and any(normalized[3:] > 0.5):
                    return "cyclic"

        if slope > 1.0:
            return "rising"
        elif slope < -1.0:
            return "falling"
        return "stable"

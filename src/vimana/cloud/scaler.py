"""Auto-scaling logic based on demand signals.

Supports reactive (threshold), predictive (ML-based), and cost-aware
scaling strategies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vimana.models import (
    ActionResult,
    DemandForecast,
    InfrastructureState,
    Resource,
    ResourceMetrics,
    ResourceStatus,
    ScaleDecision,
    ScaleDirection,
    ScaleResult,
    ScalingStrategy,
)
from vimana.cloud.provisioner import CloudProvisioner


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

@dataclass
class ScalingThresholds:
    cpu_scale_up: float = 75.0
    cpu_scale_down: float = 25.0
    memory_scale_up: float = 80.0
    memory_scale_down: float = 30.0
    latency_scale_up_ms: float = 200.0
    error_rate_scale_up: float = 0.05
    min_replicas: int = 1
    max_replicas: int = 20
    cooldown_ticks: int = 5


# ---------------------------------------------------------------------------
# AutoScaler
# ---------------------------------------------------------------------------

@dataclass
class AutoScaler:
    """Evaluates infrastructure metrics and decides how to scale.

    Three strategies:
      - reactive:   compare current metrics against thresholds
      - predictive: use a DemandForecast to scale pre-emptively
      - cost_aware: minimize cost while meeting SLA constraints
    """

    provisioner: CloudProvisioner
    strategy: ScalingStrategy = ScalingStrategy.REACTIVE
    thresholds: ScalingThresholds = field(default_factory=ScalingThresholds)
    _last_scale_tick: int = 0
    _tick: int = 0

    # -- evaluation ---------------------------------------------------------

    def evaluate(
        self,
        metrics: dict[str, float],
        state: InfrastructureState | None = None,
        forecast: DemandForecast | None = None,
    ) -> ScaleDecision:
        """Evaluate current conditions and return a scaling decision."""
        self._tick += 1

        # Cooldown check
        if (self._tick - self._last_scale_tick) < self.thresholds.cooldown_ticks:
            return ScaleDecision(
                direction=ScaleDirection.MAINTAIN,
                reason="In cooldown period.",
                confidence=1.0,
            )

        if self.strategy == ScalingStrategy.REACTIVE:
            return self._evaluate_reactive(metrics)
        elif self.strategy == ScalingStrategy.PREDICTIVE:
            return self._evaluate_predictive(metrics, forecast)
        elif self.strategy == ScalingStrategy.COST_AWARE:
            return self._evaluate_cost_aware(metrics, state)
        else:
            return ScaleDecision(direction=ScaleDirection.MAINTAIN, reason="Unknown strategy.")

    def apply(self, decision: ScaleDecision, resource_id: str) -> ScaleResult:
        """Apply a scaling decision to a specific resource."""
        resource = self.provisioner.get_resource(resource_id)
        if resource is None:
            return ScaleResult(success=False, message=f"Resource {resource_id} not found.")

        prev_replicas = resource.spec.replicas

        if decision.direction == ScaleDirection.MAINTAIN:
            return ScaleResult(
                success=True,
                previous_replicas=prev_replicas,
                new_replicas=prev_replicas,
                message="No scaling needed.",
            )

        target = decision.target_replicas
        if target is None:
            if decision.direction == ScaleDirection.UP:
                target = min(prev_replicas + 1, self.thresholds.max_replicas)
            else:
                target = max(prev_replicas - 1, self.thresholds.min_replicas)

        target = max(self.thresholds.min_replicas, min(target, self.thresholds.max_replicas))

        result = self.provisioner.scale_replicas(resource_id, target)
        self._last_scale_tick = self._tick

        return ScaleResult(
            success=result.success,
            previous_replicas=prev_replicas,
            new_replicas=target,
            cost_delta_per_hour=result.actual_cost_delta,
            message=f"Scaled from {prev_replicas} to {target} replicas. "
                    f"Reason: {decision.reason}",
        )

    # -- strategies ---------------------------------------------------------

    def _evaluate_reactive(self, metrics: dict[str, float]) -> ScaleDecision:
        """Threshold-based reactive scaling."""
        cpu = metrics.get("avg_cpu", 0)
        mem = metrics.get("avg_memory", 0)
        latency = metrics.get("avg_latency_ms", 0)
        error_rate = metrics.get("avg_error_rate", 0)

        # Scale up conditions
        reasons_up: list[str] = []
        if cpu > self.thresholds.cpu_scale_up:
            reasons_up.append(f"CPU {cpu:.1f}% > {self.thresholds.cpu_scale_up}%")
        if mem > self.thresholds.memory_scale_up:
            reasons_up.append(f"Memory {mem:.1f}% > {self.thresholds.memory_scale_up}%")
        if latency > self.thresholds.latency_scale_up_ms:
            reasons_up.append(f"Latency {latency:.0f}ms > {self.thresholds.latency_scale_up_ms}ms")
        if error_rate > self.thresholds.error_rate_scale_up:
            reasons_up.append(f"Error rate {error_rate:.4f} > {self.thresholds.error_rate_scale_up}")

        if reasons_up:
            return ScaleDecision(
                direction=ScaleDirection.UP,
                reason="Reactive scale-up: " + "; ".join(reasons_up),
                confidence=min(1.0, len(reasons_up) * 0.35),
            )

        # Scale down conditions
        reasons_down: list[str] = []
        if cpu < self.thresholds.cpu_scale_down:
            reasons_down.append(f"CPU {cpu:.1f}% < {self.thresholds.cpu_scale_down}%")
        if mem < self.thresholds.memory_scale_down:
            reasons_down.append(f"Memory {mem:.1f}% < {self.thresholds.memory_scale_down}%")

        if len(reasons_down) >= 2:
            return ScaleDecision(
                direction=ScaleDirection.DOWN,
                reason="Reactive scale-down: " + "; ".join(reasons_down),
                confidence=min(1.0, len(reasons_down) * 0.3),
            )

        return ScaleDecision(
            direction=ScaleDirection.MAINTAIN,
            reason="All metrics within normal range.",
            confidence=0.8,
        )

    def _evaluate_predictive(
        self,
        metrics: dict[str, float],
        forecast: DemandForecast | None,
    ) -> ScaleDecision:
        """Predictive scaling using demand forecast."""
        if forecast is None or not forecast.predicted_cpu:
            return self._evaluate_reactive(metrics)

        # Look at the next few predicted values
        lookahead = forecast.predicted_cpu[:5]
        peak_predicted = max(lookahead) if lookahead else 0

        current_cpu = metrics.get("avg_cpu", 0)

        if peak_predicted > self.thresholds.cpu_scale_up:
            headroom = peak_predicted - current_cpu
            extra_replicas = max(1, int(headroom / 20))
            return ScaleDecision(
                direction=ScaleDirection.UP,
                target_replicas=None,  # let apply() decide the increment
                reason=f"Predictive: expected CPU peak {peak_predicted:.1f}% "
                       f"in next 5 ticks (trend: {forecast.trend}).",
                confidence=0.7,
            )

        if peak_predicted < self.thresholds.cpu_scale_down and forecast.trend == "falling":
            return ScaleDecision(
                direction=ScaleDirection.DOWN,
                reason=f"Predictive: demand falling, peak forecast {peak_predicted:.1f}%.",
                confidence=0.6,
            )

        return ScaleDecision(
            direction=ScaleDirection.MAINTAIN,
            reason=f"Predictive: forecast stable (peak {peak_predicted:.1f}%).",
            confidence=0.7,
        )

    def _evaluate_cost_aware(
        self,
        metrics: dict[str, float],
        state: InfrastructureState | None,
    ) -> ScaleDecision:
        """Cost-aware scaling: only scale up if SLA is at risk, aggressively scale down."""
        cpu = metrics.get("avg_cpu", 0)
        error_rate = metrics.get("avg_error_rate", 0)
        latency = metrics.get("avg_latency_ms", 0)
        cost = metrics.get("total_cost_per_hour", 0)

        # Only scale up when things are truly stressed
        if error_rate > self.thresholds.error_rate_scale_up or latency > self.thresholds.latency_scale_up_ms * 1.5:
            return ScaleDecision(
                direction=ScaleDirection.UP,
                reason=f"Cost-aware: SLA at risk (errors={error_rate:.4f}, latency={latency:.0f}ms).",
                confidence=0.8,
            )

        # Aggressively scale down when utilization is low
        if cpu < self.thresholds.cpu_scale_down * 0.8:
            return ScaleDecision(
                direction=ScaleDirection.DOWN,
                reason=f"Cost-aware: low utilization {cpu:.1f}%, saving cost (${cost:.4f}/hr).",
                confidence=0.75,
            )

        return ScaleDecision(
            direction=ScaleDirection.MAINTAIN,
            reason=f"Cost-aware: acceptable balance (CPU {cpu:.1f}%, ${cost:.4f}/hr).",
            confidence=0.7,
        )

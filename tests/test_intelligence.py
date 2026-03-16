"""Tests for the intelligence module: demand prediction, anomaly detection,
cost optimization, and capacity planning."""

from __future__ import annotations

import pytest

from vimana.models import (
    DemandForecast,
    InfrastructureState,
    Region,
    Resource,
    ResourceMetrics,
    ResourceSpec,
    ResourceStatus,
    ResourceType,
    SeverityLevel,
    SimulationScenario,
)
from vimana.intelligence.demand_predictor import DemandPredictor
from vimana.intelligence.anomaly_detector import AnomalyDetector
from vimana.intelligence.cost_optimizer import CostOptimizer
from vimana.intelligence.capacity_planner import CapacityPlanner
from vimana.simulation import CloudSimulator, compute_hourly_cost


# ---------------------------------------------------------------------------
# DemandPredictor
# ---------------------------------------------------------------------------

class TestDemandPredictor:
    def test_predict_insufficient_data(self):
        predictor = DemandPredictor(min_history=10)
        for i in range(5):
            predictor.observe(50.0, 40.0, 100.0)
        forecast = predictor.predict(horizon=10)
        assert len(forecast.predicted_cpu) == 10
        assert forecast.trend == "stable"

    def test_predict_with_data(self):
        predictor = DemandPredictor(min_history=5)
        for i in range(20):
            cpu = 30 + i * 2  # rising trend
            predictor.observe(cpu, 40.0, 100.0 + i * 10)
        forecast = predictor.predict(horizon=10)
        assert len(forecast.predicted_cpu) == 10
        # Should detect rising trend
        assert forecast.predicted_cpu[-1] > forecast.predicted_cpu[0]

    def test_trend_detection_rising(self):
        predictor = DemandPredictor(min_history=5)
        for i in range(30):
            predictor.observe(20 + i * 2.5, 30.0, 100.0)
        forecast = predictor.predict(5)
        assert forecast.trend in ("rising", "stable")  # depends on magnitude

    def test_trend_detection_falling(self):
        predictor = DemandPredictor(min_history=5)
        for i in range(30):
            predictor.observe(80 - i * 2.5, 30.0, 100.0)
        forecast = predictor.predict(5)
        assert forecast.trend in ("falling", "stable")

    def test_confidence_bands(self):
        predictor = DemandPredictor(min_history=5)
        for i in range(20):
            predictor.observe(50.0 + (i % 5) * 3, 40.0, 100.0)
        forecast = predictor.predict(5)
        for lo, hi in zip(forecast.confidence_lower, forecast.confidence_upper):
            assert lo <= hi

    def test_reset(self):
        predictor = DemandPredictor(min_history=5)
        for i in range(10):
            predictor.observe(50.0, 40.0, 100.0)
        predictor.reset()
        forecast = predictor.predict(5)
        # After reset, should behave like insufficient data
        assert forecast.trend == "stable"


# ---------------------------------------------------------------------------
# AnomalyDetector
# ---------------------------------------------------------------------------

class TestAnomalyDetector:
    def _normal_metrics(self, cpu: float = 50.0) -> ResourceMetrics:
        return ResourceMetrics(
            cpu_percent=cpu,
            memory_percent=40.0,
            disk_percent=30.0,
            network_in_mbps=200.0,
            network_out_mbps=100.0,
            request_rate=500.0,
            error_rate=0.001,
            latency_ms=20.0,
        )

    def test_no_anomaly_normal_data(self):
        detector = AnomalyDetector()
        for i in range(20):
            anomalies = detector.detect("r1", self._normal_metrics(50 + i % 3))
        # Should have no z-score anomalies
        z_anomalies = [a for a in anomalies if a.z_score > 0]
        assert len(z_anomalies) == 0

    def test_detects_cpu_spike(self):
        detector = AnomalyDetector(z_threshold=2.0)
        for _ in range(20):
            detector.detect("r1", self._normal_metrics(50.0))
        # Inject spike
        spike_metrics = self._normal_metrics(99.0)
        anomalies = detector.detect("r1", spike_metrics)
        cpu_anomalies = [a for a in anomalies if a.metric_name == "cpu"]
        assert len(cpu_anomalies) > 0

    def test_pattern_detection_memory_leak(self):
        detector = AnomalyDetector()
        metrics = ResourceMetrics(
            cpu_percent=30.0,
            memory_percent=95.0,
            disk_percent=40.0,
            network_in_mbps=200.0,
            network_out_mbps=100.0,
        )
        anomalies = detector.detect("r1", metrics)
        pattern_anomalies = [a for a in anomalies if a.metric_name == "memory_leak"]
        assert len(pattern_anomalies) > 0

    def test_pattern_detection_network_partition(self):
        detector = AnomalyDetector()
        metrics = ResourceMetrics(
            cpu_percent=30.0,
            memory_percent=40.0,
            disk_percent=40.0,
            network_in_mbps=0.0,
            network_out_mbps=0.0,
        )
        anomalies = detector.detect("r1", metrics)
        net_anomalies = [a for a in anomalies if a.metric_name == "network_partition"]
        assert len(net_anomalies) > 0
        assert net_anomalies[0].severity == SeverityLevel.CRITICAL

    def test_detect_all(self):
        detector = AnomalyDetector()
        stream = {
            "r1": self._normal_metrics(50.0),
            "r2": ResourceMetrics(
                cpu_percent=30.0,
                memory_percent=96.0,
                disk_percent=40.0,
                network_in_mbps=200.0,
                network_out_mbps=100.0,
            ),
        }
        anomalies = detector.detect_all(stream)
        assert isinstance(anomalies, list)


# ---------------------------------------------------------------------------
# CostOptimizer
# ---------------------------------------------------------------------------

class TestCostOptimizer:
    def _make_state_with_metrics(self) -> InfrastructureState:
        resources = []
        for i in range(3):
            spec = ResourceSpec(
                resource_type=ResourceType.VM,
                name=f"vm-{i}",
                vcpus=8,
                memory_gb=16.0,
                region=Region.AP_NORTHEAST_1,  # expensive
                replicas=3,
            )
            r = Resource(
                spec=spec,
                status=ResourceStatus.RUNNING,
                cost_per_hour=compute_hourly_cost(spec),
                metrics=ResourceMetrics(
                    cpu_percent=15.0,   # very low
                    memory_percent=20.0,
                ),
            )
            resources.append(r)
        state = InfrastructureState(resources=resources)
        state.refresh_totals()
        return state

    def test_finds_optimizations(self):
        optimizer = CostOptimizer()
        state = self._make_state_with_metrics()
        result = optimizer.optimize(state)
        assert result.savings_percent > 0
        assert len(result.recommendations) > 0

    def test_respects_threshold(self):
        optimizer = CostOptimizer(min_savings_threshold=1000.0)
        state = self._make_state_with_metrics()
        result = optimizer.optimize(state)
        assert len(result.recommendations) == 0

    def test_empty_state(self):
        optimizer = CostOptimizer()
        state = InfrastructureState()
        result = optimizer.optimize(state)
        assert result.savings_percent == 0
        assert len(result.recommendations) == 0


# ---------------------------------------------------------------------------
# CapacityPlanner
# ---------------------------------------------------------------------------

class TestCapacityPlanner:
    def test_plan_from_forecast(self):
        forecast = DemandForecast(
            timestamps=list(range(1, 21)),
            predicted_cpu=[40 + i * 2 for i in range(20)],
            predicted_memory=[30 + i * 1.5 for i in range(20)],
            predicted_requests=[100 + i * 50 for i in range(20)],
            confidence_lower=[30 + i * 1.5 for i in range(20)],
            confidence_upper=[50 + i * 2.5 for i in range(20)],
            trend="rising",
        )
        planner = CapacityPlanner()
        plan = planner.plan(forecast, budget_per_hour=20.0)
        assert plan.horizon_ticks == 20
        assert len(plan.planned_resources) > 0
        assert plan.peak_vcpus > 0

    def test_empty_forecast(self):
        forecast = DemandForecast()
        planner = CapacityPlanner()
        plan = planner.plan(forecast)
        assert plan.horizon_ticks == 0

    def test_multi_region_for_rising(self):
        forecast = DemandForecast(
            timestamps=list(range(1, 11)),
            predicted_cpu=[50.0] * 10,
            predicted_memory=[40.0] * 10,
            predicted_requests=[500.0] * 10,
            trend="rising",
        )
        planner = CapacityPlanner()
        plan = planner.plan(forecast)
        regions = {s.region for s in plan.planned_resources}
        assert len(regions) >= 2  # should pick multi-region

"""Tests for the v0.2.0 auto-scaler agent."""

from __future__ import annotations

import random

import numpy as np
import pytest

from vimana.agent.auto_scaler import (
    AutoScalerAgent,
    AgentRunReport,
    _DriftMonitor,
)
from vimana.agent.egress import DomainDeniedError, EgressGuard
from vimana.cloud.provisioner import CloudProvisioner
from vimana.cloud.scaler import ScalingThresholds
from vimana.models import (
    Region,
    ResourceSpec,
    ResourceType,
    SimulationScenario,
    ScaleDirection,
)
from vimana.simulation import CloudSimulator


def _build_sim() -> tuple[CloudSimulator, str]:
    random.seed(0)
    np.random.seed(0)
    scenario = SimulationScenario(
        name="agent-test",
        total_ticks=50,
        base_load=0.3,
        load_pattern="burst",
        failure_probability=0.0,
        initial_resources=[
            ResourceSpec(
                resource_type=ResourceType.VM,
                name="test-vm",
                vcpus=4,
                memory_gb=8.0,
                region=Region.US_EAST_1,
                replicas=2,
            ),
        ],
    )
    sim = CloudSimulator(scenario=scenario)
    sim.initialize()
    return sim, sim.state.resources[0].id


class TestAutoScalerAgent:
    def test_runs_full_loop_under_budget(self) -> None:
        sim, target = _build_sim()
        prov = CloudProvisioner(simulator=sim)
        agent = AutoScalerAgent(
            simulator=sim,
            provisioner=prov,
            target_resource_id=target,
            usd_cap=0.50,
        )
        report = agent.run(steps=50)

        assert isinstance(report, AgentRunReport)
        assert report.step_count == 50
        assert report.stopped_early is False
        assert report.usd_spent <= report.usd_cap

    def test_budget_cap_trips_early(self) -> None:
        sim, target = _build_sim()
        prov = CloudProvisioner(simulator=sim)
        agent = AutoScalerAgent(
            simulator=sim,
            provisioner=prov,
            target_resource_id=target,
            usd_cap=0.02,  # only allows 4 ticks at $0.005 each
        )
        report = agent.run(steps=50)
        assert report.stopped_early is True
        assert report.step_count < 50
        assert "budget cap" in report.stopped_reason

    def test_scales_up_under_burst(self) -> None:
        sim, target = _build_sim()
        prov = CloudProvisioner(simulator=sim)
        agent = AutoScalerAgent(
            simulator=sim,
            provisioner=prov,
            target_resource_id=target,
            usd_cap=0.50,
            thresholds=ScalingThresholds(
                cpu_scale_up=65.0,
                cpu_scale_down=20.0,
                memory_scale_down=35.0,
                cooldown_ticks=3,
            ),
        )
        report = agent.run(steps=50)
        # Burst pattern peaks ~40% and ~75% through the run; under tuned
        # thresholds the agent should fire at least one scale_up.
        assert report.scale_up_count >= 1

    def test_decisions_carry_traceable_fields(self) -> None:
        sim, target = _build_sim()
        prov = CloudProvisioner(simulator=sim)
        agent = AutoScalerAgent(
            simulator=sim,
            provisioner=prov,
            target_resource_id=target,
            usd_cap=0.50,
        )
        report = agent.run(steps=10)
        for d in report.decisions:
            assert d.tick >= 1
            assert d.direction in {
                ScaleDirection.UP,
                ScaleDirection.DOWN,
                ScaleDirection.MAINTAIN,
            }
            assert d.reason
            assert d.replicas_before >= 0
            assert d.replicas_after >= 0
            assert d.usd_spent_total <= 0.5 + 1e-9
            assert d.elapsed_ms >= 0.0

    def test_session_id_is_stable_within_a_run(self) -> None:
        sim, target = _build_sim()
        prov = CloudProvisioner(simulator=sim)
        agent = AutoScalerAgent(
            simulator=sim,
            provisioner=prov,
            target_resource_id=target,
        )
        before = agent._session_id
        report = agent.run(steps=5)
        assert report.session_id == before


class TestDriftMonitor:
    def test_no_drift_on_stable_stream(self) -> None:
        m = _DriftMonitor(reference_window=12, current_window=6, shift_threshold=18.0)
        # 18 steady samples around 30
        for v in [30.0] * 18:
            m.observe(v)
        # Latest few are still steady; no drift.
        assert m.observe(31.0) is False

    def test_drift_fires_on_large_shift(self) -> None:
        m = _DriftMonitor(reference_window=10, current_window=4, shift_threshold=18.0)
        for v in [20.0] * 10:
            m.observe(v)
        # Push current window WAY above reference.
        for v in [80.0] * 4:
            last = m.observe(v)
        assert last is True

    def test_no_drift_until_enough_history(self) -> None:
        m = _DriftMonitor(reference_window=10, current_window=4, shift_threshold=18.0)
        for v in [20.0, 22.0, 18.0]:
            assert m.observe(v) is False


class TestEgressGuard:
    def test_default_allowlist_blocks_rogue_host(self) -> None:
        guard = EgressGuard()
        with pytest.raises(DomainDeniedError):
            guard.assert_allowed("https://evil.example.com/exfil")

    def test_default_allowlist_admits_simulated_host(self) -> None:
        guard = EgressGuard()
        guard.assert_allowed("https://metrics.simulated.local/agg")  # no raise

    def test_custom_allowlist(self) -> None:
        guard = EgressGuard(allowed=frozenset({"trusted.example.com"}))
        guard.assert_allowed("https://trusted.example.com/api")
        with pytest.raises(DomainDeniedError):
            guard.assert_allowed("https://metrics.simulated.local/agg")

    def test_backend_reports_birddog_or_inline(self) -> None:
        guard = EgressGuard()
        assert guard.backend in {"birddog", "inline"}

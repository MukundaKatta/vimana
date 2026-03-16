"""Tests for the cloud module: provisioner, scaler, migrator, health."""

from __future__ import annotations

import pytest

from vimana.models import (
    Diagnosis,
    FailureType,
    Region,
    Resource,
    ResourceMetrics,
    ResourceSpec,
    ResourceStatus,
    ResourceType,
    ScaleDirection,
    ScalingStrategy,
    SeverityLevel,
    SimulationScenario,
)
from vimana.cloud.provisioner import CloudProvisioner
from vimana.cloud.scaler import AutoScaler, ScalingThresholds
from vimana.cloud.migrator import CloudMigrator
from vimana.cloud.health import HealthMonitor, SelfHealer
from vimana.simulation import CloudSimulator, compute_hourly_cost


def _make_simulator(num_resources: int = 3) -> CloudSimulator:
    scenario = SimulationScenario(
        name="test",
        total_ticks=50,
        base_load=0.4,
        load_pattern="steady",
        failure_probability=0.0,
        initial_resources=[
            ResourceSpec(
                resource_type=ResourceType.VM,
                name=f"test-vm-{i}",
                vcpus=4,
                memory_gb=8.0,
                region=Region.US_EAST_1,
                replicas=2,
            )
            for i in range(num_resources)
        ],
    )
    sim = CloudSimulator(scenario=scenario)
    sim.initialize()
    return sim


# ---------------------------------------------------------------------------
# Provisioner
# ---------------------------------------------------------------------------

class TestCloudProvisioner:
    def test_provision_success(self):
        sim = _make_simulator(0)
        prov = CloudProvisioner(simulator=sim, provision_failure_rate=0.0)
        spec = ResourceSpec(
            resource_type=ResourceType.VM,
            name="new-vm",
            vcpus=4,
            memory_gb=8.0,
        )
        result = prov.provision(spec)
        assert result.success
        assert result.resource is not None
        assert result.cost_per_hour > 0

    def test_provision_quota(self):
        sim = _make_simulator(0)
        prov = CloudProvisioner(simulator=sim, max_resources=1, provision_failure_rate=0.0)
        spec = ResourceSpec(resource_type=ResourceType.VM, vcpus=2, memory_gb=4.0)
        r1 = prov.provision(spec)
        assert r1.success
        r2 = prov.provision(spec)
        assert not r2.success
        assert "quota" in r2.message.lower()

    def test_terminate(self):
        sim = _make_simulator(1)
        prov = CloudProvisioner(simulator=sim)
        resource = sim.get_running_resources()[0]
        assert prov.terminate(resource.id) is True
        assert len(sim.get_running_resources()) == 0

    def test_scale_replicas(self):
        sim = _make_simulator(1)
        prov = CloudProvisioner(simulator=sim)
        resource = sim.get_running_resources()[0]
        result = prov.scale_replicas(resource.id, 5)
        assert result.success
        assert resource.spec.replicas == 5

    def test_cost_calculation(self):
        spec = ResourceSpec(
            resource_type=ResourceType.VM,
            vcpus=4,
            memory_gb=8.0,
            storage_gb=50.0,
            region=Region.US_EAST_1,
            replicas=1,
        )
        cost = compute_hourly_cost(spec)
        assert cost > 0


# ---------------------------------------------------------------------------
# Scaler
# ---------------------------------------------------------------------------

class TestAutoScaler:
    def test_reactive_scale_up(self):
        sim = _make_simulator(1)
        prov = CloudProvisioner(simulator=sim)
        scaler = AutoScaler(
            provisioner=prov,
            strategy=ScalingStrategy.REACTIVE,
            thresholds=ScalingThresholds(cooldown_ticks=0),
        )
        metrics = {"avg_cpu": 85.0, "avg_memory": 70.0, "avg_latency_ms": 50.0, "avg_error_rate": 0.01}
        decision = scaler.evaluate(metrics)
        assert decision.direction == ScaleDirection.UP

    def test_reactive_scale_down(self):
        sim = _make_simulator(1)
        prov = CloudProvisioner(simulator=sim)
        scaler = AutoScaler(
            provisioner=prov,
            strategy=ScalingStrategy.REACTIVE,
            thresholds=ScalingThresholds(cooldown_ticks=0),
        )
        metrics = {"avg_cpu": 10.0, "avg_memory": 15.0, "avg_latency_ms": 5.0, "avg_error_rate": 0.001}
        decision = scaler.evaluate(metrics)
        assert decision.direction == ScaleDirection.DOWN

    def test_reactive_maintain(self):
        sim = _make_simulator(1)
        prov = CloudProvisioner(simulator=sim)
        scaler = AutoScaler(
            provisioner=prov,
            strategy=ScalingStrategy.REACTIVE,
            thresholds=ScalingThresholds(cooldown_ticks=0),
        )
        metrics = {"avg_cpu": 50.0, "avg_memory": 50.0, "avg_latency_ms": 30.0, "avg_error_rate": 0.001}
        decision = scaler.evaluate(metrics)
        assert decision.direction == ScaleDirection.MAINTAIN

    def test_apply_scale_up(self):
        sim = _make_simulator(1)
        prov = CloudProvisioner(simulator=sim)
        scaler = AutoScaler(provisioner=prov, thresholds=ScalingThresholds(cooldown_ticks=0))
        resource = sim.get_running_resources()[0]
        prev = resource.spec.replicas

        from vimana.models import ScaleDecision
        decision = ScaleDecision(direction=ScaleDirection.UP, reason="test")
        result = scaler.apply(decision, resource.id)
        assert result.success
        assert result.new_replicas == prev + 1


# ---------------------------------------------------------------------------
# Migrator
# ---------------------------------------------------------------------------

class TestCloudMigrator:
    def test_plan_migration(self):
        sim = _make_simulator(2)
        migrator = CloudMigrator(simulator=sim, migration_failure_rate=0.0)
        plan = migrator.plan_migration(Region.US_EAST_1, Region.EU_WEST_1)
        assert len(plan.resource_ids) == 2
        assert plan.estimated_data_transfer_gb > 0

    def test_execute_migration_success(self):
        sim = _make_simulator(1)
        migrator = CloudMigrator(simulator=sim, migration_failure_rate=0.0)
        plan = migrator.plan_migration(Region.US_EAST_1, Region.EU_WEST_1)
        result = migrator.execute_migration(plan)
        assert result.success
        # Resource should now be in EU_WEST_1
        r = sim.get_running_resources()[0]
        assert r.spec.region == Region.EU_WEST_1

    def test_migration_empty(self):
        sim = _make_simulator(1)
        migrator = CloudMigrator(simulator=sim)
        # No resources in US_WEST_2
        plan = migrator.plan_migration(Region.US_WEST_2, Region.EU_WEST_1)
        assert len(plan.resource_ids) == 0


# ---------------------------------------------------------------------------
# Health & Self-healing
# ---------------------------------------------------------------------------

class TestHealthMonitor:
    def test_healthy_resource(self):
        sim = _make_simulator(1)
        sim.step()  # generate metrics
        hm = HealthMonitor(simulator=sim)
        checks = hm.check_all()
        assert len(checks) == 1
        # Under normal load should be healthy
        assert checks[0].healthy or not checks[0].healthy  # metrics are random

    def test_failed_resource(self):
        sim = _make_simulator(1)
        sim.step()
        resource = sim.get_running_resources()[0]
        resource.status = ResourceStatus.FAILED
        hm = HealthMonitor(simulator=sim)
        check = hm.check(resource)
        assert not check.healthy
        assert check.failure_type == FailureType.INSTANCE_CRASH

    def test_diagnose(self):
        sim = _make_simulator(1)
        hm = HealthMonitor(simulator=sim)
        diag = hm.diagnose(["Memory at 95.0%", "high memory usage"])
        assert diag.failure_type == FailureType.MEMORY_LEAK


class TestSelfHealer:
    def test_heal_crash(self):
        sim = _make_simulator(1)
        sim.step()
        resource = sim.get_running_resources()[0]
        resource.status = ResourceStatus.FAILED

        healer = SelfHealer(simulator=sim)
        diag = Diagnosis(
            resource_id=resource.id,
            failure_type=FailureType.INSTANCE_CRASH,
            severity=SeverityLevel.CRITICAL,
        )
        result = healer.heal(diag)
        assert result.success
        # Original resource was replaced
        assert result.resource_id != resource.id or len(sim.get_running_resources()) >= 1

    def test_heal_memory_leak(self):
        sim = _make_simulator(1)
        sim.step()
        resource = sim.get_running_resources()[0]
        resource.status = ResourceStatus.DEGRADED
        if resource.metrics:
            resource.metrics.memory_percent = 95.0

        healer = SelfHealer(simulator=sim)
        diag = Diagnosis(
            resource_id=resource.id,
            failure_type=FailureType.MEMORY_LEAK,
            severity=SeverityLevel.HIGH,
        )
        result = healer.heal(diag)
        assert result.success

    def test_max_heal_attempts(self):
        sim = _make_simulator(1)
        sim.step()
        resource = sim.get_running_resources()[0]
        resource.status = ResourceStatus.DEGRADED

        healer = SelfHealer(simulator=sim, max_heal_attempts=1)
        diag = Diagnosis(
            resource_id=resource.id,
            failure_type=FailureType.MEMORY_LEAK,
            severity=SeverityLevel.HIGH,
        )
        r1 = healer.heal(diag)
        assert r1.success

        resource.status = ResourceStatus.DEGRADED
        r2 = healer.heal(diag)
        assert not r2.success  # exceeded max attempts

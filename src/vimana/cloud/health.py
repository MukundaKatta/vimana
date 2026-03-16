"""Health monitoring and self-healing for cloud resources.

HealthMonitor detects problems; SelfHealer remediates them automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vimana.models import (
    ActionType,
    Diagnosis,
    FailureType,
    HealResult,
    HealthCheck,
    Resource,
    ResourceMetrics,
    ResourceSpec,
    ResourceStatus,
    SeverityLevel,
)
from vimana.simulation import CloudSimulator


# ---------------------------------------------------------------------------
# HealthMonitor
# ---------------------------------------------------------------------------

@dataclass
class HealthMonitor:
    """Monitors infrastructure resources and produces health checks."""

    simulator: CloudSimulator
    cpu_critical: float = 95.0
    memory_critical: float = 92.0
    disk_critical: float = 90.0
    latency_critical_ms: float = 2000.0
    error_rate_critical: float = 0.1

    def check_all(self) -> list[HealthCheck]:
        """Run health checks on every resource."""
        checks: list[HealthCheck] = []
        for resource in self.simulator.state.resources:
            if resource.status == ResourceStatus.TERMINATED:
                continue
            checks.append(self.check(resource))
        return checks

    def check(self, resource: Resource) -> HealthCheck:
        """Diagnose a single resource."""
        if resource.status == ResourceStatus.FAILED:
            return HealthCheck(
                resource_id=resource.id,
                healthy=False,
                symptoms=["instance not responding"],
                failure_type=FailureType.INSTANCE_CRASH,
                severity=SeverityLevel.CRITICAL,
            )

        if resource.metrics is None:
            return HealthCheck(resource_id=resource.id, healthy=True)

        symptoms: list[str] = []
        failure_type: FailureType | None = None
        severity = SeverityLevel.LOW
        m = resource.metrics

        if m.cpu_percent > self.cpu_critical:
            symptoms.append(f"CPU at {m.cpu_percent:.1f}%")
            failure_type = FailureType.CPU_SPIKE
            severity = SeverityLevel.HIGH

        if m.memory_percent > self.memory_critical:
            symptoms.append(f"Memory at {m.memory_percent:.1f}%")
            failure_type = FailureType.MEMORY_LEAK
            severity = max(severity, SeverityLevel.HIGH, key=lambda s: list(SeverityLevel).index(s))

        if m.disk_percent > self.disk_critical:
            symptoms.append(f"Disk at {m.disk_percent:.1f}%")
            failure_type = FailureType.DISK_FULL
            severity = max(severity, SeverityLevel.HIGH, key=lambda s: list(SeverityLevel).index(s))

        if m.network_in_mbps == 0 and m.network_out_mbps == 0:
            symptoms.append("Network unreachable")
            failure_type = FailureType.NETWORK_PARTITION
            severity = SeverityLevel.CRITICAL

        if m.error_rate > self.error_rate_critical:
            symptoms.append(f"Error rate {m.error_rate:.4f}")
            severity = max(severity, SeverityLevel.MEDIUM, key=lambda s: list(SeverityLevel).index(s))

        if m.latency_ms > self.latency_critical_ms:
            symptoms.append(f"Latency {m.latency_ms:.0f}ms")
            failure_type = failure_type or FailureType.LATENCY_SPIKE
            severity = max(severity, SeverityLevel.MEDIUM, key=lambda s: list(SeverityLevel).index(s))

        healthy = len(symptoms) == 0
        if resource.status == ResourceStatus.DEGRADED:
            healthy = False

        return HealthCheck(
            resource_id=resource.id,
            healthy=healthy,
            symptoms=symptoms,
            failure_type=failure_type,
            severity=severity if not healthy else SeverityLevel.LOW,
        )

    def diagnose(self, symptoms: list[str]) -> Diagnosis:
        """Analyze a list of symptom strings and produce a diagnosis."""
        symptom_text = " ".join(symptoms).lower()

        if "not responding" in symptom_text or "crash" in symptom_text:
            return Diagnosis(
                resource_id="",
                failure_type=FailureType.INSTANCE_CRASH,
                severity=SeverityLevel.CRITICAL,
                root_cause="Instance has crashed or become unresponsive.",
                recommended_actions=[ActionType.TERMINATE, ActionType.PROVISION],
                confidence=0.9,
            )
        if "memory" in symptom_text:
            return Diagnosis(
                resource_id="",
                failure_type=FailureType.MEMORY_LEAK,
                severity=SeverityLevel.HIGH,
                root_cause="Memory usage exceeds safe threshold; possible memory leak.",
                recommended_actions=[ActionType.HEAL, ActionType.SCALE_UP],
                confidence=0.8,
            )
        if "disk" in symptom_text:
            return Diagnosis(
                resource_id="",
                failure_type=FailureType.DISK_FULL,
                severity=SeverityLevel.HIGH,
                root_cause="Disk usage near capacity.",
                recommended_actions=[ActionType.HEAL, ActionType.RECONFIGURE],
                confidence=0.85,
            )
        if "network" in symptom_text:
            return Diagnosis(
                resource_id="",
                failure_type=FailureType.NETWORK_PARTITION,
                severity=SeverityLevel.CRITICAL,
                root_cause="Network connectivity lost.",
                recommended_actions=[ActionType.HEAL, ActionType.MIGRATE],
                confidence=0.7,
            )
        if "cpu" in symptom_text:
            return Diagnosis(
                resource_id="",
                failure_type=FailureType.CPU_SPIKE,
                severity=SeverityLevel.HIGH,
                root_cause="CPU utilization at dangerous levels.",
                recommended_actions=[ActionType.SCALE_UP],
                confidence=0.8,
            )

        return Diagnosis(
            resource_id="",
            failure_type=FailureType.LATENCY_SPIKE,
            severity=SeverityLevel.MEDIUM,
            root_cause="Unknown degradation detected.",
            recommended_actions=[ActionType.HEAL],
            confidence=0.4,
        )


# ---------------------------------------------------------------------------
# SelfHealer
# ---------------------------------------------------------------------------

@dataclass
class SelfHealer:
    """Automatically remediates infrastructure issues based on diagnoses."""

    simulator: CloudSimulator
    max_heal_attempts: int = 3
    _heal_counts: dict[str, int] = field(default_factory=dict)

    def heal(self, diagnosis: Diagnosis) -> HealResult:
        """Attempt to heal a resource based on its diagnosis."""
        rid = diagnosis.resource_id
        attempts = self._heal_counts.get(rid, 0)

        if attempts >= self.max_heal_attempts:
            return HealResult(
                resource_id=rid,
                success=False,
                message=f"Max heal attempts ({self.max_heal_attempts}) exceeded for {rid}. "
                        "Manual intervention required.",
            )

        self._heal_counts[rid] = attempts + 1
        resource = self.simulator.get_resource(rid)

        if resource is None:
            return HealResult(
                resource_id=rid,
                success=False,
                message=f"Resource {rid} not found.",
            )

        actions_taken: list[ActionType] = []
        recovery_time = 0.0

        if diagnosis.failure_type == FailureType.INSTANCE_CRASH:
            # Restart: terminate and re-provision
            spec_copy = resource.spec.model_copy(deep=True)
            self.simulator.remove_resource(rid)
            new_resource = self.simulator.add_resource(spec_copy)
            actions_taken = [ActionType.TERMINATE, ActionType.PROVISION]
            recovery_time = 60.0
            return HealResult(
                resource_id=new_resource.id,
                success=True,
                actions_taken=actions_taken,
                recovery_time_seconds=recovery_time,
                message=f"Replaced crashed instance {rid} with {new_resource.id}.",
            )

        if diagnosis.failure_type == FailureType.MEMORY_LEAK:
            # Restart the resource (simulated)
            self.simulator.repair_resource(rid)
            if resource.metrics:
                resource.metrics.memory_percent = 35.0
            actions_taken = [ActionType.HEAL]
            recovery_time = 30.0

        elif diagnosis.failure_type == FailureType.DISK_FULL:
            # Clean up disk
            self.simulator.repair_resource(rid)
            if resource.metrics:
                resource.metrics.disk_percent = 40.0
            actions_taken = [ActionType.HEAL]
            recovery_time = 15.0

        elif diagnosis.failure_type == FailureType.NETWORK_PARTITION:
            # Reset networking
            self.simulator.repair_resource(rid)
            if resource.metrics:
                resource.metrics.network_in_mbps = 100.0
                resource.metrics.network_out_mbps = 50.0
                resource.metrics.error_rate = 0.001
            actions_taken = [ActionType.HEAL]
            recovery_time = 20.0

        elif diagnosis.failure_type == FailureType.CPU_SPIKE:
            self.simulator.repair_resource(rid)
            if resource.metrics:
                resource.metrics.cpu_percent = 40.0
            actions_taken = [ActionType.HEAL]
            recovery_time = 10.0

        elif diagnosis.failure_type == FailureType.LATENCY_SPIKE:
            self.simulator.repair_resource(rid)
            if resource.metrics:
                resource.metrics.latency_ms = 20.0
            actions_taken = [ActionType.HEAL]
            recovery_time = 5.0

        else:
            self.simulator.repair_resource(rid)
            actions_taken = [ActionType.HEAL]
            recovery_time = 10.0

        return HealResult(
            resource_id=rid,
            success=True,
            actions_taken=actions_taken,
            recovery_time_seconds=recovery_time,
            message=f"Healed {rid}: {diagnosis.failure_type.value} resolved.",
        )

    def reset_heal_count(self, resource_id: str) -> None:
        self._heal_counts.pop(resource_id, None)

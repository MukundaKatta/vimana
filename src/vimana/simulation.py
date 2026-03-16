"""Cloud environment simulator.

Tick-based simulation that generates realistic infrastructure metrics,
supports failure injection, and provides a sandbox for AI agents to
practice self-orchestration.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from vimana.models import (
    CloudProvider,
    FailureType,
    InfrastructureState,
    Region,
    Resource,
    ResourceMetrics,
    ResourceSpec,
    ResourceStatus,
    ResourceType,
    SimulationScenario,
)


# ---------------------------------------------------------------------------
# Pricing table (simulated $/hr)
# ---------------------------------------------------------------------------
_BASE_PRICING: dict[ResourceType, float] = {
    ResourceType.VM: 0.10,           # per vCPU-hour
    ResourceType.CONTAINER: 0.04,
    ResourceType.SERVERLESS: 0.02,
    ResourceType.DATABASE: 0.25,
    ResourceType.LOAD_BALANCER: 0.03,
    ResourceType.STORAGE: 0.005,     # per GB-hour
}

_REGION_MULTIPLIER: dict[Region, float] = {
    Region.US_EAST_1: 1.00,
    Region.US_WEST_2: 1.05,
    Region.EU_WEST_1: 1.12,
    Region.EU_CENTRAL_1: 1.15,
    Region.AP_SOUTHEAST_1: 1.08,
    Region.AP_NORTHEAST_1: 1.20,
}


def compute_hourly_cost(spec: ResourceSpec) -> float:
    """Compute the simulated hourly cost for a resource specification."""
    base = _BASE_PRICING.get(spec.resource_type, 0.10)
    region_mult = _REGION_MULTIPLIER.get(spec.region, 1.0)
    cpu_cost = base * spec.vcpus * region_mult
    mem_cost = 0.01 * spec.memory_gb * region_mult
    storage_cost = _BASE_PRICING[ResourceType.STORAGE] * spec.storage_gb
    return round((cpu_cost + mem_cost + storage_cost) * spec.replicas, 6)


# ---------------------------------------------------------------------------
# Load patterns
# ---------------------------------------------------------------------------

def _load_at_tick(tick: int, scenario: SimulationScenario) -> float:
    """Return a load factor in [0, 1] for the given tick."""
    base = scenario.base_load
    pattern = scenario.load_pattern
    t = tick / max(scenario.total_ticks, 1)

    if pattern == "steady":
        load = base
    elif pattern == "burst":
        # Burst at ~40% and ~75% through the simulation
        burst1 = math.exp(-((t - 0.4) ** 2) / 0.002) * 0.6
        burst2 = math.exp(-((t - 0.75) ** 2) / 0.003) * 0.45
        load = base + burst1 + burst2
    elif pattern == "cyclic":
        load = base + 0.3 * math.sin(2 * math.pi * t * 4)
    elif pattern == "ramp":
        load = base + (1.0 - base) * t
    elif pattern == "random":
        load = base + random.uniform(-0.2, 0.4)
    else:
        load = base

    # Add noise
    load += random.gauss(0, 0.03)
    return max(0.0, min(1.0, load))


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

@dataclass
class CloudSimulator:
    """Tick-based simulation of a cloud environment.

    Maintains a set of resources, generates metrics each tick, and
    optionally injects failures based on the scenario configuration.
    """

    scenario: SimulationScenario
    state: InfrastructureState = field(default_factory=InfrastructureState)
    tick: int = 0
    rng: random.Random = field(default_factory=lambda: random.Random(42))
    _np_rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(42)
    )
    _failure_log: list[dict] = field(default_factory=list)

    # -- lifecycle ----------------------------------------------------------

    def initialize(self) -> InfrastructureState:
        """Provision initial resources defined in the scenario."""
        self.rng = random.Random(self.scenario.parameters.get("seed", 42))
        self._np_rng = np.random.default_rng(
            self.scenario.parameters.get("seed", 42)
        )
        self.tick = 0

        for spec in self.scenario.initial_resources:
            resource = Resource(
                spec=spec,
                status=ResourceStatus.RUNNING,
                cost_per_hour=compute_hourly_cost(spec),
            )
            self.state.resources.append(resource)

        self.state.refresh_totals()
        return self.state

    def step(self) -> InfrastructureState:
        """Advance the simulation by one tick."""
        self.tick += 1
        self.state.tick = self.tick
        load = _load_at_tick(self.tick, self.scenario)

        # Update metrics for running resources
        for resource in self.state.resources:
            if resource.status == ResourceStatus.RUNNING:
                resource.metrics = self._generate_metrics(resource, load)

        # Possibly inject a failure
        if self.rng.random() < self.scenario.failure_probability:
            self._inject_failure()

        self.state.refresh_totals()
        return self.state

    def run(self, ticks: int | None = None) -> list[InfrastructureState]:
        """Run the simulation for *ticks* steps (defaults to scenario total)."""
        ticks = ticks or self.scenario.total_ticks
        snapshots: list[InfrastructureState] = []
        for _ in range(ticks):
            snapshots.append(self.step().model_copy(deep=True))
        return snapshots

    # -- resource management ------------------------------------------------

    def add_resource(self, spec: ResourceSpec) -> Resource:
        """Provision a new resource into the simulation."""
        resource = Resource(
            spec=spec,
            status=ResourceStatus.PROVISIONING,
            cost_per_hour=compute_hourly_cost(spec),
        )
        self.state.resources.append(resource)
        # Takes 1-3 ticks to become running (simulated later)
        resource.status = ResourceStatus.RUNNING
        self.state.refresh_totals()
        return resource

    def remove_resource(self, resource_id: str) -> bool:
        """Terminate and remove a resource."""
        for i, r in enumerate(self.state.resources):
            if r.id == resource_id:
                r.status = ResourceStatus.TERMINATED
                self.state.resources.pop(i)
                self.state.refresh_totals()
                return True
        return False

    def get_resource(self, resource_id: str) -> Resource | None:
        for r in self.state.resources:
            if r.id == resource_id:
                return r
        return None

    def get_running_resources(self) -> list[Resource]:
        return [r for r in self.state.resources if r.status == ResourceStatus.RUNNING]

    # -- metrics generation -------------------------------------------------

    def _generate_metrics(self, resource: Resource, load: float) -> ResourceMetrics:
        """Generate realistic-ish metrics for a resource under *load*."""
        noise = self._np_rng.normal(0, 0.05)
        cpu = min(100.0, max(0.0, load * 100 + noise * 30))
        mem_base = 30 + load * 50
        mem = min(100.0, max(0.0, mem_base + self._np_rng.normal(0, 5)))
        disk = min(100.0, max(10.0, 20 + self.tick * 0.05 + self._np_rng.normal(0, 2)))
        net_in = max(0.0, load * 500 + self._np_rng.normal(0, 30))
        net_out = max(0.0, load * 200 + self._np_rng.normal(0, 15))
        rps = max(0.0, load * 1000 + self._np_rng.normal(0, 50))
        err_rate = max(0.0, 0.001 + (load ** 3) * 0.05 + self._np_rng.normal(0, 0.002))
        latency = max(1.0, 10 + load * 90 + (load ** 4) * 200 + self._np_rng.normal(0, 5))

        return ResourceMetrics(
            cpu_percent=round(cpu, 2),
            memory_percent=round(mem, 2),
            disk_percent=round(disk, 2),
            network_in_mbps=round(net_in, 2),
            network_out_mbps=round(net_out, 2),
            request_rate=round(rps, 2),
            error_rate=round(err_rate, 5),
            latency_ms=round(latency, 2),
        )

    # -- failure injection --------------------------------------------------

    def _inject_failure(self) -> None:
        running = self.get_running_resources()
        if not running:
            return
        target = self.rng.choice(running)
        failure_type = self.rng.choice(self.scenario.failure_types)
        self._apply_failure(target, failure_type)

    def inject_failure(self, resource_id: str, failure_type: FailureType) -> bool:
        """Manually inject a specific failure (for experiments)."""
        resource = self.get_resource(resource_id)
        if resource is None or resource.status != ResourceStatus.RUNNING:
            return False
        self._apply_failure(resource, failure_type)
        return True

    def _apply_failure(self, resource: Resource, failure_type: FailureType) -> None:
        if failure_type == FailureType.INSTANCE_CRASH:
            resource.status = ResourceStatus.FAILED
        elif failure_type == FailureType.MEMORY_LEAK:
            if resource.metrics:
                resource.metrics.memory_percent = min(
                    99.0, resource.metrics.memory_percent + 40
                )
            resource.status = ResourceStatus.DEGRADED
        elif failure_type == FailureType.DISK_FULL:
            if resource.metrics:
                resource.metrics.disk_percent = 99.5
            resource.status = ResourceStatus.DEGRADED
        elif failure_type == FailureType.NETWORK_PARTITION:
            if resource.metrics:
                resource.metrics.network_in_mbps = 0.0
                resource.metrics.network_out_mbps = 0.0
                resource.metrics.error_rate = 1.0
            resource.status = ResourceStatus.DEGRADED
        elif failure_type == FailureType.CPU_SPIKE:
            if resource.metrics:
                resource.metrics.cpu_percent = 99.9
            resource.status = ResourceStatus.DEGRADED
        elif failure_type == FailureType.LATENCY_SPIKE:
            if resource.metrics:
                resource.metrics.latency_ms = 5000.0
            resource.status = ResourceStatus.DEGRADED
        elif failure_type == FailureType.DNS_FAILURE:
            if resource.metrics:
                resource.metrics.error_rate = 0.8
            resource.status = ResourceStatus.DEGRADED

        self._failure_log.append(
            {"tick": self.tick, "resource_id": resource.id, "failure": failure_type.value}
        )

    def repair_resource(self, resource_id: str) -> bool:
        """Reset a degraded/failed resource to RUNNING."""
        resource = self.get_resource(resource_id)
        if resource is None:
            return False
        if resource.status in (ResourceStatus.DEGRADED, ResourceStatus.FAILED):
            resource.status = ResourceStatus.RUNNING
            return True
        return False

    # -- queries ------------------------------------------------------------

    @property
    def current_load(self) -> float:
        return _load_at_tick(self.tick, self.scenario)

    @property
    def failure_log(self) -> list[dict]:
        return list(self._failure_log)

    def aggregate_metrics(self) -> dict[str, float]:
        """Return averaged metrics across all running resources."""
        running = self.get_running_resources()
        if not running:
            return {}
        metrics = [r.metrics for r in running if r.metrics is not None]
        if not metrics:
            return {}
        n = len(metrics)
        return {
            "avg_cpu": round(sum(m.cpu_percent for m in metrics) / n, 2),
            "avg_memory": round(sum(m.memory_percent for m in metrics) / n, 2),
            "avg_latency_ms": round(sum(m.latency_ms for m in metrics) / n, 2),
            "total_rps": round(sum(m.request_rate for m in metrics), 2),
            "avg_error_rate": round(sum(m.error_rate for m in metrics) / n, 5),
            "total_cost_per_hour": round(self.state.total_cost_per_hour, 4),
        }

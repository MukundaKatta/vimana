"""Cloud resource provisioner.

Simulates provisioning and termination of cloud compute resources with
realistic latency, cost tracking, and failure modes.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

from vimana.models import (
    ActionResult,
    Resource,
    ResourceSpec,
    ResourceStatus,
    ResourceType,
)
from vimana.simulation import CloudSimulator, compute_hourly_cost


@dataclass
class ProvisionResult:
    success: bool
    resource: Resource | None = None
    cost_per_hour: float = 0.0
    provision_latency_seconds: float = 0.0
    message: str = ""


@dataclass
class CloudProvisioner:
    """Provisions and terminates simulated cloud resources.

    Wraps the CloudSimulator to provide a higher-level API that tracks
    provisioning history and enforces quotas.
    """

    simulator: CloudSimulator
    max_resources: int = 50
    provision_failure_rate: float = 0.02
    _rng: random.Random = field(default_factory=lambda: random.Random(42))
    _history: list[ProvisionResult] = field(default_factory=list)

    # -- provisioning -------------------------------------------------------

    def provision(self, spec: ResourceSpec) -> ProvisionResult:
        """Provision a new cloud resource from a specification."""
        # Check quota
        running = self.simulator.get_running_resources()
        if len(running) >= self.max_resources:
            result = ProvisionResult(
                success=False,
                message=f"Resource quota exceeded ({self.max_resources} max).",
            )
            self._history.append(result)
            return result

        # Simulate random provision failure
        if self._rng.random() < self.provision_failure_rate:
            result = ProvisionResult(
                success=False,
                message="Provisioning failed: simulated cloud API error.",
                provision_latency_seconds=self._rng.uniform(0.5, 2.0),
            )
            self._history.append(result)
            return result

        # Provision
        latency = self._simulate_provision_latency(spec.resource_type)
        resource = self.simulator.add_resource(spec)
        cost = compute_hourly_cost(spec)

        result = ProvisionResult(
            success=True,
            resource=resource,
            cost_per_hour=cost,
            provision_latency_seconds=latency,
            message=f"Provisioned {spec.resource_type.value} '{spec.name}' "
                    f"in {spec.region.value} ({spec.vcpus} vCPU, {spec.memory_gb} GB RAM).",
        )
        self._history.append(result)
        return result

    def terminate(self, resource_id: str) -> bool:
        """Terminate a resource by ID."""
        return self.simulator.remove_resource(resource_id)

    def scale_replicas(self, resource_id: str, new_replicas: int) -> ActionResult:
        """Change the replica count for a resource."""
        resource = self.simulator.get_resource(resource_id)
        if resource is None:
            return ActionResult(
                action_id="scale",
                success=False,
                error=f"Resource {resource_id} not found.",
            )
        old_replicas = resource.spec.replicas
        resource.spec.replicas = max(0, new_replicas)
        resource.cost_per_hour = compute_hourly_cost(resource.spec)
        self.simulator.state.refresh_totals()

        if new_replicas == 0:
            resource.status = ResourceStatus.TERMINATED

        return ActionResult(
            action_id="scale",
            success=True,
            resource_id=resource_id,
            message=f"Scaled {resource_id} from {old_replicas} to {new_replicas} replicas.",
            actual_cost_delta=resource.cost_per_hour
            - compute_hourly_cost(
                resource.spec.model_copy(update={"replicas": old_replicas})
            ),
        )

    # -- queries ------------------------------------------------------------

    def list_resources(self) -> list[Resource]:
        return list(self.simulator.state.resources)

    def get_resource(self, resource_id: str) -> Resource | None:
        return self.simulator.get_resource(resource_id)

    @property
    def total_cost_per_hour(self) -> float:
        return self.simulator.state.total_cost_per_hour

    @property
    def history(self) -> list[ProvisionResult]:
        return list(self._history)

    # -- internal -----------------------------------------------------------

    @staticmethod
    def _simulate_provision_latency(resource_type: ResourceType) -> float:
        """Return a realistic provisioning latency in seconds."""
        base: dict[ResourceType, float] = {
            ResourceType.VM: 45.0,
            ResourceType.CONTAINER: 8.0,
            ResourceType.SERVERLESS: 1.5,
            ResourceType.DATABASE: 120.0,
            ResourceType.LOAD_BALANCER: 20.0,
            ResourceType.STORAGE: 5.0,
        }
        b = base.get(resource_type, 30.0)
        return round(b + random.gauss(0, b * 0.15), 2)

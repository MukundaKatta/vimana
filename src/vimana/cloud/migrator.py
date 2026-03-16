"""Cross-region and cross-cloud workload migration.

Simulates the full migration lifecycle: planning, data transfer,
DNS cutover, verification, and rollback.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from vimana.models import (
    MigrationPlan,
    MigrationResult,
    Region,
    Resource,
    ResourceSpec,
    ResourceStatus,
)
from vimana.simulation import CloudSimulator, compute_hourly_cost


# Transfer speed simulation (GB/s between regions)
_TRANSFER_SPEEDS: dict[tuple[Region, Region], float] = {}
for r1 in Region:
    for r2 in Region:
        if r1 == r2:
            _TRANSFER_SPEEDS[(r1, r2)] = 10.0
        elif r1.value[:2] == r2.value[:2]:  # same continent prefix
            _TRANSFER_SPEEDS[(r1, r2)] = 2.0
        else:
            _TRANSFER_SPEEDS[(r1, r2)] = 0.5


@dataclass
class CloudMigrator:
    """Migrates workloads between regions or cloud providers.

    Handles data transfer estimation, coordinated cutover, and rollback
    on failure.
    """

    simulator: CloudSimulator
    migration_failure_rate: float = 0.05
    _rng: random.Random = field(default_factory=lambda: random.Random(42))
    _history: list[MigrationResult] = field(default_factory=list)

    def plan_migration(
        self,
        source_region: Region,
        target_region: Region,
        resource_ids: list[str] | None = None,
    ) -> MigrationPlan:
        """Create a migration plan from source to target region.

        If resource_ids is None, migrate all running resources in the source region.
        """
        if resource_ids is None:
            resource_ids = [
                r.id
                for r in self.simulator.state.resources
                if r.spec.region == source_region and r.status == ResourceStatus.RUNNING
            ]

        total_data_gb = sum(
            self._estimate_data_gb(r)
            for r in self.simulator.state.resources
            if r.id in resource_ids
        )

        speed = _TRANSFER_SPEEDS.get((source_region, target_region), 0.5)
        estimated_transfer_time = total_data_gb / speed if speed > 0 else float("inf")
        # DNS cutover adds ~30s, verification ~15s
        estimated_downtime = estimated_transfer_time + 45.0

        return MigrationPlan(
            source_region=source_region,
            target_region=target_region,
            resource_ids=resource_ids,
            estimated_downtime_seconds=round(estimated_downtime, 2),
            estimated_data_transfer_gb=round(total_data_gb, 2),
            rollback_enabled=True,
            reason=f"Migrate {len(resource_ids)} resources from "
                   f"{source_region.value} to {target_region.value}.",
        )

    def execute_migration(self, plan: MigrationPlan) -> MigrationResult:
        """Execute a migration plan, moving resources to the target region."""
        migrated_resources: list[Resource] = []
        original_specs: dict[str, ResourceSpec] = {}

        for rid in plan.resource_ids:
            resource = self.simulator.get_resource(rid)
            if resource is None:
                continue
            original_specs[rid] = resource.spec.model_copy(deep=True)
            resource.status = ResourceStatus.MIGRATING
            migrated_resources.append(resource)

        # Simulate transfer
        speed = _TRANSFER_SPEEDS.get(
            (plan.source_region, plan.target_region), 0.5
        )
        actual_transfer_time = plan.estimated_data_transfer_gb / speed if speed > 0 else 0
        actual_downtime = actual_transfer_time + 45.0 + self._rng.uniform(-5, 10)

        # Check for migration failure
        if self._rng.random() < self.migration_failure_rate:
            # Rollback
            if plan.rollback_enabled:
                for resource in migrated_resources:
                    resource.status = ResourceStatus.RUNNING
                    resource.spec = original_specs[resource.id]
            result = MigrationResult(
                plan_id=plan.id,
                success=False,
                actual_downtime_seconds=round(actual_downtime * 0.3, 2),
                data_transferred_gb=round(plan.estimated_data_transfer_gb * 0.3, 2),
                cost=round(plan.estimated_data_transfer_gb * 0.09, 4),
                message="Migration failed during data transfer. Rollback executed.",
                rolled_back=plan.rollback_enabled,
            )
            self._history.append(result)
            return result

        # Success: update resources to target region
        for resource in migrated_resources:
            resource.spec.region = plan.target_region
            resource.cost_per_hour = compute_hourly_cost(resource.spec)
            resource.status = ResourceStatus.RUNNING

        self.simulator.state.refresh_totals()

        # Cost: data transfer cost ($0.09/GB is typical cross-region)
        transfer_cost = plan.estimated_data_transfer_gb * 0.09

        result = MigrationResult(
            plan_id=plan.id,
            success=True,
            actual_downtime_seconds=round(actual_downtime, 2),
            data_transferred_gb=plan.estimated_data_transfer_gb,
            cost=round(transfer_cost, 4),
            message=f"Successfully migrated {len(migrated_resources)} resources "
                    f"to {plan.target_region.value}.",
        )
        self._history.append(result)
        return result

    @property
    def history(self) -> list[MigrationResult]:
        return list(self._history)

    @staticmethod
    def _estimate_data_gb(resource: Resource) -> float:
        """Estimate data volume for a resource (storage + in-memory state)."""
        return resource.spec.storage_gb + resource.spec.memory_gb * 0.5

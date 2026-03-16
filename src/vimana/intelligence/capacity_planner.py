"""Long-term capacity planning.

Given a demand forecast, budget, and SLA requirements, produce a
CapacityPlan that maps out the resources needed over a planning horizon.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from vimana.models import (
    CapacityPlan,
    DemandForecast,
    Region,
    ResourceSpec,
    ResourceType,
)
from vimana.simulation import compute_hourly_cost


@dataclass
class CapacityPlanner:
    """Plans long-term infrastructure capacity.

    Translates demand forecasts into concrete resource specifications,
    respecting budget and SLA constraints. Adds a configurable headroom
    buffer above predicted peak demand.
    """

    headroom_percent: float = 20.0
    default_resource_type: ResourceType = ResourceType.VM
    default_region: Region = Region.US_EAST_1
    vcpus_per_unit: int = 4
    memory_per_unit_gb: float = 8.0

    def plan(
        self,
        forecast: DemandForecast,
        budget_per_hour: float = 50.0,
        sla_cpu_headroom: float = 30.0,
    ) -> CapacityPlan:
        """Generate a capacity plan from a demand forecast.

        Args:
            forecast: Predicted future demand.
            budget_per_hour: Maximum hourly spend.
            sla_cpu_headroom: Target headroom above peak CPU (percent points).

        Returns:
            A CapacityPlan with resource specifications and cost estimates.
        """
        if not forecast.predicted_cpu:
            return CapacityPlan(
                horizon_ticks=0,
                notes="No forecast data available.",
            )

        # Determine peak demand
        peak_cpu = max(forecast.predicted_cpu)
        peak_memory = max(forecast.predicted_memory) if forecast.predicted_memory else 50.0
        peak_requests = max(forecast.predicted_requests) if forecast.predicted_requests else 100.0

        # Apply headroom
        headroom_mult = 1.0 + self.headroom_percent / 100.0
        target_cpu_capacity = peak_cpu * headroom_mult
        target_memory_capacity = peak_memory * headroom_mult

        # How many resource units do we need?
        # Assume each unit contributes 100/N of the total CPU capacity
        # where N is the number of units
        cpu_units = max(1, math.ceil(target_cpu_capacity / (100.0 / max(1, self.vcpus_per_unit))))
        mem_units = max(1, math.ceil(target_memory_capacity / (100.0 / max(1, int(self.memory_per_unit_gb)))))
        num_units = max(cpu_units, mem_units)

        # Distribute across regions based on forecast trend
        regions = self._select_regions(forecast)

        # Build resource specs
        planned: list[ResourceSpec] = []
        units_per_region = max(1, num_units // len(regions))
        remainder = num_units - units_per_region * len(regions)

        for i, region in enumerate(regions):
            count = units_per_region + (1 if i < remainder else 0)
            if count <= 0:
                continue
            spec = ResourceSpec(
                resource_type=self.default_resource_type,
                name=f"capacity-{region.value}-{i}",
                vcpus=self.vcpus_per_unit,
                memory_gb=self.memory_per_unit_gb,
                region=region,
                replicas=count,
            )
            planned.append(spec)

        total_cost = sum(compute_hourly_cost(s) for s in planned)
        total_vcpus = sum(s.vcpus * s.replicas for s in planned)
        total_mem = sum(s.memory_gb * s.replicas for s in planned)

        # Check budget
        notes_parts: list[str] = []
        if total_cost > budget_per_hour:
            notes_parts.append(
                f"WARNING: planned cost ${total_cost:.2f}/hr exceeds "
                f"budget ${budget_per_hour:.2f}/hr. Consider smaller instances "
                f"or serverless."
            )

        notes_parts.append(
            f"Trend: {forecast.trend}. "
            f"Peak CPU: {peak_cpu:.1f}%, Peak mem: {peak_memory:.1f}%. "
            f"Planned {num_units} units across {len(regions)} region(s)."
        )

        return CapacityPlan(
            horizon_ticks=len(forecast.timestamps),
            planned_resources=planned,
            estimated_cost_total=round(total_cost * len(forecast.timestamps), 2),
            peak_vcpus=total_vcpus,
            peak_memory_gb=total_mem,
            headroom_percent=self.headroom_percent,
            notes=" ".join(notes_parts),
        )

    def _select_regions(self, forecast: DemandForecast) -> list[Region]:
        """Choose deployment regions based on demand characteristics."""
        if forecast.trend == "rising" or forecast.trend == "cyclic":
            # Multi-region for resilience under growing/variable demand
            return [Region.US_EAST_1, Region.US_WEST_2]
        return [self.default_region]

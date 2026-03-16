"""AI-driven cost optimization for cloud infrastructure.

Analyzes current resource configurations and recommends cheaper
alternatives (right-sizing, region arbitrage, spot instances,
reserved capacity) while respecting performance constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vimana.models import (
    ActionType,
    InfraAction,
    InfrastructureState,
    OptimizedConfig,
    Region,
    Resource,
    ResourceSpec,
    ResourceStatus,
    ResourceType,
)
from vimana.simulation import compute_hourly_cost, _REGION_MULTIPLIER


# ---------------------------------------------------------------------------
# Optimization strategies
# ---------------------------------------------------------------------------

def _right_size(resource: Resource) -> InfraAction | None:
    """Recommend downsizing if utilization is low."""
    m = resource.metrics
    if m is None:
        return None

    if m.cpu_percent < 20 and m.memory_percent < 30:
        new_vcpus = max(1, resource.spec.vcpus // 2)
        new_mem = max(1.0, resource.spec.memory_gb / 2)
        old_cost = compute_hourly_cost(resource.spec)
        new_spec = resource.spec.model_copy(
            update={"vcpus": new_vcpus, "memory_gb": new_mem}
        )
        new_cost = compute_hourly_cost(new_spec)
        return InfraAction(
            action_type=ActionType.RECONFIGURE,
            target_resource_id=resource.id,
            resource_spec=new_spec,
            reason=f"Right-size: CPU {m.cpu_percent:.0f}%, Mem {m.memory_percent:.0f}% -- "
                   f"reduce from {resource.spec.vcpus}vCPU/{resource.spec.memory_gb}GB "
                   f"to {new_vcpus}vCPU/{new_mem}GB "
                   f"(save ${old_cost - new_cost:.4f}/hr).",
            estimated_cost_delta=new_cost - old_cost,
        )
    return None


def _region_arbitrage(resource: Resource) -> InfraAction | None:
    """Recommend moving to a cheaper region if cost difference is significant."""
    current_mult = _REGION_MULTIPLIER.get(resource.spec.region, 1.0)
    cheapest_region = min(_REGION_MULTIPLIER, key=_REGION_MULTIPLIER.get)  # type: ignore[arg-type]
    cheapest_mult = _REGION_MULTIPLIER[cheapest_region]

    if cheapest_mult < current_mult * 0.9:  # at least 10% cheaper
        savings_pct = (1 - cheapest_mult / current_mult) * 100
        return InfraAction(
            action_type=ActionType.MIGRATE,
            target_resource_id=resource.id,
            parameters={
                "source_region": resource.spec.region.value,
                "target_region": cheapest_region.value,
            },
            reason=f"Region arbitrage: move from {resource.spec.region.value} "
                   f"(x{current_mult:.2f}) to {cheapest_region.value} "
                   f"(x{cheapest_mult:.2f}) for ~{savings_pct:.0f}% savings.",
            estimated_cost_delta=-resource.cost_per_hour * (savings_pct / 100),
        )
    return None


def _spot_instance(resource: Resource) -> InfraAction | None:
    """Recommend switching to serverless/container for non-critical workloads."""
    if resource.spec.resource_type == ResourceType.VM and resource.spec.vcpus <= 4:
        new_spec = resource.spec.model_copy(
            update={"resource_type": ResourceType.CONTAINER}
        )
        old_cost = compute_hourly_cost(resource.spec)
        new_cost = compute_hourly_cost(new_spec)
        if new_cost < old_cost:
            return InfraAction(
                action_type=ActionType.RECONFIGURE,
                target_resource_id=resource.id,
                resource_spec=new_spec,
                reason=f"Switch from VM to container: save "
                       f"${old_cost - new_cost:.4f}/hr.",
                estimated_cost_delta=new_cost - old_cost,
            )
    return None


def _consolidate_replicas(resource: Resource) -> InfraAction | None:
    """Recommend reducing replicas if over-provisioned."""
    if resource.spec.replicas > 2:
        m = resource.metrics
        if m and m.cpu_percent < 30 and m.memory_percent < 40:
            new_replicas = max(1, resource.spec.replicas - 1)
            old_spec = resource.spec.model_copy()
            new_spec = resource.spec.model_copy(update={"replicas": new_replicas})
            saving = compute_hourly_cost(old_spec) - compute_hourly_cost(new_spec)
            return InfraAction(
                action_type=ActionType.SCALE_DOWN,
                target_resource_id=resource.id,
                parameters={"replica_delta": -1},
                reason=f"Consolidate: reduce replicas {resource.spec.replicas} -> "
                       f"{new_replicas} (save ${saving:.4f}/hr).",
                estimated_cost_delta=-saving,
            )
    return None


# ---------------------------------------------------------------------------
# CostOptimizer
# ---------------------------------------------------------------------------

@dataclass
class CostOptimizer:
    """Finds cheaper infrastructure configurations that still meet SLA.

    Applies multiple optimization strategies and aggregates recommendations
    into an OptimizedConfig.
    """

    min_savings_threshold: float = 0.01  # $/hr minimum to recommend a change

    def optimize(
        self,
        state: InfrastructureState,
        constraints: dict[str, Any] | None = None,
    ) -> OptimizedConfig:
        """Analyze current infrastructure and return optimization recommendations."""
        constraints = constraints or {}
        recommendations: list[InfraAction] = []
        running = [r for r in state.resources if r.status == ResourceStatus.RUNNING]

        strategies = [_right_size, _region_arbitrage, _spot_instance, _consolidate_replicas]

        for resource in running:
            for strategy in strategies:
                action = strategy(resource)
                if action and abs(action.estimated_cost_delta) >= self.min_savings_threshold:
                    recommendations.append(action)

        # Deduplicate by resource ID (keep the highest-saving recommendation)
        best_per_resource: dict[str, InfraAction] = {}
        for rec in recommendations:
            rid = rec.target_resource_id or ""
            if rid not in best_per_resource or rec.estimated_cost_delta < best_per_resource[rid].estimated_cost_delta:
                best_per_resource[rid] = rec

        final_recs = list(best_per_resource.values())
        total_savings = sum(abs(r.estimated_cost_delta) for r in final_recs)
        original_cost = state.total_cost_per_hour
        optimized_cost = max(0.0, original_cost - total_savings)
        savings_pct = (total_savings / original_cost * 100) if original_cost > 0 else 0.0

        reasoning_parts = [r.reason for r in final_recs]
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "No optimizations found."

        return OptimizedConfig(
            original_cost_per_hour=round(original_cost, 4),
            optimized_cost_per_hour=round(optimized_cost, 4),
            savings_percent=round(savings_pct, 1),
            recommendations=final_recs,
            reasoning=reasoning,
        )

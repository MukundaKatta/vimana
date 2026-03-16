"""AI flight planner -- decides infrastructure actions.

The FlightPlanner takes the current infrastructure state and a goal,
then produces an InfrastructurePlan: an ordered sequence of actions
(provision, scale, migrate, heal, terminate) to reach the goal.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from vimana.models import (
    ActionType,
    CloudProvider,
    InfraAction,
    InfrastructurePlan,
    InfrastructureState,
    Region,
    ResourceSpec,
    ResourceStatus,
    ResourceType,
    Waypoint,
)
from vimana.navigator.waypoints import WaypointGraph, state_to_waypoint


# ---------------------------------------------------------------------------
# Heuristic planner (no LLM required)
# ---------------------------------------------------------------------------

def _heuristic_plan(
    state: InfrastructureState,
    goal: Waypoint,
    constraints: dict[str, Any] | None = None,
) -> InfrastructurePlan:
    """Build an infrastructure plan using deterministic heuristics.

    This is the fallback when no LLM API key is available.
    """
    constraints = constraints or {}
    actions: list[InfraAction] = []
    current = state_to_waypoint(state, name="current")
    reasoning_parts: list[str] = []

    # --- Heal degraded/failed resources first ---
    for r in state.resources:
        if r.status in (ResourceStatus.DEGRADED, ResourceStatus.FAILED):
            actions.append(
                InfraAction(
                    action_type=ActionType.HEAL,
                    target_resource_id=r.id,
                    reason=f"Resource {r.id} is {r.status.value}; healing required.",
                    priority=10,
                )
            )
            reasoning_parts.append(f"Heal {r.id} ({r.status.value})")

    # --- Scale: compute gap ---
    cpu_gap = goal.compute_vcpus - current.compute_vcpus
    mem_gap = goal.memory_gb - current.memory_gb

    if cpu_gap > 0 or mem_gap > 0:
        needed_vcpus = max(cpu_gap, 0)
        needed_mem = max(mem_gap, 0.0)
        # Decide how many new resources to add
        resource_type = goal.resource_types[0] if goal.resource_types else ResourceType.VM
        vcpus_per = 4 if resource_type == ResourceType.VM else 2
        mem_per = 8.0 if resource_type == ResourceType.VM else 4.0
        count = max(
            1,
            max(
                (needed_vcpus + vcpus_per - 1) // vcpus_per,
                int((needed_mem + mem_per - 1) // mem_per),
            ),
        )

        for i in range(count):
            region = goal.regions[i % len(goal.regions)] if goal.regions else Region.US_EAST_1
            spec = ResourceSpec(
                resource_type=resource_type,
                name=f"vimana-{resource_type.value}-{i}",
                vcpus=vcpus_per,
                memory_gb=mem_per,
                region=region,
            )
            actions.append(
                InfraAction(
                    action_type=ActionType.PROVISION,
                    resource_spec=spec,
                    reason=f"Need +{vcpus_per} vCPU, +{mem_per} GB RAM to reach goal.",
                    estimated_cost_delta=0.10 * vcpus_per,
                    priority=5,
                )
            )
        reasoning_parts.append(f"Provision {count} {resource_type.value}(s) for compute gap")

    elif cpu_gap < -4:
        # Over-provisioned: terminate excess
        excess = abs(cpu_gap) // 4
        running = [
            r for r in state.resources if r.status == ResourceStatus.RUNNING
        ]
        for r in running[:excess]:
            actions.append(
                InfraAction(
                    action_type=ActionType.TERMINATE,
                    target_resource_id=r.id,
                    reason=f"Over-provisioned by {abs(cpu_gap)} vCPUs; terminating {r.id}.",
                    estimated_cost_delta=-r.cost_per_hour,
                    priority=3,
                )
            )
        reasoning_parts.append(f"Terminate {excess} excess resources")

    # --- Region migration ---
    current_regions = set(current.regions)
    goal_regions = set(goal.regions)
    missing_regions = goal_regions - current_regions

    if missing_regions:
        for region in missing_regions:
            resource_type = goal.resource_types[0] if goal.resource_types else ResourceType.VM
            spec = ResourceSpec(
                resource_type=resource_type,
                name=f"vimana-{region.value}",
                vcpus=4,
                memory_gb=8.0,
                region=region,
            )
            actions.append(
                InfraAction(
                    action_type=ActionType.PROVISION,
                    resource_spec=spec,
                    reason=f"Goal requires presence in {region.value}.",
                    priority=4,
                )
            )
        reasoning_parts.append(f"Expand to regions: {[r.value for r in missing_regions]}")

    # --- Cost constraint ---
    max_cost = constraints.get("max_cost_per_hour", goal.max_cost_per_hour)
    if max_cost < float("inf"):
        estimated_cost = state.total_cost_per_hour + sum(
            a.estimated_cost_delta for a in actions
        )
        if estimated_cost > max_cost:
            reasoning_parts.append(
                f"Warning: estimated cost ${estimated_cost:.2f}/hr exceeds "
                f"budget ${max_cost:.2f}/hr"
            )

    # Sort by priority (highest first)
    actions.sort(key=lambda a: a.priority, reverse=True)

    total_cost = sum(a.estimated_cost_delta for a in actions)
    reasoning = "Heuristic plan: " + "; ".join(reasoning_parts) if reasoning_parts else "No actions needed."

    return InfrastructurePlan(
        actions=actions,
        goal_description=f"Reach waypoint '{goal.name}' ({goal.compute_vcpus} vCPU, "
                         f"{goal.memory_gb} GB, {goal.replicas} replicas, "
                         f"regions={[r.value for r in goal.regions]})",
        estimated_total_cost=total_cost,
        estimated_duration_seconds=len(actions) * 30.0,
        confidence=0.7 if actions else 1.0,
        reasoning=reasoning,
    )


# ---------------------------------------------------------------------------
# LLM-backed planner
# ---------------------------------------------------------------------------

def _llm_plan(
    state: InfrastructureState,
    goal: Waypoint,
    constraints: dict[str, Any] | None = None,
) -> InfrastructurePlan | None:
    """Attempt to build a plan using an LLM (Anthropic or OpenAI).

    Returns None if no API key is configured.
    """
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if not anthropic_key and not openai_key:
        return None

    prompt = _build_planning_prompt(state, goal, constraints)

    try:
        if anthropic_key:
            return _call_anthropic(prompt, anthropic_key)
        else:
            return _call_openai(prompt, openai_key)  # type: ignore[arg-type]
    except Exception:
        return None


def _build_planning_prompt(
    state: InfrastructureState,
    goal: Waypoint,
    constraints: dict[str, Any] | None,
) -> str:
    resources_summary = []
    for r in state.resources:
        resources_summary.append(
            {
                "id": r.id,
                "type": r.spec.resource_type.value,
                "vcpus": r.spec.vcpus,
                "memory_gb": r.spec.memory_gb,
                "region": r.spec.region.value,
                "status": r.status.value,
                "replicas": r.spec.replicas,
                "cost_per_hour": r.cost_per_hour,
            }
        )

    return f"""You are an AI infrastructure planner for the Vimana self-orchestrating system.

Current infrastructure state:
- Total vCPUs: {state.total_vcpus}
- Total memory: {state.total_memory_gb} GB
- Total cost/hr: ${state.total_cost_per_hour:.4f}
- Active regions: {[r.value for r in state.regions_active]}
- Resources: {json.dumps(resources_summary, indent=2)}

Target waypoint "{goal.name}":
- Required vCPUs: {goal.compute_vcpus}
- Required memory: {goal.memory_gb} GB
- Required replicas: {goal.replicas}
- Required regions: {[r.value for r in goal.regions]}
- Max cost/hr: ${goal.max_cost_per_hour}
- Min availability: {goal.min_availability}
- Resource types: {[t.value for t in goal.resource_types]}

Constraints: {json.dumps(constraints or {{}}, indent=2)}

Produce a JSON plan with this structure:
{{
  "actions": [
    {{
      "action_type": "provision|scale_up|scale_down|migrate|heal|terminate|reconfigure",
      "resource_type": "vm|container|serverless|database",
      "vcpus": 4,
      "memory_gb": 8.0,
      "region": "us-east-1",
      "target_resource_id": null,
      "reason": "explanation"
    }}
  ],
  "reasoning": "overall strategy explanation",
  "confidence": 0.85
}}

Respond ONLY with the JSON object, no other text."""


def _call_anthropic(prompt: str, api_key: str) -> InfrastructurePlan | None:
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text  # type: ignore[union-attr]
        return _parse_llm_response(text)
    except Exception:
        return None


def _call_openai(prompt: str, api_key: str) -> InfrastructurePlan | None:
    try:
        import openai

        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
        )
        text = response.choices[0].message.content or ""
        return _parse_llm_response(text)
    except Exception:
        return None


def _parse_llm_response(text: str) -> InfrastructurePlan | None:
    """Parse the LLM JSON response into an InfrastructurePlan."""
    try:
        # Strip markdown code fences if present
        clean = text.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1]
            clean = clean.rsplit("```", 1)[0]

        data = json.loads(clean)
        actions: list[InfraAction] = []
        for a in data.get("actions", []):
            action_type = ActionType(a["action_type"])
            spec = None
            if action_type == ActionType.PROVISION:
                spec = ResourceSpec(
                    resource_type=ResourceType(a.get("resource_type", "vm")),
                    vcpus=a.get("vcpus", 4),
                    memory_gb=a.get("memory_gb", 8.0),
                    region=Region(a.get("region", "us-east-1")),
                    name=a.get("name", ""),
                )
            actions.append(
                InfraAction(
                    action_type=action_type,
                    target_resource_id=a.get("target_resource_id"),
                    resource_spec=spec,
                    reason=a.get("reason", ""),
                )
            )

        return InfrastructurePlan(
            actions=actions,
            reasoning=data.get("reasoning", ""),
            confidence=data.get("confidence", 0.5),
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# FlightPlanner
# ---------------------------------------------------------------------------

@dataclass
class FlightPlanner:
    """AI flight planner that decides infrastructure actions.

    Uses an LLM when available, falling back to heuristic planning.
    Navigation between waypoints is handled via a WaypointGraph.
    """

    waypoint_graph: WaypointGraph = field(default_factory=WaypointGraph.default)
    use_llm: bool = True

    def plan(
        self,
        current_state: InfrastructureState,
        goal: Waypoint,
        constraints: dict[str, Any] | None = None,
    ) -> InfrastructurePlan:
        """Produce an InfrastructurePlan to move from current_state toward goal.

        Tries LLM-based planning first (if enabled and API key present),
        then falls back to heuristic planning.
        """
        # Try LLM
        if self.use_llm:
            llm_plan = _llm_plan(current_state, goal, constraints)
            if llm_plan is not None:
                return llm_plan

        # Fallback to heuristic
        return _heuristic_plan(current_state, goal, constraints)

    def plan_route(
        self,
        current_state: InfrastructureState,
        target: Waypoint,
        constraints: dict[str, Any] | None = None,
    ) -> list[InfrastructurePlan]:
        """Plan a multi-step route through waypoints to reach *target*.

        Uses the WaypointGraph to find intermediate waypoints, then
        generates a plan for each hop.
        """
        current_wp = state_to_waypoint(current_state)
        route = self.waypoint_graph.calculate_route(current_wp, target)

        plans: list[InfrastructurePlan] = []
        state = current_state
        for waypoint in route:
            p = self.plan(state, waypoint, constraints)
            plans.append(p)
        return plans

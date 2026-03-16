"""Experiment: Can AI provision its own infrastructure?

Give an AI agent a task (a target waypoint) and measure whether it can
autonomously provision the right resources, at the right scale, in the
right region -- and how efficiently it does so.
"""

from __future__ import annotations

from datetime import datetime, timezone

from rich.console import Console

from vimana.cloud.health import HealthMonitor, SelfHealer
from vimana.cloud.migrator import CloudMigrator
from vimana.cloud.provisioner import CloudProvisioner
from vimana.models import (
    ExperimentConfig,
    ExperimentResult,
    Region,
    ResourceSpec,
    ResourceType,
    SimulationScenario,
)
from vimana.navigator.autopilot import Autopilot
from vimana.navigator.planner import FlightPlanner
from vimana.navigator.waypoints import WAYPOINT_STANDARD, Waypoint
from vimana.simulation import CloudSimulator

console = Console()


def run_self_provision_experiment(
    config: ExperimentConfig | None = None,
) -> ExperimentResult:
    """Run the self-provisioning experiment.

    Starts with zero infrastructure and asks the AI to provision resources
    to reach a target waypoint. Measures success rate, cost efficiency,
    time to provision, and over/under-provisioning.
    """
    config = config or ExperimentConfig(
        name="self-provision",
        duration_ticks=100,
        seed=42,
    )

    console.print("\n[bold magenta]=== Experiment: Self-Provision ===[/bold magenta]")
    console.print(f"  Duration: {config.duration_ticks} ticks | Seed: {config.seed}")

    # Setup: empty infrastructure
    scenario = SimulationScenario(
        name="self-provision",
        description="Start from nothing, reach a target state.",
        total_ticks=config.duration_ticks,
        base_load=0.4,
        load_pattern="ramp",
        failure_probability=0.01,
        region=Region.US_EAST_1,
        initial_resources=[],
        parameters={"seed": config.seed},
    )

    simulator = CloudSimulator(scenario=scenario)
    simulator.initialize()

    provisioner = CloudProvisioner(simulator=simulator)
    migrator = CloudMigrator(simulator=simulator)
    health_monitor = HealthMonitor(simulator=simulator)
    self_healer = SelfHealer(simulator=simulator)

    # Target waypoint
    target = config.parameters.get("target_waypoint", None)
    if target is None:
        target = WAYPOINT_STANDARD

    if isinstance(target, dict):
        target = Waypoint(**target)

    console.print(f"  Target: {target.name} ({target.compute_vcpus} vCPU, "
                  f"{target.memory_gb} GB, {target.replicas} replicas)")

    # Plan
    planner = FlightPlanner(use_llm=config.parameters.get("use_llm", False))
    plan = planner.plan(simulator.state, target)

    console.print(f"  Plan: {len(plan.actions)} actions, "
                  f"confidence {plan.confidence:.0%}")
    console.print(f"  Reasoning: {plan.reasoning}")

    # Execute
    autopilot = Autopilot(
        simulator=simulator,
        provisioner=provisioner,
        migrator=migrator,
        health_monitor=health_monitor,
        self_healer=self_healer,
        verbose=True,
    )
    flight_log = autopilot.fly(plan, goal=target)

    # Continue simulation to observe stability
    console.print("\n  [dim]Running stability observation...[/dim]")
    stability_ticks = min(50, config.duration_ticks // 2)
    for _ in range(stability_ticks):
        simulator.step()

    # Compute metrics
    final_state = simulator.state
    final_state.refresh_totals()

    cpu_ratio = final_state.total_vcpus / max(1, target.compute_vcpus)
    mem_ratio = final_state.total_memory_gb / max(0.1, target.memory_gb)
    over_provisioned = cpu_ratio > 1.3 or mem_ratio > 1.3
    under_provisioned = cpu_ratio < 0.7 or mem_ratio < 0.7

    metrics = {
        "success": 1.0 if flight_log.success else 0.0,
        "actions_executed": float(len(flight_log.entries)),
        "replans": float(flight_log.replans),
        "total_cost": flight_log.total_cost,
        "final_vcpus": float(final_state.total_vcpus),
        "final_memory_gb": final_state.total_memory_gb,
        "target_vcpus": float(target.compute_vcpus),
        "target_memory_gb": target.memory_gb,
        "cpu_provision_ratio": round(cpu_ratio, 2),
        "memory_provision_ratio": round(mem_ratio, 2),
        "over_provisioned": 1.0 if over_provisioned else 0.0,
        "under_provisioned": 1.0 if under_provisioned else 0.0,
        "final_cost_per_hour": final_state.total_cost_per_hour,
        "resources_provisioned": float(len(final_state.resources)),
    }

    summary = (
        f"Self-provision {'SUCCEEDED' if flight_log.success else 'FAILED'}. "
        f"Provisioned {final_state.total_vcpus} vCPUs / {final_state.total_memory_gb} GB "
        f"(target: {target.compute_vcpus} / {target.memory_gb}). "
        f"Cost: ${final_state.total_cost_per_hour:.4f}/hr. "
        f"{'Over-provisioned.' if over_provisioned else ''}"
        f"{'Under-provisioned.' if under_provisioned else ''}"
        f"{'Right-sized.' if not over_provisioned and not under_provisioned else ''}"
    )

    console.print(f"\n  [bold]{summary}[/bold]")

    return ExperimentResult(
        config=config,
        success=flight_log.success,
        metrics=metrics,
        flight_log=flight_log,
        summary=summary,
        completed_at=datetime.now(timezone.utc),
    )

#!/usr/bin/env python3
"""Example: run a complete Vimana flight.

This script demonstrates the full self-orchestrating workflow:
1. Initialize a cloud simulation
2. Let the AI plan infrastructure
3. Execute the plan autonomously
4. Run all four experiments
5. Generate a report

Usage:
    python examples/run_vimana_flight.py
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel

from vimana.models import (
    ExperimentConfig,
    Region,
    ResourceSpec,
    ResourceType,
    SimulationScenario,
)
from vimana.cloud.health import HealthMonitor, SelfHealer
from vimana.cloud.migrator import CloudMigrator
from vimana.cloud.provisioner import CloudProvisioner
from vimana.intelligence.demand_predictor import DemandPredictor
from vimana.intelligence.anomaly_detector import AnomalyDetector
from vimana.navigator.autopilot import Autopilot
from vimana.navigator.planner import FlightPlanner
from vimana.navigator.waypoints import WAYPOINT_STANDARD, WAYPOINT_HIGH_AVAILABILITY
from vimana.report import print_experiment_report, print_state_report
from vimana.simulation import CloudSimulator

console = Console()


def main() -> None:
    console.print(Panel(
        "[bold]Vimana: Self-Orchestrating AI Infrastructure[/bold]\n"
        "Demonstrating autonomous provisioning, scaling, healing, and migration.",
        border_style="bright_cyan",
    ))

    # -----------------------------------------------------------------------
    # 1. Initialize simulation
    # -----------------------------------------------------------------------
    console.print("\n[bold cyan]Phase 1: Initializing cloud simulation[/bold cyan]")

    scenario = SimulationScenario(
        name="demo-flight",
        total_ticks=100,
        base_load=0.3,
        load_pattern="burst",
        failure_probability=0.03,
        initial_resources=[
            ResourceSpec(
                resource_type=ResourceType.VM,
                name="seed-vm",
                vcpus=2,
                memory_gb=4.0,
                region=Region.US_EAST_1,
            )
        ],
    )

    sim = CloudSimulator(scenario=scenario)
    sim.initialize()
    print_state_report(sim.state)

    # -----------------------------------------------------------------------
    # 2. AI plans infrastructure
    # -----------------------------------------------------------------------
    console.print("\n[bold cyan]Phase 2: AI planning flight to STANDARD waypoint[/bold cyan]")

    planner = FlightPlanner(use_llm=False)
    plan = planner.plan(sim.state, WAYPOINT_STANDARD)

    console.print(f"  Plan ID: {plan.id[:8]}...")
    console.print(f"  Actions: {len(plan.actions)}")
    console.print(f"  Confidence: {plan.confidence:.0%}")
    console.print(f"  Reasoning: {plan.reasoning}")

    # -----------------------------------------------------------------------
    # 3. Execute autonomously
    # -----------------------------------------------------------------------
    console.print("\n[bold cyan]Phase 3: Autopilot executing plan[/bold cyan]")

    provisioner = CloudProvisioner(simulator=sim)
    migrator = CloudMigrator(simulator=sim)
    health_monitor = HealthMonitor(simulator=sim)
    self_healer = SelfHealer(simulator=sim)

    autopilot = Autopilot(
        simulator=sim,
        provisioner=provisioner,
        migrator=migrator,
        health_monitor=health_monitor,
        self_healer=self_healer,
        verbose=True,
    )

    flight_log = autopilot.fly(plan, goal=WAYPOINT_STANDARD)

    console.print("\n[bold cyan]Post-flight state:[/bold cyan]")
    print_state_report(sim.state)

    # -----------------------------------------------------------------------
    # 4. Run experiments
    # -----------------------------------------------------------------------
    console.print("\n[bold cyan]Phase 4: Running experiments[/bold cyan]")

    from vimana.experiments.self_provision import run_self_provision_experiment
    from vimana.experiments.self_scale import run_self_scale_experiment
    from vimana.experiments.self_heal import run_self_heal_experiment
    from vimana.experiments.self_migrate import run_self_migrate_experiment

    experiments = [
        ("self-provision", run_self_provision_experiment),
        ("self-scale", run_self_scale_experiment),
        ("self-heal", run_self_heal_experiment),
        ("self-migrate", run_self_migrate_experiment),
    ]

    results = []
    for name, runner in experiments:
        config = ExperimentConfig(name=name, duration_ticks=80, seed=42)
        result = runner(config)
        results.append(result)
        print_experiment_report(result)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    console.print(Panel(
        f"[bold]Flight complete.[/bold]\n"
        f"Experiments run: {len(results)}\n"
        f"Successes: {sum(1 for r in results if r.success)}/{len(results)}",
        title="Vimana Flight Summary",
        border_style="bright_green",
    ))


if __name__ == "__main__":
    main()

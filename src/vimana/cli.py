"""CLI entry point for Vimana.

Usage:
    vimana fly --experiment self-provision --duration 100
    vimana simulate --scenario burst-traffic --ticks 200
    vimana optimize --state current
    vimana report --experiment self-provision
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="vimana")
def cli() -> None:
    """Vimana -- Self-Orchestrating AI Infrastructure Framework.

    Can AI agents autonomously provision, scale, and navigate their own
    cloud infrastructure?
    """
    pass


# ---------------------------------------------------------------------------
# fly: run an experiment
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--experiment", "-e", type=click.Choice([
    "self-provision", "self-scale", "self-heal", "self-migrate",
]), required=True, help="Which experiment to run.")
@click.option("--duration", "-d", type=int, default=100, help="Duration in ticks.")
@click.option("--seed", type=int, default=42, help="Random seed.")
@click.option("--use-llm", is_flag=True, default=False, help="Enable LLM-based planning.")
@click.option("--output", "-o", type=str, default=None, help="Save results to JSON file.")
def fly(experiment: str, duration: int, seed: int, use_llm: bool, output: str | None) -> None:
    """Run a Vimana experiment (autonomous infrastructure flight)."""
    from vimana.models import ExperimentConfig
    from vimana.report import print_experiment_report, save_results_json

    config = ExperimentConfig(
        name=experiment,
        duration_ticks=duration,
        seed=seed,
        parameters={"use_llm": use_llm},
    )

    console.print(f"\n[bold cyan]Vimana: launching {experiment} experiment...[/bold cyan]")

    if experiment == "self-provision":
        from vimana.experiments.self_provision import run_self_provision_experiment
        result = run_self_provision_experiment(config)
    elif experiment == "self-scale":
        from vimana.experiments.self_scale import run_self_scale_experiment
        result = run_self_scale_experiment(config)
    elif experiment == "self-heal":
        from vimana.experiments.self_heal import run_self_heal_experiment
        result = run_self_heal_experiment(config)
    elif experiment == "self-migrate":
        from vimana.experiments.self_migrate import run_self_migrate_experiment
        result = run_self_migrate_experiment(config)
    else:
        console.print(f"[red]Unknown experiment: {experiment}[/red]")
        sys.exit(1)

    print_experiment_report(result)

    if output:
        save_results_json(result, output)


# ---------------------------------------------------------------------------
# simulate: run a cloud simulation
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--scenario", "-s", type=click.Choice([
    "steady", "burst-traffic", "cyclic", "ramp", "random",
]), default="steady", help="Load scenario to simulate.")
@click.option("--ticks", "-t", type=int, default=200, help="Number of simulation ticks.")
@click.option("--resources", "-r", type=int, default=3, help="Number of initial resources.")
@click.option("--failure-rate", type=float, default=0.02, help="Failure injection probability.")
def simulate(scenario: str, ticks: int, resources: int, failure_rate: float) -> None:
    """Run a cloud environment simulation."""
    from vimana.models import Region, ResourceSpec, ResourceType, SimulationScenario
    from vimana.report import print_state_report
    from vimana.simulation import CloudSimulator

    pattern_map = {
        "steady": "steady",
        "burst-traffic": "burst",
        "cyclic": "cyclic",
        "ramp": "ramp",
        "random": "random",
    }

    sim_scenario = SimulationScenario(
        name=scenario,
        description=f"Simulated {scenario} scenario",
        total_ticks=ticks,
        base_load=0.3,
        load_pattern=pattern_map.get(scenario, "steady"),
        failure_probability=failure_rate,
        initial_resources=[
            ResourceSpec(
                resource_type=ResourceType.VM,
                name=f"sim-vm-{i}",
                vcpus=4,
                memory_gb=8.0,
                region=Region.US_EAST_1,
            )
            for i in range(resources)
        ],
    )

    simulator = CloudSimulator(scenario=sim_scenario)
    simulator.initialize()

    console.print(f"\n[bold cyan]Vimana: simulating {scenario} "
                  f"({ticks} ticks, {resources} resources)[/bold cyan]\n")

    for tick in range(ticks):
        simulator.step()
        if (tick + 1) % (ticks // 5) == 0:
            agg = simulator.aggregate_metrics()
            console.print(
                f"  tick {tick + 1:>4}/{ticks} | "
                f"CPU: {agg.get('avg_cpu', 0):5.1f}% | "
                f"Mem: {agg.get('avg_memory', 0):5.1f}% | "
                f"Latency: {agg.get('avg_latency_ms', 0):6.1f}ms | "
                f"Cost: ${agg.get('total_cost_per_hour', 0):.4f}/hr"
            )

    console.print()
    print_state_report(simulator.state)

    failures = simulator.failure_log
    if failures:
        console.print(f"\n  [yellow]{len(failures)} failures injected during simulation.[/yellow]")


# ---------------------------------------------------------------------------
# optimize: run cost optimization on current state
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--resources", "-r", type=int, default=5, help="Number of simulated resources.")
@click.option("--region", type=str, default="us-east-1", help="Current region.")
def optimize(resources: int, region: str) -> None:
    """Run AI cost optimization on a simulated infrastructure state."""
    from vimana.intelligence.cost_optimizer import CostOptimizer
    from vimana.models import Region, ResourceSpec, ResourceType, SimulationScenario
    from vimana.simulation import CloudSimulator

    region_enum = Region(region)

    scenario = SimulationScenario(
        name="optimize",
        total_ticks=30,
        base_load=0.3,
        load_pattern="steady",
        failure_probability=0.0,
        initial_resources=[
            ResourceSpec(
                resource_type=ResourceType.VM,
                name=f"opt-vm-{i}",
                vcpus=4 + (i % 3) * 2,
                memory_gb=8.0 + (i % 2) * 8,
                region=region_enum,
                replicas=1 + i % 3,
            )
            for i in range(resources)
        ],
    )

    sim = CloudSimulator(scenario=scenario)
    sim.initialize()

    # Run a few ticks to get metrics
    for _ in range(20):
        sim.step()

    optimizer = CostOptimizer()
    result = optimizer.optimize(sim.state)

    console.print(f"\n[bold cyan]Vimana: Cost Optimization Report[/bold cyan]\n")
    console.print(f"  Current cost:   ${result.original_cost_per_hour:.4f}/hr")
    console.print(f"  Optimized cost: ${result.optimized_cost_per_hour:.4f}/hr")
    console.print(f"  Savings:        {result.savings_percent:.1f}%\n")

    if result.recommendations:
        from rich.table import Table
        table = Table(title="Recommendations", show_header=True, header_style="bold green")
        table.add_column("Action")
        table.add_column("Resource")
        table.add_column("Savings $/hr", justify="right")
        table.add_column("Reason")

        for rec in result.recommendations:
            table.add_row(
                rec.action_type.value,
                (rec.target_resource_id or "")[:10],
                f"${abs(rec.estimated_cost_delta):.4f}",
                rec.reason[:80],
            )
        console.print(table)
    else:
        console.print("  [green]No optimizations found -- infrastructure is already efficient.[/green]")


# ---------------------------------------------------------------------------
# report: display results from a saved JSON file
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--input", "-i", "input_path", type=str, required=True,
              help="Path to a saved experiment results JSON.")
@click.option("--plots", is_flag=True, default=False, help="Generate plots.")
@click.option("--plot-dir", type=str, default="./plots", help="Directory for plots.")
def report(input_path: str, plots: bool, plot_dir: str) -> None:
    """Display a report from saved experiment results."""
    from vimana.models import ExperimentResult
    from vimana.report import generate_plots, print_experiment_report

    path = Path(input_path)
    if not path.exists():
        console.print(f"[red]File not found: {path}[/red]")
        sys.exit(1)

    data = json.loads(path.read_text())
    result = ExperimentResult(**data)
    print_experiment_report(result)

    if plots:
        generate_plots(result, plot_dir)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()

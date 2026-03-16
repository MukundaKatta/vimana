"""Report generation for Vimana experiments.

Produces rich terminal reports and optional JSON/matplotlib output
summarizing experiment results and infrastructure telemetry.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from vimana.models import ExperimentResult, FlightLog, InfrastructureState

console = Console()


def print_experiment_report(result: ExperimentResult) -> None:
    """Print a rich terminal report for an experiment result."""
    console.print()
    console.print(Panel(
        f"[bold]{result.config.name}[/bold]\n{result.summary}",
        title="Vimana Experiment Report",
        border_style="cyan",
    ))

    # Metrics table
    if result.metrics:
        table = Table(title="Metrics", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="dim")
        table.add_column("Value", justify="right")

        for key, value in sorted(result.metrics.items()):
            formatted = _format_metric(key, value)
            table.add_row(key, formatted)

        console.print(table)

    # Flight log summary
    if result.flight_log:
        _print_flight_summary(result.flight_log)

    # Timeline (abbreviated)
    if result.timeline:
        _print_timeline_summary(result.timeline)

    console.print()


def print_state_report(state: InfrastructureState) -> None:
    """Print a summary of the current infrastructure state."""
    state.refresh_totals()

    table = Table(title="Infrastructure State", show_header=True, header_style="bold green")
    table.add_column("Resource ID", style="dim")
    table.add_column("Type")
    table.add_column("Status")
    table.add_column("Region")
    table.add_column("vCPU", justify="right")
    table.add_column("Memory GB", justify="right")
    table.add_column("Replicas", justify="right")
    table.add_column("$/hr", justify="right")

    for r in state.resources:
        status_style = {
            "running": "green",
            "degraded": "yellow",
            "failed": "red",
            "terminated": "dim",
        }.get(r.status.value, "white")

        table.add_row(
            r.id[:10],
            r.spec.resource_type.value,
            f"[{status_style}]{r.status.value}[/{status_style}]",
            r.spec.region.value,
            str(r.spec.vcpus),
            f"{r.spec.memory_gb:.1f}",
            str(r.spec.replicas),
            f"${r.cost_per_hour:.4f}",
        )

    console.print(table)
    console.print(
        f"  Totals: {state.total_vcpus} vCPUs, "
        f"{state.total_memory_gb:.1f} GB, "
        f"${state.total_cost_per_hour:.4f}/hr, "
        f"{len(state.regions_active)} region(s)"
    )


def _print_flight_summary(log: FlightLog) -> None:
    """Print a summary of a flight log."""
    table = Table(title="Flight Log", show_header=True, header_style="bold yellow")
    table.add_column("Tick", justify="right")
    table.add_column("Action")
    table.add_column("Result")
    table.add_column("Message")

    for entry in log.entries[:20]:  # Show first 20
        result_style = "green" if entry.result.success else "red"
        table.add_row(
            str(entry.tick),
            entry.action.action_type.value,
            f"[{result_style}]{'OK' if entry.result.success else 'FAIL'}[/{result_style}]",
            (entry.result.message or entry.result.error or "")[:60],
        )

    if len(log.entries) > 20:
        table.add_row("...", f"+{len(log.entries) - 20} more", "", "")

    console.print(table)


def _print_timeline_summary(timeline: list[dict[str, Any]]) -> None:
    """Print an abbreviated timeline."""
    if not timeline:
        return

    console.print("\n[bold]Timeline (sampled):[/bold]")
    step = max(1, len(timeline) // 10)
    for entry in timeline[::step]:
        parts = [f"tick={entry.get('tick', '?')}"]
        for k, v in entry.items():
            if k != "tick":
                if isinstance(v, float):
                    parts.append(f"{k}={v:.2f}")
                else:
                    parts.append(f"{k}={v}")
        console.print(f"  {' | '.join(parts)}")


def _format_metric(key: str, value: float) -> str:
    """Format a metric value for display."""
    if "cost" in key or "price" in key or key.startswith("$"):
        return f"${value:.4f}"
    if "rate" in key or "ratio" in key or "percent" in key:
        return f"{value:.2%}" if value <= 1.0 else f"{value:.1f}%"
    if isinstance(value, float) and value == int(value):
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def save_results_json(result: ExperimentResult, path: str | Path) -> None:
    """Save experiment results to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = result.model_dump(mode="json")
    path.write_text(json.dumps(data, indent=2, default=str))
    console.print(f"[dim]Results saved to {path}[/dim]")


def generate_plots(result: ExperimentResult, output_dir: str | Path) -> None:
    """Generate matplotlib plots from experiment timeline data."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        console.print("[yellow]matplotlib not available; skipping plots.[/yellow]")
        return

    if not result.timeline:
        console.print("[dim]No timeline data to plot.[/dim]")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ticks = [e.get("tick", i) for i, e in enumerate(result.timeline)]

    # Plot each numeric field
    numeric_fields = set()
    for entry in result.timeline:
        for k, v in entry.items():
            if k != "tick" and isinstance(v, (int, float)):
                numeric_fields.add(k)

    if not numeric_fields:
        return

    fig, axes = plt.subplots(
        len(numeric_fields), 1,
        figsize=(12, 3 * len(numeric_fields)),
        sharex=True,
    )
    if len(numeric_fields) == 1:
        axes = [axes]

    for ax, field_name in zip(axes, sorted(numeric_fields)):
        values = [e.get(field_name, 0) for e in result.timeline]
        ax.plot(ticks, values, linewidth=1.5)
        ax.set_ylabel(field_name)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Tick")
    fig.suptitle(f"Vimana: {result.config.name}", fontsize=14)
    fig.tight_layout()

    out_path = output_dir / f"{result.config.name}_timeline.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    console.print(f"[dim]Plot saved to {out_path}[/dim]")

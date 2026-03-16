"""Experiment: Can AI scale itself optimally?

Simulates variable load and compares three scaling strategies:
  - AI (predictive) scaling
  - Static rules (fixed thresholds)
  - Reactive thresholds (respond after breach)

Metrics: cost efficiency, SLA violations, response time.
"""

from __future__ import annotations

from datetime import datetime, timezone

from rich.console import Console

from vimana.cloud.health import HealthMonitor, SelfHealer
from vimana.cloud.migrator import CloudMigrator
from vimana.cloud.provisioner import CloudProvisioner
from vimana.cloud.scaler import AutoScaler, ScalingThresholds
from vimana.intelligence.demand_predictor import DemandPredictor
from vimana.models import (
    ExperimentConfig,
    ExperimentResult,
    Region,
    ResourceSpec,
    ResourceType,
    ScalingStrategy,
    SimulationScenario,
)
from vimana.simulation import CloudSimulator

console = Console()


def _run_scaling_trial(
    strategy: ScalingStrategy,
    config: ExperimentConfig,
) -> dict[str, float]:
    """Run a single scaling trial with the given strategy."""
    scenario = SimulationScenario(
        name=f"self-scale-{strategy.value}",
        total_ticks=config.duration_ticks,
        base_load=0.3,
        load_pattern=config.parameters.get("load_pattern", "burst"),
        failure_probability=0.01,
        region=Region.US_EAST_1,
        initial_resources=[
            ResourceSpec(
                resource_type=ResourceType.VM,
                name="initial-vm",
                vcpus=4,
                memory_gb=8.0,
                region=Region.US_EAST_1,
                replicas=2,
            )
        ],
        parameters={"seed": config.seed},
    )

    sim = CloudSimulator(scenario=scenario)
    sim.initialize()
    provisioner = CloudProvisioner(simulator=sim)
    predictor = DemandPredictor()

    scaler = AutoScaler(
        provisioner=provisioner,
        strategy=strategy,
        thresholds=ScalingThresholds(cooldown_ticks=3),
    )

    total_cost = 0.0
    sla_violations = 0
    scale_actions = 0
    timeline: list[dict] = []

    for tick in range(config.duration_ticks):
        state = sim.step()
        agg = sim.aggregate_metrics()
        if not agg:
            continue

        total_cost += agg.get("total_cost_per_hour", 0) / 60  # per-tick cost

        # SLA violation: latency > 500ms or error_rate > 5%
        if agg.get("avg_latency_ms", 0) > 500 or agg.get("avg_error_rate", 0) > 0.05:
            sla_violations += 1

        # Feed predictor
        predictor.observe(
            agg.get("avg_cpu", 0),
            agg.get("avg_memory", 0),
            agg.get("total_rps", 0),
        )

        # Evaluate scaling
        forecast = predictor.predict(horizon=10) if strategy == ScalingStrategy.PREDICTIVE else None
        decision = scaler.evaluate(agg, state=state, forecast=forecast)

        if decision.direction.value != "maintain":
            running = sim.get_running_resources()
            if running:
                scaler.apply(decision, running[0].id)
                scale_actions += 1

        timeline.append({
            "tick": tick,
            "cpu": agg.get("avg_cpu", 0),
            "cost": agg.get("total_cost_per_hour", 0),
            "latency": agg.get("avg_latency_ms", 0),
        })

    return {
        "total_cost": round(total_cost, 4),
        "sla_violations": float(sla_violations),
        "sla_violation_rate": round(sla_violations / max(1, config.duration_ticks), 4),
        "scale_actions": float(scale_actions),
        "avg_cost_per_tick": round(total_cost / max(1, config.duration_ticks), 6),
    }


def run_self_scale_experiment(
    config: ExperimentConfig | None = None,
) -> ExperimentResult:
    """Run the self-scaling experiment, comparing all three strategies."""
    config = config or ExperimentConfig(
        name="self-scale",
        duration_ticks=200,
        seed=42,
        parameters={"load_pattern": "burst"},
    )

    console.print("\n[bold magenta]=== Experiment: Self-Scale ===[/bold magenta]")
    console.print(f"  Duration: {config.duration_ticks} ticks | "
                  f"Pattern: {config.parameters.get('load_pattern', 'burst')}")

    results: dict[str, dict[str, float]] = {}
    for strategy in ScalingStrategy:
        console.print(f"\n  [cyan]Running trial: {strategy.value}...[/cyan]")
        trial = _run_scaling_trial(strategy, config)
        results[strategy.value] = trial
        console.print(
            f"    Cost: ${trial['total_cost']:.4f} | "
            f"SLA violations: {trial['sla_violations']:.0f} "
            f"({trial['sla_violation_rate']:.1%}) | "
            f"Scale actions: {trial['scale_actions']:.0f}"
        )

    # Aggregate metrics
    all_metrics: dict[str, float] = {}
    for strategy_name, trial_metrics in results.items():
        for k, v in trial_metrics.items():
            all_metrics[f"{strategy_name}_{k}"] = v

    # Determine winner
    best_strategy = min(
        results.items(),
        key=lambda kv: kv[1]["total_cost"] + kv[1]["sla_violations"] * 10,
    )
    all_metrics["best_strategy_is_predictive"] = 1.0 if best_strategy[0] == "predictive" else 0.0

    summary = (
        f"Scaling comparison complete. "
        f"Best strategy: {best_strategy[0]} "
        f"(cost=${best_strategy[1]['total_cost']:.4f}, "
        f"SLA violations={best_strategy[1]['sla_violations']:.0f}). "
        f"Predictive {'won' if best_strategy[0] == 'predictive' else 'did not win'}."
    )

    console.print(f"\n  [bold]{summary}[/bold]")

    return ExperimentResult(
        config=config,
        success=True,
        metrics=all_metrics,
        summary=summary,
        completed_at=datetime.now(timezone.utc),
    )

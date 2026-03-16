"""Experiment: Can AI recover from failures?

Injects random infrastructure failures and measures the AI's ability
to detect, diagnose, and remediate them without human intervention.
"""

from __future__ import annotations

from datetime import datetime, timezone

from rich.console import Console

from vimana.cloud.health import HealthMonitor, SelfHealer
from vimana.cloud.migrator import CloudMigrator
from vimana.cloud.provisioner import CloudProvisioner
from vimana.intelligence.anomaly_detector import AnomalyDetector
from vimana.models import (
    ExperimentConfig,
    ExperimentResult,
    FailureType,
    Region,
    ResourceSpec,
    ResourceStatus,
    ResourceType,
    SimulationScenario,
)
from vimana.simulation import CloudSimulator

console = Console()


def run_self_heal_experiment(
    config: ExperimentConfig | None = None,
) -> ExperimentResult:
    """Run the self-healing experiment.

    Sets up a healthy cluster, then injects failures at random intervals.
    Measures detection time, recovery time, and success rate.
    """
    config = config or ExperimentConfig(
        name="self-heal",
        duration_ticks=150,
        seed=42,
        parameters={"failure_interval": 10},
    )

    console.print("\n[bold magenta]=== Experiment: Self-Heal ===[/bold magenta]")
    console.print(f"  Duration: {config.duration_ticks} ticks | Seed: {config.seed}")

    # Setup: healthy cluster
    scenario = SimulationScenario(
        name="self-heal",
        total_ticks=config.duration_ticks,
        base_load=0.5,
        load_pattern="steady",
        failure_probability=0.0,  # we inject manually
        region=Region.US_EAST_1,
        initial_resources=[
            ResourceSpec(
                resource_type=ResourceType.VM,
                name=f"node-{i}",
                vcpus=4,
                memory_gb=8.0,
                region=Region.US_EAST_1,
                replicas=1,
            )
            for i in range(4)
        ],
        parameters={"seed": config.seed},
    )

    sim = CloudSimulator(scenario=scenario)
    sim.initialize()

    health_monitor = HealthMonitor(simulator=sim)
    self_healer = SelfHealer(simulator=sim)
    anomaly_detector = AnomalyDetector()

    failure_interval = config.parameters.get("failure_interval", 10)
    failure_types = [
        FailureType.INSTANCE_CRASH,
        FailureType.MEMORY_LEAK,
        FailureType.DISK_FULL,
        FailureType.NETWORK_PARTITION,
        FailureType.CPU_SPIKE,
    ]

    injections = 0
    detections = 0
    heals_attempted = 0
    heals_succeeded = 0
    total_recovery_ticks = 0
    timeline: list[dict] = []

    for tick in range(config.duration_ticks):
        state = sim.step()

        # Observe metrics for anomaly detection
        for r in sim.get_running_resources():
            if r.metrics:
                anomaly_detector.observe(r.id, r.metrics)

        # Inject failure periodically
        if tick > 0 and tick % failure_interval == 0:
            running = sim.get_running_resources()
            if running:
                target = running[tick % len(running)]
                ft = failure_types[injections % len(failure_types)]
                sim.inject_failure(target.id, ft)
                injections += 1
                console.print(
                    f"  [red]tick {tick}: Injected {ft.value} on {target.id}[/red]"
                )

        # Detect and heal
        checks = health_monitor.check_all()
        unhealthy = [c for c in checks if not c.healthy]

        for check in unhealthy:
            detections += 1
            diag = health_monitor.diagnose(check.symptoms)
            diag.resource_id = check.resource_id
            heals_attempted += 1

            result = self_healer.heal(diag)
            if result.success:
                heals_succeeded += 1
                console.print(
                    f"  [green]tick {tick}: Healed {check.resource_id} "
                    f"({diag.failure_type.value})[/green]"
                )
            else:
                console.print(
                    f"  [yellow]tick {tick}: Failed to heal {check.resource_id}: "
                    f"{result.message}[/yellow]"
                )

        # Detect anomalies
        for r in state.resources:
            if r.metrics and r.status == ResourceStatus.RUNNING:
                anomalies = anomaly_detector.detect(r.id, r.metrics)
                for a in anomalies:
                    if a.severity.value in ("high", "critical"):
                        console.print(
                            f"  [yellow]tick {tick}: Anomaly on {r.id}: "
                            f"{a.description}[/yellow]"
                        )

        timeline.append({
            "tick": tick,
            "running": len(sim.get_running_resources()),
            "unhealthy": len(unhealthy),
            "total_injections": injections,
            "total_heals": heals_succeeded,
        })

    # Metrics
    heal_rate = heals_succeeded / max(1, heals_attempted)
    detection_rate = detections / max(1, injections)

    metrics = {
        "failures_injected": float(injections),
        "failures_detected": float(detections),
        "detection_rate": round(detection_rate, 4),
        "heals_attempted": float(heals_attempted),
        "heals_succeeded": float(heals_succeeded),
        "heal_success_rate": round(heal_rate, 4),
        "final_running_resources": float(len(sim.get_running_resources())),
        "initial_resources": 4.0,
    }

    summary = (
        f"Self-heal experiment complete. "
        f"Injected {injections} failures, detected {detections}, "
        f"healed {heals_succeeded}/{heals_attempted} "
        f"({heal_rate:.0%} success rate). "
        f"Final cluster: {len(sim.get_running_resources())} running resources."
    )

    console.print(f"\n  [bold]{summary}[/bold]")

    return ExperimentResult(
        config=config,
        success=heal_rate >= 0.5,
        metrics=metrics,
        timeline=timeline,
        summary=summary,
        completed_at=datetime.now(timezone.utc),
    )

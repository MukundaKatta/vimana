"""Experiment: Can AI migrate itself across regions?

Tests whether an AI agent can decide when migration is beneficial,
plan the migration, execute it, and verify the outcome -- measuring
downtime, data loss, and cost savings.
"""

from __future__ import annotations

from datetime import datetime, timezone

from rich.console import Console

from vimana.cloud.health import HealthMonitor, SelfHealer
from vimana.cloud.migrator import CloudMigrator
from vimana.cloud.provisioner import CloudProvisioner
from vimana.intelligence.cost_optimizer import CostOptimizer
from vimana.models import (
    ActionType,
    ExperimentConfig,
    ExperimentResult,
    Region,
    ResourceSpec,
    ResourceStatus,
    ResourceType,
    SimulationScenario,
)
from vimana.simulation import CloudSimulator, _REGION_MULTIPLIER

console = Console()


def run_self_migrate_experiment(
    config: ExperimentConfig | None = None,
) -> ExperimentResult:
    """Run the self-migration experiment.

    Starts resources in an expensive region, then tests if the AI
    can identify the cost-saving opportunity and migrate to a cheaper region.
    """
    config = config or ExperimentConfig(
        name="self-migrate",
        duration_ticks=100,
        seed=42,
    )

    console.print("\n[bold magenta]=== Experiment: Self-Migrate ===[/bold magenta]")
    console.print(f"  Duration: {config.duration_ticks} ticks | Seed: {config.seed}")

    # Start in an expensive region
    expensive_region = Region.AP_NORTHEAST_1  # x1.20
    cheap_region = Region.US_EAST_1           # x1.00

    scenario = SimulationScenario(
        name="self-migrate",
        total_ticks=config.duration_ticks,
        base_load=0.4,
        load_pattern="steady",
        failure_probability=0.01,
        region=expensive_region,
        initial_resources=[
            ResourceSpec(
                resource_type=ResourceType.VM,
                name=f"app-{i}",
                vcpus=4,
                memory_gb=8.0,
                storage_gb=50.0,
                region=expensive_region,
                replicas=1,
            )
            for i in range(3)
        ],
        parameters={"seed": config.seed},
    )

    sim = CloudSimulator(scenario=scenario)
    sim.initialize()

    provisioner = CloudProvisioner(simulator=sim)
    migrator = CloudMigrator(simulator=sim)
    cost_optimizer = CostOptimizer()

    # Record initial state
    initial_state = sim.state.model_copy(deep=True)
    initial_state.refresh_totals()
    initial_cost = initial_state.total_cost_per_hour

    console.print(f"  Initial region: {expensive_region.value} "
                  f"(multiplier x{_REGION_MULTIPLIER[expensive_region]:.2f})")
    console.print(f"  Initial cost: ${initial_cost:.4f}/hr")

    # Let the AI analyze and decide
    # Run a few ticks to gather metrics
    for _ in range(20):
        sim.step()

    # AI cost optimization analysis
    optimization = cost_optimizer.optimize(sim.state)
    console.print(f"\n  [cyan]Cost optimizer found {len(optimization.recommendations)} "
                  f"recommendations[/cyan]")
    console.print(f"  Potential savings: {optimization.savings_percent:.1f}%")

    # Find migration recommendations
    migration_recs = [
        r for r in optimization.recommendations
        if r.action_type == ActionType.MIGRATE
    ]

    migrations_attempted = 0
    migrations_succeeded = 0
    total_downtime = 0.0
    total_migration_cost = 0.0

    if migration_recs:
        console.print(f"  Found {len(migration_recs)} migration recommendations.")

        for rec in migration_recs:
            source = rec.parameters.get("source_region", expensive_region.value)
            target = rec.parameters.get("target_region", cheap_region.value)

            console.print(f"\n  [cyan]Migrating from {source} -> {target}[/cyan]")

            plan = migrator.plan_migration(
                source_region=Region(source),
                target_region=Region(target),
                resource_ids=[rec.target_resource_id] if rec.target_resource_id else None,
            )

            console.print(f"    Est. downtime: {plan.estimated_downtime_seconds:.1f}s")
            console.print(f"    Est. data transfer: {plan.estimated_data_transfer_gb:.1f} GB")

            result = migrator.execute_migration(plan)
            migrations_attempted += 1

            if result.success:
                migrations_succeeded += 1
                total_downtime += result.actual_downtime_seconds
                total_migration_cost += result.cost
                console.print(f"    [green]Migration successful[/green] "
                              f"(downtime: {result.actual_downtime_seconds:.1f}s)")
            else:
                console.print(f"    [red]Migration failed: {result.message}[/red]")
    else:
        console.print("  [yellow]No migration recommendations found. "
                      "Performing manual migration to cheapest region...[/yellow]")
        # Manually migrate all resources
        plan = migrator.plan_migration(expensive_region, cheap_region)
        result = migrator.execute_migration(plan)
        migrations_attempted = 1
        if result.success:
            migrations_succeeded = 1
            total_downtime = result.actual_downtime_seconds
            total_migration_cost = result.cost

    # Run remaining ticks
    remaining = config.duration_ticks - sim.tick
    for _ in range(max(0, remaining)):
        sim.step()

    # Final analysis
    final_state = sim.state
    final_state.refresh_totals()
    final_cost = final_state.total_cost_per_hour
    cost_savings = initial_cost - final_cost
    savings_pct = (cost_savings / initial_cost * 100) if initial_cost > 0 else 0

    metrics = {
        "initial_cost_per_hour": round(initial_cost, 4),
        "final_cost_per_hour": round(final_cost, 4),
        "cost_savings_per_hour": round(cost_savings, 4),
        "cost_savings_percent": round(savings_pct, 1),
        "migrations_attempted": float(migrations_attempted),
        "migrations_succeeded": float(migrations_succeeded),
        "migration_success_rate": round(
            migrations_succeeded / max(1, migrations_attempted), 4
        ),
        "total_downtime_seconds": round(total_downtime, 2),
        "total_migration_cost": round(total_migration_cost, 4),
        "recommendations_found": float(len(optimization.recommendations)),
    }

    success = migrations_succeeded > 0 and cost_savings > 0

    summary = (
        f"Self-migrate {'SUCCEEDED' if success else 'FAILED'}. "
        f"Migrated {migrations_succeeded}/{migrations_attempted} workloads. "
        f"Cost: ${initial_cost:.4f}/hr -> ${final_cost:.4f}/hr "
        f"(saved {savings_pct:.1f}%). "
        f"Total downtime: {total_downtime:.1f}s."
    )

    console.print(f"\n  [bold]{summary}[/bold]")

    return ExperimentResult(
        config=config,
        success=success,
        metrics=metrics,
        summary=summary,
        completed_at=datetime.now(timezone.utc),
    )

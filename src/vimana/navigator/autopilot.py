"""Autonomous execution of infrastructure plans.

The Autopilot takes an InfrastructurePlan and executes it against a
simulated cloud environment, monitoring progress and re-planning on failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from rich.console import Console

from vimana.cloud.health import HealthMonitor, SelfHealer
from vimana.cloud.migrator import CloudMigrator
from vimana.cloud.provisioner import CloudProvisioner
from vimana.models import (
    ActionResult,
    ActionType,
    FlightLog,
    FlightLogEntry,
    InfraAction,
    InfrastructurePlan,
    InfrastructureState,
    ResourceSpec,
    ResourceStatus,
    Waypoint,
)
from vimana.simulation import CloudSimulator

console = Console()


@dataclass
class Autopilot:
    """Executes infrastructure plans autonomously.

    Runs actions in order, advances the simulation between actions,
    monitors health, and triggers re-planning when actions fail.
    """

    simulator: CloudSimulator
    provisioner: CloudProvisioner
    migrator: CloudMigrator
    health_monitor: HealthMonitor
    self_healer: SelfHealer
    max_replans: int = 3
    dry_run: bool = False
    verbose: bool = True
    _planner: Any = None  # set lazily to avoid circular import

    @property
    def planner(self) -> Any:
        if self._planner is None:
            from vimana.navigator.planner import FlightPlanner
            self._planner = FlightPlanner(use_llm=False)
        return self._planner

    # -- main entry point ---------------------------------------------------

    def fly(
        self,
        plan: InfrastructurePlan,
        goal: Waypoint | None = None,
    ) -> FlightLog:
        """Execute the full plan against the cloud environment.

        Args:
            plan: The infrastructure plan to execute.
            goal: Optional target waypoint for re-planning on failure.

        Returns:
            A FlightLog recording every action and outcome.
        """
        flight_log = FlightLog(
            plan_id=plan.id,
            dry_run=self.dry_run,
        )

        if self.verbose:
            console.print(
                f"\n[bold cyan]>>> Vimana Autopilot: executing plan "
                f"{plan.id[:8]}... ({len(plan.actions)} actions)[/bold cyan]"
            )
            if self.dry_run:
                console.print("[yellow]  DRY-RUN mode -- no changes applied[/yellow]")

        remaining_actions = list(plan.actions)
        replans = 0

        while remaining_actions:
            action = remaining_actions.pop(0)

            if self.verbose:
                console.print(
                    f"  [dim]tick {self.simulator.tick}[/dim] "
                    f"[bold]{action.action_type.value}[/bold] "
                    f"{action.reason}"
                )

            if self.dry_run:
                result = ActionResult(
                    action_id=action.id,
                    success=True,
                    message=f"[dry-run] Would {action.action_type.value}.",
                )
            else:
                result = self._execute_action(action)

            # Advance simulation
            self.simulator.step()

            entry = FlightLogEntry(
                tick=self.simulator.tick,
                action=action,
                result=result,
                state_after=self.simulator.state.model_copy(deep=True),
            )
            flight_log.entries.append(entry)

            if not result.success:
                if self.verbose:
                    console.print(
                        f"    [red]FAILED: {result.error or result.message}[/red]"
                    )

                # Attempt self-healing for affected resources
                if action.target_resource_id:
                    self._attempt_heal(action.target_resource_id)

                # Re-plan if we have a goal and budget
                if goal and replans < self.max_replans:
                    replans += 1
                    if self.verbose:
                        console.print(
                            f"    [yellow]Re-planning (attempt {replans}/{self.max_replans})...[/yellow]"
                        )
                    new_plan = self.planner.plan(
                        self.simulator.state, goal
                    )
                    remaining_actions = list(new_plan.actions) + remaining_actions
                    flight_log.replans = replans
            else:
                if self.verbose and not self.dry_run:
                    console.print(f"    [green]OK[/green] {result.message}")

            # Run health check every 5 ticks
            if self.simulator.tick % 5 == 0 and not self.dry_run:
                self._run_health_sweep()

            flight_log.total_cost += result.actual_cost_delta

        flight_log.completed_at = datetime.now(timezone.utc)
        flight_log.success = all(e.result.success for e in flight_log.entries)

        if self.verbose:
            status = "[green]SUCCESS[/green]" if flight_log.success else "[red]PARTIAL FAILURE[/red]"
            console.print(
                f"\n[bold cyan]>>> Flight complete: {status} "
                f"({len(flight_log.entries)} actions, "
                f"{flight_log.replans} replans, "
                f"${flight_log.total_cost:.4f} cost delta)[/bold cyan]\n"
            )

        return flight_log

    # -- action dispatch ----------------------------------------------------

    def _execute_action(self, action: InfraAction) -> ActionResult:
        """Dispatch a single action to the appropriate cloud service."""
        try:
            if action.action_type == ActionType.PROVISION:
                return self._do_provision(action)
            elif action.action_type in (ActionType.SCALE_UP, ActionType.SCALE_DOWN):
                return self._do_scale(action)
            elif action.action_type == ActionType.TERMINATE:
                return self._do_terminate(action)
            elif action.action_type == ActionType.HEAL:
                return self._do_heal(action)
            elif action.action_type == ActionType.MIGRATE:
                return self._do_migrate(action)
            elif action.action_type == ActionType.RECONFIGURE:
                return ActionResult(
                    action_id=action.id,
                    success=True,
                    message="Reconfiguration noted (no-op in simulation).",
                )
            else:
                return ActionResult(
                    action_id=action.id,
                    success=False,
                    error=f"Unknown action type: {action.action_type}",
                )
        except Exception as exc:
            return ActionResult(
                action_id=action.id,
                success=False,
                error=str(exc),
            )

    def _do_provision(self, action: InfraAction) -> ActionResult:
        spec = action.resource_spec
        if spec is None:
            return ActionResult(
                action_id=action.id,
                success=False,
                error="No resource spec provided for provision action.",
            )
        result = self.provisioner.provision(spec)
        return ActionResult(
            action_id=action.id,
            success=result.success,
            resource_id=result.resource.id if result.resource else None,
            actual_cost_delta=result.cost_per_hour,
            duration_seconds=result.provision_latency_seconds,
            message=result.message,
            error=None if result.success else result.message,
        )

    def _do_scale(self, action: InfraAction) -> ActionResult:
        rid = action.target_resource_id
        if rid is None:
            return ActionResult(action_id=action.id, success=False, error="No target resource ID.")

        delta = action.parameters.get("replica_delta", 1 if action.action_type == ActionType.SCALE_UP else -1)
        resource = self.provisioner.get_resource(rid)
        if resource is None:
            return ActionResult(action_id=action.id, success=False, error=f"Resource {rid} not found.")
        new_replicas = max(0, resource.spec.replicas + delta)
        return self.provisioner.scale_replicas(rid, new_replicas)

    def _do_terminate(self, action: InfraAction) -> ActionResult:
        rid = action.target_resource_id
        if rid is None:
            return ActionResult(action_id=action.id, success=False, error="No target resource ID.")
        ok = self.provisioner.terminate(rid)
        return ActionResult(
            action_id=action.id,
            success=ok,
            message=f"Terminated {rid}." if ok else f"Failed to terminate {rid}.",
        )

    def _do_heal(self, action: InfraAction) -> ActionResult:
        rid = action.target_resource_id
        if rid is None:
            return ActionResult(action_id=action.id, success=False, error="No target resource ID.")

        check = self.health_monitor.check(
            self.simulator.get_resource(rid)  # type: ignore[arg-type]
        )
        if check.healthy:
            return ActionResult(action_id=action.id, success=True, message=f"{rid} is already healthy.")

        diag = self.health_monitor.diagnose(check.symptoms)
        diag.resource_id = rid
        heal_result = self.self_healer.heal(diag)
        return ActionResult(
            action_id=action.id,
            success=heal_result.success,
            resource_id=heal_result.resource_id,
            duration_seconds=heal_result.recovery_time_seconds,
            message=heal_result.message,
        )

    def _do_migrate(self, action: InfraAction) -> ActionResult:
        params = action.parameters
        source = params.get("source_region")
        target = params.get("target_region")
        if not source or not target:
            return ActionResult(action_id=action.id, success=False, error="Missing migration regions.")

        from vimana.models import Region
        plan = self.migrator.plan_migration(
            Region(source), Region(target),
            resource_ids=[action.target_resource_id] if action.target_resource_id else None,
        )
        result = self.migrator.execute_migration(plan)
        return ActionResult(
            action_id=action.id,
            success=result.success,
            actual_cost_delta=result.cost,
            duration_seconds=result.actual_downtime_seconds,
            message=result.message,
        )

    # -- health sweep -------------------------------------------------------

    def _run_health_sweep(self) -> None:
        checks = self.health_monitor.check_all()
        for check in checks:
            if not check.healthy and check.failure_type:
                diag = self.health_monitor.diagnose(check.symptoms)
                diag.resource_id = check.resource_id
                self.self_healer.heal(diag)

    def _attempt_heal(self, resource_id: str) -> None:
        resource = self.simulator.get_resource(resource_id)
        if resource is None:
            return
        check = self.health_monitor.check(resource)
        if not check.healthy:
            diag = self.health_monitor.diagnose(check.symptoms)
            diag.resource_id = resource_id
            self.self_healer.heal(diag)

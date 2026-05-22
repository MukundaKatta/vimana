#!/usr/bin/env python3
"""Vimana v0.2.0 auto-scaler demo.

The 2022 question: can an AI agent autonomously provision, scale, and
navigate its own cloud infrastructure?

The 2026 answer is small and concrete. This script runs a 50-step
autonomous scaling loop against a simulated cloud. The agent:

    - reads metrics from a mock cloud every tick
    - watches for distribution shift on the CPU stream
    - decides scale-up, scale-down, or hold
    - executes against the mock CloudProvisioner
    - stays inside a $0.50 USD run budget (enforced by token-budget-py)
    - rejects egress to any host not on its allowlist (birddog contract)
    - traces every decision (agenttrace, optional)

Run it:

    python examples/auto_scaler.py
"""

from __future__ import annotations

import random

import numpy as np

from vimana.agent.auto_scaler import AutoScalerAgent
from vimana.agent.egress import DomainDeniedError, EgressGuard
from vimana.cloud.provisioner import CloudProvisioner
from vimana.cloud.scaler import ScalingThresholds
from vimana.models import (
    Region,
    ResourceSpec,
    ResourceType,
    SimulationScenario,
)
from vimana.simulation import CloudSimulator


def banner(text: str) -> None:
    line = "=" * 72
    print(line)
    print(text)
    print(line)


def _safe_short_session_id(value) -> str:
    """Return a printable session id even if the trace backend is mocked."""
    if value is None:
        return "n/a"
    try:
        return str(value)[:12]
    except Exception:
        return "n/a"


def main(seed: int = 7) -> None:
    banner("Vimana v0.2.0 auto-scaler demo")
    print("Goal: 50-step autonomous scaling run, hard $0.50 USD budget.")
    print()

    random.seed(seed)
    np.random.seed(seed)

    # 1. set up a small mock cloud
    scenario = SimulationScenario(
        name="auto-scaler-demo",
        total_ticks=50,
        base_load=0.3,
        load_pattern="burst",
        failure_probability=0.0,
        initial_resources=[
            ResourceSpec(
                resource_type=ResourceType.VM,
                name="api-vm",
                vcpus=4,
                memory_gb=8.0,
                region=Region.US_EAST_1,
                replicas=2,
            ),
        ],
    )
    sim = CloudSimulator(scenario=scenario)
    sim.initialize()
    target_id = sim.state.resources[0].id

    print(f"  initial cost: ${sim.state.total_cost_per_hour:.4f}/hr")
    print(f"  initial replicas: {sim.state.resources[0].spec.replicas}")
    print(f"  target resource: {target_id[:8]}")
    print()

    # 2. prove the egress allowlist actually denies a rogue host before
    #    we hand control to the agent
    guard = EgressGuard()
    print(f"  egress backend: {guard.backend}")
    try:
        guard.assert_allowed("https://evil.example.com/exfil")
    except DomainDeniedError as exc:
        print(f"  egress guard denied rogue host as expected: {exc}")
    guard.assert_allowed("https://metrics.simulated.local/agg")
    print("  egress guard allows declared simulated host")
    print()

    # 3. run the agent
    provisioner = CloudProvisioner(simulator=sim)
    # Shorter cooldown so the agent can react within a 50-step demo.
    thresholds = ScalingThresholds(
        cpu_scale_up=65.0,
        cpu_scale_down=20.0,
        memory_scale_down=35.0,
        cooldown_ticks=3,
    )
    agent = AutoScalerAgent(
        simulator=sim,
        provisioner=provisioner,
        target_resource_id=target_id,
        usd_cap=0.50,
        thresholds=thresholds,
        verbose=True,
    )

    # Optional: wrap the run in agenttrace if available. We do this with
    # a soft import so the demo never blocks on tracing being installed.
    try:
        from agenttrace import TraceManager

        tracer = TraceManager(db_path="/tmp/vimana_agent_traces.db", colored_logging=False)

        @tracer.trace(tags=["vimana", "auto_scaler"])
        def _traced_run():
            return agent.run(steps=50)

        report = _traced_run()
        trace_sid = _safe_short_session_id(getattr(tracer, "session_id", None))
    except ImportError:
        report = agent.run(steps=50)
        trace_sid = "n/a (agenttrace not installed)"

    print()
    banner("Run report")
    print(f"  session_id:           {report.session_id}")
    print(f"  trace session:        {trace_sid}")
    print(f"  steps completed:      {report.step_count}/50")
    print(f"  scale_up decisions:   {report.scale_up_count}")
    print(f"  scale_down decisions: {report.scale_down_count}")
    print(f"  hold decisions:       {report.hold_count}")
    print(f"  drift alerts:         {report.drift_alerts}")
    print(f"  initial cost/hr:      ${report.initial_cost_per_hour:.4f}")
    print(f"  final cost/hr:        ${report.final_cost_per_hour:.4f}")
    print(f"  usd spent:            ${report.usd_spent:.4f} of ${report.usd_cap:.2f} cap")
    if report.stopped_early:
        print(f"  stopped early:        {report.stopped_reason}")
    else:
        print("  budget cap held across the full run")

    print()
    # 4. prove the budget cap actually fires when pushed.
    banner("Budget enforcement check")
    print("Re-running with a tiny $0.05 cap. Agent must stop early.")
    sim2 = CloudSimulator(scenario=scenario)
    sim2.initialize()
    target_id2 = sim2.state.resources[0].id
    prov2 = CloudProvisioner(simulator=sim2)
    tight_agent = AutoScalerAgent(
        simulator=sim2,
        provisioner=prov2,
        target_resource_id=target_id2,
        usd_cap=0.05,
        thresholds=thresholds,
        verbose=False,
    )
    tight_report = tight_agent.run(steps=50)
    print(f"  steps completed:  {tight_report.step_count}/50")
    print(f"  usd spent:        ${tight_report.usd_spent:.4f}")
    print(f"  stopped_early:    {tight_report.stopped_early}")
    if tight_report.stopped_reason:
        print(f"  stopped_reason:   {tight_report.stopped_reason}")

    print()
    print("Vimana v0.2.0 demo complete.")


if __name__ == "__main__":
    main()

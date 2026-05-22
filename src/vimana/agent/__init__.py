"""Vimana v0.2.0 autonomous agent layer.

Wraps the existing cloud simulator with a self-orchestrating decision
loop. Each tick the agent reads metrics, runs a budget-capped decision
step, and executes a scale-up / scale-down / hold against the mock
cloud. Every decision is traced and bounded.

The composition uses libraries the maintainer published in 2026:

    agenttrace        per-decision tracing
    token-budget-py   shared USD cap on the whole run
    birddog           egress allowlist (proof: rogue host is denied)
    driftvane         metric-stream drift detection

The 2022 question was "can an AI agent provision, scale, and navigate
its own cloud infrastructure?" The v0.2.0 answer is a small but honest
yes: the agent runs a 50-step scaling loop against a simulated cloud,
inside a hard $0.50 budget, with every move recorded.
"""

from vimana.agent.auto_scaler import (
    AutoScalerAgent,
    AgentDecision,
    AgentRunReport,
)

__all__ = ["AutoScalerAgent", "AgentDecision", "AgentRunReport"]

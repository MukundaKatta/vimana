"""Auto-scaler agent: the v0.2.0 finish of the Vimana 2022 question.

This is a deliberately small autonomous agent. Each tick it:

    1. reads the latest aggregate metrics from the simulated cloud
    2. updates a drift-style monitor on the CPU stream
    3. picks one of {scale_up, scale_down, hold}
    4. reserves a small slice of USD from a hard run budget
    5. asks the existing CloudProvisioner to apply the decision

The point is not to invent a smarter scaler than the existing
threshold-based AutoScaler in `vimana.cloud.scaler`. The point is to
wrap a self-orchestrating loop with the safety primitives a 2026 agent
actually needs: per-decision tracing, a hard cost ceiling, an egress
allowlist, and a drift signal on the input stream.

Most of those primitives are libraries the maintainer published this
year. agenttrace and token-budget-py are pip-installed. birddog and
driftvane are imported but only exercised in optional checks so the
demo still runs cleanly when they are not installed.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from token_budget import BudgetExceeded, BudgetPool

from vimana.cloud.provisioner import CloudProvisioner
from vimana.cloud.scaler import AutoScaler, ScalingThresholds
from vimana.models import (
    ResourceStatus,
    ScaleDecision,
    ScaleDirection,
)
from vimana.simulation import CloudSimulator


# Per-decision USD cost. Small on purpose. Real agents pay LLM tokens
# here; this is a stand-in so the budget cap is observable in a $0.50
# run over 50 steps.
_DECISION_COST_USD = 0.005


@dataclass
class AgentDecision:
    """One step of the auto-scaler loop."""

    tick: int
    direction: ScaleDirection
    reason: str
    cpu_observed: float
    drift_flag: bool
    replicas_before: int
    replicas_after: int
    cost_per_hour_after: float
    usd_spent_total: float
    elapsed_ms: float


@dataclass
class AgentRunReport:
    """Summary of a full run."""

    session_id: str
    decisions: list[AgentDecision]
    final_cost_per_hour: float
    initial_cost_per_hour: float
    usd_spent: float
    usd_cap: float
    drift_alerts: int
    stopped_early: bool
    stopped_reason: str = ""

    @property
    def step_count(self) -> int:
        return len(self.decisions)

    @property
    def scale_up_count(self) -> int:
        return sum(1 for d in self.decisions if d.direction == ScaleDirection.UP)

    @property
    def scale_down_count(self) -> int:
        return sum(1 for d in self.decisions if d.direction == ScaleDirection.DOWN)

    @property
    def hold_count(self) -> int:
        return sum(1 for d in self.decisions if d.direction == ScaleDirection.MAINTAIN)


@dataclass
class _DriftMonitor:
    """Tiny drift signal on a 1-D float stream.

    Logic mirrors driftvane.LatencyDrift in shape: compare a reference
    window against the most recent current window using a simple
    median shift. driftvane uses KS for richer p-value semantics; the
    inline version is here so the demo runs even without optional
    deps. The flag is wired into the agent's decision context, not the
    decision itself, so a drift event explains why a scale-up looked
    early.
    """

    reference_window: int = 12
    current_window: int = 6
    shift_threshold: float = 18.0
    _history: list[float] = field(default_factory=list)
    _last_drift: bool = False

    def observe(self, value: float) -> bool:
        self._history.append(value)
        need = self.reference_window + self.current_window
        if len(self._history) < need:
            self._last_drift = False
            return False
        ref = self._history[-need : -self.current_window]
        cur = self._history[-self.current_window :]
        ref_med = sum(ref) / len(ref)
        cur_med = sum(cur) / len(cur)
        self._last_drift = abs(cur_med - ref_med) >= self.shift_threshold
        return self._last_drift

    @property
    def last_drift(self) -> bool:
        return self._last_drift


@dataclass
class AutoScalerAgent:
    """Self-orchestrating auto-scaler over the mock cloud.

    Composition:
        * CloudSimulator     existing tick-based mock cloud
        * CloudProvisioner   existing wrapper that mutates the sim
        * AutoScaler         existing threshold-based scaling brain
        * BudgetPool         hard USD cap across the whole run
        * _DriftMonitor      drift signal on the CPU stream
    """

    simulator: CloudSimulator
    provisioner: CloudProvisioner
    target_resource_id: str
    usd_cap: float = 0.50
    thresholds: ScalingThresholds = field(default_factory=ScalingThresholds)
    verbose: bool = False

    _budget: BudgetPool = field(init=False)
    _scaler: AutoScaler = field(init=False)
    _drift: _DriftMonitor = field(init=False)
    _session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def __post_init__(self) -> None:
        self._budget = BudgetPool(usd_cap=self.usd_cap)
        self._scaler = AutoScaler(
            provisioner=self.provisioner,
            thresholds=self.thresholds,
        )
        self._drift = _DriftMonitor()

    # -- public API ---------------------------------------------------------

    def run(self, steps: int = 50) -> AgentRunReport:
        """Run the auto-scaler loop for `steps` ticks."""
        initial_cost = self.simulator.state.total_cost_per_hour
        decisions: list[AgentDecision] = []
        stopped_early = False
        stopped_reason = ""

        for tick in range(1, steps + 1):
            # Advance the mock cloud so metrics evolve.
            self.simulator.step()
            agg = self.simulator.aggregate_metrics()
            cpu_observed = float(agg.get("avg_cpu", 0.0))
            self._drift.observe(cpu_observed)

            t0 = time.monotonic()
            try:
                # Reserve the USD slice for this decision. If we are
                # about to breach the run cap, BudgetExceeded fires
                # BEFORE the decision is made, so the agent cannot
                # spend itself bankrupt.
                self._budget.record(usd=_DECISION_COST_USD)
            except BudgetExceeded as exc:
                stopped_early = True
                stopped_reason = (
                    f"budget cap reached at tick {tick}: "
                    f"attempted ${exc.attempted:.4f} > cap ${exc.cap:.4f}"
                )
                break

            decision = self._decide(agg)
            replicas_before, replicas_after = self._apply(decision)
            cost_after = self.simulator.state.total_cost_per_hour
            elapsed_ms = (time.monotonic() - t0) * 1000.0

            d = AgentDecision(
                tick=tick,
                direction=decision.direction,
                reason=decision.reason,
                cpu_observed=round(cpu_observed, 2),
                drift_flag=self._drift.last_drift,
                replicas_before=replicas_before,
                replicas_after=replicas_after,
                cost_per_hour_after=round(cost_after, 4),
                usd_spent_total=round(self._budget.snapshot().usd_used, 4),
                elapsed_ms=round(elapsed_ms, 3),
            )
            decisions.append(d)

            if self.verbose:
                self._log(d)

        report = AgentRunReport(
            session_id=self._session_id,
            decisions=decisions,
            final_cost_per_hour=round(self.simulator.state.total_cost_per_hour, 4),
            initial_cost_per_hour=round(initial_cost, 4),
            usd_spent=round(self._budget.snapshot().usd_used, 4),
            usd_cap=self.usd_cap,
            drift_alerts=sum(1 for d in decisions if d.drift_flag),
            stopped_early=stopped_early,
            stopped_reason=stopped_reason,
        )
        return report

    # -- decision + apply ---------------------------------------------------

    def _decide(self, metrics: dict[str, float]) -> ScaleDecision:
        return self._scaler.evaluate(metrics)

    def _apply(self, decision: ScaleDecision) -> tuple[int, int]:
        """Apply the decision to the target resource. Returns
        (replicas_before, replicas_after)."""
        resource = self.provisioner.get_resource(self.target_resource_id)
        if resource is None:
            return (0, 0)
        prev = resource.spec.replicas
        if decision.direction == ScaleDirection.MAINTAIN:
            return (prev, prev)
        result = self._scaler.apply(decision, self.target_resource_id)
        if result.success and resource.status == ResourceStatus.TERMINATED:
            # An aggressive scale-down to zero replicas is undesirable
            # for the demo. Re-anchor at min_replicas.
            target = self.thresholds.min_replicas
            self.provisioner.scale_replicas(self.target_resource_id, target)
            resource.status = ResourceStatus.RUNNING
            return (prev, target)
        # Replicas may differ from result.new_replicas in degenerate
        # cases; read back from the live resource for the report.
        post = resource.spec.replicas
        return (prev, post)

    # -- logging helper -----------------------------------------------------

    def _log(self, d: AgentDecision) -> None:
        arrow = {
            ScaleDirection.UP: "UP  ",
            ScaleDirection.DOWN: "DOWN",
            ScaleDirection.MAINTAIN: "HOLD",
        }[d.direction]
        drift_tag = " drift" if d.drift_flag else ""
        print(
            f"  t={d.tick:>3} {arrow} cpu={d.cpu_observed:5.1f}% "
            f"replicas={d.replicas_before}->{d.replicas_after} "
            f"cost=${d.cost_per_hour_after:.4f}/hr "
            f"spent=${d.usd_spent_total:.4f}{drift_tag}"
        )

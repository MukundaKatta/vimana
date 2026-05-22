# Vimana

Self-orchestrating AI infrastructure agent. Started in 2022 as a research
question. Finished in 2026 with a small but working autonomous loop.

## The 2022 question

> Can AI agents autonomously provision, scale, and navigate their own
> cloud infrastructure?

I asked it, wrote a landing page, sketched a Python scaffold, and then
got pulled into day-job work. The repo sat archived for four years.

## What changed in 2026

Between 2022 and now I ended up shipping a stack of small open-source
libraries that solve the boring parts of running an agent in production:
budget caps, per-call tracing, egress allowlists, drift detection on
input streams. Once those were on PyPI, the 2022 question turned into
homework I could actually finish.

`vimana` v0.2.0 ships a 50-step auto-scaler agent that runs against a
simulated cloud, inside a hard $0.50 USD budget, with every decision
recorded.

## Quickstart

```bash
git clone https://github.com/MukundaKatta/vimana
cd vimana
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

python examples/auto_scaler.py
```

Or via the CLI:

```bash
vimana scale --steps 50 --usd-cap 0.50
```

## What the agent actually does

Every tick:

1. Pulls aggregate metrics from the mock cloud simulator
2. Updates a drift monitor on the CPU stream (driftvane-style)
3. Records a small slice of USD against a `token-budget-py` `BudgetPool`
4. If the budget cap would be breached, stops immediately
5. Otherwise asks the existing threshold-based `AutoScaler` for a
   decision (scale up, scale down, or hold)
6. Executes the decision against the mock `CloudProvisioner`
7. Logs a structured `AgentDecision` row

The whole loop sits behind `EgressGuard`, which mirrors
`birddog.Birddog` and denies any HTTP target not on the allowlist. The
demo proves it: it tries to hit `evil.example.com` and gets a
`DomainDeniedError` before the agent starts.

## Demo output excerpt

```
========================================================================
Vimana v0.2.0 auto-scaler demo
========================================================================
  initial cost: $1.4600/hr
  initial replicas: 2
  egress backend: birddog
  egress guard denied rogue host as expected: host 'evil.example.com' not in allowlist
  egress guard allows declared simulated host

  t=  1 HOLD cpu= 29.7% replicas=2->2 cost=$1.4600/hr spent=$0.0050
  t= 18 HOLD cpu= 52.2% replicas=2->2 cost=$1.4600/hr spent=$0.0900
  t= 19 UP   cpu= 75.2% replicas=2->3 cost=$2.1900/hr spent=$0.0950
  t= 20 HOLD cpu= 86.1% replicas=3->3 cost=$2.1900/hr spent=$0.1000 drift
  t= 37 UP   cpu= 71.7% replicas=3->4 cost=$2.9200/hr spent=$0.1850
  t= 50 HOLD cpu= 26.8% replicas=4->4 cost=$2.9200/hr spent=$0.2500 drift

  scale_up decisions:   2
  scale_down decisions: 0
  drift alerts:         14
  usd spent:            $0.2500 of $0.50 cap
  budget cap held across the full run

========================================================================
Budget enforcement check
========================================================================
Re-running with a tiny $0.05 cap. Agent must stop early.
  steps completed:  10/50
  usd spent:        $0.0500
  stopped_early:    True
  stopped_reason:   budget cap reached at tick 11: attempted $0.0550 > cap $0.0500
```

## Composition

| Concern              | Library                                                  |
| -------------------- | -------------------------------------------------------- |
| USD budget cap       | [`token-budget-py`](https://pypi.org/project/token-budget-py/) |
| Per-decision tracing | [`agenttrace`](https://pypi.org/project/agenttrace/)     |
| Egress allowlist     | [`birddog`](https://pypi.org/project/birddog/)           |
| Drift on metrics     | [`driftvane`](https://pypi.org/project/driftvane/) (contract; inline implementation) |

The drift monitor is shipped inline so the demo runs without
`driftvane` installed. The shape matches `driftvane.LatencyDrift`.

## What this is, and what it isn't

It is:

* a small, honest answer to the 2022 question
* a worked example of composing four narrow agent libraries
* a sandbox that runs locally with no real cloud spend

It is not:

* a production cloud orchestrator
* a replacement for Kubernetes, Karpenter, or any real autoscaler
* connected to real AWS, GCP, or Azure

## Tests

```bash
pytest -q
```

62 tests, all passing.

## DEV.to write-up

Write-up is at `POST.md` (front-matter `published: false`). Will go live
when I publish.

## Live demo page

Landing page is at <https://MukundaKatta.github.io/vimana>.

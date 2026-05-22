---
title: "How I finished a 4-year-old AI project using 30 libraries I wrote in the meantime"
published: false
description: "In 2022 I asked whether an AI agent could run its own cloud. I gave up. In 2026 I had the right libraries to actually finish it."
tags: finishupathon, ai, agents, python
---

In August 2022 I wrote a one-paragraph research proposal called Vimana.
The question was simple:

> Can an AI agent autonomously provision, scale, and navigate its own
> cloud infrastructure?

I built a landing page, sketched some Python, pushed it to GitHub, and
then never touched it again. The repo sat archived for four years.

This is the story of finally finishing it for the DEV.to GitHub
Finish-Up-A-Thon. The trick is that I did not finish it the way I
planned in 2022. I finished it by gluing together a stack of small
libraries I ended up shipping between then and now.

## Why I gave up in 2022

Three reasons, in order of honesty.

First, the question was too big. "Self-orchestrating AI cloud
infrastructure" is a research program, not a weekend project. I did not
have a small thing I could ship.

Second, the safety story was missing. Even back then I knew that
handing an LLM an AWS account was a bad idea. There was no clean way to
say "you can spend up to fifty cents and only call these two domains".
I would have ended up either burning real money on agents that loop, or
mocking so much that the demo proved nothing.

Third, day job. I went into contractor work doing AI/ML stuff for a
big airline. The repo went quiet.

## What changed between 2022 and 2026

Honestly: I built the boring parts.

Over the last year or so I shipped a bunch of small libraries on PyPI
and crates.io. They each solve one tiny problem you hit the moment you
try to run an agent for real. A short list of the ones relevant here:

* `token-budget-py`: a thread-safe pool that holds a token and USD
  cap. Trying to record past the cap throws `BudgetExceeded`. Two
  lines to wrap any agent loop.
* `birddog`: domain allowlist for outbound HTTP. Trying to fetch a
  host that is not in the allowlist throws `DomainDeniedError`. Useful
  for scraping agents that should not be calling random URLs.
* `agenttrace`: sqlite-backed per-call tracing. Decorate a function
  and you get a row per invocation with args, result, and timing.
* `driftvane`: drift detection on numeric streams. Compare a
  reference window against a current window, get a `DriftSignal`.

None of these are revolutionary. They are infrastructure you keep
re-implementing in every agent project. Once they were on PyPI I
realized I had everything I needed to actually answer the 2022 question
at a small but honest scale.

## The finish: what Vimana v0.2.0 is now

A 50-step autonomous auto-scaler that runs against a simulated cloud,
inside a hard $0.50 USD budget, with every decision recorded.

Every tick the agent:

1. reads aggregate metrics from a mock cloud simulator
2. updates a drift monitor on the CPU stream
3. records a small slice of USD against a `BudgetPool` (and stops if
   the cap would breach)
4. asks an existing threshold-based scaler what to do
5. executes scale-up / scale-down / hold against a mock provisioner
6. logs a structured row

Before any of that runs, the egress allowlist proves itself by denying
a fake rogue host. The demo prints the denial so you can see it.

## Code walkthrough

The smallest interesting piece is the per-decision loop. Here is the
budget check, lightly trimmed:

```python
from token_budget import BudgetExceeded, BudgetPool

class AutoScalerAgent:
    def __post_init__(self) -> None:
        self._budget = BudgetPool(usd_cap=self.usd_cap)

    def run(self, steps: int = 50) -> AgentRunReport:
        for tick in range(1, steps + 1):
            self.simulator.step()
            agg = self.simulator.aggregate_metrics()

            try:
                self._budget.record(usd=_DECISION_COST_USD)
            except BudgetExceeded as exc:
                return self._stop_early(tick, exc)

            decision = self._decide(agg)
            self._apply(decision)
```

That `try/except` around `BudgetPool.record` is the entire safety
contract. The agent literally cannot run the next decision if the
budget would breach. No clever prompt, no LLM judge, just a thread-safe
counter that refuses to go past a number.

The egress guard is even shorter:

```python
from birddog import DomainDeniedError

class EgressGuard:
    allowed: frozenset[str] = frozenset({"metrics.simulated.local"})

    def assert_allowed(self, url: str) -> None:
        host = urlparse(url).hostname or ""
        if host not in self.allowed:
            raise DomainDeniedError(f"host {host!r} not in allowlist")
```

If the agent ever decides to call something it should not, the guard
raises and the operator sees it.

## Before and after

**Before, 2022.** Repo: course PDFs, a stub landing page, a README that
said "research on autonomous cloud-native intelligence". No runnable
code. Archived.

**After, 2026.** Repo: same name. A `vimana` Python package with a
`CloudSimulator`, a `CloudProvisioner`, an `AutoScalerAgent`, an
`EgressGuard`, a drift monitor, 62 passing tests, a CLI, and an example
script you can run in 20 seconds. Plus the 2022 landing page still
sitting at the same URL.

The demo output looks like this:

```
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
```

The agent reacts to the simulated load burst at tick 19, holds during
the cooldown, reacts to the second burst at tick 37, and never goes
past 25 cents of its 50 cent budget. The drift monitor flags ticks
where the CPU distribution shifts away from the reference window.

Pushing the budget down to five cents proves the safety story:

```
========================================================================
Budget enforcement check
========================================================================
Re-running with a tiny $0.05 cap. Agent must stop early.
  steps completed:  10/50
  usd spent:        $0.0500
  stopped_early:    True
  stopped_reason:   budget cap reached at tick 11
```

That is the answer to the 2022 question I would have wanted in 2022.
Not "yes" and not "no". Yes, but inside a budget, with an allowlist,
with tracing on every move.

## The Copilot piece

GitHub Copilot was the editor companion through the finish. I wired
it into VS Code and let it autocomplete the boring parts. The egress
guard wrapper, the test scaffolding for the auto-scaler, the doc
comments on AutoScalerAgent.step, the boilerplate for parsing CLI args
in vimana scale. Most were one or two keystrokes followed by accepting
a suggestion that was already close to what I would have typed myself.
Copilot Chat helped a few times when I forgot the exact shape of a
token-budget-py call or wanted a quick second opinion on a method
name. It was not the architect. The composition of agentleash +
birddog + agenttrace was already in my head from shipping those
libraries. What Copilot did was strip the typing friction so I could
finish what I started in 2022 without losing momentum to the small
stuff.

## What is not in this version

* No real AWS. The cloud is simulated. I am explicitly not putting an
  agent on a real cloud account for a hackathon demo.
* No LLM in the loop. The decision logic is threshold-based. A future
  version can swap that out for an LLM call, but the budget and trace
  primitives mean it would be safe to do so.
* The drift monitor inside `vimana` is hand-rolled. It mirrors the
  shape of `driftvane.LatencyDrift` but does not depend on it, so the
  demo runs cleanly without optional packages.

These are deliberate. The point was to finish something small and
honest, not to ship a fake "AI runs your cloud" demo that quietly
needs a million-dollar GPU budget to actually work.

## Try it

```bash
git clone https://github.com/MukundaKatta/vimana
cd vimana
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python examples/auto_scaler.py
```

The whole demo takes about a second. Tests run in under half a second.

Repo: <https://github.com/MukundaKatta/vimana>

## Closing

The lesson I took from this finish is that big questions usually do
not need bigger answers. They need smaller infrastructure underneath
them. The 2022 me wanted a self-orchestrating AI cloud. The 2026 me
got the same thing, just bounded by a $0.50 cap and a domain
allowlist. That bound is what made the question shippable.

If you have an archived repo with a research question that was too
big four years ago, ask yourself: which boring parts have I built
since then? You might already be done.

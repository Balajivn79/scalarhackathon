# Hackathon Guide — Scenario Derivation

This document explains how the three v2.1 scenarios (`tutorial`, `asymmetric`,
`deterministic`) were derived directly from specific sections of the
[Meta OpenEnv Hackathon Participant Help Guide](https://docs.google.com/document/d/1Odznuzwtb1ecDOm2t6ToZd4MuMXXfO6vWUGcxbC6mFs/edit).

The first six scenarios (`easy`, `medium`, `hard`, `night`, `rush_hour`,
`chaos`) cover the *dynamics axis* — they vary traffic density and
time-of-day to produce different difficulty levels. But the guide
emphasizes three orthogonal capabilities the scenarios should also exercise:

1. **Curriculum learning** — make success possible early (§6)
2. **Multiple independent rewards & reward-hacking detection** (§7-8)
3. **Verifiable rewards / RLVR** (§11)

The new scenarios fill those three gaps respectively.

---

## `tutorial` — Curriculum Bootstrap (Guide §6)

> *"Make success possible early. If the model never sees successful
> trajectories, learning stalls."* — Hackathon Guide §6

### The problem this solves
On `easy` and harder scenarios, a randomly-initialized RL policy can spend
many episodes with near-zero reward before stumbling onto a useful action.
Per the guide, that kills training because there's no gradient to learn from.

### How `tutorial` is constructed
- Only **N** has cars at start (8 of them); other lanes empty.
- Default green is **N** — the right answer is *already selected*.
- Light Poisson arrivals (lambda 0.4 on N, 0.05 elsewhere) so the lane keeps
  being a meaningful target without ever overwhelming the agent.
- 150 ticks, no emergencies, no pedestrians, no incidents.

### Empirical proof of the bootstrap property

| Agent | Score | What it tells us |
|---|---|---|
| `dummy` (always `keep`) | **0.600** | Even the floor agent gets a usable signal |
| `bundle_NS` cheater | 0.600 | Bundle clears N+S — only N has cars, so equivalent |
| `rule_based` | 0.898 | Strong reactive agent excels |

A **0.6 floor** — non-trivial throughput reward from tick 1 — gives RL
training the early gradient the guide demands.

---

## `asymmetric` — Anti-Reward-Hacking (Guide §7-8)

> *"If you only have a single reward signal, it is easier for the model to
> hack it. Multiple independent checks reduce that risk."* — Guide §7

> *"Reward hacking is one of the biggest practical failure modes."* — Guide §8

### The hacks this scenario catches

| Suspected hack the agent might learn | Behaviour | Why it would seem to work elsewhere |
|---|---|---|
| `bundle_NS` as default action | Always clears N+S | Wins big on `rush_hour` where N+S+E+W are all loaded |
| Stay on N forever | `keep` after default green=N | OK on `easy` where N seeds higher than W |
| `bundle_NS` → `bundle_EW` alternation without state-awareness | Half-decent throughput on balanced traffic | Wins on `deterministic` (which has scripted balanced arrivals) |

### How `asymmetric` is constructed
- N and S start with **0 cars**, E and W start with 12 and 10.
- Arrival rates: N+S = 0.05/tick, E+W = 1.5/tick (30× heavier).
- 200 ticks, no other mechanics — pure load-imbalance test.

### The independent reward signal
Beyond throughput and wait-time, `grade_asymmetric` adds a fourth
component: `ew_service_ratio` = (E+W cars cleared) / (total cars cleared).
This is the explicit "multiple independent reward function" the guide
recommends — an agent that maxes throughput on N+S can't also max
ew_service_ratio.

### Empirical proof of hack detection

| Agent | Score | Interpretation |
|---|---|---|
| `dummy` (keep on empty N) | **0.016** | Catastrophic fail — green stuck on empty lane |
| `bundle_NS` cheater | **0.024** | The hack is *exposed* — bundle on the empty axis |
| `rule_based` (busiest-lane) | **0.833** | Healthy score for a state-aware policy |

The **0.81 score gap** between rule-based and the bundle-NS cheater is
the scenario doing exactly what the guide §7-8 demands: catching a hack
that other scenarios reward by accident.

---

## `deterministic` — Verifiable Rewards (Guide §11)

> *"If the task is verifiable, build the verifier first, then plug that
> verifier into RL training."* — Guide §11

> *"Watch generated strategies over time... Periodic human inspection is
> still necessary."* — Guide §15

### The problem this solves
Stochastic environments make it impossible to tell whether a reward
change is "the agent got better" or "the seeds were luckier." The guide
recommends having a verifiable benchmark — a task where you know the
optimal score analytically and can detect drift.

### How `deterministic` is constructed
- **Arrivals are scripted** (no Poisson). An 8-tick repeating pattern:

  | Tick | N | S | E | W | Note |
  |---|---|---|---|---|---|
  | 0, 1 | 2 | 2 | 0 | 0 | NS axis burst |
  | 2, 3 | 0 | 0 | 2 | 2 | EW axis burst |
  | 4-7 | 1 | 1 | 1 | 1 | Balanced |

- **Total per cycle**: 8 cars per lane, 32 total in 8 ticks → 4 cars/tick avg.
- **FLOW_RATE = 4 single, 8 bundle** — bundle exactly matches the load.
- **Optimal policy**: alternate `bundle_NS` and `bundle_EW` every 4 ticks.
- 200 ticks total, no random elements — same trajectory every run.

### The verifier
`tasks.graders.deterministic_optimal_score()` runs the hand-tuned optimal
policy and returns its score. Current benchmark:

```python
{
  "score":             0.981,
  "total_cleared":     806,
  "avg_wait":          3.09,
  "starvation_events": 0,
  "policy":            "alternating bundle_NS / bundle_EW every 4 ticks"
}
```

### How to use it
- **As a regression test**: in CI, assert `deterministic_optimal_score()["score"] >= 0.95`.
  If a refactor drops this, the reward function drifted and the change
  is suspicious — review before merging.
- **As an RL training metric**: agent score / 0.981 = "how close to
  provably-optimal is this agent?" — directly comparable across runs.

---

---

## `pedestrian_heavy` — Demo-Improvement Headroom (Guide §19)

> *"What judges or reviewers will likely find compelling: ... evidence that
> the model improved ..."* — Guide §19

> *"A simple but strong demo format is: baseline model attempt, reward/verifier
> output, trained model attempt, measurable improvement, short explanation
> of safeguards."* — Guide §19

### The problem this solves
After the v2.1 work, an audit of baseline scores revealed that 7 of 9
scenarios were **saturating** for state-aware policies — rule_based scored
0.85+ on most, leaving an LLM/RL agent very little room to demonstrate
improvement. Per Guide §19, the demo needs at least one scenario where a
trained agent can *clearly* beat the rule-based baseline, otherwise the
"evidence the model improved" story falls flat.

### How `pedestrian_heavy` is constructed
- 200 ticks, NORMAL traffic, no emergencies, no incidents.
- **Pedestrian probability raised to 0.20/tick** (4× higher than `medium`'s 0.05).
- That's ~40 pedestrian crossings per episode — far more than the rule_based
  agent's fixed `total_cars < 15` threshold can optimize for.

### Why fixed-threshold rules underperform here
The rule_based agent uses a static rule:
```python
if obs.pedestrian_requests and not obs.pedestrian_active and total_cars < 15:
    return Action(action="pedestrian_hold")
```
A 15-tick `pedestrian_hold` freezes traffic completely. Under heavy ped
flow, the cost-benefit math depends on the *trend* of incoming traffic and
the *size* of the pedestrian backlog, not just current car count. A learned
agent should pick context-dependent thresholds.

### The independent reward signal
`grade_pedestrian_heavy` adds a fourth component beyond throughput / wait /
starvation: `ped_score = 1.0 - avg_ped_backlog / 4.0`. This independent
signal catches *both* failure modes (per Guide §7's "multiple independent
reward functions"):
- An agent that ignores peds entirely → `ped_score → 0`
- An agent that holds aggressively → `throughput_score → 0`

### Empirical proof — hack detection AND headroom (5-run mean)

| Agent | Score | Interpretation |
|---|---|---|
| `dummy` (always `keep`) | 0.214 | floor — no clearing, ped backlog grows |
| `aggressive_holds_cheat` | **0.211** | the hack is *exposed* — 15-tick freeze cost dominates |
| `bundle_NS_cheat` | 0.364 | bundle on the wrong axis — partial fail |
| `round_robin` | 0.403 | no ped handling at all |
| `rule_based` (fixed T=15) | **0.792** | decent baseline |
| `smart_bundle` (uses bundles) | **0.847** | best baseline — uses bundles to keep up |

**Two signals from this table:**

- **Anti-hack: 0.847 vs 0.211 = 0.64 score gap** — aggressive ped-hold abuse
  is caught, just like `asymmetric` catches bundle_NS abuse. Two scenarios,
  two independent hacks, both detected.
- **Headroom: 0.792 → 0.847 = 0.06 lift for `smart_bundle`** — fixed-threshold
  rules genuinely lose to better strategies here. A trained LLM/RL agent has
  measurable room to improve over the rule-based baseline, satisfying §19.

### Why this scenario and not others I considered

| Alternative | Why rejected |
|---|---|
| `gridlock_recovery` (start with 25+ cars/lane) | Optimal policy is still "serve busiest lane" — rule_based already does this. No headroom. |
| `emergency_burst` (5 emergencies in 50 ticks) | Rule-based emergency handling is near-optimal (immediate switch). Saturates. |
| `predictable_peak` (sharp known TOD transition) | Rule-based reacts fast enough that anticipation provides minimal value. |

`pedestrian_heavy` was the only candidate that gave both **measurable
headroom for advanced agents** and an **independent reward-hack target**,
without merely amplifying an existing mechanic.

---

## Summary table (all four guide-derived scenarios)

| Scenario | Guide § | Tests | Score floor (`dummy`) | Score ceiling | Anti-hack signal |
|---|---|---|---|---|---|
| tutorial | §6 curriculum | RL training has signal early | 0.60 | 0.97 | — |
| asymmetric | §7-8 anti-hacking | Multi-signal reward integrity | 0.02 | 0.95 | bundle_NS hack → 0.02 (0.93 gap) |
| deterministic | §11 verifiability | Reward function regression check | 0.18 | 0.96 (vs known-optimal 0.98) | — |
| pedestrian_heavy | §19 demo-improvement | Adaptive tradeoff + 2nd hack target | 0.21 | 0.85 | agg_holds hack → 0.21 (0.64 gap) |

---

## Why these four and not more?

The guide identifies maybe a dozen distinct concerns. Four new scenarios,
not ten, because:

1. The guide explicitly warns against **complexity for its own sake** in §6
   ("Do not begin with your hardest benchmark") and §14 ("Do not start with
   scale").
2. The original six already cover the *dynamics* axis — adding more
   variants of "heavier traffic" or "more emergencies" doesn't add a new
   capability test.
3. Four is the minimum to cover the four orthogonal principles the guide
   most strongly emphasizes (curriculum, anti-hacking, verifiability,
   demo-improvement).

If you want to extend further, the guide's §15 ("Monitor the right things")
suggests adding scenarios that produce clearly distinguishable agent
trajectories — for example a `predictable_peak` scenario with a sharp
known TOD transition that rewards anticipation. Reasonable next addition,
but not required by the guide.

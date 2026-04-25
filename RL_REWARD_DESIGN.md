# Traffic Signal Environment v2 — Design Notes

## What changed and why

This is a hackathon finals submission. v2 keeps the v1 API stable (same
endpoints, same observation/action schema) while making the reward
signal robust enough to actually train an RL agent on, and adding three
new scenarios that test capabilities the original three didn't.

---

## 1. Reward function — the core upgrade

### v1 problems

The v1 reward had seven concrete issues that hurt RL training:

| # | v1 problem | Symptom |
|---|---|---|
| 1 | `cars_cleared = sum(max(0, FLOW_RATE - cars))` rewards empty lanes, not actual throughput | Agent gets credit for inactivity |
| 2 | `-0.3 × max_wait` is unbounded — penalty grows with `avg_wait` | Per-tick reward could spike below -30, swamping all other signal |
| 3 | `np.var(waits)` squared → can dominate reward | Fairness penalty explodes during congestion |
| 4 | Emergency reward is binary (+10 / -20) | Sparse — agent gets no gradient signal between "instant" and "timeout" |
| 5 | No anti-thrashing penalty | Agent can switch every tick at zero cost |
| 6 | No reward for queue *reduction* — purely instantaneous | Agent doesn't learn that progress matters |
| 7 | Component scales differ by ~2 orders of magnitude | Gradient updates dominated by largest component |

### v2 design

Every component is **bounded** before weighting, weights are calibrated
so per-tick reward sits in roughly `[-3, +3]` (with emergency events as
designed-large spikes), and **potential-based shaping** gives a dense
per-tick learning signal without changing the optimal policy.

```python
REWARD_WEIGHTS = {
    "throughput": 1.0,   # cars_cleared / 8                  → [0, +1]
    "queue":      1.0,   # -tanh(total / 20)                 → [-1, 0]
    "max_wait":   1.0,   # -tanh(max_wait / 30)              → [-1, 0]
    "starvation": 1.5,   # -starved / 4                       → [-1.5, 0]
    "fairness":   0.5,   # -tanh(stddev_wait / 15)           → [-0.5, 0]
    "emergency":  3.0,   # graded continuous                  → [-6, +6]
    "switch":     0.5,   # -1 if agent switched this tick    → [-0.5, 0]
    "pedestrian": 1.0,   # +0.05 handled, -1 if backlog>3    → [-1, +0.05]
    "progress":   0.3,   # potential shaping (queue change)  → [-0.3, +0.3]
}
```

#### What each fix does

**Bounded penalties via tanh.** `-tanh(x/k)` saturates near -1 as `x → ∞`.
A queue of 20 cars produces -0.76; a queue of 100 cars still produces
-0.99. The agent gets a strong "this is bad" signal at moderate
congestion without the gradient blowing up at extreme congestion. This
matters because RL gradient updates use these values directly — v1's
unbounded penalties cause unstable policy updates.

**Continuous emergency response.** Instead of binary, the reward
linearly decays from +1.5 (instant response) to 0 (45-second response),
plus a +0.5 bonus for sub-15s response. Hard-fail is -2.0. This gives
the agent a usable gradient signal at *every* response time, not just
two endpoints.

```python
if elapsed > 45:
    emergency_r = -2.0
elif s["current_green"] == s["emergency_lane"]:
    base  = max(0.0, 1.5 - elapsed * (1.5 / 45.0))
    bonus = 0.5 if elapsed <= 15 else 0.0
    emergency_r = base + bonus
```

**Potential-based shaping** (Ng, Harada & Russell, ICML 1999). The
landmark result of that paper: if you add `F(s,a,s') = γΦ(s') - Φ(s)`
to your reward, the optimal policy is preserved. Here Φ(s) =
-total_cars, so reducing the queue gives a small positive bonus that
*never* misleads the agent into bad long-run behavior. This is why we
can add dense shaping without worrying about reward hacking.

**Anti-thrashing switch penalty.** A small (-0.5 weighted) penalty for
changing the green lane. This is what makes the *night* scenario hard
— a naive agent that switches whenever any lane has more cars than the
current one will rack up dozens of switches per episode. Real
intersections have meaningful switching cost (clearance phase, driver
confusion), so this is also more realistic.

---

## 2. Three new scenarios

Each new scenario tests a specific agent capability that the v1 trio
didn't.

| Scenario | Episode | TOD schedule | Tests | Failure mode |
|---|---|---|---|---|
| **night** | 200 ticks | NIGHT throughout | Minimum-intervention policy | Over-switching on tiny queues |
| **rush_hour** | 250 ticks | PEAK_HOUR throughout | Bundle action use; gridlock prevention | Queues balloon, max_wait → ∞ |
| **chaos** | 400 ticks | NIGHT → NORMAL → PEAK → NORMAL → NIGHT | Adaptive policy across regimes; final boss | Agent over-tuned to one regime |

`night` finally activates the `NIGHT` Poisson lambdas that v1 defined
but never reached. `rush_hour` makes the previously-niche bundle
actions clearly worthwhile. `chaos` adds a **lane-incident mechanic**
where lanes randomly close for 20–40 ticks and cannot clear cars,
forcing mid-episode adaptation. The rule-based agent has to be patched
to skip closed lanes — agents that don't adapt will pile cars into a
dead lane.

### Why these tests matter

The v1 trio (easy/medium/hard) form a difficulty axis but they all
test the same thing: "serve the busiest lane." A rule-based agent that
sorts lanes by `cars + avg_wait` scores 0.87 average on v1's three
tasks. But that agent **fails** on night (0.48) because it has no
notion that switching has a cost. It also misses bundle opportunities
on rush_hour. The new scenarios create real opportunities for a
learned agent to win — without them, a 20-line rule-based agent is
already near-optimal.

---

## 3. Bug fixes from v1

| Bug | Effect | Fix |
|---|---|---|
| `cars_cleared` formula was wrong | Throughput reward was based on lane emptiness, not actual cars cleared | Track real cars cleared in `_clear_green_lane` and bundle actions |
| `bundle_NS`/`bundle_EW` also cleared `current_green` | Bundle actions secretly served 3 lanes per tick, making them overpowered | Skip `_clear_green_lane` when a bundle was used (`_bundle_used` flag) |
| `NIGHT` Poisson lambdas were dead code | Defined but never reached; no scenario set NIGHT TOD | New `night` and `chaos` scenarios use NIGHT mode |
| Reward function mutated `emergency_lane` state | Side effect inside reward computation made flow harder to trace | Same fix in v2, but now also clears `emergency_tick_started` for cleanliness |
| Global `np.random` calls | Non-reproducible across seeds | `self.rng = np.random.default_rng(seed)` per env, optional `seed=` param |

---

## 4. Benchmark — what an evaluator should see

Five reference agents, three seeds each:

```
agent        |  easy   | medium  |  hard   |  night  |rush_hr  |  chaos  | average
-------------+---------+---------+---------+---------+---------+---------+--------
dummy        | 0.00    | 0.29    | 0.00    | 0.30    | 0.33    | 0.11    |  0.17
round_robin  | 0.00    | 0.30    | 0.00    | 0.27    | 0.52    | 0.20    |  0.22
rule_based   | 0.94    | 0.88    | 0.87    | 0.48    | 0.84    | 0.90    |  0.82
smart_bundle | 0.94    | 0.89    | 0.94    | 0.54    | 0.96    | 0.94    |  0.87
```

The **0.30+ point spread on night** (rule_based 0.48 vs. easy 0.94) is
the clearest evidence that the new scenarios test something the old
ones don't. The rush_hour gap between rule_based (0.84) and smart_bundle
(0.96) shows that bundle actions are now clearly worth learning to use.

---

## 5. Files

```
environment/traffic_env.py   ← new reward, new scenarios, bug fixes
tasks/graders.py             ← graders for all 6 scenarios
app.py                       ← FastAPI server (now exposes 6 tasks + seed)
benchmark.py                 ← multi-seed multi-agent comparison runner
RL_REWARD_DESIGN.md          ← this file
```

The v1 endpoints are unchanged — anything that worked against v1 still
works against v2; you just get richer reward breakdowns and three new
task IDs.

---

## 6. References

- Ng, A. Y., Harada, D., & Russell, S. J. (1999). *Policy invariance
  under reward transformations: Theory and application to reward
  shaping.* ICML. — The basis for our potential-based progress reward.

- Wei, H., Zheng, G., Yao, H., & Li, Z. (2018). *IntelliLight: A
  reinforcement learning approach for intelligent traffic light
  control.* KDD. — Survey of reward shaping approaches in TSC literature.

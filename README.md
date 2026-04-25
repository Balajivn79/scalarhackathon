---
title: Traffic Signal OpenEnv
emoji: 🚦
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Traffic Signal OpenEnv — v2.3

Adaptive traffic-signal environment built on [OpenEnv](https://github.com/openenv/openenv).
A 4-way intersection where an RL agent picks signal-timing actions every tick.

## 🔗 Project Links

- 🎬 **Video explainer (YouTube, ~7 min):** https://youtu.be/CSeLi4l6cMA
- 🚀 **Hugging Face Space (live demo + code):** https://huggingface.co/spaces/Balajivn79/traffic-openenv/tree/main
- 📓 **Colab Notebook (training & inference walkthrough):** https://colab.research.google.com/drive/1UqiooxQUdZxk-EFjWqWH0qB7zaZuFBjU?usp=sharing
- 💻 **Code repository (GitHub):** https://github.com/Balajivn79/scalarhackathon

> Built for the **OpenEnv RL Hackathon**, conducted by Scaler School of Technology and sponsored by Meta, Hugging Face, and PyTorch.

> **What's new in v2.3:** four real-world domain scenarios with new mechanics.
> `ambulance_run` (six scheduled emergencies with a triage burst), `vip_convoy`
> (Z-security convoys with 8-tick anticipation requirement), `monsoon`
> (weather reduces flow rate up to 50%), and `orchestrated_signals` (final
> boss combining VIP + ambulance + weather + incidents). New simulator
> mechanics: weather state with flow-rate scaling, scheduled emergency
> queues, and VIP convoy traversal with green-corridor anticipation.
>
> **What was new in v2.2:** `pedestrian_heavy` (demo-improvement headroom +
> second anti-hack target). **v2.1:** three guide-derived scenarios
> (`tutorial`, `asymmetric`, `deterministic`). **v2.0:** robust bounded
> reward + dynamics-driven scenarios (`night`, `rush_hour`, `chaos`).
> See [`HACKATHON_GUIDE_DERIVATION.md`](HACKATHON_GUIDE_DERIVATION.md) and
> [`RL_REWARD_DESIGN.md`](RL_REWARD_DESIGN.md) for details.

## What changed across versions

| Area | v1 | v2.0 | v2.1 | v2.2 | v2.3 |
|---|---|---|---|---|---|
| Scenarios | 3 | + 3 dynamics | + 3 guide | + 1 headroom | + 4 real-world |
| Reward components | 6, unbounded | 9, bounded | (same) | (same) | (same; VIP folds into emergency) |
| Per-tick reward range | spikes to -30+ | base in [-3, +3] | (same) | (same) | (same) |
| Emergency reward | binary +10/-20 | graded continuous | (same) | (same) | + scheduled queue mode |
| RNG | global `np.random` | per-env seedable | + scripted-arrival | (same) | + scheduled events |
| Lane incidents | — | new (chaos) | (same) | (same) | (same) |
| Curriculum bootstrap | — | — | new | (same) | (same) |
| Anti-reward-hacking | — | — | new (bundle_NS) | + ped-hold | (same) |
| Verifiable benchmark | — | — | new | (same) | (same) |
| Adaptive-tradeoff headroom | — | — | — | new | (same) |
| Weather mechanic | — | — | — | — | new (CLEAR/RAIN/HEAVY_RAIN) |
| Scheduled emergencies | — | — | — | — | new |
| VIP convoy mechanic | — | — | — | — | new (8-tick anticipation) |

## Environment Description

A 4-way intersection with North, South, East, West lanes. Each tick simulates
one second of real time. The agent controls which lane is green and for how
long. Default arrivals are Poisson; the `deterministic` scenario uses scripted
arrivals.

### Mechanics

- Poisson vehicle arrivals per lane per tick (default), or scripted arrivals (deterministic)
- Flow rate: 4 cars cleared per tick on the active green lane (8 with bundle)
- Starvation: lanes unserved for 60+ ticks incur penalties
- Emergency vehicles: 45-second response window, graded continuous reward
- Pedestrian crossings: pause all traffic for 15 ticks
- Time-of-day: NIGHT / NORMAL / PEAK_HOUR with full schedule support
- Lane incidents (v2): random closure for 20–40 ticks (chaos)
- **Weather (v2.3): CLEAR / RAIN / HEAVY_RAIN — reduces flow rate, accelerates wait**
- **VIP convoys (v2.3): pre-announced 8-tick lead time, multi-lane traversal**
- **Scheduled emergencies (v2.3): deterministic ambulance schedules (vs. Poisson)**

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| north / south / east / west | object | Per-lane `{cars: int, avg_wait: float}` |
| current_green | str | N/S/E/W |
| time_in_phase | int | Ticks since the current phase started |
| emergency_lane | str/null | Lane with emergency vehicle, if any |
| pedestrian_requests | list[str] | Lanes with pending crossing requests |
| pedestrian_active | bool | Whether a hold is currently in progress |
| pedestrian_ticks_remaining | int | — |
| time_of_day | str | NORMAL / PEAK_HOUR / NIGHT |
| tick | int | — |
| **incident_lane** *(v2)* | str/null | Lane closed by an incident |
| **incident_ticks_remaining** *(v2)* | int | — |

## Action Space

`keep`, `switch_to_{N,S,E,W}`, `extend_green`, `pedestrian_hold`,
`bundle_NS`, `bundle_EW`.

## Reward Function (v2)

Every component is normalized to roughly `[-1, +1]` *before* being weighted,
so per-tick base reward sits in `[-3, +3]` (with emergency events as
designed-large spikes).

| Component | Range | Formula |
|---|---|---|
| throughput | [0, +1] | `cars_cleared_this_tick / 8` |
| queue | [-1, 0] | `-tanh(total_queued / 20)` |
| max_wait | [-1, 0] | `-tanh(max_wait / 30)` |
| starvation | [-1, 0] | `-starved_lanes / 4` |
| fairness | [-1, 0] | `-tanh(stddev_wait / 15)` |
| emergency | [-2, +2] | linear decay over 45s + sub-15s bonus |
| switch | [-1, 0] | `-1` if agent switched lanes this tick |
| pedestrian | [-1, +0.05] | small reward for serving / penalty for backlog |
| progress | [-1, +1] | potential-based shaping: `Φ(s') - Φ(s)`, `Φ(s) = -total_cars` |

The `progress` term is **potential-based reward shaping** (Ng, Harada &
Russell, ICML 1999) — preserves the optimal policy while giving a dense
per-tick learning signal.

## Tasks

### Original three (v1, preserved)

- **easy** — Steady normal traffic. 200 ticks. Threshold 0.7.
- **medium** — Uneven traffic with pedestrians. 200 ticks. Threshold 0.6.
- **hard** — Peak-hour transitions with emergencies. 300 ticks. Threshold 0.5.

### Dynamics-driven (v2.0)

- **night** — Low-traffic NIGHT scenario, 200 ticks. Tests minimum-intervention.
- **rush_hour** — Sustained PEAK_HOUR for 250 ticks. Bundle actions matter.
- **chaos** — Final boss. 400 ticks. Day cycle + lane incidents.

### Hackathon-guide-derived (v2.1)

- **tutorial** *(Guide §6 curriculum)* — 150 ticks. Cars only on N (already
  green). Even `keep` scores 0.6 — gives RL training non-zero gradient from
  tick 1.
- **asymmetric** *(Guide §7-8 anti-hacking)* — 200 ticks. Heavy E+W, empty
  N+S. Catches `bundle_NS` and `always-keep` shortcuts. Adds an independent
  `ew_service_ratio` reward signal for hack detection.
- **deterministic** *(Guide §11 verifiable)* — 200 ticks. Scripted (non-Poisson)
  arrivals. Optimal policy is computable. Use `deterministic_optimal_score()`
  in `tasks/graders.py` as a regression test for the reward function.

### Headroom-driven (v2.2)

- **pedestrian_heavy** *(Guide §19 demo-improvement)* — 200 ticks. High
  pedestrian rate (0.20/tick, 4× medium). Tests the adaptive tradeoff between
  serving pedestrians and clearing cars; also catches aggressive ped-hold
  abuse with an independent ped_responsiveness signal.

### Real-world domain events (v2.3)

- **ambulance_run** — 250 ticks. Six scheduled ambulances; first three spaced
  out for consistent response, last three in a 10-tick burst forcing triage.
- **vip_convoy** — 250 ticks. Two Z-security convoys, each pre-announced 8
  ticks ahead. Agent must build a green corridor before convoy arrival.
  Tests anticipation, not reaction.
- **monsoon** — 250 ticks. Weather schedule reduces flow rate by up to 50%
  (HEAVY_RAIN) and accelerates wait accumulation. Tests adaptation to
  degraded throughput conditions; bundle-aware agents win.
- **orchestrated_signals** — 400 ticks. **Final boss.** VIP convoys + ambulance
  bursts + monsoon weather + lane incidents + peak-hour cycle, all
  overlapping. Tests prioritization across multiple concurrent high-priority
  events.

## Baseline Scores (3-seed mean from `benchmark.py`)

### Original + dynamics scenarios (v1, v2.0)

| agent | easy | medium | hard | night | rush_hr | chaos | **avg** |
|---|---|---|---|---|---|---|---|
| dummy (`keep`) | 0.00 | 0.29 | 0.00 | 0.30 | 0.33 | 0.11 | 0.17 |
| round_robin | 0.00 | 0.30 | 0.00 | 0.27 | 0.52 | 0.20 | 0.21 |
| rule_based | 0.94 | 0.88 | 0.87 | 0.48 | 0.84 | 0.90 | 0.82 |
| smart_bundle | 0.94 | 0.89 | 0.94 | 0.54 | 0.96 | 0.94 | **0.87** |

### Guide-derived + adaptive scenarios (v2.1, v2.2)

| agent | tutorial | asymmetric | deterministic | pedestrian_heavy |
|---|---|---|---|---|
| dummy (`keep`) | **0.60** | 0.01 | 0.18 | 0.21 |
| rule_based | 0.96 | 0.94 | 0.96 | 0.80 |
| smart_bundle | 0.96 | **0.98** | 0.97 | **0.83** |
| bundle_NS_cheat | 0.60 | **0.02** | 0.37 | 0.37 |
| agg_holds_cheat | 0.96 | 0.95 | 0.96 | **0.22** |

### Real-world domain scenarios (v2.3)

| agent | ambulance_run | vip_convoy | monsoon | orchestrated_signals |
|---|---|---|---|---|
| dummy (`keep`) | 0.09 | 0.29 | 0.22 | 0.22 |
| rule_based | **0.99** | 0.73 | 0.85 | 0.51 |
| smart_bundle | **0.99** | 0.72 | **0.94** | **0.63** |

**Five signals from these tables that judges should notice:**

1. **Tutorial floor (0.60 for `dummy`)** — every agent gets a meaningful starting score, so RL training has a non-zero gradient from tick 1 (Guide §6).

2. **Asymmetric anti-hack gap (0.94 vs 0.02 = 0.92)** — bundle_NS cheater is detected and punished. Catches the "shortcut" hack (Guide §7-8).

3. **Pedestrian_heavy anti-hack gap (0.83 vs 0.22 = 0.61)** — independent second anti-hack signal. Two scenarios catching two independent hacks.

4. **VIP convoy headroom (0.73 ceiling)** — pre-clearing a green corridor on 8-tick lead time has substantial room above the rule-based baseline. Tests anticipation, not reaction.

5. **Orchestrated_signals (0.51 / 0.63 ceiling)** — the final boss. Multiple high-priority events overlap; even smart agents struggle. Massive headroom for learned policies (Guide §19 "evidence the model improved").

The `night` scenario remains the best learned-agent showcase among the original tasks (rule_based 0.48 — over-switching is heavily punished). The `deterministic` scenario has a known-optimal benchmark of **0.981 (806 cars cleared)** via `deterministic_optimal_score()` — a CI regression test for reward-function drift.

## Quick Start

```bash
# Build & run via Docker
docker compose up

# Or run locally
pip install -r requirements.txt
python -m server.app   # serves on :7860

# Reset with a new scenario and seed (v2)
curl -X POST "http://localhost:7860/reset?task=asymmetric&seed=42"

# Step
curl -X POST http://localhost:7860/step \
     -H "Content-Type: application/json" \
     -d '{"action": "switch_to_E"}'

# List all 9 scenarios
curl http://localhost:7860/tasks
```

## Files

```
environment/traffic_env.py         ← simulator (v2.1)
tasks/graders.py                   ← graders + deterministic_optimal_score()
server/app.py                      ← FastAPI server, exposes 9 tasks
inference.py                       ← LLM agent driver, Gemini-compatible
benchmark.py                       ← multi-agent multi-seed comparison
openenv.yaml                       ← OpenEnv manifest
RL_REWARD_DESIGN.md                ← v2 reward function design doc
HACKATHON_GUIDE_DERIVATION.md      ← v2.1 scenario derivation (this is the cheat sheet)
```

## Authors

Balaji Vellineni, Sasikumar Duraisamy, Rohith Srivatsan

Built for the **OpenEnv RL Hackathon** — Scaler School of Technology, sponsored by Meta, Hugging Face, and PyTorch.

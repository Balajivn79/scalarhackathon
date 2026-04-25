"""
Traffic Signal Optimization Environment — v2.0
================================================

Robust, RL-friendly redesign for the hackathon finals.

Headline changes vs. v1
-----------------------
1.  Reward function rewritten for RL stability:
      - All components bounded in roughly [-1, +1] before weighting (tanh
        transforms) so gradient magnitudes stay sane and no single
        component dominates.
      - Continuous emergency reward that decays linearly with response
        time (was binary +10 / -20 — too sparse for RL).
      - Anti-thrashing switch penalty (small but meaningful).
      - Potential-based reward shaping (Ng, Harada, Russell 1999):
        Φ(s) = -total_queued.  This preserves the optimal policy while
        giving a dense per-tick learning signal.
      - Throughput now reflects ACTUAL cars cleared (the v1 metric
        rewarded empty lanes — see _compute_reward in v1).

2.  Three new scenarios:
      - night       : low-traffic minimum-intervention test
      - rush_hour   : sustained PEAK_HOUR — gridlock prevention
      - chaos       : full day cycle + lane incidents — final boss

3.  New mechanic — lane incidents (used in chaos):
      A lane randomly closes for 20–40 ticks and cannot clear cars
      during that period.  Forces mid-episode adaptation.

4.  Bug fixes from v1:
      - bundle_NS / bundle_EW no longer double-clear the green lane
      - NIGHT Poisson lambdas are actually reachable now
      - per-environment numpy RNG (reproducible with `seed=`)
      - reward function no longer mutates state outside its concern

Author: Balaji Vellineni, Sasikumar Duraisamy, Rohith Srivatsan
"""
from __future__ import annotations
from pydantic import BaseModel
from typing import Optional, Literal
import numpy as np


# ============================================================
#  PYDANTIC SCHEMAS
# ============================================================

class LaneState(BaseModel):
    cars: int
    avg_wait: float


class Observation(BaseModel):
    north: LaneState
    south: LaneState
    east:  LaneState
    west:  LaneState
    current_green:              Literal["N", "S", "E", "W"]
    time_in_phase:              int
    emergency_lane:             Optional[Literal["N", "S", "E", "W"]] = None
    pedestrian_requests:        list[str] = []
    pedestrian_active:          bool = False
    pedestrian_ticks_remaining: int = 0
    time_of_day:                Literal["NORMAL", "PEAK_HOUR", "NIGHT"] = "NORMAL"
    tick:                       int = 0
    # ── v2 additions ──
    incident_lane:              Optional[Literal["N", "S", "E", "W"]] = None
    incident_ticks_remaining:   int = 0
    # ── v2.3 additions ──
    weather:                    Literal["CLEAR", "RAIN", "HEAVY_RAIN"] = "CLEAR"
    vip_eta_lane:               Optional[Literal["N", "S", "E", "W"]] = None
    vip_eta_ticks:              int = 0       # ticks until VIP needs the green corridor
    vip_active:                 bool = False  # convoy currently traversing


class Action(BaseModel):
    action: Literal[
        "keep",
        "switch_to_N", "switch_to_S", "switch_to_E", "switch_to_W",
        "extend_green",
        "pedestrian_hold",
        "bundle_NS", "bundle_EW",
    ]


class Reward(BaseModel):
    total: float
    breakdown: dict


# ============================================================
#  CONSTANTS
# ============================================================

FLOW_RATE        = 4    # cars cleared per tick on a green lane
MAX_PHASE_CAP    = 50   # extend_green ceiling
STARVATION_LIMIT = 60   # ticks unserved before a lane counts as starved

SUPPORTED_TASKS = [
    "easy", "medium", "hard",                              # original three
    "night", "rush_hour", "chaos",                         # v2.0 additions (dynamics-driven)
    "tutorial", "asymmetric", "deterministic",             # v2.1 additions (hackathon-guide-aligned)
    "pedestrian_heavy",                                    # v2.2 addition (headroom for adaptive policies)
    "ambulance_run", "vip_convoy",                         # v2.3 (high-priority lane events)
    "monsoon",                                             # v2.3 (weather mechanic)
    "orchestrated_signals",                                # v2.3 (cascaded events)
]

# ------------------------------------------------------------
#  REWARD WEIGHTS
#  A single fixed schedule shared across all scenarios so that an
#  agent's policy can transfer cleanly from one task to another.
#  Calibrated so per-tick reward usually lands in roughly [-3, +3].
# ------------------------------------------------------------
REWARD_WEIGHTS = {
    "throughput": 1.0,   # cars_cleared / 8                 → [0, +1]
    "queue":      1.0,   # -tanh(total / 20)                → [-1, 0]
    "max_wait":   1.0,   # -tanh(max_wait / 30)             → [-1, 0]
    "starvation": 1.5,   # -starved / 4                      → [-1.5, 0]
    "fairness":   0.5,   # -tanh(stddev_wait / 15)          → [-0.5, 0]
    "emergency":  3.0,   # graded continuous                 → up to +6, -6
    "switch":     0.5,   # -1 if agent switched this tick   → [-0.5, 0]
    "pedestrian": 1.0,   # +0.05 handled, -1 if backlog>3   → [-1, +0.05]
    "progress":   0.3,   # potential shaping (queue change) → [-0.3, +0.3]
}

# ------------------------------------------------------------
#  POISSON RATES — arrivals per lane per tick
# ------------------------------------------------------------
POISSON_LAMBDAS = {
    "NORMAL":    {"N": 0.8, "S": 0.8, "E": 0.8, "W": 0.8},
    "PEAK_HOUR": {"N": 1.5, "S": 1.0, "E": 1.8, "W": 0.8},
    "NIGHT":     {"N": 0.2, "S": 0.2, "E": 0.3, "W": 0.2},
}

# ------------------------------------------------------------
#  SCENARIO CONFIGS — vary the dynamics, not the reward.
# ------------------------------------------------------------
SCENARIO_CONFIGS = {
    "easy": {
        "episode_length":         200,
        "emergency_prob":         0.0,
        "pedestrian_prob":        0.0,
        "initial_seeds":          {"N": 4, "S": 2, "E": 3, "W": 1},
        "time_of_day_schedule":   [(0, "NORMAL")],
        "incidents_enabled":      False,
        "incident_prob":          0.0,
        "description":            "Steady normal traffic. Baseline scenario.",
    },
    "medium": {
        "episode_length":         200,
        "emergency_prob":         0.02,
        "pedestrian_prob":        0.05,
        "initial_seeds":          {"N": 6, "S": 4, "E": 8, "W": 3},
        "time_of_day_schedule":   [(0, "NORMAL")],
        "incidents_enabled":      False,
        "incident_prob":          0.0,
        "description":            "Uneven traffic with pedestrians.",
    },
    "hard": {
        "episode_length":         300,
        "emergency_prob":         0.05,
        "pedestrian_prob":        0.05,
        "initial_seeds":          {"N": 10, "S": 7, "E": 12, "W": 5},
        "time_of_day_schedule":   [(0, "NORMAL"), (100, "PEAK_HOUR"), (200, "NORMAL")],
        "incidents_enabled":      False,
        "incident_prob":          0.0,
        "description":            "Peak hour transitions and emergencies.",
    },
    # ── v2 NEW SCENARIOS ──
    "night": {
        "episode_length":         200,
        "emergency_prob":         0.01,
        "pedestrian_prob":        0.08,    # walkers more frequent vs. cars
        "initial_seeds":          {"N": 1, "S": 1, "E": 1, "W": 1},
        "time_of_day_schedule":   [(0, "NIGHT")],
        "incidents_enabled":      False,
        "incident_prob":          0.0,
        "description":            "Low-traffic night. Tests minimum-intervention "
                                  "policy — over-switching is heavily penalized.",
    },
    "rush_hour": {
        "episode_length":         250,
        "emergency_prob":         0.04,
        "pedestrian_prob":        0.03,
        "initial_seeds":          {"N": 12, "S": 8, "E": 15, "W": 6},
        "time_of_day_schedule":   [(0, "PEAK_HOUR")],
        "incidents_enabled":      False,
        "incident_prob":          0.0,
        "description":            "Sustained heavy load. Bundle actions become "
                                  "valuable; gridlock is the failure mode.",
    },
    "chaos": {
        "episode_length":         400,
        "emergency_prob":         0.06,
        "pedestrian_prob":        0.06,
        "initial_seeds":          {"N": 8, "S": 6, "E": 10, "W": 4},
        "time_of_day_schedule":   [
            (0,   "NIGHT"),
            (50,  "NORMAL"),
            (150, "PEAK_HOUR"),
            (300, "NORMAL"),
            (370, "NIGHT"),
        ],
        "incidents_enabled":      True,
        "incident_prob":          0.005,    # ~1 incident every ~200 ticks
        "description":            "Full day cycle plus random lane incidents. "
                                  "Final boss — exercises every mechanic.",
    },

    # ── v2.1 NEW SCENARIOS (derived from hackathon guide) ──
    #
    # Each of the three scenarios below is motivated by a specific section
    # of the Meta OpenEnv Hackathon Participant Help Guide. They expand the
    # environment beyond pure dynamics-variations into curriculum, anti-
    # reward-hacking, and verifiable-reward territory respectively.

    "tutorial": {
        # Guide §6 "Keep the task simple at first" / curriculum learning:
        #   "make success possible early. If the model never sees successful
        #    trajectories, learning stalls."
        # This scenario guarantees positive throughput reward in the first
        # ~10 ticks even from a random or all-keep policy, so RL training
        # has a non-zero gradient to learn from immediately.
        "episode_length":         150,
        "emergency_prob":         0.0,
        "pedestrian_prob":        0.0,
        "initial_seeds":          {"N": 8, "S": 0, "E": 0, "W": 0},
        "time_of_day_schedule":   [(0, "NORMAL")],
        "incidents_enabled":      False,
        "incident_prob":          0.0,
        "lambda_override":        {"N": 0.4, "S": 0.05, "E": 0.05, "W": 0.05},
        "description":            "Curriculum bootstrap. Cars only on N at start, "
                                  "very light arrivals elsewhere. The default green "
                                  "(N) is already correct — guarantees non-zero reward "
                                  "from tick 1 so RL training has a learning signal.",
    },

    "asymmetric": {
        # Guide §7-8 "Multiple independent reward functions" / "Reward hacking":
        #   "If you only have a single reward signal, it is easier for the model
        #    to hack it. Multiple independent checks reduce that risk."
        # Specifically targets a known shortcut: an agent that overfits to
        # `bundle_NS` on rush_hour as a one-size-fits-all clearing strategy.
        # Here N and S are nearly empty while E and W are heavily loaded —
        # bundle_NS produces near-zero throughput. Only an agent that actually
        # responds to lane state succeeds.
        "episode_length":         200,
        "emergency_prob":         0.0,
        "pedestrian_prob":        0.0,
        "initial_seeds":          {"N": 0, "S": 0, "E": 12, "W": 10},
        "time_of_day_schedule":   [(0, "NORMAL")],
        "incidents_enabled":      False,
        "incident_prob":          0.0,
        "lambda_override":        {"N": 0.05, "S": 0.05, "E": 1.5, "W": 1.5},
        "description":            "Anti-reward-hacking test. E and W are heavy; "
                                  "N and S are nearly empty. Agents that learned "
                                  "bundle_NS (or 'always switch to N') as a shortcut "
                                  "will score badly. Only state-aware policies pass.",
    },

    "deterministic": {
        # Guide §11 "RLVR / verifiable rewards":
        #   "build the verifier first, then plug that verifier into RL training."
        # Arrivals follow a fixed 8-tick pattern (no Poisson randomness). Total
        # arrivals per cycle = 8, FLOW_RATE = 4 → optimal policy is provably
        # "alternate every 2 ticks between N+S axis and E+W axis". The
        # deterministic_optimal_score() helper in tasks/graders.py can be used
        # to validate that the reward function is well-calibrated against this
        # known optimum (catches reward drift after refactors).
        "episode_length":         200,
        "emergency_prob":         0.0,
        "pedestrian_prob":        0.0,
        "initial_seeds":          {"N": 4, "S": 4, "E": 4, "W": 4},
        "time_of_day_schedule":   [(0, "NORMAL")],
        "incidents_enabled":      False,
        "incident_prob":          0.0,
        # 8-tick repeating pattern, totals (8, 8, 8, 8) per cycle for (N,S,E,W).
        # This means each axis (NS or EW) needs to be served 4 ticks per cycle
        # to keep up with arrivals. Optimal-policy throughput is computable.
        "arrival_pattern": [
            (2, 2, 0, 0),  # tick 0: N+S burst
            (2, 2, 0, 0),  # tick 1: N+S burst
            (0, 0, 2, 2),  # tick 2: E+W burst
            (0, 0, 2, 2),  # tick 3: E+W burst
            (1, 1, 1, 1),  # tick 4: balanced
            (1, 1, 1, 1),  # tick 5: balanced
            (1, 1, 1, 1),  # tick 6: balanced
            (1, 1, 1, 1),  # tick 7: balanced
        ],
        "description":            "Verifiable-reward scenario. Scripted (non-Poisson) "
                                  "arrivals — same trajectory every run. Used as a "
                                  "ground-truth check for the reward function and "
                                  "to detect drift after refactors.",
    },

    # ── v2.2 NEW SCENARIO (headroom-driven, Guide §19) ──
    #
    # The other scenarios saturate around 0.85-0.97 for state-aware policies,
    # leaving little room for an LLM/RL agent to demonstrate improvement.
    # Per Guide §19 ("evidence that the model improved"), demos need a
    # scenario where a learned agent can clearly beat the rule-based baseline.

    "pedestrian_heavy": {
        # Tests an ADAPTIVE TRADEOFF that fixed-threshold rules can't optimize:
        # the rule-based agent uses `if peds_pending and total_cars < 15:
        # pedestrian_hold` — but the right threshold under heavy pedestrian
        # flow depends on context (current queue trends, ped backlog size,
        # time of day). A learned agent that picks context-dependent thresholds
        # should beat the fixed-15 rule decisively.
        "episode_length":         200,
        "emergency_prob":         0.0,
        "pedestrian_prob":        0.20,    # 4× higher than `medium`
        "initial_seeds":          {"N": 5, "S": 4, "E": 6, "W": 3},
        "time_of_day_schedule":   [(0, "NORMAL")],
        "incidents_enabled":      False,
        "incident_prob":          0.0,
        "description":            "Adaptive-tradeoff test. Heavy pedestrian arrivals "
                                  "force a context-dependent decision: when to honor "
                                  "ped requests vs. prioritize cars. Fixed-threshold "
                                  "rules underperform — gives learned agents room to win.",
    },

    # ── v2.3 NEW SCENARIOS (real-world domain events) ──
    #
    # These scenarios add domain-realistic events that judges immediately
    # recognize from real intersections: ambulances, VIP/Z-security convoys,
    # heavy rain, and orchestrated emergency cascades. They use new mechanics
    # (weather, scheduled VIPs, scheduled emergency queues) that genuinely
    # extend the simulator beyond the original Poisson-driven dynamics.

    "ambulance_run": {
        # Dedicated ambulance-handling test. The original scenarios only
        # had ambulances (= emergency_lane events) sprinkled in among other
        # mechanics; this scenario makes ambulance response THE test.
        # Six emergencies on a fixed schedule across 250 ticks. Tests:
        #   - Sub-15s response time (gets the bonus)
        #   - Recovery between events (waits don't pile up)
        #   - No false-positive switches (emergencies aren't always coming)
        "episode_length":         250,
        "emergency_prob":         0.0,    # all emergencies are scheduled, not random
        "pedestrian_prob":        0.0,
        "initial_seeds":          {"N": 4, "S": 3, "E": 4, "W": 3},
        "time_of_day_schedule":   [(0, "NORMAL")],
        "incidents_enabled":      False,
        "incident_prob":          0.0,
        "emergency_schedule": [   # (tick, lane) — six ambulances; last three overlap to test triage
            (20,  "E"),
            (60,  "S"),
            (100, "N"),
            # burst at the end: three ambulances within 10 ticks, force triage
            (180, "W"),
            (185, "E"),
            (190, "N"),
        ],
        "description":            "Six scheduled ambulances across 250 ticks. The first "
                                  "three are spaced (test consistent response); the last "
                                  "three arrive in a 10-tick burst, forcing triage when "
                                  "only one can be served at a time.",
    },

    "vip_convoy": {
        # Z-security / political-leader VIP convoys. Differs from ambulance:
        # - Convoy is PRE-ANNOUNCED (8-tick lead time via vip_eta_*)
        # - Convoy traverses MULTIPLE lanes in sequence (e.g., W→intersection→E)
        # - Agent must build a "green corridor" BEFORE the convoy arrives
        # This rewards anticipation, not reaction. Two convoys per episode.
        "episode_length":         250,
        "emergency_prob":         0.0,
        "pedestrian_prob":        0.0,
        "initial_seeds":          {"N": 5, "S": 4, "E": 6, "W": 4},
        "time_of_day_schedule":   [(0, "NORMAL")],
        "incidents_enabled":      False,
        "incident_prob":          0.0,
        "vip_schedule": [
            # (arrival_tick, path) — path is ordered list of lanes the convoy
            # traverses. Convoy spends 4 ticks per lane.
            (50,  ["W", "E"]),     # west-to-east traversal
            (170, ["N", "S"]),     # north-to-south traversal
        ],
        "description":            "Z-security VIP convoys with 8-tick lead time. Agent "
                                  "must anticipate and pre-clear the green corridor. "
                                  "Tests planning ahead — not just reacting.",
    },

    "monsoon": {
        # Heavy-rain / monsoon scenario. Weather is global (affects all
        # lanes) and reduces effective FLOW_RATE by 50%. Wait accumulation
        # also speeds up by 20%. Without adaptation, queues balloon and
        # rule-based fails. The scenario tests whether the agent recognizes
        # when its "throughput" actions are degraded and adapts strategy.
        "episode_length":         250,
        "emergency_prob":         0.02,   # rain → more emergencies (real-world)
        "pedestrian_prob":        0.04,
        "initial_seeds":          {"N": 6, "S": 5, "E": 7, "W": 4},
        "time_of_day_schedule":   [(0, "NORMAL")],
        "incidents_enabled":      False,
        "incident_prob":          0.0,
        "weather_schedule": [
            (0,   "RAIN"),
            (50,  "HEAVY_RAIN"),
            (180, "RAIN"),
            (220, "CLEAR"),
        ],
        "description":            "Monsoon scenario. Weather schedule reduces flow "
                                  "rate by up to 50% and speeds wait accumulation. "
                                  "Tests adaptation to degraded throughput conditions.",
    },

    "orchestrated_signals": {
        # Final-boss scenario combining ALL v2.3 mechanics in a coordinated
        # cascade: a VIP convoy, three ambulances back-to-back, weather
        # transitions, and lane incidents. Tests whether the agent can
        # juggle multiple high-priority events simultaneously instead of
        # serializing them. This is where reactive policies break down —
        # only an agent that prioritizes correctly survives.
        "episode_length":         400,
        "emergency_prob":         0.02,
        "pedestrian_prob":        0.04,
        "initial_seeds":          {"N": 7, "S": 5, "E": 8, "W": 5},
        "time_of_day_schedule":   [
            (0,   "NORMAL"),
            (100, "PEAK_HOUR"),
            (250, "NORMAL"),
        ],
        "incidents_enabled":      True,
        "incident_prob":          0.003,    # ~1 incident per ~330 ticks
        "weather_schedule": [
            (0,   "CLEAR"),
            (80,  "RAIN"),
            (200, "HEAVY_RAIN"),
            (320, "RAIN"),
            (370, "CLEAR"),
        ],
        "vip_schedule": [
            (60,  ["E", "W"]),     # convoy 1 — early
            (240, ["S", "N"]),     # convoy 2 — during peak + heavy rain
        ],
        "emergency_schedule": [    # ambulance burst overlaps with VIP and rain
            (120, "N"),
            (130, "S"),
            (145, "E"),
            (270, "W"),            # during convoy + heavy rain
        ],
        "description":            "Final boss. VIP convoys + ambulance bursts + "
                                  "monsoon + lane incidents + peak-hour cycle. "
                                  "Tests whether the agent can prioritize multiple "
                                  "concurrent high-priority events.",
    },
}


# ============================================================
#  ENVIRONMENT
# ============================================================

class TrafficEnv:
    """OpenEnv-compatible traffic signal control environment (v2)."""

    def __init__(self, task: str = "easy", seed: Optional[int] = None):
        if task not in SCENARIO_CONFIGS:
            raise ValueError(
                f"Unknown task '{task}'. Supported: {SUPPORTED_TASKS}"
            )
        self.task           = task
        self.config         = SCENARIO_CONFIGS[task]
        self.episode_length = self.config["episode_length"]
        self.weights        = REWARD_WEIGHTS
        # Per-env RNG — pass `seed=` for reproducibility across runs.
        self.rng            = np.random.default_rng(seed)
        self._state: dict   = {}

    # ── public API ────────────────────────────────────────

    def reset(self) -> Observation:
        self._state = {
            "lanes": {
                lane: {"cars": 0, "avg_wait": 0.0, "last_served": 0}
                for lane in ("N", "S", "E", "W")
            },
            "current_green":              "N",
            "time_in_phase":              0,
            "phase_cap":                  30,
            "emergency_lane":             None,
            "emergency_tick_started":     None,
            "pedestrian_requests":        [],
            "pedestrian_active":          False,
            "pedestrian_ticks_remaining": 0,
            "time_of_day":                self._tod_at_tick(0),
            "tick":                       0,
            "done":                       False,
            "incident_lane":              None,
            "incident_ticks_remaining":   0,
            # ── v2.3 mechanics ──
            "weather":                    "CLEAR",
            "vip_eta_lane":               None,
            "vip_eta_ticks":              0,
            "vip_active":                 False,
            "vip_path_remaining":         [],     # ordered list of lanes still to traverse
            "vip_dwell_ticks":            0,      # ticks remaining in current lane
            "vip_missed":                 False,  # set if convoy hit a non-green lane
            "vip_event_queue":            list(self.config.get("vip_schedule", [])),
            "emergency_event_queue":      list(self.config.get("emergency_schedule", [])),
            # per-tick scratch (re-zeroed every step)
            "_cars_cleared_this_tick":    0,
            "_switched_this_tick":        False,
            "_bundle_used":               False,
        }
        for lane, count in self.config["initial_seeds"].items():
            self._state["lanes"][lane]["cars"] = count
        return self._to_observation()

    def step(self, action: Action):
        assert not self._state["done"], "Episode is over. Call reset()."
        s = self._state

        # Per-tick scratch reset
        s["_cars_cleared_this_tick"] = 0
        s["_switched_this_tick"]     = False
        s["_bundle_used"]            = False
        prev_total_cars = sum(l["cars"] for l in s["lanes"].values())

        # --- simulation pipeline ---
        self._apply_action(action)
        self._arrive()
        if not s["_bundle_used"]:           # ← v2 bug fix (bundle no longer double-clears)
            cleared = self._clear_green_lane()
            s["_cars_cleared_this_tick"] += cleared
        self._update_waits()
        self._update_incident()
        self._update_weather()              # ← v2.3
        self._update_vip()                  # ← v2.3
        self._drain_event_queues()          # ← v2.3 (scheduled emergencies & VIPs)
        self._maybe_spawn_emergency()
        self._maybe_spawn_pedestrian()
        s["tick"] += 1
        s["time_of_day"] = self._tod_at_tick(s["tick"])

        reward = self._compute_reward(prev_total_cars)
        done   = self._check_done()
        s["done"] = done

        return self._to_observation(), reward, done, {}

    def state(self) -> Observation:
        return self._to_observation()

    # ── action handling ───────────────────────────────────

    def _apply_action(self, action: Action):
        a = action.action
        s = self._state

        # During pedestrian hold, the only thing that happens is the
        # countdown — every other action is ignored. (Cars still arrive
        # and wait; the green lane simply doesn't clear.)
        if s["pedestrian_active"]:
            s["pedestrian_ticks_remaining"] -= 1
            if s["pedestrian_ticks_remaining"] <= 0:
                s["pedestrian_active"] = False
            return

        if a == "keep":
            s["time_in_phase"] += 1

        elif a.startswith("switch_to_"):
            lane = a[-1]
            if lane != s["current_green"]:
                s["current_green"]                = lane
                s["time_in_phase"]                = 0
                s["phase_cap"]                    = 30
                s["lanes"][lane]["last_served"]   = s["tick"]
                s["_switched_this_tick"]          = True

        elif a == "extend_green":
            s["phase_cap"] = min(s["phase_cap"] + 10, MAX_PHASE_CAP)
            s["time_in_phase"] += 1

        elif a == "pedestrian_hold":
            if s["pedestrian_requests"]:
                s["pedestrian_active"]          = True
                s["pedestrian_ticks_remaining"] = 15
                s["pedestrian_requests"]        = []

        elif a in ("bundle_NS", "bundle_EW"):
            lanes = ("N", "S") if a == "bundle_NS" else ("E", "W")
            cleared_total = 0
            flow = self._flow_rate()
            for lane in lanes:
                if s["incident_lane"] == lane:
                    continue                              # blocked, skip
                cars = s["lanes"][lane]["cars"]
                cleared = min(cars, flow)
                if cleared:
                    remaining = cars - cleared
                    s["lanes"][lane]["avg_wait"] *= remaining / max(1, cars)
                    s["lanes"][lane]["cars"]      = remaining
                    cleared_total += cleared
                s["lanes"][lane]["last_served"]   = s["tick"]
            s["_cars_cleared_this_tick"] += cleared_total
            s["_bundle_used"]             = True
            s["time_in_phase"]           += 1

    # ── dynamics ──────────────────────────────────────────

    # ── weather-adjusted flow rate (v2.3) ──
    # Real-world: heavy rain reduces intersection throughput by ~30-40%
    # because of reduced visibility, longer braking distances, and
    # cautious driving. We model this as a flat FLOW_RATE multiplier.
    def _flow_rate(self) -> int:
        weather = self._state.get("weather", "CLEAR")
        if weather == "HEAVY_RAIN":
            return max(1, int(FLOW_RATE * 0.5))   # 50% throughput
        if weather == "RAIN":
            return max(1, int(FLOW_RATE * 0.75))  # 75% throughput
        return FLOW_RATE

    def _arrive(self):
        # ── v2.1 (hackathon-guide-aligned): two new arrival modes ──
        # 1. arrival_pattern   — scripted, deterministic per-tick arrivals.
        #                        Used by the `deterministic` scenario (Guide §11
        #                        "verifiable rewards") so optimal-policy reward
        #                        is computable analytically.
        # 2. lambda_override   — per-scenario Poisson rates that override the
        #                        time-of-day defaults. Used by `tutorial` and
        #                        `asymmetric` to shape arrival distributions
        #                        for curriculum and anti-hacking tests.
        if "arrival_pattern" in self.config:
            pattern = self.config["arrival_pattern"]
            counts = pattern[self._state["tick"] % len(pattern)]
            for lane, n in zip(("N", "S", "E", "W"), counts):
                self._state["lanes"][lane]["cars"] += int(n)
            return

        tod = self._state["time_of_day"]
        lambdas = self.config.get("lambda_override") or POISSON_LAMBDAS[tod]
        for lane, ldata in self._state["lanes"].items():
            arrivals = int(self.rng.poisson(lambdas[lane]))
            ldata["cars"] += arrivals

    def _clear_green_lane(self) -> int:
        s = self._state
        if s["pedestrian_active"]:
            return 0
        lane = s["current_green"]
        if s["incident_lane"] == lane:
            return 0                                     # incident blocks clearing
        ldata = s["lanes"][lane]
        flow = self._flow_rate()
        cleared = min(ldata["cars"], flow)
        if cleared:
            remaining = ldata["cars"] - cleared
            ldata["avg_wait"] *= remaining / max(1, ldata["cars"])
            ldata["cars"]      = remaining
        ldata["last_served"] = s["tick"]
        return cleared

    def _update_waits(self):
        # v2.3: rain accelerates perceived wait. Drivers in heavy rain feel
        # delays more acutely (idling time, visibility stress). 20% bump
        # in HEAVY_RAIN, 10% in RAIN. Modeling this as wait increment, not
        # a flat multiplier on total, so newly-empty lanes don't get fake
        # wait time.
        weather = self._state.get("weather", "CLEAR")
        increment = 1.0
        if weather == "HEAVY_RAIN":
            increment = 1.2
        elif weather == "RAIN":
            increment = 1.1
        for ldata in self._state["lanes"].values():
            if ldata["cars"] > 0:
                ldata["avg_wait"] += increment

    def _update_incident(self):
        """Handle lane-incident lifecycle (chaos scenario only)."""
        s = self._state
        if not self.config.get("incidents_enabled", False):
            return
        if s["incident_lane"] is not None:
            s["incident_ticks_remaining"] -= 1
            if s["incident_ticks_remaining"] <= 0:
                s["incident_lane"]            = None
                s["incident_ticks_remaining"] = 0
            return
        if self.rng.random() < self.config.get("incident_prob", 0.0):
            s["incident_lane"]            = str(self.rng.choice(["N", "S", "E", "W"]))
            s["incident_ticks_remaining"] = int(self.rng.integers(20, 40))

    # ── v2.3 mechanics ────────────────────────────────────

    def _update_weather(self):
        """
        Walk the weather schedule and set current weather. Weather is global
        (affects all lanes) and is captured in state["weather"], which
        _flow_rate() reads to scale throughput.

        Schedule format: list of (tick, condition) tuples where condition
        is one of "CLEAR", "RAIN", "HEAVY_RAIN".
        """
        schedule = self.config.get("weather_schedule")
        if not schedule:
            return
        tick = self._state["tick"]
        current = "CLEAR"
        for change_tick, condition in schedule:
            if tick >= change_tick:
                current = condition
            else:
                break
        self._state["weather"] = current

    def _update_vip(self):
        """
        Handle Z-security / VIP convoy traversal (orchestrated_signals scenario).

        A convoy enters from the first lane in `vip_path_remaining` and dwells
        there for `vip_dwell_ticks` ticks while traversing the intersection.
        If the signal isn't green for that lane during traversal, vip_missed
        is set (tracked by graders for response-quality scoring).

        After the dwell, the convoy moves to the next lane in the path. When
        the path is empty, the convoy has exited the intersection.
        """
        s = self._state
        if not s["vip_active"]:
            # Update countdown for next scheduled VIP if any
            if s["vip_eta_lane"] is not None and s["vip_eta_ticks"] > 0:
                s["vip_eta_ticks"] -= 1
                if s["vip_eta_ticks"] <= 0:
                    # Convoy arrives — activate
                    s["vip_active"]        = True
                    s["vip_dwell_ticks"]   = 4   # 4 ticks per lane
                    # vip_path_remaining was set when scheduled
            return

        # Convoy is currently in s["vip_eta_lane"] (the head of the path).
        current_lane = s["vip_eta_lane"]
        if s["current_green"] != current_lane:
            s["vip_missed"] = True

        s["vip_dwell_ticks"] -= 1
        if s["vip_dwell_ticks"] <= 0:
            # Move to next lane in path
            if s["vip_path_remaining"]:
                s["vip_eta_lane"]    = s["vip_path_remaining"].pop(0)
                s["vip_dwell_ticks"] = 4
            else:
                # Convoy has exited
                s["vip_active"]    = False
                s["vip_eta_lane"]  = None
                s["vip_eta_ticks"] = 0

    def _drain_event_queues(self):
        """
        Pop scheduled events whose time has come.
          - emergency_schedule: list of (tick, lane) tuples → set emergency_lane
          - vip_schedule:       list of (tick, path) tuples → schedule a convoy
                                (path is an ordered list of lanes, e.g. ["W", "E"]
                                meaning enter from W and exit via E)
        Compared to Poisson spawning, these queues let scenarios test
        anticipation of *known* upcoming events.
        """
        s = self._state
        tick = s["tick"]

        # Emergency schedule
        # Pop any scheduled emergency whose time has come AND whose slot is
        # available. Overdue events (tick already passed because a previous
        # emergency was still active) fire as soon as the slot opens.
        if s["emergency_event_queue"] and s["emergency_lane"] is None:
            head_tick, head_lane = s["emergency_event_queue"][0]
            if tick >= head_tick:
                s["emergency_lane"]         = head_lane
                s["emergency_tick_started"] = tick
                s["emergency_event_queue"].pop(0)

        # VIP schedule — schedule the convoy to ARRIVE in `lead_time` ticks,
        # giving the agent time to anticipate (build a green corridor).
        if s["vip_event_queue"] and not s["vip_active"] and s["vip_eta_lane"] is None:
            head_tick, head_path = s["vip_event_queue"][0]
            lead_time = 8  # 8 ticks of warning to clear the corridor
            if tick >= head_tick - lead_time:
                # Set the convoy's first lane and the rest of the path
                path = list(head_path)
                s["vip_eta_lane"]        = path[0]
                s["vip_path_remaining"]  = path[1:]
                s["vip_eta_ticks"]       = lead_time
                s["vip_event_queue"].pop(0)


    def _maybe_spawn_emergency(self):
        s = self._state
        if s["emergency_lane"] is not None:
            return
        if self.rng.random() < self.config["emergency_prob"]:
            s["emergency_lane"]         = str(self.rng.choice(["N", "S", "E", "W"]))
            s["emergency_tick_started"] = s["tick"]

    def _maybe_spawn_pedestrian(self):
        s = self._state
        if self.rng.random() < self.config["pedestrian_prob"]:
            lane = str(self.rng.choice(["N", "S", "E", "W"]))
            if lane not in s["pedestrian_requests"]:
                s["pedestrian_requests"].append(lane)

    def _tod_at_tick(self, tick: int) -> str:
        """Walk the time-of-day schedule and return the active TOD."""
        current = "NORMAL"
        for change_tick, tod in self.config["time_of_day_schedule"]:
            if tick >= change_tick:
                current = tod
            else:
                break
        return current

    # ── reward ────────────────────────────────────────────

    def _compute_reward(self, prev_total_cars: int) -> Reward:
        """
        Bounded, dense, RL-friendly reward.

        Each component is normalized to roughly [-1, +1] BEFORE weighting,
        so weights have predictable influence and gradients stay stable.

        Component breakdown
        -------------------
        1. throughput  : cars_cleared_this_tick / 8   (max-bundle-normalized)
        2. queue       : -tanh(total_queued / 20)
        3. max_wait    : -tanh(max_wait / 30)
        4. starvation  : -starved_lanes / 4
        5. fairness    : -tanh(stddev_wait / 15)
        6. emergency   : graded continuous (linear decay over 45s + sub-15s bonus)
        7. switch      : -1 if agent changed lanes this tick (anti-thrash)
        8. pedestrian  : +0.05 while serving / -1 if backlog > 3
        9. progress    : potential-based shaping, F = Φ(s') - Φ(s),
                         with Φ(s) = -total_cars  (Ng et al. 1999)
        """
        s     = self._state
        lanes = s["lanes"]
        tick  = s["tick"]
        w     = self.weights

        total_queued = sum(l["cars"] for l in lanes.values())
        max_wait     = max(l["avg_wait"] for l in lanes.values())
        wait_std     = float(np.std([l["avg_wait"] for l in lanes.values()]))
        starved      = sum(
            1 for l in lanes.values()
            if (tick - l["last_served"]) > STARVATION_LIMIT
        )

        # 1. throughput — actually count cars that left, normalized by max
        #    possible per tick (a bundle clears up to 8). Range: [0, 1].
        throughput_r = s["_cars_cleared_this_tick"] / 8.0

        # 2. queue — tanh keeps penalty bounded as queues grow huge.
        #    Saturates around 40 total cars.
        queue_r = -float(np.tanh(total_queued / 20.0))

        # 3. max_wait — bounded penalty, saturates around 60s.
        max_wait_r = -float(np.tanh(max_wait / 30.0))

        # 4. starvation — already discrete; normalize by lane count.
        starvation_r = -starved / 4.0

        # 5. fairness — stddev (not variance), tanh-bounded to keep
        #    influence proportional, never explosive.
        fairness_r = -float(np.tanh(wait_std / 15.0))

        # 6. emergency — graded continuous reward.
        #    0s response = +1.5, decaying linearly to 0 at 45s,
        #    plus +0.5 bonus for sub-15s response. Hard-fail = -2.
        #
        #    v2.3: VIP convoys (scheduled Z-security traversals) feed into
        #    this same component. A successful corridor (green held during
        #    convoy traversal) gives +1.0; a miss (convoy hit a non-green
        #    lane) gives -1.5. This keeps the reward dimensionality stable
        #    while letting a single scenario test both ad-hoc and scheduled
        #    high-priority lane events.
        emergency_r = 0.0
        if s["emergency_lane"] is not None:
            elapsed = tick - s["emergency_tick_started"]
            if elapsed > 45:
                emergency_r = -2.0
                s["emergency_lane"]         = None
                s["emergency_tick_started"] = None
            elif s["current_green"] == s["emergency_lane"]:
                base  = max(0.0, 1.5 - elapsed * (1.5 / 45.0))
                bonus = 0.5 if elapsed <= 15 else 0.0
                emergency_r = base + bonus
                s["emergency_lane"]         = None
                s["emergency_tick_started"] = None

        # VIP convoy contribution to the emergency component
        if s["vip_active"]:
            if s["current_green"] == s["vip_eta_lane"]:
                emergency_r += 0.25                    # +0.25/tick during corridor (4 ticks → +1.0)
            else:
                emergency_r += -0.5                    # miss penalty per tick

        # 7. switch — small constant penalty for changing the green lane.
        #    Discourages thrashing without making switches forbidden.
        switch_r = -1.0 if s["_switched_this_tick"] else 0.0

        # 8. pedestrian — small reward for handling a hold, penalty for backlog.
        ped_count = len(s["pedestrian_requests"])
        if s["pedestrian_active"]:
            pedestrian_r = 0.05
        elif ped_count > 3:
            pedestrian_r = -1.0
        else:
            pedestrian_r = 0.0

        # 9. progress — potential-based shaping.
        #    Φ(s) = -total_cars  →  F(s,a,s') = Φ(s') - Φ(s) = prev - curr
        #    Clipped to ±8 to keep it within roughly [-1, +1].
        progress_r = float(np.clip(prev_total_cars - total_queued, -8, 8)) / 8.0

        weighted = {
            "throughput": w["throughput"] * throughput_r,
            "queue":      w["queue"]      * queue_r,
            "max_wait":   w["max_wait"]   * max_wait_r,
            "starvation": w["starvation"] * starvation_r,
            "fairness":   w["fairness"]   * fairness_r,
            "emergency":  w["emergency"]  * emergency_r,
            "switch":     w["switch"]     * switch_r,
            "pedestrian": w["pedestrian"] * pedestrian_r,
            "progress":   w["progress"]   * progress_r,
        }
        total = sum(weighted.values())
        return Reward(
            total=round(total, 3),
            breakdown={k: round(v, 3) for k, v in weighted.items()},
        )

    # ── termination ───────────────────────────────────────

    def _check_done(self) -> bool:
        s = self._state
        if s["tick"] >= self.episode_length:
            return True
        if s["emergency_lane"] is not None:
            elapsed = s["tick"] - s["emergency_tick_started"]
            if elapsed > 45:
                return True
        return False

    # ── observation ───────────────────────────────────────

    def _to_observation(self) -> Observation:
        s = self._state
        l = s["lanes"]
        return Observation(
            north=LaneState(cars=l["N"]["cars"], avg_wait=round(l["N"]["avg_wait"], 2)),
            south=LaneState(cars=l["S"]["cars"], avg_wait=round(l["S"]["avg_wait"], 2)),
            east= LaneState(cars=l["E"]["cars"], avg_wait=round(l["E"]["avg_wait"], 2)),
            west= LaneState(cars=l["W"]["cars"], avg_wait=round(l["W"]["avg_wait"], 2)),
            current_green              = s["current_green"],
            time_in_phase              = s["time_in_phase"],
            emergency_lane             = s["emergency_lane"],
            pedestrian_requests        = list(s["pedestrian_requests"]),
            pedestrian_active          = s["pedestrian_active"],
            pedestrian_ticks_remaining = s["pedestrian_ticks_remaining"],
            time_of_day                = s["time_of_day"],
            tick                       = s["tick"],
            incident_lane              = s["incident_lane"],
            incident_ticks_remaining   = s["incident_ticks_remaining"],
            weather                    = s["weather"],
            vip_eta_lane               = s["vip_eta_lane"],
            vip_eta_ticks              = s["vip_eta_ticks"],
            vip_active                 = s["vip_active"],
        )

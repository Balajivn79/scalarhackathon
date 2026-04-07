from __future__ import annotations
from pydantic import BaseModel
from typing import Optional, Literal
import numpy as np

# ─────────────────────────────────────────
#  MODELS
# ─────────────────────────────────────────

class LaneState(BaseModel):
    cars: int
    avg_wait: float

class Observation(BaseModel):
    north: LaneState
    south: LaneState
    east:  LaneState
    west:  LaneState
    current_green:            Literal["N", "S", "E", "W"]
    time_in_phase:            int
    emergency_lane:           Optional[Literal["N", "S", "E", "W"]] = None
    pedestrian_requests:      list[str] = []
    pedestrian_active:        bool = False
    pedestrian_ticks_remaining: int = 0
    time_of_day:              Literal["NORMAL", "PEAK_HOUR", "NIGHT"] = "NORMAL"
    tick:                     int = 0

class Action(BaseModel):
    action: Literal[
        "keep",
        "switch_to_N",
        "switch_to_S",
        "switch_to_E",
        "switch_to_W",
        "extend_green",
        "pedestrian_hold",
        "bundle_NS",
        "bundle_EW",
    ]

class Reward(BaseModel):
    total:     float
    breakdown: dict

# ─────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────

FLOW_RATE = 4       # cars cleared per tick on green lane
MAX_EXTEND = 20      # max seconds a phase can be extended to
STARVATION_LIMIT = 60  # ticks before a lane is considered starved

POISSON_LAMBDAS = {
    "NORMAL":    {"N": 0.8, "S": 0.8, "E": 0.8, "W": 0.8},
    "PEAK_HOUR": {"N": 1.5, "S": 1.0, "E": 1.8, "W": 0.8},
    "NIGHT":     {"N": 0.3, "S": 0.3, "E": 0.3, "W": 0.3},
}

# ─────────────────────────────────────────
#  ENVIRONMENT
# ─────────────────────────────────────────

class TrafficEnv:

    def __init__(self, task: Literal["easy", "medium", "hard"] = "easy"):
        self.task = task
        self.episode_length = 200 if task in ("easy", "medium") else 300
        self._state: dict = {}

    # ── public API ──────────────────────────

    def reset(self) -> Observation:
        self._state = {
            "lanes": {
                "N": {"cars": 0, "avg_wait": 0.0, "last_served": 0},
                "S": {"cars": 0, "avg_wait": 0.0, "last_served": 0},
                "E": {"cars": 0, "avg_wait": 0.0, "last_served": 0},
                "W": {"cars": 0, "avg_wait": 0.0, "last_served": 0},
            },
            "current_green":               "N",
            "time_in_phase":               0,
            "phase_cap":                   30,
            "emergency_lane":              None,
            "emergency_tick_started":      None,
            "pedestrian_requests":         [],
            "pedestrian_active":           False,
            "pedestrian_ticks_remaining":  0,
            "time_of_day":                 self._starting_time_of_day(),
            "tick":                        0,
            "done":                        False,
        }
        self._seed_initial_cars()
        return self._to_observation()

    def step(self, action: Action):
        assert not self._state["done"], "Episode is over. Call reset()."

        self._apply_action(action)
        self._arrive()
        self._clear_green_lane()
        self._update_waits()
        self._maybe_spawn_emergency()
        self._maybe_spawn_pedestrian()
        self._advance_time_of_day()
        self._state["tick"] += 1

        reward  = self._compute_reward()
        done    = self._check_done()
        self._state["done"] = done

        return self._to_observation(), reward, done, {}

    def state(self) -> Observation:
        return self._to_observation()

    # ── action logic ────────────────────────

    def _apply_action(self, action: Action):
        a = action.action
        s = self._state

        if s["pedestrian_active"]:
            s["pedestrian_ticks_remaining"] -= 1
            if s["pedestrian_ticks_remaining"] <= 0:
                s["pedestrian_active"] = False
            return                          # all other actions ignored during hold

        if a == "keep":
            s["time_in_phase"] += 1

        elif a.startswith("switch_to_"):
            lane = a[-1]                    # last character is N/S/E/W
            if lane != s["current_green"]:
                s["current_green"]  = lane
                s["time_in_phase"]  = 0
                s["phase_cap"]      = 30
                s["lanes"][lane]["last_served"] = s["tick"]

        elif a == "extend_green":
            s["phase_cap"] = min(s["phase_cap"] + 10, MAX_EXTEND + 30)

        elif a == "pedestrian_hold":
            if s["pedestrian_requests"]:
                s["pedestrian_active"]           = True
                s["pedestrian_ticks_remaining"]  = 15
                s["pedestrian_requests"]         = []

        elif a == "bundle_NS":
            for lane in ("N", "S"):
                cleared = min(s["lanes"][lane]["cars"], FLOW_RATE)
                s["lanes"][lane]["cars"] -= cleared
                if cleared:
                    s["lanes"][lane]["avg_wait"] *= max(
                        0, (s["lanes"][lane]["cars"]) /
                        max(1, s["lanes"][lane]["cars"] + cleared)
                    )
                s["lanes"][lane]["last_served"] = s["tick"]
            s["time_in_phase"] += 1

        elif a == "bundle_EW":
            for lane in ("E", "W"):
                cleared = min(s["lanes"][lane]["cars"], FLOW_RATE)
                s["lanes"][lane]["cars"] -= cleared
                if cleared:
                    s["lanes"][lane]["avg_wait"] *= max(
                        0, (s["lanes"][lane]["cars"]) /
                        max(1, s["lanes"][lane]["cars"] + cleared)
                    )
                s["lanes"][lane]["last_served"] = s["tick"]
            s["time_in_phase"] += 1

    # ── simulation logic ────────────────────

    def _arrive(self):
        tod = self._state["time_of_day"]
        for lane, ldata in self._state["lanes"].items():
            arrivals = int(np.random.poisson(POISSON_LAMBDAS[tod][lane]))
            ldata["cars"] += arrivals

    def _clear_green_lane(self):
        if self._state["pedestrian_active"]:
            return
        lane  = self._state["current_green"]
        ldata = self._state["lanes"][lane]
        cleared = min(ldata["cars"], FLOW_RATE)
        if cleared:
            remaining = ldata["cars"] - cleared
            ldata["avg_wait"] = ldata["avg_wait"] * (
                remaining / max(1, ldata["cars"])
            )
            ldata["cars"] = remaining
        ldata["last_served"] = self._state["tick"]

    def _update_waits(self):
        for lane, ldata in self._state["lanes"].items():
            if ldata["cars"] > 0:
                ldata["avg_wait"] += 1.0    # each waiting car ages one tick

    # ── spawning ────────────────────────────

    def _maybe_spawn_emergency(self):
        s = self._state
        if s["emergency_lane"] is not None:
            return
        prob = {"easy": 0.0, "medium": 0.02, "hard": 0.05}[self.task]
        if np.random.random() < prob:
            s["emergency_lane"]         = np.random.choice(["N", "S", "E", "W"])
            s["emergency_tick_started"] = s["tick"]

    def _maybe_spawn_pedestrian(self):
        s = self._state
        prob = {"easy": 0.0, "medium": 0.05, "hard": 0.05}[self.task]
        if np.random.random() < prob:
            lane = np.random.choice(["N", "S", "E", "W"])
            if lane not in s["pedestrian_requests"]:
                s["pedestrian_requests"].append(lane)

    # ── time of day ─────────────────────────

    def _starting_time_of_day(self):
        return {"easy": "NORMAL", "medium": "NORMAL", "hard": "NORMAL"}[self.task]

    def _advance_time_of_day(self):
        if self.task != "hard":
            return
        tick = self._state["tick"]
        if tick < 100:
            self._state["time_of_day"] = "NORMAL"
        elif tick < 200:
            self._state["time_of_day"] = "PEAK_HOUR"
        else:
            self._state["time_of_day"] = "NORMAL"

    # ── reward ──────────────────────────────

    def _compute_reward(self) -> Reward:
        s      = self._state
        lanes  = s["lanes"]
        tick   = s["tick"]

        cars_cleared   = sum(
            max(0, FLOW_RATE - l["cars"]) for l in lanes.values()
        )
        total_queued   = sum(l["cars"] for l in lanes.values())
        max_wait       = max(l["avg_wait"] for l in lanes.values())
        wait_times     = [l["avg_wait"] for l in lanes.values()]
        fairness       = float(np.var(wait_times))
        starvation     = sum(
            1 for l in lanes.values()
            if (tick - l["last_served"]) > STARVATION_LIMIT
        )

        throughput_r   =  0.5 * cars_cleared
        queue_r        = -0.1 * total_queued
        max_wait_r     = -0.3 * max_wait
        starvation_r   = -0.5 * starvation
        fairness_r     = -0.2 * fairness

        emergency_r = 0.0
        if s["emergency_lane"] is not None:
            elapsed = tick - s["emergency_tick_started"]
            if elapsed > 45:
                emergency_r = -20.0
                s["emergency_lane"] = None
            elif s["current_green"] == s["emergency_lane"]:
                emergency_r = +10.0
                s["emergency_lane"] = None

        total = (throughput_r + queue_r + max_wait_r +
                 starvation_r + fairness_r + emergency_r)

        return Reward(
            total=round(total, 3),
            breakdown={
                "throughput":  round(throughput_r, 3),
                "queue":       round(queue_r, 3),
                "max_wait":    round(max_wait_r, 3),
                "starvation":  round(starvation_r, 3),
                "fairness":    round(fairness_r, 3),
                "emergency":   round(emergency_r, 3),
            }
        )

    # ── termination ─────────────────────────

    def _check_done(self) -> bool:
        s = self._state
        if s["tick"] >= self.episode_length:
            return True
        if s["emergency_lane"] is not None:
            elapsed = s["tick"] - s["emergency_tick_started"]
            if elapsed > 45:
                return True
        return False

    # ── helpers ─────────────────────────────

    def _seed_initial_cars(self):
        seeds = {
            "easy":   {"N": 4, "S": 2, "E": 3, "W": 1},
            "medium": {"N": 6, "S": 4, "E": 8, "W": 3},
            "hard":   {"N": 10, "S": 7, "E": 12, "W": 5},
        }[self.task]
        for lane, count in seeds.items():
            self._state["lanes"][lane]["cars"] = count

    def _to_observation(self) -> Observation:
        s = self._state
        l = s["lanes"]
        return Observation(
            north=LaneState(cars=l["N"]["cars"], avg_wait=round(l["N"]["avg_wait"], 2)),
            south=LaneState(cars=l["S"]["cars"], avg_wait=round(l["S"]["avg_wait"], 2)),
            east= LaneState(cars=l["E"]["cars"], avg_wait=round(l["E"]["avg_wait"], 2)),
            west= LaneState(cars=l["W"]["cars"], avg_wait=round(l["W"]["avg_wait"], 2)),
            current_green=s["current_green"],
            time_in_phase=s["time_in_phase"],
            emergency_lane=s["emergency_lane"],
            pedestrian_requests=s["pedestrian_requests"],
            pedestrian_active=s["pedestrian_active"],
            pedestrian_ticks_remaining=s["pedestrian_ticks_remaining"],
            time_of_day=s["time_of_day"],
            tick=s["tick"],
        )
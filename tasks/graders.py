from __future__ import annotations
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.traffic_env import TrafficEnv, Action
from typing import Literal

# ─────────────────────────────────────────
#  GRADER BASE
# ─────────────────────────────────────────

def _run_episode(task: Literal["easy", "medium", "hard"], agent_fn) -> dict:
    """
    Runs a full episode using agent_fn to pick actions.
    agent_fn receives an Observation and returns an Action.
    Returns raw episode statistics for graders to score.
    """
    env = TrafficEnv(task=task)
    obs = env.reset()

    total_reward        = 0.0
    total_cleared       = 0
    wait_time_samples   = []
    emergency_response_times = []
    starvation_events   = 0
    emergency_failed    = False

    active_emergency_tick = None

    while True:
        action = agent_fn(obs)
        new_obs, reward, done, _ = env.step(action)

        total_reward  += reward.total
        total_cleared += reward.breakdown["throughput"] / 0.5

        wait_time_samples.append(
            max(obs.north.avg_wait, obs.south.avg_wait,
                obs.east.avg_wait,  obs.west.avg_wait)
        )

        if obs.emergency_lane and active_emergency_tick is None:
            active_emergency_tick = obs.tick

        if active_emergency_tick is not None and obs.emergency_lane is None:
            response_time = obs.tick - active_emergency_tick
            emergency_response_times.append(response_time)
            active_emergency_tick = None

        if reward.breakdown["starvation"] < 0:
            starvation_events += 1

        if reward.breakdown["emergency"] == -20.0:
            emergency_failed = True

        obs = new_obs
        if done:
            break

    avg_wait        = sum(wait_time_samples) / max(1, len(wait_time_samples))
    avg_response    = (
        sum(emergency_response_times) / len(emergency_response_times)
        if emergency_response_times else None
    )

    return {
        "total_reward":       round(total_reward, 2),
        "avg_wait":           round(avg_wait, 2),
        "total_cleared":      int(total_cleared),
        "starvation_events":  starvation_events,
        "emergency_response": avg_response,
        "emergency_failed":   emergency_failed,
        "ticks":              obs.tick,
    }


# ─────────────────────────────────────────
#  TASK 1 — EASY
#  Normal traffic, clear right answer exists
#  Win: avg_wait < 30s, zero starvation
# ─────────────────────────────────────────

def grade_easy(agent_fn) -> float:
    stats = _run_episode("easy", agent_fn)

    # component 1: wait time score (0.0 → 1.0)
    # perfect = avg_wait of 0, fail = avg_wait of 50+
    wait_score = max(0.0, 1.0 - stats["avg_wait"] / 50.0)

    # component 2: starvation penalty
    # each starvation event deducts 0.05, capped at full deduction
    starvation_penalty = min(1.0, stats["starvation_events"] * 0.05)

    score = wait_score * (1.0 - starvation_penalty)
    return round(max(0.0, min(1.0, score)), 3)


# ─────────────────────────────────────────
#  TASK 2 — MEDIUM
#  Uneven traffic + pedestrians
#  Win: avg_wait < 25s, all lanes served, fairness
# ─────────────────────────────────────────

def grade_medium(agent_fn) -> float:
    stats = _run_episode("medium", agent_fn)

    # component 1: wait time (forgiving threshold)
    wait_score = max(0.0, 1.0 - stats["avg_wait"] / 60.0)

    # component 2: starvation (softer penalty)
    starvation_penalty = min(1.0, stats["starvation_events"] * 0.03)

    # component 3: throughput (easier benchmark)
    throughput_score = min(1.0, stats["total_cleared"] / 150.0)

    score = (
        0.5 * wait_score +
        0.3 * throughput_score +
        0.2 * (1.0 - starvation_penalty)
    )
    return round(max(0.0, min(1.0, score)), 3)

# ─────────────────────────────────────────
#  TASK 3 — HARD
#  Rush hour + emergency vehicles
#  Win: emergency cleared <15s, avg_wait <35s, no starvation
# ─────────────────────────────────────────

def grade_hard(agent_fn) -> float:
    stats = _run_episode("hard", agent_fn)

    # component 1: emergency handling
    if stats["emergency_failed"]:
        emergency_score = 0.0
    elif stats["emergency_response"] is None:
        emergency_score = 1.0          # no emergency spawned this episode
    else:
        # perfect = 0s response, fail = 45s+
        emergency_score = max(0.0, 1.0 - stats["emergency_response"] / 45.0)
        # bonus: sub-15s response
        if stats["emergency_response"] <= 15:
            emergency_score = min(1.0, emergency_score + 0.2)

    # component 2: wait time under pressure
    wait_score = max(0.0, 1.0 - stats["avg_wait"] / 60.0)

    # component 3: starvation during peak
    starvation_penalty = min(1.0, stats["starvation_events"] * 0.1)

    score = (
        0.5 * emergency_score +
        0.3 * wait_score +
        0.2 * (1.0 - starvation_penalty)
    )
    return round(max(0.0, min(1.0, score)), 3)


# ─────────────────────────────────────────
#  CONVENIENCE — run all three
# ─────────────────────────────────────────

def grade_all(agent_fn) -> dict:
    return {
        "easy":   grade_easy(agent_fn),
        "medium": grade_medium(agent_fn),
        "hard":   grade_hard(agent_fn),
    }


# ─────────────────────────────────────────
#  QUICK TEST — dummy agent
# ─────────────────────────────────────────

if __name__ == "__main__":
    from environment.traffic_env import Action

    def dummy_agent(obs):
        """Always keeps the current signal — baseline floor agent."""
        return Action(action="keep")

    print("Running graders with dummy agent...")
    scores = grade_all(dummy_agent)
    print(f"  Easy:   {scores['easy']}")
    print(f"  Medium: {scores['medium']}")
    print(f"  Hard:   {scores['hard']}")
    print("Graders working correctly.")
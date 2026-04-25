"""
Task Graders v2.0
=================

Updated to match the v2 reward scale and to grade the three new
scenarios (night, rush_hour, chaos).

Scoring philosophy
------------------
Each grader produces a score in [0, 1]. The component weights inside
each grader are tuned to test the *capability* the scenario is designed
to stress — wait-time discipline for easy/medium, emergency response
for hard/chaos, switch efficiency for night, throughput for rush_hour.
"""
from __future__ import annotations
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.traffic_env import TrafficEnv, Action, REWARD_WEIGHTS
from typing import Optional


# ============================================================
#  CORE EPISODE RUNNER
# ============================================================

def _run_episode(task: str, agent_fn, seed: Optional[int] = None) -> dict:
    """
    Runs a full episode and returns aggregated statistics for graders.

    Notes on metric extraction
    --------------------------
    The v2 reward decomposes into a `breakdown` dict. We reverse-engineer
    cars-cleared per tick from `breakdown["throughput"]`:

        throughput_r        = cars_cleared / 8                  (raw)
        breakdown["throughput"] = w_throughput * throughput_r   (weighted)
        cars_cleared        = breakdown["throughput"] * 8 / w_throughput

    Emergency-failure is detected when the (weighted) emergency component
    drops below a threshold — in v2 the worst-case is `w_emergency * -2.0`
    which equals -6.0, well below our -1.0 detection threshold and clearly
    distinct from any positive emergency reward.
    """
    env = TrafficEnv(task=task, seed=seed)
    obs = env.reset()

    total_reward             = 0.0
    total_cleared            = 0
    wait_time_samples        = []
    emergency_response_times = []
    starvation_events        = 0
    emergency_failed         = False
    switch_count             = 0
    active_emergency_tick    = None

    thr_w = REWARD_WEIGHTS["throughput"]
    sw_w  = REWARD_WEIGHTS["switch"]

    while True:
        action = agent_fn(obs)
        new_obs, reward, done, _ = env.step(action)

        total_reward += reward.total

        # cars-cleared this tick (see docstring above for derivation)
        cars_this_tick = round(reward.breakdown["throughput"] * 8.0 / thr_w)
        total_cleared += max(0, cars_this_tick)

        wait_time_samples.append(
            max(obs.north.avg_wait, obs.south.avg_wait,
                obs.east.avg_wait,  obs.west.avg_wait)
        )

        # Track emergencies appearing → being resolved
        if obs.emergency_lane and active_emergency_tick is None:
            active_emergency_tick = obs.tick
        if active_emergency_tick is not None and new_obs.emergency_lane is None:
            response_time = new_obs.tick - active_emergency_tick
            emergency_response_times.append(response_time)
            active_emergency_tick = None

        if reward.breakdown["starvation"] < 0:
            starvation_events += 1

        # v2: emergency hard-failure manifests as a strong negative emergency
        # component (weighted = w_emergency * -2.0). Threshold below -1.0
        # cleanly distinguishes failure from any positive response reward.
        if reward.breakdown["emergency"] <= -1.0:
            emergency_failed = True

        # v2: switch tracking
        if reward.breakdown["switch"] < 0:
            switch_count += 1

        obs = new_obs
        if done:
            break

    avg_wait = sum(wait_time_samples) / max(1, len(wait_time_samples))
    avg_response = (
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
        "switch_count":       switch_count,
        "ticks":              obs.tick,
    }


# ============================================================
#  EXISTING SCENARIO GRADERS (signatures preserved)
# ============================================================

def grade_easy(agent_fn) -> float:
    """Easy: low waits, no starvation."""
    stats = _run_episode("easy", agent_fn)
    wait_score         = max(0.0, 1.0 - stats["avg_wait"] / 50.0)
    starvation_penalty = min(1.0, stats["starvation_events"] * 0.05)
    score = wait_score * (1.0 - starvation_penalty)
    return round(max(0.0, min(1.0, score)), 3)


def grade_medium(agent_fn) -> float:
    """Medium: balance wait, throughput, starvation under uneven load."""
    stats = _run_episode("medium", agent_fn)
    wait_score         = max(0.0, 1.0 - stats["avg_wait"] / 60.0)
    starvation_penalty = min(1.0, stats["starvation_events"] * 0.03)
    throughput_score   = min(1.0, stats["total_cleared"] / 150.0)
    score = (
        0.5 * wait_score +
        0.3 * throughput_score +
        0.2 * (1.0 - starvation_penalty)
    )
    return round(max(0.0, min(1.0, score)), 3)


def grade_hard(agent_fn) -> float:
    """Hard: emergency response is the dominant component."""
    stats = _run_episode("hard", agent_fn)
    if stats["emergency_failed"]:
        emergency_score = 0.0
    elif stats["emergency_response"] is None:
        emergency_score = 1.0
    else:
        emergency_score = max(0.0, 1.0 - stats["emergency_response"] / 45.0)
        if stats["emergency_response"] <= 15:
            emergency_score = min(1.0, emergency_score + 0.2)

    wait_score         = max(0.0, 1.0 - stats["avg_wait"] / 60.0)
    starvation_penalty = min(1.0, stats["starvation_events"] * 0.1)
    score = (
        0.5 * emergency_score +
        0.3 * wait_score +
        0.2 * (1.0 - starvation_penalty)
    )
    return round(max(0.0, min(1.0, score)), 3)


# ============================================================
#  v2 NEW SCENARIO GRADERS
# ============================================================

def grade_night(agent_fn) -> float:
    """
    Night: minimum-intervention test.

    A naive 'always switch to busiest' agent thrashes here because
    every queue is tiny — the switch-cost weight punishes that.
    Components:
      50% wait-time discipline (tight standard: <30s ideal)
      30% switch efficiency (<60 switches in 200 ticks)
      20% no-starvation
    """
    stats = _run_episode("night", agent_fn)
    wait_score         = max(0.0, 1.0 - stats["avg_wait"] / 30.0)
    switch_score       = max(0.0, 1.0 - stats["switch_count"] / 60.0)
    starvation_penalty = min(1.0, stats["starvation_events"] * 0.05)
    score = (
        0.5 * wait_score +
        0.3 * switch_score +
        0.2 * (1.0 - starvation_penalty)
    )
    return round(max(0.0, min(1.0, score)), 3)


def grade_rush_hour(agent_fn) -> float:
    """
    Rush hour: throughput-dominant grader.

    Sustained PEAK_HOUR for 250 ticks. Bundle actions are valuable
    because two opposing lanes can be cleared per tick. Failure mode
    is gridlock, signaled by ballooning queues and avg_wait.
    Components:
      50% throughput (target 600 cleared cars across 250 ticks)
      30% wait-time stability under load
      20% no-starvation
    """
    stats = _run_episode("rush_hour", agent_fn)
    throughput_score   = min(1.0, stats["total_cleared"] / 600.0)
    wait_score         = max(0.0, 1.0 - stats["avg_wait"] / 80.0)
    starvation_penalty = min(1.0, stats["starvation_events"] * 0.05)
    score = (
        0.5 * throughput_score +
        0.3 * wait_score +
        0.2 * (1.0 - starvation_penalty)
    )
    return round(max(0.0, min(1.0, score)), 3)


def grade_chaos(agent_fn) -> float:
    """
    Chaos: full-mechanic final boss.

    Components weighted to reflect the multi-objective nature:
      40% emergency response (must still resolve incidents)
      25% wait-time stability across regime shifts
      20% throughput (handle peak phase well)
      15% no-starvation (despite incidents)
    """
    stats = _run_episode("chaos", agent_fn)
    if stats["emergency_failed"]:
        emergency_score = 0.0
    elif stats["emergency_response"] is None:
        emergency_score = 1.0
    else:
        emergency_score = max(0.0, 1.0 - stats["emergency_response"] / 45.0)
        if stats["emergency_response"] <= 15:
            emergency_score = min(1.0, emergency_score + 0.2)

    wait_score         = max(0.0, 1.0 - stats["avg_wait"] / 70.0)
    throughput_score   = min(1.0, stats["total_cleared"] / 700.0)
    starvation_penalty = min(1.0, stats["starvation_events"] * 0.05)
    score = (
        0.40 * emergency_score +
        0.25 * wait_score +
        0.20 * throughput_score +
        0.15 * (1.0 - starvation_penalty)
    )
    return round(max(0.0, min(1.0, score)), 3)


# ============================================================
#  v2.1 NEW SCENARIO GRADERS (hackathon-guide-derived)
# ============================================================

def grade_tutorial(agent_fn) -> float:
    """
    Curriculum-bootstrap grader (Guide §6 — make success possible early).

    Forgiving thresholds. Even a `keep` agent should clear ~50% throughput
    here because the default green is already on the only loaded lane.
    Score is dominated by total_cleared, so the agent always sees a
    non-trivial gradient even if its policy is bad.
    """
    stats = _run_episode("tutorial", agent_fn)
    # tutorial has 150 ticks, average ~0.55 cars/tick arriving total
    # (0.4 + 3*0.05) → ~82 cars + 8 initial = ~90 expected.
    # Generous threshold: 60 cleared = full marks.
    throughput_score = min(1.0, stats["total_cleared"] / 60.0)
    wait_score       = max(0.0, 1.0 - stats["avg_wait"] / 25.0)
    starve_penalty   = min(1.0, stats["starvation_events"] * 0.05)
    score = (
        0.6 * throughput_score +
        0.3 * wait_score +
        0.1 * (1.0 - starve_penalty)
    )
    return round(max(0.0, min(1.0, score)), 3)


def grade_asymmetric(agent_fn) -> float:
    """
    Anti-reward-hacking grader (Guide §7-8 — multiple checks, hack detection).

    The episode has heavy E+W traffic, near-zero N+S. We grade on:
      40% throughput (must actually clear E+W)
      25% E+W-specific service ratio (catches agents that ignore loaded lanes)
      20% wait stability
      15% no-starvation

    The `ew_service_ratio` is an INDEPENDENT reward signal — the guide
    explicitly recommends multiple independent checks so the model can't
    optimize one component at the others' expense.
    """
    # Custom episode runner so we can track lane-level service.
    from environment.traffic_env import TrafficEnv, Action  # local import
    env = TrafficEnv(task="asymmetric")
    obs = env.reset()

    total_cleared = 0
    ew_cleared    = 0
    ns_cleared    = 0
    waits         = []
    starve_events = 0
    thr_w         = REWARD_WEIGHTS["throughput"]

    while True:
        action = agent_fn(obs)
        prev_e = obs.east.cars
        prev_w = obs.west.cars
        prev_n = obs.north.cars
        prev_s = obs.south.cars

        new_obs, reward, done, _ = env.step(action)

        cars_this_tick = round(reward.breakdown["throughput"] * 8.0 / thr_w)
        total_cleared += max(0, cars_this_tick)
        # Lane-level cleared = max(0, prev - new) per lane (arrivals are small
        # enough on the empty axis that this stays a fair approximation).
        ew_delta = max(0, prev_e - new_obs.east.cars) + max(0, prev_w - new_obs.west.cars)
        ns_delta = max(0, prev_n - new_obs.north.cars) + max(0, prev_s - new_obs.south.cars)
        ew_cleared += ew_delta
        ns_cleared += ns_delta

        waits.append(max(new_obs.north.avg_wait, new_obs.south.avg_wait,
                         new_obs.east.avg_wait,  new_obs.west.avg_wait))
        if reward.breakdown["starvation"] < 0:
            starve_events += 1

        obs = new_obs
        if done:
            break

    avg_wait = sum(waits) / max(1, len(waits))

    # Independent components:
    throughput_score = min(1.0, total_cleared / 350.0)
    # ew_service_ratio: of all cars cleared, what fraction were on the busy axis?
    # Optimal is ~1.0 because N+S barely have any cars. A bundle_NS shortcut
    # agent will produce ratio close to 0.
    ew_service_ratio = ew_cleared / max(1, ew_cleared + ns_cleared)
    wait_score       = max(0.0, 1.0 - avg_wait / 80.0)
    starve_penalty   = min(1.0, starve_events * 0.05)

    score = (
        0.40 * throughput_score +
        0.25 * ew_service_ratio +
        0.20 * wait_score +
        0.15 * (1.0 - starve_penalty)
    )
    return round(max(0.0, min(1.0, score)), 3)


def grade_deterministic(agent_fn) -> float:
    """
    Verifiable-reward grader (Guide §11 — RLVR / verifiable rewards).

    Arrivals are scripted (no Poisson), so optimal throughput is computable:
      - 8-tick cycle delivers 32 cars total (8 per lane)
      - FLOW_RATE = 4 (single) or 8 (bundle)
      - 200 ticks = 25 cycles → ~800 cars in + 16 initial = ~816 total
      - A near-optimal agent clears ~780+ cars

    Use this scenario as a regression test for the reward function — see
    `deterministic_optimal_score()` below for a known-optimal benchmark.
    """
    stats = _run_episode("deterministic", agent_fn)
    # Calibrated against the deterministic_optimal_score() benchmark below
    # (which scores ~700-750 cleared with a hand-tuned alternating policy).
    throughput_score = min(1.0, stats["total_cleared"] / 700.0)
    wait_score       = max(0.0, 1.0 - stats["avg_wait"] / 50.0)
    starve_penalty   = min(1.0, stats["starvation_events"] * 0.05)
    score = (
        0.6 * throughput_score +
        0.3 * wait_score +
        0.1 * (1.0 - starve_penalty)
    )
    return round(max(0.0, min(1.0, score)), 3)


def deterministic_optimal_score() -> dict:
    """
    Run a hand-tuned near-optimal policy on the `deterministic` scenario.

    Used as a regression test for the reward function. If a refactor causes
    this score to change unexpectedly, the reward function probably drifted.
    The guide (§11, §15) explicitly recommends building a verifier first
    and watching for unexplained reward changes.

    Optimal-policy strategy: bundle_NS for 4 ticks, then bundle_EW for 4 ticks,
    repeat. Each bundle clears 8 cars/tick on its axis, matching the per-cycle
    arrival rate.
    """
    from environment.traffic_env import TrafficEnv, Action  # local import

    def optimal_policy(obs):
        # Alternate axes every 4 ticks
        return Action(action="bundle_NS" if (obs.tick // 4) % 2 == 0 else "bundle_EW")

    stats = _run_episode("deterministic", optimal_policy)
    score = grade_deterministic(optimal_policy)
    return {
        "score":             score,
        "total_cleared":     stats["total_cleared"],
        "avg_wait":          stats["avg_wait"],
        "starvation_events": stats["starvation_events"],
        "policy":            "alternating bundle_NS / bundle_EW every 4 ticks",
    }


def grade_pedestrian_heavy(agent_fn) -> float:
    """
    Adaptive-tradeoff grader (Guide §19 — evidence of improvement).

    Forces a context-dependent decision: when to honor pedestrian crossings
    vs. prioritize cars. Rule-based uses a fixed threshold (`total_cars < 15`)
    which is suboptimal under heavy ped flow — gives learned agents room
    to differentiate.

    Multiple INDEPENDENT components (per Guide §7-8 "multiple independent
    reward functions"):
      35% throughput        — must still clear cars
      30% wait stability    — can't let queues balloon while serving peds
      20% ped responsiveness — peds shouldn't sit unserved either
      15% no-starvation     — basic correctness check

    The 'ped_responsiveness' component is the independent signal that
    catches "ignore-all-pedestrians" and "always-hold" failure modes
    simultaneously.
    """
    from environment.traffic_env import TrafficEnv, Action  # local import

    env = TrafficEnv(task="pedestrian_heavy")
    obs = env.reset()

    total_cleared       = 0
    waits               = []
    ped_backlog_samples = []
    ped_holds_used      = 0
    starve_events       = 0
    thr_w               = REWARD_WEIGHTS["throughput"]

    while True:
        action = agent_fn(obs)
        new_obs, reward, done, _ = env.step(action)

        cars_this_tick = round(reward.breakdown["throughput"] * 8.0 / thr_w)
        total_cleared += max(0, cars_this_tick)

        waits.append(max(new_obs.north.avg_wait, new_obs.south.avg_wait,
                         new_obs.east.avg_wait,  new_obs.west.avg_wait))
        ped_backlog_samples.append(len(new_obs.pedestrian_requests))

        if action.action == "pedestrian_hold":
            ped_holds_used += 1
        if reward.breakdown["starvation"] < 0:
            starve_events += 1

        obs = new_obs
        if done:
            break

    avg_wait        = sum(waits) / max(1, len(waits))
    avg_ped_backlog = sum(ped_backlog_samples) / max(1, len(ped_backlog_samples))

    # Independent components:
    throughput_score = min(1.0, total_cleared / 280.0)   # ~70% of arrivals
    wait_score       = max(0.0, 1.0 - avg_wait / 70.0)
    # Ped responsiveness: backlog should rarely exceed 2 cars on average
    ped_score        = max(0.0, 1.0 - avg_ped_backlog / 4.0)
    starve_penalty   = min(1.0, starve_events * 0.05)

    score = (
        0.35 * throughput_score +
        0.30 * wait_score +
        0.20 * ped_score +
        0.15 * (1.0 - starve_penalty)
    )
    return round(max(0.0, min(1.0, score)), 3)


# ============================================================
#  v2.3 NEW SCENARIO GRADERS (real-world domain events)
# ============================================================

def grade_ambulance_run(agent_fn) -> float:
    """
    Ambulance-handling grader. The scenario has 6 scheduled emergencies.
    We grade primarily on response time, with throughput as a secondary
    signal (an agent that lets queues balloon to 'win' on emergencies
    is still failing).
      60% emergency response (per-event mean response time, capped at 45)
      25% wait stability      (queues don't pile up between emergencies)
      15% throughput          (cars still get cleared)
    """
    from environment.traffic_env import TrafficEnv, Action  # local import

    env = TrafficEnv(task="ambulance_run")
    obs = env.reset()

    response_times    = []
    waits             = []
    total_cleared     = 0
    em_failed         = False
    active_em_started = None
    thr_w             = REWARD_WEIGHTS["throughput"]

    while True:
        action = agent_fn(obs)
        new_obs, reward, done, _ = env.step(action)

        cars = round(reward.breakdown["throughput"] * 8.0 / thr_w)
        total_cleared += max(0, cars)
        waits.append(max(new_obs.north.avg_wait, new_obs.south.avg_wait,
                         new_obs.east.avg_wait,  new_obs.west.avg_wait))

        # Track emergency rising/falling edges
        if obs.emergency_lane and active_em_started is None:
            active_em_started = obs.tick
        if active_em_started is not None and new_obs.emergency_lane is None:
            response_times.append(new_obs.tick - active_em_started)
            active_em_started = None

        if reward.breakdown["emergency"] <= -1.0:
            em_failed = True

        obs = new_obs
        if done:
            break

    avg_response = sum(response_times) / len(response_times) if response_times else 45.0
    em_score     = 0.0 if em_failed else max(0.0, 1.0 - avg_response / 45.0)
    if response_times and avg_response <= 15:
        em_score = min(1.0, em_score + 0.2)

    avg_wait     = sum(waits) / max(1, len(waits))
    wait_score   = max(0.0, 1.0 - avg_wait / 60.0)
    thr_score    = min(1.0, total_cleared / 350.0)

    score = 0.60 * em_score + 0.25 * wait_score + 0.15 * thr_score
    return round(max(0.0, min(1.0, score)), 3)


def grade_vip_convoy(agent_fn) -> float:
    """
    VIP convoy grader. Two convoys traverse the intersection during the
    episode. Agent must build a green corridor BEFORE arrival (8-tick lead
    time provided via vip_eta_*). A "miss" = convoy hit a non-green lane
    on any tick during traversal.
      55% VIP corridor success (fraction of traversal ticks with green held)
      30% throughput between convoys (agent still clears cars)
      15% wait stability
    """
    from environment.traffic_env import TrafficEnv, Action  # local import

    env = TrafficEnv(task="vip_convoy")
    obs = env.reset()

    vip_active_ticks   = 0
    vip_green_ticks    = 0
    waits              = []
    total_cleared      = 0
    thr_w              = REWARD_WEIGHTS["throughput"]

    while True:
        action = agent_fn(obs)
        new_obs, reward, done, _ = env.step(action)

        if new_obs.vip_active:
            vip_active_ticks += 1
            if new_obs.current_green == new_obs.vip_eta_lane:
                vip_green_ticks += 1

        cars = round(reward.breakdown["throughput"] * 8.0 / thr_w)
        total_cleared += max(0, cars)
        waits.append(max(new_obs.north.avg_wait, new_obs.south.avg_wait,
                         new_obs.east.avg_wait,  new_obs.west.avg_wait))

        obs = new_obs
        if done:
            break

    vip_score    = vip_green_ticks / max(1, vip_active_ticks)
    avg_wait     = sum(waits) / max(1, len(waits))
    wait_score   = max(0.0, 1.0 - avg_wait / 50.0)
    thr_score    = min(1.0, total_cleared / 400.0)

    score = 0.55 * vip_score + 0.30 * thr_score + 0.15 * wait_score
    return round(max(0.0, min(1.0, score)), 3)


def grade_monsoon(agent_fn) -> float:
    """
    Monsoon grader. Weather reduces flow rate and accelerates wait
    accumulation. Tests whether the agent recognizes degraded throughput
    and adapts (e.g., uses bundles more aggressively to compensate).
      45% throughput (must keep up despite reduced flow)
      30% wait stability (queues don't run away in heavy rain)
      15% emergency response (still need to handle ambulances)
      10% no-starvation
    """
    stats = _run_episode("monsoon", agent_fn)
    if stats["emergency_failed"]:
        em_score = 0.0
    elif stats["emergency_response"] is None:
        em_score = 1.0
    else:
        em_score = max(0.0, 1.0 - stats["emergency_response"] / 45.0)
        if stats["emergency_response"] <= 15:
            em_score = min(1.0, em_score + 0.2)

    thr_score   = min(1.0, stats["total_cleared"] / 400.0)
    wait_score  = max(0.0, 1.0 - stats["avg_wait"] / 90.0)
    starve_pen  = min(1.0, stats["starvation_events"] * 0.05)

    score = (
        0.45 * thr_score +
        0.30 * wait_score +
        0.15 * em_score +
        0.10 * (1.0 - starve_pen)
    )
    return round(max(0.0, min(1.0, score)), 3)


def grade_orchestrated_signals(agent_fn) -> float:
    """
    Final-boss grader for orchestrated_signals. Combines VIP, ambulance,
    weather, and incident handling into a single multi-objective score.
      30% VIP corridor success
      30% ambulance response
      20% throughput (despite peak hour + heavy rain)
      10% wait stability
      10% no-starvation
    """
    from environment.traffic_env import TrafficEnv, Action  # local import

    env = TrafficEnv(task="orchestrated_signals")
    obs = env.reset()

    vip_active_ticks  = 0
    vip_green_ticks   = 0
    response_times    = []
    em_failed         = False
    active_em_started = None
    waits             = []
    total_cleared     = 0
    starve_events     = 0
    thr_w             = REWARD_WEIGHTS["throughput"]

    while True:
        action = agent_fn(obs)
        new_obs, reward, done, _ = env.step(action)

        if new_obs.vip_active:
            vip_active_ticks += 1
            if new_obs.current_green == new_obs.vip_eta_lane:
                vip_green_ticks += 1

        if obs.emergency_lane and active_em_started is None:
            active_em_started = obs.tick
        if active_em_started is not None and new_obs.emergency_lane is None:
            response_times.append(new_obs.tick - active_em_started)
            active_em_started = None
        if reward.breakdown["emergency"] <= -1.0:
            em_failed = True

        cars = round(reward.breakdown["throughput"] * 8.0 / thr_w)
        total_cleared += max(0, cars)
        waits.append(max(new_obs.north.avg_wait, new_obs.south.avg_wait,
                         new_obs.east.avg_wait,  new_obs.west.avg_wait))
        if reward.breakdown["starvation"] < 0:
            starve_events += 1

        obs = new_obs
        if done:
            break

    vip_score = vip_green_ticks / max(1, vip_active_ticks) if vip_active_ticks else 1.0
    avg_resp  = sum(response_times) / len(response_times) if response_times else 45.0
    em_score  = 0.0 if em_failed else max(0.0, 1.0 - avg_resp / 45.0)
    if response_times and avg_resp <= 15:
        em_score = min(1.0, em_score + 0.2)
    thr_score   = min(1.0, total_cleared / 600.0)
    avg_wait    = sum(waits) / max(1, len(waits))
    wait_score  = max(0.0, 1.0 - avg_wait / 80.0)
    starve_pen  = min(1.0, starve_events * 0.05)

    score = (
        0.30 * vip_score +
        0.30 * em_score +
        0.20 * thr_score +
        0.10 * wait_score +
        0.10 * (1.0 - starve_pen)
    )
    return round(max(0.0, min(1.0, score)), 3)


def grade_all(agent_fn) -> dict:
    return {
        "easy":             grade_easy(agent_fn),
        "medium":           grade_medium(agent_fn),
        "hard":             grade_hard(agent_fn),
        "night":            grade_night(agent_fn),
        "rush_hour":        grade_rush_hour(agent_fn),
        "chaos":            grade_chaos(agent_fn),
        "tutorial":         grade_tutorial(agent_fn),
        "asymmetric":       grade_asymmetric(agent_fn),
        "deterministic":    grade_deterministic(agent_fn),
        "pedestrian_heavy": grade_pedestrian_heavy(agent_fn),
        "ambulance_run":     grade_ambulance_run(agent_fn),
        "vip_convoy":        grade_vip_convoy(agent_fn),
        "monsoon":           grade_monsoon(agent_fn),
        "orchestrated_signals": grade_orchestrated_signals(agent_fn),
    }


# ============================================================
#  QUICK SMOKE TEST
# ============================================================

if __name__ == "__main__":
    def dummy_agent(obs):
        return Action(action="keep")

    def rule_based_agent(obs):
        # Emergency first
        if obs.emergency_lane:
            return Action(action=f"switch_to_{obs.emergency_lane}")
        # Pedestrian when load is light
        total_cars = (obs.north.cars + obs.south.cars +
                      obs.east.cars  + obs.west.cars)
        if obs.pedestrian_requests and not obs.pedestrian_active and total_cars < 15:
            return Action(action="pedestrian_hold")
        # Score lanes by (cars + wait) — natural starvation defense
        lanes = {
            "N": obs.north.cars + obs.north.avg_wait,
            "S": obs.south.cars + obs.south.avg_wait,
            "E": obs.east.cars  + obs.east.avg_wait,
            "W": obs.west.cars  + obs.west.avg_wait,
        }
        # v2: don't switch into a closed (incident) lane
        if obs.incident_lane:
            lanes.pop(obs.incident_lane, None)
        busiest = max(lanes, key=lanes.get)
        if busiest != obs.current_green:
            return Action(action=f"switch_to_{busiest}")
        return Action(action="keep")

    print("\n=== Dummy agent (always 'keep') ===")
    s = grade_all(dummy_agent)
    for k, v in s.items():
        print(f"  {k:12s}: {v}")
    print(f"  {'AVERAGE':12s}: {round(sum(s.values())/len(s), 3)}")

    print("\n=== Rule-based agent ===")
    s = grade_all(rule_based_agent)
    for k, v in s.items():
        print(f"  {k:12s}: {v}")
    print(f"  {'AVERAGE':12s}: {round(sum(s.values())/len(s), 3)}")

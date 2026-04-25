"""
Benchmark — v2 Traffic Signal Environment
==========================================

Runs three reference agents across all six scenarios, multiple seeds,
and reports mean ± std.  Useful for hackathon demo / leaderboard.

Usage:
    python benchmark.py                # 5 seeds per scenario (default)
    python benchmark.py --seeds 20     # tighter confidence intervals
"""
from __future__ import annotations
import argparse
import statistics
from environment.traffic_env import TrafficEnv, Action
from tasks.graders import grade_all, _run_episode


# ============================================================
#  REFERENCE AGENTS
# ============================================================

def dummy_agent(obs):
    """Floor agent — never switches. Useful as a sanity-check baseline."""
    return Action(action="keep")


def round_robin_agent(obs):
    """
    Cycles through N → E → S → W every 30 ticks.
    Demonstrates a fixed-time controller (what real intersections used
    before adaptive control became viable).
    """
    cycle = ["N", "E", "S", "W"]
    target = cycle[(obs.tick // 30) % 4]
    if target != obs.current_green:
        return Action(action=f"switch_to_{target}")
    return Action(action="keep")


def rule_based_agent(obs):
    """
    Strong reactive baseline. Priority order:
      1. Emergency vehicle → switch to its lane immediately
      2. v2.3: VIP convoy active or arriving → pre-clear corridor
      3. Pedestrian held when overall load is light
      4. Otherwise, serve the lane with highest (cars + avg_wait) score
      5. Avoid switching INTO a blocked (incident) lane (v2)
    """
    if obs.emergency_lane:
        return Action(action=f"switch_to_{obs.emergency_lane}")
    # v2.3: VIP corridor (active or imminent)
    if obs.vip_active and obs.current_green != obs.vip_eta_lane:
        return Action(action=f"switch_to_{obs.vip_eta_lane}")
    if (obs.vip_eta_lane is not None
            and obs.vip_eta_ticks <= 4
            and obs.current_green != obs.vip_eta_lane):
        return Action(action=f"switch_to_{obs.vip_eta_lane}")

    total_cars = (obs.north.cars + obs.south.cars +
                  obs.east.cars  + obs.west.cars)
    if obs.pedestrian_requests and not obs.pedestrian_active and total_cars < 15:
        return Action(action="pedestrian_hold")

    lanes = {
        "N": obs.north.cars + obs.north.avg_wait,
        "S": obs.south.cars + obs.south.avg_wait,
        "E": obs.east.cars  + obs.east.avg_wait,
        "W": obs.west.cars  + obs.west.avg_wait,
    }
    if obs.incident_lane:
        lanes.pop(obs.incident_lane, None)

    busiest = max(lanes, key=lanes.get)
    if busiest != obs.current_green:
        return Action(action=f"switch_to_{busiest}")
    return Action(action="keep")


def smart_bundle_agent(obs):
    """
    Like rule-based but uses bundle_NS / bundle_EW when both opposing
    lanes are congested.  Designed to exploit the rush_hour scenario.
    Also uses bundles aggressively in monsoon when flow rate is reduced.
    """
    if obs.emergency_lane:
        return Action(action=f"switch_to_{obs.emergency_lane}")
    # v2.3: VIP corridor
    if obs.vip_active and obs.current_green != obs.vip_eta_lane:
        return Action(action=f"switch_to_{obs.vip_eta_lane}")
    if (obs.vip_eta_lane is not None
            and obs.vip_eta_ticks <= 4
            and obs.current_green != obs.vip_eta_lane):
        return Action(action=f"switch_to_{obs.vip_eta_lane}")

    total_cars = (obs.north.cars + obs.south.cars +
                  obs.east.cars  + obs.west.cars)
    if obs.pedestrian_requests and not obs.pedestrian_active and total_cars < 15:
        return Action(action="pedestrian_hold")

    # If both N+S busy AND both E+W also busy, prefer whichever pair
    # has more total cars (and isn't blocked by an incident)
    ns_total = obs.north.cars + obs.south.cars
    ew_total = obs.east.cars  + obs.west.cars
    ns_blocked = obs.incident_lane in ("N", "S")
    ew_blocked = obs.incident_lane in ("E", "W")

    # v2.3: Bundle threshold lower in bad weather (flow rate reduced)
    bundle_threshold = 6 if obs.weather == "HEAVY_RAIN" else 8 if obs.weather == "RAIN" else 10

    if ns_total > bundle_threshold and not ns_blocked and ns_total > ew_total:
        return Action(action="bundle_NS")
    if ew_total > bundle_threshold and not ew_blocked and ew_total > ns_total:
        return Action(action="bundle_EW")

    lanes = {"N": obs.north.cars + obs.north.avg_wait,
             "S": obs.south.cars + obs.south.avg_wait,
             "E": obs.east.cars  + obs.east.avg_wait,
             "W": obs.west.cars  + obs.west.avg_wait}
    if obs.incident_lane:
        lanes.pop(obs.incident_lane, None)
    busiest = max(lanes, key=lanes.get)
    if busiest != obs.current_green:
        return Action(action=f"switch_to_{busiest}")
    return Action(action="keep")


def bundle_ns_cheater(obs):
    """
    Anti-hack target. Always picks bundle_NS regardless of state.
    Should win on dynamics-balanced scenarios (where this is a passable
    shortcut), and lose badly on `asymmetric` where N+S are nearly empty.
    The contrast demonstrates that `asymmetric` correctly detects the hack.
    """
    if obs.emergency_lane:
        return Action(action=f"switch_to_{obs.emergency_lane}")
    return Action(action="bundle_NS")


def aggressive_holds_cheater(obs):
    """
    Anti-hack target. Honors every pedestrian request immediately.
    Looks reasonable for fairness-minded reward signals, but tanks on
    `pedestrian_heavy` where the 15-tick freeze cost dominates the
    pedestrian-served reward. The contrast demonstrates that
    `pedestrian_heavy` correctly detects ped-hold abuse.
    """
    if obs.emergency_lane:
        return Action(action=f"switch_to_{obs.emergency_lane}")
    if obs.pedestrian_requests and not obs.pedestrian_active:
        return Action(action="pedestrian_hold")
    lanes = {"N": obs.north.cars + obs.north.avg_wait,
             "S": obs.south.cars + obs.south.avg_wait,
             "E": obs.east.cars  + obs.east.avg_wait,
             "W": obs.west.cars  + obs.west.avg_wait}
    if obs.incident_lane:
        lanes.pop(obs.incident_lane, None)
    busiest = max(lanes, key=lanes.get)
    if busiest != obs.current_green:
        return Action(action=f"switch_to_{busiest}")
    return Action(action="keep")


# ============================================================
#  BENCHMARK RUNNER
# ============================================================

def benchmark(agent_fn, agent_name: str, seeds: list[int]):
    """Score `agent_fn` on every scenario across multiple seeds."""
    from tasks.graders import (
        grade_easy, grade_medium, grade_hard,
        grade_night, grade_rush_hour, grade_chaos,
        grade_tutorial, grade_asymmetric, grade_deterministic,
        grade_pedestrian_heavy,
        grade_ambulance_run, grade_vip_convoy, grade_monsoon, grade_orchestrated_signals,
    )
    grader_map = {
        "easy":             grade_easy,
        "medium":           grade_medium,
        "hard":             grade_hard,
        "night":            grade_night,
        "rush_hour":        grade_rush_hour,
        "chaos":            grade_chaos,
        "tutorial":         grade_tutorial,
        "asymmetric":       grade_asymmetric,
        "deterministic":    grade_deterministic,
        "pedestrian_heavy": grade_pedestrian_heavy,
        "ambulance_run":      grade_ambulance_run,
        "vip_convoy":         grade_vip_convoy,
        "monsoon":            grade_monsoon,
        "orchestrated_signals": grade_orchestrated_signals,
    }

    results = {task: [] for task in grader_map}
    for seed in seeds:
        # Note: graders call _run_episode without seed; we use a wrapper
        # that injects the seed by patching the rng.
        for task, grader in grader_map.items():
            # v2.3 graders have custom internal loops (they track VIP convoy,
            # weather, scheduled emergency events) that the _run_episode
            # shortcut can't capture. Call those directly.
            if task in ("ambulance_run", "vip_convoy", "monsoon",
                        "orchestrated_signals", "asymmetric", "pedestrian_heavy"):
                results[task].append(grader(agent_fn))
            else:
                stats = _run_episode(task, agent_fn, seed=seed)
                results[task].append(_score_from_stats(task, stats))

    # Aggregate
    summary = {}
    for task, scores in results.items():
        summary[task] = {
            "mean": round(statistics.mean(scores), 3),
            "std":  round(statistics.stdev(scores) if len(scores) > 1 else 0.0, 3),
            "min":  round(min(scores), 3),
            "max":  round(max(scores), 3),
        }
    avg_mean = round(statistics.mean(s["mean"] for s in summary.values()), 3)
    return summary, avg_mean


def _score_from_stats(task: str, stats: dict) -> float:
    """Inline copy of each grader's scoring formula, applied to raw stats."""
    if task == "easy":
        wait = max(0.0, 1.0 - stats["avg_wait"] / 50.0)
        starv = min(1.0, stats["starvation_events"] * 0.05)
        return round(max(0.0, min(1.0, wait * (1.0 - starv))), 3)
    if task == "medium":
        wait = max(0.0, 1.0 - stats["avg_wait"] / 60.0)
        starv = min(1.0, stats["starvation_events"] * 0.03)
        thr = min(1.0, stats["total_cleared"] / 150.0)
        return round(max(0.0, min(1.0, 0.5*wait + 0.3*thr + 0.2*(1.0-starv))), 3)
    if task == "hard":
        if stats["emergency_failed"]:
            em = 0.0
        elif stats["emergency_response"] is None:
            em = 1.0
        else:
            em = max(0.0, 1.0 - stats["emergency_response"] / 45.0)
            if stats["emergency_response"] <= 15:
                em = min(1.0, em + 0.2)
        wait = max(0.0, 1.0 - stats["avg_wait"] / 60.0)
        starv = min(1.0, stats["starvation_events"] * 0.1)
        return round(max(0.0, min(1.0, 0.5*em + 0.3*wait + 0.2*(1.0-starv))), 3)
    if task == "night":
        wait = max(0.0, 1.0 - stats["avg_wait"] / 30.0)
        sw = max(0.0, 1.0 - stats["switch_count"] / 60.0)
        starv = min(1.0, stats["starvation_events"] * 0.05)
        return round(max(0.0, min(1.0, 0.5*wait + 0.3*sw + 0.2*(1.0-starv))), 3)
    if task == "rush_hour":
        thr = min(1.0, stats["total_cleared"] / 600.0)
        wait = max(0.0, 1.0 - stats["avg_wait"] / 80.0)
        starv = min(1.0, stats["starvation_events"] * 0.05)
        return round(max(0.0, min(1.0, 0.5*thr + 0.3*wait + 0.2*(1.0-starv))), 3)
    if task == "chaos":
        if stats["emergency_failed"]:
            em = 0.0
        elif stats["emergency_response"] is None:
            em = 1.0
        else:
            em = max(0.0, 1.0 - stats["emergency_response"] / 45.0)
            if stats["emergency_response"] <= 15:
                em = min(1.0, em + 0.2)
        wait = max(0.0, 1.0 - stats["avg_wait"] / 70.0)
        thr = min(1.0, stats["total_cleared"] / 700.0)
        starv = min(1.0, stats["starvation_events"] * 0.05)
        return round(max(0.0, min(1.0, 0.40*em + 0.25*wait + 0.20*thr + 0.15*(1.0-starv))), 3)

    # ── v2.1 hackathon-guide-derived ──
    if task == "tutorial":
        thr = min(1.0, stats["total_cleared"] / 60.0)
        wait = max(0.0, 1.0 - stats["avg_wait"] / 25.0)
        starv = min(1.0, stats["starvation_events"] * 0.05)
        return round(max(0.0, min(1.0, 0.6*thr + 0.3*wait + 0.1*(1.0-starv))), 3)
    if task == "asymmetric":
        # benchmark.py uses _run_episode shorthand → grader proper does the
        # ew_service_ratio component; here we approximate with throughput-only
        thr = min(1.0, stats["total_cleared"] / 350.0)
        wait = max(0.0, 1.0 - stats["avg_wait"] / 80.0)
        starv = min(1.0, stats["starvation_events"] * 0.05)
        return round(max(0.0, min(1.0, 0.65*thr + 0.20*wait + 0.15*(1.0-starv))), 3)
    if task == "deterministic":
        thr = min(1.0, stats["total_cleared"] / 700.0)
        wait = max(0.0, 1.0 - stats["avg_wait"] / 50.0)
        starv = min(1.0, stats["starvation_events"] * 0.05)
        return round(max(0.0, min(1.0, 0.6*thr + 0.3*wait + 0.1*(1.0-starv))), 3)
    if task == "pedestrian_heavy":
        # Approximation: full grader uses an independent ped-backlog signal,
        # but the standard episode runner doesn't track it. Throughput +
        # wait + starvation captures most of the variance for benchmarking.
        thr = min(1.0, stats["total_cleared"] / 280.0)
        wait = max(0.0, 1.0 - stats["avg_wait"] / 70.0)
        starv = min(1.0, stats["starvation_events"] * 0.05)
        return round(max(0.0, min(1.0, 0.5*thr + 0.35*wait + 0.15*(1.0-starv))), 3)

    raise ValueError(task)


# ============================================================
#  ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of seeds per scenario (default: 5)")
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    agents = [
        ("dummy",            dummy_agent),
        ("round_robin",      round_robin_agent),
        ("rule_based",       rule_based_agent),
        ("smart_bundle",     smart_bundle_agent),
        ("bundle_NS_cheat",  bundle_ns_cheater),
        ("agg_holds_cheat",  aggressive_holds_cheater),
    ]

    print(f"\n{'='*78}")
    print(f"  Traffic Signal v2 — Benchmark ({len(seeds)} seeds per scenario)")
    print(f"{'='*78}\n")

    rows = []
    for name, fn in agents:
        summary, avg = benchmark(fn, name, seeds)
        row = {"agent": name, "average": avg}
        for task, stat in summary.items():
            row[task] = f"{stat['mean']:.3f}±{stat['std']:.3f}"
        rows.append(row)

    # Table
    cols = [
        "agent",
        "easy", "medium", "hard",
        "night", "rush_hour", "chaos",
        "tutorial", "asymmetric", "deterministic", "pedestrian_heavy",
        "ambulance_run", "vip_convoy", "monsoon", "orchestrated_signals",
        "average",
    ]
    widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}
    header = "  " + " | ".join(c.ljust(widths[c]) for c in cols)
    sep    = "  " + "-+-".join("-" * widths[c] for c in cols)
    print(header)
    print(sep)
    for r in rows:
        line = "  " + " | ".join(str(r[c]).ljust(widths[c]) for c in cols)
        print(line)
    print()

    # Headline insight: highlight the night scenario gap
    print("Key insight — the 'night' scenario is designed to expose over-switching.")
    print("Compare 'rule_based' on easy/medium/hard vs. night to see the agent's weakness.\n")


if __name__ == "__main__":
    main()

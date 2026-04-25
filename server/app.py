from fastapi import FastAPI, HTTPException
from environment.traffic_env import TrafficEnv, Action, Observation, SCENARIO_CONFIGS
from typing import Literal, Optional
import uvicorn

app = FastAPI(
    title="Traffic Signal Optimization Environment",
    description=(
        "OpenEnv-compatible RL environment for adaptive traffic signal control. "
        "v2 adds bounded RL-friendly reward, potential-based shaping, and three "
        "new scenarios (night, rush_hour, chaos)."
    ),
    version="2.3.0",
)

# One global env instance per session (single-tenant simplicity).
envs: dict[str, TrafficEnv] = {}

TaskName = Literal[
    "easy", "medium", "hard",
    "night", "rush_hour", "chaos",
    "tutorial", "asymmetric", "deterministic",
    "pedestrian_heavy",
    "ambulance_run", "vip_convoy", "monsoon", "orchestrated_signals",
]


@app.get("/")
def root():
    return {
        "name":        "traffic-signal-env",
        "version":     "2.3.0",
        "description": "Adaptive traffic signal optimization environment",
        "endpoints":   ["/reset", "/step", "/state", "/tasks"],
        "scenarios":   list(SCENARIO_CONFIGS.keys()),
    }


@app.post("/reset")
def reset(task: TaskName = "easy", seed: Optional[int] = None):
    """
    Initialize a new episode.

    Args:
        task: scenario id — easy, medium, hard, night, rush_hour, or chaos.
        seed: optional integer for reproducible runs (v2).
    """
    env = TrafficEnv(task=task, seed=seed)
    envs["current"] = env
    obs = env.reset()
    return {"observation": obs.model_dump()}


@app.post("/step")
def step(action: Action):
    if "current" not in envs:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    env = envs["current"]
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward":      reward.model_dump(),
        "done":        done,
        "info":        info,
    }


@app.get("/state")
def state():
    if "current" not in envs:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    return {"state": envs["current"].state().model_dump()}


@app.get("/tasks")
def tasks():
    """List all available scenarios with their key parameters."""
    return {
        "tasks": [
            {
                "id":              tid,
                "description":     cfg["description"],
                "episode_length":  cfg["episode_length"],
                "emergency_prob":  cfg["emergency_prob"],
                "pedestrian_prob": cfg["pedestrian_prob"],
                "incidents":       cfg["incidents_enabled"],
                "schedule":        cfg["time_of_day_schedule"],
            }
            for tid, cfg in SCENARIO_CONFIGS.items()
        ]
    }


def main():
    """Entry point referenced by pyproject.toml `[project.scripts] server`."""
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()

from fastapi import FastAPI, HTTPException
from environment.traffic_env import TrafficEnv, Action, Observation
from typing import Literal
import uvicorn

app = FastAPI(
    title="Traffic Signal Optimization Environment",
    description="An OpenEnv-compatible RL environment for adaptive traffic signal control.",
    version="1.0.0"
)

# one global env instance per session
envs: dict[str, TrafficEnv] = {}

@app.get("/")
def root():
    return {
        "name": "traffic-signal-env",
        "version": "1.0.0",
        "description": "Adaptive traffic signal optimization environment",
        "endpoints": ["/reset", "/step", "/state", "/tasks"]
    }

@app.post("/reset")
def reset(task: Literal["easy", "medium", "hard"] = "easy"):
    env = TrafficEnv(task=task)
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
        "reward": reward.model_dump(),
        "done": done,
        "info": info
    }

@app.get("/state")
def state():
    if "current" not in envs:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    return {"state": envs["current"].state().model_dump()}

@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {
                "id": "easy",
                "description": "Normal traffic flow, clear right answer exists",
                "difficulty": 1,
                "episode_length": 200
            },
            {
                "id": "medium",
                "description": "Uneven traffic with pedestrian crossing requests",
                "difficulty": 2,
                "episode_length": 200
            },
            {
                "id": "hard",
                "description": "Rush hour with emergency vehicles and time-of-day transitions",
                "difficulty": 3,
                "episode_length": 300
            }
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
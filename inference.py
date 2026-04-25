import os
import json
from dotenv import load_dotenv
from environment.traffic_env import TrafficEnv, Action, Observation
from tasks.graders import grade_all

load_dotenv()

MODEL = os.getenv("MODEL_NAME", "gemini-2.0-flash")
API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai/"
)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

client = None

try:
    from openai import OpenAI
    api_key = OPENAI_API_KEY or HF_TOKEN
    if api_key:
        client = OpenAI(api_key=api_key, base_url=API_BASE_URL)
except Exception:
    client = None


def rule_based_agent(obs: Observation) -> Action:
    # Emergency vehicles are highest priority
    if obs.emergency_lane:
        if obs.current_green != obs.emergency_lane:
            return Action(action=f"switch_to_{obs.emergency_lane}")
        return Action(action="keep")

    # v2.3: VIP convoy active OR arriving within 4 ticks → pre-clear corridor
    if obs.vip_active and obs.current_green != obs.vip_eta_lane:
        return Action(action=f"switch_to_{obs.vip_eta_lane}")
    if (obs.vip_eta_lane is not None
            and obs.vip_eta_ticks <= 4
            and obs.current_green != obs.vip_eta_lane):
        return Action(action=f"switch_to_{obs.vip_eta_lane}")

    total_cars = (
        obs.north.cars + obs.south.cars +
        obs.east.cars + obs.west.cars
    )

    if obs.pedestrian_requests and not obs.pedestrian_active and total_cars < 15:
        return Action(action="pedestrian_hold")

    lane_map = {
        "N": obs.north,
        "S": obs.south,
        "E": obs.east,
        "W": obs.west,
    }

    # v2: never switch into a lane closed by an incident
    if obs.incident_lane and obs.incident_lane in lane_map:
        del lane_map[obs.incident_lane]

    for lane, state in lane_map.items():
        if lane != obs.current_green and state.avg_wait > 45 and state.cars > 0:
            return Action(action=f"switch_to_{lane}")

    if obs.time_of_day == "NIGHT" and total_cars < 8:
        ns_cars = obs.north.cars + obs.south.cars
        ew_cars = obs.east.cars + obs.west.cars
        # v2: don't bundle into closed lanes
        if obs.incident_lane in ("N", "S"):
            ns_cars = 0
        if obs.incident_lane in ("E", "W"):
            ew_cars = 0
        if ns_cars > ew_cars and ns_cars > 2:
            return Action(action="bundle_NS")
        if ew_cars > ns_cars and ew_cars > 2:
            return Action(action="bundle_EW")

    pressure = {
        lane: state.cars * max(state.avg_wait, 1.0)
        for lane, state in lane_map.items()
    }

    if not pressure:
        return Action(action="keep")

    busiest = max(pressure, key=pressure.get)
    current_pressure = pressure.get(obs.current_green, 0)
    busiest_pressure = pressure[busiest]

    if busiest != obs.current_green and busiest_pressure > current_pressure * 1.3:
        return Action(action=f"switch_to_{busiest}")

    if obs.time_in_phase > 20 and current_pressure > 30:
        return Action(action="extend_green")

    return Action(action="keep")


VALID_ACTIONS = [
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


def build_prompt(obs: Observation) -> str:
    emergency_text = (
        f"EMERGENCY VEHICLE in lane {obs.emergency_lane} — respond immediately!"
        if obs.emergency_lane else "No emergency."
    )
    pedestrian_text = (
        f"Pedestrian crossing requested at: {', '.join(obs.pedestrian_requests)}"
        if obs.pedestrian_requests else "No pedestrian requests."
    )
    # v2: surface lane incidents to the LLM so it can avoid switching into closed lanes
    incident_text = (
        f"INCIDENT: lane {obs.incident_lane} is closed for "
        f"{obs.incident_ticks_remaining} more ticks — do NOT switch to it."
        if obs.incident_lane else "No lane incidents."
    )
    # v2.3: VIP convoy and weather
    if obs.vip_active:
        vip_text = (f"VIP CONVOY ACTIVE in lane {obs.vip_eta_lane} — "
                    f"keep green on this lane until convoy exits.")
    elif obs.vip_eta_lane is not None:
        vip_text = (f"VIP CONVOY arriving in {obs.vip_eta_ticks} ticks at "
                    f"lane {obs.vip_eta_lane} — pre-clear the green corridor.")
    else:
        vip_text = "No VIP convoy."
    weather_text = (
        f"Weather: {obs.weather} — flow rate is reduced; consider bundle actions."
        if obs.weather != "CLEAR" else "Weather: CLEAR."
    )
    return (
        f"Traffic state (tick {obs.tick}, {obs.time_of_day}, {weather_text}):\n"
        f"N:{obs.north.cars}cars/{obs.north.avg_wait}s "
        f"S:{obs.south.cars}cars/{obs.south.avg_wait}s "
        f"E:{obs.east.cars}cars/{obs.east.avg_wait}s "
        f"W:{obs.west.cars}cars/{obs.west.avg_wait}s\n"
        f"Green:{obs.current_green} phase:{obs.time_in_phase}\n"
        f"{emergency_text} {pedestrian_text} {incident_text} {vip_text}\n"
        f"Actions: keep switch_to_N switch_to_S switch_to_E switch_to_W "
        f"extend_green pedestrian_hold bundle_NS bundle_EW\n"
        f'Reply ONLY with JSON: {{"action": "your_choice"}}'
    )


def llm_agent(obs: Observation) -> Action:
    if client is None:
        return rule_based_agent(obs)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=1000,
            messages=[
                {
                    "role": "system",
                    "content": "You are a traffic signal controller. Always respond with valid JSON only."
                },
                {
                    "role": "user",
                    "content": build_prompt(obs)
                }
            ]
        )

        raw = response.choices[0].message.content
        if raw is None or raw.strip() == "":
            return rule_based_agent(obs)

        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw.split("\n")[0].strip())
        action_str = parsed.get("action", "keep")

        if action_str not in VALID_ACTIONS:
            return rule_based_agent(obs)

        return Action(action=action_str)
    except Exception:
        return rule_based_agent(obs)


def agent(obs: Observation) -> Action:
    if client is not None:
        return llm_agent(obs)
    return rule_based_agent(obs)


def run_episode_structured(task: str):
    env = TrafficEnv(task=task)
    obs = env.reset()
    total_reward = 0.0
    step_count = 0

    print(f"[START] task={task}", flush=True)

    while True:
        action = agent(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward.total
        step_count += 1

        print(
            f"[STEP] task={task} step={step_count} action={action.action} "
            f"reward={reward.total:.3f} done={done}",
            flush=True
        )

        if done:
            break

    scores = grade_all(rule_based_agent)
    task_score = scores[task]

    print(
        f"[END] task={task} score={task_score:.3f} steps={step_count} "
        f"total_reward={total_reward:.3f}",
        flush=True
    )

    return task_score


# v2.3: fourteen scenarios — 6 v2.0 + 3 v2.1 + 1 v2.2 + 4 v2.3
TASKS = [
    "easy", "medium", "hard",
    "night", "rush_hour", "chaos",
    "tutorial", "asymmetric", "deterministic",
    "pedestrian_heavy",
    "ambulance_run", "vip_convoy", "monsoon", "orchestrated_signals",
]


if __name__ == "__main__":
    final_scores = {}
    for task_name in TASKS:
        final_scores[task_name] = run_episode_structured(task_name)

    avg = sum(final_scores.values()) / len(final_scores)
    score_parts = " ".join(f"{t}={final_scores[t]:.3f}" for t in TASKS)
    print(
        f"[END] task=all score={avg:.3f} steps={len(TASKS)} {score_parts}",
        flush=True
    )

import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from environment.traffic_env import TrafficEnv, Action, Observation
from tasks.graders import grade_all

load_dotenv()

# ─────────────────────────────────────────
#  CLIENT SETUP
# ─────────────────────────────────────────

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("API_BASE_URL"),
)

MODEL = os.getenv("MODEL_NAME", "gemini-2.0-flash")

# ─────────────────────────────────────────
#  PROMPT BUILDER
# ─────────────────────────────────────────

def build_prompt(obs: Observation) -> str:
    emergency_text = (
        f"🚨 EMERGENCY VEHICLE in lane {obs.emergency_lane} — respond immediately!"
        if obs.emergency_lane
        else "No emergency."
    )

    pedestrian_text = (
        f"Pedestrian crossing requested at: {', '.join(obs.pedestrian_requests)}"
        if obs.pedestrian_requests
        else "No pedestrian requests."
    )

    return f"""
You are an AI traffic signal controller managing a 4-way intersection.

CURRENT STATE (tick {obs.tick}):
  North: {obs.north.cars} cars waiting, avg wait {obs.north.avg_wait}s
  South: {obs.south.cars} cars waiting, avg wait {obs.south.avg_wait}s
  East:  {obs.east.cars}  cars waiting, avg wait {obs.east.avg_wait}s
  West:  {obs.west.cars}  cars waiting, avg wait {obs.west.avg_wait}s

  Current green light: {obs.current_green}
  Time in current phase: {obs.time_in_phase} ticks
  Time of day: {obs.time_of_day}
  {emergency_text}
  {pedestrian_text}

AVAILABLE ACTIONS:
  keep          - hold current green light for one more tick
  switch_to_N   - switch green light to North
  switch_to_S   - switch green light to South
  switch_to_E   - switch green light to East
  switch_to_W   - switch green light to West
  extend_green  - extend current phase cap by 10 ticks
  pedestrian_hold - pause all traffic for pedestrian crossing
  bundle_NS     - simultaneously release North and South lanes
  bundle_EW     - simultaneously release East and West lanes

RULES:
  - Prevent any lane from waiting more than 60 ticks (starvation)
  - Respond to emergencies within 15 ticks for full score
  - Use pedestrian_hold only when pedestrian requests exist
  - bundle_NS and bundle_EW are powerful but use sparingly

Respond with ONLY a JSON object like this:
{{"action": "switch_to_E"}}

No explanation. No markdown. Just the JSON.
""".strip()


# ─────────────────────────────────────────
#  LLM AGENT
# ─────────────────────────────────────────

VALID_ACTIONS = [
    "keep", "switch_to_N", "switch_to_S", "switch_to_E", "switch_to_W",
    "extend_green", "pedestrian_hold", "bundle_NS", "bundle_EW"
]

def llm_agent(obs: Observation) -> Action:
    prompt = build_prompt(obs)

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
                    "content": prompt
                }
            ]
        )

        raw = response.choices[0].message.content

        if raw is None or raw.strip() == "":
            return Action(action="keep")

        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw.split("\n")[0].strip())
        action_str = parsed.get("action", "keep")

        if action_str not in VALID_ACTIONS:
            action_str = "keep"

        return Action(action=action_str)

    except Exception as e:
        print(f"  [warn] LLM error: {e} — defaulting to keep")
        return Action(action="keep")

# ─────────────────────────────────────────
#  SINGLE EPISODE RUNNER (visible output)
# ─────────────────────────────────────────

def run_episode_verbose(task: str):
    env = TrafficEnv(task=task)
    obs = env.reset()
    total_reward = 0.0
    print(f"\n{'─'*50}")
    print(f"  Running task: {task.upper()}")
    print(f"{'─'*50}")

    while True:
        action = llm_agent(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward.total

        if obs.tick % 20 == 0:
            print(
                f"  tick {obs.tick:>3} | "
                f"action: {action.action:<15} | "
                f"reward: {reward.total:>6.2f} | "
                f"cumulative: {total_reward:>8.2f} | "
                f"queues N{obs.north.cars} S{obs.south.cars} "
                f"E{obs.east.cars} W{obs.west.cars}"
            )

        if done:
            print(f"\n  Episode finished at tick {obs.tick}")
            print(f"  Total reward: {round(total_reward, 2)}")
            break

    return total_reward


# ─────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*50)
    print("  TRAFFIC SIGNAL ENV — BASELINE INFERENCE")
    print("="*50)

    for task in ["easy", "medium", "hard"]:
        run_episode_verbose(task)

    def rule_based_agent(obs):
        if obs.emergency_lane:
            return Action(action=f"switch_to_{obs.emergency_lane}")
        total_cars = (obs.north.cars + obs.south.cars +
                      obs.east.cars + obs.west.cars)
        if obs.pedestrian_requests and not obs.pedestrian_active and total_cars < 15:
            return Action(action="pedestrian_hold")
        lanes = {
            "N": obs.north.cars + obs.north.avg_wait,
            "S": obs.south.cars + obs.south.avg_wait,
            "E": obs.east.cars  + obs.east.avg_wait,
            "W": obs.west.cars  + obs.west.avg_wait,
        }
        busiest = max(lanes, key=lanes.get)
        if busiest != obs.current_green:
            return Action(action=f"switch_to_{busiest}")
        return Action(action="keep")

    print(f"\n{'='*50}")
    print("  OFFICIAL GRADER SCORES")
    print(f"{'='*50}")
    scores = grade_all(rule_based_agent)
    print(f"  Easy:   {scores['easy']}")
    print(f"  Medium: {scores['medium']}")
    print(f"  Hard:   {scores['hard']}")
    print(f"\n  Average: {round(sum(scores.values()) / 3, 3)}")
    print("="*50)
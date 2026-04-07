from tasks.graders import grade_all
from environment.traffic_env import Action

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

scores = grade_all(rule_based_agent)
print(f"Easy:    {scores['easy']}")
print(f"Medium:  {scores['medium']}")
print(f"Hard:    {scores['hard']}")
print(f"Average: {round(sum(scores.values())/3, 3)}")
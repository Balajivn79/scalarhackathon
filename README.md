# Traffic Signal Optimization Environment

An OpenEnv-compatible reinforcement learning environment for adaptive traffic signal control at a 4-way intersection.

## Overview

Traditional traffic lights use fixed timing cycles, leading to unnecessary waiting, congestion during peak hours, and poor emergency vehicle handling. This environment simulates a smart intersection where an AI agent dynamically controls signal timing to minimize congestion, reduce wait times, and prioritize emergency vehicles.

The agent operates under realistic conditions — stochastic vehicle arrivals, pedestrian crossing requests, time-of-day transitions, and emergency vehicle preemption.

## Real-World Relevance

This environment models a problem actively deployed in smart cities worldwide. Adaptive signal control systems are used in Pittsburgh, Singapore, and Amsterdam to reduce urban congestion. This environment provides a clean, reproducible testbed for evaluating agent performance on this problem.

## Environment Description

A 4-way intersection with North, South, East, West lanes. Each tick simulates one second of real time. Vehicles arrive randomly via Poisson distribution. The agent controls which lane gets the green light and for how long.

### Key Mechanics

- Vehicle arrivals modeled via Poisson distribution per lane per tick
- Flow rate of 4 cars cleared per tick on the active green lane
- Starvation detection — lanes unserved for 60+ ticks incur penalties
- Emergency vehicle preemption with 45 second response window
- Pedestrian crossing requests that pause all traffic for 15 ticks
- Time-of-day transitions (NORMAL → PEAK_HOUR → NORMAL) on hard task
- Fairness reward component penalizes high variance across lane wait times

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| north.cars | int | Cars waiting in North lane |
| north.avg_wait | float | Average wait time in North lane (seconds) |
| south.cars | int | Cars waiting in South lane |
| south.avg_wait | float | Average wait time in South lane (seconds) |
| east.cars | int | Cars waiting in East lane |
| east.avg_wait | float | Average wait time in East lane (seconds) |
| west.cars | int | Cars waiting in West lane |
| west.avg_wait | float | Average wait time in West lane (seconds) |
| current_green | str | Currently active green lane (N/S/E/W) |
| time_in_phase | int | Ticks elapsed in current phase |
| emergency_lane | str/null | Lane with emergency vehicle, if any |
| pedestrian_requests | list | Lanes with pedestrian crossing requests |
| pedestrian_active | bool | Whether pedestrian hold is active |
| pedestrian_ticks_remaining | int | Ticks remaining in pedestrian hold |
| time_of_day | str | NORMAL / PEAK_HOUR / NIGHT |
| tick | int | Current episode tick |

## Action Space

| Action | Description |
|--------|-------------|
| keep | Hold current green light for one tick |
| switch_to_N | Switch green light to North |
| switch_to_S | Switch green light to South |
| switch_to_E | Switch green light to East |
| switch_to_W | Switch green light to West |
| extend_green | Extend current phase cap by 10 ticks |
| pedestrian_hold | Pause all traffic for 15 ticks for pedestrian crossing |
| bundle_NS | Simultaneously release North and South lanes |
| bundle_EW | Simultaneously release East and West lanes |

## Reward Function

Reward is computed every tick as a weighted sum of:

| Component | Formula | Description |
|-----------|---------|-------------|
| Throughput | +0.5 × cars_cleared | Reward for clearing vehicles |
| Queue | -0.1 × total_queued | Penalty for overall congestion |
| Max wait | -0.3 × max_lane_wait | Penalty for worst-off lane |
| Starvation | -0.5 × starved_lanes | Penalty for lanes unserved 60+ ticks |
| Fairness | -0.2 × variance(waits) | Penalty for unequal service |
| Emergency | -20.0 one-time if unserved after 45s, +10.0 if cleared within 15s |

## Tasks

### Easy
Normal traffic with steady arrival rates. Green light starts on North. Agent must keep average wait times low and prevent lane starvation.
- Episode length: 200 ticks
- Success threshold: 0.7

### Medium
Uneven traffic volumes with random pedestrian crossing requests. Requires balancing throughput against interruptions.
- Episode length: 200 ticks
- Success threshold: 0.6

### Hard
Peak hour congestion with emergency vehicle spawning and time-of-day transitions (NORMAL → PEAK_HOUR at tick 100, back to NORMAL at tick 200). Agent must respond to emergencies within 15 ticks while managing rush hour queues.
- Episode length: 300 ticks
- Success threshold: 0.5

## Baseline Scores

Evaluated using a rule-based greedy agent that always serves the busiest lane and prioritizes emergencies:

| Task | Score |
|------|-------|
| Easy | 0.940 |
| Medium | 0.774 |
| Hard | 0.891 |
| **Average** | **0.868** |

A dummy agent that never switches signals scores 0.0 on easy and hard, confirming the environment has a meaningful performance floor.

## Setup Instructions

### Prerequisites
- Python 3.11+
- Conda or pip

### Installation
```bash
git clone <your-repo-url>
cd traffic-signal-env
conda create -n traffic-environment python=3.11
conda activate traffic-environment
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:
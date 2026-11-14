"""
shared_state.py — Thread-Safe Memory Bridge between RL/SUMO Backend and Dash UI
===============================================================================
"""

import threading
from collections import deque

# Global re-entrant lock for thread-safe reads/writes between Thread 1 (RL/SUMO) and Thread 2 (Dash)
state_lock = threading.RLock()

# Human-readable labels for the 6 signalized intersections in cleanedindorenetwork.net.xml
INTERSECTION_NAMES = [
    "J1: MR-10 North (5697428980)",
    "J2: Ring Rd South (5697428981)",
    "J3: Ring Rd West (5697428983)",
    "J4: Vijay Nagar (Cluster 2749)",
    "J5: Radisson Sq (radisson_tls)",
    "J6: Bombay Hosp (Roundabout)",
]

# Two-Way Steering Configuration (Written by Dash UI -> Read by env.py & train.py)
global_config = {
    "control_mode": "RL",           # "RL" (PPO Agent) or "FIXED" (SUMO Default Timer)
    "traffic_density": 1.0,         # 0.5x to 3.0x dynamic traffic spawner multiplier
    "target_episodes": 50,          # Target episodes to run in current session
    "learning_rate": 3e-4,          # Live PPO learning rate
    "step_delay_ms": 0,             # Artificial delay per step (0ms to 500ms) for visual inspection
    "spawn_ev": False,              # Trigger flag set by "DISPATCH EMERGENCY VEHICLE" button
}

# Real-Time Telemetry Store (Written by env.py -> Read by dashboard.py every 1000ms)
MAX_HISTORY_LEN = 300

live_metrics = {
    # Simulation progress
    "sim_time": 0.0,
    "episode": 1,
    "episode_step": 0,
    "active_vehicles": 0,
    "current_mode": "RL",

    # KPI Card 1: Emergency Vehicle (EV) Telemetry
    "ev_active": False,
    "ev_id": "NONE",
    "ev_junction": "STANDBY",
    "ev_time_saved_s": 0.0,
    "ev_last_completed_id": "NONE",
    "ev_total_dispatched": 0,

    # KPI Card 2: Environmental Telemetry
    "co2_total_kg": 0.0,
    "co2_rate_gps": 0.0,            # Instantaneous grams per second
    "fuel_total_l": 0.0,
    "fuel_rate_mlps": 0.0,          # Instantaneous mL per second

    # KPI Card 3 & 4: Throughput & Signal Stability
    "throughput_per_min": 0.0,      # Rolling vehicles arriving at destination per minute
    "phase_flicker_rate": 0.0,      # Signal phase switches per minute across network

    # 6-Intersection Real-Time Queue Breakdown
    "intersection_queues": {name: 0 for name in INTERSECTION_NAMES},

    # Time-series deques for live Plotly charts
    "history_timestamps": deque(maxlen=MAX_HISTORY_LEN),
    "history_global_queue": deque(maxlen=MAX_HISTORY_LEN),
    "history_mode": deque(maxlen=MAX_HISTORY_LEN),
    "history_step_reward": deque(maxlen=MAX_HISTORY_LEN),
    "history_cum_reward": deque(maxlen=MAX_HISTORY_LEN),
    "episode_rewards": deque(maxlen=100),
}


def update_config(key, value):
    """Thread-safe setter for UI controls."""
    with state_lock:
        global_config[key] = value


def get_config_snapshot():
    """Thread-safe snapshot of current steering configuration."""
    with state_lock:
        return dict(global_config)


def consume_ev_trigger():
    """Atomically checks and resets the spawn_ev trigger flag."""
    with state_lock:
        if global_config.get("spawn_ev", False):
            global_config["spawn_ev"] = False
            return True
        return False


def get_metrics_snapshot():
    """Thread-safe deep snapshot of live_metrics for Dash interval callbacks."""
    with state_lock:
        return {
            "sim_time": live_metrics["sim_time"],
            "episode": live_metrics["episode"],
            "episode_step": live_metrics["episode_step"],
            "active_vehicles": live_metrics["active_vehicles"],
            "current_mode": live_metrics["current_mode"],
            "ev_active": live_metrics["ev_active"],
            "ev_id": live_metrics["ev_id"],
            "ev_junction": live_metrics["ev_junction"],
            "ev_time_saved_s": live_metrics["ev_time_saved_s"],
            "ev_last_completed_id": live_metrics["ev_last_completed_id"],
            "ev_total_dispatched": live_metrics["ev_total_dispatched"],
            "co2_total_kg": live_metrics["co2_total_kg"],
            "co2_rate_gps": live_metrics["co2_rate_gps"],
            "fuel_total_l": live_metrics["fuel_total_l"],
            "fuel_rate_mlps": live_metrics["fuel_rate_mlps"],
            "throughput_per_min": live_metrics["throughput_per_min"],
            "phase_flicker_rate": live_metrics["phase_flicker_rate"],
            "intersection_queues": dict(live_metrics["intersection_queues"]),
            "history_timestamps": list(live_metrics["history_timestamps"]),
            "history_global_queue": list(live_metrics["history_global_queue"]),
            "history_mode": list(live_metrics["history_mode"]),
            "history_step_reward": list(live_metrics["history_step_reward"]),
            "history_cum_reward": list(live_metrics["history_cum_reward"]),
            "episode_rewards": list(live_metrics["episode_rewards"]),
        }
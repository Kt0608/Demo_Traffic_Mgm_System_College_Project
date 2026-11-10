"""
evaluate.py — Performance Evaluation for Indore Traffic RL Agent
=================================================================

Metrics collected every step:
  • Average waiting time per vehicle
  • Total queue length across all monitored lanes
  • Number of vehicles that completed their trip (throughput)
  • Emergency vehicle wait time
"""

import os
import sys
import numpy as np
import datetime
import xml.etree.ElementTree as ET

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import traci

os.makedirs("logs", exist_ok=True)

try:
    import matplotlib.pyplot as plt
    PLOT = True
except ImportError:
    PLOT = False
    print("[INFO] matplotlib not found — skipping plots. Install with: pip install matplotlib")

try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("[WARNING] stable-baselines3 not found — only baseline will be evaluated.")

from env import IndoreRoundaboutEnv


# ══════════════════════════════════════════════════════════════════════════════
# Metric collector
# ══════════════════════════════════════════════════════════════════════════════

class MetricsCollector:
    def __init__(self):
        self.avg_wait_per_step    = []
        self.total_queue_per_step = []
        self.arrived_per_step     = []
        self.emergency_wait       = []
        self.total_arrived        = 0

    def collect(self, monitored_lanes):
        vehicles = traci.vehicle.getIDList()
        n = max(1, traci.vehicle.getIDCount())

        total_wait = sum(traci.vehicle.getWaitingTime(v) for v in vehicles)
        self.avg_wait_per_step.append(total_wait / n)

        total_queue = 0
        for lane in monitored_lanes:
            try:
                total_queue += traci.lane.getLastStepHaltingNumber(lane)
            except traci.TraCIException:
                pass
        self.total_queue_per_step.append(total_queue)

        arrived = traci.simulation.getArrivedNumber()
        self.total_arrived += arrived
        self.arrived_per_step.append(self.total_arrived)

        em_wait = sum(
            traci.vehicle.getWaitingTime(v)
            for v in vehicles
            if traci.vehicle.getTypeID(v) == "emergency"
        )
        self.emergency_wait.append(em_wait)

    def summary(self):
        return {
            "avg_wait_mean":          float(np.mean(self.avg_wait_per_step)),
            "avg_wait_max":           float(np.max(self.avg_wait_per_step)),
            "avg_queue_mean":         float(np.mean(self.total_queue_per_step)),
            "avg_queue_max":          float(np.max(self.total_queue_per_step)),
            "total_arrived":          int(self.total_arrived),
            "emergency_wait_total":   float(np.sum(self.emergency_wait)),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Run baseline — SUMO controls signals with its default fixed-time plan
# ══════════════════════════════════════════════════════════════════════════════

def run_baseline(use_gui=False):
    print("\n" + "="*60)
    print("  RUNNING BASELINE (Fixed-time signals, no agent)")
    print("="*60)

    sumo_bin = "sumo-gui" if use_gui else "sumo"
    sumo_cmd = [sumo_bin, "-c", "simulation.sumocfg", "--no-step-log", "true", "--start", "true"]
    traci.start(sumo_cmd)

    env_dummy = IndoreRoundaboutEnv()
    collector = MetricsCollector()
    step = 0

    while traci.simulation.getTime() < 3600:
        traci.simulationStep()
        if step % 5 == 0:
            collector.collect(env_dummy.monitored_lanes)
        step += 1

    traci.close()
    return collector


# ══════════════════════════════════════════════════════════════════════════════
# Run PPO agent
# ══════════════════════════════════════════════════════════════════════════════

def run_agent(model_path="models/indore_ppo_traffic_agent", use_gui=False):
    print("\n" + "="*60)
    print("  RUNNING PPO AGENT")
    print("="*60)

    if not SB3_AVAILABLE:
        print("[ERROR] stable-baselines3 is not installed.")
        return None

    if not os.path.exists(model_path + ".zip"):
        print(f"[ERROR] No trained model found at {model_path}.zip")
        print("        Run 'python train.py' first.")
        return None

    env = IndoreRoundaboutEnv(use_gui=use_gui)
    model = PPO.load(model_path, env=env)

    obs, _ = env.reset()
    collector = MetricsCollector()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        collector.collect(env.monitored_lanes)
        done = terminated or truncated

    env.close()
    return collector


# ══════════════════════════════════════════════════════════════════════════════
# Parse tripinfo.xml for additional metrics
# ══════════════════════════════════════════════════════════════════════════════

def parse_tripinfo(filepath="tripinfo.xml"):
    if not os.path.exists(filepath):
        return None
    tree = ET.parse(filepath)
    root = tree.getroot()
    durations, waits, speeds = [], [], []
    for trip in root.findall("tripinfo"):
        try:
            durations.append(float(trip.get("duration", 0)))
            waits.append(float(trip.get("waitingTime", 0)))
            route_len = float(trip.get("routeLength", 1))
            dur = max(1.0, float(trip.get("duration", 1)))
            speeds.append(route_len / dur)
        except (TypeError, ValueError):
            pass
    if not durations:
        return None
    return {
        "avg_trip_duration_s":    round(np.mean(durations), 2),
        "avg_waiting_time_s":     round(np.mean(waits), 2),
        "avg_speed_mps":          round(np.mean(speeds), 2),
        "total_trips_completed":  len(durations),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Report printer
# ══════════════════════════════════════════════════════════════════════════════

import datetime  # Make sure datetime is imported at the top of your file

def print_report(baseline, agent=None, model_path="baseline"):
    b = baseline.summary()
    a = agent.summary() if agent else None
    tripinfo = parse_tripinfo("tripinfo.xml")

    lines = []
    lines.append("\n" + "="*65)
    lines.append("  PERFORMANCE EVALUATION REPORT — Indore Traffic RL")
    lines.append("="*65)

    def row(label, b_val, a_val=None, unit="", lower_better=True):
        if a_val is None:
            return f"  {label:<35} {str(round(b_val,2))+unit:>12}"
        diff = a_val - b_val
        if lower_better:
            arrow = "BETTER" if diff < 0 else ("WORSE" if diff > 0 else "SAME")
        else:
            arrow = "BETTER" if diff > 0 else ("WORSE" if diff < 0 else "SAME")
        pct = f"({diff/max(abs(b_val),1e-6)*100:+.1f}%)" if b_val != 0 else ""
        return f"  {label:<35} {str(round(b_val,2))+unit:>12}  {str(round(a_val,2))+unit:>12}  {arrow} {pct}"

    header = f"\n  {'Metric':<35} {'Baseline':>12}"
    if a:
        header += f"  {'PPO Agent':>12}  Result"
    lines.append(header)
    lines.append("  " + "-"*60)
    lines.append(row("Avg wait per vehicle (mean)",   b["avg_wait_mean"],        a["avg_wait_mean"]        if a else None, "s"))
    lines.append(row("Avg wait per vehicle (peak)",   b["avg_wait_max"],         a["avg_wait_max"]         if a else None, "s"))
    lines.append(row("Avg queue length (mean)",        b["avg_queue_mean"],       a["avg_queue_mean"]       if a else None, " veh"))
    lines.append(row("Avg queue length (peak)",        b["avg_queue_max"],        a["avg_queue_max"]        if a else None, " veh"))
    lines.append(row("Total vehicles arrived",         b["total_arrived"],        a["total_arrived"]        if a else None, "", lower_better=False))
    lines.append(row("Emergency wait (total)",         b["emergency_wait_total"], a["emergency_wait_total"] if a else None, "s"))

    if tripinfo:
        lines.append("\n  --- From tripinfo.xml (last run) ---")
        lines.append(f"  {'Avg trip duration':<35} {tripinfo['avg_trip_duration_s']:>10}s")
        lines.append(f"  {'Avg waiting time (per trip)':<35} {tripinfo['avg_waiting_time_s']:>10}s")
        lines.append(f"  {'Avg speed':<35} {tripinfo['avg_speed_mps']:>8} m/s")
        lines.append(f"  {'Trips completed':<35} {tripinfo['total_trips_completed']:>10}")

    lines.append("="*65 + "\n")
    report = "\n".join(lines)
    print(report)

    # Generate unique filename using model base name and current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = os.path.basename(model_path)
    report_filename = f"logs/eval_report_{safe_model_name}_{timestamp}.txt"

    with open(report_filename, "w") as f:
        f.write(report)
    print(f"  Report saved to {report_filename}")
    return report
# ══════════════════════════════════════════════════════════════════════════════
# Plot curves
# ══════════════════════════════════════════════════════════════════════════════

def plot_comparison(baseline, agent=None, model_path="baseline"):
    if not PLOT:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Indore Traffic Control — Baseline vs PPO Agent", fontsize=14, fontweight="bold")

    datasets = [("Baseline (fixed)", baseline, "steelblue")]
    if agent:
        datasets.append(("PPO Agent", agent, "seagreen"))

    metrics = [
        ("avg_wait_per_step",    "Avg Wait Time (s)"),
        ("total_queue_per_step", "Total Queue (vehicles)"),
        ("arrived_per_step",     "Cumulative Throughput"),
    ]

    for ax, (attr, ylabel) in zip(axes, metrics):
        for name, collector, color in datasets:
            series = getattr(collector, attr)
            ax.plot(series, label=name, color=color, linewidth=1.5, alpha=0.85)
        ax.set_xlabel("Decision Step (every 5s sim-time)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Generate unique filename using model base name and timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = os.path.basename(model_path)
    plot_filename = f"logs/eval_comparison_{safe_model_name}_{timestamp}.png"

    plt.savefig(plot_filename, dpi=150, bbox_inches="tight")
    print(f"  Plot saved to {plot_filename}")
    plt.show()
# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Indore Traffic RL Agent")
    parser.add_argument("--gui",              action="store_true", help="Open SUMO-GUI during evaluation")
    parser.add_argument("--baseline-only",    action="store_true", help="Run only the baseline (no model needed)")
    parser.add_argument("--model",            default="models/indore_ppo_traffic_agent", help="Path to trained model (without .zip)")
    args = parser.parse_args()

    baseline_collector = run_baseline(use_gui=args.gui)

    agent_collector = None
    if not args.baseline_only:
        agent_collector = run_agent(model_path=args.model, use_gui=args.gui)

    print_report(baseline_collector, agent_collector, model_path=args.model)
    plot_comparison(baseline_collector, agent_collector, model_path=args.model)
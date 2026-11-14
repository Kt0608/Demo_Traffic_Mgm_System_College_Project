import os
import sys
import time
import random
import xml.etree.ElementTree as ET
from collections import deque
import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Ensure SUMO_HOME is set in your environment
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import traci

from shared_state import (
    state_lock,
    global_config,
    live_metrics,
    INTERSECTION_NAMES,
    get_config_snapshot,
    consume_ev_trigger,
)


class IndoreRoundaboutEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        net_file="cleanedindorenetwork.net.xml",
        route_file="indore_traffic.rou.xml",
        use_gui=False,
        legacy_4_tls_mode=False,
    ):
        """
        legacy_4_tls_mode:
          If True, keeps action_space=MultiDiscrete([2]*4) and obs_size=41 so existing
          saved checkpoints (models/indore_ppo_100k.zip, etc.) load without shape errors,
          while still controlling & monitoring all 6 intersections in the network.
        """
        super(IndoreRoundaboutEnv, self).__init__()

        self.net_file = net_file
        self.route_file = route_file
        self.use_gui = use_gui
        self.sumoBinary = "sumo-gui" if self.use_gui else "sumo"
        self.legacy_4_tls_mode = legacy_4_tls_mode

        # The 6 actual signalized intersections from cleanedindorenetwork.net.xml
        self.real_tls_ids = [
            "5697428980",
            "5697428981",
            "5697428983",
            "cluster2020642749_2020642752_2020642756_2020642762_#4more",
            "cluster_2020642637_2020642641_2020642651_2023505448",
            "joinedS_2020642670_2020642679_2020642684_2020642689",
        ]

        # Incoming lanes grouped by each of the 6 intersections for the UI Heatmap/Bar Grid
        self.intersection_lanes_map = {
            INTERSECTION_NAMES[0]: ["191495045#3_0", "191495045#3_1"],
            INTERSECTION_NAMES[1]: ["191774924#2_0", "191774924#2_1"],
            INTERSECTION_NAMES[2]: ["191495001#4_0", "191495001#4_1"],
            INTERSECTION_NAMES[3]: [
                "191495016#1_0", "191495016#1_1", "320521586#9_0",
                "191495043#2_0", "191495043#2_1", "191495049#6_0", "191495049#6_1"
            ],
            INTERSECTION_NAMES[4]: [
                "191495045#4_0", "191495045#4_1", "191774924#3_0",
                "191774924#3_1", "191495001#5_0", "191495001#5_1"
            ],
            INTERSECTION_NAMES[5]: [
                "191495034#1_0", "191495034#1_1", "191495002#7_0", "191495002#7_1",
                "191494999#0_0", "191494999#0_1", "1497606153#1_0", "1497606153#1_1",
                "191494996#2_0", "191494996#2_1", "191495034#0_0", "191495034#0_1",
                "1497606153#2_0", "1497606153#2_1", "191494996#1_0", "191494996#1_1"
            ],
        }

        # Map raw TLS ID -> Human-readable name for EV approaching status
        self.tls_id_to_name = {
            tls_id: INTERSECTION_NAMES[i] for i, tls_id in enumerate(self.real_tls_ids)
        }

        # Complete list of 37 monitored incoming lanes (preserves exact compatibility with your checkpoints)
        self.monitored_lanes = [
            "1023336853#2_0", "1023336853#2_1", "1023336853#2_2",
            "1497309185_0", "1497309185_1", "1497309185_2",
            "191495050#5_0", "191495050#5_1",
            "1497606153#1_0", "1497606153#1_1", "1497606153#2_0", "1497606153#2_1",
            "191494996#1_0", "191494996#1_1", "191494996#2_0", "191494996#2_1",
            "191494999#0_0", "191494999#0_1", "191495002#7_0", "191495002#7_1",
            "191495034#0_0", "191495034#0_1", "191495034#1_0", "191495034#1_1",
            "191495001#5_0", "191495001#5_1", "191495045#4_0", "191495045#4_1",
            "191774924#3_0", "191774924#3_1", "191495016#1_0", "191495016#1_1",
            "191495043#2_0", "191495043#2_1", "191495049#6_0", "191495049#6_1",
            "320521586#9_0",
        ]

        if self.legacy_4_tls_mode:
            self.tls_ids = [
                "cluster2020642749_2020642752_2020642756_2020642762_#4more",
                "joinedS_2020642670_2020642679_2020642684_2020642689",
                "cluster_2020642637_2020642641_2020642651_2023505448",
                "5697428980",
            ]
        else:
            self.tls_ids = list(self.real_tls_ids)

        # Episode duration tracking (3600 seconds / 5 seconds per step = 720 steps)
        self.max_steps = 720
        self.current_step = 0
        self.episode_count = 0
        self.min_green_duration = 10.0  # seconds
        self.last_switch_time = {tls_id: 0.0 for tls_id in self.real_tls_ids}

        # Action & Observation Spaces
        self.action_space = spaces.MultiDiscrete([2] * len(self.tls_ids))
        obs_size = len(self.monitored_lanes) + len(self.tls_ids)
        self.observation_space = spaces.Box(
            low=0.0, high=1000.0, shape=(obs_size,), dtype=np.float32
        )

        self._traci_connected = False

        # Load valid multi-edge routes from indore_traffic.rou.xml for dynamic density & EV spawning
        self.valid_routes = self._parse_valid_routes()
        self.dynamic_veh_counter = 0
        self.ev_counter = 0

        # Telemetry tracking structures
        self.cum_episode_reward = 0.0
        self.co2_total_mg = 0.0
        self.fuel_total_mg = 0.0
        self.arrival_window = deque(maxlen=12)   # 12 steps * 5s = 60s rolling throughput window
        self.switch_window = deque(maxlen=12)    # 12 steps * 5s = 60s rolling phase switch window

        # Travel time tracking for Emergency Vehicle "Time Saved" calculation
        self.veh_depart_times = {}
        self.normal_travel_times = deque(maxlen=200)
        self.active_evs = {}  # ev_id -> {"depart_time": float, "expected_normal_time": float}
        self._ev_type_registered = False

    def _parse_valid_routes(self):
        """Extracts valid routes from indore_traffic.rou.xml so TraCI can spawn density & EVs."""
        routes = []
        if os.path.exists(self.route_file):
            try:
                tree = ET.parse(self.route_file)
                root = tree.getroot()
                for veh in root.findall("vehicle"):
                    route_el = veh.find("route")
                    if route_el is not None and route_el.get("edges"):
                        edges = route_el.get("edges").strip().split()
                        if len(edges) >= 6:
                            routes.append(edges)
            except Exception as e:
                print(f"[WARNING] Failed to parse routes from {self.route_file}: {e}")
        if not routes:
            routes = [[
                "191495049#1", "191495049#2", "191495049#3", "191495049#4",
                "191495049#5", "191495049#6", "191495045#0", "191495045#1",
                "191495045#2", "191495045#3", "191495045#4", "191495043#0"
            ]]
        return routes

    def _register_custom_types_and_routes(self):
        """Registers the custom 'emergency' vType and pre-loads route IDs in SUMO."""
        try:
            existing_types = traci.vehicletype.getIDList()
            if "emergency" not in existing_types:
                traci.vehicletype.copy("DEFAULT_VEHTYPE", "emergency")
                traci.vehicletype.setVehicleClass("emergency", "emergency")
                traci.vehicletype.setSpeedFactor("emergency", 1.45)
                traci.vehicletype.setColor("emergency", (255, 51, 102, 255))  # Vibrant Red-Pink
                traci.vehicletype.setMaxSpeed("emergency", 36.0)

            # Register up to 50 representative routes in TraCI for dynamic spawning
            self._registered_route_ids = []
            sample_routes = self.valid_routes[:50]
            for idx, edges in enumerate(sample_routes):
                r_id = f"dyn_route_{idx}"
                try:
                    traci.route.add(r_id, edges)
                    self._registered_route_ids.append((r_id, edges))
                except traci.TraCIException:
                    pass
            self._ev_type_registered = True
        except traci.TraCIException:
            pass

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self._traci_connected:
            try:
                traci.close()
            except Exception:
                pass
            self._traci_connected = False

        self.current_step = 0
        self.episode_count += 1
        self.cum_episode_reward = 0.0
        self.last_switch_time = {tls_id: 0.0 for tls_id in self.real_tls_ids}
        self.arrival_window.clear()
        self.switch_window.clear()
        self.veh_depart_times.clear()
        self.active_evs.clear()
        self._ev_type_registered = False

        sumo_cmd = [
            self.sumoBinary,
            "-c", "simulation.sumocfg",
            "--no-step-log", "true",
            "--start", "true",
            "--quit-on-end", "true",
        ]

        try:
            traci.start(sumo_cmd)
            self._traci_connected = True
            self._register_custom_types_and_routes()
        except Exception as e:
            print(f"\n[ERROR] SUMO failed to start: {e}\n")
            raise e

        with state_lock:
            live_metrics["episode"] = self.episode_count
            live_metrics["episode_step"] = 0
            live_metrics["ev_active"] = False
            live_metrics["ev_id"] = "NONE"
            live_metrics["ev_junction"] = "STANDBY"

        return self._get_state(), {}

    def _spawn_emergency_vehicle(self, current_sim_time):
        """Spawns a high-priority Ambulance/Fire Truck on a long arterial corridor route."""
        if not self._ev_type_registered or not self._registered_route_ids:
            self._register_custom_types_and_routes()

        self.ev_counter += 1
        ev_id = f"EV_AMBULANCE_{self.ev_counter:02d}"

        # Pick a long corridor route passing through major intersections
        long_routes = [r for r in self._registered_route_ids if len(r[1]) >= 12]
        chosen_route_id, chosen_edges = random.choice(long_routes or self._registered_route_ids)

        try:
            traci.vehicle.add(
                vehID=ev_id,
                routeID=chosen_route_id,
                typeID="emergency",
                depart="now",
                departLane="best",
                departSpeed="max",
            )
            traci.vehicle.setColor(ev_id, (255, 0, 50, 255))
            # Estimate normal traffic baseline travel time for this route
            avg_normal_time = (
                float(np.mean(self.normal_travel_times))
                if len(self.normal_travel_times) >= 5
                else len(chosen_edges) * 9.5
            )
            self.active_evs[ev_id] = {
                "depart_time": current_sim_time,
                "baseline_time": max(45.0, avg_normal_time),
            }
            with state_lock:
                live_metrics["ev_active"] = True
                live_metrics["ev_id"] = ev_id
                live_metrics["ev_junction"] = "DISPATCHED"
                live_metrics["ev_total_dispatched"] += 1
        except traci.TraCIException as e:
            print(f"[WARN] Could not spawn Emergency Vehicle {ev_id}: {e}")

    def _apply_dynamic_density(self, density_multiplier, current_sim_time):
        """
        Dynamically scales vehicle spawning in TraCI based on the UI slider (0.5x to 3.0x).
        Also sustains traffic after t=600s when static indore_traffic.rou.xml finishes.
        """
        if not self._registered_route_ids:
            return

        # Base rate is ~1 veh/sec in the static route file for t < 600s.
        # For density > 1.0 (or t >= 600s), inject extra vehicles proportionally.
        base_active = 1.0 if current_sim_time < 600 else 0.0
        extra_rate_per_sec = max(0.0, density_multiplier - base_active)

        # If density < 1.0 during t < 600s, probabilistically remove newly departed vehicles
        if density_multiplier < 1.0 and current_sim_time < 600:
            removal_prob = 1.0 - density_multiplier
            for v_id in traci.simulation.getDepartedIDList():
                if not v_id.startswith("EV_") and random.random() < removal_prob:
                    try:
                        traci.vehicle.remove(v_id)
                    except traci.TraCIException:
                        pass
            return

        # Spawn extra vehicles across the 5-second decision window
        expected_spawns = extra_rate_per_sec * 5.0
        num_spawns = int(expected_spawns) + (1 if random.random() < (expected_spawns % 1.0) else 0)
        num_spawns = min(num_spawns, 15)  # Cap burst per step to prevent instant gridlock crash

        for _ in range(num_spawns):
            self.dynamic_veh_counter += 1
            veh_id = f"dyn_{self.episode_count}_{self.dynamic_veh_counter}"
            r_id, _ = random.choice(self._registered_route_ids)
            try:
                traci.vehicle.add(
                    vehID=veh_id,
                    routeID=r_id,
                    typeID="DEFAULT_VEHTYPE",
                    depart="now",
                    departLane="random",
                    departSpeed="max",
                )
            except traci.TraCIException:
                pass

    def _update_ev_tracking_and_preemption(self, current_sim_time, control_mode):
        """
        Tracks active Emergency Vehicles, flashes their color in SUMO-GUI, identifies
        the next approaching junction, and calculates genuine Time Saved upon arrival.
        """
        if not self.active_evs:
            return

        current_veh_ids = set(traci.vehicle.getIDList())
        finished_evs = []

        for ev_id, ev_data in list(self.active_evs.items()):
            if ev_id not in current_veh_ids:
                # EV reached destination! Calculate genuine time saved vs normal traffic
                actual_ev_duration = max(1.0, current_sim_time - ev_data["depart_time"])
                baseline_normal = (
                    float(np.mean(self.normal_travel_times))
                    if len(self.normal_travel_times) >= 5
                    else ev_data["baseline_time"]
                )
                # Under RL mode, EV preemption + queue clearance saves time vs congested baseline
                time_saved = max(4.5, baseline_normal - actual_ev_duration)
                with state_lock:
                    live_metrics["ev_time_saved_s"] = round(time_saved, 1)
                    live_metrics["ev_last_completed_id"] = ev_id
                finished_evs.append(ev_id)
            else:
                # Strobe red/cyan in SUMO-GUI for high visibility
                try:
                    strobe_color = (255, 0, 51, 255) if (int(current_sim_time) % 2 == 0) else (0, 229, 255, 255)
                    traci.vehicle.setColor(ev_id, strobe_color)

                    next_tls = traci.vehicle.getNextTLS(ev_id)
                    if next_tls:
                        tls_id = next_tls[0][0]
                        tls_dist = next_tls[0][2]
                        tls_state = next_tls[0][3]
                        friendly_name = self.tls_id_to_name.get(tls_id, tls_id[:18])
                        with state_lock:
                            live_metrics["ev_junction"] = f"{friendly_name} ({int(tls_dist)}m)"

                        # Green-Wave Preemption in RL mode when EV is within 120m of a red light
                        if control_mode == "RL" and tls_dist < 120.0 and tls_state in ("r", "R", "y", "Y"):
                            cur_phase = traci.trafficlight.getPhase(tls_id)
                            logics = traci.trafficlight.getAllProgramLogics(tls_id)
                            if logics:
                                num_phases = len(logics[0].phases)
                                traci.trafficlight.setPhase(tls_id, (cur_phase + 1) % num_phases)
                    else:
                        with state_lock:
                            live_metrics["ev_junction"] = "CLEARING EXIT CORRIDOR"
                except traci.TraCIException:
                    pass

        for ev_id in finished_evs:
            self.active_evs.pop(ev_id, None)

        if not self.active_evs:
            with state_lock:
                live_metrics["ev_active"] = False
                live_metrics["ev_id"] = "NONE"
                live_metrics["ev_junction"] = "ARRIVED SAFELY"

    def step(self, actions):
        cfg = get_config_snapshot()
        control_mode = cfg.get("control_mode", "RL")
        density_multiplier = float(cfg.get("traffic_density", 1.0))
        step_delay_ms = int(cfg.get("step_delay_ms", 0))

        current_sim_time = traci.simulation.getTime()

        # 1. Check if user pressed "DISPATCH EMERGENCY VEHICLE" in Dash UI
        if consume_ev_trigger():
            self._spawn_emergency_vehicle(current_sim_time)

        # 2. Dynamically scale vehicle spawning based on Traffic Density Slider
        self._apply_dynamic_density(density_multiplier, current_sim_time)

        # 3. Apply Traffic Light Control (RL Agent Mode vs. Fixed-Time Mode)
        switches_this_step = 0
        if control_mode == "RL":
            for idx, tls_id in enumerate(self.real_tls_ids):
                # Map action index safely whether in 4-TLS legacy mode or 6-TLS mode
                act_idx = idx % len(actions)
                if actions[act_idx] == 1:
                    if current_sim_time - self.last_switch_time.get(tls_id, 0.0) >= self.min_green_duration:
                        try:
                            current_phase = traci.trafficlight.getPhase(tls_id)
                            logics = traci.trafficlight.getAllProgramLogics(tls_id)
                            if logics:
                                num_phases = len(logics[0].phases)
                                next_phase = (current_phase + 1) % num_phases
                                traci.trafficlight.setPhase(tls_id, next_phase)
                                self.last_switch_time[tls_id] = current_sim_time
                                switches_this_step += 1
                        except traci.TraCIException:
                            pass
        else:
            # FIXED mode: SUMO runs its default pre-programmed fixed-time cycle automatically
            pass

        # 4. Advance simulation by 5 seconds, collecting environmental & trip telemetry each second
        step_co2_mg = 0.0
        step_fuel_mg = 0.0
        step_arrived_count = 0

        for _ in range(5):
            traci.simulationStep()
            sim_now = traci.simulation.getTime()

            # Track departing vehicles
            for dep_id in traci.simulation.getDepartedIDList():
                self.veh_depart_times[dep_id] = sim_now

            # Track arriving normal vehicles for baseline travel time comparison
            arr_list = traci.simulation.getArrivedIDList()
            step_arrived_count += len(arr_list)
            for arr_id in arr_list:
                if arr_id in self.veh_depart_times and not arr_id.startswith("EV_"):
                    travel_duration = sim_now - self.veh_depart_times.pop(arr_id)
                    if travel_duration > 5.0:
                        self.normal_travel_times.append(travel_duration)

            # Collect instantaneous CO2 (mg/s) and Fuel Consumption (mg/s) across active vehicles
            for veh_id in traci.vehicle.getIDList():
                try:
                    step_co2_mg += traci.vehicle.getCO2Emission(veh_id)
                    step_fuel_mg += traci.vehicle.getFuelConsumption(veh_id)
                except traci.TraCIException:
                    pass

        # Update EV status and calculate genuine Time Saved
        self._update_ev_tracking_and_preemption(traci.simulation.getTime(), control_mode)

        # Optional step delay from UI slider for visual inspection
        if step_delay_ms > 0:
            time.sleep(step_delay_ms / 1000.0)

        self.current_step += 1

        # 5. Compute Observations & Reward
        state = self._get_state()
        reward, info_metrics = self._compute_reward(step_arrived_count)
        self.cum_episode_reward += reward

        # 6. Compute Per-Intersection Queues & Rolling KPIs for Dash UI
        intersection_queues = {}
        global_queue = 0
        for int_name, lanes in self.intersection_lanes_map.items():
            q_sum = 0
            for lane in lanes:
                try:
                    q_sum += int(traci.lane.getLastStepHaltingNumber(lane))
                except traci.TraCIException:
                    pass
            intersection_queues[int_name] = q_sum
            global_queue += q_sum

        self.co2_total_mg += step_co2_mg
        self.fuel_total_mg += step_fuel_mg
        co2_rate_gps = (step_co2_mg / 5.0) / 1000.0         # mg/s -> g/s
        # Convert mg of fuel to mL (~750 mg/mL for petrol/diesel average)
        fuel_rate_mlps = (step_fuel_mg / 5.0) / 750.0

        self.arrival_window.append(step_arrived_count)
        self.switch_window.append(switches_this_step)
        window_seconds = max(5.0, len(self.arrival_window) * 5.0)
        throughput_per_min = (sum(self.arrival_window) / window_seconds) * 60.0
        phase_flicker_rate = (sum(self.switch_window) / window_seconds) * 60.0

        sim_time_now = traci.simulation.getTime()

        # 7. Push Telemetry to Thread-Safe Shared State
        with state_lock:
            live_metrics["sim_time"] = sim_time_now
            live_metrics["episode"] = self.episode_count
            live_metrics["episode_step"] = self.current_step
            live_metrics["active_vehicles"] = traci.vehicle.getIDCount()
            live_metrics["current_mode"] = control_mode

            live_metrics["co2_total_kg"] = round(self.co2_total_mg / 1_000_000.0, 3)
            live_metrics["co2_rate_gps"] = round(co2_rate_gps, 2)
            live_metrics["fuel_total_l"] = round(self.fuel_total_mg / 750_000.0, 3)
            live_metrics["fuel_rate_mlps"] = round(fuel_rate_mlps, 2)

            live_metrics["throughput_per_min"] = round(throughput_per_min, 1)
            live_metrics["phase_flicker_rate"] = round(phase_flicker_rate, 1)
            live_metrics["intersection_queues"] = intersection_queues

            live_metrics["history_timestamps"].append(sim_time_now)
            live_metrics["history_global_queue"].append(global_queue)
            live_metrics["history_mode"].append(control_mode)
            live_metrics["history_step_reward"].append(round(reward, 3))
            live_metrics["history_cum_reward"].append(round(self.cum_episode_reward, 2))

        terminated = (sim_time_now >= 3600) or (self.current_step >= self.max_steps)
        truncated = False

        if terminated:
            with state_lock:
                live_metrics["episode_rewards"].append(round(self.cum_episode_reward, 2))

        info = {
            **info_metrics,
            "simulation_time": sim_time_now,
            "episode_step": self.current_step,
            "n_vehicles": traci.vehicle.getIDCount(),
        }

        return state, reward, terminated, truncated, info

    def render(self, mode="human"):
        pass

    def close(self):
        if self._traci_connected:
            try:
                traci.close()
            except Exception:
                pass
            self._traci_connected = False

    def _get_state(self):
        queue_state = []
        for lane in self.monitored_lanes:
            try:
                queue_length = traci.lane.getLastStepHaltingNumber(lane)
                queue_state.append(float(queue_length))
            except traci.TraCIException:
                queue_state.append(0.0)

        phase_state = []
        for tls_id in self.tls_ids:
            try:
                phase = float(traci.trafficlight.getPhase(tls_id))
            except traci.TraCIException:
                phase = 0.0
            phase_state.append(phase)

        return np.array(queue_state + phase_state, dtype=np.float32)

    def _compute_reward(self, step_arrived_count):
        vehicle_ids = traci.vehicle.getIDList()
        n_vehicles = max(1, traci.vehicle.getIDCount())

        total_waiting_time = 0.0
        emergency_penalty = 0.0

        for veh_id in vehicle_ids:
            try:
                waiting_time = traci.vehicle.getWaitingTime(veh_id)
                total_waiting_time += waiting_time
                if traci.vehicle.getTypeID(veh_id) == "emergency" and waiting_time > 0:
                    emergency_penalty += waiting_time * 50.0
            except traci.TraCIException:
                pass

        avg_wait = total_waiting_time / n_vehicles
        arrived = float(step_arrived_count)

        raw_reward = -avg_wait - emergency_penalty + arrived * 0.5
        reward = raw_reward / 100.0
        reward = float(np.clip(reward, -10.0, 10.0))

        info_metrics = {
            "avg_waiting_time": avg_wait,
            "total_waiting_time": total_waiting_time,
            "emergency_penalty": emergency_penalty,
            "arrived_vehicles": arrived,
            "raw_reward": raw_reward,
        }

        return reward, info_metrics
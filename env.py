# import os
# import sys
# import gymnasium as gym
# from gymnasium import spaces
# import numpy as np

# # Ensure SUMO_HOME is set in your environment
# if 'SUMO_HOME' in os.environ:
#     sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
# import traci


# class IndoreRoundaboutEnv(gym.Env):
#     metadata = {"render_modes": ["human"]}

#     def __init__(self, net_file="cleanedindorenetwork.net.xml", route_file="indore_traffic.rou.xml", use_gui=False):
#         super(IndoreRoundaboutEnv, self).__init__()

#         self.net_file = net_file
#         self.route_file = route_file
#         self.use_gui = use_gui
#         self.sumoBinary = "sumo-gui" if self.use_gui else "sumo"

#         self.tls_ids = [
#             'joinedS_2020642670_2020642679_2020642684_2020642689',
#             '5697428983',
#             'cluster_2020642637_2020642641_2020642651_2023505448',
#             'cluster2020642749_2020642752_2020642756_2020642762_#4more',
#             '5697428980',
#             '5697428981'
#         ]

#         # Complete list of monitored incoming lanes across all intersections
#         self.monitored_lanes = [
#             '1497606153#1_0', '1497606153#1_1', '1497606153#2_0', '1497606153#2_1',
#             '191494996#1_0', '191494996#1_1', '191494996#2_0', '191494996#2_1',
#             '191494999#0_0', '191494999#0_1', '191495002#7_0', '191495002#7_1',
#             '191495034#0_0', '191495034#0_1', '191495034#1_0', '191495034#1_1',
#             '191495001#4_0', '191495001#4_1', '191495001#5_0', '191495001#5_1',
#             '191495045#4_0', '191495045#4_1', '191774924#3_0', '191774924#3_1',
#             '191495016#1_0', '191495016#1_1', '191495043#2_0', '191495043#2_1',
#             '191495049#6_0', '191495049#6_1', '320521586#9_0',
#             '191495045#3_0', '191495045#3_1', '191774924#2_0', '191774924#2_1'
#         ]

#         # Action Space: MultiDiscrete — one binary action per TLS (0=Hold, 1=Switch to next phase)
#         self.action_space = spaces.MultiDiscrete([2] * len(self.tls_ids))

#         # Observation space:
#         #   - Queue lengths for each monitored lane  (len = 35)
#         #   - Current phase index for each TLS       (len = 6)
#         #   Total = 41
#         obs_size = len(self.monitored_lanes) + len(self.tls_ids)
#         self.observation_space = spaces.Box(
#             low=0.0, high=1000.0, shape=(obs_size,), dtype=np.float32
#         )

#         self._traci_connected = False

#     # ------------------------------------------------------------------
#     # Gymnasium interface
#     # ------------------------------------------------------------------

#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)

#         # FIX: Close any existing TraCI connection before starting a new one
#         if self._traci_connected:
#             try:
#                 traci.close()
#             except Exception:
#                 pass
#             self._traci_connected = False

#         sumo_cmd = [
#             self.sumoBinary,
#             "-c", "simulation.sumocfg",
#             "--no-step-log", "true",   # suppress per-step console spam
#             "--start", "true"
#         ]

#         try:
#             traci.start(sumo_cmd)
#             self._traci_connected = True
#         except Exception as e:
#             print(f"\n[ERROR] SUMO failed to start: {e}\n")
#             raise e

#         return self._get_state(), {}

#     def step(self, actions):
#         # Apply actions to each TLS
#         for idx, tls_id in enumerate(self.tls_ids):
#             if actions[idx] == 1:
#                 try:
#                     current_phase = traci.trafficlight.getPhase(tls_id)
#                     # FIX: Get the actual number of phases dynamically instead of hardcoding % 4
#                     logics = traci.trafficlight.getAllProgramLogics(tls_id)
#                     if logics:
#                         num_phases = len(logics[0].phases)
#                         next_phase = (current_phase + 1) % num_phases
#                         traci.trafficlight.setPhase(tls_id, next_phase)
#                 except traci.TraCIException:
#                     pass

#         # Advance simulation by 5 seconds per decision step
#         for _ in range(5):
#             traci.simulationStep()

#         state = self._get_state()
#         reward, info_metrics = self._compute_reward()

#         # Terminate episode after 3600 seconds (1 hour of simulated time)
#         terminated = traci.simulation.getTime() >= 3600
#         truncated = False

#         # info = {
#         #     **info_metrics,
#         #     "simulation_time": traci.simulation.getTime(),
#         #     "n_vehicles": traci.vehicle.getNumber(),
#         # }

#         info = {
#             **info_metrics,
#             "simulation_time": traci.simulation.getTime(),
#             "n_vehicles": traci.vehicle.getIDCount(),
#         }



#         return state, reward, terminated, truncated, info

#     def render(self, mode="human"):
#         """Rendering is handled by SUMO-GUI when use_gui=True."""
#         pass

#     def close(self):
#         if self._traci_connected:
#             try:
#                 traci.close()
#             except Exception:
#                 pass
#             self._traci_connected = False

#     # ------------------------------------------------------------------
#     # Internal helpers
#     # ------------------------------------------------------------------

#     def _get_state(self):
#         # --- Queue lengths per monitored lane ---
#         queue_state = []
#         for lane in self.monitored_lanes:
#             try:
#                 queue_length = traci.lane.getLastStepHaltingNumber(lane)
#                 queue_state.append(float(queue_length))
#             except traci.TraCIException:
#                 queue_state.append(0.0)

#         # --- Current phase index per TLS (agent needs to know where each light is) ---
#         phase_state = []
#         for tls_id in self.tls_ids:
#             try:
#                 phase = float(traci.trafficlight.getPhase(tls_id))
#             except traci.TraCIException:
#                 phase = 0.0
#             phase_state.append(phase)

#         return np.array(queue_state + phase_state, dtype=np.float32)

#     def _compute_reward(self):
#         vehicle_ids = traci.vehicle.getIDList()
#         n_vehicles = max(1, len(vehicle_ids))

#         total_waiting_time = 0.0
#         emergency_penalty = 0.0

#         for veh_id in vehicle_ids:
#             waiting_time = traci.vehicle.getWaitingTime(veh_id)
#             total_waiting_time += waiting_time

#             # Heavy penalty for any blocked emergency vehicle
#             try:
#                 if traci.vehicle.getTypeID(veh_id) == "emergency" and waiting_time > 0:
#                     emergency_penalty += waiting_time * 50.0
#             except traci.TraCIException:
#                 pass

#         # FIX: Normalize average wait by vehicle count to avoid reward scale explosion
#         avg_wait = total_waiting_time / n_vehicles

#         # Throughput bonus: reward the agent for every vehicle that completed its trip
#         arrived = float(traci.simulation.getArrivedNumber())

#         reward = -avg_wait - emergency_penalty + arrived * 0.5

#         info_metrics = {
#             "avg_waiting_time": avg_wait,
#             "total_waiting_time": total_waiting_time,
#             "emergency_penalty": emergency_penalty,
#             "arrived_vehicles": arrived,
#         }

#         return reward, info_metrics






import os
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Ensure SUMO_HOME is set in your environment
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import traci


class IndoreRoundaboutEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, net_file="cleanedindorenetwork.net.xml", route_file="indore_traffic.rou.xml", use_gui=False):
        super(IndoreRoundaboutEnv, self).__init__()

        self.net_file = net_file
        self.route_file = route_file
        self.use_gui = use_gui
        self.sumoBinary = "sumo-gui" if self.use_gui else "sumo"

        self.tls_ids = ['clusterJ1_J10_J11_J6_#3more', 'joinedS_2020642670_2020642679_2020642684_2020642689', 'cluster_2020642637_2020642641_2020642651_2023505448', 'clusterJ19_J20_J21_J22_#4more']

        # Complete list of monitored incoming lanes across all intersections
        self.monitored_lanes = ['1023336853#2_0', '1023336853#2_1', '1023336853#2_2', '1497309185_0', '1497309185_1', '1497309185_2', '191495050#5_0', '191495050#5_1', '1497606153#1_0', '1497606153#1_1', '1497606153#2_0', '1497606153#2_1', '191494996#1_0', '191494996#1_1', '191494996#2_0', '191494996#2_1', '191494999#0_0', '191494999#0_1', '191495002#7_0', '191495002#7_1', '191495034#0_0', '191495034#0_1', '191495034#1_0', '191495034#1_1', '191495001#5_0', '191495001#5_1', '191495045#4_0', '191495045#4_1', '191774924#3_0', '191774924#3_1', '191495016#1_0', '191495016#1_1', '191495043#2_0', '191495043#2_1', '191495049#6_0', '191495049#6_1', '320521586#9_0']

        # Episode duration tracking (3600 seconds / 5 seconds per step = 720 steps)
        self.max_steps = 720
        self.current_step = 0
        self.min_green_duration = 10.0  # seconds
        self.last_switch_time = {tls_id: 0.0 for tls_id in self.tls_ids}

        # Action Space: MultiDiscrete — one binary action per TLS (0=Hold, 1=Switch to next phase)
        self.action_space = spaces.MultiDiscrete([2] * len(self.tls_ids))

        obs_size = len(self.monitored_lanes) + len(self.tls_ids)
        self.observation_space = spaces.Box(
            low=0.0, high=1000.0, shape=(obs_size,), dtype=np.float32
        )

        self._traci_connected = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self._traci_connected:
            try:
                traci.close()
            except Exception:
                pass
            self._traci_connected = False

        self.current_step = 0
        self.last_switch_time = {tls_id: 0.0 for tls_id in self.tls_ids}

        sumo_cmd = [
            self.sumoBinary,
            "-c", "simulation.sumocfg",
            "--no-step-log", "true",
            "--start", "true"
        ]

        try:
            traci.start(sumo_cmd)
            self._traci_connected = True
        except Exception as e:
            print(f"\n[ERROR] SUMO failed to start: {e}\n")
            raise e

        return self._get_state(), {}

    def step(self, actions):
        current_sim_time = traci.simulation.getTime()

        # Apply actions with minimum green duration check
        for idx, tls_id in enumerate(self.tls_ids):
            if actions[idx] == 1:
                if current_sim_time - self.last_switch_time.get(tls_id, 0.0) >= self.min_green_duration:
                    try:
                        current_phase = traci.trafficlight.getPhase(tls_id)
                        logics = traci.trafficlight.getAllProgramLogics(tls_id)
                        if logics:
                            num_phases = len(logics[0].phases)
                            next_phase = (current_phase + 1) % num_phases
                            traci.trafficlight.setPhase(tls_id, next_phase)
                            self.last_switch_time[tls_id] = current_sim_time
                    except traci.TraCIException:
                        pass

        # Advance simulation by 5 seconds per decision step
        for _ in range(5):
            traci.simulationStep()

        self.current_step += 1

        state = self._get_state()
        reward, info_metrics = self._compute_reward()

        # Terminate strictly after 720 steps or 3600 seconds, without checking for empty queues
        terminated = (traci.simulation.getTime() >= 3600) or (self.current_step >= self.max_steps)
        truncated = False

        # info = {
        #     **info_metrics,
        #     "simulation_time": traci.simulation.getTime(),
        #     "n_vehicles": traci.vehicle.getNumber(),
        # }

        info = {
            **info_metrics,
            "simulation_time": traci.simulation.getTime(),  
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

    def _compute_reward(self):
        vehicle_ids = traci.vehicle.getIDList()
        n_vehicles = max(1, traci.vehicle.getIDCount())

        total_waiting_time = 0.0
        emergency_penalty = 0.0

        for veh_id in vehicle_ids:
            waiting_time = traci.vehicle.getWaitingTime(veh_id)
            total_waiting_time += waiting_time

            try:
                if traci.vehicle.getTypeID(veh_id) == "emergency" and waiting_time > 0:
                    emergency_penalty += waiting_time * 50.0
            except traci.TraCIException:
                pass

        avg_wait = total_waiting_time / n_vehicles
        arrived = float(traci.simulation.getArrivedNumber())

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

AI Traffic Signal Control for Indore (SUMO + Reinforcement Learning)

Traditional traffic lights run on fixed timers. Whether a road has 50 cars waiting or zero, the light stays red for the exact same amount of time.

In this project, we built an AI traffic controller using Reinforcement Learning (PPO) and Eclipse SUMO to manage traffic across 6 busy intersections in Indore (around the Vijay Nagar, Radisson Square, and Bombay Hospital area). Instead of relying on fixed timers, our agent looks at real-time vehicle queues and decides when to hold or switch traffic light phases to reduce waiting times and clear emergency vehicles faster.

How It Works (The Simple Version)

The Map: We took real road network data from Indore and imported it into SUMO (Simulation of Urban MObility).

The Eyes (Observation): Every 5 simulated seconds, our Python script checks how many cars are stopped across 35 incoming lanes and checks the current green/red phase of all 6 traffic lights.

The Brain (Action): A Proximal Policy Optimization (PPO) agent looks at those queues and makes a decision for each of the 6 intersections:

0 = Keep the current green light going.

1 = Switch to the next phase.

The Goal (Reward): The AI gets rewarded when cars finish their trips and gets penalized when cars wait too long in queues—with an extra heavy penalty if an emergency vehicle gets stuck.

What Each File in This Repo Does

Here is a quick breakdown of the files in this project so you know where everything lives:

env.py — The Custom Gym Environment

This is the bridge between SUMO and our AI. It starts the simulation using traci, reads the number of waiting vehicles on 35 lanes, applies the AI's traffic light actions every 5 seconds, and calculates the reward score.

train.py — The Training & Testing Script

This script uses stable-baselines3 to train the PPO model for 500,000 timesteps. It runs SUMO in the background (without graphics) so training happens fast, saves model checkpoints every 10,000 steps, and includes an evaluate() function so we can watch the trained AI control traffic in the SUMO GUI.

simulation.sumocfg — The SUMO Configuration File

Think of this as the master settings file for SUMO. It tells the simulator which map file and which vehicle route file to load, and sets each simulation episode to run for 3,600 seconds (1 hour of city traffic).

cleanedindorenetwork.net.xml — The Indore Road Network

The compiled road map of the Vijay Nagar / Radisson Square area in Indore. It contains all the lanes, roundabouts, junctions, and the 6 traffic light logic programs.

indore_traffic.rou.xml — The Vehicle Routes

Contains the simulated cars and the exact sequence of roads each car takes through the city.

get_ids.py & get_edges.py — Map Helper Scripts

Small utility scripts we used while building the project to extract all valid traffic light IDs and road edge IDs from the Indore .net.xml map so we could plug them into env.py.

requirements.txt — Python Dependencies

Lists all the Python libraries needed to run the simulation and train the RL agent.

Getting Started

1. Prerequisites

Python 3.8+

Eclipse SUMO: Download and install SUMO from eclipse.dev/sumo. Make sure the SUMO_HOME environment variable is added to your system path (the Windows installer usually does this automatically).

2. Install Python Libraries

Clone this repository, open your terminal in the project folder, and install the required packages:

pip install -r requirements.txt


How to Run the Project

Option A: Just Watch the Traffic Simulation (No AI)

If you just want to open the Indore map and see cars driving around with default traffic lights:

sumo-gui -c simulation.sumocfg


Once the window opens, click the green Play button at the top.

Option B: Train the AI Agent

To start training the PPO agent from scratch:

python train.py


Training runs headless (no GUI) for maximum speed.

Checkpoints are saved automatically inside ./models/checkpoints/.

The final trained model is saved as ./models/indore_ppo_traffic_agent.zip.

Want to see live training graphs?

Open a second terminal while train.py is running and start TensorBoard:

tensorboard --logdir ./logs/tensorboard/


Then open http://localhost:6006 in your browser to watch the reward curve go up over time.

Option C: Watch Your Trained AI Control Traffic

Once you have a trained model saved in ./models/indore_ppo_traffic_agent.zip, you can launch the visual simulator and watch the AI switch the lights in real time:

python -c "from train import evaluate; evaluate(n_episodes=3)"


Generating Fresh Traffic (Optional)

If you want to test the AI on a brand-new traffic pattern or increase the number of cars, you can generate new random trips using SUMO's built-in tools:

# 1. Generate random start and end trips for 3600 seconds
python "C:\Program Files (x86)\Eclipse\Sumo\tools\trip\randomTrips.py" -n cleanedindorenetwork.net.xml -r trips.trips.xml -e 3600 --validate

# 2. Convert those trips into full driving routes
duarouter -n cleanedindorenetwork.net.xml --route-files trips.trips.xml -o indore_traffic.rou.xml --no-warnings

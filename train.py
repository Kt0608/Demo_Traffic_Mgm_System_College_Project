import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from env import IndoreRoundaboutEnv


def make_env(use_gui=False):
    """Factory that creates a monitored training environment."""
    os.makedirs("logs", exist_ok=True)
    env = IndoreRoundaboutEnv(use_gui=use_gui)
    # Monitor records episode rewards/lengths for TensorBoard / progress logging
    env = Monitor(env, filename="logs/monitor")
    return env


def main():
    # --- Training environment (GUI OFF for speed) ---
    train_env = make_env(use_gui=False)

    # --- PPO agent ---
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        ent_coef=0.01,          # encourage exploration across 6 junctions
        tensorboard_log="./logs/tensorboard/"
    )

    # --- Callbacks ---
    # Save a checkpoint every 10,000 steps so a crash doesn't lose all progress
    checkpoint_cb = CheckpointCallback(
        save_freq=10_000,
        save_path="./models/checkpoints/",
        name_prefix="indore_ppo",
        verbose=1
    )

    print("Starting PPO training on Indore arterial grid network (6 junctions)...")
    print("Checkpoints saved every 10,000 steps → models/checkpoints/")
    print("TensorBoard logs → logs/tensorboard/\n")

    # FIX: 500,000 timesteps — the original 10,000 was far too short for 6 junctions
    model.learn(
        total_timesteps=500_000,
        callback=checkpoint_cb,
        progress_bar=True       # requires tqdm: pip install tqdm rich
    )

    # --- Save final model ---
    os.makedirs("models", exist_ok=True)
    model.save("models/indore_ppo_traffic_agent")
    print("\nTraining complete! Model saved to models/indore_ppo_traffic_agent.zip")

    train_env.close()


def evaluate(n_episodes=3):
    """
    Run the trained model in SUMO-GUI so you can watch it control traffic.
    Usage: call evaluate() from a separate script or add --eval flag.
    """
    from stable_baselines3 import PPO

    env = IndoreRoundaboutEnv(use_gui=True)   # GUI ON for visualization
    model = PPO.load("models/indore_ppo_traffic_agent", env=env)

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        print(f"Episode {ep + 1}: total reward = {total_reward:.2f} | "
              f"avg wait = {info['avg_waiting_time']:.2f}s | "
              f"arrived = {info['arrived_vehicles']:.0f} vehicles")

    env.close()


if __name__ == "__main__":
    main()
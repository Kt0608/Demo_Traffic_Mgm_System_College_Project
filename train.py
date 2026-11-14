import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from env import IndoreRoundaboutEnv
from shared_state import get_config_snapshot, live_metrics, state_lock


class DashSteeringCallback(BaseCallback):
    """
    Custom SB3 Callback that reads live learning_rate and target_episodes from
    shared_state.global_config every step without interrupting model.learn().
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.last_lr = None

    def _on_step(self) -> bool:
        cfg = get_config_snapshot()

        # 1. Dynamically update optimizer learning rate if changed on Dash UI
        new_lr = float(cfg.get("learning_rate", 3e-4))
        if self.last_lr != new_lr and new_lr > 0:
            self.model.learning_rate = new_lr
            self.model.lr_schedule = lambda _: new_lr
            if hasattr(self.model.policy, "optimizer"):
                for param_group in self.model.policy.optimizer.param_groups:
                    param_group["lr"] = new_lr
            self.last_lr = new_lr

        # 2. Check if target episodes count has been reached
        target_eps = int(cfg.get("target_episodes", 50))
        with state_lock:
            current_ep = int(live_metrics.get("episode", 1))

        if current_ep > target_eps:
            print(f"\n[DASH STEERING] Reached Target Episodes ({target_eps}). Stopping training cleanly.")
            return False

        return True


def start_training_loop(use_gui=False, total_timesteps=500_000):
    """
    Entry point for Thread 1 (The RL Agent & SUMO Engine).
    Automatically detects existing checkpoints and matches action/observation dimensions.
    """
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)

    # Check if a pre-existing checkpoint exists (which uses legacy 4-TLS dimensions)
    checkpoint_files = [
        ("models/indore_ppo_400k.zip", "models/indore_ppo_500k_final"),
        ("models/indore_ppo_300k.zip", "models/indore_ppo_400k"),
        ("models/indore_ppo_200k.zip", "models/indore_ppo_300k"),
        ("models/indore_ppo_100k.zip", "models/indore_ppo_200k"),
    ]
    existing_ckpt = None
    save_name = "models/indore_ppo_100k"

    for ckpt_path, next_save in checkpoint_files:
        if os.path.exists(ckpt_path):
            existing_ckpt = ckpt_path
            save_name = next_save
            break

    # Try loading checkpoint with legacy_4_tls_mode=True first if checkpoint exists
    if existing_ckpt:
        print(f"Found checkpoint {existing_ckpt}. Loading into Digital Twin environment...")
        try:
            train_env = IndoreRoundaboutEnv(use_gui=use_gui, legacy_4_tls_mode=True)
            model = PPO.load(existing_ckpt, env=train_env, verbose=1)
        except ValueError:
            # If the checkpoint was already trained with all 6 TLS IDs
            train_env.close()
            train_env = IndoreRoundaboutEnv(use_gui=use_gui, legacy_4_tls_mode=False)
            model = PPO.load(existing_ckpt, env=train_env, verbose=1)
    else:
        print("No checkpoints found. Starting fresh 6-Intersection PPO training...")
        train_env = IndoreRoundaboutEnv(use_gui=use_gui, legacy_4_tls_mode=False)
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            gamma=0.99,
            ent_coef=0.01,
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=5_000,
        save_path="./models/checkpoints/",
        name_prefix="indore_ppo",
        verbose=1,
    )
    steering_cb = DashSteeringCallback()
    callbacks = CallbackList([checkpoint_cb, steering_cb])

    print(f"Starting Digital Twin RL training loop (Target save: {save_name})...")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            reset_num_timesteps=False,
            progress_bar=False,  # Disabled in background thread to keep console clean
        )
        model.save(save_name)
        print(f"\nPhase complete! Saved model to {save_name}.zip")
    finally:
        train_env.close()


if __name__ == "__main__":
    start_training_loop(use_gui=False, total_timesteps=100_000)
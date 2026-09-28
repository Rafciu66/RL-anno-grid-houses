import os
import glob
import time
import torch
import torch.nn as nn
from pathlib import Path
from sb3_contrib import MaskablePPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from island_env import IslandCityEnv

GRID_HEIGHT = 20
GRID_WIDTH = 20

# 1. Custom Feature Extractor required to load the policy
class CustomGridCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample_input = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


def get_latest_model(models_dir="models"):
    """Finds the most recently created zip file in the models directory."""
    path = Path(models_dir)
    zip_files = list(path.glob("*.zip"))
    if not zip_files:
        # Check without extension if user omitted .zip in save name
        all_files = [f for f in path.glob("*") if f.is_file()]
        if not all_files:
            raise FileNotFoundError(f"No saved model files found in '{models_dir}/'")
        return str(max(all_files, key=lambda f: f.stat().st_mtime))
    
    latest_file = max(zip_files, key=lambda f: f.stat().st_mtime)
    return str(latest_file)


if __name__ == "__main__":
    MODELS_DIR = "models"
    
    # Mode Settings:
    # "press_enter" -> Step manually by hitting Enter
    # "timed"       -> Auto-advance every DELAY seconds
    MODE = "press_enter" 
    DELAY = 0.3          

    # Find and load model
    try:
        model_path = get_latest_model(MODELS_DIR)
        print(f"Loading model from: {model_path}")
    except FileNotFoundError as e:
        print(e)
        exit(1)

    # Initialize Environment
    env = IslandCityEnv(height=GRID_HEIGHT, width=GRID_WIDTH, max_steps=60)

    # Pass custom policy kwargs so SB3 can reconstruct CustomGridCNN
    custom_objects = {
        "policy_kwargs": dict(
            features_extractor_class=CustomGridCNN,
            features_extractor_kwargs=dict(features_dim=128),
            normalize_images=False,
        )
    }

    model = MaskablePPO.load(
        model_path, 
        env=env, 
        custom_objects=custom_objects
    )

    # Run Interactive Debug Episode
    obs, info = env.reset()
    step_count = 0
    terminated = False
    truncated = False
    total_reward = 0

    print("\n--- INITIAL ENVIRONMENT STATE ---")
    env.render()

    while not (terminated or truncated):
        step_count += 1

        # Pause before executing step
        if MODE == "press_enter":
            input(f"\n[Step {step_count}] Press ENTER to step...")
        elif MODE == "timed":
            time.sleep(DELAY)

        # Get masked action prediction
        action_masks = env.action_masks()
        action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)

        # Step Environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Print telemetry and render
        print(f"\n--- Step {step_count} ---")
        print(f"Action: {action} | Step Reward: {reward:.2f} | Total Reward: {total_reward:.2f}")
        env.render()

    print(f"\n==========================================")
    print(f"Episode Completed in {step_count} Steps!")
    print(f"Final Total Reward: {total_reward:.2f}")
    print(f"Final Env Score: {getattr(env, 'total_score', 'N/A')}")
    print(f"==========================================")
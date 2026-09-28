import multiprocessing
import torch
import torch.nn as nn
import gymnasium as gym
import time
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib import MaskablePPO
from island_env import IslandCityEnv
import re
from contextlib import redirect_stdout
from pathlib import Path
GRID_HEIGHT = 20
GRID_WIDTH = 20
class CustomGridCNN(BaseFeaturesExtractor):
    """Custom 2D CNN Feature Extractor tailored for 10x10 grid boards."""
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]  # 5 channels

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


# Simple Gym wrapper that exposes action_masks to SubprocVecEnv
class MaskWrapper(gym.Wrapper):
    def action_masks(self):
        return self.env.action_masks()
def save_model(model):
        output_dir = Path("models")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all existing output(n).txt files and extract their numbers
        existing_numbers = [0]
        pattern = re.compile(r"^model\((\d+)\)\.txt$")

        for file in output_dir.glob("model(*).txt"):
            match = pattern.match(file.name)
            if match:
                existing_numbers.append(int(match.group(1)))

        # Determine the next file number
        next_number = max(existing_numbers) + 1
        next_filename = output_dir / f"model({next_number}).txt"

        # Save the model
        model.save(next_filename)

def print_to_txt(env, time_steps, time_elapsed, total_reward):
        output_dir = Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all existing output(n).txt files and extract their numbers
        existing_numbers = [0]
        pattern = re.compile(r"^output\((\d+)\)\.txt$")

        for file in output_dir.glob("output(*).txt"):
            match = pattern.match(file.name)
            if match:
                existing_numbers.append(int(match.group(1)))

        # Determine the next file number
        next_number = max(existing_numbers) + 1
        next_filename = output_dir / f"output({next_number}).txt"

        # Save the render output
        with open(next_filename, "w", encoding="utf-8") as f:
            with redirect_stdout(f):
                print(f"Total Time Steps: {time_steps}, Time Elapsed: {time_elapsed}, Total Reward: {total_reward}")
                env.render()

def make_env(height=10, width=10, max_steps=60):
    """Factory function to build worker environments."""
    def _init():
        env = IslandCityEnv(height=height, width=width, max_steps=max_steps)
        return MaskWrapper(env)
    return _init


if __name__ == "__main__":
    TIME_STEPS = 100_000
    start_time = time.perf_counter()
    # 1. Spawn parallel CPU environments
    num_cpu = max(1, multiprocessing.cpu_count() - 2)
    print(f"Launching {num_cpu} parallel CPU environments...")
    
    env = SubprocVecEnv([make_env(height=GRID_HEIGHT, width=GRID_WIDTH, max_steps=60) for _ in range(num_cpu)])

    # 2. Configure Custom CNN Policy kwargs
    policy_kwargs = dict(
        features_extractor_class=CustomGridCNN,
        features_extractor_kwargs=dict(features_dim=128),
        normalize_images=False,
    )

    # 3. Instantiate MaskablePPO
    model = MaskablePPO(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=3e-4,
        gamma=0.999,
        ent_coef=0.02,   # Encourages road exploration
        batch_size=256,  # GPU minibatch size
        n_steps=256,     # Worker rollout length
    )

    # 4. Train Agent
    print("Training AI model...")
    model.learn(total_timesteps=TIME_STEPS)

    # --- SAVE THE MODEL ---
    save_model(model)

    # 5. Evaluate Trained Agent on a Single Environment
    eval_env = IslandCityEnv(height=GRID_HEIGHT, width=GRID_WIDTH, max_steps=60)
    obs, info = eval_env.reset()
    terminated = False
    total_reward = 0
    
    print("\n--- Running Evaluation Episode ---")
    while not terminated:
        action_masks = eval_env.action_masks()
        action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += reward
    end_time = time.perf_counter()
    print_to_txt(eval_env, TIME_STEPS, end_time - start_time, total_reward)
    
    print(f"\nFinal Episode Score: {eval_env.total_score:.2f} points (Total Reward: {total_reward:.2f})")
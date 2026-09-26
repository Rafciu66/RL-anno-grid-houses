import multiprocessing
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib import MaskablePPO
from island_env import IslandCityEnv


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


def make_env(height=10, width=10, max_steps=60):
    """Factory function to build worker environments."""
    def _init():
        env = IslandCityEnv(height=height, width=width, max_steps=max_steps)
        return MaskWrapper(env)
    return _init


if __name__ == "__main__":
    # 1. Spawn parallel CPU environments
    num_cpu = max(1, multiprocessing.cpu_count() - 2)
    print(f"Launching {num_cpu} parallel CPU environments...")
    
    env = SubprocVecEnv([make_env(height=10, width=10, max_steps=60) for _ in range(num_cpu)])

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
        gamma=0.99,
        ent_coef=0.01,   # Encourages road exploration
        batch_size=256,  # GPU minibatch size
        n_steps=256,     # Worker rollout length
    )

    # 4. Train Agent
    print("Training AI model...")
    model.learn(total_timesteps=200_000)

    # 5. Evaluate Trained Agent on a Single Environment
    eval_env = IslandCityEnv(height=10, width=10, max_steps=60)
    obs, info = eval_env.reset()
    terminated = False
    total_reward = 0

    print("\n--- Running Evaluation Episode ---")
    while not terminated:
        action_masks = eval_env.action_masks()
        action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += reward

    eval_env.render()
    print(f"\nFinal Episode Score: {eval_env.total_score:.2f} points (Total Reward: {total_reward:.2f})")
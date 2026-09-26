import time
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from island_env import IslandCityEnv


def render_board(env, step_num=None):
    """Renders a clean ASCII view of the island board."""
    # Symbols: Empty land = '.', Road = 'R', House = 'H', Market = 'M', Water = '~'
    symbols = {0: ". ", 1: "R ", 2: "H ", 3: "M "}

    header = f" STEP {step_num} | Score: {env.total_score:.2f} " if step_num else f" FINAL CITY | Score: {env.total_score:.2f} "
    border_len = max(env.W * 3 + 2, len(header) + 4)

    print("\n" + "=" * border_len)
    print(f" {header}".center(border_len))
    print("=" * border_len)

    for y in range(env.H):
        row_str = "| "
        for x in range(env.W):
            if env.land_mask[y, x] == 0:
                row_str += "~ "
            else:
                btype = env.building_grid[y, x]
                row_str += symbols.get(btype, ". ")
        row_str += " |"
        print(row_str)

    print("=" * border_len + "\n")


def run_visualization(animate=False):
    # 1. Initialize environment and load saved model weights
    env = IslandCityEnv(height=10, width=10, max_steps=60)
    
    try:
        model = MaskablePPO.load("island_ppo_model", env=env)
        print("Loaded 'island_ppo_model.zip' successfully!")
    except FileNotFoundError:
        print("Error: Could not find 'island_ppo_model.zip'. Did you save the model after training?")
        return

    obs, info = env.reset()
    terminated = False
    step_count = 0

    print("Running evaluation episode...")

    while not terminated:
        step_count += 1
        action_masks = get_action_masks(env)
        
        # Pick best deterministic action from trained model
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if animate:
            render_board(env, step_num=step_count)
            time.sleep(0.1)  # Brief delay to watch step-by-step growth

    # Render final output layout
    if not animate:
        render_board(env)


if __name__ == "__main__":
    # Set animate=True if you want to watch the AI build step-by-step
    run_visualization(animate=False)
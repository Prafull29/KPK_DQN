import torch

from kpk_env import KPKEnv
from agent import DQNAgent


def test():

    env = KPKEnv(
        tablebase_path="/kaggle/input/datasets/prafull29/kpk-dqn-project"
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    agent = DQNAgent(
        state_size=4,
        action_size=10,
        device=device
    )

    agent.load(
        "/kaggle/working/checkpoints/final_model.pth"
    )

    episodes = 100
    max_steps = 100

    wins = 0
    draws = 0

    for episode in range(episodes):

        state, info = env.reset()

        if env.done:
            draws += 1
            continue

        # DQN must play White

        if env.turn != env.WHITE:
            draws += 1
            continue

        done = False

        for step in range(max_steps):

            if done:
                break

            mask = env.get_action_mask()

            action = agent.select_action(
                state,
                mask,
                training=False
            )

            state, reward, done, truncated, info = env.step(
                action
            )

            if truncated:
                break

        # Game did not finish within max_steps
        # Treat it as a draw.

        if not done:

            draws += 1

        else:

            result = info.get("winner")

            if result == "WHITE_WIN":
                wins += 1

            else:
                draws += 1

        if (episode + 1) % 10 == 0:

            print(
                f"Episode {episode + 1} | "
                f"Wins: {wins} | "
                f"Draws: {draws}"
            )

    print("\nResults")
    print("-------")
    print("Games:", episodes)
    print("White wins:", wins)
    print("Draws:", draws)
    print("Win rate:", wins / episodes * 100, "%")

    env.close()


if __name__ == "__main__":
    test()
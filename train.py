import torch

from kpk_env import KPKEnv
from agent import DQNAgent


def train():

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

    episodes = 5000
    max_steps = 100

    for episode in range(episodes):

        state, info = env.reset()

        if env.done:
            continue

        # Make sure White is the player
        # before asking the DQN for an action.

        if env.turn != env.WHITE:
            continue

        total_reward = 0

        for step in range(max_steps):

            mask = env.get_action_mask()

            action = agent.select_action(
                state,
                mask,
                training=True
            )

            next_state, reward, done, truncated, info = env.step(
                action
            )

            if done or truncated:

                next_mask = [0] * 10

            else:

                next_mask = env.get_action_mask()

            agent.store_transition(
                state,
                action,
                reward,
                next_state,
                done or truncated,
                next_mask
            )

            agent.train_step()

            state = next_state
            total_reward += reward

            if done or truncated:
                break

        agent.episodes += 1

        if (episode + 1) % 10 == 0:

            print(
                f"Episode {episode + 1} | "
                f"Reward: {total_reward:.2f} | "
                f"Epsilon: {agent.epsilon:.4f}"
            )

        if (episode + 1) % 100 == 0:

            agent.save(
                f"/kaggle/working/checkpoints/"
                f"model_{episode + 1}.pth"
            )

    agent.save(
        "/kaggle/working/checkpoints/final_model.pth"
    )

    env.close()


if __name__ == "__main__":
    train()
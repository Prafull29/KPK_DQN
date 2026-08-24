# KPK_DQN

A Deep Q-Network (DQN) agent for learning the **King-Pawn vs. King (KPK) chess endgame**, where the agent controls **White** and plays against an **optimal Black opponent** using the Syzygy tablebase.

## ♟️ Project Overview

The **King-Pawn vs. King** endgame consists of:

- White King
- White Pawn
- Black King

The objective of the White agent is to learn how to play the endgame effectively and achieve the best possible outcome.

In this implementation:

- **White** is controlled by the DQN agent.
- **Black** acts as an optimal opponent.
- Black's moves are selected using the **Syzygy tablebase**.
- The tablebase provides the theoretically optimal outcome for a given endgame position.
- The DQN learns through repeated interaction with the environment.

This creates an environment where the agent can learn against a strong, deterministic reference opponent.

## 🏗️ Project Structure

```text
KPK_DQN/
│
├── KPKvK.rtbw           # Syzygy tablebase file
├── KPKvK.rtbz           # Syzygy tablebase file
│
├── agent.py             # DQN agent and action-selection logic
├── config.py            # Training and environment configuration
├── dqn.py               # Deep Q-Network implementation
├── kpk_env.py           # King-Pawn vs. King environment
├── replay_buffer.py     # Experience replay buffer
│
├── train.py             # Training script
├── test.py              # Testing/evaluation script
│
└── README.md            # Project documentation
```

## ⚙️ Requirements

The project requires Python and the libraries used by the implementation.

Create a virtual environment if desired:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Training

After installing the dependencies and ensuring that the Syzygy tablebase files are available, run:

```bash
python train.py
```

Training parameters can be modified in:

```text
config.py
```

## 🧪 Testing

After training, evaluate the trained model using:

```bash
python test.py
```

The agent plays as White while Black continues to use the Syzygy tablebase for optimal play.

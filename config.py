# SIMPLE DQN CONFIGURATION FOR KPK

CONFIG = {

    # KPK

    "state_size": 4,
    "action_size": 10,

    # DQN

    "learning_rate": 1e-3,

    # Discount factor

    "gamma": 0.99,

    # Epsilon-greedy

    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.9995,

    # Replay Buffer

    "buffer_size": 50000,
    "batch_size": 64,

    # Target Network

    "target_update_freq": 1000,

    # DQN Network

    "hidden_sizes": (128, 128)
}
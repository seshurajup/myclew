# 19th Place Gold Writeup

Thanks to Kaggle for organizing a great competition! It was fun to participate 🙂

# Summary

- fast GPU simulation;
- careful game-state features and a compact model architecture;
- behavior cloning for easier experimentation and faster convergence;
- RL/PPO self-play training and fine-tuning;
- separate 2p and 4p pipelines from behavior cloning through RL/PPO and calibration.

# Fast GPU Simulator Rewrite

The official single-game environment was orders of magnitude too slow for RL. With comparable random fleet launches, the reference CPU environment ran at roughly **150-300 transitions / second**, while the batched GPU simulator reached roughly **300,000-700,000 transitions / second per GPU**.

# Architecture And Features

The model had to stay small enough for CPU inference, with roughly a 1 second per-turn budget in the tournament runtime. The representation also needed to be relative rather than tied to absolute board positions, and it needed to include both the current state and useful future trajectory information.

The model viewed the game as a set of planets, all possible directed launches between planets, and a grid of candidate fleet sizes for each launch. Inputs were organized into these feature groups:

- **Global features:** number of live opponents, current turn, current rank by ships, total fleet ships, total garrison ships, total production, non-comet production, number of non-comet planets, whether comets are present, turns until the next comet spawn, and per-player shares of garrison, fleet ships, production, and planet count;
- **Planet features:** owner, signed garrison size, production, comet time-to-live, and comet flag;
- **Future planet features:** over 32 future time buckets, incoming fleet ships by player, target balance, and reachable garrison by player;
- **Source-target features:** over the same future time buckets, time, relative distance from source to future target position, and sun clearance;
- **Fleet-size features:** over 64 candidate fleet sizes per source-target pair, candidate ship count, arrival time, estimated defenders at arrival, commitment fraction, and post-combat margin.

The model predicted four things:

- **launch count:** how many fleets to send this turn;
- **source-target logits:** which directed planet pairs to use;
- **fleet-size logits:** how many ships to send for each chosen launch;
- **value estimate:** expected position strength for PPO training.

Those predictions were converted into legal environment actions by selecting the launch count, choosing source-target pairs and fleet sizes, converting each target into an angle by aiming at the predicted future target position, and dropping invalid or over-budget launches.

|  ![Model architecture](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F2631834%2F0473f84a0c807214f8f12dfd160c2fcf%2FArchitecture.png?generation=1783619584982458&alt=media) | ![Block architecture](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F2631834%2F342275919ee30445575a710eac2770d6%2FArchitecture2.png?generation=1783619598556729&alt=media) |
| --- | --- |

# Behavior Cloning

Behavior cloning made experimentation much faster by giving RL a strong starting point. It also made architecture iteration easier: pure behavior-cloning performance was meaningful enough to compare model variants before spending compute on RL fine-tuning or training RL from scratch.

Training data came from two sources: historical Kaggle replays from competitors, and simulated games from internal models. This was useful in two ways:

- strong games provided high-quality demonstrations;
- previous learned agents provided a way to ramp model size and transfer behavior into newer or larger architectures.

The target reconstruction was important: actions were paired with the state before the move, launched fleets were tracked to infer the intended target, and the model learned launch count, target choice, and fleet size.

# RL / PPO Fine-Tuning

PPO was mostly used after behavior cloning. Training from scratch could also reach quite good results, but it took significantly longer, so behavior cloning was the practical starting point for most runs. A few details made it work well:

- initialize from a behavior-cloned policy and warm up the value head before policy updates;
- train with self-play using multiple past versions as opponents;
- use clipped PPO updates, KL checks, entropy regularization, and learning-rate schedules;
- evaluate checkpoints with direct head-to-head games instead of relying only on training loss.

# Final Deployment

The 2-player and 4-player agents were trained as separate pipelines from start to finish: behavior cloning, RL/PPO fine-tuning, and final calibration were all done separately for each format.

Two kinds of biases were used for deployment calibration. After applying the biases, the action was selected greedily by taking the highest-logit option.

1. **Launch-count bias** shifted how many fleets the agent wanted to launch.
2. **Fleet-size bias** shifted the chosen number of ships per launch.

Calibration values were searched with head-to-head games. The goal was not to change the strategy entirely, but to correct systematic deployment tendencies such as under-sending or consistently choosing fleet sizes that were too large or too small.

Separate 2-player and 4-player agents mattered because 4-player games were strategically different and much harder on the CPU runtime
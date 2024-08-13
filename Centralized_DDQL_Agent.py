from itertools import chain

from classes.SFC import SFC
import random
from classes.DDQL_ENV import DdqlEnv
import gymnasium as gym
import torch
import numpy as np
from torch import nn
import random
import torch.nn.functional as F
import collections
from torch.optim.lr_scheduler import StepLR
from classes.Memory import Memory
from classes.Models import QNetwork
from create_sfc_dataset import create_dataset
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_state(state):
    # print('state to parse ' + str(state))
    # Flatten the dictionary values into a single list
    flattened_values = list(chain.from_iterable(
        value.flatten() if isinstance(value, np.ndarray) else (value if isinstance(value, list) else [value])
        for value in state.values()
    ))

    # print(flattened_values)
    state_array = np.array(flattened_values)

    return state_array


def select_action(model, env, state, eps):
    state = parse_state(state)
    state = torch.Tensor(state).to(device)
    with torch.no_grad():
        values = model(state)

    # select a random action wih probability eps
    if random.random() <= eps:
        action = np.random.randint(0, env.action_space.n)
        print('random action ' + str(action))
    else:
        action = np.argmax(values.cpu().numpy())
        print('action selected ' + str(action))

    return action


def train(batch_size, current, target, optim, memory, gamma):
    states, actions, next_states, rewards, is_done = memory.sample(batch_size)

    q_values = current(states)

    next_q_values = current(next_states)
    next_q_state_values = target(next_states)

    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
    expected_q_value = rewards + gamma * next_q_value * (1 - is_done)

    loss = (q_value - expected_q_value.detach()).pow(2).mean()

    optim.zero_grad()
    loss.backward()
    optim.step()


def evaluate(Qmodel, env, repeats):
    """
    Runs a greedy policy with respect to the current Q-Network for "repeats" many episodes. Returns the average
    episode reward.
    """
    Qmodel.eval()
    perform = 0
    for _ in range(repeats):
        state = env.reset()
        done = False
        while not done:
            state = parse_state(state)
            state = torch.Tensor(state).to(device)
            with torch.no_grad():
                values = Qmodel(state)
            action = np.argmax(values.cpu().numpy())
            state, reward, done, _ = env.step(action)
            perform += reward
    Qmodel.train()
    return perform / repeats


def update_parameters(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())


def main(gamma=0.99, lr=0.000025, min_episodes=20, eps=1, eps_decay=0.9998, eps_min=0.001, update_step=20,
         batch_size=64, update_repeats=50,
         num_episodes=10000, seed=42, max_memory_size=10000, lr_gamma=1, lr_step=100, measure_step=100,
         measure_repeats=100, hidden_dim=64, horizon=np.inf, render=True, render_step=50):
    """
    Remark: Convergence is slow. Wait until around episode 2500 to see good performance.

    :param gamma: reward discount factor
    :param lr: learning rate for the Q-Network
    :param min_episodes: we wait "min_episodes" many episodes in order to aggregate enough data before starting to train
    :param eps: probability to take a random action during training
    :param eps_decay: after every episode "eps" is multiplied by "eps_decay" to reduces exploration over time
    :param eps_min: minimal value of "eps"
    :param update_step: after "update_step" many episodes the Q-Network is trained "update_repeats" many times with a
    batch of size "batch_size" from the memory.
    :param batch_size: see above
    :param update_repeats: see above
    :param num_episodes: the number of episodes played in total
    :param seed: random seed for reproducibility
    :param max_memory_size: size of the replay memory
    :param lr_gamma: learning rate decay for the Q-Network
    :param lr_step: every "lr_step" episodes we decay the learning rate
    :param measure_step: every "measure_step" episode the performance is measured
    :param measure_repeats: the amount of episodes played in to asses performance
    :param hidden_dim: hidden dimensions for the Q_network
    :param env_name: name of the gym environment
    :param cnn: set to "True" when using environments with image observations like "Pong-v0"
    :param horizon: number of steps taken in the environment before terminating the episode (prevents very long episodes)
    :param render: if "True" renders the environment every "render_step" episodes
    :param render_step: see above
    :return: the trained Q-Network and the measured performances
    """
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)

    sfc_dataset = create_dataset(num_episodes)
    env = DdqlEnv(5, 5, 2, sfc_dataset)

    Q_1 = QNetwork(V_s=env.pop_n, E_s=env.edge_n).to(device)
    Q_2 = QNetwork(V_s=env.pop_n, E_s=env.edge_n).to(device)
    # transfer parameters from Q_1 to Q_2
    update_parameters(Q_1, Q_2)

    # we only train Q_1
    for param in Q_2.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(Q_1.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

    memory = Memory(max_memory_size)
    performance = []
    i = 0
    scores = []
    optimal_count = 0
    optimal_percent = []
    for episode in range(num_episodes):
        # display the performance
        # if (episode % measure_step == 0) and episode >= min_episodes:
        #     performance.append([episode, evaluate(Q_1, env, measure_repeats)])
        #     print("Episode: ", episode)
        #     print("rewards: ", performance[-1][1])
        #     print("lr: ", scheduler.get_last_lr()[0])
        #     print("eps: ", eps)

        state = env.reset()
        p_state = parse_state(state)
        memory.state.append(p_state)

        done = False

        score = 0
        while not done:
            i += 1
            action = select_action(Q_2, env, state, eps)
            state, reward, done, is_optimal, _ = env.step(action)
            score += reward

            # save state, action, reward sequence
            p_state = parse_state(state)
            memory.update(p_state, action, reward, done)

            # if episode >= min_episodes and episode % update_step == 0:
            #     for _ in range(update_repeats):
            if i > 64:
                train(batch_size, Q_1, Q_2, optimizer, memory, gamma)

                # transfer new parameter from Q_1 to Q_2
                if episode % update_step == 0:
                    update_parameters(Q_1, Q_2)

                # update learning rate and eps
                # scheduler.step()
                eps = max(eps * eps_decay, eps_min)
        print('episode ' + str(episode) + ' curr sfc ' + str(env.curr_sfc_no) + 'epsilon ' + str(eps) + ' score ' + str(
            score))
        if is_optimal:
            optimal_count += 1
        optimal_percent.append(optimal_count / (episode + 1))
        scores.append(score)
        # average_scores = 0
    mov_avg = np.convolve(scores, np.ones(100) / 100, mode='valid')

    with open('central_scores.txt', 'w') as fw:
        fw.write(str(scores))
    with open('optimal_percent_central.txt', 'w') as fw:
        fw.write(str(optimal_percent))

    x_sma = np.arange(100 - 1, len(scores))
    # x_opt = np.arange(100, len(optimal_percent))

    plt.plot(x_sma, mov_avg)
    # plt.plot(x_opt, optimal_percent)
    plt.show()

    return Q_1, performance


if __name__ == '__main__':
    main()

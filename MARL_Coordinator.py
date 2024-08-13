from itertools import chain

from classes.SFC import SFC
import random
from classes.DMARL_ENV import DmarlEnv
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


def main(gamma=0.99, lr=0.000025, min_episodes=20, eps=1, eps_decay=0.9998, eps_min=0.0001, update_step=20,
         batch_size=64, update_repeats=50,
         num_episodes=10000, seed=42, max_memory_size=10000, lr_gamma=1, lr_step=100, measure_step=100,
         measure_repeats=100, hidden_dim=64, horizon=np.inf, render=True, render_step=50, n_actions=3):
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
    env = DmarlEnv(5, 10, 5, 3, sfc_dataset)

    all_pops = env.pop_topology.pops
    marl_agents = []
    for pop in all_pops:
        pop_agent = pop.initialize_agent(env.pop_topology, n_actions, pop.get_id(), max_memory_size)
        marl_agents.append(pop_agent)

    # Q_1 = QNetwork(V_s=env.pop_n, E_s=env.edge_n).to(device)
    # Q_2 = QNetwork(V_s=env.pop_n, E_s=env.edge_n).to(device)
    # # transfer parameters from Q_1 to Q_2
    # update_parameters(Q_1, Q_2)

    for agent in marl_agents:
        agent.initialize_q_network(lr)

    # # we only train Q_1
    # for param in Q_2.parameters():
    #     param.requires_grad = False
    #
    # optimizer = torch.optim.Adam(Q_1.parameters(), lr=lr)
    # scheduler = StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

    memory = Memory(max_memory_size)
    performance = []
    i = 0
    scores = []
    optimal_count = 0
    optimal_percent = []
    for episode in range(num_episodes):

        state = env.reset()
        for agent in marl_agents:
            agent.add_state(state, eps)

        done = False

        score = 0
        while not done:
            i += 1
            soft_actions = []
            for agent in marl_agents:
                s_action = agent.select_action(env, state, eps)
                soft_actions.append(s_action)

            firm_action = soft_actions.index(max(soft_actions))

            state, reward, done, is_optimal,  _ = env.step(firm_action)
            score += reward

            s_i = 0
            for agent in marl_agents:
                agent.update_memory(state, eps, soft_actions[s_i], reward, done)
                s_i += 1

            if i > 64:
                for agent in marl_agents:
                    agent.train(batch_size, gamma)

                if episode % update_step == 0:
                    for agent in marl_agents:
                        agent.update_parameters()

                # scheduler.step()
                eps = max(eps * eps_decay, eps_min)
        print('episode ' + str(episode) + ' curr sfc ' + str(env.curr_sfc_no) + 'epsilon ' + str(eps) + ' score ' + str(
            score))
        if is_optimal:
            optimal_count += 1
        optimal_percent.append(optimal_count / (episode + 1))
        scores.append(score)
        # average_scores = 0

    x_sma = np.arange(100 - 1, len(scores))
    mov_avg = np.convolve(scores, np.ones(100) / 100, mode='valid')

    with open('marl_scores.txt', 'w') as fw:
        fw.write(str(scores))
    with open('optimal_percent_marl.txt', 'w') as fw:
        fw.write(str(optimal_percent))

    plt.plot(x_sma, mov_avg)
    plt.show()



if __name__ == '__main__':
    main()

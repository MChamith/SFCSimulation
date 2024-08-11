from itertools import chain

from classes.SFC import SFC
import random
from DDQL_ENV import DdqlEnv
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


class Agent:
    def __init__(self, topology, n_actions):
        self.topology = topology
        self.n_actions = n_actions

    def get_neighbour_data(self):
        pass

    def get_flag_data(self):
        pass

    def parse_state(self, state):
        # print('state to parse ' + str(state))
        # Flatten the dictionary values into a single list
        flattened_values = list(chain.from_iterable(
            value.flatten() if isinstance(value, np.ndarray) else (value if isinstance(value, list) else [value])
            for value in state.values()
        ))

        # print(flattened_values)
        state_array = np.array(flattened_values)

        return state_array

    def select_action(self, model, env, state, eps):
        state = self.parse_state(state)
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

    def train(self, batch_size, current, target, optim, memory, gamma):
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

    def evaluate(self, Qmodel, env, repeats):
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
                state = self.parse_state(state)
                state = torch.Tensor(state).to(device)
                with torch.no_grad():
                    values = Qmodel(state)
                action = np.argmax(values.cpu().numpy())
                state, reward, done, _ = env.step(action)
                perform += reward
        Qmodel.train()
        return perform / repeats

    def update_parameters(self, current_model, target_model):
        target_model.load_state_dict(current_model.state_dict())

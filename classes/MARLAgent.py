from itertools import chain
import torch
import numpy as np
import random
from classes.Models import MARL_QNetwork
from classes.Memory import Memory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    def __init__(self, topology, n_actions, pop_id, max_memory_size):
        self.topology = topology
        self.n_actions = n_actions
        self.pop_id = pop_id
        self.Q1 = None
        self.Q2 = None
        self.memory  = Memory(max_memory_size)
        self.optimizer = None

    # might be able optimize both get functions by initially getting all the neighbours and storing the neighbours.
    # Now each time neighbours are calculated
    def initialize_q_network(self, lr):
        # input size of q network is determined by number of neighbours  (check observation space in paper)
        num_server = self.topology.get_number_of_servers(self.pop_id)
        num_neigbours = self.topology.get_number_of_neighbours(self.pop_id)
        input_size = 14 + num_server + 4 * num_neigbours
        self.Q1 = MARL_QNetwork(input_size, self.n_actions)
        self.Q2 = MARL_QNetwork(input_size, self.n_actions)
        self.update_parameters()
        for param in self.Q2.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.Adam(self.Q1.parameters(), lr=lr)

    def get_neighbour_state(self):
        nbr_locs, nbr_resources = self.topology.get_neighbours_state(self.pop_id)
        flattened_locs = [item for sublist in nbr_locs for item in sublist]
        nbr_data = [flattened_locs, nbr_resources]
        return nbr_data

    def get_local_state(self):
        server_cpus = self.topology.get_server_cpu(self.pop_id)
        return server_cpus

    def get_neighbour_bandwidths(self):
        return self.topology.get_neighbours_edges(self.pop_id)

    def get_flag_data(self, vnf_order, src, dst):
        return self.topology.get_flag_data(self.pop_id, vnf_order, src, dst)

    def parse_state(self, state, eps):
        # print('state to parse ' + str(state))
        # Flatten the dictionary values into a single list
        flattened_state = list(chain.from_iterable(
            value.flatten() if isinstance(value, np.ndarray) else (value if isinstance(value, list) else [value])
            for value in state.values()
        ))

        src = state['src_loc']
        dst = state['dst_loc']
        # TODO check if below is correct
        vnf_order = state['order'] - 1
        local_cpu = self.get_local_state()
        local_loc = self.topology.get_pop_coordinates(self.pop_id)
        n_edges = self.get_neighbour_bandwidths()
        nbr_data = self.get_neighbour_state()
        flag_data = self.get_flag_data(vnf_order, src, dst)
        # print(flattened_values)
        state_array = np.array([flattened_state, nbr_data, flag_data, eps])

        return state_array

    def select_action(self, env, state, eps):
        state = self.parse_state(state, eps)
        state = torch.Tensor(state).to(device)
        with torch.no_grad():
            values = self.Q2(state)

        # select a random action wih probability eps
        if random.random() <= eps:
            action = np.random.randint(0, env.action_space.n)
            print('random action ' + str(action))
        else:
            action = np.argmax(values.cpu().numpy())
            print('action selected ' + str(action))

        return action

    def train(self, batch_size, gamma):
        states, actions, next_states, rewards, is_done = self.memory.sample(batch_size)

        q_values = self.Q1(states)

        next_q_values = self.Q1(next_states)
        next_q_state_values = self.Q2(next_states)

        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = rewards + gamma * next_q_value * (1 - is_done)

        loss = (q_value - expected_q_value.detach()).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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

    def update_parameters(self):
        self.Q2.load_state_dict(self.Q1.state_dict())

    def add_state(self, state, eps):
        p_state = self.parse_state(state, eps)
        self.memory.state.append(p_state)

    def update_memory(self, state, eps, action, reward, done):
        p_state = self.parse_state(state, eps)
        self.memory.update(p_state, action, reward, done)

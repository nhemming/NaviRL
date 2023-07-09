"""
A DQN Learning algorithm implementation
"""

# native modules

# 3rd part modules
import copy

import numpy as np
import torch
from nn_builder.pytorch.NN import NN

# own modules
from agents.BaseAgent import BaseAgent


class DQN(BaseAgent):

    def __init__(self,action_operation,controlled_entity, device, exploration_strategy, name, network_description, observation_information):
        super(DQN,self).__init__(action_operation,controlled_entity,device,exploration_strategy, name, observation_information)

        layers_info = [int(i) for i in network_description['hidden_layers'].split(',')]

        layers_info.append(len(action_operation.action_bounds))

        self.q_network = NN(input_dim=len(observation_information), layers_info=layers_info, output_activation=None,
                 hidden_activations=network_description['hidden_activations'], dropout=network_description['dropout'],
                            initialiser=network_description['initializer'], batch_norm=network_description['batch_norm'],
                 columns_of_data_to_be_embedded=[], embedding_dimensions=[], y_range=tuple([float(i) for i in network_description['output_range'].split(',')]), random_seed=network_description['seed'])
        self.target_network = copy.deepcopy(self.q_network)

    def create_state_action(self, entities, ep_num, sensors, sim_time, use_exploration):

        # determine if new action shall be determined. If so set is_new_action to true

        # collect state information and normalize state information
        norm_state, state = self.normalize_state(entities,sensors)
        self.state_info = {'norm_state':norm_state, 'state':state}

        # produce action
        raw_action, q_values = self.forward(state)

        # exploration perturbations
        mutated_action = None
        if use_exploration:
            mutated_action = self.exploration_strategy.add_perturbation(raw_action, ep_num)

        # un-normalize action? I think handle this in the action operation at a later stage

        # partial log own data (input, raw output, perturbed output)
        self.action_info = {'raw_action':raw_action, 'mutated_action': mutated_action, 'q_values': q_values}

    def train(self):
        pass

    def update_memory(self):

        # if is_new_action == True save data, else don't save the data

        # set is_new_action to false.
        pass

    def forward(self,state):
        with torch.no_grad():
            output = self.q_network.forward(torch.reshape(torch.from_numpy(state).type(torch.float),(1,len(state))))

            output = output.numpy()
            output = np.reshape(output, (len(output[0]),))

            return np.argmax(output), output


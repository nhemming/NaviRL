"""
A DQN Learning algorithm implementation
"""

# native modules
import copy

# 3rd part modules
import numpy as np
import torch
import torch.nn.functional as F

# own modules
from agents.BaseLearningAlgorithm import BaseLearningAlgorithm
from environment.NetworkBuilder import Network


class DQN(BaseLearningAlgorithm):

    def __init__(self,device, exploration_strategy, general_params, name, network_description, observation_information, optimizer_dict, output_dim, replay_buffer, seed):

        # device, exploration_strategy, general_params, name, observation_information
        super(DQN,self).__init__(device,exploration_strategy, general_params, name, observation_information)

        # create the q network and the target network
        self.q_network = Network(device,observation_information,network_description, output_dim)
        self.target_network = copy.deepcopy(self.q_network)

        # create replay buffer
        self.replay_buffer = replay_buffer

        # setup optimizer
        self.load_optimizer(self.name, optimizer_dict, self.q_network.parameters())

        # save extra general parameters
        self.target_update_rate = general_params['target_update_rate']

        self.last_target_update = -np.infty

    def create_state_action(self, action_operation, entities, ep_num, sensors, sim_time, use_exploration):

        # determine if new action shall be determined. If so set is_new_action to true
        if sim_time - self.last_reset_time >= action_operation.frequency:

            self.last_reset_time = sim_time

            # collect state information and normalize state information
            norm_state, state = self.normalize_state(entities,sensors)
            self.state_info = {'norm_state':norm_state, 'state':state}

            # produce action
            q_values = self.forward(norm_state)
            if self.device == 'cuda':
                q_values = q_values.cpu()
            q_values = q_values.numpy()
            raw_action = np.argmax(q_values)

            # exploration perturbations
            mutated_action = None
            if use_exploration:
                mutated_action = self.exploration_strategy.add_perturbation(raw_action, ep_num)

            # un-normalize action? I think handle this in the action operation at a later stage

            # partial log own data (input, raw output, perturbed output)
            self.action_info = {'raw_action':raw_action, 'mutated_action': mutated_action, 'q_values': q_values}

    def train(self, ep_num):

        if len(self.replay_buffer) >= self.batch_size:
            # train the agent
            for _ in range(self.num_batches):
                # TODO look at own code maybe? and compare?
                states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

                # TODO check if target needs gradients on it. I think not
                q_target_next = self.forward_target(next_states)
                q_target = rewards + (self.gamma * q_target_next * (1-dones))

                q_expected = self.forward_with_grad(states)#.gather(1,actions.long())

                loss = F.mse_loss(q_expected, q_target)

                # take optimization step
                self.optimizer[self.name].zero_grad() # reset gradients
                loss.backward()
                self.optimizer[self.name].step()

            # check and update target network if needed
            if self.last_target_update - ep_num > self.target_update_rate:
                self.last_target_update = ep_num
                # TODO copy weights of q network to target network

    def update_memory(self, action_operation, done, entities, reward, sensors, sim_time):

        # if is_new_action == True save data, else don't save the data
        if sim_time - self.last_reset_time >= action_operation.frequency:
            self.state_info['norm_next_state'], self.state_info['next_state'] = self.normalize_state( entities, sensors)

            self.replay_buffer.add_experience(self.state_info['norm_state'], self.action_info['mutated_action'], reward, self.state_info['norm_next_state'], done)

            # save tuple to history
            self.add_sample_history(reward, done)

    def forward(self,state):
        with torch.no_grad():
            # the with grad function is called but the gradients are no saved
            return self.forward_with_grad(state)

    def forward_target(self,state):
        with torch.no_grad():
            # the with grad function is called but the gradients are no saved
            return self.forward_with_grad_target(state)

    def forward_with_grad(self,state):

        """
        #output = self.q_network.forward(torch.reshape(torch.from_numpy(state).type(torch.float), (1, len(state))))
        output = self.q_network.forward(state)

        if self.device == 'cuda':
            output = output.cpu()
        output = output.numpy()
        #output = np.reshape(output, (len(output),))

        return np.argmax(output), output
        """
        return self.q_network.forward(state)

    def forward_with_grad_target(self,state):
        """
        output = self.target_network.forward(state)
        if self.device == 'cuda':
            output = output.cpu()
        output = output.numpy()

        return np.argmax(output), output
        """
        return self.target_network.forward(state)




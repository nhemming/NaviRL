"""
A DQN Learning algorithm implementation
"""

# native modules
import copy
import os

# 3rd part modules
import numpy as np
import torch
import torch.nn.functional as F

# own modules
from agents.BaseLearningAlgorithm import BaseLearningAlgorithm


# TODO delete this network after testing is complete
#from environment.NetworkBuilder import Network
import torch
import torch.nn as nn


class Network(nn.Module):

    def __init__(self, device):
        nn.Module.__init__(self)

        # create layers
        inter_layer = 64
        self.fc1 = nn.Linear(3, inter_layer, device=device)
        self.fc2 = nn.Linear(inter_layer, inter_layer, device=device)
        self.out = nn.Linear(inter_layer, 3, device=device)
        self.active = torch.nn.ReLU()

        self.device = device

    def forward(self, input_data):
        x = input_data['head0']
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)

        x = self.fc1(x)
        x = self.active(x)
        x = self.fc2(x)
        x = self.active(x)
        x = self.out(x)
        #x = self.out_active(x)  # should be one for discrete outputs or untransformed outputs

        return x



class DQN(BaseLearningAlgorithm):

    def __init__(self,device, exploration_strategy, general_params, name, network_description, observation_information, optimizer_dict, output_dim, replay_buffer, seed):

        # device, exploration_strategy, general_params, name, observation_information
        super(DQN,self).__init__(device,exploration_strategy, general_params, name, observation_information)

        # create the q network and the target network
        self.q_network = Network(device)
        self.target_network = Network(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        '''
        self.q_network = Network(device,observation_information,network_description, output_dim)
        self.target_network = copy.deepcopy(self.q_network)
        '''

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

    def train(self, ep_num,file_path):

        if len(self.replay_buffer) >= self.batch_size:
            # train the agent
            loss_save = np.zeros(self.num_batches)
            for i in range(self.num_batches):
                # TODO look at own code maybe? and compare?
                states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

                """

                #state_action_values = self.q_network(states).gather(1, actions.type(torch.int64))
                state_action_values = self.q_network(states).gather(1, actions)

                next_state_values = torch.zeros(self.batch_size, device=self.device)
                next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1)[0].detach()

                expected_state_action_values = torch.add(
                    torch.mul(torch.reshape(next_state_values, (len(next_state_values), 1)),
                              self.gamma), rewards)

                loss = F.mse_loss(state_action_values.type(torch.double),
                                  expected_state_action_values.type(torch.double))

                # Optimize the model
                self.optimizer.zero_grad()
                loss.backward()
                for param in self.q_network.parameters():
                    param.grad.data.clamp_(-1, 1)
                self.optimizer.step()

                """
                #tmp0 = q_target_next = self.forward_target(next_states)
                #tmp1 = rewards + (self.gamma * q_target_next * (1 - dones))
                q_target_next = self.forward_target(next_states).detach().max(1)[0].unsqueeze(1)
                q_target = rewards + (self.gamma * q_target_next * (1-dones))

                q_expected = self.forward_with_grad(states).gather(1,actions.long())
                #q_expected_2 = self.forward_with_grad(states)

                # TODO save loss
                loss = F.mse_loss(q_expected, q_target)
                loss_save[i] = loss

                # take optimization step
                self.optimizer[self.name].zero_grad() # reset gradients
                loss.backward()
                self.optimizer[self.name].step()


            # save the loss value
            #self.save_loss

            # check and update target network if needed
            if ep_num - self.last_target_update > self.target_update_rate:
                self.last_target_update = ep_num

                self.target_network.load_state_dict(self.q_network.state_dict())

                self.save_model(ep_num,file_path)

    def update_memory(self, action_operation, done, entities, reward, sensors, sim_time):

        # if is_new_action == True save data, else don't save the data
        if sim_time - self.last_reset_time >= action_operation.frequency:
            self.state_info['norm_next_state'], self.state_info['next_state'] = self.normalize_state( entities, sensors)

            self.replay_buffer.add_experience(self.state_info['norm_state'], self.action_info['mutated_action'], reward, self.state_info['norm_next_state'], done)

            # save tuple to history
            self.add_sample_history(reward, done, sim_time)

    def forward(self,state):
        with torch.no_grad():
            # the with grad function is called but the gradients are no saved
            return self.forward_with_grad(state)

    def forward_target(self,state):
        with torch.no_grad():
            # the with grad function is called but the gradients are no saved
            return self.forward_with_grad_target(state)

    def forward_with_grad(self,state):

        return self.q_network.forward(state)

    def forward_with_grad_target(self,state):

        return self.target_network.forward(state)

    def save_model(self, episode_number, file_path):

        file_path = os.path.join(file_path, 'models')
        torch.save(self.q_network.state_dict(), os.path.abspath(os.path.join(file_path, str(self.name) + '_epnum-' + str(episode_number) + '.mdl')))





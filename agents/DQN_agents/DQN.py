"""
A DQN Learning algorithm implementation
"""

# native modules
import copy
import os

# 3rd part modules
import numpy as np
import torch.nn.functional as F

# own modules
from agents.BaseLearningAlgorithm import BaseLearningAlgorithm
from environment.NetworkBuilder import Network
import torch


class DQN(BaseLearningAlgorithm):

    def __init__(self,device, exploration_strategy, general_params, name, network_description, observation_information, optimizer_dict, output_dim, replay_buffer, save_rate, seed):

        # device, exploration_strategy, general_params, name, observation_information
        super(DQN,self).__init__(device,exploration_strategy, general_params, name, observation_information)

        # create the q network and the target network
        self.q_network = Network(device,observation_information,network_description, output_dim)
        self.target_network = Network(device,observation_information,network_description, output_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # create replay buffer
        self.replay_buffer = replay_buffer

        # setup optimizer
        self.load_optimizer(self.name, optimizer_dict, self.q_network.parameters())

        # save extra general parameters
        self.target_update_rate = general_params['target_update_rate']

        self.last_target_update = -np.infty
        self.loss_header = "Episode_number,loss\n"

        self.save_rate = save_rate

    def create_state_action(self, delta_t, action_operation, entities, ep_num, sensors, sim_time, use_exploration):

        # determine if new action shall be determined. If so set is_new_action to true
        if np.abs(sim_time - self.last_reset_time)+delta_t/2.0 >= action_operation.frequency:

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
            else:
                mutated_action = copy.deepcopy(raw_action)

            # partial log own data (input, raw output, perturbed output)
            self.action_info = {'raw_action':raw_action, 'mutated_action': mutated_action, 'q_values': q_values}

            # save information that it is persistent for the action operation until the next time an action is selected.
            self.action_info['persistent_info'] = action_operation.setPersistentInfo(entities, sensors, mutated_action)

    def train(self, ep_num,file_path):

        if len(self.replay_buffer) >= self.batch_size:
            # train the agent
            loss_save = np.zeros(self.num_batches)
            for i in range(self.num_batches):
                states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

                q_target_next = self.forward_target(next_states).detach().max(1)[0].unsqueeze(1)
                q_target = rewards + (self.gamma * q_target_next * (1-dones))

                q_expected = self.forward_with_grad(states).gather(1,actions.long())

                loss = F.mse_loss(q_expected, q_target)
                loss_save[i] = loss

                # take optimization step
                self.optimizer[self.name].zero_grad() # reset gradients
                loss.backward()
                self.optimizer[self.name].step()


            # save the loss value
            self.append_to_loss_file( ep_num, {'loss':loss_save})

            # check and update target network if needed
            if ep_num - self.last_target_update > self.target_update_rate:
                self.last_target_update = ep_num
                self.target_network.load_state_dict(self.q_network.state_dict())

        # check if model should be saved
        if ep_num % self.save_rate == 0:
            self.save_model(ep_num, file_path)

    def update_memory(self, delta_t, action_operation, done, entities, reward, sensors, sim_time):

        # if is_new_action == True save data, else don't save the data
        if np.abs(sim_time - self.last_reset_time)+delta_t/2.0 >= action_operation.frequency or done:
            self.state_info['norm_next_state'], self.state_info['next_state'] = self.normalize_state( entities, sensors)

            self.replay_buffer.add_experience(self.state_info['norm_state'], self.action_info['mutated_action'], reward, self.state_info['norm_next_state'], done)

            # save tuple to history
            self.add_sample_history(reward, done, sim_time)

    def load_networks(self, model_path, model_num):
        critic_file = os.path.join(model_path, self.name + '_epnum-' + str(model_num) + '.mdl')
        self.q_network.load_state_dict(torch.load(critic_file))
        self.q_network.eval()

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





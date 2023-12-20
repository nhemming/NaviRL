"""
A DDPG learning algorithm implementation
"""

# native modules
import copy
from collections import OrderedDict
import os

# 3rd part modules
import numpy as np
import torch.nn.functional as F

# own modules
from agents.BaseLearningAlgorithm import BaseLearningAlgorithm
from environment.NetworkBuilder import Network
import torch


class DDPG(BaseLearningAlgorithm):

    def __init__(self,device, exploration_strategy, general_params, name, network_description_actor, network_description_critic, observation_information, optimizer_dict, output_dim, replay_buffer, save_rate, seed):
        # device, exploration_strategy, general_params, name, observation_information
        super(DDPG, self).__init__(device, exploration_strategy, general_params, name, observation_information)

        # add a row to the observation_information to build a network that also takes in the selected action
        critic_observation_information = copy.deepcopy(observation_information)
        for name, value in critic_observation_information.items():
            tmp_dict = {key: None for key in list(value.columns)}

            # add empty columns. One for each dimension of the action space
            for _ in range(output_dim):
                value.loc[len(value)] = tmp_dict

        # create the critic networks
        self.critic_network = Network(device, critic_observation_information, network_description_critic, output_dim)
        self.critic_target_network = Network(device, critic_observation_information, network_description_critic, output_dim)
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # create the actor networks
        self.actor_network = Network(device, observation_information, network_description_actor, output_dim)
        self.actor_target_network = Network(device, observation_information, network_description_actor, output_dim)
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())

        # create replay buffer
        self.replay_buffer = replay_buffer

        # save extra general parameters
        self.target_update_rate = general_params['target_update_rate']
        self.tau = general_params['tau']

        # TODO come up with a way to have seperate optimizers for the actor and the critic
        # parse optimizer
        self.load_optimizer(self.name+'_critic', optimizer_dict, self.critic_network.parameters())
        self.load_optimizer(self.name+'_actor', optimizer_dict, self.actor_network.parameters())

        self.last_target_update = -np.infty
        self.loss_header = "Episode_number,actor_loss,critic_loss\n"
        self.save_rate = save_rate

    def init_state_action(self,action_operation,entities,sensors):
        # call the action operation preperation step.
        action_operation.init_state_action(entities,sensors)

    def prep_state_action(self,action_operation,entities,sensors,sim_time ):
        action_operation.prep_state_action(entities,sensors,sim_time )

    def create_state_action(self, action_operation, entities, ep_num, sensors, sim_time, use_exploration):
        """
        parses the current state of the simulation to build the state required for the agent.
        :param action_operation:
        :param entities:
        :param ep_num:
        :param sensors:
        :param sim_time:
        :param use_exploration:
        :return:
        """

        # determine if new action shall be determined. If so set is_new_action to true
        if sim_time - self.last_reset_time >= action_operation.frequency:

            self.last_reset_time = sim_time

            # collect state information and normalize state information
            norm_state, state = self.normalize_state(entities,sensors)
            self.state_info = {'norm_state':norm_state, 'state':state}

            # produce action
            raw_action = self.forward_actor(norm_state)
            if self.device == 'cuda':
                raw_action = raw_action.cpu()
            raw_action = raw_action.numpy()

            # produce critic values
            critic_values = self.forward_critic(norm_state, raw_action) # TODO is this the raw action, or should the perturbed action be used? Thinking the perturbed action is the correct one.
            if self.device == 'cuda':
                critic_values = critic_values.cpu()
            critic_values = critic_values.numpy()

            # exploration perturbations
            mutated_action = None
            if use_exploration:
                mutated_action = self.exploration_strategy.add_perturbation(raw_action, ep_num)

            # partial log own data (input, raw output, perturbed output)
            self.action_info = {'raw_action':raw_action, 'mutated_action': mutated_action, 'q_values': critic_values}

            # save information that it is persistent for the action operation until the next time an action is selected.
            self.action_info['persistent_info'] = action_operation.setPersistentInfo(entities, sensors, mutated_action)

    def train(self, ep_num,file_path):

        if len(self.replay_buffer) >= self.batch_size:
            # train the agent
            loss_save_actor = np.zeros(self.num_batches)
            loss_save_critic = np.zeros(self.num_batches)
            for i in range(self.num_batches):
                states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)


                # update the critic
                actions_next = self.forward_actor_target(next_states) #self.actor_target_network(next_states)
                critic_targets_next = self.forward_critic_target(next_states, actions_next) #self.critic_target_network(torch.cat((next_states, actions_next), 1))
                critic_targets = rewards + (
                            self.gamma * critic_targets_next * (1.0 - dones))

                critic_expected = self.forward_critic_with_grad(states,actions) #self.critic_network(torch.cat((states, actions), 1))
                critic_loss = F.mse_loss(critic_expected, critic_targets)

                # step critic optimizer. # TODO change to have two optimizers
                self.optimizer[self.name+'_critic'].zero_grad()  # reset gradients
                critic_loss.backward()
                self.optimizer[self.name+'_critic'].step()

                # update the actor
                actions_pred = self.forward_actor_with_grad(states) #self.actor_network(states)
                actor_loss = -self.forward_critic_with_grad(states,actions_pred).mean() #-self.critic_network(torch.cat((states, actions_pred), 1)).mean()

                # step actor optimizer. # TODO change to have two optimizers
                self.optimizer[self.name+'_actor'].zero_grad()  # reset gradients
                actor_loss.backward()
                self.optimizer[self.name+'_actor'].step()


                loss_save_actor[i] = actor_loss
                loss_save_critic[i] = critic_loss

            loss_dict = OrderedDict()
            loss_dict['actor'] = loss_save_actor
            loss_dict['critic'] = loss_save_critic
            # save the loss value
            self.append_to_loss_file( ep_num, loss_dict)

            # check and update target network if needed
            if ep_num - self.last_target_update > self.target_update_rate:

                self.last_target_update = ep_num

                # soft update critic target network
                for target_param, local_param in zip(self.critic_target_network.parameters(),
                                                     self.critic_network.parameters()):
                    target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

                # soft update actor target network
                for target_param, local_param in zip(self.actor_target_network.parameters(),
                                                     self.actor_network.parameters()):
                    target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

            # check if model should be saved
            if ep_num % self.save_rate == 0:
                self.save_model(ep_num, file_path)

    def update_memory(self, action_operation, done, entities, reward, sensors, sim_time):

        # if is_new_action == True save data, else don't save the data
        if sim_time - self.last_reset_time >= action_operation.frequency or done:
            self.state_info['norm_next_state'], self.state_info['next_state'] = self.normalize_state( entities, sensors)

            self.replay_buffer.add_experience(self.state_info['norm_state'], self.action_info['mutated_action'], reward, self.state_info['norm_next_state'], done)

            # save tuple to history
            self.add_sample_history(reward, done, sim_time)

    def load_networks(self, model_path, model_num):

        critic_file = os.path.join(model_path,self.name+'_epnum-Critic'+str(model_num)+'.mdl')
        self.critic_network.load_state_dict(torch.load(critic_file))
        self.critic_network.eval()

        actor_file = os.path.join(model_path, self.name + '_epnum-Actor' + str(model_num) + '.mdl')
        self.actor_network.load_state_dict(torch.load(actor_file))
        self.actor_network.eval()

    def forward_actor(self,state):
        with torch.no_grad():
            # the with grad function is called but the gradients are no saved
            return self.forward_actor_with_grad(state)

    def forward_critic(self,state, action):
        with torch.no_grad():
            # the with grad function is called but the gradients are no saved
            return self.forward_critic_with_grad(state, action)

    def forward_critic_target(self,state, action):
        with torch.no_grad():
            # the with grad function is called but the gradients are no saved
            return self.forward_critic_with_grad_target(state, action)

    def forward_actor_target(self,state):
        with torch.no_grad():
            # the with grad function is called but the gradients are no saved
            return self.forward_actor_with_grad_target(state)

    def forward_critic_with_grad(self,state, action):
        sa = copy.deepcopy(state) # TODO try and come up with a way to avoid this copy
        for name, value in state.items():
            if isinstance(action, np.ndarray):
                sa[name] = np.concatenate((value, action))
            else:
                sa[name] = torch.cat((value, action),dim=1)
        return self.critic_network.forward(sa)

    def forward_actor_with_grad(self,state):
        return self.actor_network.forward(state)

    def forward_critic_with_grad_target(self,state, action):
        sa = copy.deepcopy(state) # TODO try and come up with a way to avoid this copy
        for name, value in state.items():
            if isinstance(action, np.ndarray):
                sa[name] = np.concatenate((value, action))
            else:
                sa[name] = torch.cat((value, action), dim=1)
        return self.critic_target_network.forward(sa)

    def forward_actor_with_grad_target(self,state):
        return self.actor_target_network.forward(state)

    def save_model(self, episode_number, file_path):

        file_path = os.path.join(file_path, 'models')
        torch.save(self.critic_network.state_dict(), os.path.abspath(os.path.join(file_path, str(self.name) + '_epnum-Critic' + str(episode_number) + '.mdl')))
        torch.save(self.actor_network.state_dict(), os.path.abspath(
            os.path.join(file_path, str(self.name) + '_epnum-Actor' + str(episode_number) + '.mdl')))



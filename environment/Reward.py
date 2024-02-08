"""
Defines how the reward function used to train the agent. The total reward function is built up via combination of
reward function pieces.
"""

# native modules
from abc import ABC, abstractmethod
from collections import OrderedDict

# 3rd party modules
import numpy as np

# own modules


class RewardDefinition:

    def __init__(self, agent_names):
        self.reward_components = OrderedDict()
        self.overall_adj_factor = None
        self.reward_agents = {key: OrderedDict() for key in agent_names}

    def calculate_reward(self, sim_time, entities, sensors):

        for name, value in self.reward_components.items():
            value.calculate_reward(sim_time,entities, sensors, self.reward_agents)
        return self.reward_agents


class RewardComponent(ABC):

    def __init__(self, adj_factor, name, target_agent, target_lrn_alg):
        self.adj_factor = adj_factor
        self.name = name
        self.target_agent = target_agent
        self.target_lrn_alg = target_lrn_alg

    @abstractmethod
    def calculate_reward(self,sim_time,entities, sensors, reward_agents):
        pass

    def reset(self, entities, sensors, reward_agents):
        # do nothing for default reset function
        pass

    def get_key(self):
        return self.target_lrn_alg + ':' + self.target_agent


class AlignedHeadingReward(RewardComponent):

    def __init__(self, adj_factor, aligned_angle, aligned_reward,  destination_sensor, target_agent,target_lrn_alg):
        name = "Aligned_Heading_Reward"
        super(AlignedHeadingReward, self).__init__(adj_factor, name, target_agent,target_lrn_alg)

        self.aligned_angle = aligned_angle
        self.aligned_reward = aligned_reward
        self.destination_sensor = destination_sensor
        self.old_heading_offset = None

    def calculate_reward(self,sim_time,entities, sensors, reward_agents):

        heading_offset = sensors[self.destination_sensor].state_dict['angle']

        # reward for pointing entity near or directly at goal
        if heading_offset <= self.aligned_angle:
            reward_agents[self.target_agent][self.target_lrn_alg] += self.aligned_reward*self.adj_factor


class CloseDistanceReward(RewardComponent):

    def __init__(self, adj_factor, destination_sensor, target_agent, target_lrn_alg):
        name = "Closer_To_Goal_Reward"
        super(CloseDistanceReward, self).__init__(adj_factor, name, target_agent, target_lrn_alg)

        self.destination_sensor = destination_sensor
        self.old_dst = None

    def reset(self, entities, sensors, reward_agents):
        self.old_dst = sensors[self.destination_sensor].state_dict['distance']
        reward_agents[self.target_agent][self.target_lrn_alg] = 0.0


    def calculate_reward(self, sim_time,entities, sensors, reward_agents):

        dst = sensors[self.destination_sensor].state_dict['distance']

        if self.old_dst > dst:
            reward_agents[self.target_agent][self.target_lrn_alg] += (self.old_dst-dst)*self.adj_factor


class ImproveHeadingReward(RewardComponent):

    def __init__(self, adj_factor,  destination_sensor, target_agent, target_lrn_alg):
        name = "Improve_Heading_Reward"
        super(ImproveHeadingReward, self).__init__(adj_factor, name, target_agent, target_lrn_alg)

        self.destination_sensor = destination_sensor
        self.old_heading_offset = None

    def reset(self, entities, sensors, reward_agents):
        self.old_heading_offset = sensors[self.destination_sensor].state_dict['angle']
        reward_agents[self.target_agent][self.target_lrn_alg] = 0.0

    def calculate_reward(self,sim_time,entities, sensors, reward_agents):

        heading_offset = sensors[self.destination_sensor].state_dict['angle']

        # reward for improving heading
        delta_heading = np.abs(self.old_heading_offset) - np.abs(heading_offset)
        if delta_heading >= 0.0:
            reward_agents[self.target_agent][self.target_lrn_alg] += delta_heading * self.adj_factor

        self.old_heading_offset = heading_offset


class ReachDestinationReward(RewardComponent):

    def __init__(self, adj_factor, destination_sensor, goal_dst, reward, target_agent, target_lrn_alg):
        name = "Reach_Destination_Reward"
        super(ReachDestinationReward, self).__init__(adj_factor, name, target_agent, target_lrn_alg)
        self.destination_sensor = destination_sensor
        self.goal_dst = goal_dst
        self.reward = reward

    def calculate_reward(self,sim_time,entities, sensors, reward_agents):

        if sim_time > 1e-12:
            # not allowed to recieve reward for being at the goal location at the intiailization of the simulation. Fix for sub desntionation based sensors

            curr_dst = sensors[self.destination_sensor].state_dict['distance']

            if curr_dst <= self.goal_dst + 0.05:

                reward_agents[self.target_agent][self.target_lrn_alg] += self.reward

    def reset(self,entities, sensors, reward_agents):
        reward_agents[self.target_agent][self.target_lrn_alg] = 0.0
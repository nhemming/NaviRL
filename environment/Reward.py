"""
Defines how the reward function used to train the agent. The total reward function is built up via combination of
reward function pieces.
"""

# native modules
from abc import ABC, abstractmethod

# 3rd party modules

# own modules


class RewardDefinition:

    def __init__(self):
        self.reward_components = dict()
        self.overall_adj_factor = None

    def calculate_reward(self, entities, sensors):

        reward = 0.0
        for name, value in self.reward_components.items():
            reward += value.calc_reward(entities, sensors)

        return reward


class RewardComponent(ABC):

    def __init__(self, adj_factor, name):
        self.adj_factor = adj_factor
        self.name = name

    @abstractmethod
    def calculate_reward(self,entities, sensors):
        pass

    def reset(self, entities, sensors):
        # do nothing for default reset function
        pass


class AlignedHeadingReward(RewardComponent):

    def __init__(self, adj_factor, aligned_angle, aligned_reward,  destination_sensor):
        name = "Aligned_Heading_Reward"
        super(AlignedHeadingReward, self).__init__(adj_factor, name)

        self.aligned_angle = aligned_angle
        self.aligned_reward = aligned_reward
        self.destination_sensor = destination_sensor
        self.old_heading_offset = None

    def calculate_reward(self,entities, sensors):

        heading_offset = sensors[self.destination_sensor].state_dict['angle']

        # reward for pointing entity near or directly at goal
        if heading_offset <= self.aligned_angle:
            return self.aligned_reward
        return 0.0


class CloseDistanceReward(RewardComponent):

    def __init__(self, adj_factor, destination_sensor):
        name = "Closer_To_Goal_Reward"
        super(CloseDistanceReward, self).__init__(adj_factor, name)

        self.destination_sensor = destination_sensor
        self.old_dst = None

    def reset(self, entities, sensors):
        self.old_dst = sensors[self.destination_sensor].state_dict['distance']

    def calculate_reward(self, entities, sensors):

        dst = sensors[self.destination_sensor].state_dict['distance']

        if self.old_dst > dst:
            return (self.old_dst-dst)*self.adj_factor

        return 0.0


class ImproveHeadingReward(RewardComponent):

    def __init__(self, adj_factor,  destination_sensor):
        name = "Improve_Heading_Reward"
        super(ImproveHeadingReward, self).__init__(adj_factor, name)

        self.destination_sensor = destination_sensor
        self.old_heading_offset = None

    def reset(self, entities, sensors):
        self.old_heading_offset = sensors[self.destination_sensor].state_dict['angle']

    def calculate_reward(self,entities, sensors):

        heading_offset = sensors[self.destination_sensor].state_dict['angle']

        # reward for improving heading
        reward = 0.0
        delta_heading = self.old_heading_offset - heading_offset
        if delta_heading >= 0.0:
            reward += delta_heading

        self.old_heading_offset = heading_offset

        return reward*self.adj_factor


class ReachDestinationReward(RewardComponent):

    def __init__(self, adj_factor, destination_sensor, goal_dst, reward):
        name = "Reach_Destination_Reward"
        super(ReachDestinationReward, self).__init__(adj_factor, name)
        self.destination_sensor = destination_sensor
        self.goal_dst = goal_dst
        self.reward = reward

    def calculate_reward(self,entities, sensors):

        curr_dst = sensors[self.destination_sensor].state_dict['distance']

        if curr_dst <= self.goal_dst:
            return self.reward
        return 0.0

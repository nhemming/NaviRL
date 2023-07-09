"""
Describes the base class for all learning agents. Each agent has a learning algorithm, an entity(ies) that
it applies too, and an action operation that converts raw neural network outputs to entity state changes.
"""

# native modules
from abc import ABC, abstractmethod

# 3rd party modules
import numpy as np

# own modules


class BaseAgent(ABC):

    def __init__(self, action_operation, controlled_entity, device, exploration_strategy, name, observation_information):
        self.action_operation = action_operation
        self.controlled_entity = controlled_entity
        self.device = device
        self.exploration_strategy = exploration_strategy
        self.name = name
        self.observation_information = observation_information
        self.is_new_action = False

        self.state_info = dict()
        self.action_info = dict()

    @abstractmethod
    def create_state_action(self, entities, episode_num, sensors, sim_time, use_exploration):

        # determine if new action shall be determined. If so set is_new_action to true

        # collect state information

        # normalize state information

        # produce action

        # exploration perturbations

        # partial log own data (input, raw output, perturbed output)

        # return state and action information as dictionary

        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def update_memory(self):

        # if is_new_action == True save data, else don't save the data

        # set is_new_action to false.
        pass

    @abstractmethod
    def forward(self,state):
        """
        runs the neural network and returns the output
        :param state:
        :return: raw action vector
        """
        pass

    def reset(self):
        # do nothing in base class
        pass

    def apply_action(self, entities):
        """
        This method takes the raw actions and forwards them to the entity to change its control mechanism.
        :param entities: dictionary of entities in the simulation
        :return:
        """

        ce = entities.get(self.controlled_entity,None)
        if ce is None:
            raise ValueError('Entity controlled by the agent not in the simulation')

        # each entity is responsible for parsing the vector to apply changes.
        ce.update_control(self.action_info['applied_action'])

    def normalize_state(self,entities, sensors):

        state = np.zeros(len(self.observation_information))
        norm_state = np.zeros(len(self.observation_information))
        for index, row in self.observation_information.iterrows():
            item = entities.get(row['name'],None)
            if item is None:
                item = sensors.get(row['name'],None)
            norm_state[index] = ((item.state_dict[row['data']]-row['min']) / (row['max'] - row['min'])) * (row['norm_max']-row['norm_min']) + row['norm_min']
            state[index] = item.state_dict[row['data']]

        return norm_state, state

    def execute_action_operation(self, entities):

        # scale the output to the robots action?
        self.action_info['applied_action'] = self.action_operation.convert_action(self.action_info['mutated_action'])

        # apply the action
        entities[self.controlled_entity].apply_action(self.action_info['applied_action'])
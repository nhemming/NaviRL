"""
Base learning agent that acts in the simulation. It can have multiple learning algorithms or only one
depending on how many independent decisions it needs to make.
"""

# native modules
from abc import ABC, abstractmethod
from collections import OrderedDict

# 3rd party modules

# own modules


class BaseAgent(ABC):

    def __init__(self, action_operation,controlled_entity,name, save_rate):
        self.action_operation = action_operation
        self.controlled_entity = controlled_entity
        self.learning_algorithms = OrderedDict()
        self.name = name
        self.save_rate = save_rate

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

    def execute_action_operation(self, delta_t, entities, sensors):

        for name, value in self.learning_algorithms.items():
            value.execute_action_operation(self.action_operation, delta_t, self.controlled_entity, entities, sensors)

    def reset(self):
        for name, value in self.learning_algorithms.items():
            value.reset()

    def init_state_action(self, entities, sensors):
        for name, value in self.learning_algorithms.items():
            value.init_state_action(self.action_operation, entities, sensors)

    def prep_state_action( self, entities, sensors, sim_time):
        for name, value in self.learning_algorithms.items():
            value.prep_state_action(self.action_operation,entities,sensors, sim_time)

    def create_state_action(self, delta_t, entities, episode_num, sensors, sim_time, use_exploration):
        for name, value in self.learning_algorithms.items():
            value.create_state_action(delta_t, self.action_operation,entities, episode_num, sensors, sim_time, use_exploration)

    def update_memory(self,delta_t, tmp_done, entities, tmp_reward, sensors, sim_time):
        for name, reward in tmp_reward.items():
            self.learning_algorithms[name].update_memory(delta_t,self.action_operation,tmp_done, entities, reward, sensors, sim_time)

    def train(self, episode_num, file_path):
        for name, agent in self.learning_algorithms.items():
            agent.train(episode_num, file_path)

    def save_model(self,episode_num,history_path):
        for _, lrn_alg in self.learning_algorithms.items():
            lrn_alg.save_model(episode_num,history_path)


class SingleLearningAlgorithmAgent(BaseAgent):

    def __init__(self,action_operation,controlled_entity,name, save_rate):
        super(SingleLearningAlgorithmAgent, self).__init__(action_operation,controlled_entity,name,save_rate)


class VoidAgent(BaseAgent):

    def __init__(self,action_operation,controlled_entity,name, save_rate):
        super(VoidAgent, self).__init__(action_operation,controlled_entity,name,save_rate)

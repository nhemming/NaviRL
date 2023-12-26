"""
Algorithm used when no RL learning is used.
"""

# native modules
from collections import OrderedDict

# own modules
from agents.BaseLearningAlgorithm import BaseLearningAlgorithm


class NoLearning(BaseLearningAlgorithm):

    def __init__(self):
        super(NoLearning, self).__init__('cpu', None, None, 'NoLearning', None)

        self.action_info = OrderedDict()

    def create_state_action(self, action_operation, entities, ep_num, sensors, sim_time, use_exploration):
        self.action_info['mutated_action'] = None

    def load_networks(self, model_path, model_num):
        pass

    def update_memory(self, action_operation, done, entities, reward, sensors, sim_time):
        pass

    def train(self, ep_num, file_path):
        pass

    def save_model(self, episode_number, file_path):
        pass

    # Override base type
    def create_loss_file(self, base_dir, agent):
        pass
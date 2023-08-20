"""
Epilson greedy exploration strategy
"""

# 3rd party modules
import numpy as np
import torch


class EpsilonGreedy:

    def __init__(self, device, is_continuous, name, threshold_schedule, mutation_definition):
        self.device = device
        self.is_continuous = is_continuous
        self.name = name
        self.threshold_schedule = threshold_schedule

        # if continuous, this is the number of total actions. If continuous, this describes how perturbations are sampled
        num_options = 1
        for name, value in mutation_definition.items():
            if isinstance(value,str):
                # there is more than one option so split them. If this if statement is not taken, that means there is
                # only one option.
                num_options *= len([float(i) for i in value.split(',')])

        self.mutation_definition = num_options

    def add_perturbation(self,actions, ep_num):

        # get the threshold for the random action
        eps_threshold = 0.0
        for i, point in enumerate(self.threshold_schedule):

            if i == len(self.threshold_schedule) - 1:
                # hold last value as past the last defined point
                eps_threshold = point[1]
                break
            elif point[0] <= ep_num < self.threshold_schedule[i+1][0]:
                slope = (self.threshold_schedule[i+1][1]-point[1])/(self.threshold_schedule[i+1][0]-point[0])
                intercept = point[1]-slope*point[0]
                eps_threshold = slope*ep_num + intercept
                break

        # draw random sample add check if random perturbation should be added
        sample = np.random.uniform(low=0,high=1)
        if sample <= eps_threshold:
            # use mutated action

            if self.is_continuous:
                for i, tmp_action in enumerate(actions):
                    # add perturbation
                    # TODO
                    pass
            else:
                return np.random.randint(0,self.mutation_definition)

        # do  not add mutation. Use original action
        return actions

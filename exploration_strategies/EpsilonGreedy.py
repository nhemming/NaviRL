"""
Epilson greedy exploration strategy
"""

# 3rd party modules
import copy

import numpy as np
import pandas as pd
import torch


class EpsilonGreedy:

    def __init__(self, device, is_continuous, name, threshold_schedule, action_definition, perturbation_dist):
        """
        Creates the epsilon greedy exploration method. During training, epsilon greedy adds a random perturbation some
        percentage of the time. The rate at which random perturbation is added to an action is defined with a schedule.
        This helps the RL agent with exploration by forcing non-preferred actions to be taken.

        :param device: What device the actions produced by a neural network are on.
        :param is_continuous: Boolean for if the actions are continuous or discrete. Controls how pertubations are done.
        :param name: A string helpful for debugging.
        :param threshold_schedule: A schedule defining points with x being the episode number and y being a threshold
            for the rate of random actions. 1 is every action is random, and 0 is no actions have random perturbations
            added to it.
        :param action_definition: Meta data defining how actions are defined. This is primarily needed by the learning
            agent itself, but helps define how perturbation can be used. Typically, used for the discrete case.
        :param perturbation_dist: Defining gaussian distributions defining how perturbations are added for the
            continuous case. The mean 'mu' and 'std' are defined in this dictionary. They are in order with the action
            outputs.
        """
        self.device = device
        self.is_continuous = is_continuous
        self.name = name
        self.threshold_schedule = threshold_schedule

        # save distribution perturbation information
        if is_continuous:
            data = np.zeros((len(perturbation_dist),2))
            k = 0
            for name, value in perturbation_dist.items():
                dist_info = [float(i) for i in value.split(',')]
                data[k,:] = dist_info
                k += 1
            self.mutation_definition_continuous = pd.DataFrame(columns=['Mean','Std'],data=data)
        else:
            num_options = 1
            for name, value in action_definition.items():
                if isinstance(value, str):
                    # there is more than one option so split them. If this if statement is not taken, that means there is
                    # only one option.
                    num_options *= len([float(i) for i in value.split(',')])

            self.mutation_definition_discrete = num_options

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
                new_actions = copy.deepcopy(actions)
                i = np.random.randint(0,len(new_actions))
                new_actions[i] = new_actions[i] + np.random.normal(self.mutation_definition_continuous['Mean'].iloc[i],self.mutation_definition_continuous['Std'].iloc[i])

                return new_actions
            else:
                return np.random.randint(0,self.mutation_definition_discrete)

        # do  not add mutation. Use original action
        return actions

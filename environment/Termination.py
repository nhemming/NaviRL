"""
Termination function is a composite of conditions that check for what state determines what stops a simulation.
"""

# native modules
from abc import ABC, abstractmethod

# 3rd party modules
import numpy as np

# own modules
from environment.Entity import CollideEntity


class TerminationDefinition:

    def __init__(self):
        self.components = dict()


class TerminationComponent(ABC):

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def calculate_termination(self, entities, sensors):
        pass


class ReachDestinationTermination(TerminationComponent):

    def __init__(self, destination_sensor, goal_dst, name):
        super(ReachDestinationTermination, self).__init__(name)

        self.destination_sensor = destination_sensor
        self.goal_dst = goal_dst

    def calculate_termination(self, entities, sensors):

        dst = sensors[self.destination_sensor].state_dict['distance']
        if dst <= self.goal_dst:
            return True
        return False


class AnyCollisionsTermination(TerminationComponent):

    def __init__(self, name):
        super(AnyCollisionsTermination, self).__init__(name)

    def calculate_termination(self, entities, sensors):


        # TODO change to look at entites is collided flag


        return False
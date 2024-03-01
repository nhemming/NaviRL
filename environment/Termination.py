"""
Termination function is a composite of conditions that check for what state determines what stops a simulation.
"""

# native modules
from abc import ABC, abstractmethod
from collections import OrderedDict

# 3rd party modules
import numpy as np

# own modules
from environment.Entity import CollideEntity


class TerminationDefinition:

    def __init__(self, agent_names):
        self.components = OrderedDict()
        self.done_agents = {key: False for key in agent_names}

    def calculate_termination(self, entities, sensors):

        for name, value in self.components.items():
            value.calculate_termination(entities, sensors, self.done_agents)

        overall_done = False
        if any(list(self.done_agents.values())):
            overall_done = True

        return overall_done, self.done_agents

    def reset(self):
        for name, value in self.done_agents.items():
            self.done_agents[name] = False


class TerminationComponent(ABC):

    def __init__(self, name, target_agent):
        self.name = name
        self.target_agent = target_agent

    @abstractmethod
    def calculate_termination(self, entities, sensors, done_agents):
        pass


class ReachDestinationTermination(TerminationComponent):

    def __init__(self, destination_sensor, goal_dst, name, target_agent):
        super(ReachDestinationTermination, self).__init__(name, target_agent)

        self.destination_sensor = destination_sensor
        self.goal_dst = goal_dst

    def calculate_termination(self, entities, sensors, done_agents):

        dst = sensors[self.destination_sensor].state_dict['distance']
        if dst <= self.goal_dst:
            done_agents[self.target_agent] = True
        done_agents[self.target_agent] = done_agents[self.target_agent] or False


class AnyCollisionsTermination(TerminationComponent):

    def __init__(self, name, target_agent):
        super(AnyCollisionsTermination, self).__init__(name, target_agent)

    def calculate_termination(self, entities, sensors, done_agents):

        # TODO change to only look at target agent

        for name, entity in entities.items():
            if isinstance(entity,CollideEntity):
                if entity.state_dict['is_collided']:
                    done_agents[self.target_agent] = True
        done_agents[self.target_agent] = done_agents[self.target_agent] or False

class TooFarAwayTermination(TerminationComponent):

    def __init__(self, destination_sensor, max_dst, name, target_agent):
        super(TooFarAwayTermination, self).__init__(name, target_agent)

        self.destination_sensor = destination_sensor
        self.max_dst = max_dst

    def calculate_termination(self, entities, sensors, done_agents):

        dst = sensors[self.destination_sensor].state_dict['distance']
        if dst >= self.max_dst:
            done_agents[self.target_agent] = True
        done_agents[self.target_agent] = done_agents[self.target_agent] or False

"""
Sensors are used to aggregate relationship data between entities.
"""

# native modules
from abc import ABC, abstractmethod

# 3rd party modules
import numpy as np

# own modules
from environment.Entity import get_angle_between_vectors


class Sensor(ABC):

    def __init__(self, id, name, owner):
        self.id = id
        self.name = name
        self.owner = owner
        self.state_dict = dict()

    @abstractmethod
    def update(self, time, entities, sensors):
        pass


class DestinationSensor(Sensor):

    def __init__(self, id, name, owner, target):
        super(DestinationSensor,self).__init__(id, name, owner)
        self.target = target
        self.state_dict['angle'] = 0.0  # [rad]
        self.state_dict['distance'] = 0.0  # [m]

    def update(self, time, entities, sensors):
        # update the observation information from entites

        # look if item is in entities
        owner_entity = entities.get(self.owner, None)
        if owner_entity is None:
            owner_entity = sensors.get(self.owner, None)

        target_entity = entities.get(self.target, None)
        if target_entity is None:
            target_entity = sensors.get(self.target, None)

        # get angle from owner to destination
        owner_vec = [np.cos(owner_entity.state_dict['phi']),np.sin(owner_entity.state_dict['phi'])]
        sep_vec = [target_entity.state_dict['x_pos']-owner_entity.state_dict['x_pos'],target_entity.state_dict['y_pos']-owner_entity.state_dict['y_pos']]
        self.state_dict['angle'] = get_angle_between_vectors(owner_vec, sep_vec,True)

        # update the distance between the owner entity and the goal entity
        self.state_dict['distance'] = np.sqrt(sep_vec[0]*sep_vec[0] + sep_vec[1]*sep_vec[1])
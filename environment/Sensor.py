"""
Sensors are used to aggregate relationship data between entities.
"""

# native modules
from abc import ABC, abstractmethod
from collections import OrderedDict
import copy
import os

# 3rd party modules
import numpy as np
import pandas as pd

# own modules
from environment.Entity import get_angle_between_vectors


class Sensor(ABC):

    def __init__(self, id, name, owner):
        """

        :param id:
        :param name:
        :param owner:
        """
        self.id = id
        self.name = name
        self.owner = owner
        self.state_dict = OrderedDict()
        self.history = []

    @abstractmethod
    def update(self, time, entities, sensors):
        pass

    def add_step_history(self, sim_time):
        """
        Store the current state dict information of the sensor. This is eventually saved for post training analysis
        and visualization.

        :param sim_time: The current simulation time to function as a time stamp.
        :return:
        """
        self.state_dict['sim_time'] = sim_time
        self.history.append(copy.deepcopy(self.state_dict))

    def reset_history(self):
        """
        Clears out the history. Typically called at the end of the episode after the history has been saved.
        :return:
        """
        self.history = []

    def reset(self):
        self.reset_history()

    def write_history(self, episode_number, file_path, eval_num=''):
        # TODO change from csv to sqlite data base
        # write history to csv
        df = pd.DataFrame(self.history)
        file_path = os.path.join(file_path,'sensors')
        df.to_csv(os.path.abspath(os.path.join(file_path,str(self.name)+'_epnum-'+str(episode_number)+eval_num+'.csv')), index=False)


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

        # get the target entity
        target_entity = entities.get(self.target, None)
        if target_entity is None:
            target_entity = sensors.get(self.target, None)

        # get angle from owner to destination
        owner_vec = [np.cos(owner_entity.state_dict['phi']),np.sin(owner_entity.state_dict['phi'])]
        sep_vec = [target_entity.state_dict['x_pos']-owner_entity.state_dict['x_pos'],target_entity.state_dict['y_pos']-owner_entity.state_dict['y_pos']]
        self.state_dict['angle'] = get_angle_between_vectors(owner_vec, sep_vec,True)

        # update the distance between the owner entity and the goal entity
        self.state_dict['distance'] = np.sqrt(sep_vec[0]*sep_vec[0] + sep_vec[1]*sep_vec[1])

    def draw_angle(self,ax, data, sim_time):
        ax.plot(data['sim_time'], data['angle'], label='dest')

    def draw_distance(self,ax, data, sim_time):
        ax.plot(data['sim_time'], data['distance'], label='dest')

    def draw_at_time(self,ax, data, sim_time):
        pass

class SubDestinationSensor(Sensor):

    def __init__(self, id, name, owner):
        super(SubDestinationSensor,self).__init__(id, name, owner)
        self.state_dict['angle'] = 0.0  # [rad]
        self.state_dict['distance'] = 0.0  # [m]

    def update(self, time, entities, sensors):

        """
        # update the observation information from entites

        # look if item is in entities
        owner_entity = entities.get(self.owner, None)
        if owner_entity is None:
            owner_entity = sensors.get(self.owner, None)

        # get the target entity
        target_entity = entities.get(self.target, None)
        if target_entity is None:
            target_entity = sensors.get(self.target, None)

        # get angle from owner to destination
        owner_vec = [np.cos(owner_entity.state_dict['phi']),np.sin(owner_entity.state_dict['phi'])]
        sep_vec = [target_entity.state_dict['x_pos']-owner_entity.state_dict['x_pos'],target_entity.state_dict['y_pos']-owner_entity.state_dict['y_pos']]
        self.state_dict['angle'] = get_angle_between_vectors(owner_vec, sep_vec,True)

        # update the distance between the owner entity and the goal entity
        self.state_dict['distance'] = np.sqrt(sep_vec[0]*sep_vec[0] + sep_vec[1]*sep_vec[1])
        """

        # this sensor is updated manually by RL_PRM action operation method
        pass

    def draw_angle(self,ax, data, sim_time):
        ax.plot(data['sim_time'], data['angle'], label='sub_dest')

    def draw_distance(self,ax, data, sim_time):
        ax.plot(data['sim_time'], data['distance'], label='sub_dest')

    def draw_at_time(self,ax, data, sim_time):
        pass
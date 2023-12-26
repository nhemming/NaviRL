"""
Sensors are used to aggregate relationship data between entities.
"""

# native modules
from abc import ABC, abstractmethod
from collections import OrderedDict
import copy
import os

# 3rd party modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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


    def draw_traj(self, ax, data, sim_time):
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

    def draw_traj(self, ax, data, sim_time):
        pass


class DestinationPRMSensor(Sensor):

    def __init__(self, graph_frequency,  id, max_connect_dst, name, n_samples, owner, sample_domain, target, trans_dst,  model_path='', model_radius=None):
        super(DestinationPRMSensor,self).__init__(id, name, owner)

        # information the agent will use as input variables
        self.target = target # should be the destination object
        self.state_dict['angle'] = 0.0  # [rad]
        self.state_dict['distance'] = 0.0  # [m]

        # information for the sensor to maintain the PRM map
        self.vertices = []  # list of the vertices in the graph
        self.max_connect_dst = max_connect_dst
        self.n_samples = n_samples
        self.trans_dst = trans_dst  # distance to subgoal to acheive before moving to the next subgoal
        self.sample_domain = sample_domain # euclidean distance around owner entity to draw samples from for building the PRM
        self.model_path = model_path
        self.model_radius = model_radius
        self.use_simple_model = True if model_radius is not None else False
        self.last_reset_time = -np.infty  # [s]
        self.graph_frequency = graph_frequency

        # TODO enaable methodology for a surrogate transition model for the entity that owns this sensor

    def reset(self):

        # call super reset method
        super().reset()

        # reset time for calculating when to upate the prm
        self.last_reset_time = -np.infty  # [s]
        self.state_dict.clear()
        self.state_dict['angle'] = 0.0  # [rad]
        self.state_dict['distance'] = 0.0  # [m]

        self.path = []

    def update(self, time, entities, sensors):

        # look if item is in entities
        owner_entity = entities.get(self.owner, None)
        if owner_entity is None:
            owner_entity = sensors.get(self.owner, None)

        # get the target entity
        target_entity = entities.get(self.target, None)
        if target_entity is None:
            target_entity = sensors.get(self.target, None)

        if time - self.last_reset_time >= self.graph_frequency:
            self.last_reset_time = time

            goal_loc = [target_entity.state_dict['x_pos'], target_entity.state_dict['y_pos']]

            start_loc = [owner_entity.state_dict['x_pos'],
                         owner_entity.state_dict['y_pos']]
            state = {'phi': owner_entity.state_dict['phi']}

            # build a graph
            raw_path = self.build_prm(goal_loc, start_loc, state)

            # change path to be (x,y,theta)
            self.path = self.format_path(raw_path)

            # write path to state_dict
            for i, point in enumerate(self.path):
                self.state_dict['path_x_'+str(i)] = point[0]
                self.state_dict['path_y_' + str(i)] = point[1]
                self.state_dict['path_phi_' + str(i)] = point[2]

            # set current sub goal
            self.sub_goal_idx = 1

        # update sub destination sensor to point to the nearest point in the path
        curr_loc = [owner_entity.state_dict['x_pos'], owner_entity.state_dict['y_pos']]
        tmp_dst = np.sqrt(
            (curr_loc[0] - self.path[self.sub_goal_idx, 0]) ** 2 + (curr_loc[1] - self.path[self.sub_goal_idx, 1]) ** 2)
        if (tmp_dst <= self.trans_dst):
            self.sub_goal_idx += 1

        if self.sub_goal_idx >= len(self.path):
            self.sub_goal_idx = len(self.path) - 1  # correct for walking off the end of the array
            sub_goal = self.path[len(self.path) - 1, :2]
        else:
            sub_goal = self.path[self.sub_goal_idx, :2]

        dst = np.sqrt((curr_loc[0] - sub_goal[0]) ** 2 + (curr_loc[1] - sub_goal[1]) ** 2)
        '''
        target_unit_vec = [np.cos(target_entity.state_dict['phi']),
                           np.sin(target_entity.state_dict['phi'])]
        goal_vec = [sub_goal[0] - curr_loc[0], sub_goal[1] - curr_loc[1]]
        mu = get_angle_between_vectors(target_unit_vec, goal_vec, True)
        '''
        owner_unit_vec = [np.cos(owner_entity.state_dict['phi']),
                           np.sin(owner_entity.state_dict['phi'])]
        goal_vec = [sub_goal[0] - curr_loc[0], sub_goal[1] - curr_loc[1]]
        mu = get_angle_between_vectors(owner_unit_vec, goal_vec, True)
        if mu < 0:
            mu += 2.0 * np.pi
        elif mu > 2.0 * np.pi:
            mu -= 2.0 * np.pi

        # update the sub_goal sensor to point to the nearest point
        self.state_dict['angle'] = mu
        self.state_dict['distance'] = dst

    def format_path(self, path):
        # change path to be (x,y,theta)
        samples = np.zeros((len(path), 3))
        for i, vert in enumerate(reversed(path)):
            samples[i, 0] = vert.location[0]
            samples[i, 1] = vert.location[1]
            samples[i, 2] = vert.state['phi']

        return samples

    def build_prm(self, goal_loc, start_loc, state):
        # build the verticies of the PRM
        self.vertices = []
        self.vertices.append(VertexPRM(start_loc, state=state))
        for i in range(self.n_samples):

            # TODO need to check if the point is in an obstacle.

            # draw random point
            location = np.zeros((2,))
            location[0] = np.random.uniform(low=self.sample_domain[0,0]+start_loc[0], high=self.sample_domain[1,0]+start_loc[0])
            location[1] = np.random.uniform(low=self.sample_domain[0,1]+start_loc[1], high=self.sample_domain[1,1]+start_loc[1])

            self.vertices.append(VertexPRM(location))

        self.vertices.append(VertexPRM(goal_loc))

        # TODO remove after debugging
        '''
        sns.set_theme()
        fig = plt.figure(0,figsize=(9,9))
        x = []
        y = []
        for vert in self.vertices:
            x.append(vert.location[0])
            y.append(vert.location[1])
        ax = fig.add_subplot(1,1,1)
        plt.tight_layout()
        ax.plot(x,y,'o')
        ax.plot(start_loc[0],start_loc[1],'o',label='start')
        ax.plot(goal_loc[0], goal_loc[1], 'o', label='end')
        ax.legend()
        '''
        # TODO ^^^

        # try to connect the nodes in the PRM with a simple arc
        open_verts = [self.vertices[0]]
        counts = 0
        while len(open_verts) > 0 and counts < len(self.vertices) + 1:
            counts += 1
            current_vert = open_verts.pop()
            for i, tmp_vert in enumerate(self.vertices):
                if tmp_vert != current_vert:

                    curr_state = current_vert.state

                    dst = np.sqrt((current_vert.location[0] - tmp_vert.location[0]) ** 2 + (
                            current_vert.location[1] - tmp_vert.location[1]) ** 2)

                    if self.use_simple_model:
                        # use a simple arc model for determining if the node is reachable

                        # get angle from current heading to tmp vertex
                        delta_x = tmp_vert.location[0] - current_vert.location[0]
                        delta_y = tmp_vert.location[1] - current_vert.location[1]
                        theta = np.arctan2(delta_y, delta_x)
                        mu1 = theta - curr_state['phi']
                        if mu1 >= 0:
                            mu2 = np.pi * 2.0 - mu1  # explementary angle
                        else:
                            mu2 = np.pi * 2.0 + mu1  # explementary angle
                        mu_v = [mu1, mu2]
                        ind = np.argmin(np.abs(mu_v))
                        mu = mu_v[ind]

                        if np.abs(mu) > np.pi / 2.0:
                            gamma = np.abs(mu) - np.pi / 2.0
                        else:
                            gamma = np.pi / 2.0 - np.abs(mu)

                        arc_radius = dst * np.sin(gamma) / np.sin(np.pi - 2.0 * gamma)

                        # plot the  circle connecting everything
                        if mu < 0:
                            angle_off = -np.pi / 2.0

                        else:
                            angle_off = np.pi / 2.0
                        radius_vec = [arc_radius * np.cos(state['phi']), arc_radius * np.sin(state['phi'])]
                        radius_offset = [radius_vec[0] * np.cos(angle_off) - radius_vec[1] * np.sin(angle_off),
                                         radius_vec[0] * np.sin(angle_off) + radius_vec[1] * np.cos(angle_off)]
                        radius_point = [current_vert.location[0] + radius_offset[0],
                                        current_vert.location[1] + radius_offset[1]]

                        vec1 = [current_vert.location[0] - radius_point[0], current_vert.location[1] - radius_point[1]]
                        vec2 = [tmp_vert.location[0] - radius_point[0], tmp_vert.location[1] - radius_point[1]]
                        delta1 = get_angle_between_vectors(vec1, vec2, True)
                        unit_vec = [np.cos(state['phi']), np.sin(state['phi'])]
                        delta2 = get_angle_between_vectors(vec1, unit_vec, True)
                        circumfrance = 2.0 * np.pi * arc_radius
                        if (delta1 > 0 and delta2 < 0) or (delta1 < 0 and delta2 > 0):
                            # long way around
                            arc_length = max([circumfrance - np.abs(delta1) * arc_radius, np.abs(delta1) * arc_radius])
                        else:
                            # short way around
                            arc_length = min([circumfrance - np.abs(delta1) * arc_radius, np.abs(delta1) * arc_radius])

                        # get end angle
                        swept_angle = arc_length / arc_radius  # angle to rotate initial psi to to get ending angle
                        end_vec = [unit_vec[0] * np.cos(swept_angle) - unit_vec[1] * np.sin(swept_angle),
                                   unit_vec[0] * np.sin(swept_angle) + unit_vec[1] * np.cos(swept_angle)]
                        end_angle = np.arctan2(end_vec[1], end_vec[0])

                        if arc_radius >= self.model_radius and dst <= self.max_connect_dst and arc_length <= 2.0 * self.max_connect_dst:
                            # can reach the point. Make the connection and add the node to the open set.

                            if tmp_vert not in current_vert.children and current_vert not in tmp_vert.children:
                                tmp_vert.state = {'phi': end_angle}
                                tmp_vert.dst_to_par = arc_length
                                current_vert.children.append(tmp_vert)
                                open_verts.append(tmp_vert)
                    else:
                        # TODO
                        # use a surrogate model for determining if the next node is reachable
                        pass

        # TODO remove after debugging
        '''
        for vert in self.vertices:
            for child in vert.children:

                ax.plot([vert.location[0],child.location[0]],[vert.location[1],child.location[1]],'k--',alpha=0.05)
        '''
        # TODO ^^

        # extract path from graph with astar
        path, count = self.astar(self.vertices[0], self.vertices[len(self.vertices) - 1], len(self.vertices))

        # TODO remove after debugging
        '''
        x = []
        y = []
        for vert in path:
            x.append(vert.location[0])
            y.append(vert.location[1])
        ax.plot(x,y,':',label='pathOrg')
        ax.legend()
        '''
        # TODO ^^

        # set current waypoint to goal if no path exists.
        if path == [] or count >= len(self.vertices):
            path = []
            # I don't think the stat values matter here
            path.append(VertexPRM(goal_loc, state=state))
            path.append(VertexPRM(start_loc, state={'phi':0.0}))

            # TODO remove after debugging
            '''
            x = []
            y = []
            for vert in path:
                x.append(vert.location[0])
                y.append(vert.location[1])
            ax.plot(x, y, ':', label='pathAdg')
            ax.legend()
            '''
            # TODO ^^

        return path

    def astar(self, start, goal, n_verts):

        open_lst = [start]
        closed_lst = []
        is_complete = False
        count = 0
        while len(open_lst) > 0 and not is_complete: #and count < n_verts + 1:
            count += 1
            min_f = np.infty
            min_idx = None
            for i, node in enumerate(open_lst):
                if node.f < min_f:
                    min_idx = i
                    min_f = node.f
            q = open_lst.pop(min_idx)

            for child in q.children:

                if child == goal:
                    child.parent = q
                    is_complete = True
                    break

                if child not in open_lst and child not in closed_lst:
                    child.h = np.sqrt(
                        (child.location[0] - goal.location[0]) ** 2 + (
                                child.location[1] - goal.location[1]) ** 2)
                    # child.h = dst_to_goal
                    child.g = q.g + child.dst_to_par  # distance to reach parent node
                    child.f = child.h + child.g

                    child.parent = q
                    open_lst.append(child)

            closed_lst.append(q)

        # build the path
        path = []
        current_node = goal
        if goal.parent is not None:
            path_count = 0
            while current_node.parent is not None and path_count < n_verts + 1:
                path_count += 1
                path.append(current_node)
                current_node = current_node.parent
            # if len(path) == 0:

            # add the start node to the path
            path.append(start)

        return path, count


    def draw_at_time(self,ax, data, sim_time):
        pass

    def draw_angle(self,ax, data, sim_time):
        ax.plot(data['sim_time'], data['angle'], label='dest')

    def draw_distance(self,ax, data, sim_time):
        ax.plot(data['sim_time'], data['distance'], label='dest')

    def draw_traj(self,ax, data, sim_time):

        # get the prm path.
        n_rows = len(data)-1
        row = data.iloc[n_rows]
        col_names = data.columns
        x_str = 'path_x'
        y_str = 'path_y'
        n_x = len([i for i in col_names if x_str in i])
        x = np.zeros((n_x,))
        y = np.zeros_like(x)
        for i in range(n_x):
            x[i] = data[x_str+'_'+str(i)].iloc[n_rows]
            y[i] = data[y_str + '_' + str(i)].iloc[n_rows]

        ax.plot(x,y,'o-',color='black')
        ax.plot(x[0],y[0],'rx')
        ax.plot(x[len(x)-1], y[len(y)-1], 'm^')
        #ax.plot(data['angle'].iloc[n_rows],)

class VertexPRM():

    def __init__(self, location, state=None):
        self.children = []
        self.parent = None
        self.location = location
        self.state = state
        self.dst_to_par = None
        self.g = 0
        self.h = 0
        self.f = 0
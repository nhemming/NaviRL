"""
Action operations are the functions that convert raw neural network outputs to the action consumed by the
entities to update themselves.
"""
# native modules
from abc import ABC, abstractmethod
from collections import OrderedDict
import itertools


# 3rd party modules
import numpy as np
from scipy.special import comb

# own modules


class ActionOperation(ABC):

    def __init__(self,action_options_dict, controller, frequency, is_continuous, name, number_controls, output_range=None):

        if is_continuous:
            # TODO need to test
            self.action_bounds = []
            for name, value in action_options_dict.items():
                self.action_bounds.append([float(i) for i in value.split(",")])
        else:
            # reshape to vector
            action_option_vals = []
            for key, value in action_options_dict.items():
                action_option_vals.append([float(i) for i in value.split(',')])

            self.action_options = list(itertools.product(*action_option_vals))

        self.controller = controller
        self.frequency = frequency
        self.is_continuous = is_continuous
        self.name = name

        if is_continuous:
            self.output_range = [float(i) for i in output_range.split(",")]
        else:
            self.output_range = None
        self.num_controls = number_controls

    @abstractmethod
    def convert_action(self, action_vector, delta_t, entities, sensors):
        pass


class DirectVectorControl(ActionOperation):

    def __init__(self, action_options_dict, controller, frequency, is_continuous, name, number_controls, output_range=None):
        super(DirectVectorControl, self).__init__(action_options_dict,controller,frequency,is_continuous,name, number_controls,output_range)

    def convert_action(self, action_vector, delta_t, entities, sensors):
        """
        Simple scaling of outputs of the network to the dimensions of the action control
        :param action_vector:
        :return:
        """

        if self.is_continuous:
            transformed_action = np.zeros_like(action_vector)
            for i, action in enumerate(action_vector):
                transformed_action[i] = (action - self.output_range[0]) * (self.action_bounds[0][1]-self.action_bounds[0][0]) / (self.output_range[1]-self.output_range[0]) + self.action_bounds[0][0]
        else:
            # convert index to case
            transformed_action = self.action_options[action_vector][0]

        return transformed_action

    def setPersistentInfo(self,entities,sensors):
        pass


class BSplineControl(ActionOperation):

    def __init__(self, action_options_dict, controller, frequency, is_continuous, name, number_controls,output_range,segment_length, target_entity):
        """
        Action operation that uses a b spline to generate local paths for the agent to follow. A controller converts
        the path into actuations.

        :param action_options_dict:
        :param frequency:
        :param is_continuous:
        :param name:
        :param number_controls:
        :param output_range:
        """
        super(BSplineControl, self).__init__(action_options_dict, controller, frequency, is_continuous, name, number_controls,
                                                  output_range)

        self.segment_length = segment_length # the length of the segments between the control points that define the b-spline
        self.n_samples = 10  # number of points along the bspline path used for navigating
        self.target_entity = target_entity

        # save information for refreshing.
        self.start_location = []
        self.start_angle = None

    def convert_action(self, action_vector, delta_t, entities, sensors):

        if self.is_continuous:
            # TODO Don't know if this block is needed. May need scaling.
            path_angles = action_vector
        else:
            # discrete
            path_angles = self.action_options[action_vector]

        # build the b spline curve. from the called out angles
        control_points = np.zeros((len(path_angles)+1,2))
        control_points[0,:] = np.array(self.start_location)
        cp_angle = np.zeros(len(path_angles))
        cp_angle[0] = self.start_angle + path_angles[0]
        for i in range(len(path_angles)):
            control_points[i+1,0] = control_points[i,0]+self.segment_length*np.cos(cp_angle[i])
            control_points[i+1, 1] = control_points[i, 1] + self.segment_length * np.sin(cp_angle[i])
            if i < len(path_angles)-1:
                cp_angle[i+1] = cp_angle[i] + path_angles[i+1]

        samples = self.bezier_curve( control_points, self.n_samples)

        # get the point that is nearest to the agent, then advance one
        idx = 0
        min_dst = np.infty

        # get the current position of the entity
        entity_x = entities[self.target_entity].state_dict['x_pos']
        entity_y = entities[self.target_entity].state_dict['y_pos']
        for i, samp in enumerate(samples):
            tmp_dst = np.sqrt( (samp[0]-entity_x)**2 + (samp[1]-entity_y)**2)
            if tmp_dst < min_dst:
                min_dst = tmp_dst
                idx = i
        # increment the index by 1 if possible.
        if idx < len(samples)-1:
            idx += 1

        # use the controller to produce a change to the agent
        heading = entities[self.target_entity].state_dict['phi']

        rot_mat = [[np.cos(heading), np.sin(heading), 0.0],
                   [-np.sin(heading), np.cos(heading), 0.0],
                   [0.0, 0.0, 1.0]]
        rot_mat = np.reshape(rot_mat, (3, 3))

        diff = np.subtract([samples[idx,0],samples[idx,1],samples[idx,2]], [entity_x,entity_y,heading])

        error_vec = np.matmul(rot_mat, diff)

        v_mag = np.sqrt(entities[self.target_entity].state_dict['v_mag'])
        command = self.controller.get_command(delta_t,error_vec,v_mag)

        # return the transfromed action
        return command

    def bernstein_poly(self, i, n, t):
        """
        helper that generates bernstein polynomials
        """
        return comb(n, i) * (np.power(t, (n - i))) * np.power(1.0 - t, i)

    def bezier_curve(self, control_points, n_samples):
        """
        Given a list of (x, y) points that serve as control points, a path is generated. The path is created with
        n_samples of points to define the path.

        :param control_points: an array of x,y control points that define a bezier curve
        :param n_samples: the number of samples to collect spanning 0 to 1
        """

        n_cp = len(control_points)

        t = np.linspace(0.0, 1.0, n_samples)

        pa = np.zeros((n_cp, n_samples))
        for i in range(n_cp):
            pa[i, :] = (self.bernstein_poly(i, n_cp - 1, t))
        pa = np.reshape(pa, (len(pa), len(pa[0])))

        samples = np.zeros((n_samples, 4))
        samples[:, 0] = np.dot(control_points[:, 0], pa)  # x
        samples[:, 1] = np.dot(control_points[:, 1], pa)  # y

        samples = np.flip(samples, axis=0)

        # calculate angles
        for i in range(len(samples)):
            if i == len(samples) - 1:
                tmp_vec = [samples[i, 0] - samples[i - 1, 0], samples[i, 1] - samples[i - 1, 1]]
            elif i == 0:
                tmp_vec = [samples[i + 1, 0] - samples[i, 0], samples[i + 1, 1] - samples[i, 1]]
            else:
                tmp_vec = [samples[i + 1, 0] - samples[i - 1, 0], samples[i + 1, 1] - samples[i - 1, 1]]

            samples[i, 2] = np.arctan2(tmp_vec[1], tmp_vec[0])

        # samples columns = x, y, theta, ?
        return samples

    def setPersistentInfo(self,entities,sensors):

        # save the root of the bsline to build the spline from
        self.start_location = [entities[self.target_entity].state_dict['x_pos'],entities[self.target_entity].state_dict['y_pos']]
        self.start_angle = entities[self.target_entity].state_dict['phi']

        presistentInfo = OrderedDict()
        presistentInfo['x_init'] = self.start_location[0]
        presistentInfo['y_init'] = self.start_location[1]
        presistentInfo['phi_init'] = self.start_angle
        return presistentInfo


class DubinsControl(ActionOperation):

    def __init__(self, action_bounds_dict, frequency, is_continous, name, number_controls, output_range):
        super(DubinsControl, self).__init__(action_bounds_dict, frequency, is_continous, name, number_controls,
                                                  output_range)

    def convert_action(self, action_vector):
        pass

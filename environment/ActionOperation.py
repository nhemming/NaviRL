"""
Action operations are the functions that convert raw neural network outputs to the action consumed by the
entities to update themselves.
"""
# native modules
from abc import ABC, abstractmethod
import itertools


# 3rd party modules
import numpy as np
from scipy.special import comb

# own modules


class ActionOperation(ABC):

    def __init__(self,action_options_dict, frequency, is_continuous, name, number_controls, output_range=None):

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

        self.frequency = frequency
        self.is_continuous = is_continuous
        self.name = name

        if is_continuous:
            self.output_range = [float(i) for i in output_range.split(",")]
        else:
            self.output_range = None
        self.num_controls = number_controls

    @abstractmethod
    def convert_action(self, action_vector):
        pass


class DirectVectorControl(ActionOperation):

    def __init__(self, action_options_dict, frequency, is_continuous, name, number_controls, output_range=None):
        super(DirectVectorControl, self).__init__(action_options_dict,frequency,is_continuous,name, number_controls,output_range)

    def convert_action(self, action_vector):
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


class BSplineControl(ActionOperation):

    def __init__(self, action_options_dict, frequency, is_continuous, name, number_controls, output_range):
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
        super(BSplineControl, self).__init__(action_options_dict, frequency, is_continuous, name, number_controls,
                                                  output_range)

    def convert_action(self, action_vector):
        pass

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

        # calculate angles
        for i in range(len(samples)):
            if i == len(samples) - 1:
                tmp_vec = [samples[i, 0] - samples[i - 1, 0], samples[i, 1] - samples[i - 1, 1]]
            elif i == 0:
                tmp_vec = [samples[i + 1, 0] - samples[i, 0], samples[i + 1, 1] - samples[i, 1]]
            else:
                tmp_vec = [samples[i + 1, 0] - samples[i - 1, 0], samples[i + 1, 1] - samples[i - 1, 1]]

            samples[i, 2] = np.arctan2(tmp_vec[1], tmp_vec[0])

        samples = np.flip(samples, axis=0)

        return samples


class DubinsControl(ActionOperation):

    def __init__(self, action_bounds_dict, frequency, is_continous, name, number_controls, output_range):
        super(DubinsControl, self).__init__(action_bounds_dict, frequency, is_continous, name, number_controls,
                                                  output_range)

    def convert_action(self, action_vector):
        pass


class PointControl(ActionOperation):

    def __init__(self, action_bounds_dict, frequency, is_continous, name, number_controls, output_range):
        super(PointControl, self).__init__(action_bounds_dict, frequency, is_continous, name, number_controls,
                                                  output_range)

    def convert_action(self, action_vector):
        pass

"""
Defines PD controller for producing actions.
"""

# native packages
from abc import ABC, abstractmethod
from collections import namedtuple

# 3rd party packages
import numpy as np

# own packages


class Controller:

    def __init__(self, coeffs: namedtuple):
        self.coeffs = coeffs
        self.old_error = [0.0,0.0,0.0]
        self.error = [0.0,0.0,0.0]

    @abstractmethod
    def get_command(self, dt, error, v_mag):
        """
        given the time step and current state information, produce a change.

        :param dt: time step size [s]
        :param state: a dictionary containing the current state information
        :return: output change
        """
        pass


class PDController(Controller):

    def __init__(self,coeffs):
        super(PDController, self).__init__(coeffs)

    def get_command(self, dt, error, v_mag):

        y_dot = (error[1] - self.old_error[1]) / dt

        prop = self.coeffs.p * error[1] / v_mag
        deriv = self.coeffs.d * y_dot

        command = prop + deriv

        self.old_error = error

        return command
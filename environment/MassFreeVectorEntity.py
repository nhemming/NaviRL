"""
A basic agent is an agent that is a massless point that can move anywhere in 2 dimensions
"""

# native modules

# 3rd party modules
import numpy as np

# own modules
from environment.Entity import CollideEntity


class MassFreeVectorEntity(CollideEntity):

    def __init__(self, collision_shape, id, name):
        super(MassFreeVectorEntity, self).__init__( collision_shape, id, name)

        self.state_dict['phi'] = 0.0  # heading in global coordinates [rad]
        self.state_dict['velocity'] = 1.0  # velocity the agent moves at in the simulation [m/s]

    def step(self, delta_t):

        dx = self.state_dict['velocity'] * np.cos(self.state_dict['phi']) * delta_t
        dy = self.state_dict['velocity'] * np.sin(self.state_dict['phi']) * delta_t

        self.state_dict['x_pos'] += dx
        self.state_dict['y_pos'] += dy

    def set_heading(self, phi):
        # correct angle to be between 0 and pi
        if phi > 2.0*np.pi:
            phi -= 2.0*np.pi
        if phi < 0:
            phi += 2.0*np.pi
        self.state_dict['phi'] = phi

    def reset(self):
        # reset the heading to a random vector
        self.state_dict['phi'] = np.random.uniform(low=0, high=2.0*np.pi)


    def apply_action(self, action_vec):
        self.set_heading(action_vec)
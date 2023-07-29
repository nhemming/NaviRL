"""
A basic agent is an agent that is a massless point that can move anywhere in 2 dimensions
"""

# native modules

# 3rd party modules
import matplotlib.pyplot as plt
import numpy as np

# own modules
from environment.Entity import CollideEntity, CollisionCircle, CollisionRectangle


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
        self.set_heading(action_vec+self.state_dict['phi'])

    def draw_trajectory(self, ax, data, sim_time):

        # draw trajectory
        ax.plot(data['x_pos'],data['y_pos'])

        # draw shape
        if isinstance(self.collision_shape,CollisionCircle):
            row = data.loc[data['sim_time'] == sim_time]
            circle = plt.Circle((row['x_pos'],row['y_pos']),radius=self.collision_shape.radius,alpha=0.3)
            ax.add_patch(circle)

    def draw_telemetry_trajectory(self, ax, data, sim_time):
        ax.plot(data['sim_time'],data['x_pos'],label='X')
        ax.plot(data['sim_time'], data['y_pos'], label='Y')
        ax.legend()

    def draw_telemetry_heading(self, ax, data, sim_time):
        ax.plot(data['sim_time'],data['phi'],label='X')

    def draw_telemetry_velocity(self, ax, data, sim_time):
        ax.plot(data['sim_time'],data['velocity'],label='X')
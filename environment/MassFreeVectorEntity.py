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
        self.state_dict['v_mag'] = 1.0  # velocity the agent moves at in the simulation [m/s]

    def step(self, delta_t):
        """
        Use Euler stepping to step the entity one step in time.
        :param delta_t: Time step [s] to step over.
        :return:
        """

        dx = self.state_dict['v_mag'] * np.cos(self.state_dict['phi']) * delta_t
        dy = self.state_dict['v_mag'] * np.sin(self.state_dict['phi']) * delta_t

        self.state_dict['x_pos'] += dx
        self.state_dict['y_pos'] += dy

    def adj_heading(self, phi_adj):
        """
        Changes the heading (phi) of the entity by the passed in amount.
        :param phi_adj: The amount [rad] to change the heading.
        :return:
        """

        if isinstance(phi_adj,np.ndarray):
            phi_adj = phi_adj[0]

        # correct for the bounds of the change
        if phi_adj > np.deg2rad(45.0):
            phi_adj = np.deg2rad(45.0)
        elif phi_adj < -np.deg2rad(45.0):
            phi_adj = -np.deg2rad(45.0)

        phi = self.state_dict['phi'] + phi_adj

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
        """
        Accepts a vector produced by a neural network. It may not be the direct output, as the actions may have been
        mutated. This function dispatches the vector to the appropriate model changes.
        :param action_vec: Vector describing the value of change for the actuators.
        :return:
        """

        # adjust the heading of the entity
        self.adj_heading(action_vec)

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
        ax.plot(data['sim_time'],data['v_mag'],label='X')
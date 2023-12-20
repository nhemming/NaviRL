"""
A basic agent is an agent that is a massless point that can move anywhere in 2 dimensions
"""

# native modules

# 3rd party modules
import matplotlib.pyplot as plt
import numpy as np

# own modules
from environment.Entity import CollideEntity, CollisionCircle, CollisionRectangle, Entity


class StaticEntity(Entity):

    def __init__(self, id, name, shape):
        super(StaticEntity, self).__init__(id, name)

        self.shape = shape

    def step(self, delta_t):
        # do nothing
        pass

    def reset(self):
        # do nothing
        pass

    def reset_random(self):
        # do nothing
        pass

    def draw_trajectory(self, ax, data, sim_time):
        # draw trajectory
        ax.plot(data['x_pos'], data['y_pos'])

        # draw shape
        #if isinstance(self.collision_shape, CollisionCircle):
        row = data.loc[data['sim_time'] == sim_time]
        circle = plt.Circle((row['x_pos'], row['y_pos']), radius=self.shape.radius, color='tab:green',alpha=0.3)
        ax.add_patch(circle)

    def draw_telemetry_trajectory(self, ax, data, sim_time):
        pass

    def draw_telemetry_heading(self, ax, data, sim_time):
        pass

    def draw_telemetry_velocity(self, ax, data, sim_time):
        pass


class StaticEntityCollide(CollideEntity):

    def __init__(self, collision_shape, id, name):
        super(StaticEntityCollide, self).__init__(collision_shape, id, name)

    def step(self, delta_t):
        # do nothing
        pass

    def reset(self):
        # do nothing
        pass

    def draw_trajectory(self, ax, data, sim_time):
        # draw trajectory
        ax.plot(data['x_pos'], data['y_pos'])

        # draw shape
        if isinstance(self.collision_shape, CollisionCircle):
            row = data.loc[data['sim_time'] == sim_time]
            circle = plt.Circle((row['x_pos'], row['y_pos']), radius=self.collision_shape.radius, color='tab:green',alpha=0.3)
            ax.add_patch(circle)

    def draw_telemetry_trajectory(self, ax, data, sim_time):
        pass

    def draw_telemetry_heading(self, ax, data, sim_time):
        pass

    def draw_telemetry_velocity(self, ax, data, sim_time):
        pass
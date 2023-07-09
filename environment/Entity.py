"""
Houses the base entity classes. All items in the simulation are build upon these abstract classes.
"""

import numpy as np


def get_angle_between_vectors( v1, v2, keep_sign):
    """
    uses the dot product or cross product to get the angle between two vectors. 2D vectors are assumed for this

    :param v1: first vector
    :param v2: second vector
    :param keep_sign: boolean for if the sign of the angle should be maintained or ignored
    :return: angle between vectors 1 and 2 [rad]
    """
    if keep_sign:
        # the sign the angle matters
        angle = np.arctan2(v2[1] * v1[0] - v2[0] * v1[1], v1[0] * v2[0] + v1[1] * v2[1])
    else:
        # the sign should be ignored
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    return angle


class Entity:

    def __init__(self, id, name):
        self.id = id
        self.name = name
        self.state_dict = dict()
        self.history = None
        self.is_active = False

    def add_step_history(self):
        pass

    def reset_history(self):
        pass

    def reset(self):
        pass


class CollideEntity(Entity):

    def __init__(self, collision_shape, id, name):

        super(CollideEntity, self).__init__(id, name)
        self.collision_shape = collision_shape

        # add cg position of the entity
        self.state_dict['x_pos'] = 0.0
        self.state_dict['y_pos'] = 0.0
        self.state_dict['z_pos'] = 0.0


class CollisionCircle:

    def __init__(self, heading, radius):
        self.heading = heading
        self.radius = radius


class CollisionRectangle:

    def __init__(self, heading, height, width):
        self.heading = heading
        self.height = height
        self.width = width

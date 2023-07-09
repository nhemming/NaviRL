"""
A basic agent is an agent that is a massless point that can move anywhere in 2 dimensions
"""

# native modules

# 3rd party modules
import numpy as np

# own modules
from environment.Entity import CollideEntity


class StaticEntity(CollideEntity):

    def __init__(self, collision_shape, id, name):
        super(StaticEntity, self).__init__(collision_shape, id, name)

    def step(self, delta_t):
        # do nothing
        pass

    def reset(self):
        # do nothing
        pass

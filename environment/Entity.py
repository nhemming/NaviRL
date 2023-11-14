"""
Houses the base entity classes. All items in the simulation are build upon these abstract classes.
"""

# native modules
from abc import ABC, abstractmethod
import copy
import os

# 3rd party modules
import numpy as np
import pandas as pd

# own modules


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


def get_collision_status(entity_1, entity_2):
    """
    Returns a boolean for if the two entities have collided. Returns true if collided, and false if not collided.

    :param entity_1: The first simulation entity to be checked.
    :param entity_2:  The second simulation entity to be checked.
    :return: boolean for collision status.
    """

    if isinstance(entity_1,CollideEntity) and isinstance(entity_2,CollideEntity):

        entity_1.collision_shape.update_shape_definition(entity_1.state_dict['phi'],entity_1.state_dict['x_pos'],entity_1.state_dict['y_pos'])
        entity_2.collision_shape.update_shape_definition(entity_2.state_dict['phi'], entity_2.state_dict['x_pos'],
                                         entity_2.state_dict['y_pos'])

        if isinstance(entity_1.collision_shape, CollisionCircle) and isinstance(entity_2.collision_shape, CollisionCircle):
            # both objects are circles. Measure the distances minus the radiuses to check for collision

            dx = entity_1.collision_shape.center_x - entity_2.collision_shape.center_x
            dy = entity_1.collision_shape.center_y - entity_2.collision_shape.center_y
            #dz = entity_1.state_dict['z_pos'] - entity_2.state_dict['z_pos']
            #dst = np.sqrt(dx*dx + dy*dy + dz*dz)
            dst = np.sqrt(dx * dx + dy * dy)

            if dst < (entity_1.collision_shape.radius + entity_2.collision_shape.radius):
                # collision
                return True
            return False

        elif isinstance(entity_1.collision_shape, CollisionRectangle) and isinstance(entity_2.collision_shape, CollisionRectangle):
            # both are rectangles
            if rect_to_rect_helper(entity_1, entity_2):
                return True
            if rect_to_rect_helper(entity_2, entity_1):
                return True
            return False

        else:

            # one is a rectangle and one is a circle
            if isinstance(entity_1.collision_shape, CollisionRectangle):
                phi = entity_1.state_dict['phi']
                rectangle = entity_1
                rectangle_shape = entity_1.collision_shape
                circle = entity_2
                circle_shape = entity_2.collision_shape
            else:
                phi = entity_2.state_dict['phi']
                rectangle = entity_2
                rectangle_shape = entity_2.collision_shape
                circle = entity_1
                circle_shape = entity_1.collision_shape

            # rotate circle vector by - angle of rectangle
            x = circle.state_dict['x_pos']-rectangle.state_dict['x_pos']
            y = circle.state_dict['y_pos'] - rectangle.state_dict['y_pos']

            x_circle_p = np.abs(x*np.cos(phi) - y * np.sin(phi))
            y_circle_p = np.abs(x * np.sin(phi) + y * np.cos(phi))

            # check for collision
            if x_circle_p <= rectangle_shape.width/2.0+circle_shape.radius and y_circle_p <= rectangle_shape.height/2.0+circle_shape.radius:
                # collision happended
                return True
            return False
    else:
        return False


def rect_to_rect_helper(entity_1,entity_2):
    """
    Creates a ray from each corner of entity 1 to the cg of entity 2. If the ray intersects an odd number of segments
    the objects have collided. The case that the entities are exactly coincident is not checked. It is assumed
    that will not happen in simulation.

    :param entity_1: Entity that has a rectangular collision box.
    :param entity_2: A different entity that has a rectangular collision box.
    :return: True if a collision has occurred. False if no collision has occurred.
    """

    # cast a ray from each corner of one rectangle to the cg of the other
    # if only one edge intersects, a collision has occurred. Need to check both rectangles

    cg = [entity_2.collision_shape.center_x, entity_2.collision_shape.center_y]
    for i, corner1 in enumerate(entity_1.collision_shape.corners):

        # create ray
        if cg[0] == corner1[0]:
            m_ray = np.infty
            b_ray = None
        else:
            m_ray = (cg[1] - corner1[1]) / (cg[0] - corner1[0])
            b_ray = cg[1] - m_ray * cg[0]

        # check if cg is coincident with the corner. If so, they are collided
        if cg[0] == corner1[0] and cg[1] == corner1[1]:
            return True

        # count intersections of the ray and the second entities edges.
        n_intersections = 0
        for j, corner2 in enumerate(entity_2.collision_shape.corners):

            x2a = corner2[0]
            y2a = corner2[1]

            x2b = entity_2.collision_shape.corners[(j + 1) % 4][0]
            y2b = entity_2.collision_shape.corners[(j + 1) % 4][1]



            if x2a == x2b:
                m = np.infty
                b = None
            else:
                m = (y2b - y2a) / (x2b - x2a)
                b = y2b - m * x2b

            if m == m_ray:
                # parallel lines
                pass
            else:

                if np.isinf(m):
                    # edge of entity 2 is vertical
                    x_intercept = x2a
                    y_intercept = m_ray * x_intercept + b_ray

                elif np.isinf(m_ray):
                    x_intercept = cg[0]
                    y_intercept = m * x_intercept + b

                else:

                    x_intercept = (b - b_ray) / (m_ray - m)
                    y_intercept = m_ray * x_intercept + b_ray

                angle = get_angle_between_vectors([corner1[0] - cg[0], corner1[1] - cg[1]],
                                                  [corner1[0] - x_intercept, corner1[1] - y_intercept], True)
                if np.abs(angle) <= 1e-3:
                    # vectors pointing in the same direction. Can count the intersection
                    if (x2a <= x_intercept <= x2b or x2a >= x_intercept >= x2b) and (
                            y2a <= y_intercept <= y2b or y2a >= y_intercept >= y2b):
                        n_intersections += 1

        if n_intersections % 2 != 0:
            # collision has happened
            return True
    # no collision has happened
    return False


class Entity:

    def __init__(self, id, name):
        self.id = id
        self.name = name
        self.state_dict = {'sim_time' : None}

        # add cg position of the entity
        self.state_dict['x_pos'] = 0.0
        self.state_dict['y_pos'] = 0.0
        self.state_dict['z_pos'] = 0.0

        # add heading
        self.state_dict['phi'] = 0.0

        self.history = []
        self.is_active = False

    def add_step_history(self, sim_time):
        """
        save the state information for the current time steo
        :return:
        """
        self.state_dict['sim_time'] = sim_time
        self.history.append(copy.deepcopy(self.state_dict))

    def reset_history(self):
       self.history = []

    def write_history(self, episode_number, file_path, eval_num=''):
        # TODO change from csv to sqlite data base
        # write history to csv
        df = pd.DataFrame(self.history)
        file_path = os.path.join(file_path, 'entities')
        df.to_csv(os.path.abspath(os.path.join(file_path,str(self.name)+'_epnum-'+str(episode_number)+eval_num+'.csv')), index=False)

    def reset_base(self):
        self.reset()
        self.reset_history()

    def reset(self):
        pass


class CollideEntity(Entity):

    def __init__(self, collision_shape, id, name):

        super(CollideEntity, self).__init__(id, name)
        self.collision_shape = collision_shape

        # collision status with other entites
        self.state_dict['is_collided'] = False

    def reset_base(self):
        self.state_dict['is_collided'] = False
        self.history = []
        self.reset()


class CollisionShape(ABC):

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def update_shape_definition(self, heading, x, y):
        pass


class CollisionCircle(CollisionShape):

    def __init__(self, heading, name, radius):
        super(CollisionCircle, self).__init__(name)
        self.heading = heading
        self.radius = radius

        # the center location [m]
        self.center_x = None
        self.center_y = None

    def update_shape_definition(self, heading, x, y):
        self.center_x = x
        self.center_y = y

        self.heading = heading


class CollisionRectangle(CollisionShape):

    def __init__(self, heading, height, name, width):
        super(CollisionRectangle, self).__init__(name)
        self.heading = heading
        self.height = height
        self.width = width

        # the center location [m]
        self.center_x = None
        self.center_y = None

        # points move CCW (right hand rule) from heading vector
        self.corners = np.zeros((4,2))

    def update_shape_definition(self, heading, x, y):
        self.center_x = x
        self.center_y = y
        self.heading = heading

        org_vector = [self.width / 2.0, self.height / 2.0]

        # first corner
        #c1_angle = np.arctan2(org_vector[1],org_vector[0])
        self.corners[0,0] = org_vector[0]*np.cos(self.heading) - org_vector[1]*np.sin(self.heading) + self.center_x
        self.corners[0, 1] = org_vector[0] * np.sin(self.heading) + org_vector[1] * np.cos(self.heading) + self.center_y

        # second corner
        #c2_angle = np.arctan2(-org_vector[1], org_vector[0])
        self.corners[1, 0] = -org_vector[0] * np.cos(self.heading) - org_vector[1] * np.sin(self.heading) + self.center_x
        self.corners[1, 1] = -org_vector[0] * np.sin(self.heading) + org_vector[1] * np.cos(self.heading) + self.center_y

        # third corner
        #c3_angle = np.arctan2(-org_vector[1], -org_vector[0])
        self.corners[2, 0] = -org_vector[0] * np.cos(self.heading) + org_vector[1] * np.sin(self.heading) + self.center_x
        self.corners[2, 1] = -org_vector[0] * np.sin(self.heading) - org_vector[1] * np.cos(self.heading) + self.center_y

        # fourth corner
        #c4_angle = np.arctan2(org_vector[1], -org_vector[0])
        self.corners[3, 0] = org_vector[0] * np.cos(self.heading) + org_vector[1] * np.sin(self.heading) + self.center_x
        self.corners[3, 1] = org_vector[0] * np.sin(self.heading) - org_vector[1] * np.cos(self.heading) + self.center_y





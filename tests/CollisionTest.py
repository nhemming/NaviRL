"""
Unit tests for verifying the collisions are calculated correctly.
"""

# native modules
from unittest import TestCase

# 3rd party modules
import numpy as np

# own modules
from environment.Entity import CollisionCircle, CollisionRectangle, get_collision_status
from environment.StaticEntity import StaticEntity


class CollisionTest(TestCase):

    def test_no_collision_two_circles(self):
        """
        Place two entities with circle collision bounds that are not collided and verify the collision status is false
        :return:
        """

        circle_1 = CollisionCircle(0.0, 'circ_0', 1.0)
        entity1 = StaticEntity(circle_1, 0, 'circ_0')
        entity1.state_dict['x_pos'] = 0.0
        entity1.state_dict['y_pos'] = 0.0

        circle_2 = CollisionCircle(0.0, 'circ_1', 1.0)
        entity2 = StaticEntity(circle_2, 0, 'circ_1')
        entity2.state_dict['x_pos'] = 3.0
        entity2.state_dict['y_pos'] = 0.0

        # should not be a collision
        self.assertFalse(get_collision_status(entity1, entity2))

    def test_collision_two_circles(self):
        """
        Place two entities with circle collision bounds that are collided and verify the collision status is true
        :return:
        """

        circle_1 = CollisionCircle(0.0, 'circ_0', 1.0)
        entity1 = StaticEntity(circle_1, 0, 'circ_0')
        entity1.state_dict['x_pos'] = 0.0
        entity1.state_dict['y_pos'] = 0.0

        circle_2 = CollisionCircle(0.0, 'circ_1', 1.0)
        entity2 = StaticEntity(circle_2, 1, 'circ_1')
        entity2.state_dict['x_pos'] = 1.9
        entity2.state_dict['y_pos'] = 0.0

        # should not be a collision
        self.assertTrue(get_collision_status(entity1, entity2))

    def test_no_collision_two_rectangles(self):
        """
        Tests to check collisions did not happen between two rectangular collision objects
        :return:
        """

        rectangle_1 = CollisionRectangle(0.0, 1.0, 'rect_1', 2.0)
        entity1 = StaticEntity(rectangle_1, 0, 'rect_1')
        entity1.state_dict['x_pos'] = 0.0
        entity1.state_dict['y_pos'] = 0.0

        rectangle_2 = CollisionRectangle(0.0, 1.0, 'rect_2', 2.0)
        entity2 = StaticEntity(rectangle_2, 1, 'rect_2')
        entity2.state_dict['x_pos'] = 4.0
        entity2.state_dict['y_pos'] = 0.5

        self.assertFalse(get_collision_status(entity1,entity2))

        rectangle_1 = CollisionRectangle(0.0, 1.0, 'rect_1', 2.0)
        entity1 = StaticEntity(rectangle_1, 0, 'rect_1')
        entity1.state_dict['x_pos'] = 0.0
        entity1.state_dict['y_pos'] = 0.0

        rectangle_2 = CollisionRectangle(0.0, 1.0, 'rect_2', 2.0)
        entity2 = StaticEntity(rectangle_2, 1, 'rect_2')
        entity2.state_dict['x_pos'] = 0.5
        entity2.state_dict['y_pos'] = 4.0

        self.assertFalse(get_collision_status(entity1, entity2))

        rectangle_1 = CollisionRectangle(0.0, 1.0, 'rect_1', 2.0)
        entity1 = StaticEntity(rectangle_1, 0, 'rect_1')
        entity1.state_dict['x_pos'] = 0.0
        entity1.state_dict['y_pos'] = 0.0
        entity1.state_dict['phi'] = -0.1

        rectangle_2 = CollisionRectangle(0.0, 1.0, 'rect_2', 2.0)
        entity2 = StaticEntity(rectangle_2, 1, 'rect_2')
        entity2.state_dict['x_pos'] = 4.0
        entity2.state_dict['y_pos'] = 0.5
        entity2.state_dict['phi'] = 0.1

        self.assertFalse(get_collision_status(entity1, entity2))

        rectangle_1 = CollisionRectangle(0.0, 1.0, 'rect_1', 2.0)
        entity1 = StaticEntity(rectangle_1, 0, 'rect_1')
        entity1.state_dict['x_pos'] = 0.0
        entity1.state_dict['y_pos'] = 0.0
        entity1.state_dict['phi'] = -0.1

        rectangle_2 = CollisionRectangle(0.0, 1.0, 'rect_2', 2.0)
        entity2 = StaticEntity(rectangle_2, 1, 'rect_2')
        entity2.state_dict['x_pos'] = 0.5
        entity2.state_dict['y_pos'] = 4.0
        entity2.state_dict['phi'] = 0.1

        self.assertFalse(get_collision_status(entity1, entity2))

    def test_collision_two_rectangles(self):
        """
        Tests to check collisions did happen between two rectangular collision objects
        :return:
        """

        rectangle_1 = CollisionRectangle(0.0, 1.0, 'rect_1', 2.0)
        entity1 = StaticEntity(rectangle_1, 0, 'rect_1')
        entity1.state_dict['x_pos'] = 0.0
        entity1.state_dict['y_pos'] = 0.0

        rectangle_2 = CollisionRectangle(0.0, 1.0, 'rect_2', 2.0)
        entity2 = StaticEntity(rectangle_2, 1, 'rect_2')
        entity2.state_dict['x_pos'] = 1.0
        entity2.state_dict['y_pos'] = 0.5

        self.assertTrue(get_collision_status(entity1, entity2))

        rectangle_1 = CollisionRectangle(0.0, 1.0, 'rect_1', 2.0)
        entity1 = StaticEntity(rectangle_1, 0, 'rect_1')
        entity1.state_dict['x_pos'] = 0.0
        entity1.state_dict['y_pos'] = 0.0

        rectangle_2 = CollisionRectangle(0.0, 1.0, 'rect_2', 2.0)
        entity2 = StaticEntity(rectangle_2, 1, 'rect_2')
        entity2.state_dict['x_pos'] = 1.1
        entity2.state_dict['y_pos'] = 0.5

        self.assertTrue(get_collision_status(entity1, entity2))

        rectangle_1 = CollisionRectangle(0.0, 1.0, 'rect_1', 2.0)
        entity1 = StaticEntity(rectangle_1, 0, 'rect_1')
        entity1.state_dict['x_pos'] = 0.0
        entity1.state_dict['y_pos'] = 0.0
        entity2.state_dict['phi'] = -0.1

        rectangle_2 = CollisionRectangle(0.0, 1.0, 'rect_2', 2.0)
        entity2 = StaticEntity(rectangle_2, 1, 'rect_2')
        entity2.state_dict['x_pos'] = 1.1
        entity2.state_dict['y_pos'] = 0.5
        entity2.state_dict['phi'] = 0.1

        self.assertTrue(get_collision_status(entity1, entity2))

    def test_no_collision_one_circle_one_rectangle(self):
        """
        tests that collisions do not happen when a circle and rectangle are farther apart
        :return:
        """

        rectangle_1 = CollisionRectangle(0.0, 1.0, 'rect_1', 2.0)
        entity1 = StaticEntity(rectangle_1, 0, 'rect_1')
        entity1.state_dict['x_pos'] = 0.0
        entity1.state_dict['y_pos'] = 0.0
        entity1.state_dict['phi'] = 0.01

        circle_1 = CollisionCircle(0.0, 'circ_0', 1.0)
        entity2 = StaticEntity(circle_1, 1, 'circ_0')
        entity2.state_dict['x_pos'] = 4.0
        entity2.state_dict['y_pos'] = 0.0

        self.assertFalse(get_collision_status(entity1, entity2))

        angles = [0.0, np.pi/2.0, np.pi, 3.0*np.pi/2.0]
        for angle in angles:
            entity1.state_dict['phi'] = angle
            self.assertFalse(get_collision_status(entity1, entity2))

        entity1.state_dict['phi'] = 0.0
        for angle in angles:
            entity2.state_dict['x_pos'] = 4.0*np.cos(angle)
            entity2.state_dict['y_pos'] = 4.0 * np.sin(angle)
            self.assertFalse(get_collision_status(entity1, entity2))

    def test_collision_one_circle_one_rectangle(self):
        """
        tests that collisions do not happen when a circle and rectangle are farther apart
        :return:
        """

        rectangle_1 = CollisionRectangle(0.0, 1.0, 'rect_1', 2.0)
        entity1 = StaticEntity(rectangle_1, 0, 'rect_1')
        entity1.state_dict['x_pos'] = 0.0
        entity1.state_dict['y_pos'] = 0.0
        entity1.state_dict['phi'] = 0.01

        circle_1 = CollisionCircle(0.0, 'circ_0', 1.0)
        entity2 = StaticEntity(circle_1, 1, 'circ_0')
        entity2.state_dict['x_pos'] = 0.5
        entity2.state_dict['y_pos'] = 0.0

        self.assertTrue(get_collision_status(entity1, entity2))

        angles = [0.0, np.pi/2.0, np.pi, 3.0*np.pi/2.0]
        for angle in angles:
            entity1.state_dict['phi'] = angle
            self.assertTrue(get_collision_status(entity1, entity2))

        entity1.state_dict['phi'] = 0.0
        for angle in angles:
            entity2.state_dict['x_pos'] = 0.5*np.cos(angle)
            entity2.state_dict['y_pos'] = 0.5 * np.sin(angle)
            self.assertTrue(get_collision_status(entity1, entity2))
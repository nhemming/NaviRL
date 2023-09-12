"""
Action operations are the functions that convert raw neural network outputs to the action consumed by the
entities to update themselves.
"""
# native modules
import copy
from abc import ABC, abstractmethod
from collections import OrderedDict
import itertools


# 3rd party modules
import numpy as np
from scipy.special import comb

# own modules
from environment.Entity import get_angle_between_vectors


class ActionOperation(ABC):

    def __init__(self,action_options_dict, controller, frequency, is_continuous, name, number_controls):

        if is_continuous:
            # TODO need to test
            self.action_bounds = []
            for name, value in action_options_dict.items():
                if isinstance(value, str):
                    self.action_bounds.append([float(i) for i in value.split(",")])
                else:
                    self.action_bounds.append([value])
        else:
            # reshape to vector
            action_option_vals = []
            for key, value in action_options_dict.items():
                if isinstance(value,str):
                    action_option_vals.append([float(i) for i in value.split(',')])
                else:
                    action_option_vals.append([value])

            self.action_options = list(itertools.product(*action_option_vals))

        self.controller = controller
        self.frequency = frequency
        self.is_continuous = is_continuous
        self.name = name

        self.num_controls = number_controls
        self.output_range = [-1,1] # This assumes all actors or continuous agents have a tanh ending activation function.

    @abstractmethod
    def convert_action(self, action_vector, delta_t, entities, sensors):
        pass


class DirectVectorControl(ActionOperation):

    def __init__(self, action_options_dict, controller, frequency, is_continuous, name, number_controls):
        super(DirectVectorControl, self).__init__(action_options_dict,controller,frequency,is_continuous,name, number_controls)

    def convert_action(self, action_vector, delta_t, entities, sensors):
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

    def setPersistentInfo(self,entities,sensors):
        pass


class BSplineControl(ActionOperation):

    def __init__(self, action_options_dict, controller, frequency, is_continuous, name, number_controls,output_range,segment_length, target_entity):
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
        super(BSplineControl, self).__init__(action_options_dict, controller, frequency, is_continuous, name, number_controls)

        self.segment_length = segment_length # the length of the segments between the control points that define the b-spline
        self.n_samples = 10  # number of points along the bspline path used for navigating
        self.target_entity = target_entity

        # save information for refreshing.
        self.start_location = []
        self.start_angle = None

    def convert_action(self, action_vector, delta_t, entities, sensors):
        """
        Converts the output of a neural network into (an) actuator(s) change. Various methods can be used to make this
        conversion. Here a B-spline path is built from the outputs of a neural network. Then a controller works to
        follow the b-spline path.

        :param action_vector: The vector of outputs from the neural network. In the discrete case, only an integer is
            provided. The integer has the index of the meta data from a combinatorial previously calculated. For the
            continuous case, the changes are directly provided.
        :param delta_t: The time step of the simulation. Needed for the controller to calculate the actuator changes
            for the agents.
        :param entities: An ordered dictionary containing the entities in the simulation.
        :param sensors: An ordered dictionary containing the sensors in a simulation.
        :return: The command that is in the dimensions that are appropriate for the entity to update its control scheme
            with.
        """

        if self.is_continuous:
            # TODO Don't know if this block is needed. May need scaling.
            path_angles = copy.deepcopy(action_vector)
            for i, action in enumerate(action_vector):
                path_angles[i] = (action - self.output_range[0]) * (
                            self.action_bounds[i][1] - self.action_bounds[i][0]) / (
                                               self.output_range[1] - self.output_range[0]) + self.action_bounds[i][0]

        else:
            # discrete
            path_angles = self.action_options[action_vector]

        # build the b spline curve. from the called out angles
        control_points = np.zeros((len(path_angles)+1,2))
        control_points[0,:] = np.array(self.start_location)
        cp_angle = np.zeros(len(path_angles))
        cp_angle[0] = self.start_angle + path_angles[0]
        for i in range(len(path_angles)):
            control_points[i+1,0] = control_points[i,0]+self.segment_length*np.cos(cp_angle[i])
            control_points[i+1, 1] = control_points[i, 1] + self.segment_length * np.sin(cp_angle[i])
            if i < len(path_angles)-1:
                cp_angle[i+1] = cp_angle[i] + path_angles[i+1]

        samples = self.bezier_curve( control_points, self.n_samples)

        # get the point that is nearest to the agent, then advance one
        idx = 0
        min_dst = np.infty

        # get the current position of the entity
        entity_x = entities[self.target_entity].state_dict['x_pos']
        entity_y = entities[self.target_entity].state_dict['y_pos']
        for i, samp in enumerate(samples):
            tmp_dst = np.sqrt( (samp[0]-entity_x)**2 + (samp[1]-entity_y)**2)
            if tmp_dst < min_dst:
                min_dst = tmp_dst
                idx = i
        # increment the index by 1 if possible.
        if idx < len(samples)-1:
            idx += 1

        # use the controller to produce a change to the agent
        heading = entities[self.target_entity].state_dict['phi']

        rot_mat = [[np.cos(heading), np.sin(heading), 0.0],
                   [-np.sin(heading), np.cos(heading), 0.0],
                   [0.0, 0.0, 1.0]]
        rot_mat = np.reshape(rot_mat, (3, 3))

        diff = np.subtract([samples[idx,0],samples[idx,1],samples[idx,2]], [entity_x,entity_y,heading])

        error_vec = np.matmul(rot_mat, diff)

        v_mag = np.sqrt(entities[self.target_entity].state_dict['v_mag'])
        command = self.controller.get_command(delta_t,error_vec,v_mag)

        # return the transfromed action
        return command

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

        samples = np.flip(samples, axis=0)

        # calculate angles
        for i in range(len(samples)):
            if i == len(samples) - 1:
                tmp_vec = [samples[i, 0] - samples[i - 1, 0], samples[i, 1] - samples[i - 1, 1]]
            elif i == 0:
                tmp_vec = [samples[i + 1, 0] - samples[i, 0], samples[i + 1, 1] - samples[i, 1]]
            else:
                tmp_vec = [samples[i + 1, 0] - samples[i - 1, 0], samples[i + 1, 1] - samples[i - 1, 1]]

            samples[i, 2] = np.arctan2(tmp_vec[1], tmp_vec[0])

        # samples columns = x, y, theta, ?
        return samples

    def setPersistentInfo(self,entities,sensors):
        """
        Save information about the action chosen at the simulation step it is chosen. This enables only storing copies
        of the needed information needed to reconstruct the action. For the bspline, the angles and starting conditions
        are what is needed to rebuild the bspline. s

        :param entities: An ordered dictionary containing the entities in the simulation.
        :param sensors: An ordered dictionary containing the sensors in a simulation.
        :return:
        """

        # save the root of the bsline to build the spline from
        self.start_location = [entities[self.target_entity].state_dict['x_pos'],entities[self.target_entity].state_dict['y_pos']]
        self.start_angle = entities[self.target_entity].state_dict['phi']

        presistentInfo = OrderedDict()
        presistentInfo['x_init'] = self.start_location[0]
        presistentInfo['y_init'] = self.start_location[1]
        presistentInfo['phi_init'] = self.start_angle
        return presistentInfo


class DubinsControl(ActionOperation):

    def __init__(self, action_options_dict, controller, frequency, is_continuous, name, number_controls,output_range, target_entity):
        """
        An action operation that converts a raw neural network output to a series of samples that are along a dubins
        path. A controller then uses the samples along the path to generate actuator commands to minimize the distance
        to the sample point. The action of the agent is effectively a dubins path.

        :param action_options_dict:
        :param controller:
        :param frequency:
        :param is_continuous:
        :param name:
        :param number_controls:
        :param output_range:
        :param target_entity:
        """
        super(DubinsControl, self).__init__(action_options_dict, controller, frequency, is_continuous, name, number_controls)

        self.n_samples = 15  # number of points along the bspline path used for navigating
        self.target_entity = target_entity

        # save information for refreshing.
        self.start_location = []
        self.start_angle = None

    def convert_action(self, action_vector, delta_t, entities, sensors):
        """
        Converts the output of a neural network into (an) actuator(s) change. Various methods can be used to make this
        conversion. Here a Dubins path is built from the outputs of a neural network. Then a controller works to
        follow the dubins path.

        :param action_vector: The vector of outputs from the neural network. In the discrete case, only an integer is
            provided. The integer has the index of the meta data from a combinatorial previously calculated. For the
            continuous case, the changes are directly provided.
        :param delta_t: The time step of the simulation. Needed for the controller to calculate the actuator changes
            for the agents.
        :param entities: An ordered dictionary containing the entities in the simulation.
        :param sensors: An ordered dictionary containing the sensors in a simulation.
        :return: The command that is in the dimensions that are appropriate for the entity to update its control scheme
            with.
        """

        if self.is_continuous:
            # TODO Don't know if this block is needed. May need scaling.
            path_description = copy.deepcopy(action_vector)
            for i, action in enumerate(action_vector):
                path_description[i] = (action - self.output_range[0]) * (self.action_bounds[i][1]-self.action_bounds[i][0]) / (self.output_range[1]-self.output_range[0]) + self.action_bounds[i][0]
        else:
            # discrete
            path_description = self.action_options[action_vector]

        # build the dubins path and get eht samples
        start = [entities[self.target_entity].state_dict['x_pos'],entities[self.target_entity].state_dict['y_pos'],entities[self.target_entity].state_dict['phi']]
        end_theta = entities[self.target_entity].state_dict['phi']+path_description[0]+path_description[2]
        if end_theta > np.pi*2.0:
            end_theta -= np.pi*2.0
        elif end_theta < 0.0:
            end_theta += np.pi*2.0
        end = [entities[self.target_entity].state_dict['x_pos'] + path_description[1]*np.cos(entities[self.target_entity].state_dict['phi']+path_description[0]),
               entities[self.target_entity].state_dict['y_pos'] + path_description[1]*np.sin(entities[self.target_entity].state_dict['phi']+path_description[0]),
               end_theta]
        radius = path_description[3]
        samples = self.build_shortest_dubins( start, end, radius, radius)

        # get the point that is nearest to the agent, then advance one
        idx = 0
        min_dst = np.infty

        # get the current position of the entity
        entity_x = entities[self.target_entity].state_dict['x_pos']
        entity_y = entities[self.target_entity].state_dict['y_pos']
        for i, samp in enumerate(samples):
            tmp_dst = np.sqrt( (samp[0]-entity_x)**2 + (samp[1]-entity_y)**2)
            if tmp_dst < min_dst:
                min_dst = tmp_dst
                idx = i
        # increment the index by 1 if possible.
        if idx < len(samples)-1:
            idx += 1

        # use the controller to produce a change to the agent
        heading = entities[self.target_entity].state_dict['phi']

        rot_mat = [[np.cos(heading), np.sin(heading), 0.0],
                   [-np.sin(heading), np.cos(heading), 0.0],
                   [0.0, 0.0, 1.0]]
        rot_mat = np.reshape(rot_mat, (3, 3))

        diff = np.subtract([samples[idx,0],samples[idx,1],samples[idx,2]], [entity_x,entity_y,heading])

        error_vec = np.matmul(rot_mat, diff)

        v_mag = np.sqrt(entities[self.target_entity].state_dict['v_mag'])
        command = self.controller.get_command(delta_t,error_vec,v_mag)

        # return the transfromed action
        return command

    def add_arc_sample_points(self, c_x, c_y, control_point, dir, n_samples, offset, radius, samples, sample_length):
        """
        Generates the [x,y,theta] points along an arc on the dubins path. The new points are placed into the
        samples numpy array.

        :param c_x: The x location of the center of the arc.
        :param c_y: The y location of the center of the arc.
        :param control_point: The [x,y,theta] point that the arc starts on.
        :param dir: The direction of the turn. Must be either 'left' or 'right'.
        :param n_samples: The number of samples along the curve to generate.
        :param offset: The index offset for saving the positions.
        :param radius: The radius of the arc.
        :param samples: The numpy array containing all of the samples for the dubins path.
        :param sample_length: The spacing of the sample points along the arc.
        :return:
        """
        for i in range(n_samples):
            dst_from_start = float(i) * sample_length
            delta_theta = dst_from_start / radius

            center_to_cp = [control_point[0] - c_x, control_point[1] - c_y]
            if dir == 'right':
                delta_theta *= -1.0

            rot_mat = [[np.cos(delta_theta), -np.sin(delta_theta)],
                       [np.sin(delta_theta), np.cos(delta_theta)]]
            rot_mat = np.reshape(rot_mat, (2, 2))
            new_vec = np.matmul(rot_mat, center_to_cp)
            sample_point = np.add(new_vec, [c_x, c_y])
            samples[i + offset, 0:2] = sample_point

            # sample angle
            point_angle = np.arctan2(new_vec[1], new_vec[0])
            if dir == 'right':
                point_angle -= np.pi / 2.0
            else:
                point_angle += np.pi / 2.0
            samples[i + offset, 2] = point_angle

    def build_dubins(self, start, end, radius1, radius2, dir_1, dir_2, n_samples):
        """
        Builds a set of [x,y,theta] points that are near equally spaced along a dubins path. The dubins path is built to
        be continuous. There is a bug in the tangent point calculation when the two radii are not-equal. The error is
        small and because the planned use case is to always use two radii of the same size.

        :param start: [x,y,theta] point defining the initial location for the dubins path
        :param end: [x,y,theta] point defining the ending location for the dubins path
        :param radius1: the radius [m] of the first turn in the path.
        :param radius2: the radius [m] of the second (and last) turn in the path.
        :param dir_1: The direction of the first turn. Must be 'left' or 'right'.
        :param dir_2: The direction of the second turn. Must be 'left' or 'right'.
        :param n_samples: The number of samples along the dubins path to be returned.
        :return: a series of [x,y,theta] points that define a coarse dubins path.
        """

        # calculate the center of the first circles
        c1x = None
        c1y = None
        if dir_1 == 'right':
            c1x = start[0] + radius1 * np.cos(start[2] - np.pi / 2.0)
            c1y = start[1] + radius1 * np.sin(start[2] - np.pi / 2.0)
        elif dir_1 == 'left':
            c1x = start[0] + radius1 * np.cos(start[2] + np.pi / 2.0)
            c1y = start[1] + radius1 * np.sin(start[2] + np.pi / 2.0)

        # calculate the center of the second circle
        c2x = None
        c2y = None
        if dir_2 == 'right':
            c2x = end[0] + radius2 * np.cos(end[2] - np.pi / 2.0)
            c2y = end[1] + radius2 * np.sin(end[2] - np.pi / 2.0)
        elif dir_2 == 'left':
            c2x = end[0] + radius2 * np.cos(end[2] + np.pi / 2.0)
            c2y = end[1] + radius2 * np.sin(end[2] + np.pi / 2.0)

        # get tangent points. The tangent points have a small innaccuracy in it. IT is not enough to make a difference
        # for the use case but it exists.
        cen_to_cen_x = (end[0] - start[0])
        cen_to_cen_y = (end[1] - start[1])
        dst = np.sqrt(cen_to_cen_x ** 2 + cen_to_cen_y ** 2)
        cen_to_cen_x /= dst
        cen_to_cen_y /= dst

        if dst < (radius1 - radius2) * (radius1 - radius2):
            # circles cannot have tangent lines drawn towards eachother
            return None, None

        tan_points = np.zeros([4, 4])
        k = 0
        sign1_vec = [1.0, -1.0]
        for sign1 in sign1_vec:
            c = (radius1 - sign1 * radius2) / dst
            if c * c > 1.0:
                continue
            h = np.sqrt(np.max([0.0, 1.0 - c * c]))

            sign2_vec = [1.0, -1.0]
            for sign2 in sign2_vec:
                n_x = cen_to_cen_x * c - sign2 * h * cen_to_cen_y
                n_y = cen_to_cen_y * c + sign2 * h * cen_to_cen_x
                tan_points[k, 0] = c1x + radius1 * n_x
                tan_points[k, 1] = c1y + radius1 * n_y
                tan_points[k, 2] = c2x + sign1 * radius2 * n_x
                tan_points[k, 3] = c2y + sign1 * radius2 * n_y
                k += 1

        # find the one continuous path. The start and end vectors are rotated around the circles. If the signs of the
        # vectors match they are continuous when the roots are coincident. If not, the vectors are in opposite directions
        # and the path is not continuous. When radius1 == radius2 the lines are tangent.
        valid_idx = [i for i in range(len(tan_points))]
        for idx in reversed(valid_idx):

            # check starting arc
            center_to_cp = [start[0] - c1x, start[1] - c1y]
            cp = [np.cos(start[2]), np.sin(start[2])]
            angle_start = get_angle_between_vectors(center_to_cp, cp, True)

            to_tan_vect = tan_points[idx, 0:2] - [c1x, c1y]
            tp = [tan_points[idx, 2] - tan_points[idx, 0], tan_points[idx, 3] - tan_points[idx, 1]]
            angle_tan = get_angle_between_vectors(to_tan_vect, tp, True)

            if np.sign(angle_start) != np.sign(angle_tan):
                # this point is not valid so remove it from the list
                valid_idx.pop(idx)
            else:
                # check the points on the second circle for validity
                center_to_cp = [end[0] - c2x, end[1] - c2y]
                cp = [np.cos(end[2]), np.sin(end[2])]
                angle_start = get_angle_between_vectors(center_to_cp, cp, True)

                to_tan_vect = tan_points[idx, 2:4] - [c2x, c2y]
                tp = [tan_points[idx, 2] - tan_points[idx, 0], tan_points[idx, 3] - tan_points[idx, 1]]
                angle_tan = get_angle_between_vectors(to_tan_vect, tp, True)

                if np.sign(angle_start) != np.sign(angle_tan):
                    valid_idx.pop(idx)

        # save the index of the tangent points that is the one solution that is continuous
        valid_idx = valid_idx[0]

        """
        Calculate the length of the three segments of the dubins path
        """
        # calculate the distance of the first arc
        center_to_cp = [start[0] - c1x, start[1] - c1y]
        cp = [np.cos(start[2]), np.sin(start[2])]
        angle_start = get_angle_between_vectors(center_to_cp, cp, True)
        to_tan_vect = tan_points[valid_idx, 0:2] - [c1x, c1y]
        angle_tan = get_angle_between_vectors(center_to_cp, to_tan_vect, True)
        if np.sign(angle_start) == np.sign(angle_tan):
            sector_1_length = np.abs(get_angle_between_vectors(to_tan_vect, center_to_cp, True)) * radius1
        else:
            sector_1_length = (2.0 * np.pi - np.abs(
                get_angle_between_vectors(to_tan_vect, center_to_cp, True))) * radius1

        # calculate the distance of the second arc
        center_to_cp = [end[0] - c2x, end[1] - c2y]
        cp = [np.cos(end[2]), np.sin(end[2])]
        angle_start = get_angle_between_vectors(center_to_cp, cp, True)
        to_tan_vect = tan_points[valid_idx, 2:4] - [c2x, c2y]
        angle_tan = get_angle_between_vectors(center_to_cp, to_tan_vect, True)
        if np.sign(angle_start) != np.sign(angle_tan):
            sector_2_length = np.abs(get_angle_between_vectors(to_tan_vect, center_to_cp, True)) * radius2
        else:
            sector_2_length = (2.0 * np.pi - np.abs(
                get_angle_between_vectors(to_tan_vect, center_to_cp, True))) * radius2

        # get the length fo the tangent line and the total path
        tangent_length = np.sqrt((tan_points[valid_idx, 2] - tan_points[valid_idx, 0]) ** 2 + (
                    tan_points[valid_idx, 3] - tan_points[valid_idx, 1]) ** 2)
        path_length = sector_1_length + sector_2_length + tangent_length

        """
        Generate samples along the dubins path.
        """

        # create n points along the dubins path that are near equally spaced. The columns are (x,y,theta)
        samples = np.zeros((n_samples, 3))

        # determine the number of samples in each segment
        sample_segment_length = path_length / (n_samples - 1)
        sector_1_n_samples = int(np.ceil(sector_1_length / sample_segment_length))
        sector_2_n_samples = int(np.ceil(sector_2_length / sample_segment_length))
        tangent_n_samples = (n_samples - 1) - sector_1_n_samples - sector_2_n_samples

        self.add_arc_sample_points(c1x, c1y, start, dir_1, sector_1_n_samples, 0, radius1, samples,
                              sector_1_length / sector_1_n_samples)

        # convert the tangent line into samples.
        tangent_sample_length = tangent_length / tangent_n_samples
        tangent_angle = np.arctan2((tan_points[valid_idx, 3] - tan_points[valid_idx, 1]),
                                   (tan_points[valid_idx, 2] - tan_points[valid_idx, 0]))
        for i in range(tangent_n_samples):
            samples[i + sector_1_n_samples, 0] = tan_points[valid_idx, 0] + tangent_sample_length * i * np.cos(
                tangent_angle)
            samples[i + sector_1_n_samples, 1] = tan_points[valid_idx, 1] + tangent_sample_length * i * np.sin(
                tangent_angle)
            samples[i + sector_1_n_samples, 2] = tangent_angle

        self.add_arc_sample_points(c2x, c2y, tan_points[valid_idx, 2:4], dir_2, sector_2_n_samples,
                              sector_1_n_samples + tangent_n_samples, radius2, samples,
                              sector_2_length / sector_2_n_samples)

        # add end point to samples
        samples[len(samples) - 1, :] = end

        return path_length, samples

    def build_shortest_dubins(self, start, end, radius1, radius2):
        """
        Builds four dubins paths, one with each combination of turn directions, and returns the path that is the shortest
        distance. The path is represented by a number of samples along it. The agent will then consume the points/samples
        as a series of navigation points.

        :param start: a vector of [x,y,theta] describing the initial location and orientation of the path following entity.
        :param end: a vector of [x,y,theta] describing the ending location and orientation of the path following entity.
        :param radius1: The radius [m] of the first turn to make for the dubins path.
        :param radius2: THe radius [m] of the second turn to make for the dubins path.
        :return: A list of samples [x,y,theta] that describe the shortest length dubins path.
        """

        n_samples = 40
        turn_combos = [['right', 'right'],
                       ['right', 'left'],
                       ['left', 'right'],
                       ['left', 'left']]

        # loop over the different turn combinations and save the shortest path.
        min_path_length = np.infty
        samples = []
        for turns in turn_combos:
            path_lenth, tmp_samples = self.build_dubins(start, end, radius1, radius2, turns[0], turns[1], n_samples)
            if path_lenth < min_path_length:
                samples = tmp_samples
                min_path_length = path_lenth

        return samples

    def setPersistentInfo(self,entities,sensors):
        """
        Save information about the first time an action is built. This allows for the simulation to get more
        actuator command updates while keep the original information needed to build the dubins path.
        :param entities: An ordered dictionary containing the entities in the simulation.
        :param sensors: An ordered dictionary containing the sensors in a simulation.
        :return:
        """

        # save the root of the bsline to build the spline from
        self.start_location = [entities[self.target_entity].state_dict['x_pos'],
                               entities[self.target_entity].state_dict['y_pos']]
        self.start_angle = entities[self.target_entity].state_dict['phi']

        presistentInfo = OrderedDict()
        presistentInfo['x_init'] = self.start_location[0]
        presistentInfo['y_init'] = self.start_location[1]
        presistentInfo['phi_init'] = self.start_angle
        return presistentInfo

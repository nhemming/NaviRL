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
import pandas as pd
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

    def setPersistentInfo(self,entities,sensors, action_vector):
        pass

    def draw_persistent(self, ax, df, sim_time):
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
        self.samples = []

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
        # get the point that is nearest to the agent, then advance one
        idx = 0
        min_dst = np.infty

        # get the current position of the entity
        entity_x = entities[self.target_entity].state_dict['x_pos']
        entity_y = entities[self.target_entity].state_dict['y_pos']
        for i, samp in enumerate(self.samples):
            tmp_dst = np.sqrt( (samp[0]-entity_x)**2 + (samp[1]-entity_y)**2)
            if tmp_dst < min_dst:
                min_dst = tmp_dst
                idx = i
        # increment the index by 1 if possible.
        if idx < len(self.samples)-1:
            idx += 1

        # use the controller to produce a change to the agent
        heading = entities[self.target_entity].state_dict['phi']

        rot_mat = [[np.cos(heading), np.sin(heading), 0.0],
                   [-np.sin(heading), np.cos(heading), 0.0],
                   [0.0, 0.0, 1.0]]
        rot_mat = np.reshape(rot_mat, (3, 3))

        diff = np.subtract([self.samples[idx,0],self.samples[idx,1],self.samples[idx,2]], [entity_x,entity_y,heading])

        error_vec = np.matmul(rot_mat, diff)

        v_mag = np.sqrt(entities[self.target_entity].state_dict['v_mag'])
        command = self.controller.get_command(delta_t,error_vec,v_mag)

        # return the transfromed action
        return command

    def build_spline_from_angles(self, path_angles):

        control_points = np.zeros((len(path_angles) + 1, 2))
        control_points[0, :] = np.array(self.start_location)
        cp_angle = np.zeros(len(path_angles))
        cp_angle[0] = self.start_angle + path_angles[0]
        for i in range(len(path_angles)):
            control_points[i + 1, 0] = control_points[i, 0] + self.segment_length * np.cos(cp_angle[i])
            control_points[i + 1, 1] = control_points[i, 1] + self.segment_length * np.sin(cp_angle[i])
            if i < len(path_angles) - 1:
                cp_angle[i + 1] = cp_angle[i] + path_angles[i + 1]

        samples = self.bezier_curve(control_points, self.n_samples)
        return samples

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

    def setPersistentInfo(self,entities,sensors, action_vector):
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

        if self.is_continuous:
            path_angles = copy.deepcopy(action_vector)
            for i, action in enumerate(action_vector):
                path_angles[i] = (action - self.output_range[0]) * (
                            self.action_bounds[i][1] - self.action_bounds[i][0]) / (
                                               self.output_range[1] - self.output_range[0]) + self.action_bounds[i][0]

        else:
            # discrete
            path_angles = self.action_options[action_vector]

        # build the b spline curve. from the called out angles
        self.samples = self.build_spline_from_angles(path_angles)

        presistentInfo = OrderedDict()
        presistentInfo['x_init'] = self.start_location[0]
        presistentInfo['y_init'] = self.start_location[1]
        presistentInfo['phi_init'] = self.start_angle
        return presistentInfo

    def draw_persistent(self, ax, df, sim_time):

        # collect the path at the current time
        slice = df[df['sim_time'] <= sim_time]
        slice_len = len(slice)-1

        if len(slice) != 0:

            # get path starting locations
            x_pos = slice['persistent_info_x_init'].iloc[slice_len]
            y_pos = slice['persistent_info_y_init'].iloc[slice_len]
            phi = slice['persistent_info_phi_init'].iloc[slice_len]

            self.start_location = np.reshape([x_pos,y_pos],(2,))
            self.start_angle = phi

            path_cols = slice[[col for col in df.columns if "mutated_action" in col]]
            col_names = list(path_cols.columns)
            path_dict = dict()
            for col in col_names:
                part_col = col.split('_')
                point_num = int(part_col[len(part_col) - 1])

                path_dict[point_num] = path_cols[col].iloc[slice_len]

            angle_df = pd.DataFrame(path_dict.items(), columns=['angle', 'val'])
            angle_df.sort_values(by=['angle'], inplace=True)
            path_angles = angle_df['val'].to_numpy()

            # un-normalize angles
            if self.is_continuous:
                for i, action in enumerate(path_angles):
                    path_angles[i] = (action - self.output_range[0]) * (
                                self.action_bounds[i][1] - self.action_bounds[i][0]) / (
                                                   self.output_range[1] - self.output_range[0]) + self.action_bounds[i][0]
            else:
                # discrete
                path_angles = self.action_options[path_angles]

            samples = self.build_spline_from_angles(path_angles)

            ax.plot(samples[:,0], samples[:,1], 'o--', color='gray')


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
        self.samples = []

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
        # get the point that is nearest to the agent, then advance one
        idx = 0
        min_dst = np.infty

        # get the current position of the entity
        entity_x = entities[self.target_entity].state_dict['x_pos']
        entity_y = entities[self.target_entity].state_dict['y_pos']
        for i, samp in enumerate(self.samples):
            tmp_dst = np.sqrt( (samp[0]-entity_x)**2 + (samp[1]-entity_y)**2)
            if tmp_dst < min_dst:
                min_dst = tmp_dst
                idx = i
        # increment the index by 1 if possible.
        if idx < len(self.samples)-1:
            idx += 1

        # use the controller to produce a change to the agent
        heading = entities[self.target_entity].state_dict['phi']

        rot_mat = [[np.cos(heading), np.sin(heading), 0.0],
                   [-np.sin(heading), np.cos(heading), 0.0],
                   [0.0, 0.0, 1.0]]
        rot_mat = np.reshape(rot_mat, (3, 3))

        diff = np.subtract([self.samples[idx,0],self.samples[idx,1],self.samples[idx,2]], [entity_x,entity_y,heading])

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

        n_samples = 15
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

    def setPersistentInfo(self,entities,sensors, action_vector):
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

        # build the path
        if self.is_continuous:
            path_description = copy.deepcopy(action_vector)
            for i, action in enumerate(action_vector):
                path_description[i] = (action - self.output_range[0]) * (self.action_bounds[i][1]-self.action_bounds[i][0]) / (self.output_range[1]-self.output_range[0]) + self.action_bounds[i][0]
        else:
            # discrete
            path_description = self.action_options[action_vector]

        # build the dubins path and get eht samples
        #start = [entities[self.target_entity].state_dict['x_pos'],entities[self.target_entity].state_dict['y_pos'],entities[self.target_entity].state_dict['phi']]
        #end_theta = entities[self.target_entity].state_dict['phi']+path_description[0]+path_description[2]
        start = [self.start_location[0], self.start_location[1],
                 self.start_angle]
        end_theta = self.start_angle + path_description[0] + path_description[2]
        if end_theta > np.pi*2.0:
            end_theta -= np.pi*2.0
        elif end_theta < 0.0:
            end_theta += np.pi*2.0
        end = [self.start_location[0] + path_description[1]*np.cos(self.start_angle + path_description[0]),
               self.start_location[1] + path_description[1]*np.sin(self.start_angle + path_description[0]),
               end_theta]
        radius = path_description[3]
        self.samples = self.build_shortest_dubins( start, end, radius, radius)


        presistentInfo = OrderedDict()
        presistentInfo['x_init'] = self.start_location[0]
        presistentInfo['y_init'] = self.start_location[1]
        presistentInfo['phi_init'] = self.start_angle
        return presistentInfo

    def draw_persistent(self, ax, df, sim_time):

        # collect the path at the current time
        slice = df[df['sim_time'] <= sim_time]
        slice_len = len(slice)-1

        if len(slice) != 0:

            path_description = []
            # un-normalize angles
            if self.is_continuous:

                #path_def = []
                path_cols = slice[[col for col in df.columns if "mutated_action" in col]]
                col_names = list(path_cols.columns)
                path_dict = dict()
                for col in col_names:
                    part_col = col.split('_')
                    point_num = int(part_col[len(part_col) - 1])

                    path_dict[point_num] = path_cols[col].iloc[slice_len]

                angle_df = pd.DataFrame(path_dict.items(), columns=['def', 'val'])
                angle_df.sort_values(by=['def'], inplace=True)
                path_def = angle_df['val'].to_numpy()



                path_description = copy.deepcopy(path_def)
                for i, action in enumerate(path_def):
                    path_description[i] = (action - self.output_range[0]) * (
                                self.action_bounds[i][1] - self.action_bounds[i][0]) / (
                                                      self.output_range[1] - self.output_range[0]) + \
                                          self.action_bounds[i][0]

            else:
                # discrete
                #path_description = self.action_options[action_vector]
                pass

            x_pos = slice['persistent_info_x_init'].iloc[slice_len]
            y_pos = slice['persistent_info_y_init'].iloc[slice_len]
            phi = slice['persistent_info_phi_init'].iloc[slice_len]

            # build the dubins path and get eht samples
            start = [x_pos,
                     y_pos,
                     phi]

            end_theta = phi + path_description[0] + path_description[2]
            if end_theta > np.pi * 2.0:
                end_theta -= np.pi * 2.0
            elif end_theta < 0.0:
                end_theta += np.pi * 2.0
            end = [x_pos + path_description[1] * np.cos(phi + path_description[0]),
                   y_pos + path_description[1] * np.sin(phi + path_description[0]),
                   end_theta]
            radius = path_description[3]
            samples = self.build_shortest_dubins(start, end, radius, radius)
            ax.plot(samples[:,0], samples[:,1], 'o--', color='gray')


class RLProbablisticRoadMap(ActionOperation):

    def __init__(self, action_options_dict, controller, domain, frequency, graph_frequency, is_continuous, max_connect_dst, name, number_controls, n_samples,output_range, target_entity, target_sensor, trans_dst, use_simple_model, model_path='', model_radius=None):
        """
        Action operation that a probabilistic road map to help navigate to the goal. The connections in the graph are
        either built with a simple circle radius model or using a neural network based transition model.

        :param action_options_dict:
        :param frequency:
        :param is_continuous:
        :param name:
        :param number_controls:
        :param output_range:
        """
        super(RLProbablisticRoadMap, self).__init__(action_options_dict, controller, frequency, is_continuous, name, number_controls)

        self.domain = domain
        self.n_samples = n_samples  # number of points in the PRM
        self.target_entity = target_entity
        self.target_sensor = target_sensor

        # save information for what transition model to use
        self.model_path = model_path
        self.model_radius = model_radius
        self.use_simple_model = True if model_radius is not None else False
        self.last_reset_time = 0.0  # [s]
        self.graph_frequency = graph_frequency

        # save information for refreshing.
        self.vertices = [] # list of the vertices in the graph
        self.max_connect_dst = max_connect_dst
        self.model_radius = model_radius
        self.n_samples = n_samples
        self.trans_dst = trans_dst #distance to subgoal to acheive before moving to the next subgoal
        self.use_simple_model = use_simple_model
        self.path = None  # path that leads to the goal location

    def init_state_action(self, entities, sensors):

        goal_loc = [entities['destination'].state_dict['x_pos'], entities['destination'].state_dict['y_pos']]

        start_loc = [entities[self.target_entity].state_dict['x_pos'], entities[self.target_entity].state_dict['y_pos']]
        state = {'phi': entities[self.target_entity].state_dict['phi']}

        # build a graph
        raw_path = self.build_prm(goal_loc, start_loc, state)

        self.path = self.format_path(raw_path)
        self.sub_goal_idx = 1
        sub_goal = self.path[self.sub_goal_idx,:]

        dst = np.sqrt( (start_loc[0]-sub_goal[0])**2 + (start_loc[1]-sub_goal[1])**2)
        target_unit_vec = [np.cos(entities[self.target_entity].state_dict['phi']), np.sin(entities[self.target_entity].state_dict['phi'])]
        goal_vec = [sub_goal[0]-start_loc[0],sub_goal[1]-start_loc[1]]
        mu = get_angle_between_vectors(target_unit_vec,goal_vec,True)
        if mu < 0:
            mu += 2.0*np.pi
        elif mu > 2.0*np.pi:
            mu -= 2.0 * np.pi

        # update the sub_goal sensor to point to the nearest point
        sensors[self.target_sensor].state_dict['angle'] = mu
        sensors[self.target_sensor].state_dict['distance'] = dst

    def prep_state_action(self, entities, sensors, sim_time):

        # reset last regraph time. Quick hack.
        if sim_time <= 1e-12:
            self.last_reset_time = -np.infty

        if sim_time - self.last_reset_time >= self.graph_frequency:

            self.last_reset_time = sim_time

            goal_loc = [entities['destination'].state_dict['x_pos'], entities['destination'].state_dict['y_pos']]

            start_loc = [entities[self.target_entity].state_dict['x_pos'],
                         entities[self.target_entity].state_dict['y_pos']]
            state = {'phi': entities[self.target_entity].state_dict['phi']}

            # build a graph
            raw_path = self.build_prm(goal_loc, start_loc, state)

            # change path to be (x,y,theta)
            self.path = self.format_path(raw_path)

            # set current sub goal
            self.sub_goal_idx = 1

        # update sub destination sensor to point to the nearest point in the path
        curr_loc = [entities[self.target_entity].state_dict['x_pos'],entities[self.target_entity].state_dict['y_pos']]
        tmp_dst = np.sqrt((curr_loc[0] - self.path[self.sub_goal_idx,0]) ** 2 + (curr_loc[1] - self.path[self.sub_goal_idx,1]) ** 2)
        if (tmp_dst <= self.trans_dst):
            self.sub_goal_idx += 1

        if self.sub_goal_idx >= len(self.path)-2:
            self.sub_goal_idx = len(self.path)-1 # correct for walking off the end of the array
            sub_goal = self.path[len(self.path)-1, :2]
        else:
            sub_goal = self.path[self.sub_goal_idx+1,:2]

        dst = np.sqrt((curr_loc[0] - sub_goal[0]) ** 2 + (curr_loc[1] - sub_goal[1]) ** 2)
        target_unit_vec = [np.cos(entities[self.target_entity].state_dict['phi']),
                           np.sin(entities[self.target_entity].state_dict['phi'])]
        goal_vec = [sub_goal[0] - curr_loc[0], sub_goal[1] - curr_loc[1]]
        mu = get_angle_between_vectors(target_unit_vec, goal_vec, True)
        if mu < 0:
            mu += 2.0 * np.pi
        elif mu > 2.0 * np.pi:
            mu -= 2.0 * np.pi

        # update the sub_goal sensor to point to the nearest point
        sensors[self.target_sensor].state_dict['angle'] = mu
        sensors[self.target_sensor].state_dict['distance'] = dst

    def format_path(self, path):

        # change path to be (x,y,theta)
        samples = np.zeros((len(path), 3))
        for i, vert in enumerate(reversed(path)):
            samples[i, 0] = vert.location[0]
            samples[i, 1] = vert.location[1]
            samples[i, 2] = vert.state['phi']

        return samples

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
            transformed_action = np.zeros_like(action_vector)
            for i, action in enumerate(action_vector):
                transformed_action[i] = (action - self.output_range[0]) * (
                            self.action_bounds[0][1] - self.action_bounds[0][0]) / (
                                                    self.output_range[1] - self.output_range[0]) + \
                                        self.action_bounds[0][0]
        else:
            # convert index to case
            transformed_action = self.action_options[action_vector][0]

        return transformed_action

    def build_prm(self, goal_loc, start_loc, state):
        # build the verticies of the PRM
        self.vertices = []
        self.vertices.append(VertexPRM(start_loc, state=state))
        for i in range(self.n_samples):

            # TODO need to check if the point is in an obstacle.

            # draw random point
            location = np.zeros((2,))
            location[0] = np.random.uniform(low=self.domain['min_x'], high=self.domain['max_x'])
            location[1] = np.random.uniform(low=self.domain['min_y'], high=self.domain['max_y'])

            self.vertices.append(VertexPRM(location))

        self.vertices.append(VertexPRM(goal_loc))

        # try to connect the nodes in the PRM with a simple arc
        open_verts = [self.vertices[0]]
        counts = 0
        while len(open_verts) > 0 and counts < len(self.vertices) + 1:
            counts += 1
            current_vert = open_verts.pop()
            for i, tmp_vert in enumerate(self.vertices):
                if tmp_vert != current_vert:

                    curr_state = current_vert.state

                    dst = np.sqrt((current_vert.location[0] - tmp_vert.location[0]) ** 2 + (
                            current_vert.location[1] - tmp_vert.location[1]) ** 2)

                    if self.use_simple_model:
                        # use a simple arc model for determining if the node is reachable

                        # get angle from current heading to tmp vertex
                        delta_x = tmp_vert.location[0] - current_vert.location[0]
                        delta_y = tmp_vert.location[1] - current_vert.location[1]
                        theta = np.arctan2(delta_y, delta_x)
                        mu1 = theta - curr_state['phi']
                        if mu1 >= 0:
                            mu2 = np.pi * 2.0 - mu1  # explementary angle
                        else:
                            mu2 = np.pi * 2.0 + mu1  # explementary angle
                        mu_v = [mu1, mu2]
                        ind = np.argmin(np.abs(mu_v))
                        mu = mu_v[ind]

                        if np.abs(mu) > np.pi / 2.0:
                            gamma = np.abs(mu) - np.pi / 2.0
                        else:
                            gamma = np.pi / 2.0 - np.abs(mu)

                        arc_radius = dst * np.sin(gamma) / np.sin(np.pi - 2.0 * gamma)

                        # plot the  circle connecting everything
                        if mu < 0:
                            angle_off = -np.pi / 2.0

                        else:
                            angle_off = np.pi / 2.0
                        radius_vec = [arc_radius * np.cos(state['phi']), arc_radius * np.sin(state['phi'])]
                        radius_offset = [radius_vec[0] * np.cos(angle_off) - radius_vec[1] * np.sin(angle_off),
                                         radius_vec[0] * np.sin(angle_off) + radius_vec[1] * np.cos(angle_off)]
                        radius_point = [current_vert.location[0] + radius_offset[0],
                                        current_vert.location[1] + radius_offset[1]]

                        vec1 = [current_vert.location[0] - radius_point[0], current_vert.location[1] - radius_point[1]]
                        vec2 = [tmp_vert.location[0] - radius_point[0], tmp_vert.location[1] - radius_point[1]]
                        delta1 = get_angle_between_vectors(vec1, vec2, True)
                        unit_vec = [np.cos(state['phi']), np.sin(state['phi'])]
                        delta2 = get_angle_between_vectors(vec1, unit_vec, True)
                        circumfrance = 2.0 * np.pi * arc_radius
                        if (delta1 > 0 and delta2 < 0) or (delta1 < 0 and delta2 > 0):
                            # long way around
                            arc_length = max([circumfrance - np.abs(delta1) * arc_radius, np.abs(delta1) * arc_radius])
                        else:
                            # short way around
                            arc_length = min([circumfrance - np.abs(delta1) * arc_radius, np.abs(delta1) * arc_radius])

                        # get end angle
                        swept_angle = arc_length / arc_radius  # angle to rotate initial psi to to get ending angle
                        end_vec = [unit_vec[0] * np.cos(swept_angle) - unit_vec[1] * np.sin(swept_angle),
                                   unit_vec[0] * np.sin(swept_angle) + unit_vec[1] * np.cos(swept_angle)]
                        end_angle = np.arctan2(end_vec[1], end_vec[0])

                        if arc_radius >= self.model_radius and dst <= self.max_connect_dst and arc_length <= 2.0 * self.max_connect_dst:
                            # can reach the point. Make the connection and add the node to the open set.

                            if tmp_vert not in current_vert.children and current_vert not in tmp_vert.children:
                                tmp_vert.state = {'phi': end_angle}
                                tmp_vert.dst_to_par = arc_length
                                current_vert.children.append(tmp_vert)
                                open_verts.append(tmp_vert)
                    else:
                        # TODO
                        # use a surrogate model for determining if the next node is reachable
                        pass

        # extract path from graph with astar
        path, count = self.astar(self.vertices[0], self.vertices[len(self.vertices) - 1], len(self.vertices))

        # set current waypoint to goal if no path exists.
        if path == [] or count >= len(self.vertices):
            path = []
            # I don't think the stat values matter here
            path.append(VertexPRM(goal_loc, state=state))
            path.append(VertexPRM(start_loc, state={'phi':0.0}))


        return path

    def astar(self, start, goal, n_verts):

        open_lst = [start]
        closed_lst = []
        is_complete = False
        count = 0
        while len(open_lst) > 0 and not is_complete: #and count < n_verts + 1:
            count += 1
            min_f = np.infty
            min_idx = None
            for i, node in enumerate(open_lst):
                if node.f < min_f:
                    min_idx = i
                    min_f = node.f
            q = open_lst.pop(min_idx)

            for child in q.children:

                if child == goal:
                    child.parent = q
                    is_complete = True
                    break

                if child not in open_lst and child not in closed_lst:
                    child.h = np.sqrt(
                        (child.location[0] - goal.location[0]) ** 2 + (
                                child.location[1] - goal.location[1]) ** 2)
                    # child.h = dst_to_goal
                    child.g = q.g + child.dst_to_par  # distance to reach parent node
                    child.f = child.h + child.g

                    child.parent = q
                    open_lst.append(child)

            closed_lst.append(q)

        # build the path
        path = []
        current_node = goal
        if goal.parent is not None:
            path_count = 0
            while current_node.parent is not None and path_count < n_verts + 1:
                path_count += 1
                path.append(current_node)
                current_node = current_node.parent
            # if len(path) == 0:

            # add the start node to the path
            path.append(start)

        return path, count

    def setPersistentInfo(self,entities,sensors,action_vector):
        """
        Save information about the action chosen at the simulation step it is chosen. This enables only storing copies
        of the needed information needed to reconstruct the action. For the bspline, the angles and starting conditions
        are what is needed to rebuild the bspline. s

        :param entities: An ordered dictionary containing the entities in the simulation.
        :param sensors: An ordered dictionary containing the sensors in a simulation.
        :return:
        """
        presistentInfo = OrderedDict()
        k = 0
        for i, vert in enumerate(self.path):
            presistentInfo['path_x_'+str(k)] = vert[0]
            presistentInfo['path_y_' + str(k)] = vert[1]
            k += 1

        return presistentInfo

    def draw_persistent(self, ax, df, sim_time):

        # collect the path at the current time
        slice = df[df['sim_time'] == sim_time]

        if len(slice) != 0:

            # reorganize into the correct order
            path_cols = slice[[col for col in df.columns if "persistent_info_path" in col]]
            col_names = list(path_cols.columns)
            x_dict = dict()
            y_dict = dict()
            for col in col_names:
                if not np.isnan(path_cols[col].iloc[0]):

                    part_col = col.split('_')
                    point_num = int(part_col[len(part_col)-1])
                    point_dim = str(part_col[len(part_col)-2])

                    if point_dim == 'x':
                        x_dict[point_num] = path_cols[col].iloc[0]
                    elif point_dim == 'y':
                        y_dict[point_num] = path_cols[col].iloc[0]

            x_df = pd.DataFrame(x_dict.items(),columns=['x','val'])
            x_df.sort_values(by=['x'], inplace=True)
            y_df = pd.DataFrame(y_dict.items(),columns=['y','val'])
            y_df.sort_values(by=['y'], inplace=True)
            # plot
            ax.plot(x_df['val'], y_df['val'],'o--',color='gray')

class VertexPRM():

    def __init__(self, location, state=None):
        self.children = []
        self.parent = None
        self.location = location
        self.state = state
        self.dst_to_par = None
        self.g = 0
        self.h = 0
        self.f = 0

"""
Script for testing math for creating the dubins path

inspired by https://github.com/gieseanw/Dubins/blob/master/Includes.cpp
    https://gieseanw.wordpress.com/2012/10/21/a-comprehensive-step-by-step-tutorial-to-computing-dubins-paths/
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


from environment.Entity import get_angle_between_vectors


def add_arc_sample_points(c_x,c_y,control_point,dir,n_samples,offset,radius, samples, sample_length):
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
        samples[i+offset, 0:2] = sample_point

        # sample angle
        point_angle = np.arctan2(new_vec[1], new_vec[0])
        if dir == 'right':
            point_angle -= np.pi / 2.0
        else:
            point_angle += np.pi / 2.0
        samples[i+offset, 2] = point_angle


def build_dubins(start,end,radius1,radius2,dir_1, dir_2, n_samples):
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
        c1x = start[0] + radius1*np.cos(start[2]-np.pi/2.0)
        c1y = start[1] + radius1 * np.sin(start[2] - np.pi / 2.0)
    elif dir_1 == 'left':
        c1x = start[0] + radius1*np.cos(start[2]+np.pi/2.0)
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
    cen_to_cen_x = (end[0]-start[0])
    cen_to_cen_y = (end[1] - start[1])
    dst = np.sqrt(cen_to_cen_x**2 + cen_to_cen_y**2)
    cen_to_cen_x /= dst
    cen_to_cen_y /= dst

    if dst < (radius1-radius2)*(radius1-radius2):
        # circles cannot have tangent lines drawn towards eachother
        return None,None

    tan_points = np.zeros([4,4])
    k = 0
    sign1_vec = [1.0,-1.0]
    for sign1 in sign1_vec:
        c = (radius1-sign1*radius2)/dst
        if c*c > 1.0:
            continue
        h = np.sqrt(np.max([0.0,1.0-c*c]))

        sign2_vec = [1.0,-1.0]
        for sign2 in sign2_vec:
            n_x = cen_to_cen_x*c - sign2*h*cen_to_cen_y
            n_y = cen_to_cen_y * c + sign2 * h * cen_to_cen_x
            tan_points[k,0] = c1x+radius1*n_x
            tan_points[k, 1] = c1y+radius1*n_y
            tan_points[k, 2] = c2x + sign1*radius2 * n_x
            tan_points[k, 3] = c2y + sign1*radius2 * n_y
            k += 1

    # find the one continuous path. The start and end vectors are rotated around the circles. If the signs of the
    # vectors match they are continuous when the roots are coincident. If not, the vectors are in opposite directions
    # and the path is not continuous. When radius1 == radius2 the lines are tangent.
    valid_idx = [i for i in range(len(tan_points))]
    for idx in reversed(valid_idx):

        # check starting arc
        center_to_cp = [start[0]-c1x,start[1]-c1y]
        cp = [np.cos(start[2]),np.sin(start[2])]
        angle_start = get_angle_between_vectors(center_to_cp, cp, True)

        to_tan_vect = tan_points[idx, 0:2] - [c1x, c1y]
        tp = [ tan_points[idx,2]-tan_points[idx,0],tan_points[idx,3]-tan_points[idx,1]]
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
    angle_tan = get_angle_between_vectors(center_to_cp,to_tan_vect , True)
    if np.sign(angle_start) == np.sign(angle_tan):
        sector_1_length = np.abs(get_angle_between_vectors(to_tan_vect, center_to_cp, True))*radius1
    else:
        sector_1_length = (2.0*np.pi-np.abs(get_angle_between_vectors(to_tan_vect, center_to_cp, True))) * radius1

    # calculate the distance of the second arc
    center_to_cp = [end[0] - c2x, end[1] - c2y]
    cp = [np.cos(end[2]), np.sin(end[2])]
    angle_start = get_angle_between_vectors(center_to_cp, cp, True)
    to_tan_vect = tan_points[valid_idx, 2:4] - [c2x, c2y]
    angle_tan = get_angle_between_vectors(center_to_cp,to_tan_vect, True)
    if np.sign(angle_start) != np.sign(angle_tan):
        sector_2_length = np.abs(get_angle_between_vectors(to_tan_vect, center_to_cp, True))*radius2
    else:
        sector_2_length = (2.0*np.pi-np.abs(get_angle_between_vectors(to_tan_vect, center_to_cp, True))) * radius2

    # get the length fo the tangent line and the total path
    tangent_length = np.sqrt( (tan_points[valid_idx, 2]-tan_points[valid_idx, 0])**2 +(tan_points[valid_idx, 3]-tan_points[valid_idx, 1])**2 )
    path_length = sector_1_length+sector_2_length+tangent_length

    """
    Generate samples along the dubins path.
    """

    # create n points along the dubins path that are near equally spaced. The columns are (x,y,theta)
    samples = np.zeros((n_samples, 3))

    # determine the number of samples in each segment
    sample_segment_length = path_length/(n_samples-1)
    sector_1_n_samples = int(np.ceil(sector_1_length / sample_segment_length))
    sector_2_n_samples = int(np.ceil(sector_2_length / sample_segment_length))
    tangent_n_samples = (n_samples-1)-sector_1_n_samples-sector_2_n_samples

    add_arc_sample_points(c1x, c1y, start, dir_1, sector_1_n_samples, 0, radius1, samples, sector_1_length / sector_1_n_samples)

    # convert the tangent line into samples.
    tangent_sample_length = tangent_length / tangent_n_samples
    tangent_angle = np.arctan2((tan_points[valid_idx, 3] - tan_points[valid_idx, 1]),(tan_points[valid_idx, 2] - tan_points[valid_idx, 0]))
    for i in range(tangent_n_samples):
        samples[i+sector_1_n_samples,0] =  tan_points[valid_idx, 0] + tangent_sample_length*i*np.cos(tangent_angle)
        samples[i + sector_1_n_samples, 1] = tan_points[valid_idx, 1] + tangent_sample_length * i * np.sin(
            tangent_angle)
        samples[i + sector_1_n_samples, 2] = tangent_angle

    add_arc_sample_points(c2x, c2y, tan_points[valid_idx, 2:4], dir_2, sector_2_n_samples, sector_1_n_samples+tangent_n_samples, radius2, samples,
                          sector_2_length / sector_2_n_samples)

    # add end point to samples
    samples[len(samples)-1,:] = end

    return path_length, samples


def build_shortest_dubins(start,end,radius1,radius2):
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
    turn_combos = [['right','right'],
                   ['right','left'],
                   ['left','right'],
                   ['left','left']]

    # loop over the different turn combinations and save the shortest path.
    min_path_length = np.infty
    samples = []
    for turns in turn_combos:
        path_lenth, tmp_samples = build_dubins(start, end, radius1, radius2, turns[0], turns[1], n_samples)
        if path_lenth < min_path_length:
            samples = tmp_samples
            min_path_length = path_lenth

    return samples

def main():

    start = [0.0,1.0,np.pi/2.0]
    end = [10.0, 5.0, -np.pi / 2.0]
    radius1 = 1.0
    radius2 = 1.0

    samples = build_shortest_dubins(start, end, radius1, radius2)

    sns.set_theme()
    fig = plt.figure(0,figsize=(14,8))
    ax = fig.add_subplot(111)

    ax.plot([start[0],start[0]+np.cos(start[2])],[start[1],start[1]+np.sin(start[2])])
    ax.plot([end[0], end[0] + np.cos(end[2])], [end[1], end[1] + np.sin(end[2])])
    ax.plot(samples[:,0], samples[:,1],'o-')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    main()
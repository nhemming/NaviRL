"""
Script for testing and working out RRT* fpr navigation
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def build_graph_RRTStar(start, goal, n_samples, domain, link_dst, goal_dst = 0.3, neighbor_radius=0.5):
    start_vert = VertexPRM(start, dict())
    start_vert.state['phi'] = np.pi / 2.0  # point due north
    vertices = [start_vert]

    samples_added = 0
    n_attempts = 0
    max_attempts = n_samples * 2
    while samples_added < n_samples and n_attempts < max_attempts:

        # draw a random sample with respect to the start location. The start location is assumed where the robot currently is
        location = np.random.uniform(low=domain[0], high=domain[1], size=(2,))

        # find the nearest vertex
        near_vert = getNearestVertex(vertices, location)

        # generate vertex
        new_vert = create_vertex(near_vert, location)

        # TODO handle not added node
        vertices.append(new_vert)

        # get distance between closest node
        new_vert.rrt_cost = getDistance(new_vert, near_vert) + near_vert.rrt_cost

        # get the neighbors
        neighbors, best_neighbor = getNeighbors(vertices, new_vert, neighbor_radius)

        # add a link
        add_link(best_neighbor, new_vert)

        # check for local improvements
        for neigh in neighbors:
            if new_vert.rrt_cost + getDistance(neigh,near_vert) < neigh.rrt_cost:
                # update parents
                neigh.rrt_cost = new_vert.rrt_cost + getDistance(new_vert, neigh)
                #new_vert.children.append(neigh)
                neigh.parent = new_vert



        if getDistance(new_vert,VertexPRM(goal,{})) < goal_dst:
            # found a path
            #break
            pass

        n_attempts += 1

    # use Astar to get the path to the goal
    goal_vert = getNearestVertex(vertices, goal)
    path, count = astar(vertices[0], goal_vert, len(vertices))

    # draw the graph
    x = []
    y = []
    for vert in vertices:
        x.append(vert.location[0])
        y.append(vert.location[1])

    sns.set_theme()
    fig = plt.figure(0, figsize=(14, 8))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(x, y, 'o')
    ax.plot(start[0], start[1], 'o', label='Start')
    ax.plot(goal[0], goal[1], 'o', label='goal')

    # draw edge
    for vert in vertices:
        try:
            ax.plot([vert.location[0], vert.parent.location[0]], [vert.location[1], vert.parent.location[1]], 'k--', alpha=0.2)
        except:
            pass
        #for child in vert.children:
        #    ax.plot([vert.location[0], child.location[0]], [vert.location[1], child.location[1]], 'k--', alpha=0.2)

    # plot the path
    x_path = []
    y_path = []
    for i, vert in enumerate(path):
        x_path.append(vert.location[0])
        y_path.append(vert.location[1])

    ax.plot(x_path, y_path, ':', label='path')

    ax.legend()
    plt.tight_layout()
    plt.show()

def add_link(near_vert, new_vert):
    new_vert.state['phi'] = 0.0  # TODO update to use circular model
    #near_vert.children.append(new_vert)
    new_vert.parent = near_vert
    new_vert.dst_to_par = np.sqrt(
        (new_vert.location[0] - near_vert.location[0]) ** 2 + (new_vert.location[1] - near_vert.location[1]) ** 2)\

def create_vertex(near_vert, location):
    # get angle conecting the points
    angle = np.arctan2(location[1] - near_vert.location[1], location[0] - near_vert.location[0])

    new_location = [near_vert.location[0] + np.cos(angle) * link_dst[2],
                    near_vert.location[1] + np.sin(angle) * link_dst[2]]

    new_vert = VertexPRM(new_location,dict())

    return new_vert

def getDistance(n1, n2):
    # TODO update to cicrcular link
    return np.sqrt( (n1.location[0]-n2.location[0])**2 + (n1.location[1]-n2.location[1])**2)

def getNeighbors(vertices, new_vert, radius):

    best_dst = np.infty
    neighbors = []
    best_neighbor = None
    for vert in vertices:

        if vert != new_vert:

            n_dst = getDistance(vert, new_vert)
            if n_dst <= radius:
                neighbors.append(vert)

                if n_dst < best_dst:
                    best_dst = n_dst
                    best_neighbor = vert

    return neighbors, best_neighbor

def build_graph_RRT(start, goal, n_samples, domain, link_dst, goal_dst = 0.3):
    # link_dst - min, max, start

    start_vert = VertexPRM(start,dict())
    start_vert.state['phi'] = np.pi/2.0 # point due north
    vertices = [start_vert]

    samples_added = 0
    n_attempts = 0
    max_attempts = n_samples*2
    while samples_added < n_samples and n_attempts < max_attempts:

        # draw a random sample with respect to the start location. The start location is assumed where the robot currently is
        location = np.random.uniform(low=domain[0], high=domain[1], size=(2,))

        # TODO check for location collision

        # find the nearest vertex
        near_vert = getNearestVertex(vertices, location)

        # attempt to create link
        if not create_link(vertices, near_vert,location, link_dst):
            # failed to create a link
            continue
        n_samples += 1

        # check if location is within reaching distance of goal. If so return, else continue. May want to do additional refinement
        recent_vert = near_vert.children[len(near_vert.children)-1]
        dst_to_goal = np.sqrt((recent_vert.location[0]-goal[0])**2 +(recent_vert.location[1]-goal[1])**2)
        if dst_to_goal < goal_dst:
            # found a suitable path
            print('Goal reached')
            break

        n_attempts += 1

    print('Number of attempts {:d}'.format(n_attempts))

    # use Astar to get the path to the goal
    goal_vert = getNearestVertex(vertices, goal)
    path, count = astar(vertices[0], goal_vert, len(vertices))

    # draw the graph
    x = []
    y = []
    for vert in vertices:
        x.append(vert.location[0])
        y.append(vert.location[1])

    sns.set_theme()
    fig = plt.figure(0, figsize=(14,8))
    ax = fig.add_subplot(1,1,1)

    ax.plot(x,y,'o')
    ax.plot(start[0], start[1], 'o', label='Start')
    ax.plot(goal[0], goal[1], 'o', label='goal')

    # draw edge
    for vert in vertices:
        for child in vert.children:
            ax.plot([vert.location[0],child.location[0]],[vert.location[1],child.location[1]],'k--',alpha=0.2)

    # plot the path
    x_path = []
    y_path = []
    for i, vert in enumerate(path):
        x_path.append(vert.location[0])
        y_path.append(vert.location[1])

    ax.plot(x_path, y_path, ':',label='path')

    ax.legend()
    plt.tight_layout()
    plt.show()

def getNearestVertex(vertices, location):

    closest_vert = None
    min_dst = np.infty
    for vert in vertices:
        tmp_dst = np.sqrt( (vert.location[0]-location[0])**2 + (vert.location[1]-location[1])**2)
        if tmp_dst < min_dst:
            min_dst = tmp_dst
            closest_vert = vert

    return closest_vert

def create_link(verticies, near_vert,location, link_dst):

    # get angle conecting the points
    angle = np.arctan2(location[1]-near_vert.location[1],location[0]-near_vert.location[0])

    new_location = [near_vert.location[0]+np.cos(angle)*link_dst[2],near_vert.location[1]+np.sin(angle)*link_dst[2]]

    # TODO check if new location can be reached. Return None if cannot. Attempt to link points

    # create the link
    new_vert = VertexPRM(new_location,dict())
    new_vert.state['phi'] = 0.0 # TODO update to use circular model
    near_vert.children.append(new_vert)
    new_vert.dst_to_par = np.sqrt( (new_vert.location[0]-near_vert.location[0])**2 + (new_vert.location[1]-near_vert.location[1])**2)

    verticies.append(new_vert)

    return True

def astar( start, goal, n_verts):
    open_lst = [start]
    closed_lst = []
    is_complete = False
    count = 0
    while len(open_lst) > 0 and not is_complete:  # and count < n_verts + 1:
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
        self.rrt_cost = 0.


if __name__ == '__main__':

    trial = 2

    if trial == 0:
        start = [0,0]
        goal = [1,0]
        n_samples = 1000
        domain = [-0.5,1.5]
        link_dst = [0.01, 1,0.05] # min, max , ideal
    elif trial == 1:
        start = [0, 0]
        goal = [2, 0]
        n_samples = 1000
        domain = [-1.0, 2.5]
        link_dst = [0.01, 1, 0.05]  # min, max , ideal
    elif trial == 2:
        start = [0, 0]
        goal = [4, 0]
        n_samples = 1000
        domain = [-1.0, 5.0]
        link_dst = [0.01, 1, 0.05]  # min, max , ideal


    np.random.seed(0)
    #build_graph_RRT(start, goal, n_samples, domain, link_dst)
    build_graph_RRTStar(start, goal, n_samples, domain, link_dst)
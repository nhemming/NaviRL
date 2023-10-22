"""
A script to work out the details for ReinforcementLearning-Probablistic Road Map (RL-PRM)
"""
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from environment.Entity import get_angle_between_vectors

class RLPRM:

    def __init__(self, domain, max_connect_dst, model_radius, n_samples, use_simple_model=True):
        self.domain = domain
        self.vertices = []
        self.max_connect_dst = max_connect_dst
        self.model_radius = model_radius
        self.n_samples = n_samples
        self.use_simple_model = use_simple_model

    def build_prm(self, goal_loc, start_loc, state):
        # build the verticies of the PRM
        self.vertices = []
        self.vertices.append(VertexPRM(start_loc, state=state))
        for i in range(self.n_samples):
            # TODO need domain for low and high values
            location = np.random.uniform(low=self.domain[0], high=self.domain[1], size=(2,))
            self.vertices.append(VertexPRM(location))

        self.vertices.append(VertexPRM(goal_loc))

        # plot nodes
        fig = plt.figure(0,figsize=(14,8))
        ax = fig.add_subplot(111)
        plt.tight_layout()
        #verts = np.reshape(self.vertices,(len(self.vertices),2))
        verts = np.zeros((len(self.vertices),2))
        for i, vert in enumerate(self.vertices):
            verts[i] = vert.location
        ax.plot(verts[:,0],verts[:,1],'o')
        ax.plot(verts[0, 0], verts[0, 1], 'o', label='start')
        ax.plot(verts[len(verts)-1, 0], verts[len(verts)-1, 1], 'o', label='end')
        ax.legend()

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
                    dst_to_org = 0.0  # distance from current location to current_vert

                    if self.use_simple_model:
                        # use a simple arc model for determining if the node is reachable

                        # get angle from current heading to tmp vertex
                        delta_x = tmp_vert.location[0] - current_vert.location[0]
                        delta_y = tmp_vert.location[1] - current_vert.location[1]
                        theta = np.arctan2(delta_y, delta_x)
                        mu1 = theta - curr_state['psi']
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
                            angle_off = -np.pi/2.0

                        else:
                            angle_off = np.pi / 2.0
                        radius_vec = [arc_radius*np.cos(state['psi']),arc_radius*np.sin(state['psi'])]
                        radius_offset = [radius_vec[0]*np.cos(angle_off)-radius_vec[1]*np.sin(angle_off),radius_vec[0]*np.sin(angle_off)+radius_vec[1]*np.cos(angle_off)]
                        radius_point = [current_vert.location[0]+radius_offset[0],current_vert.location[1]+radius_offset[1]]

                        vec1 = [current_vert.location[0]-radius_point[0],current_vert.location[1]-radius_point[1]]
                        vec2 = [tmp_vert.location[0] - radius_point[0], tmp_vert.location[1] - radius_point[1]]
                        delta1 = get_angle_between_vectors(vec1,vec2,True)
                        unit_vec = [np.cos(state['psi']),np.sin(state['psi'])]
                        delta2 = get_angle_between_vectors(vec1,unit_vec,True)
                        circumfrance = 2.0*np.pi*arc_radius
                        if (delta1 > 0 and delta2 < 0) or (delta1 <0 and delta2 > 0):
                            # long way around
                            arc_length = max([circumfrance-np.abs(delta1)*arc_radius,np.abs(delta1)*arc_radius])


                        else:
                            # short way around
                            arc_length = min([circumfrance - np.abs(delta1) * arc_radius, np.abs(delta1) * arc_radius])

                        # get end angle
                        swept_angle = arc_length / arc_radius  # angle to rotate initial psi to to get ending angle
                        end_vec = [unit_vec[0] * np.cos(swept_angle) - unit_vec[1] * np.sin(swept_angle),
                                   unit_vec[0] * np.sin(swept_angle) + unit_vec[1] * np.cos(swept_angle)]
                        end_angle = np.arctan2(end_vec[1],end_vec[0])

                        '''
                        ax.plot(radius_point[0],radius_point[1],'x')
                        theta = np.linspace(0,2.0*np.pi,361)
                        tx = []
                        ty = []
                        for t in theta:
                            tx.append(np.cos(t)*arc_radius+radius_point[0])
                            ty.append(np.sin(t) * arc_radius + radius_point[1])
                        ax.plot(tx,ty,'--')
                        '''

                        if arc_radius >= self.model_radius and dst <= self.max_connect_dst and arc_length <= 2.0*self.max_connect_dst:
                            # can reach the point. Make the connection and add the node to the open set.

                            if tmp_vert not in current_vert.children and current_vert not in tmp_vert.children:
                                tmp_vert.state = {'psi':end_angle}
                                tmp_vert.dst_to_par =arc_length
                                current_vert.children.append(tmp_vert)
                                open_verts.append(tmp_vert)
                    else:
                        # TODO
                        # use a surrogate model for determining if the next node is reachable
                        pass

        # extract path from graph
        for i, vert in enumerate(self.vertices):
            #verts[i] = vert.location
            for child in vert.children:
                ax.plot([child.location[0],vert.location[0]],[child.location[1],vert.location[1]],'k--',alpha=0.1)

        # astar
        path, count = self.astar(self.vertices[0], self.vertices[len(self.vertices) - 1], len(self.vertices))

        # set current waypoint to goal if no path exists.
        if path == [] or count >= len(self.vertices):
            path = [VertexPRM(start_loc),VertexPRM(goal_loc)]

        x_path = []
        y_path = []
        for i, vert in enumerate(path):
            x_path.append(vert.location[0])
            y_path.append(vert.location[1])

        ax.plot(x_path,y_path,':')

        plt.show()

        return path



    def astar(self, start, goal, n_verts):

        open_lst = [start]
        closed_lst = []
        is_complete = False
        count = 0
        while len(open_lst) > 0 and not is_complete and count < n_verts + 1:
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
            #if len(path) == 0:

            # add the start node to the path
            path.append(start)

        return reversed(path), count

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

def main():

    sns.set_theme()
    np.random.seed(0)

    domain = [0,10]
    max_connect_dst = 3.0
    model_radius = 1.0
    n_samples = 200

    prm = RLPRM(domain,max_connect_dst,model_radius,n_samples)

    start_loc = [4.0,4.0]
    goal_loc = [9.0,9.0]
    #goal_loc = [15., 15.0]
    state = OrderedDict()
    state['psi'] = 0.0
    prm.build_prm(goal_loc,start_loc,state)


if __name__ == '__main__':

    main()
"""
Generates a space filling latin hyper cube sampling plan to use for DOEs. Python port from the forester text.
Still need to verify this is working as intended.
"""

# native modules
import copy
import random

# 3rd party modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns


def random_latin_hyper_cube(n_points,n_vars):
    """
    Generates a random Latin hypercube within the [0,1]^k hypercube.
    :param n_points: number of points in the sampling plan
    :param n_vars: number of dimensions/variables in the sampling plan
    :return: A randomly generated LHS
    """

    lhs = np.zeros((n_points,n_vars))

    # randomly fill the slots in the LHS
    for i in range(n_vars):
        lhs[:,i] = random.sample(range(n_points), n_points)

    # normalize points
    return lhs/(n_points-1)


def mmlhs(n_points,n_vars,lhs_init,pop_size,max_iter,q,p):
    """
    Evolutionary algorithm to search for the best space filling Latin Hyper Cube sampling plan.

    :param n_points: number of points in the plan
    :param n_vars: the number of variables in the plan
    :param lhs_init: The initial plan
    :param pop_size: population size for the lhs
    :param max_iter: maxiumum number of generations
    :param q: Values to optimze phi_q for
    :param p: distance power metric
    :return: optimized space filling latin hypercube sampling plan
    """

    lhs_best = copy.deepcopy(lhs_init)

    phi_best = mmphi(n_points,lhs_best, q, p)

    level_off = np.floor(0.85*max_iter)

    for i in range(max_iter):
        print('Iteration: {:d} of {:d}'.format(i+1,max_iter))
        if i < level_off:
            mutations = int(np.round(1+(0.5*n_points-1)*(level_off-i)/(level_off-1)))
        else:
            mutations = 1

        lhs_improved = copy.deepcopy(lhs_best)
        phi_improved = phi_best

        for offspring in range(pop_size):
            lhs_try = perturb(n_points,n_vars,lhs_best, mutations)
            phi_try = mmphi(n_points,lhs_try,q,p)

            if phi_try < phi_improved:
                lhs_improved = lhs_try
                phi_improved = phi_try

        if phi_improved < phi_best:
            lhs_best = copy.deepcopy(lhs_improved)
            phi_best = phi_improved

    return lhs_best


def mmphi(n_points,lhs, q, p):
    """
    Calculates the sampling plan quality criterion of Morris and Mitchell.

    :param lhs: sampling plan
    :param q: exponent used in the calculation of the metric
    :param p: distance metric factor(p=1 rectangular - default, p=2 Euclidean)
    :return: sampling plan `space-fillingness' metric
    """

    # Calculate the distances between all pairs of points (using the p-norm) and build multiplicity array J
    j, d = jD(n_points,lhs,p)

    return np.power(np.sum( np.dot( j,np.power(d,-q))  ),1.0/q)


def jD(n_points,lhs,p):
    """
    Computes the distances between all pairs of points in a sampling plan X using the p-norm, sorts them in
    ascending order and removes multiple occurences.

    :param n_points: The number of points in the sampling plan
    :param lhs: The sampling plan
    :param p: distance norm (p=1 rectangular - default, p=2 Euclidean)
    :return:
        multiplicity array (that is, the number of pairs separated by each distance value).
        list of distinct distance values
    """

    # Compute the distances between all pairs of points
    dst = np.zeros((int(n_points * (n_points - 1) / 2),))
    for i in range(1,n_points):
        for j in range(i+1,n_points+1):
            # Distance metric: p - norm
            idx = int(((i - 1) * n_points - (i - 1) * i / 2 + j - i)-1)
            dst[idx] = scipy.spatial.distance.minkowski(lhs[i-1,:],lhs[j-1,:], p)

    # remove the non-unqiue occurances
    distinct_dst = np.unique(dst)

    j = np.zeros_like(distinct_dst)
    for i, tmp_dst in enumerate(distinct_dst):
        j[i] = len([k for k in dst if k == distinct_dst[i]])

    return j, distinct_dst


def perturb(n_points,n_vars,lhs, n_pert):
    """
    Interchanges pairs of randomly chosen elements within randomly chosen columns of a sampling plan a number of times.
    If the plan is a Latin hypercube, the result of this operation will also be a Latin hypercube.

    :param n_points: number of points in the sampling plan
    :param n_vars: number of dimensions in the sampling plan
    :param lhs: sampling plan
    :param n_pert: number of perturbations to attempt
    :return: A mutated sampling plan
    """

    for i in range(n_pert):
        col = int(np.floor(np.random.uniform()*n_vars))

        # choose two distinct random points
        el1 = 0
        el2 = 0
        while el1 == el2:
            el1 = int(np.floor(np.random.uniform() * n_points))
            el2 = int(np.floor(np.random.uniform() * n_points))

        # swap the choosen elements
        lhs[el1,col] , lhs[el2,col] = lhs[el2,col], lhs[el1,col]

    return lhs


def mmsort(n_points, lhs_lst, p):
    """
    Ranks sampling plans according to the Morris-Mitchell criterion definition. Note: similar to phisort,
    which uses the numerical quality criterion Phiq as a basis for the ranking.
    :param n_points: number of points in the sampling plan
    :param lhs_lst: list of sampling plans
    :param p: distance metric exponent
    :return:
        best plan
        index of the best plan
    """

    idx = [i for i in range(len(lhs_lst))]

    # bubble sort
    swap_flag = 1

    while swap_flag == 1:
        swap_flag = 0
        i = 0
        while i < len(idx)-1:
            if mm(n_points, lhs_lst[idx[i]], lhs_lst[idx[i+1]], p) == 2:
                idx[i], idx[i+1] = idx[i+1], idx[i]
                swap_flag = 1
            i += 1

    return lhs_lst[idx[0]], idx


def mm(n_points, lhs_1, lhs_2, p):
    """
    Given two samplig plans chooses the one with the better space-filling properties
    (as per the Morris-Mitchell criterion).

    :param n_points: number of points in the sampling plan
    :param lhs_1: first sampling plan
    :param lhs_2: second sampling plan
    :param p: distance metric factor
    :return:
        if c = 0, identical plans or equally space-filling,
        if c = 1, lhs_1 is more space-filling,
        if c = 2, lhs_2 is more space-filling.
    """

    if (lhs_1 == lhs_2).all():
        # check if the plans are the same
        return 0

    j1, dst_1 = jD(n_points,lhs_1,p)
    m1 = len(dst_1)
    j2, dst_2 = jD(n_points, lhs_2, p)
    m2 = len(dst_2)

    v1 = np.zeros((int(2.0*np.max((m1,m2)))))
    for i, dst in enumerate(dst_1):
        v1[i*2] = dst
    for i, j in enumerate(j1):
        v1[i*2+1] = j

    v2 = np.zeros_like(v1)
    for i, dst in enumerate(dst_2):
        v2[i*2] = dst
    for i, j in enumerate(j2):
        v2[i * 2 + 1] = j

    m = np.min((m1,m2))
    v1 = v1[:m]
    v2 = v2[:m]

    c = (v1 > v2) + 2*(v1 < v2)

    if np.sum(c) == 0:
        return 0
    else:
        i = 0
        while c[i] == 0:
            i += 1
        return c[i]


def space_filling_latin_hyper_cube(n_points,n_vars, pop_size, max_iter):
    """
    Generates an optimized Latin hypercube by optimizing the Morris-Mitchell criterion for a range of exponents and
    plots the first two dimensions of the current hypercube throughout the optimization process. The resulting plan
    is saved to the same directory as this folder. If n_vars == 2, then a graph is created showing the plan.

    :param n_points: number of points in the sampling plan
    :param n_vars: number of dimensions/variables in the sampling plan
    :param pop_size: population size for each generation in the evolutionary optimization
    :param max_iter: maximum number of iterations to run
    :return:
    """

    if n_vars < 2:
        raise ValueError('Must have at least two dimensions for the LHS')

    # set seed
    random.seed(0)

    # list of q values to optimize Phi q for
    q = [1,2,5,10,20,50,100]

    # Set the distance norm to rectangular for a faster search. This can be
    # changed to p = 2 if the Euclidean norm is required.
    p = 1

    # build initial plan
    lhs_init = random_latin_hyper_cube(n_points,n_vars)

    # optimize for each q
    lhs_plans = []
    for i, q_tmp in enumerate(q):
        print('Optimizing for q of {:.1f}'.format(q_tmp))
        lhs_plans.append(mmlhs(n_points, n_vars,lhs_init,pop_size,max_iter,q_tmp,p))

    # get optimal plan by sorting the plans
    lhs_plan, order = mmsort(n_points, lhs_plans,p)

    # save the lhs
    df = pd.DataFrame(data=lhs_plan)
    df.to_csv('Npoints-' + str(n_points) + '_Nvars-' + str(n_vars)+'_sflhs.csv')

    if len(lhs_plan[0,:]) == 2:
        sns.set_theme()
        fig = plt.figure(0)
        ax = fig.add_subplot()
        ax.plot(lhs_plan[:,0],lhs_plan[:,1],'o')
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':

    n_points = 30
    n_vars = 3
    pop = 20
    max_iter = 500

    space_filling_latin_hyper_cube(n_points, n_vars, pop, max_iter)
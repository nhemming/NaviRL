"""
Script for working out how the adapative sample should work
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import experiments.OptimizeHyperParameters as ohp


def truth_func(x):
    adj = -1.0
    return adj*np.power(6.0*x-2.0,2.0)*np.sin(12.0*x-4.0)


if __name__ == '__main__':

    sns.set_theme()
    krig_hp = {'init_length_scale': 0.2,
               'init_variance': 0.05,
               'n_restarts': 10}
    nm_hp = {'n_restarts': 10}

    # create sample data
    np.random.seed(5)
    x_init = np.random.uniform(0,1,(5,))
    y_init = np.zeros_like(x_init)
    for i, x in enumerate(x_init):
        y_init[i] = truth_func(x)

    x_space = np.linspace(0,1,50)
    y_space = np.zeros_like(x_space)
    for i, x in enumerate(x_space):
        y_space[i] = truth_func(x)

    fig = plt.figure(0,figsize=(14,8))
    ax1 = fig.add_subplot(2,6,1)
    ax1.set_title('init pred')
    ax1.plot(x_space,y_space,label='truth')
    ax1.plot(x_init, y_init, 'o',label='samples')

    # build kriging method
    x_init_in = np.reshape(x_init,(len(x_init),1))
    model = ohp.build_krig(x_init_in,y_init,krig_hp)
    x_space_2 = np.reshape(x_space,(len(x_space),1))
    y_hat, std_hat = model.predict(x_space_2)

    ax1.plot(x_space,y_hat,label='pred')

    y_hat = np.reshape(y_hat,(len(y_hat),))
    std_hat = np.reshape(std_hat, (len(std_hat),))
    ax1.fill_between(x_space, y_hat + std_hat,  y_hat - std_hat, alpha=0.3)
    ax1.legend()

    # graph the expectation
    ei = np.zeros_like(x_space)
    for i, x in enumerate(x_space):
        x_tmp = np.reshape(x,(1,1))
        y_tmp, std_tmp = model.predict(x_tmp)
        ei[i] = ohp.get_expected_improvement(y_tmp,y_init.min(),std_tmp)

    ax2 = fig.add_subplot(2,6,7)
    ax2.plot(x_space,ei)
    ax2.set_title('Expected improvement init')

    # get the optimal max expected imporvement
    sol = ohp.get_max_expected_improvement(model, 1,y_init.min(), nm_hp, True)
    ax2.plot([sol,sol],[0,max(ei)],'--',label='max ei')
    ax2.legend()
    print(sol)

    # do infill
    n_infill = 5
    for i in range(n_infill):

        ax_tmp1 = fig.add_subplot(2,6,i+2)
        ax_tmp2 = fig.add_subplot(2, 6, i + 8)

        x_init = np.append(x_init,sol)
        y_init = np.append(y_init, truth_func(sol))
        x_init_in = np.reshape(x_init, (len(x_init), 1))
        model = ohp.build_krig(x_init_in, y_init, krig_hp)

        x_space_2 = np.reshape(x_space, (len(x_space), 1))
        y_hat, std_hat = model.predict(x_space_2)

        ax_tmp1.plot(x_space, y_space, label='truth')
        ax_tmp1.plot(x_init, y_init, 'o', label='samples')
        ax_tmp1.plot(x_space, y_hat, label='pred')

        y_hat = np.reshape(y_hat, (len(y_hat),))
        std_hat = np.reshape(std_hat, (len(std_hat),))
        ax_tmp1.fill_between(x_space, y_hat + std_hat, y_hat - std_hat, alpha=0.3)
        ax_tmp1.legend()

        ei = np.zeros_like(x_space)
        for i, x in enumerate(x_space):
            x_tmp = np.reshape(x, (1, 1))
            y_tmp, std_tmp = model.predict(x_tmp)
            ei[i] = ohp.get_expected_improvement(y_tmp, y_init.min(), std_tmp)

        ax_tmp2.plot(x_space, ei)
        ax_tmp2.set_title('Expected improvement init')

        sol = ohp.get_max_expected_improvement(model, 1, y_init.min(), nm_hp, True)
        ax_tmp2.plot([sol, sol], [0, max(ei)], '--', label='max ei')
        ax_tmp2.legend()

    plt.tight_layout()
    plt.show()
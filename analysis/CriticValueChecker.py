"""
One off script for checking the critic value given a state condition. Used to check if the critic is learning appropriatly.
"""

# native packages
import copy
from collections import Counter
import os

# 3rd party packages
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter
import seaborn as sns
import torch

# own modules
from environment.NavigationEnvironment import NavigationEnvironment
from environment.Reward import ReachDestinationReward
from environment.Sensor import DestinationSensor


def load_environment(base_folder, set_name, trial_num):
    abs_path = os.getcwd().replace('\\analysis', '\\experiments')
    base_dir = os.path.join(abs_path, base_folder)
    base_dir = os.path.join(base_dir, 'output')
    base_dir = os.path.join(base_dir, set_name)
    base_dir = os.path.join(base_dir, str(trial_num))
    file_name = os.path.join(base_dir, 'hyper_parameters.yaml')

    env = NavigationEnvironment()
    env.build_env_from_yaml(file_name, '', False)
    return env


def main():
    # set experiments to evaluate
    base_folder = 'demo_to_test_DQN'
    set_name = 'DebugDQN'
    trial_num = 2

    abs_path = os.getcwd().replace('\\analysis', '\\experiments')
    base_dir = os.path.join(abs_path, base_folder)
    base_dir = os.path.join(base_dir, 'output')
    base_dir = os.path.join(base_dir, set_name)
    base_dir = os.path.join(base_dir, str(trial_num))

    env = load_environment(base_folder, set_name, trial_num)

    x_loc_entity = [0.0,0.0 ]
    y_loc_entity = [0.0,3.0]
    phi_headings = [0.0,0.0]
    x_loc_goal = 5.0
    y_loc_goal = 0.0

    entity = env.entities['learning_entity']
    goal = env.entities['destination']
    goal.state_dict['x_pos'] = x_loc_goal
    goal.state_dict['y_pos'] = y_loc_goal
    lrn_alg = env.agents['general_nav_0'].learning_algorithms['DQN_0']
    action_operation = env.agents['general_nav_0'].action_operation

    sns.set_theme()
    fig = plt.figure(0,figsize=(14,8))
    ax = fig.add_subplot(111)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    line_styles =['-',':','--','-.']

    for i in range(len(x_loc_entity)):

        entity.state_dict['x_pos'] = x_loc_entity[i]
        entity.state_dict['y_pos'] = y_loc_entity[i]
        entity.state_dict['phi'] = phi_headings[i]

        # get number of files in folder
        file_dir = os.path.join(base_dir, 'models')
        dir_list = os.listdir(file_dir)
        model_lst = [i for i in dir_list if 'DQN_0' in i]

        q_values = []
        # loop over models in time
        for model_path in model_lst:

            # set q network
            #print(lrn_alg.q_network)
            lrn_alg.q_network.load_state_dict(torch.load(os.path.join(file_dir,model_path)))

            # update sensors
            for name, tmp_sensor in env.sensors.items():
                tmp_sensor.update(0.0, env.entities, env.sensors)

            # get the q values
            lrn_alg.create_state_action(action_operation, env.entities, 0, env.sensors, 0.0, False)
            lrn_alg.last_reset_time = -np.infty

            q_values.append(lrn_alg.action_info['q_values'])

        q_values = np.reshape(q_values,(len(q_values),len(q_values[0])))

        for j in range(len(q_values[0,:])):
            ax.plot([i for i in range(len(q_values))],q_values[:,j],line_styles[j],color=colors[i],label=str(i)+'_'+str(j))

    ax.legend()
    plt.tight_layout()
    plt.show()






if __name__ == '__main__':

    main()
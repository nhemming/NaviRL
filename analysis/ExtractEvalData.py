"""
This script parses evaluation data and combines it into a succinct file.
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

# own modules
from environment.NavigationEnvironment import NavigationEnvironment
from environment.Reward import ReachDestinationReward
from environment.Sensor import DestinationSensor

def graph_results(data):

    sns.set_theme()

    # plot success rate
    fig = plt.figure(0,figsize=(14,8))
    ax1 = fig.add_subplot(111)
    ax1.plot(data['ep_num'],data['success'],label='mean')
    ax1.fill_between(data['ep_num'], data['success'] + data['success_std'],data['success'] - data['success_std'], facecolor='tab:blue', alpha=0.3, label='1 $\sigma$')
    # graph max line
    ax1.plot(data['ep_num'], data['success_max'], '-', color='tab:purple', label='Max')
    # graph min line
    ax1.plot(data['ep_num'], data['success_min'], '-', color='tab:green', label='Min')
    ax1.set_xlabel('Episode Number [-]')
    ax1.set_ylabel('Average Success Rate [-]')
    ax1.legend()
    plt.tight_layout()

    # plot crash rate
    fig = plt.figure(1, figsize=(14, 8))
    ax1 = fig.add_subplot(111)
    ax1.plot(data['ep_num'], data['crashed'],label='mean')
    ax1.fill_between(data['ep_num'], data['crashed'] + data['crashed_std'], data['crashed'] - data['crashed_std'],
                     facecolor='tab:blue', alpha=0.3, label='1 $\sigma$')
    # graph max line
    ax1.plot(data['ep_num'], data['crashed_max'], '-', color='tab:purple', label='Max')
    # graph min line
    ax1.plot(data['ep_num'], data['crashed_min'], '-', color='tab:green', label='Min')
    ax1.set_xlabel('Episode Number [-]')
    ax1.set_ylabel('Average Crash Rate [-]')
    ax1.legend()
    plt.tight_layout()

    # plot distance to reach goal
    fig = plt.figure(2, figsize=(14, 8))
    ax1 = fig.add_subplot(111)
    ax1.plot(data['ep_num'], data['distance_traveled[m]'],label='mean')
    ax1.fill_between(data['ep_num'], data['distance_traveled[m]'] + data['distance_traveled[m]_std'], data['distance_traveled[m]'] - data['distance_traveled[m]_std'],
                     facecolor='tab:blue', alpha=0.3, label='1 $\sigma$')

    # graph max line
    ax1.plot(data['ep_num'], data['distance_traveled[m]_max'], '-', color='tab:purple', label='Max')
    # graph min line
    ax1.plot(data['ep_num'], data['distance_traveled[m]_min'], '-', color='tab:green', label='Min')
    ax1.set_xlabel('Episode Number [-]')
    ax1.set_ylabel('Average Distance Traveled [m]')
    ax1.legend()
    plt.tight_layout()

    # plot normalized distance to reach goal
    fig = plt.figure(3, figsize=(14, 8))
    ax1 = fig.add_subplot(111)
    ax1.plot(data['ep_num'], data['normalized_distance_traveled'],label='mean')
    ax1.fill_between(data['ep_num'], data['normalized_distance_traveled'] + data['normalized_distance_traveled_std'],
                     data['normalized_distance_traveled'] - data['normalized_distance_traveled_std'],
                     facecolor='tab:blue', alpha=0.3, label='1 $\sigma$')

    # graph max line
    ax1.plot(data['ep_num'], data['normalized_distance_traveled_max'],'-',color='tab:purple',label='Max')
    # graph min line
    ax1.plot(data['ep_num'], data['normalized_distance_traveled_min'], '-', color='tab:green',label='Min')

    # TODO add timeout exclusion graph also.
    ax1.plot([0,data['ep_num'].max()],[1.0,1.0],'--',color='tab:gray',label='Crow Flies Distance')
    ax1.set_xlabel('Episode Number [-]')
    ax1.set_ylabel('Normalized Average Distance Traveled [-]')
    ax1.legend()
    plt.tight_layout()

    plt.show()

def load_environment(base_folder, set_name, trial_num):
    file_name = os.path.join(base_folder, 'initial_condition_eval_set.yaml')

    env = NavigationEnvironment()
    env.build_eval_env_from_yaml(file_name,os.getcwd(), create=False) # TODO investigate correct input dir
    return env

def extract_data():

    base_folder = 'demo_to_test_DDPG'
    set_name = 'DebugDDPGRLPRM'
    trial_num = 10
    eval_trial_num = 0

    """
    Edit above ^^
    """

    abs_path = os.getcwd().replace('\\analysis', '\\experiments')
    base_dir = os.path.join(abs_path, base_folder)
    base_dir = os.path.join(base_dir, 'output')
    base_dir = os.path.join(base_dir, set_name)
    base_dir = os.path.join(base_dir, str(trial_num))
    base_dir = os.path.join(base_dir, "evaluation")
    base_dir = os.path.join(base_dir, str(eval_trial_num))


    env = load_environment(base_dir, set_name, trial_num)

    # get the names of the agents
    file_dir = os.path.join(base_dir, 'learning_algorithm')
    dir_list = os.listdir(file_dir)
    dir_list = [i for i in dir_list if 'loss' not in i]
    for i, file in enumerate(dir_list):
        dir_list[i] = file.split('_epnum')[0]

    agent_names = list(Counter(dir_list).keys())

    # get file path to sensors
    file_path_sensors = file_dir.replace('learning_algorithm','sensors')
    file_path_entities = file_dir.replace('learning_algorithm', 'entities')

    # get goal distance
    for name, reward_comp in env.reward_function.reward_components.items():
        if isinstance(reward_comp, ReachDestinationReward):
            goal_dst = reward_comp.goal_dst

    # Loop over each initial contidion
    ic_dict = dict()
    for ic_num in range(env.eval_set_size):
    #for ic_num in range(16,20):

        cols = ['ep_num','success','crashed','distance_traveled[m]','normalized_distance_traveled']#,'amount_of_data_seen','amount_of_data_trained']
        df_ic = pd.DataFrame(data=np.zeros((len(env.model_nums),len(cols))),columns=cols)

        # loop over episode number
        for j, ep_num in enumerate(env.model_nums):

            # TODO loop over agents

            # open agent file
            #
            df_ic['ep_num'].iloc[j] = ep_num

            # determine if the agent succeeded in navigation
            file_name = os.path.join(file_path_sensors,'destination_sensor_0_epnum-' + str(ep_num) + '-evalnum' + str(ic_num) + '.csv')
            tmp_df = pd.read_csv(file_name)
            min_dst = tmp_df['distance'].min()
            success = 0.0
            if min_dst <= goal_dst:
                success = 1.0
            df_ic['success'].iloc[j] = success

            # get distance traveled from entity

            entity_name = ''
            for tmp_name, tmp_value in env.agents.items():
                if tmp_value.name in agent_names[0]:
                    entity_name = tmp_value.controlled_entity
                    break

            file_name = os.path.join(file_path_entities,
                                     entity_name + '_epnum-' + str(ep_num) + '-evalnum' + str(ic_num) + '.csv')
            tmp_df = pd.read_csv(file_name)
            total_dst = 0.0
            #result = [total_dst + np.sqrt() for x, y in zip(tmp_df['col1'], tmp_df['col2'])]
            #result = [total_dst + np.sqrt() for i, row in enumerate(tmp_df[['x_pos','y_pos']].to_numpy())]
            for k in range(len(tmp_df)): # TODO find a way to use list comprehensions or vectorization
                if k != 0:
                    total_dst += np.sqrt( (tmp_df['x_pos'].iloc[k]-tmp_df['x_pos'].iloc[k-1])**2  + (tmp_df['y_pos'].iloc[k]-tmp_df['y_pos'].iloc[k-1])**2 )
            df_ic['distance_traveled[m]'].iloc[j] = total_dst

            # init location
            x_init = tmp_df['x_pos'].iloc[0]
            y_init = tmp_df['y_pos'].iloc[0]
            #print(x_init,y_init)
            # goal location
            file_name = os.path.join(file_path_entities,
                                    'destination_epnum-' + str(ep_num) + '-evalnum' + str(ic_num) + '.csv')
            tmp_df = pd.read_csv(file_name)
            x_goal = tmp_df['x_pos'].iloc[0]
            y_goal = tmp_df['y_pos'].iloc[0]
            org_dst = np.sqrt((x_goal-x_init)**2+(y_goal-y_init)**2)
            df_ic['normalized_distance_traveled'].iloc[j] = total_dst/org_dst

        ic_dict[ic_num] = df_ic

    # save the data

    # generate overall average data
    #df_concat = pd.concat((ic_dict[0], ic_dict[1]))
    df_concat = pd.DataFrame()
    for name, value in ic_dict.items():
        df_concat = pd.concat((df_concat,value))
    by_row_index = df_concat.groupby(df_concat.index)
    df_means = by_row_index.mean()
    df_std = by_row_index.std()
    df_max = by_row_index.max()
    df_min = by_row_index.min()
    df_std_col_names = list(df_std.columns)
    df_std_col_names_std = [i + '_std' for i in df_std_col_names]
    df_max_col_names_max = [i+'_max' for i in df_std_col_names]
    df_min_col_names_min = [i + '_min' for i in df_std_col_names]
    name_dict = {}
    for i, _ in enumerate(df_std_col_names_std):
        name_dict[df_std_col_names[i]] = df_std_col_names_std[i]
    df_std = df_std.rename(columns=name_dict)
    df_avg = pd.concat((df_means,df_std),axis=1)
    name_dict = {}
    for i, _ in enumerate(df_max_col_names_max):
        name_dict[df_std_col_names[i]] = df_max_col_names_max[i]
    df_max = df_max.rename(columns=name_dict)
    df_avg = pd.concat((df_avg, df_max), axis=1)
    name_dict = {}
    for i, _ in enumerate(df_min_col_names_min):
        name_dict[df_std_col_names[i]] = df_min_col_names_min[i]
    df_min = df_min.rename(columns=name_dict)
    df_avg = pd.concat((df_avg, df_min), axis=1)



    # TODO move to seperate script once complete
    graph_results(df_avg)

if __name__ == '__main__':

    # extract data
    extract_data()

    # combine data that needs to be combined
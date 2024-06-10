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
from scipy.integrate import trapezoid
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter
import seaborn as sns

# own modules
from environment.NavigationEnvironment import NavigationEnvironment
from environment.Reward import ReachDestinationReward
from environment.Sensor import DestinationSensor
from environment.Termination import ReachDestinationTermination


def load_environment(base_folder, set_name, trial_num):
    file_name = os.path.join(base_folder, 'initial_condition_eval_set.yaml')

    env = NavigationEnvironment()
    env.build_eval_env_from_yaml(file_name,os.getcwd(), create=False) # TODO investigate correct input dir
    return env

def add_window_distributions(df_avg, agg_lst, field):
    windows = [5, 10, 20]
    for win in windows:
        avg = uniform_filter1d(df_avg[field], win)

        # TODO remove after testing
        '''
        sns.set_theme()
        fig = plt.figure(0)
        plt.plot(df_avg['ep_num'],df_avg[field])
        plt.plot(df_avg['ep_num'],avg)
        '''

        min_idx = np.argmin(avg)

        half = int(win / 2.0)
        if len(avg) - half < min_idx:
            # close to the end of the sequence
            stop = len(avg)
            start = len(avg) - win
        elif min_idx < win:
            # too close to teh start
            start = 0
            stop = win - 1
        else:
            start = win - half + min_idx
            stop = win + half+1 + min_idx
        win_data = df_avg[field][start:stop]
        avg_dst = np.mean(win_data)
        std_dst = np.std(win_data)

        tmp_dict = {'name': field+'_' + str(win) + '_min', 'value': np.min(avg)}
        agg_lst.append(tmp_dict)
        tmp_dict = {'name':field+'_' + str(win) + '_avg', 'value':  avg_dst}
        agg_lst.append(tmp_dict)
        tmp_dict = {'name':field+'_' + str(win) + '_std', 'value':  std_dst}
        agg_lst.append(tmp_dict)


def extract_data(base_folder, set_name, trial_num, eval_trial_num):

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
    goal_dst = None
    for name, reward_comp in env.reward_function.reward_components.items():
        if isinstance(reward_comp, ReachDestinationReward):
            goal_dst = reward_comp.goal_dst

    if goal_dst is None:
        # search in termination function for non-learning agent
        for name, term_func in env.termination_function.components.items():
            if isinstance(term_func, ReachDestinationTermination):
                goal_dst = term_func.goal_dst

    # Loop over each initial contidion
    ic_dict = dict()
    for ic_num in range(env.eval_set_size):
    #for ic_num in range(0,3):

        print("Extracting IC : {:d}".format(ic_num))

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

    # save the data
    df_avg.to_csv(os.path.join(base_dir,'OverallResults.csv'),index=False)

    # build aggregate metrics
    if len(df_avg) == 1:
        aoc = float(df_avg['success'].iloc[0])
    else:
        aoc = trapezoid(y=df_avg['success'], x=df_avg['ep_num'])
    min_norm_dst_travel = df_avg['normalized_distance_traveled'].min()
    min_dst_travel = df_avg['distance_traveled[m]'].min()
    success = list(df_avg['success'])

    window = 0
    max_window = 0
    for i, s in enumerate(success):

        if not np.abs(s - 1.0) < 1e-5:
            # update window
            window = 0
        else:
            window += 1

        if window > max_window:
            max_window = window
    longest_success_rate = max_window

    aggregate_dict_0 = {'name': 'AOC', 'value': aoc}
    aggregate_dict_1 = {'name': 'min_norm_dst_travel', 'value': min_norm_dst_travel}
    aggregate_dict_2 = {'name': 'min_dst_travel', 'value': min_dst_travel}
    aggregate_dict_3 = {'name': 'longest_success_rate', 'value': longest_success_rate}
    agg_lst = [aggregate_dict_0, aggregate_dict_1, aggregate_dict_2, aggregate_dict_3]

    # get average over sliding window with std.
    add_window_distributions(df_avg, agg_lst, 'distance_traveled[m]')
    add_window_distributions(df_avg, agg_lst, 'normalized_distance_traveled')


    df_agg = pd.DataFrame(agg_lst)
    df_agg.to_csv(os.path.join(base_dir,'AggregateResults.csv'),index=False)


if __name__ == '__main__':

    base_folder = 'tune_boat_bspline_DDPG_Sparse'
    set_name = 'DDPGBSplineSparse'
    trial_num = 107
    eval_trial_num = 0

    """
    Edit above ^^
    """

    # extract data
    extract_data(base_folder, set_name, trial_num, eval_trial_num)
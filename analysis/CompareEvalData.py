"""
This script creates graphs for comparing models over the evaluation set or for graphing only one models performance.
"""

# native modules
import os

# 3rd party modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def get_color(idx):
    colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan','k']
    return colors[idx % len(colors)]

def graph_frame(frame, frame_name, ax, x_col_name, y_col_name, len_frames, graph_std, k, max_frame=0):
    if len_frames == 1:
        label = 'mean'
    else:
        label = frame_name
    if len(frame) == 1:
        ax.plot([0, max_frame], np.full((2,), frame[y_col_name]), label=label, color=get_color(k))
    else:
        ax.plot(frame[x_col_name], frame[y_col_name], label=label, color=get_color(k))

    if graph_std and len(frame) > 1:
        if len(frames) == 1:
            label = '1 $\sigma$'
        else:
            label = ''
        ax.fill_between(frame[x_col_name], frame[y_col_name] + frame[y_col_name+'_std'],
                         frame[y_col_name] - frame[y_col_name+'_std'],
                         facecolor=get_color(k), alpha=0.3, label=label)

    if len_frames == 1 and len(frame) > 1:
        # graph max line
        ax.plot(frame[x_col_name], frame[y_col_name+'_max'], '-', color='tab:purple', label='Max')
        # graph min line
        ax.plot(frame[x_col_name], frame[y_col_name+'_min'], '-', color='tab:green', label='Min')


def create_graphs(sets_to_compare, graph_std, include_training_data):

    # load the data files
    frames = dict()
    for name, value in sets_to_compare.items():
        # open data file
        path = '..\\'
        folders_to_add = ['experiments',value['base_folder'],'output',value['set_name'],value['trial_num'],'evaluation',value['eval_trial_num'],'OverallResults.csv']
        for fta in folders_to_add:
            path = os.path.join(path,str(fta))
        frames[name] = pd.read_csv(path)

    # load aggregate data
    aggregate = dict()
    for name, value in sets_to_compare.items():
        # open data file
        path = '..\\'
        folders_to_add = ['experiments',value['base_folder'],'output',value['set_name'],value['trial_num'],'evaluation',value['eval_trial_num'],'AggregateResults.csv']
        for fta in folders_to_add:
            path = os.path.join(path,str(fta))
        aggregate[name] = pd.read_csv(path)

    # get the maximum amount episodes
    max_frame = 0
    for frame_name, frame in frames.items():
        tmp_max = frame['ep_num'].max()
        if tmp_max > max_frame:
            max_frame = tmp_max

    # graph the success rate
    sns.set_theme()
    fig = plt.figure(0, figsize=(8, 6))
    ax = fig.add_subplot(111)
    k = 0
    x_col_name = 'ep_num'
    y_col_name = 'success'
    for frame_name, frame in frames.items():
        graph_frame(frame, frame_name, ax, x_col_name, y_col_name, len(frames), graph_std, k, max_frame)
        k += 1
    ax.set_xlabel('Episode Number [-]')
    ax.set_ylabel('Average Success Rate [-]')
    ax.legend()
    plt.tight_layout()

    # graph crash rate
    fig = plt.figure(1, figsize=(8, 6))
    ax = fig.add_subplot(111)
    k = 0
    x_col_name = 'ep_num'
    y_col_name = 'crashed'
    for frame_name, frame in frames.items():
        graph_frame(frame, frame_name, ax, x_col_name, y_col_name, len(frames), graph_std, k, max_frame)
        k += 1
    ax.set_xlabel('Episode Number [-]')
    ax.set_ylabel('Average Crash Rate [-]')
    ax.legend()
    plt.tight_layout()

    # plot distance to reach goal
    fig = plt.figure(2, figsize=(8, 6))
    ax = fig.add_subplot(111)
    k = 0
    x_col_name = 'ep_num'
    y_col_name = 'distance_traveled[m]'
    for frame_name, frame in frames.items():
        graph_frame(frame, frame_name, ax, x_col_name, y_col_name, len(frames), graph_std, k, max_frame)
        k += 1
    ax.set_xlabel('Episode Number [-]')
    ax.set_ylabel('Average Distance Traveled [m]')
    ax.legend()
    plt.tight_layout()

    # plot normalized distance to reach goal
    fig = plt.figure(3, figsize=(8, 6))
    ax = fig.add_subplot(111)
    k = 0
    x_col_name = 'ep_num'
    y_col_name = 'normalized_distance_traveled'
    for frame_name, frame in frames.items():
        graph_frame(frame, frame_name, ax, x_col_name, y_col_name, len(frames), graph_std, k, max_frame)
        k += 1

    # TODO add timeout exclusion graph also.
    ax.plot([0, max_frame], [1.0, 1.0], '--', color='tab:gray', label='Crow Flies Distance')
    ax.set_xlabel('Episode Number [-]')
    ax.set_ylabel('Normalized Average Distance Traveled [-]')
    ax.legend()
    plt.tight_layout()

    # plot bar chart of path length
    #bar_df = pd.DataFrame(columns=['name','AOC','longest success rate','min average distance traveled [m]',''])
    bar_dict_lst = []
    for agg_name, agg in aggregate.items():
        tmp_dict = {'name':agg_name,
                    'AOC':agg.loc[agg['name'] == 'AOC'].iloc[0]['value'],
                    'Minimum Average Distance Traveled':agg.loc[agg['name'] == 'min_dst_travel'].iloc[0]['value'],
                    'Longest Success Rate':agg.loc[agg['name'] == 'longest_success_rate'].iloc[0]['value'],
                    'Distance Traveled 5 Min':agg.loc[agg['name'] == 'distance_traveled[m]_5_min'].iloc[0]['value'],
                    'Distance Traveled 5 Avg':agg.loc[agg['name'] == 'distance_traveled[m]_5_avg'].iloc[0]['value'],
                    'Distance Traveled 5 Std':agg.loc[agg['name'] == 'distance_traveled[m]_5_std'].iloc[0]['value'],
                    'Distance Traveled 10 Min': agg.loc[agg['name'] == 'distance_traveled[m]_10_min'].iloc[0]['value'],
                    'Distance Traveled 10 Avg': agg.loc[agg['name'] == 'distance_traveled[m]_10_avg'].iloc[0]['value'],
                    'Distance Traveled 10 Std': agg.loc[agg['name'] == 'distance_traveled[m]_10_std'].iloc[0]['value'],
                    'Distance Traveled 20 Min': agg.loc[agg['name'] == 'distance_traveled[m]_20_min'].iloc[0]['value'],
                    'Distance Traveled 20 Avg': agg.loc[agg['name'] == 'distance_traveled[m]_20_avg'].iloc[0]['value'],
                    'Distance Traveled 20 Std': agg.loc[agg['name'] == 'distance_traveled[m]_20_std'].iloc[0]['value']
                    }
        bar_dict_lst.append(tmp_dict)

    bar_df = pd.DataFrame(bar_dict_lst)

    ind = np.arange(len(bar_df))  # the x locations for the groups
    width = 0.5  # the width of the bars

    fig = plt.figure(4, figsize=(8, 6))
    ax = fig.add_subplot(111)
    bars = ax.bar(ind,bar_df['Minimum Average Distance Traveled'], width)
    ax.bar_label(bars, fmt='%.2f')
    ax.set_ylabel('Minimum Average Distance Traveled in Evaluation Set [m]')
    ax.set_xticks(ind)
    ax.set_xticklabels(bar_df['name'].values)
    plt.tight_layout()

    fig = plt.figure(5, figsize=(8, 6))
    ax = fig.add_subplot(111)
    bars = ax.bar(ind, bar_df['AOC'], width)
    ax.bar_label(bars, fmt='%.2f')
    ax.set_ylabel('Area Under The Curve For Success [AOC]')
    ax.set_xticks(ind)
    ax.set_xticklabels(bar_df['name'].values)
    plt.tight_layout()

    fig = plt.figure(6, figsize=(8, 6))
    ax = fig.add_subplot(111)
    bars = ax.bar(ind, bar_df['Longest Success Rate'], width)
    ax.bar_label(bars, fmt='%.2f')
    ax.set_ylabel('Longest Success Rate')
    ax.set_xticks(ind)
    ax.set_xticklabels(bar_df['name'].values)
    plt.tight_layout()

    width = 0.25
    fig = plt.figure(8, figsize=(8, 6))
    ax = fig.add_subplot(111)
    rects1 = ax.bar(ind - width , bar_df['Distance Traveled 5 Avg'], width, yerr=bar_df['Distance Traveled 5 Std'],label='Distance Traveled 5')
    rects2 = ax.bar(ind , bar_df['Distance Traveled 10 Avg'], width, yerr=bar_df['Distance Traveled 10 Std'],
                    label='Distance Traveled 10')
    rects3 = ax.bar(ind + width , bar_df['Distance Traveled 20 Avg'], width, yerr=bar_df['Distance Traveled 20 Std'],
                    label='Distance Traveled 20')
    ax.set_ylabel('Average Distance Traveled Over Best Window')
    ax.set_xticks(ind)
    ax.set_xticklabels(bar_df['name'].values)
    ax.legend()
    plt.tight_layout()

    # ------------------------------------------------------------------------------------------------------------------
    # graph learning efficinces
    # ------------------------------------------------------------------------------------------------------------------
    frames_training = dict()
    for name, value in sets_to_compare.items():
        # open data file
        path = '..\\'
        folders_to_add = ['experiments', value['base_folder'], 'output', value['set_name'], value['trial_num'],
                          'progress', 'Training_summary.csv']
        for fta in folders_to_add:
            path = os.path.join(path, str(fta))
        frames_training[name] = pd.read_csv(path)

    # drop data from training frames not needed in evaluation set
    for tmp_name, tmp_frame in frames.items():
        eval_nums = list(tmp_frame['ep_num'])
        tmp_training_frame = frames_training[tmp_name]
        #frames_training[tmp_name] = tmp_training_frame.drop(tmp_training_frame[tmp_training_frame['ep_num'] in eval_nums].index)
        frames_training[tmp_name] = tmp_training_frame[tmp_training_frame['ep_num'].isin(eval_nums)]

    max_data_gen = 0
    for frame_name, frame in frames_training.items():
        tmp_max = frame['data_generated'].max()
        if tmp_max > max_data_gen:
            max_data_gen = tmp_max

    # graph training efficiencies for success rate and distance to reach the goal.
    fig = plt.figure(9, figsize=(8, 6))
    ax = fig.add_subplot(111)
    for tmp_name, tmp_frame in frames.items():
        if len(tmp_frame) == 1:
            ax.semilogx([0,max_data_gen], [tmp_frame['success'],tmp_frame['success']], label=tmp_name)
        else:
            ax.semilogx(frames_training[tmp_name]['data_generated'],tmp_frame['success'],label=tmp_name)

    ax.legend()
    ax.set_xlabel('Amount of Data Generated in Training[-]')
    ax.set_ylabel('Average Success Rate [-]')
    plt.tight_layout()

    fig = plt.figure(10, figsize=(8, 6))
    ax = fig.add_subplot(111)
    for tmp_name, tmp_frame in frames.items():
        if len(tmp_frame) == 1:
            ax.semilogx([0,max_data_gen], [tmp_frame['success'],tmp_frame['distance_traveled[m]']], label=tmp_name)
        else:
            ax.semilogx(frames_training[tmp_name]['data_generated'],tmp_frame['distance_traveled[m]'],label=tmp_name)

    ax.legend()
    ax.set_xlabel('Amount of Data Generated in Training[-]')
    ax.set_ylabel('Average Distance Travelled [m]')
    plt.tight_layout()

    max_data_train = 0
    for frame_name, frame in frames_training.items():
        tmp_max = frame['data_trained_on'].max()
        if tmp_max > max_data_train:
            max_data_train = tmp_max

    fig = plt.figure(11, figsize=(8, 6))
    ax = fig.add_subplot(111)
    for tmp_name, tmp_frame in frames.items():
        if len(tmp_frame) == 1:
            ax.semilogx([0,max_data_train], [tmp_frame['success'],tmp_frame['success']], label=tmp_name)
        else:
            ax.semilogx(frames_training[tmp_name]['data_trained_on'],tmp_frame['success'],label=tmp_name)

    ax.legend()
    ax.set_xlabel('Amount of Data Seen by Network [-]')
    ax.set_ylabel('Average Success Rate [-]')
    plt.tight_layout()

    fig = plt.figure(12, figsize=(8, 6))
    ax = fig.add_subplot(111)
    for tmp_name, tmp_frame in frames.items():
        if len(tmp_frame) == 1:
            ax.semilogx([0,max_data_train], [tmp_frame['success'],tmp_frame['distance_traveled[m]']], label=tmp_name)
        else:
            ax.semilogx(frames_training[tmp_name]['data_trained_on'],tmp_frame['distance_traveled[m]'],label=tmp_name)

    ax.legend()
    ax.set_xlabel('Amount of Data Seen by Network [-]')
    ax.set_ylabel('Average Distance Travelled [m]')
    plt.tight_layout()

    max_data_grad = 0
    for frame_name, frame in frames_training.items():
        tmp_max = frame['n_grad_steps'].max()
        if tmp_max > max_data_grad:
            max_data_grad = tmp_max

    fig = plt.figure(13, figsize=(8, 6))
    ax = fig.add_subplot(111)
    for tmp_name, tmp_frame in frames.items():
        if len(tmp_frame) == 1:
            ax.semilogx([0, max_data_grad], [tmp_frame['success'], tmp_frame['success']], label=tmp_name)
        else:
            ax.semilogx(frames_training[tmp_name]['n_grad_steps'], tmp_frame['success'], label=tmp_name)

    ax.legend()
    ax.set_xlabel('Number of Gradient Steps [-]')
    ax.set_ylabel('Average Success Rate [-]')
    plt.tight_layout()

    fig = plt.figure(14, figsize=(8, 6))
    ax = fig.add_subplot(111)
    for tmp_name, tmp_frame in frames.items():
        if len(tmp_frame) == 1:
            ax.semilogx([0, max_data_grad], [tmp_frame['success'], tmp_frame['distance_traveled[m]']], label=tmp_name)
        else:
            ax.semilogx(frames_training[tmp_name]['n_grad_steps'], tmp_frame['distance_traveled[m]'], label=tmp_name)

    ax.legend()
    ax.set_xlabel('Number of Gradient Steps [-]')
    ax.set_ylabel('Average Distance Travelled [m]')
    plt.tight_layout()

    '''
    if include_training_data:
        # ------------------------------------------------------------------------------------------------------------------
        # graph training progress of training episodes
        # ------------------------------------------------------------------------------------------------------------------

        frames = dict()
        for name, value in sets_to_compare.items():
            # open data file
            path = '..\\'
            folders_to_add = ['experiments', value['base_folder'], 'output', value['set_name'], value['trial_num'],
                              'progress', 'Training_summary.csv']
            for fta in folders_to_add:
                path = os.path.join(path, str(fta))
            frames[name] = pd.read_csv(path)

        max_frame = 0
        for frame_name, frame in frames.items():
            tmp_max = frame['ep_num'].max()
            if tmp_max > max_frame:
                max_frame = tmp_max

        fig = plt.figure(4, figsize=(14, 8))
        ax = fig.add_subplot(111)
        k = 0
        x_col_name = 'ep_num'
        y_col_name = 'success'
        for frame_name, frame in frames.items():
            graph_frame(frame, frame_name, ax, x_col_name, y_col_name, len(frames), graph_std, k)
            k += 1
        ax.set_xlabel('Episode Number [-]')
        ax.set_ylabel('Average Success Rate Over Training [-]')
        ax.legend()
        plt.tight_layout()

        fig_num = 5
        window_sizes = [21,51,101,201] # TODO convert to scan the data frame for windowsizes used instead of hard codeing
        for ws in window_sizes:
            fig = plt.figure(fig_num, figsize=(14, 8))
            ax = fig.add_subplot(111)
            k = 0
            x_col_name = 'ep_num'
            y_col_name = 'success_avg_'+str(ws)
            for frame_name, frame in frames.items():
                graph_frame(frame, frame_name, ax, x_col_name, y_col_name, len(frames), graph_std, k, max_frame)
                k += 1
            ax.set_xlabel('Episode Number [-]')
            ax.set_ylabel('Smoothed Average Success Rate Over Training with Window Size = '+str(ws)+' [-]')
            ax.legend()
            plt.tight_layout()

            fig_num +=1

        fig = plt.figure(fig_num, figsize=(14, 8))
        ax = fig.add_subplot(111)
        k = 0
        x_col_name = 'data_generated'
        y_col_name = 'success'
        for frame_name, frame in frames.items():
            graph_frame(frame, frame_name, ax, x_col_name, y_col_name, len(frames), graph_std, k)
            k += 1
        ax.set_xlabel('Number of MDP Tuples Generated [-]')
        ax.set_ylabel('Average Success Rate Over Training [-]')
        ax.legend()
        plt.tight_layout()
        fig_num += 1

        for ws in window_sizes:
            fig = plt.figure(fig_num, figsize=(14, 8))
            ax = fig.add_subplot(111)
            k = 0
            x_col_name = 'data_generated'
            y_col_name = 'success_avg_'+str(ws)
            for frame_name, frame in frames.items():
                graph_frame(frame, frame_name, ax, x_col_name, y_col_name, len(frames), graph_std, k, max_frame)
                k += 1
            ax.set_xlabel('Number of MDP Tuples Generated [-]')
            ax.set_ylabel('Smoothed Average Success Rate Over Training with Window Size = '+str(ws)+' [-]')
            ax.legend()
            plt.tight_layout()

            fig_num +=1

        fig = plt.figure(fig_num, figsize=(14, 8))
        ax = fig.add_subplot(111)
        k = 0
        x_col_name = 'data_trained_on'
        y_col_name = 'success'
        for frame_name, frame in frames.items():
            graph_frame(frame, frame_name, ax, x_col_name, y_col_name, len(frames), graph_std, k)
            k += 1
        ax.set_xlabel('Number of MDP Tuples Trained On [-]')
        ax.set_ylabel('Average Success Rate Over Training [-]')
        ax.legend()
        plt.tight_layout()
        fig_num += 1

        for ws in window_sizes:
            fig = plt.figure(fig_num, figsize=(14, 8))
            ax = fig.add_subplot(111)
            k = 0
            x_col_name = 'data_trained_on'
            y_col_name = 'success_avg_'+str(ws)
            for frame_name, frame in frames.items():
                graph_frame(frame, frame_name, ax, x_col_name, y_col_name, len(frames), graph_std, k, max_frame)
                k += 1
            ax.set_xlabel('Number of MDP Tuples Trained On [-]')
            ax.set_ylabel('Smoothed Average Success Rate Over Training with Window Size = '+str(ws)+' [-]')
            ax.legend()
            plt.tight_layout()

            fig_num +=1


            fig = plt.figure(fig_num, figsize=(14, 8))
            ax = fig.add_subplot(111)
            k = 0
            x_col_name = 'ep_num'
            y_col_name = 'crash'
            for frame_name, frame in frames.items():
                graph_frame(frame, frame_name, ax, x_col_name, y_col_name, len(frames), graph_std, k)
                k += 1
            ax.set_xlabel('Episode Number [-]')
            ax.set_ylabel('Average Crash Rate Over Training [-]')
            ax.legend()
            plt.tight_layout()
            fig_num += 1

        window_sizes = [21, 51, 101,
                        201]  # TODO convert to scan the data frame for windowsizes used instead of hard codeing
        for ws in window_sizes:
            fig = plt.figure(fig_num, figsize=(14, 8))
            ax = fig.add_subplot(111)
            k = 0
            x_col_name = 'ep_num'
            y_col_name = 'crash_avg_' + str(ws)
            for frame_name, frame in frames.items():
                graph_frame(frame, frame_name, ax, x_col_name, y_col_name, len(frames), graph_std, k, max_frame)
                k += 1
            ax.set_xlabel('Episode Number [-]')
            ax.set_ylabel('Smoothed Average Crash Rate Over Training with Window Size = ' + str(ws) + ' [-]')
            ax.legend()
            plt.tight_layout()

            fig_num += 1

        fig = plt.figure(fig_num, figsize=(14, 8))
        ax = fig.add_subplot(111)
        k = 0
        x_col_name = 'ep_num'
        y_col_name = 'dst_traveled[m]'
        for frame_name, frame in frames.items():
            graph_frame(frame, frame_name, ax, x_col_name, y_col_name, len(frames), graph_std, k)
            k += 1
        ax.set_xlabel('Episode Number [-]')
        ax.set_ylabel('Average Distance Traveled Over Training [m]')
        ax.legend()
        plt.tight_layout()
        fig_num += 1

        window_sizes = [21, 51, 101,
                        201]  # TODO convert to scan the data frame for windowsizes used instead of hard codeing
        for ws in window_sizes:
            fig = plt.figure(fig_num, figsize=(14, 8))
            ax = fig.add_subplot(111)
            k = 0
            x_col_name = 'ep_num'
            y_col_name = 'dst_traveled[m]_avg_' + str(ws)
            for frame_name, frame in frames.items():
                graph_frame(frame, frame_name, ax, x_col_name, y_col_name, len(frames), graph_std, k, max_frame)
                k += 1
            ax.set_xlabel('Episode Number [-]')
            ax.set_ylabel('Average Distance Traveled Over Training with Window Size = ' + str(ws) + ' [m]')
            ax.legend()
            plt.tight_layout()

            fig_num += 1

        fig = plt.figure(fig_num, figsize=(14, 8))
        ax = fig.add_subplot(111)
        k = 0
        x_col_name = 'ep_num'
        y_col_name = 'reward'
        for frame_name, frame in frames.items():
            graph_frame(frame, frame_name, ax, x_col_name, y_col_name, len(frames), graph_std, k)
            k += 1
        ax.set_xlabel('Episode Number [-]')
        ax.set_ylabel('Average Reward Over Training [-]')
        ax.legend()
        plt.tight_layout()
        fig_num += 1

        window_sizes = [21, 51, 101,
                        201]  # TODO convert to scan the data frame for windowsizes used instead of hard codeing
        for ws in window_sizes:
            fig = plt.figure(fig_num, figsize=(14, 8))
            ax = fig.add_subplot(111)
            k = 0
            x_col_name = 'ep_num'
            y_col_name = 'reward_avg_' + str(ws)
            for frame_name, frame in frames.items():
                graph_frame(frame, frame_name, ax, x_col_name, y_col_name, len(frames), graph_std, k, max_frame)
                k += 1
            ax.set_xlabel('Episode Number [-]')
            ax.set_ylabel('Average Reward Over Training with Window Size = ' + str(ws) + ' [-]')
            ax.legend()
            plt.tight_layout()

            fig_num += 1
    '''


    plt.show()


if __name__ == '__main__':

    graph_std = False
    sets_to_compare = dict()
    include_training_data = False

    # ------------------------------------------------------------------------------------------------------------------
    # method comparison vvv
    # ------------------------------------------------------------------------------------------------------------------

    '''
    # DDPG b spline
    set_5 = {
        'base_folder': 'tune_boat_bspline_DDPG',
        'set_name': 'DDPGBSpline',
        'trial_num': 5,
        'eval_trial_num': 0}
    sets_to_compare['DDPG_BSpline'] = set_5

    # DDPG dubins selected solution
    set_2 = {
        'base_folder': 'tune_boat_dubins_DDPG',
        'set_name': 'DDPGDubins',
        'trial_num': 2,
        'eval_trial_num': 0}
    sets_to_compare['DDPG_Dubins'] = set_2

    # DDPG vector control
    set_1008 = {
        'base_folder': 'tune_boat_vector_control_DDPG',
        'set_name': 'DDPGVector',
        'trial_num': 1008,
        'eval_trial_num': 0}
    sets_to_compare['DDPG_Vanilla'] = set_1008

    # DQN bspline
    set_100 = {
        'base_folder': 'tune_boat_bspline_DQN',
        'set_name': 'DQNBSpline',
        'trial_num': 100,
        'eval_trial_num': 0}
    sets_to_compare['DQN_Bspline'] = set_100

    # DQN dubins
    set_100a = {
        'base_folder': 'tune_boat_dubins_DQN',
        'set_name': 'DQNDubins',
        'trial_num': 100,
        'eval_trial_num': 0}
    sets_to_compare['DQN_Dubins'] = set_100a

    # DQN vector control
    set_1002 = {
        'base_folder': 'tune_boat_vector_control_DQN',
        'set_name': 'DQNVector',
        'trial_num': 1002,
        'eval_trial_num': 0}
    sets_to_compare['DQN_Vanilla'] = set_1002

    # RRTStar

    set_0 = {
        'base_folder': 'tune_boat_RRTstar',
        'set_name': 'RRTstar',
        'trial_num': 34,
        'eval_trial_num': 0}
    sets_to_compare['RRT'] = set_0

    # PRM
    set_49 = {
        'base_folder': 'tune_boat_PRM',
        'set_name': 'TunePRM',
        'trial_num': 49,
        'eval_trial_num': 0}
    sets_to_compare['PRM'] = set_49
    '''


    # ------------------------------------------------------------------------------------------------------------------
    # method comparison ^^^
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # DRP compare spare reward vvv
    # ------------------------------------------------------------------------------------------------------------------


    set_100 = {
        'base_folder': 'tune_boat_bspline_DDPG_Sparse',
        'set_name': 'DDPGBSplineSparse',
        'trial_num': 100,
        'eval_trial_num': 0}
    sets_to_compare['0.6'] = set_100

    set_101 = {
        'base_folder': 'tune_boat_bspline_DDPG_Sparse',
        'set_name': 'DDPGBSplineSparse',
        'trial_num': 101,
        'eval_trial_num': 0}
    sets_to_compare['0.7'] = set_101

    set_102 = {
        'base_folder': 'tune_boat_bspline_DDPG_Sparse',
        'set_name': 'DDPGBSplineSparse',
        'trial_num': 102,
        'eval_trial_num': 0}
    sets_to_compare['0.8'] = set_102

    set_103 = {
        'base_folder': 'tune_boat_bspline_DDPG_Sparse',
        'set_name': 'DDPGBSplineSparse',
        'trial_num': 103,
        'eval_trial_num': 0}
    sets_to_compare['0.9'] = set_103

    set_104 = {
        'base_folder': 'tune_boat_bspline_DDPG_Sparse',
        'set_name': 'DDPGBSplineSparse',
        'trial_num': 104,
        'eval_trial_num': 0}
    sets_to_compare['0.95'] = set_104

    set_105 = {
        'base_folder': 'tune_boat_bspline_DDPG_Sparse',
        'set_name': 'DDPGBSplineSparse',
        'trial_num': 105,
        'eval_trial_num': 0}
    sets_to_compare['0.99'] = set_105


    # ------------------------------------------------------------------------------------------------------------------
    # DRP compare spare reward ^^^
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # Vanilla RL compare spare reward vvv
    # ------------------------------------------------------------------------------------------------------------------

    '''
    set_0 = {
        'base_folder': 'tune_boat_vector_control_DDPG_Sparse',
        'set_name': 'DDPGVectorSparse',
        'trial_num': 0,
        'eval_trial_num': 0}
    sets_to_compare['0.6'] = set_0

    set_1 = {
        'base_folder': 'tune_boat_vector_control_DDPG_Sparse',
        'set_name': 'DDPGVectorSparse',
        'trial_num': 1,
        'eval_trial_num': 0}
    sets_to_compare['0.7'] = set_1

    set_2 = {
        'base_folder': 'tune_boat_vector_control_DDPG_Sparse',
        'set_name': 'DDPGVectorSparse',
        'trial_num': 2,
        'eval_trial_num': 0}
    sets_to_compare['0.8'] = set_2

    set_3 = {
        'base_folder': 'tune_boat_vector_control_DDPG_Sparse',
        'set_name': 'DDPGVectorSparse',
        'trial_num': 3,
        'eval_trial_num': 0}
    sets_to_compare['0.9'] = set_3

    set_4 = {
        'base_folder': 'tune_boat_vector_control_DDPG_Sparse',
        'set_name': 'DDPGVectorSparse',
        'trial_num': 4,
        'eval_trial_num': 0}
    sets_to_compare['0.95'] = set_4

    set_5 = {
        'base_folder': 'tune_boat_vector_control_DDPG_Sparse',
        'set_name': 'DDPGVectorSparse',
        'trial_num': 5,
        'eval_trial_num': 0}
    sets_to_compare['0.999735'] = set_5
    '''

    # ------------------------------------------------------------------------------------------------------------------
    # Vanilla RL compare spare reward ^^^
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # Vanilla RL and DRP compare spare reward vvv
    # ------------------------------------------------------------------------------------------------------------------

    '''
    set_102 = {
        'base_folder': 'tune_boat_bspline_DDPG_Sparse',
        'set_name': 'DDPGBSplineSparse',
        'trial_num': 102,
        'eval_trial_num': 0}
    sets_to_compare['DRP Sparse'] = set_102

    set_5a = {
        'base_folder': 'tune_boat_bspline_DDPG',
        'set_name': 'DDPGBSpline',
        'trial_num': 5,
        'eval_trial_num': 0}
    sets_to_compare['DRP Dense'] = set_5a

    set_5 = {
        'base_folder': 'tune_boat_vector_control_DDPG_Sparse',
        'set_name': 'DDPGVectorSparse',
        'trial_num': 5,
        'eval_trial_num': 0}
    sets_to_compare['Vanilla Sparse'] = set_5

    set_1008 = {
        'base_folder': 'tune_boat_vector_control_DDPG',
        'set_name': 'DDPGVector',
        'trial_num': 1008,
        'eval_trial_num': 0}
    sets_to_compare['Vanilla Dense'] = set_1008
    '''

    """
    Edit above ^^
    """

    create_graphs(sets_to_compare, graph_std,include_training_data)
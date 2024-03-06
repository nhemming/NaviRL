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

    # get the maximum amount episodes
    max_frame = 0
    for frame_name, frame in frames.items():
        tmp_max = frame['ep_num'].max()
        if tmp_max > max_frame:
            max_frame = tmp_max

    # graph the success rate
    sns.set_theme()
    fig = plt.figure(0, figsize=(14, 8))
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
    fig = plt.figure(1, figsize=(14, 8))
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
    fig = plt.figure(2, figsize=(14, 8))
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
    fig = plt.figure(3, figsize=(14, 8))
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



    plt.show()


if __name__ == '__main__':

    graph_std = False
    sets_to_compare = dict()
    include_training_data = False

    set_0 = {
        'base_folder': 'tune_boat_bspline_DQN',
        'set_name': 'DQNBSpline',
        'trial_num': 0,
        'eval_trial_num': 0}
    sets_to_compare['DQN_Bspline0'] = set_0

    # DDPG vector control

    set_1008 = {
        'base_folder': 'tune_boat_vector_control_DDPG',
        'set_name': 'DDPGVector',
        'trial_num': 1008,
        'eval_trial_num': 0}
    sets_to_compare['DDPG_Vector1008'] = set_1008

    # dubins selected solution
    set_2 = {
        'base_folder': 'tune_boat_dubins_DDPG',
        'set_name': 'DDPGDubins',
        'trial_num': 2,
        'eval_trial_num': 0}
    sets_to_compare['DDPG_Dubins2'] = set_2

    # bspline
    set_1011 = {
        'base_folder': 'demo_to_test_boat_DDPG',
        'set_name': 'DebugDDPGBSpline',
        'trial_num': 1011,
        'eval_trial_num': 0}
    sets_to_compare['DDPG_BSpline1011'] = set_1011
    
    """
    Edit above ^^
    """

    create_graphs(sets_to_compare, graph_std,include_training_data)
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

def create_graphs(sets_to_compare, graph_std):

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
    ax1 = fig.add_subplot(111)
    k = 0
    for frame_name, frame in frames.items():
        if len(frames) == 1:
            label = 'mean'
        else:
            label = frame_name
        if len(frame) == 1:
            ax1.plot([0,max_frame], np.full((2,),frame['success']), label=label, color=get_color(k))
        else:
            ax1.plot(frame['ep_num'], frame['success'], label=label,color=get_color(k))

        if graph_std and len(frame) > 1:
            if len(frames) == 1:
                label = '1 $\sigma$'
            else:
                label = ''
            ax1.fill_between(frame['ep_num'], frame['success'] + frame['success_std'], frame['success'] - frame['success_std'],
                             facecolor=get_color(k), alpha=0.3, label=label)

        if len(frames) == 1 and len(frame) > 1:
            # graph max line
            ax1.plot(frame['ep_num'], frame['success_max'], '-', color='tab:purple', label='Max')
            # graph min line
            ax1.plot(frame['ep_num'], frame['success_min'], '-', color='tab:green', label='Min')

        k += 1
    ax1.set_xlabel('Episode Number [-]')
    ax1.set_ylabel('Average Success Rate [-]')
    ax1.legend()
    plt.tight_layout()

    # graph crash rate
    fig = plt.figure(1, figsize=(14, 8))
    ax1 = fig.add_subplot(111)
    k = 0
    for frame_name, frame in frames.items():
        if len(frames) == 1:
            label = 'mean'
        else:
            label = frame_name
        if len(frame) == 1:
            ax1.plot([0, max_frame], np.full((2,),frame['crashed']) , label=label, color=get_color(k))
        else:
            ax1.plot(frame['ep_num'], frame['crashed'], label=label, color=get_color(k))

        if graph_std:
            if len(frames) == 1:
                label = '1 $\sigma$'
            else:
                label = ''
            ax1.fill_between(frame['ep_num'], frame['crashed'] + frame['crashed_std'], frame['crashed'] - frame['crashed_std'],
                             facecolor=get_color(k), alpha=0.3, label=label)
        if len(frames) == 1:
            # graph max line
            ax1.plot(frame['ep_num'], frame['crashed_max'], '-', color='tab:purple', label='Max')
            # graph min line
            ax1.plot(frame['ep_num'], frame['crashed_min'], '-', color='tab:green', label='Min')

        k += 1
    ax1.set_xlabel('Episode Number [-]')
    ax1.set_ylabel('Average Crash Rate [-]')
    ax1.legend()
    plt.tight_layout()

    # plot distance to reach goal
    fig = plt.figure(2, figsize=(14, 8))
    ax1 = fig.add_subplot(111)
    k = 0
    for frame_name, frame in frames.items():
        if len(frames) == 1:
            label = 'mean'
        else:
            label = frame_name
        if len(frame) == 1:
            ax1.plot([0, max_frame], np.full((2,),frame['distance_traveled[m]']) , label=label,color=get_color(k))
        else:
            ax1.plot(frame['ep_num'], frame['distance_traveled[m]'], label=label,color=get_color(k))
        if graph_std:
            if len(frames) == 1:
                label = '1 $\sigma$'
            else:
                label = ''
            ax1.fill_between(frame['ep_num'], frame['distance_traveled[m]'] + frame['distance_traveled[m]_std'],
                             frame['distance_traveled[m]'] - frame['distance_traveled[m]_std'],
                             facecolor=get_color(k), alpha=0.3, label=label)
        if len(frames) == 1:
            # graph max line
            ax1.plot(frame['ep_num'], frame['distance_traveled[m]_max'], '-', color='tab:purple', label='Max')
            # graph min line
            ax1.plot(frame['ep_num'], frame['distance_traveled[m]_min'], '-', color='tab:green', label='Min')
        k += 1
    ax1.set_xlabel('Episode Number [-]')
    ax1.set_ylabel('Average Distance Traveled [m]')
    ax1.legend()
    plt.tight_layout()

    # plot normalized distance to reach goal
    fig = plt.figure(3, figsize=(14, 8))
    ax1 = fig.add_subplot(111)
    k = 0
    for frame_name, frame in frames.items():
        if len(frames) == 1:
            label = 'mean'
        else:
            label = frame_name
        if len(frame) == 1:
            ax1.plot([0, max_frame], np.full((2,),frame['normalized_distance_traveled']) , label=label,color=get_color(k))
        else:
            ax1.plot(frame['ep_num'], frame['normalized_distance_traveled'], label=label,color=get_color(k))
        if graph_std:
            if len(frames) == 1:
                label = '1 $\sigma$'
            else:
                label = ''
            ax1.fill_between(frame['ep_num'], frame['normalized_distance_traveled'] + frame['normalized_distance_traveled_std'],
                             frame['normalized_distance_traveled'] - frame['normalized_distance_traveled_std'],
                             facecolor=get_color(k), alpha=0.3, label=label)

        if len(frames) == 1:
            # graph max line
            ax1.plot(frame['ep_num'], frame['normalized_distance_traveled_max'], '-', color='tab:purple', label='Max')
            # graph min line
            ax1.plot(frame['ep_num'], frame['normalized_distance_traveled_min'], '-', color='tab:green', label='Min')



        k += 1

    # TODO add timeout exclusion graph also.
    ax1.plot([0, max_frame], [1.0, 1.0], '--', color='tab:gray', label='Crow Flies Distance')
    ax1.set_xlabel('Episode Number [-]')
    ax1.set_ylabel('Normalized Average Distance Traveled [-]')
    ax1.legend()
    plt.tight_layout()

    plt.show()


if __name__ == '__main__':

    graph_std = False
    sets_to_compare = dict()

    set_0 = {
        'base_folder': 'demo_to_test_DDPG',
        'set_name' : 'DebugDDPGRLPRM',
        'trial_num': 0,
        'eval_trial_num' : 0}
    sets_to_compare['DDPG_RLPRM'] = set_0

    set_1 = {
        'base_folder': 'demo_to_test_DDPG',
        'set_name': 'DebugDDPGBSpline',
        'trial_num': 6,
        'eval_trial_num': 0}
    sets_to_compare['DDPG_BSpline'] = set_1

    set_2 = {
        'base_folder': 'demo_to_test_DDPG',
        'set_name': 'DebugDDPGDubins',
        'trial_num': 2,
        'eval_trial_num': 0}
    sets_to_compare['DDPG_Dubins'] = set_2


    set_3 = {
        'base_folder': 'demo_to_test_non_learning',
        'set_name': 'DebugPRM',
        'trial_num': 0,
        'eval_trial_num': 0}
    sets_to_compare['PRM'] = set_3

    """
    Edit above ^^
    """

    create_graphs(sets_to_compare, graph_std)
"""
Extract and graph the trends in navigation success and reward over time
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
    trial_num = 1

    abs_path = os.getcwd().replace('\\analysis', '\\experiments')
    base_dir = os.path.join(abs_path, base_folder)
    base_dir = os.path.join(base_dir, 'output')
    base_dir = os.path.join(base_dir, set_name)
    base_dir = os.path.join(base_dir, str(trial_num))

    env = load_environment(base_folder,set_name,trial_num)

    # get the names of the agents
    file_dir = os.path.join(base_dir, 'training', 'learning_algorithm')
    dir_list = os.listdir(file_dir)
    for i, file in enumerate(dir_list):
        dir_list[i] = file.split('_epnum')[0]

    agent_names = Counter(dir_list).keys()
    df = pd.DataFrame(data=np.zeros((len(dir_list),len(agent_names))),columns=agent_names)
    df['ep_num'] = 0.0
    for i, an in enumerate(agent_names):
        for j,_ in enumerate(dir_list):
            tmp_file = os.path.join(file_dir,an+'_epnum-'+str(j)+'.csv')
            tmp_df = pd.read_csv(tmp_file)
            cum_reward = tmp_df['reward'].sum()
            df[an].iloc[j] = cum_reward
            df['ep_num'].iloc[j] = j

    # get the success rate of the agent

    # get goal distance
    for name, reward_comp in env.reward_function.reward_components.items():
        if isinstance(reward_comp,ReachDestinationReward):
            goal_dst = reward_comp.goal_dst

    file_dir = os.path.join(base_dir, 'training', 'sensors')
    dir_list = os.listdir(file_dir)
    dir_list = [i for i in dir_list if 'destination_sensor_0' in i]
    df['success'] = 0.0
    for i, file in enumerate(dir_list):
        tmp_file = os.path.join(file_dir,'destination_sensor_0_epnum-' + str(i) + '.csv')
        tmp_df = pd.read_csv(tmp_file)
        min_dst = tmp_df['distance'].min()
        success = 0.0
        if min_dst <= goal_dst:
            success = 1.0
        if i < len(df):
            # allows for running while training
            df['success'].iloc[i] = success

    sns.set_theme()
    fig = plt.figure(0,figsize=(14,8))
    ax = fig.add_subplot(111)
    ax.set_xlabel('Episode Number [-]')
    ax.set_ylabel('Reward [-]')
    for an in agent_names:
        ax.plot(df['ep_num'],df[an],label=an,alpha=0.3)
        an_smooth = uniform_filter1d(df[an],100)
        ax.plot(df['ep_num'],an_smooth, label=an)
    plt.tight_layout()
    file_dir = os.path.join(base_dir, 'progress')
    plt.savefig(os.path.join(file_dir,'Reward.png'))

    fig = plt.figure(1, figsize=(14, 8))
    ax = fig.add_subplot(111)
    ax.set_xlabel('Episode Number [-]')
    ax.set_ylabel('Success Rate [-]')
    ax.plot(df['ep_num'], df['success'], label='success', alpha=0.3)
    success_smooth = uniform_filter1d(df['success'], 100)
    ax.plot(df['ep_num'], success_smooth, label='success mean')
    plt.tight_layout()
    plt.savefig(os.path.join(file_dir, 'SuccessRate.png'))

    plt.close()


if __name__ == '__main__':

    main()
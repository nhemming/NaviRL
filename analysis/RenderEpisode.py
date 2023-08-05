"""
renders an training episode
"""

# native packages
import os

# 3rd party packages
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# own modules
from environment.NavigationEnvironment import NavigationEnvironment
from environment.Sensor import DestinationSensor

def graph_episode(env, ep_num, ep_num_key, frames, save_path):

    sns.set_theme()
    fig = plt.figure(figsize=(14,8))
    gs = GridSpec(4, 4)
    ax1 = fig.add_subplot(gs[0:2,0:2])
    ax1.set_xlabel('X Position [m]')
    ax1.set_ylabel('Y Position [m]')
    #ax1.set_xlim([env.domain['min_x']-env.domain['buffer'],env.domain['max_x']+env.domain['buffer']])
    #ax1.set_ylim([env.domain['min_y'] - env.domain['buffer'], env.domain['max_y'] + env.domain['buffer']])
    ax2 = fig.add_subplot(gs[2,0])
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Position [m]')

    ax3 = fig.add_subplot(gs[2, 1])
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Heading [rad]')

    ax4 = fig.add_subplot(gs[3, 0])
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Velocity [m/s]')

    ax5 = fig.add_subplot(gs[3, 2])
    ax5.set_xlabel('Time [s]')
    ax5.set_ylabel('Destination Heading [rad]')

    ax6 = fig.add_subplot(gs[3, 3])
    ax6.set_xlabel('Time [s]')
    ax6.set_ylabel('Destination Distance [m]')

    ax7 = fig.add_subplot(gs[0,2])
    ax7.set_xlabel('Time [s]')
    ax7.set_ylabel('Reward [-]')

    ax8 = fig.add_subplot(gs[0, 3])
    ax8.set_xlabel('Time [s]')
    ax8.set_ylabel('Applied Action [-]')

    ax9 = fig.add_subplot(gs[1, 2])
    ax9.set_xlabel('Time [s]')
    ax9.set_ylabel('Q vals [-]')


    sim_time = np.infty

    # graph entities
    for name, entity in env.entities.items():
        if np.isinf(sim_time):
            sim_time = frames[name]['sim_time'].max()

        # slice the frame to relevant time
        frames[name]= frames[name].loc[frames[name]['sim_time'] <= sim_time]

        # draw the trajectory
        entity.draw_trajectory(ax1, frames[name], sim_time)

        # draw trajectory telemetry
        entity.draw_telemetry_trajectory(ax2, frames[name], sim_time)

        # draw heading telemetry
        entity.draw_telemetry_heading(ax3, frames[name], sim_time)

        # draw velocity telemetry
        entity.draw_telemetry_velocity(ax4, frames[name], sim_time)

    # graph sensors
    for name, sensor in env.sensors.items():
        # slice the frame to relevant time
        frames[name] = frames[name].loc[frames[name]['sim_time'] <= sim_time]

        if isinstance(sensor,DestinationSensor):
            sensor.draw_angle(ax5, frames[name], sim_time)
            sensor.draw_distance(ax6, frames[name], sim_time)

        sensor.draw_at_time( ax1, frames[name], sim_time)

    # graph learning entities
    for name, agent in env.agents.items():
        for tmp_name, lrn_alg in agent.learning_algorithms.items():
            # slice the frame to relevant time
            frames[name+'_'+tmp_name] = frames[name+'_'+tmp_name].loc[frames[name+'_'+tmp_name]['sim_time'] <= sim_time]
            ax7.plot(frames[name+'_'+tmp_name]['sim_time'],frames[name+'_'+tmp_name]['reward'])

            ax8.plot(frames[name + '_' + tmp_name]['sim_time'], frames[name + '_' + tmp_name]['applied_action'])

            # need a better solution for q values
            cols = frames[name+'_'+tmp_name].columns
            q_val_header = [i for i in cols if 'q_values' in i]
            for i, qv in enumerate(q_val_header):
                ax9.plot(frames[name + '_' + tmp_name]['sim_time'], frames[name + '_' + tmp_name][qv], label=str(i))
            #ax9.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_path,str(ep_num)+'.png'))
    plt.close()

def render_episode(env, ep_num, ep_num_key, frames):
    pass

def main():

    # set experiments to evaluate
    base_folder = 'demo_to_test_DQN'
    set_name = 'DebugDQN'
    trial_num = 3
    ep_num_vec = range(950,1150)
    create_video = False

    abs_path = os.getcwd().replace('\\analysis','\\experiments')
    base_dir = os.path.join(abs_path,base_folder)
    base_dir = os.path.join(base_dir,'output')
    base_dir = os.path.join(base_dir,set_name)
    base_dir = os.path.join(base_dir, str(trial_num))
    file_name = os.path.join(base_dir,'hyper_parameters.yaml')

    env = NavigationEnvironment()
    env.build_env_from_yaml(file_name, '', False)

    for episode_num in ep_num_vec:

        ep_num_key = 'epnum-'+str(episode_num)+'.csv'

        frames = dict()
        for name, entity in env.entities.items():
            file_dir = os.path.join(base_dir,'training','entities')
            dir_list = os.listdir(file_dir)
            file_of_item = [i for i in dir_list if name+'_'+ep_num_key in i][0]
            file_of_item = os.path.join(file_dir,file_of_item)
            tmp_frame = pd.read_csv(file_of_item,index_col=False)
            frames[name] = tmp_frame
        for name, sensor in env.sensors.items():
            file_dir = os.path.join(base_dir,'training','sensors')
            dir_list = os.listdir(file_dir)
            file_of_item = [i for i in dir_list if name+'_'+ep_num_key in i][0]
            file_of_item = os.path.join(file_dir,file_of_item)
            tmp_frame = pd.read_csv(file_of_item,index_col=False)
            frames[name] = tmp_frame
        for name, agent in env.agents.items():
            file_dir = os.path.join(base_dir,'training','learning_algorithm')
            dir_list = os.listdir(file_dir)

            for tmp_name, lrn_alg in agent.learning_algorithms.items():

                file_of_item = [i for i in dir_list if name+'_'+tmp_name+'_'+ep_num_key in i][0]
                file_of_item = os.path.join(file_dir,file_of_item)
                tmp_frame = pd.read_csv(file_of_item,index_col=False)
                frames[name+'_'+tmp_name] = tmp_frame

        if create_video:
            render_episode(env,episode_num, ep_num_key, frames)
        else:
            graph_episode(env,episode_num, ep_num_key, frames, os.path.join(base_dir,'training','graphs'))


if __name__ == '__main__':

    main()
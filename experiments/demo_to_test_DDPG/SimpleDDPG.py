"""
used for testing and debugging DDPG using a simple massless agent
"""

# native modules
import os

# 3rd party modules0

# own modules
from environment.NavigationEnvironment import NavigationEnvironment


def main():

    env = NavigationEnvironment()
    test_scenario_number = 1
    input_file_name = ''
    if test_scenario_number == 0:
        input_file_name = 'experiment_setup_DDPG_vector_control.yaml'
    elif test_scenario_number == 1:
        input_file_name = 'experiment_setup_DDPG_bspline_control.yaml'
    elif test_scenario_number == 2:
        input_file_name = 'experiment_setup_DDPG_dubins_control.yaml'

    # get directory of this script
    cur_dir = os.getcwd()

    # create the environment that constructs all the objects.
    env.build_env_from_yaml(input_file_name, cur_dir)

    # run the training
    env.train_agent()


if __name__ == '__main__':

    main()
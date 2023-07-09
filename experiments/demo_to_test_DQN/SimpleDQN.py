"""
Runs a DQN agent on the basic agent for simple navigation
"""

# native modules

# 3rd party modules

# own modules
from environment.NavigationEnvironment import NavigationEnvironment


def main():

    env = NavigationEnvironment()
    test_scenario_number = 1
    input_file_name = ''
    if test_scenario_number == 0:
        input_file_name = 'experiment_setup_DQN_vector_control.yaml'
    elif test_scenario_number == 1:
        input_file_name = 'experiment_setup_DQN_bspline_control.yaml'
    elif test_scenario_number == 2:
        input_file_name = 'experiment_setup_DQN_dubins_control.yaml'
    elif test_scenario_number == 3:
        input_file_name = 'experiment_setup_DQN_point_control.yaml'

    env.build_env_from_yaml(input_file_name)

    # run the training
    env.train_agent()


if __name__ == '__main__':

    main()
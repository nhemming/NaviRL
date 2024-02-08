"""
This scripts launches an evaluation set for trained robots with no obstacles
"""

# native modules
import os

# 3rd party modules0

# own modules
from environment.NavigationEnvironment import NavigationEnvironment

def main():

    input_file = 'no_obstacle_mass_free_evaluation_set.yaml'

    # create environment
    env = NavigationEnvironment()

    # get directory of this script
    cur_dir = os.getcwd()

    env.build_eval_env_from_yaml(input_file, cur_dir)

    # create the environment that constructs all the objects.
    env.run_evaluation_set()

if __name__ == '__main__':

    main()
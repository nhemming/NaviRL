"""
Environment for navigation tasks. Controls the building, running, and saving the learning agents
"""

# native modules
import os

# 3rd party modules
import numpy as np
import pandas as pd
import yaml

# own modules
from environment.ActionOperation import BSplineControl, DirectVectorControl
from agents.DQN_agents.DQN import DQN
from environment.MassFreeVectorEntity import MassFreeVectorEntity
from environment.Entity import CollisionCircle, CollisionRectangle, get_collision_status
from environment.Reward import AlignedHeadingReward, CloseDistanceReward, ImproveHeadingReward, RewardDefinition, ReachDestinationReward
from environment.Sensor import DestinationSensor
from environment.StaticEntity import StaticEntity
from exploration_strategies.EpsilonGreedy import EpsilonGreedy
from environment.Termination import AnyCollisionsTermination, ReachDestinationTermination, TerminationDefinition


class NavigationEnvironment:

    def __init__(self):
        self.domain = dict()
        self.h_params = dict()
        self.delta_t = None # time step of the simulation
        self.max_training_time = 0.0
        self.output_dir = None

        # a dictionary of entities in the simulation. These are the objects in the simulation that act. They are typically
        # physical entities, but do not have to be.
        self.entities = dict()

        # a dictionary of sensors for the system. The sensors are used to generate data from interactions of entities.
        self.sensors = dict()

        # a dictionary of agents that act in the environment. Not all agents are learning agents.
        self.agents = dict()

        # a dictionary of exploration strategies used by the agents to perturb actions during learning.
        self.exploration_strategies = dict()

        # object that defines how reward is recieved
        self.reward_function = None

        # object that defines how the simulation is terminated
        self.termination_function = None

        # minimum distance all entities in the simulation must be apart in the reset function.
        # TODO should be loaded from file not hard coded
        self.reset_seperation_dst = 2

    def build_env_from_yaml(self, file_name, base_dir):
        """
        Given a yaml file that has configuration details for the environment and hyperparameters for the agents,
        a dictionary is built and returned to the user

        :param file_name: a file name that has all the hyperparameters. This should be a yaml file
        """
        with open(file_name, "r") as stream:
            try:
                hp_data = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        self.h_params = hp_data

        # load domain information
        domain_data = self.h_params['Environment']['domain']
        self.domain = domain_data
        self.delta_t = self.h_params['Environment']['time_step']
        self.max_training_time = self.h_params['MetaData']['training_sim_time']

        # create the entities
        self.load_entities()

        # create sensors
        self.load_sensors()

        # create exploration strategies
        self.load_exploration_strategies()

        # create learning agent
        self.load_learning_agent()

        # create reward function
        self.load_reward_function()

        # create folders to save information
        output_dir = os.path.join(base_dir,'output')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        output_dir = os.path.join(output_dir, self.h_params['MetaData']['set'])
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        output_dir = os.path.join(output_dir, str(self.h_params['MetaData']['trial_num']))
        self.output_dir = output_dir
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        model_dir = os.path.join(output_dir,'models')
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        eval_dir = os.path.join(output_dir, 'evaluation')
        if not os.path.isdir(eval_dir):
            os.mkdir(eval_dir)
        training_dir = os.path.join(output_dir, 'training')
        if not os.path.isdir(training_dir):
            os.mkdir(training_dir)
        progress_dir = os.path.join(output_dir, 'progress')
        if not os.path.isdir(progress_dir):
            os.mkdir(progress_dir)

        # save the hyper parameter set
        with open(os.path.join(output_dir,'hyper_parameters.yaml'), 'w') as file:
            yaml.safe_dump(self.h_params, file)

    def load_entities(self):
        """
        From the data in the input file, entities are created and saved.
        :return:
        """
        for value, item in self.h_params['Entities'].items():

            # create the collision object
            collision_shape = None
            collision_dict = item['collision_object']
            if collision_dict['type'] == 'circle':
                # create a circle collision object
                collision_shape = CollisionCircle(0.0,'circle',collision_dict['radius'])
            elif collision_dict['type'] == 'rectangle':
                # create a rectangular collision object
                collision_shape = CollisionRectangle(0.0, collision_dict['height'], 'rectangle',collision_dict['width'])
            else:
                raise ValueError('Invalid collision object type.')

            # create the entities
            if value == 'MassFreeVectorEntity':
                # create a basic agent and add it to the entity list
                ba = MassFreeVectorEntity(collision_shape, item['id'], item['name'])
                self.entities[ba.name] = ba
            elif value == 'StaticEntity':
                # create a basic entity that does not move. Typically, used for static obstacles and destination areas.
                se = StaticEntity(collision_shape,item['id'],item['name'] )
                self.entities[se.name] = se
            else:
                raise ValueError('Invalid entity type.')

    def load_sensors(self):

        for value, item in self.h_params['Sensors'].items():
            if item['type'] == 'destination_sensor':
                ds = DestinationSensor(item['id'], item['name'], item['owner'], item['target'])
                self.sensors[ds.name] = ds
            else:
                raise ValueError('Invalid Sensor Type')

    def load_exploration_strategies(self):

        es = self.h_params['ExplorationStrategies']
        for value, item in es.items():
            if item['type'] == 'EpsilonGreedy':

                # parse the schedule for the threshold for choosing a random action
                ts = item['threshold_schedule']
                threshold_schedule = np.zeros((len(ts),2))
                k = 0
                for tmp_value, tmp_item in ts.items():
                    point = [float(i) for i in tmp_item.split(',')]
                    threshold_schedule[k,:] = point
                    k += 1

                # parse the perterbation distribution definitions
                if self.h_params['LearningAgent']['ActionOperation']['is_continuous']:
                    # TODO
                    ds = item['distribution']
                    perturb = None
                else:
                    perturb = self.h_params['LearningAgent']['ActionOperation']['action_options']

                eg = EpsilonGreedy(self.h_params['LearningAgent']['Network']['device'],self.h_params['LearningAgent']['ActionOperation']['is_continuous'], item['name'], threshold_schedule, perturb)
                self.exploration_strategies[eg.name] = eg

    def load_learning_agent(self):

        la = self.h_params['LearningAgent']

        # parse action operation
        ao = self.load_action_operation()

        alg = la['Algorithm']
        la_input = la['Input']
        if alg['type'] == 'DQN':
            # create a DQN agent
            row_list = []
            for value, item in la_input.items():
                row_list.append(item)

            obs_df = pd.DataFrame(columns=['name', 'data', 'min', 'max', 'norm_min', 'norm_max'], data=row_list)
            agent = DQN(ao,
                        la['ControlledEntity'],
                        la['Network']['device'],
                        self.exploration_strategies[alg['exploration_strategy']],
                        la['name'],
                        la['Network'],
                        obs_df)
            self.agents[agent.name] = agent

        else:
            raise ValueError('Learning Algorithm not supported')

    def load_action_operation(self):
        op = None
        ao = self.h_params['LearningAgent']['ActionOperation']
        if ao['name'] == 'direct_vector_control':
            op = DirectVectorControl(ao['action_options'], ao['frequency'], ao['is_continuous'], ao['name'],
                                     ao['number_controls'],
                                     self.h_params['LearningAgent']['Network']['output_range'])
        elif ao['name'] == 'bspline_control':

            op = BSplineControl(ao['action_options'], ao['frequency'], ao['is_continuous'], ao['name'],
                                     ao['number_controls'],
                                     self.h_params['LearningAgent']['Network']['output_range'])

        else:
            raise ValueError('Invalid action operation designation')

        return op

    def load_reward_function(self):

        self.reward_function = RewardDefinition()

        rd = self.h_params['RewardDefinition']
        self.reward_function.overall_adj_factor = rd['overall_adj_factor']
        for name, value in rd.items():
            if 'component' in name:
                if value['type'] == 'aligned_heading':
                    ahr = AlignedHeadingReward(value['adj_factor'], value['aligned_angle'], value['aligned_reward'],  value['destination_sensor'])
                    self.reward_function.reward_components[ahr.name] = ahr
                elif value['type'] == 'close_distance':
                    cdr = CloseDistanceReward(value['adj_factor'], value['destination_sensor'])
                    self.reward_function.reward_components[cdr.name] = cdr
                elif value['type'] == 'heading_improvement':
                    ihr = ImproveHeadingReward(value['adj_factor'], value['destination_sensor'])
                    self.reward_function.reward_components[ihr.name] = ihr
                elif value['type'] == 'reach_destination':
                    rdr = ReachDestinationReward(value['adj_factor'], value['destination_sensor'], value['goal_dst'], value['reward'])
                    self.reward_function.reward_components[rdr.name] = rdr
                else:
                    raise ValueError('Unsupported reward component')

    def load_termination_function(self):

        term = self.h_params['Termination']
        self.termination_function = TerminationDefinition()
        for name, value in term.items():
            if value['type'] == 'any_collisions':
                self.termination_function.components[value['name']] = AnyCollisionsTermination(value['name'])
            elif value['type'] == 'reach_destination':
                self.termination_function.components[value['name']] = \
                    ReachDestinationTermination(value['destination_sensor'],value['goal_dst'],value['name'])
            else:
                raise ValueError('Invalid termination function')

    def train_agent(self):

        max_num_episodes = self.h_params['MetaData']['num_episodes']
        for episode_num in range(max_num_episodes):

            print("Episode Number={}".format(episode_num))

            # run training simulation
            history_path = os.path.join(self.output_dir,'training')
            self.run_simulation(episode_num, history_path, self.max_training_time, True)

    def run_simulation(self, episode_num, history_path, max_time, use_exploration=True):

        sim_time = 0.0
        done = False # false if the simulation is still active

        # call the reset methods for everything
        # reset the simulation
        self.reset()

        # reset the agents
        for name, tmp_agent in self.agents.items():
            tmp_agent.reset()

        # entity reset function
        for name, tmp_entity in self.entities.items():
            tmp_entity.reset()

        # reward function reset
        self.reward_function.reset()

        # add the history of the initial entity states after reseting
        for name, tmp_entity in self.entities.items():
            tmp_entity.add_step_history(sim_time)

        while not done: # TODO add hard time limit but in termination function

            # update sensors
            for name, tmp_sensor in self.sensors.items():
                tmp_sensor.update(sim_time, self.entities, self.sensors)

            # loop over agents both learning and non-learning
            # select action -> state_dict, action_dict
            for name, tmp_agent in self.agents.items():
                tmp_agent.create_state_action(self.entities, episode_num, self.sensors, sim_time, use_exploration)

            # convert action and apply the change
            for name, tmp_agent in self.agents.items():
                tmp_agent.execute_action_operation(self.entities)

            # step simulation for each entity -> new_state_dict
            for name_entity, tmp_entity in self.entities.items():
                tmp_entity.step(self.delta_t)

            # get reward of the step
            reward = self.reward_function.calculate_reward(self.entities, self.sensors)

            # look for collisions and set agent to inactive if collided
            self.set_collision_status()

            # update time of the simulation
            sim_time += self.delta_t

            # update history
            for name, tmp_entity in self.entities.items():
                tmp_entity.add_step_history(sim_time)

            # check for termination
            if sim_time >= max_time:
                done = True
            else:
                self.termination_function.calculate_termination(self.entities, self.sensors)

            # give MDP tuple back to agent. The agent chooses if it is going to save the data. It already has the state, action, and next state


        # write entity history
        for name, tmp_entity in self.entities.items():
            tmp_entity.write_history(episode_num,history_path)

    def reset(self):
        # reset the environment. Sensors do not have positions

        new_loc_lst = []
        for name, value in self.entities.items():

            min_dst = 0
            new_x = 0.0
            new_y = 0.0
            while min_dst <= self.reset_seperation_dst:

                new_x = np.random.uniform(low=self.domain['min_x'], high=self.domain['max_x'])
                new_y = np.random.uniform(low=self.domain['min_y'], high=self.domain['max_y'])
                min_dst = np.inf
                for loc in new_loc_lst:
                    tmp_dst = np.sqrt( (new_x-loc[0])**2 + (new_y-loc[1])**2 )
                    if tmp_dst < min_dst:
                        min_dst = tmp_dst

            value.state_dict['x_pos'] = new_x
            value.state_dict['y_pos'] = new_y

            # save the new location of the entity
            new_loc_lst.append([new_x, new_y])

            # each entity should have its own reset function
            value.reset()

    def set_collision_status(self):
        """
        Works through the entities in the simulation and checks for collisions between entities.
        :return:
        """

        entity_names = list(self.entities.keys())
        for i, entity_1_name in enumerate(entity_names):
            for j in range(i+1,len(entity_names)):
                is_collided = get_collision_status(self.entities[entity_1_name],self.entities[entity_names[j]])
                self.entities[entity_1_name].state_dict['is_collided'] = is_collided or self.entities[entity_1_name].state_dict['is_collided']
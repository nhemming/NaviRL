"""
Environment for navigation tasks. Controls the building, running, and saving the learning agents
"""

# native modules
from collections import OrderedDict
import os

# 3rd party modules
import numpy as np
import pandas as pd
import yaml

# own modules
from environment.ActionOperation import BSplineControl, DirectVectorControl
from environment.BaseAgent import SingleLearningAlgorithmAgent
from agents.DQN_agents.DQN import DQN
from environment.MassFreeVectorEntity import MassFreeVectorEntity
from environment.Entity import CollisionCircle, CollisionRectangle, get_collision_status
from environment.Reward import AlignedHeadingReward, CloseDistanceReward, ImproveHeadingReward, RewardDefinition, ReachDestinationReward
from environment.Sensor import DestinationSensor
from environment.StaticEntity import StaticEntity, StaticEntityCollide
from exploration_strategies.EpsilonGreedy import EpsilonGreedy
from environment.Termination import AnyCollisionsTermination, ReachDestinationTermination, TerminationDefinition
from replay_buffer.ReplayBuffer import ReplayBuffer


class NavigationEnvironment:

    def __init__(self):
        self.domain = OrderedDict()
        self.h_params = OrderedDict()
        self.delta_t = None # time step of the simulation
        self.max_training_time = 0.0
        self.output_dir = None

        # a dictionary of entities in the simulation. These are the objects in the simulation that act. They are typically
        # physical entities, but do not have to be.
        self.entities = OrderedDict()

        # a dictionary of sensors for the system. The sensors are used to generate data from interactions of entities.
        self.sensors = OrderedDict()

        # a dictionary of agents that act in the environment. Not all agents are learning agents.
        self.agents = OrderedDict()

        # a dictionary of exploration strategies used by the agents to perturb actions during learning.
        self.exploration_strategies = OrderedDict()

        # object that defines how reward is recieved
        self.reward_function = None

        # object that defines how the simulation is terminated
        self.termination_function = None

        # minimum distance all entities in the simulation must be apart in the reset function.
        # TODO should be loaded from file not hard coded
        self.reset_seperation_dst = 2

    def build_env_from_yaml(self, file_name, base_dir, create=True):
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

        # create the termination function
        self.load_termination_function()

        if create:
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
            train_sub_dir = ['sensors','entities','learning_algorithm','graphs']
            for tsd in train_sub_dir:
                tmp_dir = os.path.join(training_dir, tsd)
                if not os.path.isdir(tmp_dir):
                    os.mkdir(tmp_dir)
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
            collision_dict = item.get('collision_object',None)
            if collision_dict is None:
                # do nothing
                pass
            elif collision_dict['type'] == 'circle':
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
                se = StaticEntity(item['id'],item['name'])
                self.entities[se.name] = se
            elif value == 'StaticEntityCollide':
                # create a basic entity that does not move. Typically, used for static obstacles and destination areas.
                sec = StaticEntityCollide(collision_shape,item['id'],item['name'] )
                self.entities[sec.name] = sec
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

                eg = EpsilonGreedy(self.h_params['LearningAgent']['device'],self.h_params['LearningAgent']['ActionOperation']['is_continuous'], item['name'], threshold_schedule, perturb)
                self.exploration_strategies[eg.name] = eg

    def load_learning_agent(self):

        # create agents
        la = self.h_params['LearningAgent']

        # parse action operation
        ao = self.load_action_operation()
        if la['type'] == 'SingleLearningAlgorithmAgent':
            agent = SingleLearningAlgorithmAgent(ao,la['ControlledEntity'],la['name'],la['save_rate'])
        else:
            raise ValueError('Agent type not supported')

        # get learning algorithms
        alg = la['LearningAlgorithm']
        for lrn_alg_name, lrn_alg_data in alg.items():
            # optimizer dict

            optimizer_dict = lrn_alg_data['Optimizer']

            la_input = lrn_alg_data['Input']
            if lrn_alg_data['type'] == 'DQN':
                # create a DQN agent

                # parse replay buffer
                replay_buffer = self.load_replay_buffer(lrn_alg_data, la['device'])

                head_dict = OrderedDict()
                for head_name, head_data in la_input.items():
                    row_list = []
                    for value, item in head_data.items():
                        row_list.append(item)
                    obs_df = pd.DataFrame(columns=['name', 'data', 'min', 'max', 'norm_min', 'norm_max'], data=row_list)
                    head_dict[head_name] = obs_df
                dqn = DQN(la['device'],
                            self.exploration_strategies[lrn_alg_data['exploration_strategy']],
                            lrn_alg_data,
                            lrn_alg_data['name'],
                            lrn_alg_data['Network'],
                            head_dict,
                            optimizer_dict,
                            len(ao.action_options),
                            replay_buffer,
                            self.h_params['MetaData']['seed'])
                agent.learning_algorithms[dqn.name] = dqn
                #@self.agents[agent.name] = agent

            else:
                raise ValueError('Learning Algorithm not supported')
        self.agents[agent.name] = agent

    def load_action_operation(self):
        op = None
        ao = self.h_params['LearningAgent']['ActionOperation']

        # get algorithm that is linked
        alg_name = ao['alg_name']
        la = self.h_params['LearningAgent']['LearningAlgorithm']
        for name, value in la.items():
            if value['name'] == alg_name:
                alg_num = name

        if ao['name'] == 'direct_vector_control':
            op = DirectVectorControl(ao['action_options'], ao['frequency'], ao['is_continuous'], ao['name'],
                                     ao['number_controls'])
                                     #la[alg_num]['Network']['output_range'])
        elif ao['name'] == 'bspline_control':

            op = BSplineControl(ao['action_options'], ao['frequency'], ao['is_continuous'], ao['name'],
                                     ao['number_controls'],
                                     self.h_params['LearningAgent']['Network']['output_range'])

        else:
            raise ValueError('Invalid action operation designation')

        return op

    def load_replay_buffer(self, la_data, device):


        rb_data = la_data['ReplayBuffer']
        if rb_data['type'] == 'vanilla':
            seed = self.h_params['MetaData']['seed']
            return ReplayBuffer(rb_data['buffer_size'],la_data['batch_size'],seed, device)
        else:
            raise ValueError('Invalid replay buffer type')

    def load_reward_function(self):

        agent_names = []
        for name, value in self.agents.items():
            agent_names.append(value.name)
        self.reward_function = RewardDefinition(agent_names)

        rd = self.h_params['RewardDefinition']
        self.reward_function.overall_adj_factor = rd['overall_adj_factor']
        for name, value in rd.items():
            if 'component' in name:
                if value['type'] == 'aligned_heading':
                    ahr = AlignedHeadingReward(value['adj_factor'], value['aligned_angle'], value['aligned_reward'],  value['destination_sensor'], value['target_agent'], value['target_lrn_alg'])
                    self.reward_function.reward_components[ahr.name] = ahr
                elif value['type'] == 'close_distance':
                    cdr = CloseDistanceReward(value['adj_factor'], value['destination_sensor'], value['target_agent'], value['target_lrn_alg'])
                    self.reward_function.reward_components[cdr.name] = cdr
                elif value['type'] == 'heading_improvement':
                    ihr = ImproveHeadingReward(value['adj_factor'], value['destination_sensor'], value['target_agent'], value['target_lrn_alg'])
                    self.reward_function.reward_components[ihr.name] = ihr
                elif value['type'] == 'reach_destination':
                    rdr = ReachDestinationReward(value['adj_factor'], value['destination_sensor'], value['goal_dst'], value['reward'], value['target_agent'], value['target_lrn_alg'])
                    self.reward_function.reward_components[rdr.name] = rdr
                else:
                    raise ValueError('Unsupported reward component')

    def load_termination_function(self):

        term = self.h_params['TerminationDefinition']
        agent_names = []
        for name, value in self.agents.items():
            agent_names.append(value.name)
        self.termination_function = TerminationDefinition(agent_names)
        for name, value in term.items():
            if value['type'] == 'any_collisions':
                self.termination_function.components[value['name']] = AnyCollisionsTermination(value['name'],value['target_agent'])
            elif value['type'] == 'reach_destination':
                self.termination_function.components[value['name']] = \
                    ReachDestinationTermination(value['destination_sensor'],value['goal_dst'],value['name'],value['target_agent'])
            else:
                raise ValueError('Invalid termination function')

    def train_agent(self):

        # save initial agents
        for name, value in self.agents.items():
            # train the agent
            value.save_model(0, self.output_dir)

        max_num_episodes = self.h_params['MetaData']['num_episodes']
        for episode_num in range(max_num_episodes):

            print("Episode Number={}".format(episode_num))

            # run evaluation set

            # run training simulation
            history_path = os.path.join(self.output_dir,'training')
            self.run_simulation(episode_num, history_path, self.max_training_time, True)

            # train agent
            for name, value in self.agents.items():
                # train the agent
                value.train(episode_num,self.output_dir)

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
            tmp_entity.reset_base()

        # entity reset sensors
        for name, tmp_sensor in self.sensors.items():
            tmp_sensor.reset()

        # reset termination
        self.termination_function.reset()

        # add the history of the initial entity states after reseting
        for name, tmp_entity in self.entities.items():
            tmp_entity.add_step_history(sim_time)

        # update sensors
        for name, tmp_sensor in self.sensors.items():
            tmp_sensor.update(sim_time, self.entities, self.sensors)
        for name, tmp_sensor in self.sensors.items():
            tmp_sensor.add_step_history(sim_time)

        # reward function reset
        for name, reward_comp in self.reward_function.reward_components.items():
            reward_comp.reset(self.entities, self.sensors, self.reward_function.reward_agents)

        while not done and sim_time < max_time:

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

            # update sensors
            for name, tmp_sensor in self.sensors.items():
                tmp_sensor.update(sim_time, self.entities, self.sensors)

            # get reward of the step
            reward = self.reward_function.calculate_reward(self.entities, self.sensors)

            # look for collisions and set agent to inactive if collided
            self.set_collision_status()

            # update time of the simulation
            sim_time += self.delta_t

            # update history
            for name, tmp_entity in self.entities.items():
                tmp_entity.add_step_history(sim_time)
            for name, tmp_sensor in self.sensors.items():
                tmp_sensor.add_step_history(sim_time)

            # check for termination
            done, done_dict = self.termination_function.calculate_termination(self.entities, self.sensors)

            # give MDP tuple back to agent. The agent chooses if it is going to save the data. It already has the state, action, and next state
            for name, value in self.agents.items():
                tmp_reward = reward[name]
                tmp_done = done_dict[name]
                value.update_memory(tmp_done, self.entities, tmp_reward,self.sensors,sim_time)

                # ask agent if reward functions need to be reset ?
                for tmp_name, tmp_value in self.reward_function.reward_components.items():
                    if tmp_value.target_agent == name:
                        for r_name, r_value in tmp_reward.items():
                            if tmp_value.target_lrn_alg == r_name:
                                tmp_value.reset(self.entities, self.sensors, self.reward_function.reward_agents)

            '''
            # check for leaving the domain. This is seperate of termination conditions
            for name, entity in self.entities.items():
                if self.domain['min_x']-self.domain['buffer'] <= entity.state_dict['x_pos'] <= self.domain['max_x']+self.domain['buffer'] or self.domain['min_y']-self.domain['buffer'] <= entity.state_dict['y_pos'] <= self.domain['max_y']+self.domain['buffer']:
                    done = True
            '''


        # write entity history
        for name, tmp_entity in self.entities.items():
            tmp_entity.write_history(episode_num,history_path)

        # write learning alg history
        for name, tmp_agent in self.agents.items():
            for tmp_name, tmp_lrn_agent in tmp_agent.learning_algorithms.items():
                tmp_lrn_agent.write_history(name,episode_num,history_path)

        # write sensor history
        for name, tmp_sensor in self.sensors.items():
            tmp_sensor.write_history(episode_num,history_path)

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
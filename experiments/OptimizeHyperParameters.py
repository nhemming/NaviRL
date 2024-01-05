"""
Using a kriging model, optimize the hyper parameters for an agent. The kriging model is built with an initial seed
of points from a space filling latin-hyper cube (SF-LHS) sampling plan. Then an adaptive sampling plan is used for
infill using maximum expected improvement as the goal. After the budget is complete, the optimum for a few metrics
is run to train the final agent.

Procedure:
- An input script will need to be manually created. The input script defines the simulation and the agent. This script
will change user specified parameters in this file to optimize the hyperparameters for the agent.
- In the main block of this script, the hyperparameters to change must be manually defined. Which ones, and there max
and min values are needed.
- In the 'utilities' folder, the script 'GeneratesSFLHS.py' can be used to generate a plan. Ideally this is done
before hand but will be executed here if a suitable plan does not exist. The hyperparameters defined in the main body of
this script define the sflhs, with exception of when a new LHS plan is needed. Then there are a few hyper parameters to
add for the sampling plan.
- Next the infill budget is specified. Note the number of infill points defined here will be doubled as the two models
are used for hyper-parameter optimization. Area under the evaluation success curve, and minimum distance to reach the
goal are both optimized. At every hyper-parameter model evaluation step, an infill point, one from each model, is
generated and used for the adaptive sampling.

"""

# native modules
from collections import OrderedDict
import os

# 3rd party modules
import GPy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import yaml

# own modules
from analysis.ExtractEvalData import extract_data
from environment.NavigationEnvironment import NavigationEnvironment
from utilities.GenerateSFLHS import space_filling_latin_hyper_cube



def train_and_eval(set_name,input_file_name, eval_input_file_name):

    env = NavigationEnvironment()
    # get directory of this script
    cur_dir = os.getcwd()
    cur_dir = os.path.join(cur_dir,set_name)

    # create the environment that constructs all the objects.
    env.build_env_from_yaml(input_file_name, cur_dir)
    trial_num = env.h_params['MetaData']['trial_num']
    base_folder = env.h_params['MetaData']['set']

    # run the training. This call isn't actually training, but running an episode to setup the enviornment for evaluation
    env.train_agent() # TODO remove comment

    # run the evaluation script
    env.build_eval_env_from_yaml(eval_input_file_name, cur_dir)

    # create the environment that constructs all the objects.
    env.run_evaluation_set() # TODO remove comment

    # run extract evaluation data
    extract_data(set_name, base_folder , trial_num, env.eval_trial_num) # TODO uncomment

    return base_folder, trial_num, env.eval_trial_num


def edit_simulation_definition_files(sim_def_file,eval_sim_def_file, set_name, index, row):
    # open input file to edit it
    with open(sim_def_file, "r") as stream:
        try:
            sim_def = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # edit input file
    sim_def['MetaData']['trial_num'] = index

    i = 0
    # loop over variables to change
    for name, value in hp_vars_dict.items():
        tabs = name.split(';')
        tmp_var = tabs.pop()
        tmp_def = sim_def
        for tab in tabs:
            tmp_def = tmp_def[tab]

        # convert from normalized DOE point to value in target domain
        if value['type'] == 'int':
            tmp_def[tmp_var] = int(row[i] * (value['max'] - value['min']) + value['min'])
        elif value['type'] == 'float':
            tmp_def[tmp_var] = float(row[i] * (value['max'] - value['min']) + value['min'])

        i += 1

    # save the adjusted input file
    with open(sim_def_file, 'w') as file:
        yaml.safe_dump(sim_def, file)

    # edit eval file
    with open(eval_sim_def_file, "r") as stream:
        try:
            eval_sim_def = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    eval_sim_def['BaseExperiment']['trial_num'] = index
    eval_sim_def['BaseExperiment']['base_folder'] = set_name
    with open(eval_sim_def_file, 'w') as file:
        yaml.safe_dump(eval_sim_def, file)

    return sim_def, eval_sim_def

def add_metrics_to_file(sim_def, results_file_name, agg_df, idx, row, is_infill):
    # open model tracking information, add evaluation metric information and close the file
    df_results = pd.read_csv(results_file_name, index_col=False)

    dict_tmp = dict()
    dict_tmp['Idx'] = idx #j + len(lhs)
    # loop over input vars
    k = 0
    for name, value in hp_vars_dict.items():
        tabs = name.split(';')
        _ = tabs.pop()
        tmp_def = sim_def
        for tab in tabs:
            tmp_def = tmp_def[tab]

        # convert from normalized DOE point to value in target domain
        if value['type'] == 'int':
            dict_tmp[name] = int(row[k] * (value['max'] - value['min']) + value['min'])
            dict_tmp[name + ':norm'] = float(row[k])
        elif value['type'] == 'float':
            dict_tmp[name] = float(row[k] * (value['max'] - value['min']) + value['min'])
            dict_tmp[name + ':norm'] = float(row[k])

        k += 1

    dict_tmp['isInfill'] = is_infill
    dict_tmp['AOC [t-]'] = agg_df[agg_df['name'] == 'AOC']['value'].iloc[0]
    dict_tmp['AvgDstToDest [m]'] = agg_df[agg_df['name'] == 'min_norm_dst_travel']['value'].iloc[0]
    dict_tmp['LongestSuccessRate [-]'] = agg_df[agg_df['name'] == 'longest_success_rate']['value'].iloc[0]
    df_tmp = pd.DataFrame.from_dict([dict_tmp])

    df_results = pd.concat((df_results, df_tmp))
    df_results.to_csv(results_file_name, index=False)

def graph_model(model, n_dim, train_x, train_y, max_ei, model_num, model_name, isOutFlipped, base_path):

    if n_dim == 1:
        sns.set_theme()
        fig = plt.figure(0,figsize=(14,8))
        ax1 = fig.add_subplot(2,1,1)
        x_space = np.linspace(0,1,100)
        x_space_in = np.reshape(x_space,(len(x_space),1))
        y_hat, std_hat = model.predict(x_space_in)
        if isOutFlipped:
            y_hat *= -1.0
        ax1.plot(x_space, y_hat, label='pred')

        y_hat = np.reshape(y_hat, (len(y_hat),))
        std_hat = np.reshape(std_hat, (len(std_hat),))
        ax1.fill_between(x_space, y_hat + std_hat, y_hat - std_hat, alpha=0.3)
        if isOutFlipped:
            ax1.plot(train_x, -1.0*train_y,'o')
        else:
            ax1.plot(train_x, train_y, 'o')
        ax1.legend()

        # plot expected improvement
        ei = np.zeros_like(x_space)
        y_hat, std_hat = model.predict(x_space_in)
        for i, x in enumerate(x_space):
            ei[i] = get_expected_improvement(y_hat[i],train_y.min(),std_hat[i])
        ax2 = fig.add_subplot(2,1,2)
        ax2.plot(x_space,ei)
        ax2.plot([max_ei, max_ei],[0,max(ei)],label='Max EI')
        ax2.set_ylabel('Expected Improvement')
        plt.tight_layout()
        plt.savefig(os.path.join(base_path,str(model_name)+'_'+str(model_num)+'.png'))
        plt.close()

    else:
        # create slices with all x = 0.5 except the varying one vatiable at a time

        var_names = list(hp_vars_dict.keys())
        for i in range(n_dim):
            sns.set_theme()
            fig = plt.figure(0, figsize=(14, 8))
            ax1 = fig.add_subplot(2, 1, 1)

            x_in = np.ones((100,len(train_x[0,:])))*0.5
            x_in[:,i] = np.linspace(0,1,len(x_in))

            y_hat, std_hat = model.predict(x_in)
            y_hat = np.reshape(y_hat, (len(y_hat),))

            y_hat = np.reshape(y_hat, (len(y_hat),))
            std_hat = np.reshape(std_hat, (len(std_hat),))
            if isOutFlipped:
                y_hat *= -1.0
            ax1.plot(np.linspace(0,1,len(x_in)),y_hat)
            ax1.fill_between(np.linspace(0,1,len(x_in)), y_hat + std_hat, y_hat - std_hat, alpha=0.3)
            ax1.set_xlabel(var_names[i])
            ax1.set_ylabel(model_name)
            ax1.legend()

            ei = np.zeros((len(x_in),))
            y_hat, std_hat = model.predict(x_in)
            for j, x in enumerate(x_in):
                ei[j] = get_expected_improvement(y_hat[j], train_y.min(), std_hat[j])
            ax2 = fig.add_subplot(2, 1, 2)
            ax2.plot(x_in[:,i], ei)
            ax2.plot([x_in[np.argmax(ei),i], x_in[np.argmax(ei),i]], [0, max(ei)], label='Max EI')
            ax2.set_xlabel(var_names[i])
            ax2.set_ylabel('Expected Improvement')
            plt.tight_layout()
            plt.savefig(os.path.join(base_path, str(model_name) + '_' + str(model_num) + 'Slice_'+str(var_names[i])+'.png'))
            plt.close()

        if n_dim == 2:
            # make contour plots
            xv, yv = np.meshgrid(np.linspace(0, 1, 25), np.linspace(0, 1, 25), indexing='ij')
            x1 = np.reshape(xv, (len(xv) * len(xv[0, :]),))
            x2 = np.reshape(yv, (len(yv) * len(yv[0, :]),))
            x = np.zeros((len(x1), 2))
            x[:, 0] = x1
            x[:, 1] = x2
            y_hat, _ = model.predict(x)
            y_hat = np.reshape(y_hat, (len(y_hat),))
            if isOutFlipped:
                y_hat *= -1.0

            fig = plt.figure(0, figsize=(14, 8))
            ax = fig.add_subplot(1, 1, 1)
            cs = ax.tricontourf(x[:, 0], x[:, 1], y_hat, levels=20)
            plt.colorbar(cs)
            ax.set_xlabel(list(hp_vars_dict.keys())[0])
            ax.set_ylabel(list(hp_vars_dict.keys())[1])
            ax.set_title(str(model_name)+'_' + str(model_num))
            plt.tight_layout()
            plt.savefig(os.path.join(base_path,str(model_name)+'_'+str(model_num)+'.png'))
            plt.close()

def build_krig(x,y, hp_dict):
    """
    Builds a kriging model given input and target data along with some hyperparameter info.

    :param x: Input data. The rows are a single input. The columns are unique inputs.
    :param y: Target data. Can only be a single column of data with the same length as the input data.
    :param hp_dict: Dictionary for the hyper parameters for the kernel, and other training parameters
    :return: A built kriging model.
    """

    #kernel = GPy.kern.RBF(input_dim=len(x[0,:]),variance=hp_dict['init_variance'],lengthscale=hp_dict['init_length_scale']) + GPy.kern.White(len(x[0,:]))
    kernel = GPy.kern.Matern32(input_dim=len(x[0,:]),variance=hp_dict['init_variance'],lengthscale=hp_dict['init_length_scale']) #+ GPy.kern.White(len(x[0,:]))

    y = np.reshape(y,(len(y),1))
    krig = GPy.models.GPRegression(x,y,kernel)
    krig.optimize_restarts(hp_dict['n_restarts'], messages=False, max_f_eval=1000)

    return krig


def get_max_expected_improvement(krig, n_dim, y_min, nm_hp):
    """
    Use an optimizer to find the point that has the maximum expected improvement of the Kriging model.

    :param krig: A pre-built kriging model
    :param y_min: The minimum output value for the entire data set used for training the kriging model.
    :param nm_hp: A dictionary of hyperparameters for the optimizer.
    :return: The vector of normalized input values that create the maximum expected improvement.
    """
    min_sol = None
    min_fun_val = np.infty
    for i in range(nm_hp['n_restarts']):

        x0 = np.random.uniform(low=0, high=1, size=(n_dim,))
        bounds = []
        for _ in range(n_dim):
            bounds.append([0,1])

        opt_min = scipy.optimize.minimize(opt_expected_improvement_fun,x0,args=(krig, y_min),method='Nelder-Mead', bounds=bounds)

        sol = opt_min.x
        fun_val = opt_min.fun
        if fun_val < min_fun_val:
            min_fun_val = fun_val
            min_sol = sol

    return min_sol


def opt_expected_improvement_fun(x, krig, y_min):
    """
    Gets the expected improvement of the kKriging model at a specfied point. This method should not be called direcetly.

    :param x: The sample point to get the expected improvement at
    :param krig: A pre-built Kriging model
    :param y_min: The minimum output value for the training set used for the kriging model
    :return: The negative of the maximum expected improvement.
    """
    x = np.reshape(x,(1,len(x)))
    mean, std = krig.predict(x)
    mean = mean[0][0]
    ei = get_expected_improvement(mean, y_min, std[0][0])
    return -ei


def get_expected_improvement(y_hat,y_min,s):
    """
    Helper function that calculates the expected improvement.

    :param y_hat: Predicted output
    :param y_min:  The minimum output value for the training set used for the kriging model
    :param s: standard deviation from the model
    :return: expected improvement.
    """
    return (y_min - y_hat)*(0.5 + 0.5*scipy.special.erf((y_min-y_hat)/(s*np.sqrt(2.0))))+s/np.sqrt(2.0*np.pi)*np.exp(-(y_min-y_hat)**2/(2*s*s))


def get_optimal_settings(krig, nm_hp):
    min_sol = None
    min_fun_val = np.infty
    for i in range(nm_hp['n_restarts']):

        x0 = np.random.uniform(low=0, high=1, size=(2,))

        opt_min = scipy.optimize.minimize(opt_fun, x0, args=(krig), method='Nelder-Mead',
                                          bounds=[(0, 1), (0, 1)])

        sol = opt_min.x
        fun_val = opt_min.fun
        if fun_val < min_fun_val:
            min_fun_val = fun_val
            min_sol = sol

    return min_sol


def opt_fun(x, krig):
    x = np.reshape(x, (1, len(x)))
    mean, std = krig.predict(x)
    return mean[0][0]


def optimize_h_params(set_name, sim_def_file_name, eval_sim_def_file_name, hp_vars_dict,doe_hp, n_infill, krig_hp, nm_hp):
    """
    Works to optimize a set of hyperparameters for an agent that performs navigation. This works for both learning and
    non learning agent. A space filling latin hyper cube sampling plan is used for the initial DOE. Each point in that
    DOE is run, evaluation episodes are run, and metrics are extracted. After the initial DOE is complete, adaptive
    sampling is used for infill points. A Kriging model is built using all of the previous samples. Then the point of
    maximum expected improvement is found from the Kriging model. Then that point is run and evaluated. The Kriging
    model builder and running process ius repeated until a budget of infill points are exhausted. If the process
    is stopped early, it can be restarted. Two metrics are used for optimizing. Area under the curve (AOC) for the
    integration under the success rate curve over training episode. The second is average distance to reach the goal in
    the evaluation set. Both have Kriging models built and propose new sampling points. If a sample point is proposed
    that is too near an existing point, a random point is drawn in the design space. A csv of results are saved
    in the set_name folder. If more infill points are desired, increase the number of infill points and recall this
    function.

    :param set_name: The name of the folder to place all of the simulations in side of. This should be created prior
        to running this  method.
    :param sim_def_file_name: The name of the file defining simulation. This file must be in the set_name folder
    :param eval_sim_def_file_name: The name of the file defining the evaluation set. This file must be in the set_name
        folder.
    :param hp_vars_dict: A dictionary of variables to optimize over. Each entry in the dictionary is another dictionary.
        An example for one of the entries:
            hp_0 = {'name': 'Sensors;sensor1;n_samples','min':100,'max':1000,'type':'int'}
            The name field should be the scope to variable in the sim_def_file that is delimited with semicolons.
            The min and max fields are the minimum and maximum values to vary the variable.
            The type is the variable type to help the definition files be edited correctly.
        The dictionary entries must be saved in the hp_vars_dict with their name field as their key.
    :param doe_hp: A dictionary containing hyper parameter(s) for which SFLHS DOE to use.
    :param n_infill: The number of times to build a Kriging model and sample new points.
    :param krig_hp: A dictionary of hyper parameters for how to build the Kriging model. Manly initial guesses for the
        kernel and how many random restarts to do.
    :param nm_hp: A dictionary for the hyper parameters for the optimizer. Currently only the number of random restarts.
    :return:
    """
    sns.set_theme()

    # create empty file for storing model information w.r.t. DOE progress
    if not os.path.isdir(set_name):
        os.mkdir(set_name) # file should mannualy be created by the user

    # make a folder for model images
    if not os.path.isdir(os.path.join(set_name,'model_images')):
        os.mkdir(os.path.join(set_name,'model_images'))

    sim_def_file = os.path.join(os.getcwd(), set_name, sim_def_file_name)
    if not os.path.exists(sim_def_file):
        raise ValueError('Specified simulation file does not exist')

    eval_sim_def_file = os.path.join(os.getcwd(), set_name, eval_sim_def_file_name)
    if not os.path.exists(eval_sim_def_file):
        raise ValueError('Specified simulation file does not exist')

    cols = ['Idx'] + list(hp_vars_dict.keys()) + list([i+':norm' for i in hp_vars_dict.keys()]) + ['isInfill','AOC [t-]','AvgDstToDest [m]','LongestSuccessRate [-]']
    df = pd.DataFrame(columns=cols)
    results_file_name = os.path.join(set_name,'HParamOptResults.csv')
    if not os.path.exists(results_file_name):
        df.to_csv(results_file_name, index=False)
    else:
        tmp_df = pd.read_csv(results_file_name,index_col=False)
        if len(tmp_df) == 0:
            # only create file of the data is empty so redoing points is not needed.
            df.to_csv(results_file_name,index=False)
    tmp_df = pd.read_csv(results_file_name,index_col=False)

    # generate SF-LHS plan as needed
    doe_file_path = os.path.join('..','utilities','Npoints-'+str(doe_hp['n_points'])+'_Nvars-'+str(len(list(hp_vars_dict.keys())))+'_sflhs.csv')
    if not os.path.exists(doe_file_path):
        #space_filling_latin_hyper_cube(doe_hp['n_points'],len(list(hp_vars_dict.keys())), doe_hp['population'], doe_hp['generations'])
        raise ValueError('DOE does not exist. Please run GenerateSFLHS.py to create a plan matching the requested data')

    # open DOE
    lhs = pd.read_csv(doe_file_path,index_col=False)

    # loop over DOE entries
    for index, row in lhs.iterrows():

        print('Point {:d} of {:d} from the SFLHS'.format(int(index + 1), len(lhs)))

        if index >= len(tmp_df):
            # point has not been run yet so run the point

            # change the input files to use the variables called out in the DOE
            sim_def, eval_sim_def = edit_simulation_definition_files(sim_def_file, eval_sim_def_file, set_name, index, row)

            # launch training
            base_folder, trial_num, eval_trial_num = train_and_eval(set_name,sim_def_file, eval_sim_def_file)

            # get evaluation metrics
            agg_file = sim_def_file.replace(sim_def_file_name,'')
            agg_file = os.path.join(agg_file,'output',str(base_folder), str(trial_num),'evaluation',str(eval_trial_num),'AggregateResults.csv')
            agg_df = pd.read_csv(agg_file,index_col=False)

            add_metrics_to_file(sim_def, results_file_name, agg_df, index, row, False) # TODO uncomment this

    # loop over number of infill instances
    tot_points = n_infill+len(lhs)
    tmp_df = pd.read_csv(results_file_name, index_col=False)
    n_infill = tot_points-len(tmp_df) # adjust for already having run some infill points if needed.
    for i in range(n_infill):

        print("Infill Point {:d}".format(i+1))

        # open data
        df_results = pd.read_csv(results_file_name, index_col=False)

        # build model for AOC
        inp_names = [i+':norm' for i in hp_vars_dict.keys()]
        train_x = df_results[inp_names].to_numpy()
        train_y = -1.0*df_results['AOC [t-]'].to_numpy() # make trainy negative because we want to maximize the data
        krig_aoc = build_krig(train_x, train_y, krig_hp)

        # get point of maximum expectation for AOC model
        point_max_exp_aoc = get_max_expected_improvement(krig_aoc, len(hp_vars_dict.keys()), train_y.min(), nm_hp)

        # save AOC model ... why?
        graph_model(krig_aoc, len(hp_vars_dict.keys()),train_x, train_y, point_max_exp_aoc, i, 'AOC', True, os.path.join(set_name,'model_images'))

        # build model for time to destination
        train_y = df_results['AvgDstToDest [m]'].to_numpy()
        krig_adtd = build_krig(train_x, train_y, krig_hp)

        # get point of maximum expectation for time to destination model
        point_max_exp_adtd = get_max_expected_improvement(krig_adtd,len(hp_vars_dict.keys()), train_y.min(), nm_hp)

        # save the time to destination model ... why?
        graph_model(krig_adtd,len(hp_vars_dict.keys()),train_x, train_y, point_max_exp_adtd, i, 'AvgTimeToDest', False, os.path.join(set_name,'model_images'))

        infill_points = [point_max_exp_aoc, point_max_exp_adtd]
        # loop over two new infill points
        for j, ip in enumerate(infill_points):

            # check if ip is near a point already sampled. If so, draw a random point to test
            n_attempts = 0
            while n_attempts < 50:
                diff = np.zeros(len(train_x))
                for k, tx in enumerate(train_x):
                    diff[k] = np.max(np.abs(tx-ip))
                if diff.min() <= 0.02: # 2 percent near another point.
                    # draw a random sample
                    ip = np.random.uniform(low=0, high=1, size=(len(ip),))
                else:
                    break
                n_attempts += 1

            # change the input files to use the variables called out in the DOE
            sim_def, eval_sim_def = edit_simulation_definition_files(sim_def_file, eval_sim_def_file, set_name, j+len(df_results), ip)

            # launch training and evaluatiuon of the agent
            base_folder, trial_num, eval_trial_num = train_and_eval(set_name,sim_def_file, eval_sim_def_file)

            # get evaluation metrics
            agg_file = sim_def_file.replace(sim_def_file_name, '')
            agg_file = os.path.join(agg_file, 'output', str(base_folder), str(trial_num), 'evaluation',
                                    str(eval_trial_num), 'AggregateResults.csv')
            agg_df = pd.read_csv(agg_file, index_col=False)

            add_metrics_to_file(sim_def, results_file_name, agg_df, j+len(df_results), ip, True) # TODO uncomment

    # build model for AOC
    df_results = pd.read_csv(results_file_name, index_col=False)
    inp_names = [i + ':norm' for i in hp_vars_dict.keys()]
    train_x = df_results[inp_names].to_numpy()
    train_y = -1.0*df_results['AOC [t-]'].to_numpy() # max negative as we are maximizing
    krig_aoc = build_krig(train_x, train_y, krig_hp)

    # get point of maximum expectation for AOC model
    point_max_exp_aoc = get_max_expected_improvement(krig_aoc, len(hp_vars_dict.keys()), train_y.min(), nm_hp)
    graph_model(krig_aoc, len(hp_vars_dict.keys()), train_x, train_y, point_max_exp_aoc, n_infill, 'AOC_Last', True,
                os.path.join(set_name, 'model_images'))

    # get optimal hyper-parameter set from AOC model
    curr_best_settings_aoc = get_optimal_settings(krig_aoc, nm_hp)

    # run simulation for optimal AOC set
    # change the input files to use the variables called out in the DOE
    sim_def, eval_sim_def = edit_simulation_definition_files(sim_def_file, eval_sim_def_file, set_name, len(df_results), curr_best_settings_aoc)

    # launch training and evaluatiuon of the agent
    base_folder, trial_num, eval_trial_num = train_and_eval(set_name, sim_def_file, eval_sim_def_file)

    # get evaluation metrics
    agg_file = sim_def_file.replace(sim_def_file_name, '')
    agg_file = os.path.join(agg_file, 'output', str(base_folder), str(trial_num), 'evaluation',
                            str(eval_trial_num), 'AggregateResults.csv')
    agg_df = pd.read_csv(agg_file, index_col=False)

    add_metrics_to_file(sim_def, results_file_name, agg_df, len(df_results), curr_best_settings_aoc, True)

    # build model for time to destination
    train_y = df_results['AvgDstToDest [m]'].to_numpy()
    krig_adtd = build_krig(train_x, train_y, krig_hp)

    # get point of maximum expectation for time to destination model
    point_max_exp_adtd = get_max_expected_improvement(krig_adtd, len(hp_vars_dict.keys()), train_y.min(), nm_hp)
    # save the time to destination model ... why?
    graph_model(krig_adtd, len(hp_vars_dict.keys()), train_x, train_y, point_max_exp_adtd, n_infill, 'AvgTimeToDest_Last', False,
                os.path.join(set_name, 'model_images'))

    # get optimal hyper-parameter set from time to destination model
    curr_best_settings_adtd = get_optimal_settings(krig_adtd, nm_hp)

    # change the input files to use the variables called out in the DOE
    sim_def, eval_sim_def = edit_simulation_definition_files(sim_def_file, eval_sim_def_file, set_name, len(df_results),
                                                             curr_best_settings_adtd)

    # launch training and evaluatiuon of the agent
    base_folder, trial_num, eval_trial_num = train_and_eval(set_name, sim_def_file, eval_sim_def_file)

    # get evaluation metrics
    agg_file = sim_def_file.replace(sim_def_file_name, '')
    agg_file = os.path.join(agg_file, 'output', str(base_folder), str(trial_num), 'evaluation',
                            str(eval_trial_num), 'AggregateResults.csv')
    agg_df = pd.read_csv(agg_file, index_col=False)

    add_metrics_to_file(sim_def, results_file_name, agg_df, len(df_results), curr_best_settings_adtd, True)


if __name__ == '__main__':

    # definition of the hyper parameters
    set_name = 'demo_to_test_2Dhparam_opt'
    sim_def_file_name = 'experiment_setup_RRTStar.yaml'
    eval_sim_def_file_name = 'no_obstacle_mass_free_evaluation_set.yaml'
    hp_vars_dict = OrderedDict()

    hp_0 = {'name': 'Sensors;sensor1;n_samples','min':100,'max':1000,'type':'int'}
    hp_vars_dict[hp_0['name']] = hp_0

    hp_1 = {'name': 'Sensors;sensor1;link_dst', 'min': 0.3, 'max': 2.5,'type':'float'} # 0.5 default
    hp_vars_dict[hp_1['name']] = hp_1

    # number of initial doe samples
    doe_hp = {'n_points' : 20}

    # number of infill searches. Note the number of infill points will be double this value
    n_infill = 10

    # kriging model hyper parameters
    krig_hp = {'init_length_scale':0.2,
        'init_variance':0.05,
        'n_restarts':25}

    # optimizer (nelder mead) hyper parameters
    nm_hp = {'n_restarts':25}


    # edit above ^^

    optimize_h_params(set_name, sim_def_file_name, eval_sim_def_file_name, hp_vars_dict,doe_hp, n_infill, krig_hp, nm_hp)
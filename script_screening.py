# Copyright by Paul Rudolph
# Research Group Applied Systems Biology - Head: Prof. Dr. Marc Thilo Figge
# https://www.leibniz-hki.de/en/applied-systems-biology.html
# HKI-Center for Systems Biology of Infection
# Leibniz Institute for Natural Product Research and Infection Biology - Hans Knöll Insitute (HKI)
# Adolf-Reichwein-Straße 23, 07745 Jena, Germany
# This code is licensed under BSD 2-Clause
# See the LICENSE file provided with this code for the full license.

import multiprocessing
import functools
from scipy.stats import qmc
import numpy as np
import pandas as pd
import sys
import os
from model import system_size,time_scaling,ldh_model,cal_model,cal_nb_model,candida_model
from fitting import simulate, check_objective,get_parameters,distance,distance_max,log_distance,log_distance_max, parallelize
from model_utils import objective_LDH,objective_CAL,objective_Nb,generate_data_LDH,generate_data_CAL,generate_data_Nb,prepare_data_LDH,prepare_data_CaL,prepare_data_Nb,objective_Candida, prepare_data_Candida,generate_data_Candida
import preprocessing
from tqdm import tqdm

MAX_CPUS = int(multiprocessing.cpu_count()* 0.785) # Maximum number of cpus used, in total do around 80% since we have an AMD architecure
MAX_ITERATIONS = 10000 # Maximum iterations of scipy optimize fitting steps
NSP = 1000000 # Number of sampling points for fitting as initial values

ldh_model_ptr = ldh_model.address
cal_model_ptr = cal_model.address
ab_cal_model_ptr = cal_nb_model.address
candida_model_ptr = candida_model.address

DISTANCE_MEASURE = distance
DISTANCE_MAX_MEASURE = distance_max
DATA_AGG = "mean"
DATA_FOLDER = os.path.abspath("../DATA/")
RESULT_FOLDER = os.path.abspath("../PUBLICATION/SIMULATION")

file_name = os.path.join(DATA_FOLDER,"Experimental Data For Modelling.xlsx")
# The loop iterates over different aggregate sizes and then fits in sequential fashion 4 different models with increasing complexity
# For each model:
# 1) The bounds for parameter fit are set (which are very broad)
# 2) An objective function is created
# 3) Initial values for the fitting are computed using LatinHypercube sampling
# 4) The fitting is done in parallel
# 5) All results are saved but ordered by the best objective evaluation
for adjust_CaL in [8,16,32,64,128,256,512,1024]:
    # Create Output Folder
    OUTPUT_FOLDER = os.path.join(RESULT_FOLDER,str(adjust_CaL))
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    # Read in and preprocess data
    df_ex5,df_ex3,df_ex3_candida,df_ex1,df_fig_cal,df_fig_candida = preprocessing.read_data(file_name)
    df_data_LDH,df_data_sim,df_data_pre,df_data_pre_co,df_data_post,df_data_candida_sim,df_data_candida_pre,df_data_candida_post = preprocessing.preprocess_data(file_name,time_scaling, adjust_CaL=adjust_CaL, adjust_LDH=True,system_size=system_size)

    # LDH Model setup
    funcptr = ldh_model_ptr
    fitting_parameters = ["k_l", "beta"]
    bounds = {"k_l":[1e-6,1e6],
              "beta":[1e-6,1e6]}
    output_file = os.path.join(OUTPUT_FOLDER,f"LDH.feather")
    order_parameters = ["k_l","beta"]
    all_parameters ={"k_l":.1,"beta":2}
    no_fitting_dict = {key:value for key,value in all_parameters.items() if key not in fitting_parameters}
    prepared_data = prepare_data_LDH(df_data_LDH)
    numbek_starting_points = NSP
    def objective_wrapper_ldh(parameters):
        return objective_LDH(funcptr, prepared_data, log_distance, *get_parameters(parameters, fitting_parameters, no_fitting_dict,order_parameters))
    list_parameters = [all_parameters[i] for i in order_parameters]
    wrap_obj = lambda condition,*paras: objective_LDH(funcptr, condition, log_distance, *paras)
    wrap_gen = lambda *paras: generate_data_LDH(funcptr,  *paras)
    print("Check LSE for true parameters: ",check_objective(wrap_obj,wrap_gen, *list_parameters))
    print("Check LSE on real data:",objective_LDH(funcptr, prepared_data, log_distance, *list_parameters))

    # LDH Model fitting procedure
    list_bounds= [(bounds[i]) for i in fitting_parameters]
    sampler = qmc.LatinHypercube(d=len(fitting_parameters))
    sample = sampler.random(n=numbek_starting_points)
    starting_points = np.exp(qmc.scale(sample, np.log(np.array(list_bounds)[:,0]), np.log(np.array(list_bounds)[:,1])))
    constant_args = (objective_wrapper_ldh, list_bounds, MAX_ITERATIONS)
    partial_parallelize = functools.partial(parallelize, *constant_args)
    with multiprocessing.Pool(np.minimum(numbek_starting_points,MAX_CPUS)) as pool:
        results = list(tqdm(pool.imap(partial_parallelize, starting_points), total=len(starting_points), file=sys.stdout))
    df_fit_ldh = pd.DataFrame([(*result.x,result.fun) for result in results],columns=fitting_parameters+["Fun"]).sort_values("Fun",ascending=True)
    df_fit_ldh.reset_index(drop=True).to_feather(output_file)
    df_fit_ldh.sort_values("Fun").head(1)
    print("Check LSE after fit on real data:",objective_LDH(funcptr, prepared_data, log_distance, *df_fit_ldh.iloc[0][fitting_parameters].to_list()))

    # CAL Model setup
    funcptr = cal_model_ptr
    numbek_starting_points = NSP
    all_parameters ={
        'k_a': 1e-5,
        'k_c': 0.0,
        "k_l":df_fit_ldh.iloc[0]["k_l"],
        'k_d': 1e4,
        'alpha': 1e-6,
        "beta":df_fit_ldh.iloc[0]["beta"]}
    order_parameters = ['k_a', 'k_c','k_l', 'k_d', 'alpha', 'beta']
    fitting_parameters = ['k_a', 'k_d', 'alpha']
    no_fitting_dict = {key:value for key,value in all_parameters.items() if key not in fitting_parameters}
    bounds = {"k_a":[1e-7,1e4],
             "k_d":[1e0,1e12],
             "alpha":[1e-8,1e-4]}
    data_obj = prepare_data_CaL(df_data_sim.query("Nb == 0.0"), all_parameters["beta"], agg=DATA_AGG)
    output_file = os.path.join(OUTPUT_FOLDER,f"CAL.feather")
    def objective_wrapper_cal(parameters):
        return objective_CAL(funcptr, data_obj, DISTANCE_MEASURE, DISTANCE_MAX_MEASURE, *get_parameters(parameters, fitting_parameters, no_fitting_dict,order_parameters))
    # CAL Model fitting procedure
    list_parameters = [all_parameters[i] for i in order_parameters]
    wrap_obj = lambda condition,*paras: objective_CAL(funcptr, condition, DISTANCE_MEASURE, DISTANCE_MAX_MEASURE, *paras)
    wrap_gen = lambda *paras: generate_data_CAL(funcptr, *paras)
    print("Check LSE for true parameters: ",check_objective(wrap_obj,wrap_gen, *list_parameters))
    print("Check LSE on real data:",objective_CAL(funcptr, data_obj, DISTANCE_MEASURE, DISTANCE_MAX_MEASURE, *list_parameters))
    list_bounds= [(bounds[i]) for i in fitting_parameters]
    sampler = qmc.LatinHypercube(d=len(fitting_parameters))
    sample = sampler.random(n=numbek_starting_points)
    starting_points = np.exp(qmc.scale(sample, np.log(np.array(list_bounds)[:,0]), np.log(np.array(list_bounds)[:,1])))
    constant_args = (objective_wrapper_cal, list_bounds, MAX_ITERATIONS)
    partial_parallelize = functools.partial(parallelize, *constant_args)
    with multiprocessing.Pool(np.minimum(numbek_starting_points,MAX_CPUS)) as pool:
        results = list(tqdm(pool.imap(partial_parallelize, starting_points), total=len(starting_points), file=sys.stdout))
    df_fit_cal = pd.DataFrame([(*result.x,result.fun) for result in results],columns=fitting_parameters+["Fun"]).sort_values("Fun",ascending=True)
    df_fit_cal.reset_index(drop=True).to_feather(output_file)
    df_fit_cal.sort_values("Fun").head(1)
    print("Check LSE after fit on real data:",objective_CAL(funcptr, data_obj, DISTANCE_MEASURE, DISTANCE_MAX_MEASURE, *get_parameters(df_fit_cal.iloc[0][fitting_parameters].to_list(), fitting_parameters, no_fitting_dict,order_parameters)))

    # NB CAL Model setup
    funcptr = ab_cal_model_ptr
    numbek_starting_points = NSP
    all_parameters ={
        'k_a': df_fit_cal.iloc[0]["k_a"],
        'k_n':0.0,
        'k_b':0.0,
        'k_c':  0.0,
        "k_l":df_fit_ldh.iloc[0]["k_l"],
        'k_d':  df_fit_cal.iloc[0]["k_d"],
        'A':  adjust_CaL,
        'alpha':  df_fit_cal.iloc[0]["alpha"],
        "beta":df_fit_ldh.iloc[0]["beta"]}
    order_parameters = ['k_a', 'k_c', 'k_n','k_l','k_b', 'k_d','A',  'alpha', 'beta']
    fitting_parameters = ['k_b', 'k_n']
    no_fitting_dict = {key:value for key,value in all_parameters.items() if key not in fitting_parameters}
    bounds = {"k_n":[1e-20,1e10],
             "k_b":[1e1,1e5]}
    output_file = os.path.join(OUTPUT_FOLDER,f"NB_CAL.feather")
    data_obj = prepare_data_Nb(df_data_sim,df_data_pre,df_data_pre_co,df_data_post, all_parameters["beta"], agg=DATA_AGG)
    def objective_wrapper_ab_cal(parameters):
        return objective_Nb(funcptr, data_obj, DISTANCE_MEASURE, DISTANCE_MAX_MEASURE,*get_parameters(parameters, fitting_parameters, no_fitting_dict,order_parameters))
    list_parameters = [all_parameters[i] for i in order_parameters]
    wrap_obj = lambda condition,*paras: objective_Nb(funcptr, condition, DISTANCE_MEASURE, DISTANCE_MAX_MEASURE, *paras)
    wrap_gen = lambda *paras: generate_data_Nb(funcptr, *paras)
    print("Check LSE for true parameters: ",check_objective(wrap_obj,wrap_gen, *list_parameters))
    print("Check LSE on real data:",objective_Nb(funcptr, data_obj, DISTANCE_MEASURE, DISTANCE_MAX_MEASURE, *list_parameters))
    # NB CAL Model fitting procedure
    list_bounds= [(bounds[i]) for i in fitting_parameters]
    sampler = qmc.LatinHypercube(d=len(fitting_parameters))
    sample = sampler.random(n=numbek_starting_points)
    starting_points = np.exp(qmc.scale(sample, np.log(np.array(list_bounds)[:,0]), np.log(np.array(list_bounds)[:,1])))
    constant_args = (objective_wrapper_ab_cal, list_bounds, MAX_ITERATIONS)
    partial_parallelize = functools.partial(parallelize, *constant_args)
    with multiprocessing.Pool(np.minimum(numbek_starting_points,MAX_CPUS)) as pool:
        results = list(tqdm(pool.imap(partial_parallelize, starting_points), total=len(starting_points), file=sys.stdout))
    df_fit_ab = pd.DataFrame([(*result.x,result.fun) for result in results],columns=fitting_parameters+["Fun"]).sort_values("Fun",ascending=True)
    df_fit_ab.reset_index(drop=True).to_feather(output_file)
    df_fit_ab.sort_values("Fun").head(1)
    print("Check LSE after fit on real data:",objective_Nb(funcptr, data_obj, DISTANCE_MEASURE, DISTANCE_MAX_MEASURE, *get_parameters(df_fit_ab.iloc[0][fitting_parameters].to_list(), fitting_parameters, no_fitting_dict,order_parameters)))

    # CANDIDA Model setup
    funcptr = candida_model_ptr
    numbek_starting_points = NSP
    all_parameters ={
        'k_a': df_fit_cal.iloc[0]["k_a"],
        'k_n':df_fit_ab.iloc[0]["k_n"],
        'k_b':df_fit_ab.iloc[0]["k_b"],
        'k_c':  0.0,
        "k_l":df_fit_ldh.iloc[0]["k_l"],
        'k_d':  df_fit_cal.iloc[0]["k_d"],
        'A':  adjust_CaL,
        'alpha':  df_fit_cal.iloc[0]["alpha"],
        "beta":df_fit_ldh.iloc[0]["beta"],
        "k_s": 1.0}
    order_parameters = ['k_a', 'k_c', 'k_n','k_l','k_b', 'k_d','A', 'alpha', 'beta','k_s']
    fitting_parameters = ['k_s']
    no_fitting_dict = {key:value for key,value in all_parameters.items() if key not in fitting_parameters}
    bounds = {"k_s":[1e-20,1e8]}
    output_file = os.path.join(OUTPUT_FOLDER,f"CANDIDA.feather")
    data_obj = prepare_data_Candida(df_data_candida_sim,df_data_candida_pre,df_data_candida_post,df_fit_ldh.iloc[0]["beta"])
    def objective_wrapper_candida(parameters):
        return objective_Candida(funcptr, data_obj, DISTANCE_MEASURE, DISTANCE_MAX_MEASURE,*get_parameters(parameters, fitting_parameters, no_fitting_dict,order_parameters))
    list_parameters = [all_parameters[i] for i in order_parameters]
    wrap_obj = lambda condition,*paras: objective_Candida(funcptr, condition, DISTANCE_MEASURE, DISTANCE_MAX_MEASURE, *paras)
    wrap_gen = lambda *paras: generate_data_Candida(funcptr, *paras)

    print("Check LSE for true parameters: ",check_objective(wrap_obj,wrap_gen, *list_parameters))
    print("Check LSE on real data:",objective_Candida(funcptr, data_obj, DISTANCE_MEASURE, DISTANCE_MAX_MEASURE, *list_parameters))
    # CANDIDA Model fitting procedure
    list_bounds= [(bounds[i]) for i in fitting_parameters]
    sampler = qmc.LatinHypercube(d=len(fitting_parameters))
    sample = sampler.random(n=numbek_starting_points)
    starting_points = np.exp(qmc.scale(sample, np.log(np.array(list_bounds)[:,0]), np.log(np.array(list_bounds)[:,1])))
    constant_args = (objective_wrapper_candida, list_bounds, MAX_ITERATIONS)
    partial_parallelize = functools.partial(parallelize, *constant_args)
    with multiprocessing.Pool(np.minimum(numbek_starting_points,MAX_CPUS)) as pool:
        results = list(tqdm(pool.imap(partial_parallelize, starting_points), total=len(starting_points), file=sys.stdout))
    df_fit_candida= pd.DataFrame([(*result.x,result.fun) for result in results],columns=fitting_parameters+["Fun"]).sort_values("Fun",ascending=True)
    df_fit_candida.reset_index(drop=True).to_feather(output_file)
    print("Check LSE after fit on real data:",objective_Candida(funcptr, data_obj, DISTANCE_MEASURE, DISTANCE_MAX_MEASURE, *get_parameters(df_fit_candida.iloc[0][fitting_parameters].to_list(), fitting_parameters, no_fitting_dict,order_parameters)))

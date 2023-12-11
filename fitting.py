# Copyright by Paul Rudolph
# Research Group Applied Systems Biology - Head: Prof. Dr. Marc Thilo Figge
# https://www.leibniz-hki.de/en/applied-systems-biology.html
# HKI-Center for Systems Biology of Infection
# Leibniz Institute for Natural Product Research and Infection Biology - Hans Knöll Insitute (HKI)
# Adolf-Reichwein-Straße 23, 07745 Jena, Germany
# This code is licensed under BSD 2-Clause
# See the LICENSE file provided with this code for the full license.

from numbalsoda import lsoda
from numba import njit
import os
import pandas as pd
import numpy as np
from scipy import optimize
import sys 
import multiprocessing
import functools

from tqdm import tqdm

MAX_CPUS = int(multiprocessing.cpu_count()* 0.785) # Maximum number of cpus used, in total do around 80% since we have an AMD architecure


@njit
def distance(x,y):
    return (x-y)**2
@njit
def log_distance(x,y):
    shift = 1e-13
    return (np.log(x+shift)-np.log(y+shift))**2
@njit
def distance_max(x,y):
    return np.maximum(y-x,0.0)**2
@njit
def log_distance_max(x,y):
    shift = 1e-13
    res = np.log(y+shift)-np.log(x+shift)
    return np.where(y-x>=0,res,0.0)**2

@njit
def simulate(params, initials, t_eval,funcptr):
    '''Simulate a given ODE model with parameters and intital values for certain time point (t_eval)'''
    num_species = len(initials)
    solution_fill_value= 10 # if lsoda stops without finishing, all time points which have not been computed are replaced by this value (to avoid nans)
    solution = np.ones((num_species,len(t_eval)))*solution_fill_value
    sol, _ = lsoda(funcptr, np.copy(initials), t_eval, data = params)
    solution[:,:sol.T.shape[1]] = sol.T
    return np.where(solution >= 0.0,solution,0.0)

def check_objective(fun_obj,fun_gen, *parameters):
    conditions = fun_gen(*parameters)
    return fun_obj(conditions, *parameters)

def get_parameters(parameters, fitting_parameters, no_fitting_dict,order_parameters):
    fitting_dict = dict(zip(fitting_parameters,parameters))
    fitting_dict.update(no_fitting_dict)
    return [fitting_dict[i] for i in order_parameters]

def parallelize(objective_wrapper, list_bounds, max_iterations, starting_point):
    return optimize.minimize(objective_wrapper,bounds=list_bounds,x0=starting_point)

def parallelize_validation(current_value,critical_value, factor, para, fitting_parameters, no_fitting, likelihood_fun,do_parameters,order_parameters,initial_parameters,null_nllh,test_value):
        no_fitting_dict ={para: test_value}
        no_fitting_dict.update(no_fitting)
        optimize_fun = lambda paras: likelihood_fun(*get_parameters(paras, do_parameters, no_fitting_dict,order_parameters))
        if len(fitting_parameters) != 1:
            initial_parameters = optimize.minimize(optimize_fun, x0=initial_parameters,bounds=[(para2*1/(factor),para2*(factor))for para2 in initial_parameters]).x
        new_test_parameters = get_parameters(initial_parameters, do_parameters, no_fitting_dict,order_parameters)
        alt_nllh = -likelihood_fun(*new_test_parameters)
        test_statistic = 2*(null_nllh-alt_nllh)
        return (para,current_value,test_value,test_statistic,test_statistic>critical_value)
    
def validation_interval(likelihood_fun, critical_value, fitting_parameters, no_fitting, order_parameters, *parameters, steps =100000, factor=3):
    '''Generate confidence intervals using the profile likelihood methode'''
    null_nllh = -likelihood_fun(*parameters)
    all_parameters = dict(zip(order_parameters,parameters))
    results = list()
    for para in fitting_parameters:
        idx  = order_parameters.index(para)
        current_value = parameters[idx]
        test_parameters = np.copy(parameters)
        test_parameters = np.delete(test_parameters, idx)
        do_parameters = np.copy(fitting_parameters)
        do_parameters = np.delete(do_parameters, np.where(do_parameters==para)[0])
        initial_parameters = [all_parameters[para2] for para2 in do_parameters]
        starting_points = np.logspace(np.log10(current_value)-factor, np.log10(current_value)+factor,steps)

        constant_args = (current_value, critical_value, factor, para, fitting_parameters, no_fitting, likelihood_fun,do_parameters,order_parameters,initial_parameters,null_nllh)
        partial_parallelize = functools.partial(parallelize_validation, *constant_args)    
        with multiprocessing.Pool(np.minimum(len(starting_points),MAX_CPUS)) as pool:
            results += list(tqdm(pool.imap(partial_parallelize, starting_points), total=len(starting_points), file=sys.stdout))
    return pd.DataFrame(results, columns=["Parameter","Mean","Test Value","Test Statistic","Rejection"])

def validation_interval_not_paralell(likelihood_fun, critical_value, fitting_parameters, no_fitting, order_parameters, *parameters, steps =1000, factor=2):
    '''Generate confidence intervals using the profile likelihood methode'''
    null_nllh = -likelihood_fun(*parameters)
    all_parameters = dict(zip(order_parameters,parameters))
    results = list()
    for para in fitting_parameters:
        idx  = order_parameters.index(para)
        current_value = parameters[idx]
        test_parameters = np.copy(parameters)
        test_parameters = np.delete(test_parameters, idx)
        do_parameters = np.copy(fitting_parameters)
        do_parameters = np.delete(do_parameters, np.where(do_parameters==para)[0])
        initial_parameters = [all_parameters[para2] for para2 in do_parameters]
        for test_value in np.logspace(np.log10(current_value)-factor, np.log10(current_value)+factor,steps):
            no_fitting_dict ={para: test_value}
            no_fitting_dict.update(no_fitting)
            optimize_fun = lambda paras: likelihood_fun(*get_parameters(paras, do_parameters, no_fitting_dict,order_parameters))
            if len(fitting_parameters) != 1:
                initial_parameters = optimize.minimize(optimize_fun, x0=initial_parameters,bounds=[(para2*1/(factor+1),para2*(factor+1))for para2 in initial_parameters]).x
            new_test_parameters = get_parameters(initial_parameters, do_parameters, no_fitting_dict,order_parameters)
            alt_nllh = -likelihood_fun(*new_test_parameters)
            test_statistic = 2*(null_nllh-alt_nllh)
            results.append((para,current_value,test_value,test_statistic,test_statistic>critical_value))
    return pd.DataFrame(results, columns=["Parameter","Mean","Test Value","Test Statistic","Rejection"])

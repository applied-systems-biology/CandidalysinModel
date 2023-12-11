# Copyright by Paul Rudolph
# Research Group Applied Systems Biology - Head: Prof. Dr. Marc Thilo Figge
# https://www.leibniz-hki.de/en/applied-systems-biology.html
# HKI-Center for Systems Biology of Infection
# Leibniz Institute for Natural Product Research and Infection Biology - Hans Knöll Insitute (HKI)
# Adolf-Reichwein-Straße 23, 07745 Jena, Germany
# This code is licensed under BSD 2-Clause
# See the LICENSE file provided with this code for the full license.

import numpy as np
import pandas as pd

from model import time_scaling,adjust_CaL,system_size
from fitting import simulate


# For each model:
# Objective function:  Defines the objective function during fitting procedure by simulating the experiment and computing the error between real data and simulation
# Generate data: Create synthetic data for testing purpose and validation if algorith works
# Prepare data: Process the data into a form in which it can easily be iterated over in the objective function


### Model with only LDH
def objective_LDH(funcptr, conditions,distance_fun, k_l, beta):
    error = 0.0
    denominator = len(conditions)
    for data,times in conditions:
        initials = np.ones(1)*beta
        time_points = np.linspace(0,np.max(times),np.max(times)+1)
        simulated_data = simulate(np.array([k_l,beta]),initials,time_points,funcptr)[:,times]
        error += np.sum(distance_fun(data,simulated_data))
    return error


def generate_data_LDH(funcptr, k_l, beta):
    times = np.array([1,2,3,10])*time_scaling
    time_points = np.linspace(0,np.max(times), np.max(times)+1)
    initials=np.ones(1)*beta
    data = simulate(np.array([k_l,beta]),initials,time_points,funcptr)[:,times]
    return [(data,times)]

def prepare_data_LDH(df, agg="sample"):
    if agg != "sample":
        df = df.groupby("Time").agg(agg).reset_index()
        return [(df.Readout.to_numpy(dtype="float64"),df.Time.to_numpy(dtype="int"))]
    return [(df.Readout.to_numpy(dtype="float64"),df.Time.to_numpy(dtype="int"))]

### Model with CaL and LDH

def objective_CAL(funcptr, conditions, distance_fun,distance_max_fun, k_a, k_c, k_l, k_d, alpha, beta): 
    error = 0.0
    parameters = np.array([k_a, k_c, k_l, k_d, alpha, beta])
    for CaL_initial, data_EC, data_LDH,times in conditions:
        initials = np.array([CaL_initial,0.0,1.0,0.0,0.0])
        time_points = np.linspace(0,np.max(times),np.max(times)+1)
        simulated_data = simulate(parameters, initials, time_points, funcptr)[:,times]
        error += np.sum(distance_max_fun(data_EC,simulated_data[3,:]))
        error += np.sum(distance_fun(data_LDH/beta,simulated_data[4,:]/beta))
    return error


def generate_data_CAL(funcptr, k_a, k_c, k_l, k_d, alpha, beta):
    parameters = np.array([k_a, k_c, k_l, k_d, alpha, beta])
    data_sim = list()
    for CaL in [0,1,10,32,70]:
        CaL_initial = CaL/adjust_CaL*system_size*1e-3
        times = np.array([1,3,24,48,72])*time_scaling
        initials = np.array([CaL_initial,0.0,1.0,0.0,0.0])
        time_points = np.linspace(0,np.max(times),np.max(times)+1)
        readout = simulate(parameters,initials,time_points,funcptr)[:,times]
        data_sim.append((CaL_initial,readout[3,:],readout[4,:],times))                         
    return data_sim

def prepare_data_CaL(df, beta, agg="mean"):
    data_sim = list()
    df_prep = df
    if agg != "sample":
        df_prep = df.groupby(["CaL","Nb","Time"]).agg(agg).reset_index()
    for key, row in df_prep.groupby(["CaL","Nb"]):
        data_sim.append((key[0], row.Readout.to_numpy(dtype="float64")/beta, row.Readout.to_numpy(dtype="float64"), row.Time.to_numpy(dtype="int")))
    return data_sim


### Model with Nb, CaL and LDH


def generate_data_Nb(funcptr, k_a, k_c, k_n,k_l,k_b, k_d, A, alpha, beta):
    parameters = np.array([k_a, k_c, k_n,k_l,k_b, k_d, A, alpha, beta])
    data = dict()
    data["sim"] = list()
    for Nb in [0,1,2,4,8,16]:
        for CaL in [0,1,10,32,70]:
            CaL_initial = CaL/adjust_CaL*system_size*1e-3
            Nb_initial = Nb*system_size*1e-3
            times = np.array([1,3,24,48,72])*time_scaling
            initials = np.array([CaL_initial,Nb_initial,0.0,1.0,0.0,0.0])
            time_points = np.linspace(0,np.max(times),np.max(times)+1)
            readout = simulate(parameters,initials,time_points,funcptr)[:,times]
            data["sim"].append((CaL_initial,Nb_initial,readout[4,:],readout[5,:],times))
            
    data["pre_co"] = list()
    for time_delay in [3,24,48,72]*time_scaling:
        for Nb in [0,1,2,4,8,16]:
            for CaL in [0,1,10,32,70]:
                CaL_initial = CaL/adjust_CaL*system_size*1e-3
                Nb_initial = Nb*system_size*1e-3
                times_pre = np.array([time_delay])
                initials = np.array([CaL_initial,Nb_initial,0.0,0.0,0.0,0.0])
                time_points = np.linspace(0,np.max(times_pre),np.max(times_pre)+1)
                initials = simulate(parameters,initials,time_points,funcptr)[:,times_pre]
                initials[3] = 1.0
                times = np.array([1,3,24,48,72])*time_scaling
                time_points = np.linspace(0,np.max(times),np.max(times)+1)
                readout = simulate(parameters,initials,time_points,funcptr)[:,times]
                data["pre_co"].append((time_delay,CaL_initial,Nb_initial,readout[4,:],readout[5,:],times))  
    data["pre"] = list()
    for time_delay in [3,24,48,72]*time_scaling:
        for Nb in [0,1,2,4,8,16]:
            for CaL in [0,1,10,32,70]:
                CaL_initial = CaL/adjust_CaL*system_size*1e-3
                Nb_initial = Nb*system_size*1e-3
                times = np.array([time_delay])
                initials = np.array([0.0,Nb_initial,0.0,1.0,0.0,0.0])
                time_points = np.linspace(0,np.max(times),np.max(times)+1)
                initials = simulate(parameters,initials,time_points,funcptr)[:,times]
                initials[0] = CaL_initial
                times = np.array([1,3,24,48,72])*time_scaling
                time_points = np.linspace(0,np.max(times),np.max(times)+1)
                readout = simulate(parameters,initials,time_points,funcptr)[:,times]
                data["pre"].append((time_delay,CaL_initial,Nb_initial,readout[4,:],readout[5,:],times))  
    data["post"] = list()
    for time_delay in [3,24,48,72]*time_scaling:
        for Nb in [0,1,2,4,8,16]:
            for CaL in [0,1,10,32,70]:
                CaL_initial = CaL/adjust_CaL*system_size*1e-3
                Nb_initial = Nb*system_size*1e-3
                times_pre = np.array([time_delay])
                initials = np.array([CaL_initial,0,0.0,1.0,0.0,0.0])
                time_points = np.linspace(0,np.max(times_pre),np.max(times_pre)+1)
                initials = simulate(parameters,initials,time_points,funcptr)[:,times_pre]
                initials[1] = Nb_initial
                times = np.array([1,3,24,48,72])*time_scaling
                time_points = np.linspace(0,np.max(times),np.max(times)+1)
                readout = simulate(parameters,initials,time_points,funcptr)[:,times]
                data["post"].append((time_delay,CaL_initial,Nb_initial,readout[4,:],readout[5,:],times))  
    return data

def objective_Nb(funcptr, conditions, distance_fun,distance_max_fun, k_a, k_c, k_n,k_l,k_b, k_d, A, alpha, beta): 
    error = 0.0
    parameters = np.array([k_a, k_c, k_n,k_l,k_b, k_d, A, alpha, beta])
    if "sim" in conditions:
        for CaL_initial, Nb_initial, data_EC, data_LDH,times in conditions["sim"]:
            initials = np.array([CaL_initial,Nb_initial,0.0,1.0,0.0,0.0])
            time_points = np.linspace(0,np.max(times),np.max(times)+1)
            simulated_data = simulate(parameters, initials, time_points, funcptr)[:,times]
            error += np.sum(distance_max_fun(data_EC,simulated_data[4,:]))
            error += np.sum(distance_fun(data_LDH/beta,simulated_data[5,:]/beta))
    if "pre2" in conditions:
        for time_delay, CaL_initial, Nb_initial, data_EC, data_LDH,times in conditions["pre"]:
            times_pre = np.array([time_delay])
            initials = np.array([0.0,Nb_initial,0.0,1.0,0.0,0.0])
            time_points = np.linspace(0,np.max(times_pre),np.max(times_pre)+1)
            initials = simulate(parameters,initials,time_points,funcptr)[:,times_pre]
            initials[0] = CaL_initial
            time_points = np.linspace(0,np.max(times),np.max(times)+1)
            simulated_data = simulate(parameters,initials,time_points,funcptr)[:,times]
            error += np.sum(distance_max_fun(data_EC,simulated_data[4,:]))
            error += np.sum(distance_fun(data_LDH/beta,simulated_data[5,:]/beta)) 
    if "pre_co" in conditions:
        for time_delay, CaL_initial, Nb_initial, data_EC, data_LDH,times in conditions["pre_co"]:
            times_pre = np.array([time_delay])
            initials = np.array([CaL_initial,Nb_initial,0.0,0.0,0.0,0.0])
            time_points = np.linspace(0,np.max(times_pre),np.max(times_pre)+1)
            initials = simulate(parameters,initials,time_points,funcptr)[:,times_pre]
            initials[3] = 1.0
            time_points = np.linspace(0,np.max(times),np.max(times)+1)
            simulated_data = simulate(parameters,initials,time_points,funcptr)[:,times]
            error += np.sum(distance_max_fun(data_EC,simulated_data[4,:]))
            error += np.sum(distance_fun(data_LDH/beta,simulated_data[5,:]/beta))
    if "post" in conditions:
        for time_delay, CaL_initial, Nb_initial, data_EC, data_LDH,times in conditions["post"]:
            times_post = np.array([time_delay])
            initials = np.array([CaL_initial,0.0,0.0,1.0,0.0,0.0])
            time_points = np.linspace(0,np.max(times_post),np.max(times_post)+1)
            initials = simulate(parameters,initials,time_points,funcptr)[:,times_post]
            initials[1] = Nb_initial
            time_points = np.linspace(0,np.max(times),np.max(times)+1)
            simulated_data = simulate(parameters,initials,time_points,funcptr)[:,times]
            error += np.sum(distance_max_fun(data_EC,simulated_data[4,:]))
            error += np.sum(distance_fun(data_LDH/beta,simulated_data[5,:]/beta))
    return error


def prepare_data_Nb(df_data_sim,df_data_pre,df_data_pre_co,df_data_post, beta, agg="mean"):
    data = dict()
    if agg != "sample":
        if len(df_data_sim) > 0:
            data["sim"] = list()
            for key, row in df_data_sim.groupby(["CaL","Nb","Time"]).agg(agg).reset_index().groupby(["CaL","Nb"]):
                data["sim"].append((key[0],key[1], row.Readout.to_numpy(dtype="float64")/beta, row.Readout.to_numpy(dtype="float64"), row.Time.to_numpy(dtype="int")))
        if len(df_data_pre) > 0:
            data["pre"] = list()
            for key, row in df_data_pre.groupby(["CaL","Nb","Time","Condition"]).agg(agg).reset_index().groupby(["CaL","Nb","Condition"]):
                data["pre"].append((np.array([int(key[2])]),key[0],key[1], row.Readout.to_numpy(dtype="float64")/beta, row.Readout.to_numpy(dtype="float64"), row.Time.to_numpy(dtype="int")))
        if len(df_data_pre_co) > 0:
            data["pre_co"] = list()
            for key, row in df_data_pre_co.groupby(["CaL","Nb","Time","Condition"]).agg(agg).reset_index().groupby(["CaL","Nb","Condition"]):
                data["pre_co"].append((np.array([int(key[2])]),key[0],key[1], row.Readout.to_numpy(dtype="float64")/beta, row.Readout.to_numpy(dtype="float64"), row.Time.to_numpy(dtype="int")))
        if len(df_data_post) > 0:
            data["post"] = list()
            for key, row in df_data_post.groupby(["CaL","Nb","Time","Condition"]).agg(agg).reset_index().groupby(["CaL","Nb","Condition"]):
                data["post"].append((np.array([int(key[2])]),key[0],key[1], row.Readout.to_numpy(dtype="float64")/beta, row.Readout.to_numpy(dtype="float64"), row.Time.to_numpy(dtype="int")))    
    else:
        if len(df_data_sim) > 0:
            data["sim"] = list()
            for key, row in df_data_sim.reset_index().groupby(["CaL","Nb"]):
                data["sim"].append((key[0],key[1], row.Readout.to_numpy(dtype="float64")/beta, row.Readout.to_numpy(dtype="float64"), row.Time.to_numpy(dtype="int")))
        if len(df_data_pre) > 0:
            data["pre"] = list()
            for key, row in df_data_pre.reset_index().groupby(["CaL","Nb","Condition"]):
                data["pre"].append((np.array([int(key[2])]),key[0],key[1], row.Readout.to_numpy(dtype="float64")/beta, row.Readout.to_numpy(dtype="float64"), row.Time.to_numpy(dtype="int")))
        if len(df_data_post) > 0:
            data["post"] = list()
            for key, row in df_data_post.reset_index().groupby(["CaL","Nb","Condition"]):
                data["post"].append((np.array([int(key[2])]),key[0],key[1], row.Readout.to_numpy(dtype="float64")/beta, row.Readout.to_numpy(dtype="float64"), row.Time.to_numpy(dtype="int")))
        if len(df_data_pre_co) > 0:
            data["pre_co"] = list()
            for key, row in df_data_pre_co.reset_index().groupby(["CaL","Nb","Condition"]):
                data["pre"].append((np.array([int(key[2])]),key[0],key[1], row.Readout.to_numpy(dtype="float64")/beta, row.Readout.to_numpy(dtype="float64"), row.Time.to_numpy(dtype="int")))
    return data


def prepare_data_Candida(df_data_candida_sim,df_data_candida_pre,df_data_candida_post, beta, agg="mean"):
    data = dict()
    data["sim"] = list()
    max_dmg = float(df_data_candida_sim.query("Nb == 0").pipe(lambda df_:df_.query(f"Time == {df_.Time.max()}"))["Readout"].mean())/beta
    df_prep = df_data_candida_sim#.query("Nb == 0")
    if agg != "sample":
        df_prep = df_prep.groupby(["Nb","Time"]).agg(agg).reset_index()
    for key, row in df_prep.groupby(["Nb"]):
        data["sim"].append((max_dmg,key[0],row.Readout.to_numpy(dtype="float64"), row.Readout.to_numpy(dtype="float64"), row.Time.to_numpy(dtype="int")))
        
    data["pre"] = list()
    df_prep = df_data_candida_pre#.query("Nb == 0")
    if agg != "sample":
        df_prep = df_prep.groupby(["Nb","Time","Condition"]).agg(agg).reset_index()
    for key, row in df_prep.groupby(["Nb","Condition"]):
        data["pre"].append((max_dmg,key[0],row.Readout.to_numpy(dtype="float64"), row.Readout.to_numpy(dtype="float64"), row.Time.to_numpy(dtype="int"),key[1]))
        
    data["post"] = list()#
    df_prep = df_data_candida_post#.query("Nb == 0")
    if agg != "sample":
        df_prep = df_prep.groupby(["Nb","Time","Condition"]).agg(agg).reset_index()
    for key, row in df_prep.groupby(["Nb","Condition"]):
        data["post"].append((max_dmg,key[0],row.Readout.to_numpy(dtype="float64"), row.Readout.to_numpy(dtype="float64"), row.Time.to_numpy(dtype="int"),key[1]))
    return data


def generate_data_Candida(funcptr, k_a, k_c,k_n, k_l, k_b,  k_d, A, alpha, beta,k_s):
    parameters = np.array([k_a, k_c, k_n, k_l, k_b, k_d, A, alpha, beta,k_s])
    data = dict()
    data["sim"] = list()
    Y0 = 2.0e4 # Intial concentration of yeast C. Albicans given from experimental setup
    for Nb in [0,2,4,8]:
        Nb_initial = Nb*system_size*1e-3
        times = np.array([1,3,24,48,72])*time_scaling
        initials = np.array([0.0,Nb_initial,0.0,0.6,0.0,0.0,Y0,0.0,0.0])
        time_points = np.linspace(0,np.max(times),np.max(times)+1)
        readout = simulate(parameters,initials,time_points,funcptr)[:,times]
        data["sim"] .append((0.6,Nb_initial,readout[4,:],readout[5,:],times))
    data["pre"] = list()
    for Nb in [0,2,4,8]:
        condition = 1*time_scaling
        Nb_initial = Nb*system_size*1e-3
        pre_times = np.array([condition])*time_scaling
        initials = np.array([0.0,Nb_initial,0.0,0.0,0.0,0.0,Y0,0.0,0.0])
        time_points = np.linspace(0,np.max(pre_times),np.max(pre_times)+1)
        initials_pre = simulate(parameters,initials,time_points,funcptr)[:,-1]
        times = np.array([1,3,24,48,72])*time_scaling
        initials_pre[3] = 0.6
        time_points = np.linspace(0,np.max(times),np.max(times)+1)
        readout = simulate(parameters,initials_pre,time_points,funcptr)[:,times]
        data["pre"] .append((0.6,Nb_initial,readout[4,:],readout[5,:],times,condition))    
    data["post"] = list()
    for Nb in [0,2,4,8]:
        condition = 3*time_scaling
        post_times = np.array([condition])*time_scaling
        initials = np.array([0.0,0.0,0.0,0.6,0.0,0.0,Y0,0.0,0.0])
        time_points = np.linspace(0,np.max(post_times),np.max(post_times)+1)
        initials_post = simulate(parameters,initials,time_points,funcptr)[:,-1]        
        
        times = np.array([1,3,24,48,72])*time_scaling
        initials_post[1] = Nb_initial
        time_points = np.linspace(0,np.max(times),np.max(times)+1)
        readout = simulate(parameters,initials_post,time_points,funcptr)[:,times]
        data["post"].append((0.6,Nb_initial,readout[4,:],readout[5,:],times,condition))   
    return data

def objective_Candida(funcptr, conditions, distance_fun,distance_max_fun, k_a, k_c, k_n, k_l, k_b, k_d, A, alpha, beta,k_s): 
    error = 0.0
    Y0 = 2.0e4 # Intial concentration of yeast C. Albicans given from experimental setup
    parameters = np.array([k_a, k_c,k_n, k_l, k_b, k_d, A, alpha, beta,k_s])
    if "sim" in conditions:
        for E_infected, Nb_initial, data_EC, data_LDH,times in conditions["sim"]:
            initials = np.array([0.0,Nb_initial,0.0,E_infected,0.0,0.0,Y0,0.0,0.0])
            time_points = np.linspace(0,np.max(times),np.max(times)+1)
            bu = simulate(parameters, initials, time_points, funcptr)
            simulated_data  = bu[:,times]
            error += np.sum(distance_max_fun(data_EC,simulated_data[4,:]))
            error += np.sum(distance_fun(data_LDH/beta,simulated_data[5,:]/beta))
    if "pre" in conditions:
        for E_infected, Nb_initial, data_EC, data_LDH,times,condition in conditions["pre"]:
            pre_times = np.array([condition])
            initials = np.array([0.0,Nb_initial,0.0,0.0,0.0,0.0,Y0,0.0,0.0])
            time_points = np.linspace(0,np.max(pre_times),np.max(pre_times)+1)
            initials_pre = simulate(parameters,initials,time_points,funcptr)[:,-1]
            initials_pre[3] = E_infected
            time_points = np.linspace(0,np.max(times),np.max(times)+1)
            readout = simulate(parameters,initials_pre,time_points,funcptr)[:,times]
            error += np.sum(distance_max_fun(data_EC,readout[4,:]))
            error += np.sum(distance_fun(data_LDH/beta,readout[5,:]/beta))
    if "post" in conditions:
        for E_infected, Nb_initial, data_EC, data_LDH,times,condition in conditions["post"]:
            post_times = np.array([condition])
            initials = np.array([0.0,0.0,0.0,E_infected,0.0,0.0,Y0,0.0,0.0])
            time_points = np.linspace(0,np.max(post_times),np.max(post_times)+1)
            initials_post = simulate(parameters,initials,time_points,funcptr)[:,-1]        
            initials_post[1] = Nb_initial
            time_points = np.linspace(0,np.max(times),np.max(times)+1)
            simulated_data = simulate(parameters,initials_post,time_points,funcptr)[:,times]
            error += np.sum(distance_max_fun(data_EC,simulated_data[4,:]))
            error += np.sum(distance_fun(data_LDH/beta,simulated_data[5,:]/beta))
    return error

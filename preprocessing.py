# Copyright by Paul Rudolph
# Research Group Applied Systems Biology - Head: Prof. Dr. Marc Thilo Figge
# https://www.leibniz-hki.de/en/applied-systems-biology.html
# HKI-Center for Systems Biology of Infection
# Leibniz Institute for Natural Product Research and Infection Biology - Hans Knöll Insitute (HKI)
# Adolf-Reichwein-Straße 23, 07745 Jena, Germany
# This code is licensed under BSD 2-Clause
# See the LICENSE file provided with this code for the full license.

import pandas as pd
import numpy as np

def read_data(file_name):
    '''This functions converts the data into panda dataframes'''
    df_ex5 = (pd.read_excel(file_name,sheet_name=2,header=None)
                .iloc[-3:,:]
                .rename(columns = dict(zip(np.arange(5),np.array([1,3,24,48,72]))))
                .melt(var_name="Time",value_name="LDH")
                .assign(LDH = lambda df_:df_.LDH)
                )

    df_ex1 =(pd.read_excel(file_name,sheet_name=0,header=None)
                .pipe(lambda df_:pd.concat((df_.iloc[:,:6].T, df_.iloc[:,6:].T)).T)
                .dropna(axis=1,how='all')
                .dropna(axis=0,how='all')
                .iloc[3:,:]
                .reset_index(drop=True)
                .pipe(lambda df_:pd.concat((
                                pd.concat((df_.iloc[2:5,:3].set_axis(["Ab","CaL","Ab+CaL"],axis=1).assign(Interval= 24),
                                df_.iloc[2:5,3:6].set_axis(["Ab","CaL","Ab+CaL"],axis=1).assign(Interval= 48),
                                df_.iloc[2:5,6:9].set_axis(["Ab","CaL","Ab+CaL"],axis=1).assign(Interval= 72))).assign(QVQ= 0),
                                pd.concat((df_.iloc[6:9,:3].set_axis(["Ab","CaL","Ab+CaL"],axis=1).assign(Interval= 24),
                                df_.iloc[6:9,3:6].set_axis(["Ab","CaL","Ab+CaL"],axis=1).assign(Interval= 48),
                                df_.iloc[6:9,6:9].set_axis(["Ab","CaL","Ab+CaL"],axis=1).assign(Interval= 72))).assign(QVQ= 1),
                                pd.concat((df_.iloc[10:13,:3].set_axis(["Ab","CaL","Ab+CaL"],axis=1).assign(Interval= 24),
                                df_.iloc[10:13,3:6].set_axis(["Ab","CaL","Ab+CaL"],axis=1).assign(Interval= 48),
                                df_.iloc[10:13,6:9].set_axis(["Ab","CaL","Ab+CaL"],axis=1).assign(Interval= 72))).assign(QVQ= 2),
                                pd.concat((df_.iloc[14:17,:3].set_axis(["Ab","CaL","Ab+CaL"],axis=1).assign(Interval= 24),
                                df_.iloc[14:17,3:6].set_axis(["Ab","CaL","Ab+CaL"],axis=1).assign(Interval= 48),
                                df_.iloc[14:17,6:9].set_axis(["Ab","CaL","Ab+CaL"],axis=1).assign(Interval= 72))).assign(QVQ= 4)))
                                )
                .melt(id_vars=["Interval","QVQ"],var_name="Condition",value_name="LDH")
                .dropna()
                .rename(columns={"QVQ":"Ab"})
                .assign(CaL = lambda df_: pd.Series(66.67, index =df_.index).where(df_.Condition != "Ab",0.0),
                        Nb = lambda df_:df_.Ab.where(df_.Condition != "CaL",0.0),
                        Time = 24,
                        Condition = lambda df_:df_.Interval,
                        LDH = lambda df_:df_.LDH)
                .drop(columns=["Interval"])
                )

    df_ex3 =(pd.read_excel(file_name,sheet_name=1,header=None)
                .pipe(lambda df_:
                pd.concat((
                df_.iloc[4:7,:].assign(Time = 1),
                df_.iloc[8:11,:].assign(Time = 3),
                df_.iloc[12:15,:].assign(Time = 24),
                df_.iloc[16:19,:].assign(Time = 48),
                df_.iloc[20:23,:].assign(Time = 72),))
            )
            .rename(columns = dict(zip(np.arange(5),["Candida",1,10,35,70])))
            .melt(id_vars=["Time"],var_name="CaL",value_name="LDH")
            .query("CaL != 'Candida'")
            .assign(CaL = lambda df_:df_.CaL,
                    Nb = 0.0,
                    LDH = lambda df_:df_.LDH)
            )
    df_ex3_candida =(pd.read_excel(file_name,sheet_name=1,header=None)
                .pipe(lambda df_:
                pd.concat((
                df_.iloc[4:7,:].assign(Time = 1),
                df_.iloc[8:11,:].assign(Time = 3),
                df_.iloc[12:15,:].assign(Time = 24),
                df_.iloc[16:19,:].assign(Time = 48),
                df_.iloc[20:23,:].assign(Time = 72),))
            )
            .rename(columns = dict(zip(np.arange(5),["Candida",1,10,35,70])))
            .melt(id_vars=["Time"],var_name="CaL",value_name="LDH")
            .query("CaL == 'Candida'")
            .assign(Nb = 0.0,
                    LDH = lambda df_:df_.LDH)
            .drop(columns= ["CaL"])
            )
    df_fig_cal = (pd.read_excel(file_name,sheet_name=4,header=None)
                    .pipe(lambda df_:pd.concat((
                    df_.iloc[6:10,:].set_axis(["Ab","Ab+CaL","CaL"],axis=1).assign(QVQ= 4,Condition= "CaL_Pre"),
                    df_.iloc[11:15,:].set_axis(["Ab","Ab+CaL","CaL"],axis=1).assign(QVQ= 8,Condition= "CaL_Pre"),
                    df_.iloc[16:20,:].set_axis(["Ab","Ab+CaL","CaL"],axis=1).assign(QVQ= 16,Condition= "CaL_Pre"),
                    df_.iloc[25:29,:].set_axis(["Ab","Ab+CaL","CaL"],axis=1).assign(QVQ= 4,Condition= "CaL_Sim"),
                    df_.iloc[30:34,:].set_axis(["Ab","Ab+CaL","CaL"],axis=1).assign(QVQ= 8,Condition= "CaL_Sim"),
                    df_.iloc[35:39,:].set_axis(["Ab","Ab+CaL","CaL"],axis=1).assign(QVQ= 16,Condition= "CaL_Sim"),
                    df_.iloc[43:47,:].set_axis(["Ab","Ab+CaL","CaL"],axis=1).assign(QVQ= 4,Condition= "CaL_Post"),
                    df_.iloc[48:52,:].set_axis(["Ab","Ab+CaL","CaL"],axis=1).assign(QVQ= 8,Condition= "CaL_Post"),
                    df_.iloc[53:57,:].set_axis(["Ab","Ab+CaL","CaL"],axis=1).assign(QVQ= 16,Condition= "CaL_Post"),))
                    )
                    .melt(id_vars=["QVQ","Condition"],var_name="Condition2",value_name="LDH")
                    .dropna()
                    .rename(columns={"QVQ":"Ab"})
                    .assign(CaL = lambda df_:pd.Series(70, index =df_.index).where(df_.Condition2 != "Ab",0.0),
                            Nb = lambda df_:df_.Ab.where(df_.Condition2 != "CaL",0.0),
                            LDH = lambda df_:df_.LDH,
                            Time = 24)
                    .drop(columns = ["Condition2"])
                    )
    df_fig_candida = (pd.read_excel(file_name,sheet_name=3,header=None)
                    .pipe(lambda df_:pd.concat((
                    df_.iloc[6:10,:].set_axis(["Ab","Ab+Ca","Ca"],axis=1).assign(QVQ= 4,Condition= "Ca_Pre"),
                    df_.iloc[11:15,:].set_axis(["Ab","Ab+Ca","Ca"],axis=1).assign(QVQ= 8,Condition= "Ca_Pre"),
                    df_.iloc[16:20,:].set_axis(["Ab","Ab+Ca","Ca"],axis=1).assign(QVQ= 16,Condition= "Ca_Pre"),
                    df_.iloc[25:29,:].set_axis(["Ab","Ab+Ca","Ca"],axis=1).assign(QVQ= 4,Condition= "Ca_Sim"),
                    df_.iloc[30:34,:].set_axis(["Ab","Ab+Ca","Ca"],axis=1).assign(QVQ= 8,Condition= "Ca_Sim"),
                    df_.iloc[35:39,:].set_axis(["Ab","Ab+Ca","Ca"],axis=1).assign(QVQ= 16,Condition= "Ca_Sim"),
                    df_.iloc[43:47,:].set_axis(["Ab","Ab+Ca","Ca"],axis=1).assign(QVQ= 4,Condition= "Ca_Post"),
                    df_.iloc[48:52,:].set_axis(["Ab","Ab+Ca","Ca"],axis=1).assign(QVQ= 8,Condition= "Ca_Post"),
                    df_.iloc[53:57,:].set_axis(["Ab","Ab+Ca","Ca"],axis=1).assign(QVQ= 16,Condition= "Ca_Post"),))
                    )
                    .melt(id_vars=["QVQ","Condition"],var_name="Condition2",value_name="LDH")
                    .dropna()
                    .rename(columns={"QVQ":"Ab"})
                    .assign(CaL = lambda df_:pd.Series(70, index =df_.index).where(df_.Condition2 != "Ab",0.0),
                            Nb = lambda df_:df_.Ab.where(df_.Condition2 != "Ca",0.0),
                            LDH = lambda df_:df_.LDH,
                            Time = 24)
                    .drop(columns = ["Condition2"])
                    )
    return df_ex5,df_ex3,df_ex3_candida,df_ex1,df_fig_cal,df_fig_candida


def preprocess_data(file_name,time_scaling, adjust_CaL=8.0, adjust_LDH=True, system_size = 1.0):
    '''Data is preprocessed into the right units and scaled to account for different LDH concentrations due to technical replicates'''
    df_ex5,df_ex3,df_ex3_candida,df_ex1,df_fig_cal,df_fig_candida = read_data(file_name)
    LDH_scale = 1.0
    # Reference scale is taken from the simultaneous experiments from 70 CaL
    # If we assume an aggregate size we have to divide by the it in order to get a linearization for the tranistion. This division only applies to the intial CaL concentration
    if adjust_LDH:
        cal_ref=70
        LDH_scale = 1/(df_ex3.query(f"Nb ==0.0 and Time == 24 and CaL=={cal_ref}").LDH.mean()/df_fig_cal.query("Nb ==0.0 and Condition == 'CaL_Sim'").LDH.mean())
    # LDH : µg/ml -> 1e-3*g/ml (LDH_scale)
    # CaL : µM = µmol/l -> 1e-3µmol/ml * system_size/adjust_CaL
    # Nb : µM = µmol/l -> 1e-3µmol/ml * system_size
    df_data_LDH = (df_ex5
                    .assign(Time = lambda df_:df_.Time*time_scaling,
                            Readout = lambda df_:df_.LDH*1e-3)
                    .drop(columns=["LDH"])
                    )
    df_data_sim = (
        pd.concat(((df_fig_cal
                    .assign(Time = lambda df_: df_.Time * time_scaling,
                            Nb = lambda df_: df_.Nb*system_size*1e-3,
                            CaL = lambda df_: df_.CaL*system_size/adjust_CaL*1e-3,
                            Readout = lambda df_:df_.LDH.where(df_.CaL != 0,0.0)*1e-3*1/LDH_scale)
                    .query("Condition == 'CaL_Sim'")
                    [["CaL","Nb","Time","Readout"]]),
                    (df_ex3
                    .assign(Nb = lambda df_: df_.Nb*system_size*1e-3,
                            CaL = lambda df_: df_.CaL*system_size/adjust_CaL*1e-3,
                            Time = lambda df_:df_.Time*time_scaling,
                            Readout = lambda df_:df_.LDH.where(df_.CaL != 0,0.0)*1e-3)
                    [["CaL","Nb","Time","Readout"]]
                    )
    ))
                   .sort_values("Time").reset_index(drop=True)
                    )

    df_data_pre = (df_ex1.assign(Time = lambda df_:df_.Time*time_scaling,
                                Condition = lambda df_: df_.Condition*time_scaling)
                   .groupby("Condition",group_keys=False).apply(lambda grp_:grp_.assign(Readout = lambda df_:df_.LDH.where(df_.CaL != 0,0.0)*66.66/70*1e-3*(df_ex3.query(f"Nb ==0.0 and Time == 24 and CaL=={cal_ref}").LDH.mean()/grp_.query("CaL == 66.67 and Nb == 0").LDH.mean()),
                                                                        Nb = lambda df_: df_.Nb*system_size*1e-3,
                                                                        CaL = lambda df_: df_.CaL*system_size/adjust_CaL*1e-3)).reset_index()
                    [["Nb","Condition","CaL","Time","Readout"]]
                    
                    .sort_values("Time").reset_index(drop=True)
                    )
    df_data_pre_co = (df_fig_cal
                    .query("Condition == 'CaL_Pre'")
                    .assign(Time = lambda df_: df_.Time * time_scaling,
                            Nb = lambda df_: df_.Nb*system_size*1e-3,
                            CaL = lambda df_: df_.CaL*system_size/adjust_CaL*1e-3,
                            Readout = lambda df_:df_.LDH.where(df_.CaL != 0,0.0)*1e-3*1/LDH_scale,
                            Condition = 1*time_scaling)
                    [["Nb","Condition","CaL","Time","Readout"]]
                      .sort_values("Time").reset_index(drop=True))

    df_data_post = (df_fig_cal
                    .query("Condition == 'CaL_Post'")
                    .assign(Time=21*time_scaling,
                            Nb = lambda df_: df_.Nb*system_size*1e-3,
                            CaL = lambda df_: df_.CaL*system_size/adjust_CaL*1e-3,
                            Readout = lambda df_:df_.LDH.where(df_.CaL != 0,0.0)*1e-3*1/LDH_scale,
                            Condition = 3*time_scaling)
                    [["Nb","Condition","CaL","Time","Readout"]]
                    .sort_values("Time").reset_index(drop=True)
                    )

    LDH_scale_candida = 1#/(df_ex3_candida.query(f"Nb ==0.0 and Time == 24").LDH.mean()/df_fig_candida.query("Nb ==0.0 and Condition == 'Ca_Sim'").LDH.mean())

    df_data_candida_sim = (
        pd.concat(((df_fig_candida
                    .assign(Time = lambda df_: df_.Time * time_scaling,
                            Nb = lambda df_: df_.Nb*system_size*1e-3,
                            Readout = lambda df_:df_.LDH.where(df_.CaL != 0,0.0)*1e-3*1/LDH_scale_candida)
                    .query("Condition == 'Ca_Sim'")
                    [["Nb","Time","Readout"]]),
                    (df_ex3_candida
                    .assign(Nb = lambda df_: df_.Nb*system_size*1e-3,
                            Time = lambda df_:df_.Time*time_scaling,
                            Readout = lambda df_:df_.LDH*1e-3)
                    [["Nb","Time","Readout"]]
                    )
    ))
                   .sort_values("Time").reset_index(drop=True)
                    )

    df_data_candida_pre = (df_fig_candida
                .query("Condition == 'Ca_Pre'")
                .assign(Time = lambda df_: df_.Time * time_scaling,
                        Nb = lambda df_: df_.Nb*system_size*1e-3,
                        Readout = lambda df_:df_.LDH.where(df_.CaL != 0,0.0)*1e-3*1/LDH_scale_candida,
                        Condition = 1*time_scaling)
                [["Nb","Condition","Time","Readout"]]
                  .sort_values("Time").reset_index(drop=True))

    df_data_candida_post = (df_fig_candida
                .query("Condition == 'Ca_Post'")
                .assign(Time=lambda df_:(df_.Time-3)*time_scaling,
                        Nb = lambda df_: df_.Nb*system_size*1e-3,
                        Readout = lambda df_:df_.LDH.where(df_.CaL != 0,0.0)*1e-3*1/LDH_scale_candida,
                        Condition = 3*time_scaling)
                [["Nb","Condition","Time","Readout"]]
                .sort_values("Time").reset_index(drop=True)
                )
    return df_data_LDH,df_data_sim,df_data_pre,df_data_pre_co,df_data_post,df_data_candida_sim,df_data_candida_pre,df_data_candida_post
# Copyr_ight by Paul Rudolph
# Research Group Applied Systems Biology - Head: Prof. Dr. Marc Thilo Figge
# https://www.leibniz-hki.de/en/applied-systems-biology.html
# HKI-Center for Systems Biology of Infection
# Leibniz Institute for Natural Product Research and Infection Biology - Hans Knöll Insitute (HKI)
# Adolf-Reichwein-Straße 23, 07745 Jena, Germany
# This code is licensed under BSD 2-Clause
# See the LICENSE file provided with this code for the full license.

from numbalsoda import lsoda_sig
from numba import cfunc
import numpy as np

time_scaling = 1
system_size = 1e-3
adjust_CaL = 8.0

@cfunc(lsoda_sig)
def candida_model(t, y, dy, p):   

    # SPECIES
    CAL = 0
    NB = 1
    CAL_A = 2
    E_CANDIDA = 3
    E_DEAD = 4
    LDH = 5
    CANDIDA = 6
    F_I = 8
    F_NI = 7
    # PARAMETERS
    k_a = 0
    k_c = 1
    k_n = 2
    k_l = 3
    k_b = 4
    k_d = 5
    A = 6
    alpha = 7
    beta = 8
    k_s = 9
    r_i = 0.0049*60
    r = 0.0159*60

    dy[CAL] = -p[k_a]*y[CAL] -p[k_c]*y[CAL]-p[k_b]/(p[A])*y[CAL]*y[NB] + p[k_s]*y[F_I]*y[E_CANDIDA]
    dy[NB] = -p[k_n]*y[NB]-p[k_b]*y[NB]*(y[CAL]+y[CAL_A]) 
    dy[CAL_A] = p[k_a]*y[CAL] - p[alpha]*p[k_d]*y[CAL_A]*y[E_CANDIDA] -p[k_b]*y[CAL_A]*y[NB]
    dy[E_CANDIDA] = -p[k_d]*y[CAL_A]*y[E_CANDIDA]  
    dy[E_DEAD] = p[k_d]*y[CAL_A]*y[E_CANDIDA]
    dy[LDH] = p[beta]*p[k_d]*y[CAL_A]*y[E_CANDIDA]-p[k_l]*y[LDH] 
    dy[CANDIDA] = -r *y[CANDIDA]
    dy[F_NI] = r *y[CANDIDA]-r_i* y[F_NI]
    dy[F_I] = r_i* y[F_NI]
    
@cfunc(lsoda_sig)
def cal_nb_model(t, y, dy, p):
    # SPECIES
    CAL = 0
    NB = 1
    CAL_A = 2
    E_ALIVE = 3
    E_DEAD = 4
    LDH = 5
    # PARAMETERS
    k_a = 0
    k_c = 1
    k_n = 2
    k_l = 3
    k_b = 4
    k_d = 5
    A = 6
    alpha = 7
    beta = 8

    # REACTIONS
    dy[CAL] = -p[k_a]*y[CAL] -p[k_c]*y[CAL] -p[k_b]/(p[A])*y[CAL]*y[NB]
    dy[NB] = -p[k_n]*y[NB]-p[k_b]*y[NB]*(y[CAL]+y[CAL_A])   
    dy[CAL_A] = p[k_a]*y[CAL] - p[alpha]*p[k_d]*y[CAL_A]*y[E_ALIVE] -p[k_b]*y[CAL_A]*y[NB]    
    dy[E_ALIVE] = -p[k_d]*y[CAL_A]*y[E_ALIVE]
    dy[E_DEAD] = p[k_d]*y[CAL_A]*y[E_ALIVE]
    dy[LDH] = p[beta]*p[k_d]*y[CAL_A]*y[E_ALIVE]-p[k_l]*y[LDH]  

@cfunc(lsoda_sig)
def cal_nb_with_binding_model(t, y, dy, p):
    # SPECIES
    CAL = 0
    NB = 1
    CAL_A = 2
    E_ALIVE = 3
    E_DEAD = 4
    LDH = 5
    CAL_NB = 6
    CALA_NB = 7
    # PARAMETERS
    k_a = 0
    k_c = 1
    k_n = 2
    k_l = 3
    k_b = 4
    k_d = 5
    A = 6
    alpha = 7
    beta = 8

    # REACTIONS
    dy[CAL] = -p[k_a]*y[CAL] -p[k_c]*y[CAL] -p[k_b]/(p[A])*y[CAL]*y[NB]
    dy[NB] = -p[k_n]*y[NB]-p[k_b]*y[NB]*(y[CAL]+y[CAL_A])   
    dy[CAL_A] = p[k_a]*y[CAL] - p[alpha]*p[k_d]*y[CAL_A]*y[E_ALIVE] -p[k_b]*y[CAL_A]*y[NB]    
    dy[E_ALIVE] = -p[k_d]*y[CAL_A]*y[E_ALIVE]
    dy[E_DEAD] = p[k_d]*y[CAL_A]*y[E_ALIVE]
    dy[LDH] = p[beta]*p[k_d]*y[CAL_A]*y[E_ALIVE]-p[k_l]*y[LDH]  
    dy[CAL_NB] = p[k_b]/(p[A])*y[CAL]*y[NB]
    dy[CALA_NB] =p[k_b]*y[CAL_A]*y[NB]

@cfunc(lsoda_sig)
def cal_model(t, y, dy, p):
    # SPECIES
    CAL = 0
    CAL_A = 1
    E_ALIVE = 2
    E_DEAD = 3
    LDH = 4
    # PARAMETERS
    k_a = 0
    k_c = 1
    k_l = 2
    k_d = 3
    alpha = 4
    beta = 5

    # REACTIONS
    dy[CAL] = -p[k_a]*y[CAL] -p[k_c]*y[CAL]
    dy[CAL_A] = p[k_a]*y[CAL] - p[alpha]*p[k_d]*y[CAL_A]*y[E_ALIVE]
    dy[E_ALIVE] = -p[k_d]*y[CAL_A]*y[E_ALIVE]
    dy[E_DEAD] = p[k_d]*y[CAL_A]*y[E_ALIVE]
    dy[LDH] = p[beta]*p[k_d]*y[CAL_A]*y[E_ALIVE]-p[k_l]*y[LDH]    
    
@cfunc(lsoda_sig)    
def ldh_model(t, y, dy, p):
    # SPECIES
    LDH = 0
    # PARAMETERS
    k_l = 0
    # REACTIONS
    dy[LDH] = -p[k_l]*y[LDH]
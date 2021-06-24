"""
CCC-MODEL-MODIFIER
Written by Feiyu Yang
y_f_y@sina.cn
LBV_Module Written by Xin Lyu
lvanddan@gmail.com
"""

from __future__ import division
from __future__ import print_function
import sys
sys.path.append(r'G:\ProgramData\Anaconda3\envs\spam\Lib\site-packages')

import cantera as ct
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import utils_lbv
import copy
import write_reactions
import soln2ck
import soln2cti
import os
import datetime
print('''
/========================================================================\\
|                                                                         |
|                             CCC-MODEL-OPTIMIZER                         |
|                                                                         |
\========================================================================/''')
##############################################################################
# inputs
##############################################################################
LOG_DIR                     = 'log'
# create checkpoint file
now = datetime.datetime.now()
filename = 'Session_{:%Y%m%dT%H%M%S}'.format(now)
LOG_DIR = os.path.join(LOG_DIR, filename)
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
    
INPUT_LBV                   = 'C4H6_test/input_LBVs.txt'
INPUT_UNCERTAINTIES         = 'C4H6_test/input_uncertainties.txt'
INPUT_MECH                  = 'C4H6_test/USC_mech/scheme.uscmech2-base.1atm.cti'
INDICATOR                   = 'P' 		# T, P, oh ...
MODE                        = 'derivative' 	# max, halfmax  
OUTPUT_REACTONS_FILE_NAME   = os.path.join(LOG_DIR, 'new_reactions')
OUTPUT_FACTORS_FILE_NAME    = os.path.join(LOG_DIR, 'factors')
OUTPUT_CK                   = True
OUTPUT_CTI                  = True
LBV_PLOT                    = True
EXP_PRED_PLOT               = True
MIN_LOSS                    = 0.01
MAX_GAUSS                   = 7

def END_TIME(T):
    if T > 1260:
        return 0.001
    elif T > 1190:
        return 0.003
    else:
        return 0.004
#def END_TIME(T):
    #if T > 1260:
        #return 0.000002
    #elif T > 1190:
        #return 0.000002
    #else:
        #return 0.000002
    
# output info
print('Inputs summary: ')
print("INPUT_LBV:                   %s"%(INPUT_LBV))
print("INPUT_UNCERTAINTIES:         %s"%(INPUT_UNCERTAINTIES))
print("INPUT_MECH:                  %s"%(INPUT_MECH))
print("INDICATOR:                   %s"%(INDICATOR))
print("MODE:                        %s"%(MODE))
print("OUTPUT_REACTONS_FILE_NAME:   %s"%(OUTPUT_REACTONS_FILE_NAME))
print("OUTPUT_FACTORS_FILE_NAME:    %s"%(OUTPUT_FACTORS_FILE_NAME))
print("OUTPUT_CK:                   %s"%(OUTPUT_CK))
print("OUTPUT_CTI:                  %s"%(OUTPUT_CTI))
print("LBV_PLOT:                    %s"%(LBV_PLOT))
print("EXP_PRED_PLOT:               %s"%(EXP_PRED_PLOT))
print("MIN_LOSS:                    %s"%(MIN_LOSS))
print("MAX_GAUSS:                   %s"%(MAX_GAUSS))
print('===========================================================================')
# suppress thermo warning
ct.suppress_thermo_warnings()
# read information
info, LBVs, PHIs = utils_lbv.parse_exprimental_conditions(INPUT_LBV)
uncertainties = utils_lbv.parse_uncertainties(INPUT_UNCERTAINTIES)
reactions = ct.Reaction.listFromFile(INPUT_MECH)
species = ct.Species.listFromFile(INPUT_MECH)
#%%
##############################################################################
# main loop
##############################################################################
# calc original predictions, loop through every gt exprimental point
original_taus = []
pre_set_orignal_file = os.path.abspath(os.path.join(LOG_DIR, '../Original_taus.txt'))
# calc original IDTs is the Original_taus.txt file is not given in log directory
time_cost = 0.
if not os.path.isfile(pre_set_orignal_file):
    print('Calculating orignal IDTs ...' )
    t_0 = time.time()
    for ind, condition in info.items():
        gas = ct.Solution(INPUT_MECH)
        gas.TPX = condition
        width=0.03 # m
        freeflame=ct.FreeFlame(gas,width=width)
        freeflame.set_refine_criteria(ratio=3, slope=0.02, curve=0.02)
        freeflame.solve(loglevel=1, auto=True)
        # calc lbv using TP or species        
        if not INDICATOR in ['T', 'P']:
            LBV=float(freeflame.u[0])
            original_taus.append(LBV)
            print('\nmixture-averaged flamespeed = {:7f} m/s\n'.format(freeflame.u[0]))
        else:
            LBV=float(freeflame.u[0])
            original_taus.append(LBV)
            print('\nmixture-averaged flamespeed = {:7f} m/s\n'.format(freeflame.u[0]))            
    t_1 = time.time() 
    time_cost = t_1 - t_0
# read original IDTs is the Original_taus.txt file is given in log directory
else:
    print('Reading original IDTs from %s' % pre_set_orignal_file)
    with open(pre_set_orignal_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line == '\n':
                continue
            else:
                LBV = line.split(',')[-1]
                LBV = float(LBV.strip())
                original_taus.append(LBV)
                
# initialize loss with original loss
original_loss = utils_lbv.l2_loss(LBVs, original_taus)
loss = original_loss

# write original IDTs to file
with open(os.path.join(LOG_DIR, 'Original_taus.txt'), 'w') as f:
    print('Original LBVs: ')
    for i, LBV in enumerate(original_taus):
        print('%d, %.3f m/s' % (i+1, LBV))
        f.write('%d, %f \n' % (i+1, LBV))
 
print('===========================================================================')
print('Original: %.5f, Time cost: %.2f min' % (loss, (float(time_cost)/60)))
print('===========================================================================')
"""
everything goes well upon here
"""

# modify mech, calc new predictions, compare with gt and save the best        
counter = 1

# place holders
best_reactions = None
best_factors = None
best_lbvs = None
#%%
# random loop
t0 = time.time()
while True:
    # generate a mech
    t_iteration_0 = time.time()
    reactions_var = ct.Reaction.listFromFile(INPUT_MECH)  
    
    # use Gaussian ditribution after 4 loops
    if counter <= 4:
        means = None
        print('Uniform search, ', end=' ')
    else:
        means = best_factors
        print('Gaussian search, ', end=' ')
        
    # make predictions with new mech
    reactions_new, factors = utils_lbv.generate_new_reactions(reactions_var, uncertainties, means)
    lbvs = []
    for ind, condition in info.items():
        gas = ct.Solution(thermo='IdealGas', kinetics='GasKinetics',species=species, reactions=reactions_new)
        gas.TPX = condition
        freeflame =ct.FreeFlame(gas,width=width)
        freeflame.set_refine_criteria(ratio=3, slope=0.07, curve=0.14)
        freeflame.solve(loglevel=1, auto=True)
        # calc lbv using TP or species        
        if not INDICATOR in ['T', 'P']:
            lbv=float(freeflame.u[0])
            lbvs.append(lbv)
            print('\nmixture-averaged flamespeed = {:7f} m/s\n'.format(freeflame.u[0]))
        else:
            lbv=float(freeflame.u[0])
            lbvs.append(lbv)
            print('\nmixture-averaged flamespeed = {:7f} m/s\n'.format(freeflame.u[0]))            
        
    # calc idt loss
    l = utils_lbv.l2_loss(LBVs, lbvs)

    # save the best
    if l < loss:
        loss = l
        best_reactions = reactions_new
        best_factors = factors
        best_lbvs = lbvs
        write_reactions.write_reactions(best_reactions, OUTPUT_REACTONS_FILE_NAME)
        write_reactions.write_factors(best_factors, OUTPUT_FACTORS_FILE_NAME)
        with open(os.path.join(LOG_DIR, 'Best_LBV_%.5f.txt' % loss), 'w') as f:
            for i, LBV in enumerate(best_lbvs):
                f.write('%d, %f \n' % (i+1, LBV))
                
        # use solution to save ck or cti file
        if OUTPUT_CK:
            gas = ct.Solution(thermo='IdealGas', kinetics='GasKinetics', species=species, reactions=best_reactions)            
            soln2ck.write(gas, LOG_DIR)
        if OUTPUT_CTI:
            gas = ct.Solution(thermo='IdealGas', kinetics='GasKinetics', species=species, reactions=best_reactions)            
            soln2cti.write(gas, LOG_DIR)
    
    t_iteration_1 = time.time()    
    num_space = 5 - len(str(counter))
    print('===========================================================================')
    print('Counter:%s%d, Loss: %.5f, Time cost: %.2f min' % (num_space*' ', counter, loss, float(t_iteration_1 - t_iteration_0)/60))
    print('===========================================================================')

    # end metrics
    if loss <= MIN_LOSS or counter >= MAX_GAUSS:
        if loss == original_loss:
            print('Did not find a better mech, please increase MAX_GAUSS and retry.')
            exit()
        else:
            break

    counter += 1
t1 = time.time()
print('Total time cost: %.2f min' % (float(t1 - t0)/60.))
print('===========================================================================')
#%%
##############################################################################
# plot results
##############################################################################
# plot pred-true figure, valid for abitary inputs
if EXP_PRED_PLOT:
    fig2 = plt.figure()
    ax = fig2.add_subplot(111)
    ax.loglog(np.array(LBVs), np.array(original_taus), 's', color='r')
    ax.loglog(np.array(LBVs), np.array(best_lbvs), 'o', color='g')
    ax.legend(['Exp-New','Exp-Old'], loc='lower right')

    # set identity limits
    low_lim = np.minimum(ax.get_xlim()[0], ax.get_ylim()[0])
    high_lim = np.maximum(ax.get_xlim()[1], ax.get_ylim()[1])
    lim = (low_lim, high_lim)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.plot(lim, lim, color='b')

    # add labels
    ax.set_ylabel('Experimental data', fontsize=16)
    ax.set_xlabel('Mech predictions', fontsize=16)
    
    plt.savefig(os.path.join(LOG_DIR, 'Fig2'))
    
# plot tau-T figure, only valid when the input data share all the parameters but temperature.
if LBV_PLOT:
    # get Temperatures
    Temperatures = [ condition[0] for ind, condition in info.items()]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.semilogy(np.array(PHIs), np.array(LBVs), 'o', color='b')
    ax.semilogy(np.array(PHIs), np.array(original_taus), color='r')
    ax.semilogy(np.array(PHIs), np.array(best_lbvs), color='g')
    ax.legend(['Experimental data','Original predctions','New predictions'], loc='lower right')

    ax.set_ylabel('Lamianr burning velocity (m/s)', fontsize=16)
    ax.set_xlabel('PHIs', fontsize=16)

    # Add a second axis on top to plot the temperature for better readability
    ax2 = ax.twiny()
    ticks = ax.get_xticks()
    ax2.set_xticks(ticks)
    ax2.set_xticklabels((1000/ticks).round(1))
    ax2.set_xlim(ax.get_xlim())

    
    plt.savefig(os.path.join(LOG_DIR, 'Fig1'))
    plt.show()
    
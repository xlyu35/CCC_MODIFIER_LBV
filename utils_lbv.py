
"""
CCC-MODEL-OPTIMIZER
Written by Feiyu Yang
y_f_y@sina.cn
"""

from __future__ import division
from __future__ import print_function
import sys
sys.path.append(r'/home/yang/anaconda3/lib/python3.6/site-packages')
import cantera as ct
import time
import numpy as np
import pandas as pd
import copy

def parse_uncertainties(file):
    with open(file, 'r') as f:
        inputs = f.readlines()
        
    uncertainties = {}
    for line in inputs:
        if line == '\n':
            continue
        ind, uncer = line.split(',')
        ind = int(ind.strip())
        uncer = float(uncer.strip())
        
        uncertainties[ind] = uncer
    return uncertainties

def parse_exprimental_conditions(file):
    '''
    input in the following format:
    1, 1200, 101325, 'CH4:1, O2:2, N2:7.52', 3.0E-3
    2, 1500, 101325, 'CH4:1, O2:2, N2:7.52', 2.8E-3
    returns:
        info: a dictionary of index and conditions
        lbvs: laminar burning velocities
        
        eg:
            info:
                {1: [1200.0, 101325.0, 'CH4:1, O2:2, N2:7.52'], 
                2: [1500.0, 101325.0, 'CH4:1, O2:2, N2:7.52']} 
            idts:
                [0.003, 0.0028]
    '''
    with open(file, 'r') as f:
        inputs = f.readlines()
    
    info = {}
    lbvs = []
    PHIs = []
    for line in inputs:
        if line == '\n':
            continue
        nums, species, lbv = line.split("'")

        nums= [n.strip() for n in nums.split(',')]
        ind = int(nums[0])
        T = float(nums[1])
        P = float(nums[2])
        PHI=float(nums[3])
        PHIs.append(PHI)

        lbv = float(lbv.split(',')[1].strip())
        lbvs.append(lbv)

        condition = [T, P, species]
        info[ind] = condition
        
    return info, lbvs, PHIs

#def random_factor(uncertainty):
    #x = np.random.uniform(1.0, uncertainty)
    #x = np.random.choice([1./x, x])
    #return x

def random_factor(uncertainty, mean=None):
    if mean == None:
        x = np.random.uniform(1.0, uncertainty)
        x = np.random.choice([1./x, x])
        return x
    else:
        x = np.random.normal(mean)
        if x < 1./uncertainty:
            x = 1./uncertainty
        if x > uncertainty:
            x = uncertainty
        return x
    
def generate_new_reactions(old_rxns, uncertainties, means):
    '''
    update uncertainties to reactios
    '''
    factors = {}
    for ind, uncert in uncertainties.items():

        # find the reaction
        new_rxns = copy.copy(old_rxns)
        r = new_rxns[ind - 1]
        
        # deal with different reaction types
        # elementary and three body reaction
        if means != None:
            mean = means[ind]
        else:
            mean = None
            
        if r.reaction_type in [1, 2]:
            # calc new rate
            factor = random_factor(uncert, mean)
            
            A_old = r.rate.pre_exponential_factor
            n = r.rate.temperature_exponent
            E = r.rate.activation_energy
            
            A_new = A_old * factor
            rate_new = ct.Arrhenius(A_new, n, E)
            
            # update reaction
            r.rate = rate_new
        
        # fall-off and chemical reaction
        elif r.reaction_type in [3, 4]:
            # calc new rate
            factor = random_factor(uncert, mean)
            # for low
            A_old_low = r.low_rate.pre_exponential_factor
            n_low = r.low_rate.temperature_exponent
            E_low = r.low_rate.activation_energy
            
            A_new_low = A_old_low * factor
            rate_new_low = ct.Arrhenius(A_new_low, n_low, E_low)
            
            # for high
            A_old_high = r.high_rate.pre_exponential_factor
            n_high = r.high_rate.temperature_exponent
            E_high = r.high_rate.activation_energy
            
            A_new_high = A_old_high * factor
            rate_new_high = ct.Arrhenius(A_new_high, n_high, E_high)
            
            # update reaction
            r.low_rate = rate_new_low            
            r.high_rate = rate_new_high    
            
        else:
            print('The raction type is {} !'.format(r.reaction_type))
            
        new_rxns[ind - 1] = r 
        factors[ind] = factor
    return new_rxns, factors

# show modified mech
#for i, r in enumerate(reactions_new):
    #if r.reaction_type in [1, 2]:
        #print('ind: ', i, ' ', r.reaction_type, ' ', reactions[i].rate,'---', r.rate, '\n')
    #elif r.reaction_type in [3, 4]:
        #print('ind: ', i, ' ', r.reaction_type, ' ', reactions[i].low_rate,'---', r.low_rate, '\n')
        #print('ind: ', i, ' ', r.reaction_type, ' ', reactions[i].high_rate,'---', r.high_rate, '\n')
    #else:
        #print('unknown reaction type: ', r.reaction_type)
        
def get_idt(df, species, mode='derivative'):
    """
    This function computes the ignition delay using different method given
    """
    assert mode in ['derivative', 'halfmax', 'max']
    if mode == 'derivative':
        s1 = list(df[species])
        index = list(df.index)
        
        s2 = s1[1:]
        s2.append(s1[-1])

        derivative_s = np.array(s2) - np.array(s1) 
        return index[derivative_s.argmax()]
    
    if mode == 'halfmax':
        s = list(df[species])
        index = list(df.index)
        half_max_s = max(s) / 2
        
        for i, x in enumerate(s):
            if x < half_max_s:
                continue
            else:
                return index[i]
            
    if mode == 'max':
        return df[species].argmax()
"""    
def calculate_idt_with_TP(reactorNetwork, gas, indicator, mode, endtime):
    '''
    calc ignition delay time for a T/P indicator
    '''
    assert indicator in ['T', 'P']
    
    # starte time
    t0 = time.time()
    
    # counter for saving
    counter = 1
    
    # calc idt
    run_time = 0.
    history = pd.DataFrame(columns=[indicator])
    print('Calculating IDT at T=%.1f K, P=%.1f atm, END_TIME=%.3f ms ' % (gas.T, (gas.P/101325), (1000*endtime)))
    while(run_time < endtime):
        run_time = reactorNetwork.step()
        
        if counter % 20 == 0:
            if indicator == 'T':
                history.loc[str(run_time)] = gas.T
            if indicator == 'P':
                history.loc[str(run_time)] = gas.P
        counter += 1
        
    # end time        
    t1 = time.time()
    delta_t = t1 - t0
    
    # get idt
    lbv = float(get_idt(history, indicator, mode))
    print('TAU=%.3f ms, ' % (1000*tau), end='')
    return tau, delta_t
    
def calculate_idt_with_species(reactorNetwork, indicator, indicator_index, mode, endtime):
    '''
    calc ignition delay time for a specific indicator
    '''
    # starte time
    t0 = time.time()
    
    # counter for saving
    counter = 1
    
    # calc idt
    run_time = 0.
    history = pd.DataFrame(columns=[indicator])
    print('Calculating IDT with end time %.3f s ' % ((1000*endtime)))
    while(run_time < endtime):
        run_time = reactorNetwork.step()
        if counter % 20 == 0:
            state = reactorNetwork.get_state()
            history.loc[str(run_time)] = (state[indicator_index])
        counter += 1
        
    # end time        
    t1 = time.time()
    delta_t = t1 - t0
    
    # get idt
    tau = float(get_idt(history, indicator, mode))
    print('TAU=%.3f ms, ' % (1000*tau), end='')
    return tau, delta_t
"""
def get_indicator_index(reactor, indicator='OH'):
    '''
    get the index of a specific indicator
    '''
    indicator_index = None
    for i in range(reactor.n_vars):
        if reactor.component_name(i) == indicator:
            indicator_index = i
    if indicator_index == None:
        print('Could not find indicator ', indicator)
    else:
        return indicator_index

def l2_loss(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    loss = np.mean(np.sqrt(np.square(np.log(y_true) - np.log(y_pred))))
    return loss

def sparse_uncertainties(file):
    with open(file, 'r') as f:
        inputs = f.readlines()
        
    uncertainties = {}
    for line in inputs:
        if line == '\n':
            continue
        ind, uncer = line.split(',')
        ind = int(ind.strip())
        uncer = float(uncer.strip())
        
        uncertainties[ind] = uncer
    return uncertainties

def sparse_exprimental_conditions(file):
    '''
    input in the following format:
    1, 1200, 101325, 'CH4:1, O2:2, N2:7.52', 3.0E-3
    2, 1500, 101325, 'CH4:1, O2:2, N2:7.52', 2.8E-3
    returns:
        info: a dictionary of index and conditions
        lbvs: ignition delay times
        
        eg:
            info:
                {1: [1200.0, 101325.0, 'CH4:1, O2:2, N2:7.52'], 
                2: [1500.0, 101325.0, 'CH4:1, O2:2, N2:7.52']} 
            lbvs:
                [0.003, 0.0028]
    '''
    with open(file, 'r') as f:
        inputs = f.readlines()
    
    info = {}
    lbvs = []
    for line in inputs:
        if line == '\n':
            continue
        nums, species, lbv = line.split("'")

        nums= [n.strip() for n in nums.split(',')]
        ind = int(nums[0])
        T = float(nums[1])
        P = float(nums[2])

        lbv = float(lbv.split(',')[1].strip())
        lbvs.append(lbv)

        condition = [T, P, species]
        info[ind] = condition
        
    return info, lbvs


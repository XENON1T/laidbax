import numpy as np

##
# Get arguments from command line (for batch queue integration)
##
import argparse
parser = argparse.ArgumentParser(description="Compute limits for sensitivity study")
parser.add_argument('--setting', default='er')
parser.add_argument('--change_arg', default='rate_multipliers')
parser.add_argument('--n_trials', default=int(4e3))
parser.add_argument('--sampling_multiplier', default=0.2)
parser.add_argument('--multipliers', default=[0.5, 0.75, 1, 1.25, 1.5], nargs='+')
args = parser.parse_args()
n_trials = int(args.n_trials)
multipliers = np.array(args.multipliers)
change_arg = args.change_arg
sampling_multiplier = args.sampling_multiplier
setting = args.setting
base_folder = change_arg

##
# Compute limits
##
import os
from copy import deepcopy
import pandas as pd
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt

import blueice as bi
from blueice.utils import save_pickle
from laidbax import base_model

data_dir = base_model.THIS_DIR + '/data/'
base_model.config['livetime_days'] = 365.25

def compute_limits(config=None,
                   true_config=None,
                   meta_config=None,
                   rate_multipliers=None,
                   true_rate_multipliers=None,
                   n_trials=n_trials, 
                   pdf_sampling_multiplier=sampling_multiplier,
                   tqdm_on=False):
    """
    If true_ is not explicitly passed, will be assumed equal to the model config
    """    
    if config is None:
        config = dict()
    if true_config is None:
        true_config = config
    if meta_config is None:
        meta_config = dict()
    if rate_multipliers is None:
        rate_multipliers = dict()
    if true_rate_multipliers is None:
        true_rate_multipliers = rate_multipliers
    for c in config, true_config:
        c['pdf_sampling_multiplier'] = pdf_sampling_multiplier
        
    limit_options = dict(bestfit_routine='scipy',
                     kind='upper',
                     minimize_kwargs=dict(method='Powell',))
        
    # Make likelihood function
    c = deepcopy(base_model.config)
    c.update(config)
    ll = bi.UnbinnedLogLikelihood(c)
    ll.add_rate_parameter('wimp')
    ll.add_rate_parameter('er')
    for sname in rate_multipliers.keys():
        ll.add_rate_parameter(sname)
    ll.prepare()
    
    # True model
    ctrue = deepcopy(base_model.config)
    ctrue.update(true_config)
    true_model = bi.Model(ctrue)
    
    results = np.zeros(n_trials)
    
    if tqdm_on:
        q = tqdm(range(n_trials))
    else:
        q = range(n_trials)
        
    for iter_i in q:
        # Simulate a background-only dataset using the true model
        d = true_model.simulate(rate_multipliers=dict(wimp=0, 
                                                      **true_rate_multipliers))
        
        # If we believe the electron lifetime is different, we have to change the S2 correction
        # I will assume the drift velocity is unchanged though.
        t_true = true_model.config['e_lifetime']
        t_assumed = ll.base_model.config['e_lifetime']
        if t_true != t_assumed:
            drift_time = d['z'] / true_model.config['v_drift']
            d['cs2'] *= np.exp(-drift_time/t_true) * np.exp(drift_time/t_assumed)
            
        ll.set_data(d)
        
        try:
            kwargs = deepcopy(limit_options)
            kwargs.update({k + '_rate_multiplier': v 
                           for k, v in rate_multipliers.items()})
            results[iter_i] = ll.one_parameter_interval('wimp_rate_multiplier', 
                                                        bound=100,
                                                        confidence_level=meta_config.get('confidence_level', 0.9),
                                                        **kwargs)
        except ValueError:
            results[iter_i] = float('nan')

    return results

limits = np.zeros((len(multipliers), n_trials))
for i, x in enumerate(multipliers):   
    print(x)
    limits[i] = compute_limits(**{change_arg: {setting: x}})

##
# Save results to pickle
##

if not os.path.exists(base_folder):
    os.makedirs(base_folder)

save_pickle(
    dict(setting=setting,
         change_arg=change_arg,
         multipliers=multipliers,
         limits=limits),
    os.path.join(base_folder, setting + '.pkl'))
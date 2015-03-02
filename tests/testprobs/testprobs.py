
from bsfh.likelihood import LikelihoodFunction
from bsfh import model_setup
import numpy as np
import os, fsps


'''
skeleton:
load up some model, instantiate an sps, and load some observations
generate spectrum, compare to observations, assess likelihood
can be run in different environments to ensure underlying stellar pops are consistent
'''

def test_likelihood(param_file=None, sps=None, model=None, obs=None, thetas=None):

	if not param_file:
		param_file = os.getenv('APPS')+'/python/threedhst_bsfh/parameter_files/dtau_intmet/dtau_intmet_params_66.py'

	if not sps:
		# load stellar population, set up custom filters
		sps = fsps.StellarPopulation(zcontinuous=1, compute_vega_mags=False)
		custom_filter_keys = os.getenv('APPS')+'/threedhst_bsfh/filters/filter_keys_threedhst.txt'
		fsps.filters.FILTERS = model_setup.custom_filter_dict(custom_filter_keys)

	if not model:
		model = model_setup.load_model(param_file)
	
	if not obs:
		run_params = model_setup.get_run_params(param_file=param_file)
		obs = model_setup.load_obs(**run_params)

	if thetas is None:
		thetas = model.initial_theta

	likefn = LikelihoodFunction(obs=obs, model=obs)
	mu, phot, x = model.mean_model(thetas, obs, sps = sps)
	lnp_phot = likefn.lnlike_phot(phot, obs=obs, gp=None)

	print lnp_phot
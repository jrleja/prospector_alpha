
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
		param_file = os.getenv('APPS')+'/threedhst_bsfh/parameter_files/dtau_intmet/dtau_intmet_params_66.py'

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
		thetas = np.array([2.06382260e+09,1.55273910e+10,-7.94072640e-01,5.82656906e+01,2.89232690e-01,5.94558117e+00,9.10674270e-01,2.42980691e-01,7.21169529e-01,-1.53203324e+00])

	likefn = LikelihoodFunction(obs=obs, model=model)
	mu, phot, x = model.mean_model(thetas, obs, sps = sps)
	lnp_phot = likefn.lnlike_phot(phot, obs=obs, gp=None)

	print lnp_phot
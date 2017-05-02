import prosp_dutils
from prospect.models import model_setup
import os
import numpy as np
from prospect.likelihood import LikelihoodFunction


param_name = '/Users/joel/code/python/threedhst_bsfh/parameter_files/nonparametric_mocks/nonparametric_mocks_params_1.py'


###### POST_PROCESSING
param_file = model_setup.import_module_from_file(param_name)
outname = param_file.run_params['outfile']
outfolder = os.getenv('APPS')+'/threedhst_bsfh/plots/'+outname.split('/')[-2]+'/'

sample_results, powell_results, model = prosp_dutils.load_prospector_data(outname)



#### CALC_EXTRA_QUANTITIES
parnames = sample_results['model'].theta_labels()

##### modify nebon status
# we want to be able to turn it on and off at will
if sample_results['model'].params['add_neb_emission'] == 2:
	sample_results['model'].params['add_neb_emission'] = np.array(True)

##### initialize sps
sps = model_setup.load_sps(**sample_results['run_params'])


###### MAXPROB_MODEL
# grab maximum probability, plus the thetas that gave it
maxprob = np.max(sample_results['lnprobability'])
probind = sample_results['lnprobability'] == maxprob
thetas = sample_results['chain'][probind,:]
if type(thetas[0]) != np.dtype('float64'):
	thetas = thetas[0]


###### TEST LIKELIHOOD
run_params = model_setup.get_run_params(param_file=param_name)
gp_spec, gp_phot = model_setup.load_gp(**run_params)
likefn = LikelihoodFunction()
mu, phot, x = model.mean_model(model.initial_theta, sample_results['obs'], sps=sps)

mu, phot, x = model.mean_model(thetas, sample_results['obs'], sps = sps)
lnp_phot = likefn.lnlike_phot(phot, obs=sample_results['obs'], gp=gp_phot)
lnp_prior = model.prior_product(thetas)

print lnp_phot + lnp_prior


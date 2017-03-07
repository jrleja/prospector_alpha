import numpy as np
from prospect.io import read_results
import threed_dutils, corner, pickle, math
import matplotlib.pyplot as plt
from prospect.models import model_setup
from copy import copy

### globals
pickle_file = 'chain.pickle'
param_file = 'logsfr_priortest_params.py'

def plot_all(recollate=False,nsample=40000):

	try:
		with open(pickle_file, "rb") as f:
			chain = pickle.load(f)
	except:
		chain = {}

	if (chain.get('sfr',None) is None) or (recollate):
		chain = calc_sfr(nsample=nsample)
		pickle.dump(chain,open(pickle_file, "wb"))

	### corner plot in sSFR
	subcorner_ssfr(chain)
	subcorner(chain)

def return_fake_obs(obs):

	obs['wave_effective'] = None
	obs['filters'] = None
	obs['phot_mask'] = None
	obs['maggies'] = None
	obs['maggies_unc'] = None
	obs['wavelength'] = None
	obs['spectrum'] = None

	return obs

def calc_sfr(nsample=40000):
	
	### setup SPS
	run_params = model_setup.get_run_params(param_file=param_file)
	sps = model_setup.load_sps(**run_params)
	model = model_setup.load_model(**run_params)
	obs = return_fake_obs({})
	nbins = model.params['sfr_fraction'].shape[0]

	#### create chain to sample from
	flatchain = np.random.dirichlet(tuple(1.0 for x in xrange(6)),nsample)
	flatchain = flatchain[:,:-1]

	### define time array for SFHs
	in_years = 10**model.params['agebins']/1e9
	t = in_years.sum(axis=1)/2.

	### output bins
	sfr = np.zeros(shape=(t.shape[0],nsample))

	#### sample the posterior
	for jj in xrange(nsample):
		
		if jj % 100 == 0:
			print float(jj)/nsample

		##### model call, to set parameters
		thetas = flatchain[jj,:]
		_,_,sm = model.mean_model(thetas, obs, sps=sps)

		##### extract sfh parameters
		# pass stellar mass to avoid extra model call
		sfh_params = threed_dutils.find_sfh_params(model,thetas,
			                                       obs,sps,sm=sm)

		#### SFR
		sfr[:,jj] = threed_dutils.return_full_sfh(t, sfh_params,minsfr=-np.inf)

	out = {}
	out['sfr'] = sfr
	out['flatchain'] = flatchain[:nsample,:]
	out['model'] = model

	return out

def subcorner_ssfr(out):

	# pull out the parameter names and flatten the thinned chains
	parnames = out['model'].theta_labels()
	pnew = ['log(SFR '+p.split('_')[-1]+')' for p in parnames] + ['log(SFR 6)']

	'''
	## calculate mformed
	time_per_bin = []
	for (t1, t2) in out['model'].params['agebins']: time_per_bin.append(10**t2-10**t1)

	mformed = np.sum(10**out['flatchain']*np.array(time_per_bin),axis=1)

	## calculate sSFR
	flatchain = np.log10(out['sfr']/mformed)
	'''
	fig = corner.corner(np.log10(out['sfr']).swapaxes(1,0), labels = pnew,
	                    quantiles=[0.16, 0.5, 0.84], show_titles=True)

	fig.savefig('ssfr.corner.png')
	plt.close(fig)

def subcorner(out):
	"""
	Make a corner plot of the (thinned, latter) samples of the posterior
	parameter space.  Optionally make the plot only for a supplied subset
	of the parameters.
	"""

	# pull out the parameter names and flatten the thinned chains
	parnames = out['model'].theta_labels()
	flatchain = out['flatchain']

	fig = corner.corner(flatchain, labels = parnames,
	                    quantiles=[0.16, 0.5, 0.84], show_titles=True)

	fig.savefig('logm.corner.png')
	plt.close(fig)


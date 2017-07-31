import numpy as np
from prospect.io import read_results
import prosp_dutils, corner, pickle, math
import matplotlib.pyplot as plt
from prospect.models import model_setup
from copy import copy

def plot_all(recollate_idp=False,recollate_dir=False):

	outfolder = '/Users/joel/code/python/prospector_alpha/results/prior_test/'
	mcmc_filename = outfolder+'prior_test_dp_mcmc'
	model_filename = outfolder+'prior_test_dp_model'
	sample_results, powell_results, model = read_results.read_pickles(mcmc_filename, model_file=model_filename,inmod=None)

	if (sample_results.get('sfr',None) is None) or (recollate_idp):
		nsample = 40000
		sample_results['obs'] = return_fake_obs({})
		sample_results = calc_sfr(sample_results,nsample=nsample)
		pickle.dump(sample_results,open(mcmc_filename, "wb"))

	if (sample_results.get('sfr_dirichlet',None) is None) or (recollate_dir):
		nsample = 40000
		sample_results['obs'] = return_fake_obs({})
		sample_results = calc_sfr_dirichlet(sample_results,nsample=nsample)
		pickle.dump(sample_results,open(mcmc_filename, "wb"))

	### corner plot in sSFR
	subcorner_ssfr(sample_results,outfolder)
	subcorner_dirichlet(sample_results,outfolder)

	#### plot
	subcorner(sample_results,outfolder)

def return_fake_obs(obs):

	obs['wave_effective'] = None
	obs['filters'] = None
	obs['phot_mask'] = None
	obs['maggies'] = None
	obs['maggies_unc'] = None
	obs['wavelength'] = None
	obs['spectrum'] = None

	return obs

def calc_sfr(sample_results,nsample=40000):
	
	### setup SPS
	sps = model_setup.load_sps(**sample_results['run_params'])

	#### choose chain to sample from
	in_priors = np.isfinite(prosp_dutils.chop_chain(sample_results['lnprobability'])) == True
	flatchain = copy(prosp_dutils.chop_chain(sample_results['chain'])[in_priors])
	np.random.shuffle(flatchain)

	### define time array for SFHs
	in_years = 10**sample_results['model'].params['agebins']/1e9
	t = in_years.sum(axis=1)/2.

	### output bins
	sfr = np.zeros(shape=(t.shape[0],nsample))

	#### sample the posterior
	for jj in xrange(nsample):
		
		##### model call, to set parameters
		thetas = flatchain[jj,:]
		_,_,sm = sample_results['model'].mean_model(thetas, sample_results['obs'], sps=sps)

		##### extract sfh parameters
		# pass stellar mass to avoid extra model call
		sfh_params = prosp_dutils.find_sfh_params(sample_results['model'],thetas,
			                                       sample_results['obs'],sps,sm=sm)

		#### SFR
		sfr[:,jj] = prosp_dutils.return_full_sfh(t, sfh_params)

	sample_results['sfr'] = sfr
	sample_results['flatchain'] = flatchain[:nsample,:]

	return sample_results

def calc_sfr_dirichlet(sample_results,nsample=40000):
	
	### setup SPS
	sps = model_setup.load_sps(**sample_results['run_params'])

	#### define chain to sample from
	nbins = 6
	flatchain = np.random.dirichlet(tuple(1.0 for x in xrange(nbins)),nsample)
	for ii in xrange(nsample): flatchain[ii,:] /= np.sum(flatchain[ii,:])
	flatchain = flatchain[:,:-1]

	### define time array for SFHs
	in_years = 10**sample_results['model'].params['agebins']/1e9
	t = in_years.sum(axis=1)/2.

	### output bins
	sfr = np.zeros(shape=(t.shape[0],nsample))

	#### sample the posterior
	for jj in xrange(nsample):
		
		##### model call, to set parameters
		thetas = flatchain[jj,:]
		_,_,sm = sample_results['model'].mean_model(thetas, sample_results['obs'], sps=sps)

		##### extract sfh parameters
		# pass stellar mass to avoid extra model call
		sfh_params = prosp_dutils.find_sfh_params(sample_results['model'],thetas,
			                                       sample_results['obs'],sps,sm=sm)

		#### SFR
		sfr[:,jj] = prosp_dutils.return_full_sfh(t, sfh_params)

	sample_results['sfr_dirichlet'] = sfr
	sample_results['flatchain_dirichlet'] = flatchain

	return sample_results


def marginalized_dirichlet_pdf(x, alpha_one, alpha_sum):
	''' 
	returns the marginalized Dirichlet PDF (i.e., for ONE component of the distribution).
	From Wikipedia, the marginalized Dirichlet PDF is the Beta function, 
	where a = alpha_1 and b = sum(alpha)-alpha_1. Here, "alpha" are the concentration 
	parameters of the Dirichlet distribution, and alpha_1 is the dimension to marginalize over.

	input "x" is the x-vector
	'''

	return (math.gamma(alpha_one+alpha_sum) * x**(alpha_one-1) * \
			 (1-x)**(alpha_sum-1)) / \
			 (math.gamma(alpha_one)*math.gamma(alpha_sum)) 

def subcorner_dirichlet(sample_results, outfolder):
	"""
	Make a corner plot of the (thinned, latter) samples of the posterior
	parameter space.  Optionally make the plot only for a supplied subset
	of the parameters.
	"""

	# pull out the parameter names and flatten the thinned chains
	parnames = sample_results['model'].theta_labels()
	pnew = ['sSFR '+p.split('_')[-1] for p in parnames]
	pnew.append('sSFR_6')
	flatchain = np.log10(sample_results['sfr_dirichlet']/10**sample_results['model'].params['logmass'])

	fig = corner.corner(flatchain.swapaxes(1,0), labels = pnew,
	                    quantiles=[0.16, 0.5, 0.84], show_titles=True)

	#### plot Dirichlet prior
	nbins = 6
	test_x = np.linspace(1e-5,1,1000,endpoint=False)
	test_dirichlet = marginalized_dirichlet_pdf(test_x, 1,nbins-1)

	axes = fig.get_axes()
	print len(axes)
	try:
		print axes.shape
	except:
		pass

	for i in xrange(nbins-1):
		idx = i*(nbins-1)+i

		# normalize dirichlet distribution
		ylim = axes[idx].get_ylim()
		test_dirichlet *= ylim[1] / test_dirichlet[0]

		# plot it
		axes[idx].plot(test_x,test_dirichlet,linestyle='-',color='red',lw=2,alpha=0.5)

	fig.savefig(outfolder+'sfr.dirichlet.corner.png')
	plt.close(fig)

def subcorner_ssfr(sample_results, outfolder):
	"""
	Make a corner plot of the (thinned, latter) samples of the posterior
	parameter space.  Optionally make the plot only for a supplied subset
	of the parameters.
	"""

	# pull out the parameter names and flatten the thinned chains
	parnames = sample_results['model'].theta_labels()
	pnew = ['sSFR '+p.split('_')[-1] for p in parnames]
	pnew.append('sSFR_6')
	flatchain = np.log10(sample_results['sfr']/10**sample_results['model'].params['logmass'])

	fig = corner.corner(flatchain.swapaxes(1,0), labels = pnew,
	                    quantiles=[0.16, 0.5, 0.84], show_titles=True)

	#### plot Dirichlet prior
	nbins = 6
	test_x = np.linspace(1e-5,1,1000,endpoint=False)
	test_dirichlet = marginalized_dirichlet_pdf(test_x, 1,nbins-1)

	axes = fig.get_axes()
	print len(axes)
	try:
		print axes.shape
	except:
		pass

	for i in xrange(nbins-1):
		idx = i*(nbins-1)+i

		# normalize dirichlet distribution
		ylim = axes[idx].get_ylim()
		test_dirichlet *= ylim[1] / test_dirichlet[0] * 0.3

		# plot it
		axes[idx].plot(test_x,test_dirichlet,linestyle='-',color='red',lw=2,alpha=0.5)

	fig.savefig(outfolder+'sfr.corner.png')
	plt.close(fig)

def subcorner(sample_results, outfolder):
	"""
	Make a corner plot of the (thinned, latter) samples of the posterior
	parameter space.  Optionally make the plot only for a supplied subset
	of the parameters.
	"""

	# pull out the parameter names and flatten the thinned chains
	parnames = sample_results['model'].theta_labels()
	flatchain = prosp_dutils.chop_chain(sample_results['chain'])

	fig = corner.corner(flatchain, labels = parnames,
	                    quantiles=[0.16, 0.5, 0.84], show_titles=True)

	#### plot Dirichlet prior
	nbins = 6
	test_x = np.linspace(1e-5,1,1000,endpoint=False)
	test_dirichlet = marginalized_dirichlet_pdf(test_x, 1,nbins-1)

	axes = fig.get_axes()
	print len(axes)
	try:
		print axes.shape
	except:
		pass

	for i in xrange(nbins-1):
		idx = i*(nbins-1)+i

		# normalize dirichlet distribution
		ylim = axes[idx].get_ylim()
		test_dirichlet *= ylim[1] / test_dirichlet[0] * 0.8

		# plot it
		axes[idx].plot(test_x,test_dirichlet,linestyle='-',color='red',lw=2,alpha=0.5)

	fig.savefig(outfolder+'prior_test.corner.png')
	plt.close(fig)


import read_sextractor, read_data, random, os, threed_dutils
import numpy as np
from astropy.table import Table, vstack
from astropy.io import ascii
from astropy import units as u
import nonparametric_mocks_params as nparams

#### NONPARAMETRIC GLOBALS
sps = nparams.load_sps(**nparams.run_params)
model = nparams.load_model(**nparams.run_params)
obs = nparams.load_obs(**nparams.run_params)
sps.update(**model.params)

#### RANDOM SEED
random.seed(25001)

def return_bounds(parname,model,i):

	'''
	returns parameter boundaries
	if test_sfhs is on, puts special constraints on certain variables
	these special constraints are defined in return_test_sfhs
	'''

	bounds = model.theta_bounds()[i]

	return bounds[0],bounds[1]

def construct_mocks(basename,outname=None,add_zp_err=False, plot_mock=False):

	'''
	Generate model SEDs and add noise
	IMPORTANT: linked+outlier noise will NOT be added if those variables are not free 
	parameters in the passed parameter file!
	'''

	#### output names ####
	outname = '/Users/joel/code/python/threedhst_bsfh/data/'+basename

	#### basic parameters ####
	# hack to make it run 100x times in for-loop
	# so i can calculate sfr each time
	noise               = 0.02            # perturb fluxes
	reported_noise      = 0.02            # reported noise
	ntest               = 100             # number of mock galaxies to generate
	total_mass          = 1e10            # total mass of galaxy

	#### generate random model parameters ####
	nparams = len(model.initial_theta)
	testparms = np.zeros(shape=(ntest,nparams))
	parnames = np.array(model.theta_labels())

	#### generate nonparametric SFH distribution
	# expectation value is constant SFH
	# get this by using Dirichlet distribution, with expectation = (size of time bins) / total time
	# do it in LOG SPACE, so that we don't have most of the SFH in oldest bin...
	nbins = model.params['sfh_logmass'].shape[0]
	bin_weight = np.zeros(nbins)
	exp = 1.7 # this controls the mean fractional mass in each age bin
	for i in xrange(nbins): bin_weight[i] = exp**model.params['agebins'][i,1]-exp**model.params['agebins'][i,0]
	norm_bin_weight = bin_weight / 1e2 # this numeric factor controls the dispersion of the Dirichlet distribution (see docs)
	norm_bin_tuple = tuple(bin for bin in norm_bin_weight)

	sfh_distribution = np.random.dirichlet(norm_bin_tuple, ntest)

	'''
	print 'MEAN'
	for i in xrange(nbins): print np.mean(sfh_distribution[:,i])
	print 'STANDARD DEVIATION'
	for i in xrange(nbins): print np.std(sfh_distribution[:,i])/np.mean(sfh_distribution[:,i])
	'''

	for ii in xrange(nparams):
		
		#### nonparametric bins, using Dirichlet distribution
		if 'sfh_logmass' in parnames[ii]:
			component = int(parnames[ii][-1])-1
			testparms[:,ii] = np.log10(sfh_distribution[:,component]*total_mass)
			bad = testparms[:,ii] < model.theta_bounds()[ii][0]
			testparms[bad,ii] = model.theta_bounds()[ii][0]

		#### choose reasonable amounts of dust ####
		elif parnames[ii] == 'dust2':
			min = 0.0
			max = 0.5
			for kk in xrange(ntest): testparms[kk,ii] = random.random()*(max-min)+min

		#### dust1 must match the dust1/dust2 prior #### 
		elif parnames[ii] == 'dust1':
			dust2 = testparms[:,parnames == 'dust2']
			min = dust2 * 0.0
			max = dust2 * 1.5
			for kk in xrange(ntest): testparms[kk,ii] = random.random()*(max[kk]-min[kk])+min[kk]

		#### apply dust_index prior, and clip the upper bounds! ####
		elif parnames[ii] == 'dust_index':
			min = -1.4
			max = model.theta_bounds()[ii][1]
			for kk in xrange(ntest): testparms[kk,ii] = np.clip(random.gauss(0.0, 0.5),min,max)
		
		else:
			min = model.theta_bounds()[ii][0]
			max = model.theta_bounds()[ii][1]
			for kk in xrange(ntest): testparms[kk,ii] = random.random()*(max-min)+min

	#### make sure priors are satisfied
	for ii in xrange(ntest):
		assert np.isfinite(model.prior_product(testparms[ii,:]))

	#### write out thetas ####
	with open(outname+'.dat', 'w') as f:
		
		### header ###
		f.write('# ')
		for theta in model.theta_labels():
			f.write(theta+' ')
		f.write('\n')

		### data ###
		for ii in xrange(ntest):
			for kk in xrange(nparams):
				f.write(str(testparms[ii,kk])+' ')
			f.write('\n')

	#### set up photometry output ####
	nfilters = len(obs['filters'])
	maggies     = np.zeros(shape=(ntest,nfilters))
	maggies_unc = np.zeros(shape=(ntest,nfilters))

	#### generate photometry, add noise ####
	for ii in xrange(ntest):
		model.initial_theta = testparms[ii,:]
		spec,maggiestemp,_ = model.mean_model(model.initial_theta, obs, sps=sps)

		if plot_mock:
			if ii == 0:
				import matplotlib.pyplot as plt
				plt.ioff() # don't pop up a window for each plot
				fig, axarr = plt.subplots(ncols=ntest/10, nrows=10, figsize=(30,30))
				ax = np.ravel(axarr)
			good = (sps.wavelengths > 1e3) & (sps.wavelengths < 1e6)
			factor = 3e18 / sps.wavelengths[good]
			ax[ii].plot(np.log10(sps.wavelengths[good]),np.log10(spec[good]*factor))
			for nn in xrange(nparams):
				if 'sfh_logmass' in parnames[nn]:
					ax[ii].text(0.05,0.95-nn*0.05,"{:.2f}".format(model.initial_theta[nn]),fontsize=8,transform = ax[ii].transAxes)
			if ii < ntest-10:
				ax[ii].xaxis.get_major_ticks()[0].label1On = False
			else:
				ax[ii].set_xlabel(r'log($\lambda$)')

		maggies[ii,:] = maggiestemp

		#### record noise ####
		maggies_unc[ii,:] = maggies[ii,:]*reported_noise

		#### add noise ####
		for kk in xrange(nfilters): 
			
			###### general noise
			add_noise = random.gauss(0, noise)
			print obs['filters'][kk].name+': ' + "{:.3f}".format(add_noise)
			maggies[ii,kk] += add_noise*maggies[ii,kk]

	if plot_mock:
		plt.tight_layout()
		plt.savefig(plot_mock,dpi=150)
		plt.close()

	#### output ####
	#### ids first ####
	ids =  np.arange(ntest)+1
	with open(outname+'.ids', 'w') as f:
	    for id in ids:
	        f.write(str(id)+'\n')

	#### photometry ####
	with open(outname+'.cat', 'w') as f:
		
		### header ###
		f.write('# id ')
		for filter in obs['filters']:
			f.write('f_'+filter.name+' e_' +filter.name+' ')
		f.write('\n')

		### data ###
		for ii in xrange(ntest):
			f.write(str(ids[ii])+' ')
			for kk in xrange(nfilters):
				f.write(str(maggies[ii,kk])+' '+str(maggies_unc[ii,kk]) + ' ')
			f.write('\n')
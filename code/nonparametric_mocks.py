import read_sextractor, read_data, random, os
import numpy as np
from astropy.table import Table, vstack
from astropy.io import ascii
from astropy import units as u
import nonparametric_mocks_params as nonparam
import threed_dutils

#### NONPARAMETRIC GLOBALS
sps = nonparam.load_sps(**nonparam.run_params)
model = nonparam.load_model(**nonparam.run_params)
obs = nonparam.load_obs(**nonparam.run_params)
sps.update(**model.params)

#### RANDOM SEED
random.seed(25001)
np.random.seed(50)

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
	noise               = 0.05            # perturb fluxes
	reported_noise      = 0.05            # reported noise
	ntest               = 100             # number of mock galaxies to generate

	#### generate random model parameters ####
	nparams = len(model.initial_theta)
	testparms = np.zeros(shape=(ntest,nparams))
	parnames = np.array(model.theta_labels())

	#### generate nonparametric SFH distribution
	# expectation value is constant SFH
	# get this by using Dirichlet distribution, with expectation = (size of time bins) / total time
	nbins = model.params['sfr_fraction'].shape[0]+1
	sfh_distribution = np.random.dirichlet(tuple(1.0 for x in xrange(nbins)),ntest)
	for ii in xrange(ntest): sfh_distribution[ii,:] /= np.sum(sfh_distribution[ii,:])

	#### generate all parameters
	for ii in xrange(nparams):
		
		#### nonparametric bins, using Dirichlet distribution
		if 'sfr_fraction' in parnames[ii]:
			component = int(parnames[ii][-1])-1
			testparms[:,ii] = sfh_distribution[:,component]

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
		spec,maggiestemp,sm = model.mean_model(testparms[ii,:], obs, sps=sps)

		if plot_mock:
			if ii == 0:
				import matplotlib.pyplot as plt
				plt.ioff() # don't pop up a window for each plot
				fig, axarr = plt.subplots(ncols=ntest/10, nrows=10, figsize=(30,30))
				ax = np.ravel(axarr)
			good = (sps.wavelengths > 1e3) & (sps.wavelengths < 1e6)
			factor = 3e18 / sps.wavelengths[good]
			ax[ii].plot(np.log10(sps.wavelengths[good]),np.log10(spec[good]*factor))

			## write mass in each bin
			frac = 0.0
			for nn in xrange(nparams):
				if 'sfr_fraction' in parnames[nn]:
					ax[ii].text(0.98,0.95-nn*0.05,"{:.2f}".format(testparms[ii,nn]),fontsize=8,transform = ax[ii].transAxes,ha='right')
					frac += testparms[ii,nn]
			ax[ii].text(0.98,0.95-(nbins)*0.05,"{:.2f}".format(1-frac),fontsize=8,transform = ax[ii].transAxes,ha='right')

			## write sSFR(10 Myr, 100 Myr, 1 Gyr)
			sfh_params = threed_dutils.find_sfh_params(model,testparms[ii,:],obs,sps,sm=sm)

			if ii == 0:
				print sfh_params.keys()
			ssfr = np.array([threed_dutils.calculate_sfr(sfh_params, 0.01, minsfr=-np.inf, maxsfr=np.inf),\
			                 threed_dutils.calculate_sfr(sfh_params, 0.1,  minsfr=-np.inf, maxsfr=np.inf),\
			                 threed_dutils.calculate_sfr(sfh_params, 1.0,  minsfr=-np.inf, maxsfr=np.inf)])/sfh_params['mass']
			ssfr_label = ['10 Myr','100 Myr','1 Gyr']
			for nn in xrange(ssfr.shape[0]): ax[ii].text(0.05,0.2-nn*0.05,"{:.2e}".format(ssfr[nn])+' '+ssfr_label[nn],fontsize=8,transform = ax[ii].transAxes)

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

	print 1/0

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
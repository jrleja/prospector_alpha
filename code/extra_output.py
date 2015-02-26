from bsfh import model_setup,read_results
import os, threed_dutils, triangle, threedhst_diag, pickle, sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import WMAP9
from scipy.optimize import brentq
from copy import copy
from astropy import constants

def calc_emp_ha(mass,tage,tau,sf_start,tuniv,dust2,dustindex):

	ncomp = len(mass)
	ha_flux=0.0
	oiii_flux=0.0
	for kk in xrange(ncomp):
		sfr = threed_dutils.integrate_sfh(np.asarray([tage[kk]-0.1]),
			                              np.asarray([tage[kk]]),
			                              np.asarray([mass[kk]]),
			                              np.asarray([tage[kk]]),
			                              np.asarray([tau[kk]]),
	                                      np.asarray([sf_start[kk]]))*mass[kk]/(0.1*1e9)
		x=threed_dutils.synthetic_emlines(mass[kk],
				                          sfr,
				                          0.0,
				                          dust2[kk],
				                          dustindex)
		oiii_flux = oiii_flux + x['flux'][x['name'] == '[OIII]']
		ha_flux = ha_flux + x['flux'][x['name'] == 'Halpha']
	
	return ha_flux,oiii_flux

def sfh_half_time(x,mass,tage,tau,sf_start,c):
	 return threed_dutils.integrate_sfh(sf_start,x,mass,tage,tau,sf_start)-c

def halfmass_assembly_time(mass,tage,tau,sf_start,tuniv):

	# calculate half-mass assembly time
	# c = 0.5 if half-mass assembly time occurs before burst
	half_time = brentq(sfh_half_time, 0,14,
                       args=(mass,tage,tau,sf_start,0.5),
                       rtol=1.48e-08, maxiter=100)

	return tuniv-half_time

def measure_emline_lum(w,spec,emline,wavelength,sideband,saveplot=False):
	
	'''
	takes spec(on)-spec(off) to measure emission line luminosity
	sideband is defined for each emission line after visually 
	inspecting the spectral sampling density around each line
	'''

	emline_flux = np.zeros(len(wavelength))
	
	for jj in xrange(len(wavelength)):
		center = (np.abs(w-wavelength[jj]) == np.min(np.abs(w-wavelength[jj]))).nonzero()[0][0]
		bot,top = center-sideband[jj],center+sideband[jj]+1
		
		# spectral units on arrival are flux density (maggies)
		# we convert first to flux density in fnu, in cgs
		# then integrate to get the line flux
		factor = 3e18 / w**2
		wings = spec[bot:top]*factor[bot:top]
		emline_flux[jj] = np.trapz(wings, w[bot:top])

		if saveplot and jj==4:
			plt.plot(w,spec*factor,'ro',linestyle='-')
			plt.plot(w[bot:top], wings, 'bo',linestyle=' ')
			plt.xlim(wavelength[jj]-800,wavelength[jj]+800)
			plt.ylim(-np.max(wings)*0.2,np.max(wings)*1.2)
			
			plotlines=['[SIII]','[NII]','Halpha','[SII]']
			plotlam  =np.array([6312,6583,6563,6725])
			for kk in xrange(len(plotlam)):
				plt.vlines(plotlam[kk],plt.ylim()[0],plt.ylim()[1],color='0.5',linestyle='--')
				plt.text(plotlam[kk],(plt.ylim()[0]+plt.ylim()[1])/2.*(1.0-kk/6.0),plotlines[kk])
			plt.savefig(os.getenv('APPS')+'/threedhst_bsfh/plots/testem/emline_'+str(saveplot)+'.png',dpi=300)
			plt.close()
	return emline_flux

def calc_extra_quantities(sample_results, nsamp_mc=1000):

	'''' 
	CALCULATED QUANTITIES
	model nebular emission line strength
	model star formation history parameters (ssfr,sfr,half-mass time)
	'''

    # save nebon/neboff status
	nebstatus = sample_results['model'].params['add_neb_emission']

    # find SFH parameters that are variables in the chain
    # save their indexes for future use
	str_sfh_parms = ['mass','tau','sf_start','tage']
	parnames = sample_results['model'].theta_labels()
	indexes = []
	for string in str_sfh_parms:
		indexes.append(np.char.find(parnames,string) > -1)
	indexes = np.array(indexes,dtype='bool')

    # initialize output arrays for SFH parameters
	nwalkers,niter = sample_results['chain'].shape[:2]
	half_time,sfr_10,sfr_100,sfr_1000,ssfr_100,totmass,emp_ha,emp_oiii = [np.zeros(shape=(nwalkers,niter)) for i in range(8)]

    # get constants for SFH calculation [MODEL DEPENDENT]
	z = sample_results['model'].params['zred'][0]
	tuniv = WMAP9.age(z).value
	deltat=[0.01,0.1,1.0] # in Gyr
    
	##### TEMPORARY CALCULATION TO DO EMPIRICAL EMISSION LINES IN CHAIN ######
	dust2_index = np.array([True if x[:-2] == 'dust2' else False for x in parnames])
	dust_index_index = np.array([True if x == 'dust_index' else False for x in parnames])

	######## SFH parameters #########
	for jj in xrange(nwalkers):
		for kk in xrange(niter):
	    
			# extract sfh parameters
			# the ELSE finds SFH parameters that are NOT part of the chain
			sfh_parms = []
			for ll in xrange(len(str_sfh_parms)):
				if np.sum(indexes[ll]) > 0:
					sfh_parms.append(sample_results['chain'][jj,kk,indexes[ll]])
				else:
					_ = [x['init'] for x in sample_results['model'].config_list if x['name'] == str_sfh_parms[ll]][0]
					if len(np.atleast_1d(_)) != np.sum(indexes[0]):
						_ = np.zeros(np.sum(indexes[0]))+_
					sfh_parms.append(_)
			mass,tau,sf_start,tage = sfh_parms

			# calculate half-mass assembly time, sfr
			half_time[jj,kk] = halfmass_assembly_time(mass,tage,tau,sf_start,tuniv)

			# empirical halpha
			emp_ha[jj,kk],emp_oiii[jj,kk] = calc_emp_ha(mass,tage,tau,sf_start,tuniv,
				                                        sample_results['chain'][jj,kk,dust2_index],sample_results['chain'][jj,kk,dust_index_index])

			# calculate sfr
			sfr_10[jj,kk] = threed_dutils.integrate_sfh(tage-deltat[0],tage,mass,tage,tau,
	                                                    sf_start)*np.sum(mass)/(deltat[0]*1e9)
			sfr_100[jj,kk] = threed_dutils.integrate_sfh(tage-deltat[1],tage,mass,tage,tau,
	                                                     sf_start)*np.sum(mass)/(deltat[1]*1e9)
			sfr_1000[jj,kk] = threed_dutils.integrate_sfh(tage-deltat[2],tage,mass,tage,tau,
	                                                     sf_start)*np.sum(mass)/(deltat[2]*1e9)

			ssfr_100[jj,kk] = sfr_100[jj,kk] / np.sum(mass)

			totmass[jj,kk] = np.sum(mass)
	     
	# CALCULATE Q16,Q50,Q84 FOR VARIABLE PARAMETERS
	ntheta = len(sample_results['initial_theta'])
	q_16, q_50, q_84 = (np.zeros(ntheta)+np.nan for i in range(3))
	for kk in xrange(ntheta): q_16[kk], q_50[kk], q_84[kk] = triangle.quantile(sample_results['flatchain'][:,kk], [0.16, 0.5, 0.84])

	# CALCULATE Q16,Q50,Q84 FOR EXTRA PARAMETERS
	extra_chain = np.dstack((half_time.reshape(half_time.shape[0],half_time.shape[1],1), 
		                     sfr_10.reshape(sfr_10.shape[0],sfr_10.shape[1],1), 
		                     sfr_100.reshape(sfr_100.shape[0],sfr_100.shape[1],1), 
		                     sfr_1000.reshape(sfr_1000.shape[0],sfr_1000.shape[1],1), 
		                     ssfr_100.reshape(ssfr_100.shape[0],ssfr_100.shape[1],1),
		                     totmass.reshape(ssfr_100.shape[0],totmass.shape[1],1),
		                     emp_ha.reshape(emp_ha.shape[0],emp_ha.shape[1],1),
		                     emp_oiii.reshape(emp_oiii.shape[0],emp_oiii.shape[1],1)
		                     ))
	extra_flatchain = threed_dutils.chop_chain(extra_chain)
	nextra = extra_chain.shape[-1]
	q_16e, q_50e, q_84e = (np.zeros(nextra)+np.nan for i in range(3))
	for kk in xrange(nextra): q_16e[kk], q_50e[kk], q_84e[kk] = triangle.quantile(extra_flatchain[:,kk], [0.16, 0.5, 0.84])

    ######## MODEL CALL PARAMETERS ########
    # measure emission lines
	emline = ['[OII]','Hbeta','[OIII]1','[OIII]2','Halpha','[SII]']
	wavelength = np.array([3728,4861.33,4959,5007,6562,6732.71])
	sideband   = np.array([20,1,1,1,2,1])
	nline = len(emline)

	# initialize sps
	# check to see if we want zcontinuous=2 (i.e., the MDF)
	if np.sum([1 for x in sample_results['model'].config_list if x['name'] == 'pmetals']) > 0:
		sps = threed_dutils.setup_sps()
	else:
		sps = threed_dutils.setup_sps(zcontinuous=1)

	# set up MIPS + fake L_IR filter
	mips_flux = np.zeros(nsamp_mc)
	mips_index = [i for i, s in enumerate(sample_results['obs']['filters']) if 'mips' in s]
	botlam = np.atleast_1d(8e4-1)
	toplam = np.atleast_1d(1000e4+1)
	edgetrans = np.atleast_1d(0)
	lir_filter = [[np.concatenate((botlam,np.linspace(8e4, 1000e4, num=100),toplam))],
	              [np.concatenate((edgetrans,np.ones(100),edgetrans))]]

    # setup outputs
	lineflux = np.empty(shape=(nsamp_mc,nline))
	lir      = np.zeros(nsamp_mc)

	# save initial states
	neb_em = sample_results['model'].params['add_neb_emission']
	con_em = sample_results['model'].params['add_neb_continuum']
	z      = np.atleast_1d(sample_results['model'].params['zred'])

    # use randomized, flattened, thinned chain for posterior draws
	flatchain = copy(sample_results['flatchain'])
	np.random.shuffle(flatchain)
	for jj in xrange(nsamp_mc):
		thetas = flatchain[jj,:]
		
		# nebon
		sample_results['model'].params['add_neb_emission'] = np.array(2)
		sample_results['model'].params['add_neb_continuum'] = np.array(True)
		spec,mags,w = sample_results['model'].mean_model(thetas, sample_results['obs'], sps=sps,norm_spec=False)
		
		# neboff
		sample_results['model'].params['add_neb_emission'] = np.array(False)
		sample_results['model'].params['add_neb_continuum'] = np.array(False)
		spec_neboff,mags_neboff,w = sample_results['model'].mean_model(thetas, sample_results['obs'], sps=sps,norm_spec=False)

		# randomly save emline fig
		if jj == 5:
			saveplot=sample_results['run_params']['objname']
		else:
			saveplot=False

		lineflux[jj]= measure_emline_lum(w/(1+z),spec-spec_neboff,emline,wavelength,sideband,saveplot=saveplot)

		# calculate redshifted mips magnitudes
		mips_flux[jj] = mags_neboff[mips_index][0]*1e10 # comes out in maggies, convert to flux such that AB zeropoint is 25 mags

		# now calculate z=0 magnitudes
		sample_results['model'].params['zred'] = np.atleast_1d(0.00)
		spec_neboff,mags_neboff,w = sample_results['model'].mean_model(thetas, sample_results['obs'], sps=sps,norm_spec=False)

		_,lir[jj]     = threed_dutils.integrate_mag(w,spec_neboff,lir_filter, z=None, alt_file=None) # comes out in ergs/s
		#tmips  = threed_dutils.integrate_mag(w,spec_neboff,'MIPS_24um_AEGIS', z=sps.params['zred'], alt_file=None) # comes out in ergs/s
		#tmips_intrin  = threed_dutils.integrate_mag(w,spec_neboff,'MIPS_24um_AEGIS', z=None, alt_file=None) # comes out in ergs/s

		#print mips_flux[jj]
		#print 1/0
		lir[jj]       = lir[jj] / 3.846e33 #  convert to Lsun

		# revert
		sample_results['model'].params['zred'] = np.atleast_1d(z)

	
	# restore initial states
	sample_results['model'].params['add_neb_emission'] = neb_em
	sample_results['model'].params['add_neb_continuum'] = con_em

	##### MAXIMUM PROBABILITY
	# grab best-fitting model

	# if we're stupid and pre-cut the chain but not the lnprobability,
	# this cuts lnprobability to the same shape
	if sample_results['lnprobability'].shape[1] != sample_results['chain'].shape[1]:
		sample_results['lnprobability'] = sample_results['lnprobability'][:,-sample_results['chain'].shape[1]:]

	maxprob = np.max(sample_results['lnprobability'])
	probind = sample_results['lnprobability'] == maxprob
	thetas = sample_results['chain'][probind,:]
	if type(thetas[0]) != np.dtype('float64'):
		thetas = thetas[0]
	maxhalf_time,maxsfr_10,maxsfr_100,maxsfr_1000,maxssfr_100,maxtotmass,maxemp_ha,maxemp_oiii = extra_chain[probind.nonzero()[0][0],probind.nonzero()[1][0],:]

	# grab most likely emlines
	sample_results['model'].params['add_neb_emission'] = np.array(True)
	spec,mags,w = sample_results['model'].mean_model(thetas, sample_results['obs'], sps=sps)
	sample_results['model'].params['add_neb_emission'] = np.array(False)
	spec_neboff,mags_neboff,w = sample_results['model'].mean_model(thetas, sample_results['obs'], sps=sps)
	lineflux_maxprob = measure_emline_lum(w,spec-spec_neboff,emline,wavelength,sideband)

	##### FORMAT EMLINE OUTPUT #####
	q_16em, q_50em, q_84em, thetamaxem = (np.zeros(nline)+np.nan for i in range(4))
	for kk in xrange(nline): q_16em[kk], q_50em[kk], q_84em[kk] = triangle.quantile(lineflux[:,kk], [0.16, 0.5, 0.84])
	emline_info = {'name':emline,'lam':wavelength,
	               'fluxchain':lineflux,
	               'q16':q_16em,
	               'q50':q_50em,
	               'q84':q_84em,
	               'maxprob':lineflux_maxprob}
	sample_results['model_emline'] = emline_info

	###### FORMAT MIPS OUTPUT
	mips = {'mips_flux':mips_flux,'L_IR':lir}
	sample_results['mips'] = mips

	# EXTRA PARAMETER OUTPUTS #
	extras = {'chain': extra_chain,
			  'flatchain': extra_flatchain,
			  'parnames': np.array(['half_time','sfr_10','sfr_100','sfr_1000','ssfr_100','totmass','emp_ha','emp_oiii']),
			  'maxprob': np.array([maxhalf_time,maxsfr_10,maxsfr_100,maxsfr_1000,maxssfr_100,maxtotmass,maxemp_ha,maxemp_oiii]),
			  'q16': q_16e,
			  'q50': q_50e,
			  'q84': q_84e}
	sample_results['extras'] = extras

	# QUANTILE OUTPUTS #
	quantiles = {'q16':q_16,
				 'q50':q_50,
				 'q84':q_84,
				 'maxprob_params':thetas,
				 'maxprob':maxprob}
	sample_results['quantiles'] = quantiles

	# reset nebon/neboff status in model
	sample_results['model'].params['add_neb_emission'] = np.array(nebstatus)

	return sample_results

def post_processing(param_name, add_extra=True, nsamp_mc=1000):

	'''
	Driver. Loads output, makes all plots for a given galaxy.
	'''
	
	print 'begun post-processing'
	parmfile = model_setup.import_module_from_file(param_name)
	print 'loaded param file'
	outname = parmfile.run_params['outfile']
	outfolder = os.getenv('APPS')+'/threedhst_bsfh/plots/'+outname.split('/')[-2]+'/'
	print 'defined outfolder'

	# thin and chop the chain?
	thin=1
	chop_chain=1.666

	print 'about to check output folder existence'
	# make sure the output folder exists
	try:
		os.makedirs(outfolder)
	except OSError:
		pass
	print 'checked output folder existence'

	# find most recent output file
	# with the objname
	folder = "/".join(outname.split('/')[:-1])
	filename = outname.split("/")[-1]
	files = [f for f in os.listdir(folder) if "_".join(f.split('_')[:-2]) == filename]	
	times = [f.split('_')[-2] for f in files]

	# if we found no files, skip this object
	if len(times) == 0:
		print 'Failed to find any files in ' + folder + ' of form ' + filename
		return 0

	print 'found files'
	# load results
	mcmc_filename=outname+'_'+max(times)+"_mcmc"
	model_filename=outname+'_'+max(times)+"_model"
	
	print 'loading ' + mcmc_filename +', ' + model_filename

	try:
		sample_results, powell_results, model = read_results.read_pickles(mcmc_filename, model_file=model_filename,inmod=None)
	except (ValueError,EOFError) as e:
		print mcmc_filename + ' failed during output writing'
		print e
		return 0
	except IOError as e:
		print mcmc_filename + ' does not exist!'
		print e
		return 0

	print 'Successfully loaded file'

	if add_extra:
		print 'ADDING EXTRA OUTPUT FOR ' + filename + ' in ' + outfolder
		sample_results['flatchain'] = threed_dutils.chop_chain(sample_results['chain'])
		sample_results = calc_extra_quantities(sample_results,nsamp_mc=nsamp_mc)
	
		### SAVE OUTPUT HERE
		pickle.dump(sample_results,open(mcmc_filename, "wb"))

	### PLOT HERE
	threedhst_diag.make_all_plots(sample_results=sample_results,filebase=outname,outfolder=outfolder)

if __name__ == "__main__":
	post_processing(sys.argv[1])








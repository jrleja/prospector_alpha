from bsfh import model_setup,read_results
import os, threed_dutils, triangle, threedhst_diag, pickle, sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import WMAP9
from scipy.optimize import brentq
from copy import copy
from astropy import constants

def calc_emp_ha(mass,tage,tau,sf_start,tuniv,dust2,dustindex,ncomp=1):

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

def maxprob_model(sample_results,sps):

	# grab maximum probability, plus the thetas that gave it
	maxprob = np.max(sample_results['lnprobability'])
	probind = sample_results['lnprobability'] == maxprob
	thetas = sample_results['chain'][probind,:]
	if type(thetas[0]) != np.dtype('float64'):
		thetas = thetas[0]

	# ensure that maxprob stored is the same as calculated now
	current_maxprob = threed_dutils.test_likelihood(sps=sps, model=sample_results['model'], obs=sample_results['obs'], thetas=thetas)
	print current_maxprob
	print maxprob
	np.testing.assert_array_almost_equal(current_maxprob,maxprob,decimal=4)

	return thetas, maxprob

def calc_extra_quantities(sample_results, nsamp_mc=1000):

	'''' 
	CALCULATED QUANTITIES
	model nebular emission line strength
	model star formation history parameters (ssfr,sfr,half-mass time)
	'''

    # save nebon/neboff status
	nebstatus = sample_results['model'].params['add_neb_emission']

	# initialize sps
	# check to see if we want zcontinuous=2 (i.e., the MDF)
	if np.sum([1 for x in sample_results['model'].config_list if x['name'] == 'pmetals']) > 0:
		sps = threed_dutils.setup_sps(zcontinuous=2)
	else:
		sps = threed_dutils.setup_sps(zcontinuous=1)

	# maxprob
	thetas, maxprob = maxprob_model(sample_results,sps)

	# calculate number of components
	sample_results['ncomp'] = np.max([len(np.atleast_1d(x['init'])) for x in sample_results['model'].config_list if x['isfree'] == True])

    # find SFH parameters that are variables in the chain
    # save their indexes for future use
	str_sfh_parms = ['mass','tau','sf_start','tage']
	parnames = sample_results['model'].theta_labels()
	indexes = []
	for string in str_sfh_parms:
		indexes.append(np.char.find(parnames,string) > -1)
	indexes = np.array(indexes,dtype='bool')

    # initialize output arrays for SFH parameters
	#nchain = sample_results['flatchain'].shape[0]
	nchain = 2000
	half_time,sfr_10,sfr_100,sfr_1000,ssfr_100,totmass,emp_ha,emp_oiii = [np.zeros(shape=(nchain)) for i in range(8)]

    # get constants for SFH calculation [MODEL DEPENDENT]
	z = sample_results['model'].params['zred'][0]
	tuniv = WMAP9.age(z).value
	deltat=[0.01,0.1,1.0] # in Gyr
    
	##### DO EMPIRICAL EMISSION LINES IN CHAIN ######
	dust2_index = np.array([True if (x[:-sample_results['ncomp']] == 'dust2') or 
		                            (x == 'dust2') else 
		                            False for x in parnames])
	dust_index_index = np.array([True if x == 'dust_index' else False for x in parnames])

	# use randomized, flattened, thinned chain for posterior draws
	flatchain = copy(sample_results['flatchain'])
	np.random.shuffle(flatchain)

	######## SFH parameters #########
	for jj in xrange(nchain):
		    
		# extract sfh parameters
		# the ELSE finds SFH parameters that are NOT part of the chain
		sfh_parms = []
		for ll in xrange(len(str_sfh_parms)):
			if np.sum(indexes[ll]) > 0:
				sfh_parms.append(flatchain[jj,indexes[ll]])
			else:
				_ = [x['init'] for x in sample_results['model'].config_list if x['name'] == str_sfh_parms[ll]][0]
				if len(np.atleast_1d(_)) != np.sum(indexes[0]):
					_ = np.zeros(np.sum(indexes[0]))+_
				sfh_parms.append(_)
		mass,tau,sf_start,tage = sfh_parms

		# calculate half-mass assembly time, sfr
		half_time[jj] = halfmass_assembly_time(mass,tage,tau,sf_start,tuniv)

		if np.sum(dust_index_index) > 0:
			dindex = flatchain[jj,dust_index_index]
		else:
			dindex = None

		# empirical halpha
		emp_ha[jj],emp_oiii[jj] = calc_emp_ha(mass,tage,tau,sf_start,tuniv,
			                                  flatchain[jj,dust2_index],dindex,
			                                  ncomp=sample_results['ncomp'])

		# calculate sfr
		sfr_10[jj] = threed_dutils.integrate_sfh(tage-deltat[0],tage,mass,tage,tau,
                                                    sf_start)*np.sum(mass)/(deltat[0]*1e9)
		sfr_100[jj] = threed_dutils.integrate_sfh(tage-deltat[1],tage,mass,tage,tau,
                                                     sf_start)*np.sum(mass)/(deltat[1]*1e9)
		sfr_1000[jj] = threed_dutils.integrate_sfh(tage-deltat[2],tage,mass,tage,tau,
                                                     sf_start)*np.sum(mass)/(deltat[2]*1e9)

		ssfr_100[jj] = sfr_100[jj] / np.sum(mass)

		totmass[jj] = np.sum(mass)

	# CALCULATE Q16,Q50,Q84 FOR VARIABLE PARAMETERS
	ntheta = len(sample_results['initial_theta'])
	q_16, q_50, q_84 = (np.zeros(ntheta)+np.nan for i in range(3))
	for kk in xrange(ntheta): q_16[kk], q_50[kk], q_84[kk] = triangle.quantile(sample_results['flatchain'][:,kk], [0.16, 0.5, 0.84])
	
	# CALCULATE Q16,Q50,Q84 FOR EXTRA PARAMETERS
	extra_flatchain = np.dstack((half_time, sfr_10, sfr_100, sfr_1000, ssfr_100, totmass, emp_ha, emp_oiii))[0]
	nextra = extra_flatchain.shape[1]
	q_16e, q_50e, q_84e = (np.zeros(nextra)+np.nan for i in range(3))
	for kk in xrange(nextra): q_16e[kk], q_50e[kk], q_84e[kk] = triangle.quantile(extra_flatchain[:,kk], [0.16, 0.5, 0.84])

    ######## MODEL CALL PARAMETERS ########
	# set up outputs
	nline = 6 # set by number of lines measured in threed_dutils
	mips_flux = np.zeros(nsamp_mc)
	lineflux = np.empty(shape=(nsamp_mc,nline))
	lir      = np.zeros(nsamp_mc)

	# save initial states
	neb_em = sample_results['model'].params.get('add_neb_emission', np.array(False))
	con_em = sample_results['model'].params.get('add_neb_continuum', np.array(False))
	met_save = sample_results['model'].params.get('logzsol', np.array(0.0))

    # use randomized, flattened, thinned chain for posterior draws
	flatchain = copy(sample_results['flatchain'])
	np.random.shuffle(flatchain)
	for jj in xrange(nsamp_mc):
		thetas = flatchain[jj,:]
		
		# randomly save emline fig
		if jj == 5:
			saveplot=sample_results['run_params']['objname']
		else:
			saveplot=False

		modelout = threed_dutils.measure_emline_lum(sps, thetas = thetas,
			 										model=sample_results['model'], obs = sample_results['obs'],
											        saveplot=saveplot, measure_ir=True)
		
		lineflux[jj,:] = modelout['emline_flux']
		mips_flux[jj]  = modelout['mips']
		lir[jj]        = modelout['lir']

	# restore initial states
	sample_results['model'].params['add_neb_emission'] = neb_em
	sample_results['model'].params['add_neb_continuum'] = con_em
	#sample_results['model'].params['logzsol'] = met_save

	##### MAXIMUM PROBABILITY
	# grab best-fitting model
	thetas, maxprob = maxprob_model(sample_results,sps)

	##### FORMAT EMLINE OUTPUT #####
	q_16em, q_50em, q_84em, thetamaxem = (np.zeros(nline)+np.nan for i in range(4))
	for kk in xrange(nline): q_16em[kk], q_50em[kk], q_84em[kk] = triangle.quantile(lineflux[:,kk], [0.16, 0.5, 0.84])
	emline_info = {'name':modelout['emline_name'],
	               'fluxchain':lineflux,
	               'q16':q_16em,
	               'q50':q_50em,
	               'q84':q_84em}
	sample_results['model_emline'] = emline_info

	###### FORMAT MIPS OUTPUT
	mips = {'mips_flux':mips_flux,'L_IR':lir}
	sample_results['mips'] = mips

	# EXTRA PARAMETER OUTPUTS #
	extras = {'flatchain': extra_flatchain,
			  'parnames': np.array(['half_time','sfr_10','sfr_100','sfr_1000','ssfr_100','totmass','emp_ha','emp_oiii']),
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








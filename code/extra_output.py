from bsfh import model_setup,read_results
import os, threed_dutils, triangle, threedhst_diag, pickle, sys
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from astropy import constants

def calc_emp_ha(mass,sfr,dust1,dust2,dustindex,ncomp=1):

	# dust1 is hardcoded here, be careful!!!!

	ha_flux=0.0
	oiii_flux=0.0
	for kk in xrange(ncomp):
		x=threed_dutils.synthetic_emlines(mass[kk],
				                          np.atleast_1d(sfr)[kk],
				                          dust1[kk],
				                          dust2[kk],
				                          dustindex)
		oiii_flux = oiii_flux + x['flux'][x['name'] == '[OIII]']
		ha_flux = ha_flux + x['flux'][x['name'] == 'Halpha']
	
	return ha_flux,oiii_flux

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
	#np.testing.assert_array_almost_equal(current_maxprob,maxprob,decimal=4)

	return thetas, maxprob


def calc_extra_quantities(sample_results, ncalc=2000):

	'''' 
	CALCULATED QUANTITIES
	model nebular emission line strength
	model star formation history parameters (ssfr,sfr,half-mass time)
	'''

	parnames = sample_results['model'].theta_labels()

	##### modify nebon status
	# we want to be able to turn it on and off at will
	if sample_results['model'].params['add_neb_emission'] == 2:
		sample_results['model'].params['add_neb_emission'] = np.array(True)

	##### initialize sps
	# check to see if we want zcontinuous=2 (i.e., the MDF)
	if np.sum([1 for x in sample_results['model'].config_list if x['name'] == 'pmetals']) > 0:
		sps = threed_dutils.setup_sps(zcontinuous=2,
			                          custom_filter_key=sample_results['run_params'].get('custom_filter_key',None))
		print 'zcontinuous=2'
	else:
		sps = threed_dutils.setup_sps(zcontinuous=1,
			                          custom_filter_key=sample_results['run_params'].get('custom_filter_key',None))
		print 'zcontinuous=1'

	##### maxprob
	# also confirm probability calculations are consistent with fit
	maxthetas, maxprob = maxprob_model(sample_results,sps)

	##### set call parameters
	sample_results['ncomp'] = np.sum(['mass' in x for x in sample_results['model'].theta_labels()])
	deltat=[0.01,0.1,1.0] # for averaging SFR over, in Gyr
	nline = 6 # set by number of lines measured in threed_dutils

    ##### initialize output arrays for SFH + emission line posterior draws #####
	half_time,sfr_10,sfr_100,sfr_1000,ssfr_100,totmass,emp_ha,mips_flux,lir,dust_mass = [np.zeros(shape=(ncalc)) for i in range(10)]
	lineflux = np.empty(shape=(ncalc,nline))

	##### information for empirical emission line calculation ######
	dust1_index = np.array([True if (x[:-sample_results['ncomp']] == 'dust1') or 
		                            (x == 'dust1') else 
		                            False for x in parnames])
	dust2_index = np.array([True if (x[:-sample_results['ncomp']] == 'dust2') or 
		                            (x == 'dust2') else 
		                            False for x in parnames])
	dust_index_index = np.array([True if x == 'dust_index' else False for x in parnames])

	##### use randomized, flattened, thinned chain for posterior draws
	# don't allow things outside the priors
	# make maxprob the first stop
	in_priors = np.isfinite(threed_dutils.chop_chain(sample_results['lnprobability'])) == True
	flatchain = copy(sample_results['flatchain'][in_priors])
	np.random.shuffle(flatchain)
	flatchain[0,:] = maxthetas

	##### set up time vector for full SFHs
	nt = 100
	idx = np.array(sample_results['model'].theta_labels()) == 'tage'
	maxtime = np.max(flatchain[:ncalc,idx])
	t = np.linspace(0,maxtime,num=nt)
	intsfr = np.zeros(shape=(nt,ncalc))

	##### set up model flux vectors
	mags = np.zeros(shape=(len(sample_results['obs']['filters']),ncalc))
	spec = np.zeros(shape=(len(sps.wavelengths),ncalc))

	######## posterior sampling #########
	for jj in xrange(ncalc):
		
		##### model call, to set parameters
		thetas = flatchain[jj,:]
		spec[:,jj],mags[:,jj],sm = sample_results['model'].mean_model(thetas, sample_results['obs'], sps=sps)

		##### extract sfh parameters
		# pass stellar mass to avoid extra model call
		sfh_params = threed_dutils.find_sfh_params(sample_results['model'],flatchain[jj,:],
			                                       sample_results['obs'],sps,sm=sm)
		dust_mass[jj] = sps.dust_mass * sfh_params['mformed']

		##### calculate SFH
		intsfr[:,jj] = threed_dutils.return_full_sfh(t, sfh_params)

		##### solve for half-mass assembly time
		# this is half-time in the sense of integral of SFR, i.e.
		# mass loss is NOT taken into account.
		half_time[jj] = threed_dutils.halfmass_assembly_time(sfh_params,sfh_params['tage'])

		##### calculate time-averaged SFR
		sfr_inst     = threed_dutils.calculate_sfr(sfh_params, 0.00000000001, minsfr=-np.inf, maxsfr=np.inf)
		sfr_10[jj]   = threed_dutils.calculate_sfr(sfh_params, 0.01, minsfr=-np.inf, maxsfr=np.inf)
		sfr_100[jj]  = threed_dutils.calculate_sfr(sfh_params, 0.1,  minsfr=-np.inf, maxsfr=np.inf)
		sfr_1000[jj] = threed_dutils.calculate_sfr(sfh_params, 1.0,  minsfr=-np.inf, maxsfr=np.inf)

		##### calculate mass, sSFR
		totmass[jj] = np.sum(sfh_params['mass'])
		ssfr_100[jj] = sfr_100[jj] / totmass[jj]

		##### empirical halpha
		emp_ha[jj] = threed_dutils.synthetic_halpha(sfr_inst,flatchain[jj,dust1_index],
			                          flatchain[jj,dust2_index],-1.0,
			                          flatchain[jj,dust_index_index],
			                          kriek = (sample_results['model'].params['dust_type'] == 4)[0])


		##### model Halpha, L_IR, and mips flux
		modelout = threed_dutils.measure_emline_lum(sps, thetas = thetas,
			 										model=sample_results['model'], obs = sample_results['obs'],
											        savestr=sample_results['run_params']['objname'], 
											        saveplot=False,measure_ir=True)
		
		lineflux[jj,:] = modelout['emline_flux']
		mips_flux[jj]  = modelout['mips']
		lir[jj]        = modelout['lir']


	##### CALCULATE Q16,Q50,Q84 FOR VARIABLE PARAMETERS
	ntheta = len(sample_results['initial_theta'])
	q_16, q_50, q_84 = (np.zeros(ntheta)+np.nan for i in range(3))
	for kk in xrange(ntheta): q_16[kk], q_50[kk], q_84[kk] = triangle.quantile(sample_results['flatchain'][:,kk], [0.16, 0.5, 0.84])
	
	##### CALCULATE Q16,Q50,Q84 FOR EXTRA PARAMETERS
	extra_flatchain = np.dstack((half_time, sfr_10, sfr_100, sfr_1000, ssfr_100, totmass, emp_ha, dust_mass))[0]
	nextra = extra_flatchain.shape[1]
	q_16e, q_50e, q_84e = (np.zeros(nextra)+np.nan for i in range(3))
	for kk in xrange(nextra): q_16e[kk], q_50e[kk], q_84e[kk] = triangle.quantile(extra_flatchain[:,kk], [0.16, 0.5, 0.84])

	##### FORMAT EMLINE OUTPUT 
	q_16em, q_50em, q_84em, thetamaxem = (np.zeros(nline)+np.nan for i in range(4))
	for kk in xrange(nline): q_16em[kk], q_50em[kk], q_84em[kk] = triangle.quantile(lineflux[:,kk], [0.16, 0.5, 0.84])
	emline_info = {'name':modelout['emline_name'],
	               'fluxchain':lineflux,
	               'q16':q_16em,
	               'q50':q_50em,
	               'q84':q_84em}
	sample_results['model_emline'] = emline_info

	#### EXTRA PARAMETER OUTPUTS 
	extras = {'flatchain': extra_flatchain,
			  'parnames': np.array(['half_time','sfr_10','sfr_100','sfr_1000','ssfr_100','totmass','emp_ha','dust_mass']),
			  'q16': q_16e,
			  'q50': q_50e,
			  'q84': q_84e,
			  'sfh': intsfr,
			  't_sfh': t}
	sample_results['extras'] = extras

	#### OBSERVABLES
	observables = {'spec': spec,
	               'mags': mags,
	               'lam_obs': sps.wavelengths,
	               'L_IR':lir}
	sample_results['observables'] = observables

	#### QUANTILE OUTPUTS #
	quantiles = {'parnames': parnames,
				 'q16':q_16,
				 'q50':q_50,
				 'q84':q_84}
	sample_results['quantiles'] = quantiles

	#### BEST-FITS
	bfit      = {'maxprob_params':maxthetas,
				 'maxprob':maxprob,
	             'emp_ha': emp_ha[0],
	             'sfh': intsfr[:,0],
	             'half_time': half_time[0],
	             'sfr_10': sfr_10[0],
	             'sfr_100':sfr_100[0],
	             'sfr_1000':sfr_1000[0],
	             'lir':lir[0],
	             'mips_flux':mips_flux[0],
	             'halpha_flux':lineflux[0,4],
	             'hbeta_flux':lineflux[0,1],
	             'spec':spec[:,0],
	             'mags':mags[:,0]}
	sample_results['bfit'] = bfit

	print 1/0
	return sample_results

def post_processing(param_name, add_extra=True, **extras):

	'''
	Driver. Loads output, makes all plots for a given galaxy.
	'''
	
	print 'begun post-processing'
	parmfile = model_setup.import_module_from_file(param_name)
	outname = parmfile.run_params['outfile']
	outfolder = os.getenv('APPS')+'/threedhst_bsfh/plots/'+outname.split('/')[-2]+'/'

	# thin and chop the chain?
	thin=1
	chop_chain=1.666

	# make sure the output folder exists
	try:
		os.makedirs(outfolder)
	except OSError:
		pass

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
		sample_results = calc_extra_quantities(sample_results,**extras)
	
		### SAVE OUTPUT HERE
		pickle.dump(sample_results,open(mcmc_filename, "wb"))

	### PLOT HERE
	threedhst_diag.make_all_plots(sample_results=sample_results,filebase=outname,outfolder=outfolder)

if __name__ == "__main__":
	post_processing(sys.argv[1])








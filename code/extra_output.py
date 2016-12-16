from prospect.models import model_setup
from prospect.io import read_results
import os, threed_dutils, pickle, sys
import numpy as np
from copy import copy
from astropy import constants

try:
	import threedhst_diag
except IOError:
	pass

def calc_emp_ha(mass,sfr,dust1,dust2,dustindex,ncomp=1):

	# calculate empirical halpha

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

	### grab maximum probability, plus the thetas that gave it
	maxprob = sample_results['flatprob'].max()
	maxtheta = sample_results['flatchain'][sample_results['flatprob'].argmax()]

	### ensure that maxprob stored is the same as calculated now 
	current_maxprob = threed_dutils.test_likelihood(sps,
		                                            sample_results['model'],
		                                            sample_results['obs'],
		                                            maxtheta,
		                                            sample_results['run_params']['param_file'])

	print 'Probability during sampling: {0}'.format(maxprob)
	print 'Probability right now: {0}'.format(current_maxprob)

	return maxtheta, maxprob

def measure_model_phot(sample_results, flatchain, sps):


	from sedpy.observate import load_filters
	filters = ['bessell_U','bessell_V','twomass_J','bessell_B','bessell_R','twomass_Ks']
	obs = {'filters': load_filters(filters), 'wavelength': None}
	zsave = sample_results['model'].params['zred']
	sample_results['model'].params['zred'] = np.array([0.0])

	ncalc = flatchain.shape[0]
	mags = np.zeros(shape=(len(filters),ncalc))

	for i in xrange(ncalc):
		_,mags[:,i],sm = sample_results['model'].mean_model(flatchain[i,:], obs, sps=sps)
	sample_results['model'].params['zred'] = zsave
	sample_results['mphot'] = {}
	sample_results['mphot']['mags'] = mags
	sample_results['mphot']['name'] = np.array(filters)

	return sample_results

def measure_spire_phot(sample_results, flatchain, sps):

	'''
	for plots for kim-vy tran on 10/26/16
	'''

	from sedpy.observate import load_filters
	filters = ['herschel_spire_250','herschel_spire_350','herschel_spire_500']
	obs = {'filters': load_filters(filters), 'wavelength': None}

	ncalc = flatchain.shape[0]
	mags = np.zeros(shape=(len(filters),ncalc))
	for i in xrange(ncalc):
		_,mags[:,i],sm = sample_results['model'].mean_model(flatchain[i,:], obs, sps=sps)
		print i

	sample_results['hphot'] = {}
	sample_results['hphot']['mags'] = mags
	sample_results['hphot']['name'] = np.array(filters)

	return sample_results

def sample_flatchain(chain, lnprob, parnames, ir_priors=False, include_maxlnprob=True, nsamp=2000):

	'''
	CURRENTLY UNDER DEVELOPMENT
	goal: sample the flatchain in a smart way
	'''

	##### use randomized, flattened chain for posterior draws
	# don't allow draws which are outside the priors
	good = np.isfinite(lnprob) == True

	### cut in IR priors
	if ir_priors:
		gamma_idx = parnames.index('duste_gamma')
		umin_idx = parnames.index('duste_umin')
		qpah_idx = parnames.index('duste_qpah')
		gamma_prior = 0.15
		umin_prior = 15
		qpah_prior = 7
		good = (flatchain[:,gamma_idx] < gamma_prior) & \
		       (flatchain[:,umin_idx] < umin_prior) & \
		       (flatchain[:,qpah_idx] < qpah_prior)
		if good.sum() > ncalc:
			flatchain = flatchain[np.squeeze(good),:]
	flatchain[0,:] = maxthetas

	return sample_idx

def calc_extra_quantities(sample_results, ncalc=2000, ir_priors=True):

	'''' 
	CALCULATED QUANTITIES
	model nebular emission line strength
	model star formation history parameters (ssfr,sfr,half-mass time)
	'''

	parnames = sample_results['model'].theta_labels()

	##### describe number of components in Prospector model [legacy]
	sample_results['ncomp'] = np.sum(['mass' in x for x in sample_results['model'].theta_labels()])

	##### array indexes over which to sample the flatchain
	sample_idx = sample_flatchain(sample_results['flatchain'], sample_results['flatprob'], 
		                          parnames, ir_priors=False, include_maxlnprob=True, nsamp=ncalc)

    ##### initialize output arrays for SFH + emission line posterior draws
	half_time,sfr_10,sfr_100,sfr_1000,ssfr_100,totmass,emp_ha,lir,luv, \
	bdec_cloudy,bdec_calc,ext_5500,dn4000,ssfr_10,xray_lum, luv0 = [np.zeros(shape=(ncalc)) for i in range(16)]
	if 'fagn' in parnames:
		l_agn = np.zeros(shape=(ncalc))

	##### information for empirical emission line calculation ######
	d1_idx = np.array(parnames) == 'dust1'
	d2_idx = np.array(parnames) == 'dust2'
	didx = np.array(parnames) == 'dust_index'

	##### set up time vector for full SFHs
	# if parameterized, calculate linearly in 100 steps from t=0 to t=tage
	# if nonparameterized, calculate at bin edges.
	if 'tage' in sample_results['model'].theta_labels():
		nt = 100
		idx = np.array(sample_results['model'].theta_labels()) == 'tage'
		maxtime = np.max(flatchain[:ncalc,idx])
		t = np.linspace(0,maxtime,num=nt)
	elif 'agebins' in sample_results['model'].params:
		in_years = 10**sample_results['model'].params['agebins']/1e9
		t = np.concatenate((np.ravel(in_years)*0.9999, np.ravel(in_years)*1.001))
		t.sort()
		t = t[1:-1] # remove older than oldest bin, younger than youngest bin
		t = np.clip(t,1e-3,np.inf) # nothing younger than 1 Myr!
	else:
		print 'not sure how to set up the time array here...'
		print 1/0
	intsfr = np.zeros(shape=(t.shape[0],ncalc))

	##### set up model flux vectors
	mags = np.zeros(shape=(len(sample_results['obs']['filters']),ncalc))
	mags_nodust = copy(mags)
	spec = np.zeros(shape=(len(sps.wavelengths),ncalc))

	# sample_flatchain = flatchain[:ncalc,:]
	#sample_results = measure_spire_phot(sample_results, sample_flatchain, sps)

	##### modify nebon status
	# don't cache
	if sample_results['model'].params['add_neb_emission'] == 2:
		sample_results['model'].params['add_neb_emission'] = np.array(True)

	##### initialize sps, calculate maxprob
	# also confirm probability calculations are consistent with fit
	sps = model_setup.load_sps(**sample_results['run_params'])
	maxthetas, maxprob = maxprob_model(sample_results,sps)

	######## posterior sampling #########
	for jj,idx in enumerate(sample_idx):
		
		##### model call, to set parameters
		thetas = copy(sample_flatchain[jj,:])
		spec[:,jj],mags[:,jj],sm = sample_results['model'].mean_model(thetas, sample_results['obs'], sps=sps)

		##### extract sfh parameters
		# pass stellar mass to avoid extra model call
		sfh_params = threed_dutils.find_sfh_params(sample_results['model'],sample_fatchain[jj,:],
			                                       sample_results['obs'],sps,sm=sm)

		##### calculate SFH
		intsfr[:,jj] = threed_dutils.return_full_sfh(t, sfh_params)

		##### solve for half-mass assembly time
		# this is half-time in the sense of integral of SFR, i.e.
		# mass loss is NOT taken into account.
		half_time[jj] = threed_dutils.halfmass_assembly_time(sfh_params)

		##### calculate time-averaged SFR
		sfr_10[jj]   = threed_dutils.calculate_sfr(sfh_params, 0.01, minsfr=-np.inf, maxsfr=np.inf)
		sfr_100[jj]  = threed_dutils.calculate_sfr(sfh_params, 0.1,  minsfr=-np.inf, maxsfr=np.inf)
		sfr_1000[jj] = threed_dutils.calculate_sfr(sfh_params, 1.0,  minsfr=-np.inf, maxsfr=np.inf)

		##### calculate mass, sSFR
		totmass[jj] = sfh_params['mass'].sum()
		ssfr_10[jj] = sfr_10[jj] / totmass[jj]
		ssfr_100[jj] = sfr_100[jj] / totmass[jj]

		##### calculate L_AGN if necessary
		if 'fagn' in parnames:
			l_agn[jj] = threed_dutils.measure_agn_luminosity(thetas[np.array(parnames)=='fagn'],sps,sfh_params['mformed'])
		xray_lum[jj] = threed_dutils.estimate_xray_lum(sfr_100[jj])

		##### empirical halpha
		emp_ha[jj] = threed_dutils.synthetic_halpha(sfr_10[jj],sample_fatchain[jj,d1_idx],
			                          flatchain[jj,d2_idx],-1.0,
			                          flatchain[jj,didx],
			                          kriek = (sample_results['model'].params['dust_type'] == 4)[0])

		##### dust extinction at 5500 angstroms
		ext_5500[jj] = sample_fatchain[jj,d1_idx] + sample_fatchain[jj,d2_idx]

		##### spectral quantities (emission line flux, Balmer decrement, Hdelta absorption, Dn4000)
		##### and magnitudes (LIR, LUV)
		modelout = threed_dutils.measure_emline_lum(sps, thetas = thetas,
			 										model=sample_results['model'], obs = sample_results['obs'],
											        measure_ir=True, measure_luv=True)

		##### Balmer decrements
		bdec_cloudy[jj] = modelout['emlines']['Halpha']['flux'] / modelout['emlines']['Hbeta']['flux']
		bdec_calc[jj] = threed_dutils.calc_balmer_dec(sample_fatchain[jj,d1_idx], sample_fatchain[jj,d2_idx], -1.0, 
			                                          sample_fatchain[jj,didx],
			                                          kriek = (sample_results['model'].params['dust_type'] == 4)[0])
		#### save names!
		if jj == 0:
			emnames = np.array(modelout['emlines'].keys())
			nline = len(emnames)
			emflux = np.empty(shape=(ncalc,nline))
			emeqw = np.empty(shape=(ncalc,nline))

			absnames = np.array(modelout['abslines'].keys())
			nabs = len(absnames)
			absflux = np.empty(shape=(ncalc,nabs))
			abseqw = np.empty(shape=(ncalc,nabs))
			absflux_elines_on = np.empty(shape=(ncalc,nabs))
			abseqw_elines_on = np.empty(shape=(ncalc,nabs))

		absflux_elines_on[jj,:] = np.array([modelout['abslines_elines_on'][line]['flux'] for line in absnames])
		abseqw_elines_on[jj,:] = np.array([modelout['abslines_elines_on'][line]['eqw'] for line in absnames])
		absflux[jj,:]  = np.array([modelout['abslines'][line]['flux'] for line in absnames])
		abseqw[jj,:]  = np.array([modelout['abslines'][line]['eqw'] for line in absnames])
		emflux[jj,:] = np.array([modelout['emlines'][line]['flux'] for line in emnames])
		emeqw[jj,:] = np.array([modelout['emlines'][line]['eqw'] for line in emnames])

		lir[jj]        = modelout['lir']
		luv[jj]        = modelout['luv']
		dn4000[jj]     = modelout['dn4000']

		#### no dust
		nd_thetas = copy(thetas)
		nd_thetas[d1_idx] = np.array([0.0])
		nd_thetas[d2_idx] = np.array([0.0])
		_,mags_nodust[:,jj],sm = sample_results['model'].mean_model(nd_thetas, sample_results['obs'], sps=sps)
		modelout = threed_dutils.measure_emline_lum(sps, thetas = nd_thetas,
			 										model=sample_results['model'], obs = sample_results['obs'],
											        measure_ir=False, measure_luv=True)
		luv0[jj]       = modelout['luv']

	sample_results = measure_model_phot(sample_results, sample_fatchain, sps)

	##### CALCULATE Q16,Q50,Q84 FOR VARIABLE PARAMETERS
	ntheta = len(sample_results['initial_theta'])
	q_16, q_50, q_84 = (np.zeros(ntheta)+np.nan for i in range(3))
	for kk in xrange(ntheta): q_16[kk], q_50[kk], q_84[kk] = np.percentile(sample_fatchain[:,kk], [16.0, 50.0, 84.0])
	
	##### CALCULATE Q16,Q50,Q84 FOR EXTRA PARAMETERS
	extra_flatchain = np.dstack((half_time, sfr_10, sfr_100, sfr_1000, ssfr_10, ssfr_100, totmass, emp_ha, bdec_cloudy,bdec_calc, ext_5500, xray_lum))[0]
	if 'fagn' in parnames:
		extra_flatchain = np.append(extra_flatchain, l_agn[:,None],axis=1)
	nextra = extra_flatchain.shape[1]
	q_16e, q_50e, q_84e = (np.zeros(nextra)+np.nan for i in range(3))
	for kk in xrange(nextra): q_16e[kk], q_50e[kk], q_84e[kk] = np.percentile(extra_flatchain[:,kk], [16.0, 50.0, 84.0])

	##### FORMAT EMLINE OUTPUT 
	q_16flux, q_50flux, q_84flux, q_16eqw, q_50eqw, q_84eqw = (np.zeros(nline)+np.nan for i in range(6))
	for kk in xrange(nline): q_16flux[kk], q_50flux[kk], q_84flux[kk] = np.percentile(emflux[:,kk], [16.0, 50.0, 84.0])
	for kk in xrange(nline): q_16eqw[kk], q_50eqw[kk], q_84eqw[kk] = np.percentile(emeqw[:,kk], [16.0, 50.0, 84.0])
	emline_info = {}
	emline_info['eqw'] = {'chain':emeqw,
						'q16':q_16eqw,
						'q50':q_50eqw,
						'q84':q_84eqw}
	emline_info['flux'] = {'chain':emflux,
						'q16':q_16flux,
						'q50':q_50flux,
						'q84':q_84flux}
	emline_info['emnames'] = emnames
	sample_results['model_emline'] = emline_info

	##### SPECTRAL QUANTITIES
	q_16flux, q_50flux, q_84flux, q_16eqw, q_50eqw, q_84eqw, \
	q_16eflux, q_50eflux, q_84eflux, q_16eeqw, q_50eeqw, q_84eeqw = (np.zeros(nabs)+np.nan for i in range(12))
	for kk in xrange(nabs): q_16eflux[kk], q_50eflux[kk], q_84eflux[kk] = np.percentile(absflux_elines_on[:,kk], [16.0, 50.0, 84.0])
	for kk in xrange(nabs): q_16eeqw[kk], q_50eeqw[kk], q_84eeqw[kk] = np.percentile(abseqw_elines_on[:,kk], [16.0, 50.0, 84.0])
	for kk in xrange(nabs): q_16flux[kk], q_50flux[kk], q_84flux[kk] = np.percentile(absflux[:,kk], [16.0, 50.0, 84.0])
	for kk in xrange(nabs): q_16eqw[kk], q_50eqw[kk], q_84eqw[kk] = np.percentile(abseqw[:,kk], [16.0, 50.0, 84.0])
	q_16dn, q_50dn, q_84dn = np.percentile(dn4000, [16.0, 50.0, 84.0])
	
	spec_info = {}
	spec_info['dn4000'] = {'chain':dn4000,
						   'q16':q_16dn,
						   'q50':q_50dn,
						   'q84':q_84dn}
	spec_info['eqw_elines_on'] = {'chain':abseqw_elines_on,
						'q16':q_16eeqw,
						'q50':q_50eeqw,
						'q84':q_84eeqw}
	spec_info['flux_elines_on'] = {'chain':absflux_elines_on,
						'q16':q_16eflux,
						'q50':q_50eflux,
						'q84':q_84eflux}
	spec_info['eqw'] = {'chain':abseqw,
						'q16':q_16eqw,
						'q50':q_50eqw,
						'q84':q_84eqw}
	spec_info['flux'] = {'chain':absflux,
						'q16':q_16flux,
						'q50':q_50flux,
						'q84':q_84flux}
	spec_info['absnames'] = absnames
	sample_results['spec_info'] = spec_info

	#### EXTRA PARAMETER OUTPUTS 
	extras = {'flatchain': extra_flatchain,
			  'parnames': np.array(['half_time','sfr_10','sfr_100','sfr_1000','ssfr_10','ssfr_100','totmass','emp_ha','bdec_cloudy','bdec_calc','total_ext5500', 'xray_lum']),
			  'q16': q_16e,
			  'q50': q_50e,
			  'q84': q_84e,
			  'sfh': intsfr,
			  't_sfh': t}
	if 'fagn' in parnames:
		extras['parnames'] = np.append(extras['parnames'],np.atleast_1d('l_agn'))
	sample_results['extras'] = extras

	#### OBSERVABLES
	observables = {'spec': spec,
	               'mags': mags,
	               'mags_nodust': mags_nodust,
	               'lam_obs': sps.wavelengths,
	               'L_IR':lir,
	               'L_UV':luv,
	               'L_UV_INTRINSIC':luv0}
	sample_results['observables'] = observables

	#### QUANTILE OUTPUTS #
	quantiles = {'sample_fatchain': sample_fatchain,
				 'parnames': parnames,
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
	             'luv':luv[0],
	             'luv_intrinsic':luv0[0],
	             'halpha_flux':emflux[0,emnames == 'Halpha'],
	             'hbeta_flux':emflux[0,emnames == 'Hbeta'],
	             'hdelta_flux':emflux[0,emnames == 'Hdelta'],
	             'halpha_abs':absflux[0,absnames == 'halpha_wide'],
	             'hbeta_abs':absflux[0,absnames == 'hbeta'],
	             'hdelta_abs':absflux[0,absnames == 'hdelta_wide'],	             
	             'bdec_cloudy':bdec_cloudy[0],
	             'bdec_calc':bdec_calc[0],
	             'dn4000':dn4000[0],
	             'spec':spec[:,0],
	             'mags':mags[:,0],
	             'mags_nodust': mags_nodust[:,0]}
	sample_results['bfit'] = bfit

	return sample_results

def update_all(runname, **kwargs):
	'''
	change some parameters, need to update the post-processing?
	run this!
	'''
	filebase, parm_basename, ancilname=threed_dutils.generate_basenames(runname)
	for param in parm_basename:
		post_processing(param, **kwargs)

def post_processing(param_name, **extras):

	'''
	Driver. Loads output. Creates 
	'''

	from brown_io import load_prospector_data, create_prosp_filename

	# I/O
	parmfile = model_setup.import_module_from_file(param_name)
	outname = parmfile.run_params['outfile']
	outfolder = os.getenv('APPS')+'/threedhst_bsfh/plots/'+outname.split('/')[-2]+'/'

	# check for output folder, create if necessary
	if not os.path.isdir(outfolder):
		os.makedirs(outfolder)

	try:
 		sample_results, powell_results, model = load_prospector_data(outname,hdf5=True)
 	except AttributeError:
 		print 'Failed to load chain for '+sample_results['run_params']['objname']+'. Returning.'
 		return

	print 'Performing post-processing on ' + sample_results['run_params']['objname']

	### create flatchain, run post-processing
	sample_results['flatchain'] = threed_dutils.chop_chain(sample_results['chain'])
	sample_results['flatprob'] = threed_dutils.chop_chain(sample_results['lnprobability'])
	post_processing = calc_extra_quantities(sample_results,**extras)
	
	### create post-processing name, dump info
	mcmc_filename, model_filename, postname = create_prosp_filename(outname)
	pickle.dump(post_processing,open(postname, "wb"))

	### MAKE PLOTS HERE
	try:
		threedhst_diag.make_all_plots(sample_results=sample_results,filebase=outname,outfolder=outfolder,param_name=param_name)
	except NameError:
		print "Unable to make plots for "+sample_results['run_params']['objname']+" due to import error. Passing."
		pass

if __name__ == "__main__":
	post_processing(sys.argv[1])


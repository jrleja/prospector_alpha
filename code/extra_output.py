from prospect.models import model_setup
from prospect.io import read_results
import os, threed_dutils, corner, threedhst_diag, pickle, sys
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from astropy import constants

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

	# grab maximum probability, plus the thetas that gave it
	maxprob = np.max(sample_results['lnprobability'])
	probind = sample_results['lnprobability'] == maxprob
	thetas = sample_results['chain'][probind,:]
	if type(thetas[0]) != np.dtype('float64'):
		thetas = thetas[0]

	# ensure that maxprob stored is the same as calculated now
	current_maxprob = threed_dutils.test_likelihood(sps,sample_results['model'],sample_results['obs'],thetas,sample_results['run_params']['param_file'])
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
	sps = model_setup.load_sps(**sample_results['run_params'])

	##### maxprob
	# also confirm probability calculations are consistent with fit
	maxthetas, maxprob = maxprob_model(sample_results,sps)

	##### set call parameters
	sample_results['ncomp'] = np.sum(['mass' in x for x in sample_results['model'].theta_labels()])
	deltat=[0.01,0.1,1.0] # for averaging SFR over, in Gyr

    ##### initialize output arrays for SFH + emission line posterior draws #####
	half_time,sfr_10,sfr_100,sfr_1000,ssfr_100,totmass,emp_ha,mips_flux,lir, \
	bdec_cloudy,bdec_calc,ext_5500,dn4000,bdec_nodust,ssfr_10 = [np.zeros(shape=(ncalc)) for i in range(15)]
	

	##### information for empirical emission line calculation ######
	dust1_index = np.array([True if (x[:-sample_results['ncomp']] == 'dust1') or 
		                            (x == 'dust1') else 
		                            False for x in parnames])
	dust2_index = np.array([True if (x[:-sample_results['ncomp']] == 'dust2') or 
		                            (x == 'dust2') else 
		                            False for x in parnames])
	dust_index_index = np.array([True if x == 'dust_index' else False for x in parnames])
	met_idx = np.array([True if x == 'logzsol' else False for x in parnames])

	##### use randomized, flattened, thinned chain for posterior draws
	# don't allow things outside the priors
	# make maxprob the first stop
	in_priors = np.isfinite(threed_dutils.chop_chain(sample_results['lnprobability'])) == True
	flatchain = copy(sample_results['flatchain'][in_priors])
	np.random.shuffle(flatchain)
	flatchain[0,:] = maxthetas

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
	else:
		print 'not sure how to set up the time array here...'
		print 1/0
	intsfr = np.zeros(shape=(t.shape[0],ncalc))

	##### set up model flux vectors
	mags = np.zeros(shape=(len(sample_results['obs']['filters']),ncalc))
	spec = np.zeros(shape=(len(sps.wavelengths),ncalc))

	######## posterior sampling #########
	for jj in xrange(ncalc):
		
		##### model call, to set parameters
		thetas = flatchain[jj,:]
		sample_results['model'].params['gas_logz'] = thetas[met_idx]
		spec[:,jj],mags[:,jj],sm = sample_results['model'].mean_model(thetas, sample_results['obs'], sps=sps)

		##### extract sfh parameters
		# pass stellar mass to avoid extra model call
		sfh_params = threed_dutils.find_sfh_params(sample_results['model'],flatchain[jj,:],
			                                       sample_results['obs'],sps,sm=sm)

		##### calculate SFH
		intsfr[:,jj] = threed_dutils.return_full_sfh(t, sfh_params)

		##### solve for half-mass assembly time
		# this is half-time in the sense of integral of SFR, i.e.
		# mass loss is NOT taken into account.
		half_time[jj] = threed_dutils.halfmass_assembly_time(sfh_params,sfh_params['tage'])

		##### calculate time-averaged SFR
		sfr_10[jj]   = threed_dutils.calculate_sfr(sfh_params, 0.01, minsfr=-np.inf, maxsfr=np.inf)
		sfr_100[jj]  = threed_dutils.calculate_sfr(sfh_params, 0.1,  minsfr=-np.inf, maxsfr=np.inf)
		sfr_1000[jj] = threed_dutils.calculate_sfr(sfh_params, 1.0,  minsfr=-np.inf, maxsfr=np.inf)

		##### calculate mass, sSFR
		totmass[jj] = np.sum(sfh_params['mass'])
		ssfr_10[jj] = sfr_10[jj] / totmass[jj]
		ssfr_100[jj] = sfr_100[jj] / totmass[jj]

		##### empirical halpha
		emp_ha[jj] = threed_dutils.synthetic_halpha(sfr_10[jj],flatchain[jj,dust1_index],
			                          flatchain[jj,dust2_index],-1.0,
			                          flatchain[jj,dust_index_index],
			                          kriek = (sample_results['model'].params['dust_type'] == 4)[0])

		##### dust extinction at 5500 angstroms
		ext_5500[jj] = flatchain[jj,dust1_index] + flatchain[jj,dust2_index]

		##### spectral quantities (emission line flux, Balmer decrement, Hdelta absorption, Dn4000)
		##### and magnitudes (L_IR, MIPS)
		modelout = threed_dutils.measure_emline_lum(sps, thetas = thetas,
			 										model=sample_results['model'], obs = sample_results['obs'],
											        #savestr=sample_results['run_params']['objname'], 
											        measure_ir=True)

		##### no dust, to get the intrinsic balmer decrement
		nd_thetas = copy(thetas)
		nd_thetas[dust1_index] = 0.0
		nd_thetas[dust2_index] = 0.0
		modelout_nodust = threed_dutils.measure_emline_lum(sps, thetas = nd_thetas,
			 										       model=sample_results['model'], obs = sample_results['obs'],
											               measure_ir=False)

		##### Balmer decrements
		bdec_cloudy[jj] = modelout['emlines']['Halpha']['flux'] / modelout['emlines']['Hbeta']['flux']
		bdec_calc[jj] = threed_dutils.calc_balmer_dec(flatchain[jj,dust1_index], flatchain[jj,dust2_index], -1.0, 
			                                          flatchain[jj,dust_index_index],
			                                          kriek = (sample_results['model'].params['dust_type'] == 4)[0])
		bdec_nodust[jj] = modelout_nodust['emlines']['Halpha']['flux']  / modelout_nodust['emlines']['Hbeta']['flux']
		
		if jj == 0:
			emnames = np.array(modelout['emlines'].keys())
			nline = len(emnames)
			emflux = np.empty(shape=(ncalc,nline))
			emeqw = np.empty(shape=(ncalc,nline))

			absnames = np.array(modelout['abslines'].keys())
			nabs = len(absnames)
			absflux = np.empty(shape=(ncalc,nabs))
			abseqw = np.empty(shape=(ncalc,nabs))

		absflux[jj,:]  = np.array([modelout['abslines'][line]['flux'] for line in absnames])
		abseqw[jj,:]  = np.array([modelout['abslines'][line]['eqw'] for line in absnames])
		emflux[jj,:] = np.array([modelout['emlines'][line]['flux'] for line in emnames])
		emeqw[jj,:] = np.array([modelout['emlines'][line]['eqw'] for line in emnames])

		mips_flux[jj]  = modelout['mips']
		lir[jj]        = modelout['lir']
		dn4000[jj] = modelout['dn4000']
		print 1/0


	##### CALCULATE Q16,Q50,Q84 FOR VARIABLE PARAMETERS
	ntheta = len(sample_results['initial_theta'])
	q_16, q_50, q_84 = (np.zeros(ntheta)+np.nan for i in range(3))
	for kk in xrange(ntheta): q_16[kk], q_50[kk], q_84[kk] = corner.quantile(sample_results['flatchain'][:,kk], [0.16, 0.5, 0.84])
	
	##### CALCULATE Q16,Q50,Q84 FOR EXTRA PARAMETERS
	extra_flatchain = np.dstack((half_time, sfr_10, sfr_100, sfr_1000, ssfr_10, ssfr_100, totmass, emp_ha, bdec_cloudy,bdec_calc, bdec_nodust,ext_5500))[0]
	nextra = extra_flatchain.shape[1]
	q_16e, q_50e, q_84e = (np.zeros(nextra)+np.nan for i in range(3))
	for kk in xrange(nextra): q_16e[kk], q_50e[kk], q_84e[kk] = corner.quantile(extra_flatchain[:,kk], [0.16, 0.5, 0.84])

	##### FORMAT EMLINE OUTPUT 
	q_16flux, q_50flux, q_84flux, q_16eqw, q_50eqw, q_84eqw = (np.zeros(nline)+np.nan for i in range(6))
	for kk in xrange(nline): q_16flux[kk], q_50flux[kk], q_84flux[kk] = corner.quantile(emflux[:,kk], [0.16, 0.5, 0.84])
	for kk in xrange(nline): q_16eqw[kk], q_50eqw[kk], q_84eqw[kk] = corner.quantile(emeqw[:,kk], [0.16, 0.5, 0.84])
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
	q_16flux, q_50flux, q_84flux, q_16eqw, q_50eqw, q_84eqw = (np.zeros(nabs)+np.nan for i in range(6))
	for kk in xrange(nabs): q_16flux[kk], q_50flux[kk], q_84flux[kk] = corner.quantile(absflux[:,kk], [0.16, 0.5, 0.84])
	for kk in xrange(nabs): q_16eqw[kk], q_50eqw[kk], q_84eqw[kk] = corner.quantile(abseqw[:,kk], [0.16, 0.5, 0.84])
	q_16dn, q_50dn, q_84dn = corner.quantile(dn4000, [0.16, 0.5, 0.84])
	
	spec_info = {}
	spec_info['dn4000'] = {'chain':dn4000,
						   'q16':q_16dn,
						   'q50':q_50dn,
						   'q84':q_84dn}
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
			  'parnames': np.array(['half_time','sfr_10','sfr_100','sfr_1000','ssfr_10','ssfr_100','totmass','emp_ha','bdec_cloudy','bdec_calc','bdec_nodust','total_ext5500']),
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
	             'halpha_flux':emflux[0,emnames == 'Halpha'],
	             'hbeta_flux':emflux[0,emnames == 'Hbeta'],
	             'hdelta_flux':emflux[0,emnames == 'Hdelta'],
	             'halpha_abs':absflux[0,absnames == 'halpha_wide'],
	             'hbeta_abs':absflux[0,absnames == 'hbeta'],
	             'hdelta_abs':absflux[0,absnames == 'hdelta_wide'],	             
	             'bdec_cloudy':bdec_cloudy[0],
	             'bdec_calc':bdec_calc[0],
	             'bdec_nodust':bdec_nodust[0],
	             'dn4000':dn4000[0],
	             'spec':spec[:,0],
	             'mags':mags[:,0]}
	sample_results['bfit'] = bfit

	return sample_results

def update_all(runname):
	'''
	change some parameters, need to update the post-processing?
	run this!
	'''
	filebase, parm_basename, ancilname=threed_dutils.generate_basenames(runname)
	for param in parm_basename:
		post_processing(param)

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

 	sample_results, powell_results, model = threed_dutils.load_prospector_data(outname)

	if add_extra:
		print 'ADDING EXTRA OUTPUT FOR ' + sample_results['run_params']['objname'] + ' in ' + outfolder
		sample_results['flatchain'] = threed_dutils.chop_chain(sample_results['chain'])
		sample_results = calc_extra_quantities(sample_results,**extras)

		### SAVE OUTPUT HERE
		mcmc_filename, model_filename = threed_dutils.create_prosp_filename(outname)
		pickle.dump(sample_results,open(mcmc_filename, "wb"))

	### PLOT HERE
	threedhst_diag.make_all_plots(sample_results=sample_results,filebase=outname,outfolder=outfolder)

if __name__ == "__main__":
	post_processing(sys.argv[1])

def write_kinney_txt():
	
	filebase, parm_basename, ancilname=threed_dutils.generate_basenames('virgo')
	ngals = len(filebase)
	mass, sfr10, sfr100, cloudyha, dmass = [np.zeros(shape=(3,ngals)) for i in xrange(5)]
	names = []
	for jj in xrange(ngals):
		sample_results, powell_results, model = threed_dutils.load_prospector_data(filebase[jj])
		
		if jj == 0:
			sfr10_ind = sample_results['extras']['parnames'] == 'sfr_10'
			sfr100_ind = sample_results['extras']['parnames'] == 'sfr_100'
			mass_ind = sample_results['quantiles']['parnames'] == 'mass'
			ha_ind = sample_results['model_emline']['name'] == 'Halpha'

			nwav = len(sample_results['observables']['lam_obs'])
			nmag = len(sample_results['observables']['mags'])
			lam_obs = np.zeros(shape=(nwav,ngals))
			spec_obs = np.zeros(shape=(4,nwav,ngals)) # q50, q84, q16, best-fit
			mag_obs = np.zeros(shape=(4,nmag,ngals)) # q50, q84, q16, best-fit


		lam_obs[:,jj] = sample_results['observables']['lam_obs']
		for kk in xrange(nwav):
			spec_obs[:,kk,jj] = np.percentile(sample_results['observables']['spec'][kk,:],[5.0,50.0,95.0]).tolist()+\
		                        [sample_results['observables']['spec'][kk,0]]
		for kk in xrange(nmag):
			mag_obs[:,kk,jj] = np.percentile(sample_results['observables']['mags'][kk,:],[5.0,50.0,95.0]).tolist()+\
		                        [sample_results['observables']['mags'][kk,0]]

		mass[:,jj] = [sample_results['quantiles']['q50'][mass_ind],
					  sample_results['quantiles']['q84'][mass_ind],
					  sample_results['quantiles']['q16'][mass_ind]]
		
		sfr10[:,jj] = [sample_results['extras']['q50'][sfr10_ind],
					  sample_results['extras']['q84'][sfr10_ind],
					  sample_results['extras']['q16'][sfr10_ind]]
		
		sfr100[:,jj] = [sample_results['extras']['q50'][sfr100_ind],
					  sample_results['extras']['q84'][sfr100_ind],
					  sample_results['extras']['q16'][sfr100_ind]]
		
		dmass[:,jj] = [sample_results['extras']['q50'][dmass_ind],
					  sample_results['extras']['q84'][dmass_ind],
					  sample_results['extras']['q16'][dmass_ind]]

		cloudyha[:,jj] = [sample_results['model_emline']['q50'][ha_ind],
					  sample_results['model_emline']['q84'][ha_ind],
					  sample_results['model_emline']['q16'][ha_ind]]
		names.append(sample_results['run_params']['objname'])

	outobs = '/Users/joel/code/python/threedhst_bsfh/data/virgo/observables.txt'
	outpars = '/Users/joel/code/python/threedhst_bsfh/data/virgo/parameters.txt'

	# write out observables
	with open(outpars, 'w') as f:
		
		### header ###
		f.write('# name mass mass_errup mass_errdown sfr10 sfr10_errup sfr10_errdown sfr100 sfr100_errup sfr100_errdown dustmass dustmass_errup dustmass_errdown halpha_flux halpha_flux_errup halpha_flux_errdown')
		f.write('\n')

		### data ###
		for jj in xrange(ngals):
			f.write(names[jj]+' ')
			for kk in xrange(3): f.write("{:.2f}".format(mass[kk,jj])+' ')
			for kk in xrange(3): f.write("{:.2e}".format(sfr10[kk,jj])+' ')
			for kk in xrange(3): f.write("{:.2e}".format(sfr100[kk,jj])+' ')
			for kk in xrange(3): f.write("{:.2f}".format(dmass[kk,jj])+' ')
			for kk in xrange(3): f.write("{:.2e}".format(cloudyha[kk,jj])+' ')
			f.write('\n')

	# write out observables
	with open(outobs, 'w') as f:
		
		### header ###
		f.write('# lambda, best-fit spectrum, median spectrum, 84th percentile, 16th percentile, best-fit fluxes, median fluxes, 84th percentile flux, 16th percentile flux')
		for jj in xrange(ngals):
			for kk in xrange(nwav): f.write("{:.1f}".format(lam_obs[kk,jj])+' ')
			f.write('\n')
			for kk in xrange(nwav): f.write("{:.3e}".format(spec_obs[3,kk,jj])+' ')
			f.write('\n')
			for kk in xrange(nwav): f.write("{:.3e}".format(spec_obs[0,kk,jj])+' ')
			f.write('\n')
			for kk in xrange(nwav): f.write("{:.3e}".format(spec_obs[1,kk,jj])+' ')
			f.write('\n')
			for kk in xrange(nwav): f.write("{:.3e}".format(spec_obs[2,kk,jj])+' ')
			f.write('\n')
			
			for kk in xrange(nmag): f.write("{:.3e}".format(mag_obs[3,kk,jj])+' ')
			f.write('\n')
			for kk in xrange(nmag): f.write("{:.3e}".format(mag_obs[0,kk,jj])+' ')
			f.write('\n')
			for kk in xrange(nmag): f.write("{:.3e}".format(mag_obs[1,kk,jj])+' ')
			f.write('\n')
			for kk in xrange(nmag): f.write("{:.3e}".format(mag_obs[2,kk,jj])+' ')
			f.write('\n')




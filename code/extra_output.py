from bsfh import model_setup,read_results
import os, threed_dutils, triangle, threedhst_diag, pickle, sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import WMAP9
from scipy.optimize import brentq
from copy import copy

def sfh_half_time(x,tage,tau,sf_start,tburst,fburst,c):
	 return threed_dutils.integrate_sfh(sf_start,x,tage,tau,sf_start,tburst,fburst)-c

def halfmass_assembly_time(tage,tau,sf_start,tburst,fburst,tuniv):

    # calculate half-mass assembly time
    # c = 0.5 if half-mass assembly time occurs before burst
    half_time = brentq(sfh_half_time, 0,14,
                   args=(tage,tau,sf_start,tburst,fburst,0.5),
                   rtol=1.48e-08, maxiter=100)
    if half_time > tburst:
        
        # if this triggers, make sure to check that 
        # brentq properly found the zero
        c = 0.5+fburst
        
        half_time = brentq(sfh_half_time, 0,20,
                   args=(tage,tau,sf_start,tburst,fburst,c),
                   rtol=1.48e-08, maxiter=100)
        print 1/0
    return tuniv-half_time

def measure_emline_flux(w,spec,z,emline,wavelength,sideband,saveplot=False):
	
	'''
	takes spec(on)-spec(off) to measure emission line flux
	sideband is defined for each emission line after visually 
	inspecting the spectral sampling density around each line

	I've noticed the [SII] line at 6732.71 can be monstrous!
	'''

	emline_flux = np.zeros(len(wavelength))
	
	for jj in xrange(len(wavelength)):
		center = (np.abs(w-wavelength[jj]) == np.min(np.abs(w-wavelength[jj]))).nonzero()[0][0]
		wings = spec[center-sideband[jj]:center+sideband[jj]+1]
		emline_flux[jj] = np.sum(wings[wings>0.01*spec[center]])

		if saveplot and jj==4:
			plt.plot(w,spec,'ro',linestyle='-')
			plt.plot(w[center-sideband[jj]:center+sideband[jj]+1], spec[center-sideband[jj]:center+sideband[jj]+1], 'bo',linestyle=' ')
			plt.xlim(wavelength[jj]-800,wavelength[jj]+800)
			plt.ylim(-spec[center]*0.2,spec[center]*1.2)
			
			plotlines=['[SIII]','[NII]','Halpha','[SII]']
			plotlam  =np.array([6312,6583,6563,6725])*(1+z)
			for kk in xrange(len(plotlam)):
				plt.vlines(plotlam[kk],plt.ylim()[0],plt.ylim()[1],color='0.5',linestyle='--')
				plt.text(plotlam[kk],(plt.ylim()[0]+plt.ylim()[1])/2.*(1.0-kk/6.0),plotlines[kk])
			plt.savefig(os.getenv('APPS')+'/threedhst_bsfh/plots/testem/emline_'+str(saveplot)+'.png',dpi=300)
			plt.close()
	return emline_flux

def calc_extra_quantities(sample_results, nsamp_mc=1000):

    # transform SFH values in chain to half-mass assembly time, SFR
    # write analytical expression, based on tau, tage, etc
    # that returns SFR (== int from tage-(DELTAT_SF) to tage)
    # and half-mass assembly time (== t such that int from zero to t gives int = 0.5, do analytically)
    
    # save nebon/neboff status
	nebstatus = sample_results['model'].params['add_neb_emission']

    # setup parameter background
	str_sfh_parms = ['tau','tburst','fburst','sf_start']
	parnames = sample_results['model'].theta_labels()

    # initialize output arrays for SFH parameters
	nwalkers,niter = sample_results['chain'].shape[:2]
	half_time,sfr_100,ssfr_100 = [np.zeros(shape=(nwalkers,niter)) for i in range(3)]

    # get constants for SFH calculation [MODEL DEPENDENT]
	tage = sample_results['model'].params['tage'][0]
	z = sample_results['model'].params['zred'][0]
	tuniv = WMAP9.age(z).value*1.2
	deltat=0.1 # in Gyr
    
    ######## SFH parameters #########
	for jj in xrange(nwalkers):
		for kk in xrange(niter):
        
			# extract sfh parameters
			sfh_parms = [sample_results['chain'][jj,kk,i] for i in xrange(len(parnames)) if parnames[i] in str_sfh_parms]
			tau,tburst,fburst,sf_start = sfh_parms
	        
			# calculate half-mass assembly time, sfr
			half_time[jj,kk] = halfmass_assembly_time(tage,tau,sf_start,tburst,fburst,tuniv)
	        
			# calculate sfr
			sfr_100[jj,kk] = threed_dutils.integrate_sfh(tage-deltat,tage,tage,tau,
	                                                     sf_start,tburst,fburst)*sample_results['chain'][jj,kk,parnames.index('mass')]/deltat/1e9

			ssfr_100[jj,kk] = sfr_100[jj,kk] / sample_results['chain'][jj,kk,sample_results['model'].theta_labels() == 'mass']
	     
	# CALCULATE Q16,Q50,Q84 FOR VARIABLE PARAMETERS
	ntheta = len(sample_results['initial_theta'])
	q_16, q_50, q_84 = (np.zeros(ntheta)+np.nan for i in range(3))
	for kk in xrange(ntheta): q_16[kk], q_50[kk], q_84[kk] = triangle.quantile(sample_results['flatchain'][:,kk], [0.16, 0.5, 0.84])

	# CALCULATE Q16,Q50,Q84 FOR EXTRA PARAMETERS
	extra_chain = np.dstack((half_time.reshape(half_time.shape[0],half_time.shape[1],1), 
		                     sfr_100.reshape(sfr_100.shape[0],sfr_100.shape[1],1), 
		                     ssfr_100.reshape(ssfr_100.shape[0],ssfr_100.shape[1],1)))
	extra_flatchain = threed_dutils.chop_chain(extra_chain)
	nextra = extra_chain.shape[-1]
	q_16e, q_50e, q_84e = (np.zeros(nextra)+np.nan for i in range(3))
	for kk in xrange(nextra): q_16e[kk], q_50e[kk], q_84e[kk] = triangle.quantile(extra_flatchain[:,kk], [0.16, 0.5, 0.84])

    ######## MODEL CALL PARAMETERS ########
    # measure emission lines
	emline = ['[OII]','Hbeta','[OIII]1','[OIII]2','Halpha','[SII]']
	wavelength = np.array([3728,4861.33,4959,5007,6562,6732.71])*(1+z)
	sideband   = np.array([20,1,1,1,2,1])
	nline = len(emline)

	# initialize sps
	sps = threed_dutils.setup_sps()

    # first randomize
    # use flattened and thinned chain for random posterior draws
	flatchain = copy(sample_results['flatchain'])
	lineflux = np.empty(shape=(nsamp_mc,nline))
	np.random.shuffle(flatchain)
	for jj in xrange(nsamp_mc):
		thetas = flatchain[jj,:]
		sample_results['model'].params['add_neb_emission'] = np.array(True)
		spec,mags,w = sample_results['model'].mean_model(thetas, sample_results['obs'], sps=sps, norm_spec=True)
		sample_results['model'].params['add_neb_emission'] = np.array(False)
		spec_neboff,mags_neboff,w = sample_results['model'].mean_model(thetas, sample_results['obs'], sps=sps, norm_spec=True)

		# randomly save emline fig
		if jj == 5:
			saveplot=sample_results['run_params']['objname']
		else:
			saveplot=False

		lineflux[jj]= measure_emline_flux(w,spec-spec_neboff,z,emline,wavelength,sideband,saveplot=saveplot)

	##### MAXIMUM PROBABILITY
	# grab best-fitting model
	maxprob = np.max(sample_results['lnprobability'])
	probind = sample_results['lnprobability'] == maxprob
	thetas = sample_results['chain'][probind,:]
	if type(thetas[0]) != np.dtype('float64'):
		thetas = thetas[0]
	maxhalf_time,maxsfr_100,maxssfr_100 = extra_chain[probind.nonzero()[0][0],probind.nonzero()[1][0],:]

	# grab most likely emlines
	sample_results['model'].params['add_neb_emission'] = np.array(True)
	spec,mags,w = sample_results['model'].mean_model(thetas, sample_results['obs'], sps=sps, norm_spec=True)
	sample_results['model'].params['add_neb_emission'] = np.array(False)
	spec_neboff,mags_neboff,w = sample_results['model'].mean_model(thetas, sample_results['obs'], sps=sps, norm_spec=True)
	lineflux_maxprob = measure_emline_flux(w,spec-spec_neboff,z,emline,wavelength,sideband)

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

	# EXTRA PARAMETER OUTPUTS #
	extras = {'chain': extra_chain,
			  'flatchain': extra_flatchain,
			  'parnames': np.array(['half_time','sfr_100','ssfr_100']),
			  'maxprob': np.array([maxhalf_time,maxsfr_100,maxssfr_100]),
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
	
	parmfile = model_setup.import_module_from_file(param_name)
	outname = parmfile.run_params['outfile']
	outfolder = os.getenv('APPS')+'/threedhst_bsfh/plots/'+outname.split('/')[-2]+'/'

	# thin and chop the chain?
	thin=1
	chop_chain=1.666

	# make sure the output folder exists
	if not os.path.isdir(outfolder):
		os.makedirs(outfolder)

	# find most recent output file
	# with the objname
	folder = "/".join(outname.split('/')[:-1])
	filename = outname.split("/")[-1]
	files = [f for f in os.listdir(folder) if "_".join(f.split('_')[:-2]) == filename]	
	times = [f.split('_')[-2] for f in files]

	# if we found no files, skip this object
	if len(times) == 0:
		print 'Failed to find any files to extract times in ' + folder + ' of form ' + filename
		return 0

	# load results
	mcmc_filename=outname+'_'+max(times)+"_mcmc"
	model_filename=outname+'_'+max(times)+"_model"
	
	try:
		sample_results, powell_results, model = read_results.read_pickles(mcmc_filename, model_file=model_filename,inmod=None)
	except:
		print 'Failed to open '+ mcmc_filename +','+model_filename
		return 0
	
	
	if add_extra:
		print 'ADDING EXTRA OUTPUT FOR ' + filename + ' in ' + outfolder
		sample_results['flatchain'] = threed_dutils.chop_chain(sample_results['chain'])
		sample_results = calc_extra_quantities(sample_results,nsamp_mc=nsamp_mc)
	
		### SAVE OUTPUT HERE
		pickle.dump(sample_results,open(mcmc_filename, "wb"))

	### PLOT HERE
	threedhst_diag.make_all_plots(sample_results=sample_results,filebase=outname)

if __name__ == "__main__":
	post_processing(sys.argv[1])








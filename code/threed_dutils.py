import numpy as np
import os, fsps
import matplotlib.pyplot as plt
from bsfh import model_setup
from scipy.interpolate import interp1d
from scipy.integrate import simps
from calc_ml import load_filter_response
from bsfh.likelihood import LikelihoodFunction
from astropy.cosmology import WMAP9
import copy

def return_lir(lam,spec,z=None,alt_file=None):

	# integrates input over wavelength
	# input must be Lsun / hz
	# returns erg/s

	# fake LIR filter
	# 8-1000 microns
	# note that lam must be in angstroms
	botlam = np.atleast_1d(8e4-1)
	toplam = np.atleast_1d(1000e4+1)
	edgetrans = np.atleast_1d(0)
	lir_filter = [[np.concatenate((botlam,np.linspace(8e4, 1000e4, num=100),toplam))],
	              [np.concatenate((edgetrans,np.ones(100),edgetrans))]]

	# calculate integral
	_,lir     = integrate_mag(lam,spec,lir_filter, z=z, alt_file=alt_file) # comes out in ergs/s

	return lir

def return_luv(lam,spec,z=None,alt_file=None):

	# integrates input over wavelength
	# input must be Lsun / hz
	# returns erg/s

	# fake LUV filter
	# over 1216-3000 angstroms
	# note that lam must be in angstroms
	botlam = np.atleast_1d(1216)
	toplam = np.atleast_1d(3000)
	edgetrans = np.atleast_1d(0)
	luv_filter =  [[np.concatenate((botlam-1,np.linspace(botlam, toplam, num=100),toplam+1))],
	               [np.concatenate((edgetrans,np.ones(100),edgetrans))]]

	# calculate integral
	_,luv     = integrate_mag(lam,spec,luv_filter, z=z, alt_file=alt_file) # comes out in ergs/s

	return luv

def mips_to_lir(mips_flux,z):

	'''
	input flux must be in mJy
	output is in Lsun
	L_IR [Lsun] = fac_<band>(redshift) * flux [milliJy]
	'''

	dale_helou_txt = '/Users/joel/code/python/threedhst_bsfh/data/MIPS/dale_helou.txt'
	with open(dale_helou_txt, 'r') as f: hdr = f.readline().split()[1:]
	conversion = np.loadtxt(dale_helou_txt, comments = '#', dtype = np.dtype([(n, np.float) for n in hdr]))
	
	# if we're at higher redshift, interpolate
	# it decrease error due to redshift relative to rest-frame template (good)
	# but adds nonlinear error due to distances (bad)
	# else, scale the nearest conversion factor by the 
	# ratio of luminosity distances, since nonlinear error due to distances will dominate
	if z > 0.1:
		intfnc = interp1d(conversion['redshift'],conversion['fac_MIPS24um'], bounds_error = True, fill_value = 0)
		fac = intfnc(z)
	else:
		near_idx = np.abs(conversion['redshift']-z).argmin()
		lumdist_ratio = (WMAP9.luminosity_distance(z).value / WMAP9.luminosity_distance(conversion['redshift'][near_idx]).value)**2
		zfac_ratio = (1.+conversion['redshift'][near_idx]) / (1.+z)
		fac = conversion['fac_MIPS24um'][near_idx]*lumdist_ratio*zfac_ratio

	return fac*mips_flux

def sfr_uvir(lir,luv):

	# inputs in Lsun
	# from Whitaker+14
	# output is Msun/yr, in Chabrier IMF
	return 1.09e-10*(lir + 2.2*luv)

def smooth_spectrum(lam,spec,sigma,
	                minlam=0.0,maxlam=1e50):     

	'''
	ripped from Charlie Conroy's smoothspec.f90
	the 'fast way'
	integration is truncated at +/-4*sigma
	'''
	c_kms = 2.99e5
	int_trunc=4
	spec_out = copy.copy(spec)

	for ii in xrange(len(lam)):
		if lam[ii] < minlam or lam[ii] > maxlam:
			spec_out[ii] = spec[ii]
			continue

		dellam = lam[ii]*(int_trunc*sigma/c_kms+1)-lam[ii]
		integrate_lam = (lam > lam[ii]-dellam) & (lam < lam[ii]+dellam)

		if np.sum(integrate_lam) <= 1:
			spec_out[ii] = spec[ii]
		else:
			vel = (lam[ii]/lam[integrate_lam]-1)*c_kms
			func = 1/np.sqrt(2*np.pi)/sigma * np.exp(-vel**2/2./sigma**2)
			dx=np.abs(np.diff(vel)) # we want absolute value
			func = func / np.trapz(func,dx=dx)
			spec_out[ii] = np.trapz(func*spec[integrate_lam],dx=dx)

	return spec_out

def offset_and_scatter(x,y,biweight=True):

	n = len(x)
	mean_offset = np.nanmean(x-y)

	if biweight:
		diff = y-x
		Y0  = np.median(diff)

		# calculate MAD
		MAD = np.median(np.abs(diff-Y0))/0.6745

		# biweighted value
		U   = (diff-Y0)/(6.*MAD)
		UU  = U*U
		Q   = UU <= 1.0
		if np.sum(Q) < 3:
			print 'distribution is TOO WEIRD, returning -1'
			scat=-1

		N = len(diff)
		numerator = np.sum( (diff[Q]-Y0)**2 * (1-UU[Q])**4)
		den1      = np.sum( (1.-UU[Q])*(1.-5.*UU[Q]))
		siggma    = N*numerator/(den1*(den1-1.))

		scat      = np.sqrt(siggma)

	else:
		scat=np.sqrt(np.sum((x-y-mean_offset)**2.)/(n-2))

	return mean_offset,scat

def find_sfh_params(model,theta,obs,sps,sm=None):

	str_sfh_parms = ['sfh','mass','tau','sf_start','tage','sf_trunc','sf_slope']
	parnames = model.theta_labels()
	sfh_out = []

	# set parameters, in case of dependencies
	model.set_parameters(theta)

	for string in str_sfh_parms:
		
		# find SFH parameters that are variables in the chain
		index = np.char.find(parnames,string) > -1

		# if not found, look in fixed parameters
		if np.sum(index) == 0:
			sfh_out.append(np.atleast_1d(model.params.get(string,0.0)))
		else:
			sfh_out.append(theta[index])

	iterable = [(str_sfh_parms[ii],sfh_out[ii]) for ii in xrange(len(sfh_out))]
	out = {key: value for (key, value) in iterable}

	# Need this because mass is 
	# current mass, not total mass formed!
	
	# if we pass sm from a prior model call,
	# we don't have to calculate it here
	if sm is None:
		_,_,_=model.sed(theta, obs, sps=sps)
		sm = sps.stellar_mass
	out['mformed'] = out['mass'] / sm

	return out

def test_likelihood(param_file=None, sps=None, model=None, obs=None, thetas=None, verbose=False):

	'''
	skeleton:
	load up some model, instantiate an sps, and load some observations
	generate spectrum, compare to observations, assess likelihood
	can be run in different environments as a test
	'''

	if param_file is None:
		param_file = os.getenv('APPS')+'/threedhst_bsfh/parameter_files/testsed_simha/testsed_simha_params.py'

	if sps is None:
		# load stellar population, set up custom filters
		sps = fsps.StellarPopulation(zcontinuous=2, compute_vega_mags=False)
		custom_filter_keys = os.getenv('APPS')+'/threedhst_bsfh/filters/filter_keys_threedhst.txt'
		fsps.filters.FILTERS = model_setup.custom_filter_dict(custom_filter_keys)

	if model is None:
		model = model_setup.load_model(param_file)
	
	if obs is None:
		run_params = model_setup.get_run_params(param_file=param_file)
		obs = model_setup.load_obs(**run_params)

	if thetas is None:
		thetas = np.array(model.initial_theta)

	# setup gp
	from bsfh import gp
	gp_phot = gp.PhotOutlier()
	try:
		s, a, l = model.phot_gp_params(obs=obs)
		print 'phot gp parameters'
		print s, a, l
		gp_phot.kernel = np.array( list(a) + list(l) + [s])
	except(AttributeError):
		#There was no phot_gp_params method
		pass

	likefn = LikelihoodFunction(obs=obs, model=model)
	mu, phot, x = model.mean_model(thetas, obs, sps = sps)
	lnp_phot = likefn.lnlike_phot(phot, obs=obs, gp=gp_phot)
	lnp_prior = model.prior_product(thetas)

	if verbose:
		print 'photometry:'
		print phot
		print 'phot likelihood, prior likelihood'
		print lnp_phot,lnp_prior

	return lnp_phot + lnp_prior

def setup_sps(zcontinuous=2,compute_vega_magnitudes=False,custom_filter_key=None):

	'''
	easy way to define an SPS
	must rewrite filter_key functionality after update to most recent python-fsps filter functionality (8/3/15)
	'''

	# load stellar population, set up custom filters
	sps = fsps.StellarPopulation(zcontinuous=zcontinuous, compute_vega_mags=compute_vega_magnitudes)
	if custom_filter_key is not None:
		# os.getenv('APPS')+'/threedhst_bsfh/filters/filter_keys_threedhst.txt'
		fsps.filters.FILTERS = model_setup.custom_filter_dict(custom_filter_key)

	return sps

def synthetic_halpha(sfr,dust1,dust2,dust1_index,dust2_index,kriek=False):

	'''
	SFR in Msun/yr
	mass in Msun
	'''

	# calculate Halpha luminosity from Kennicutt relationship
	# comes out in units of [ergs/s]
	# correct from Chabrier to Salpeter with a factor of 1.7
	flux = 1.26e41 * (sfr*1.7)
	lam     = 6563.0

	# correct for dust
	# if dust_index == None, use Calzetti
	if dust2_index is not None:
		flux=flux*charlot_and_fall_extinction(lam,dust1,dust2,dust1_index,dust2_index,kriek=kriek)
	else:
		Rv   = 4.05
		klam = 2.659*(-2.156+1.509/(lam/1e4)-0.198/(lam/1e4)**2+0.011/(lam/1e4)**3)+Rv
		A_lam = klam/Rv*dust2

		flux = flux*10**(-0.4*A_lam)

	# comes out in ergs/s
	return flux

def charlot_and_fall_extinction(lam,dust1,dust2,dust1_index,dust2_index, kriek=False, nobc=False, nodiff=False):

	dust1_ext = np.exp(-dust1*(lam/5500.)**dust1_index)
	dust2_ext = np.exp(-dust2*(lam/5500.)**dust2_index)

	# sanitize inputs
	lam = np.atleast_1d(lam)

	# are we using Kriek & Conroy 13?
	if kriek == True:
		dd63=6300.00
		lamv=5500.0
		dlam=350.0
		lamuvb=2175.0

		#Calzetti curve, below 6300 Angstroms, else no addition
		cal00 = np.zeros_like(lam)
		gt_dd63 = lam > dd63
		le_dd63 = ~gt_dd63
		if np.sum(gt_dd63) > 0:
			cal00[gt_dd63] = 1.17*( -1.857+1.04*(1e4/lam[gt_dd63]) ) + 1.78
		if np.sum(le_dd63) > 0:
			cal00[le_dd63]  = 1.17*(-2.156+1.509*(1e4/lam[le_dd63])-0.198*(1E4/lam[le_dd63])**2 + 0.011*(1E4/lam[le_dd63])**3) + 1.78
		cal00 = cal00/0.44/4.05 

		eb = 0.85 - 1.9 * dust2_index  #KC13 Eqn 3

		#Drude profile for 2175A bump
		drude = eb*(lam*dlam)**2 / ( (lam**2-lamuvb**2)**2 + (lam*dlam)**2 )

		attn_curve = dust2*(cal00+drude/4.05)*(lam/lamv)**dust2_index
		dust2_ext = np.exp(-attn_curve)

	if nobc:
		ext_tot = dust2_ext
	elif nodiff:
		ext_tot = dust1_ext
	else:
		ext_tot = dust2_ext*dust1_ext

	return ext_tot


def calc_balmer_dec(tau1, tau2, ind1, ind2,kriek=False):

	ha_lam = 6562.801
	hb_lam = 4861.363
	balm_dec = 2.86*charlot_and_fall_extinction(ha_lam,tau1,tau2,ind1,ind2,kriek=kriek) / \
	                charlot_and_fall_extinction(hb_lam,tau1,tau2,ind1,ind2,kriek=kriek)
	return balm_dec

def sfh_half_time(x,sfh_params,c):

	'''
	wrapper for use with halfmass assembly time
	'''

	return integrate_sfh(sfh_params['sf_start'],x,sfh_params)-c

def halfmass_assembly_time(sfh_params,tuniv):

	from scipy.optimize import brentq

	# calculate half-mass assembly time
	# c = 0.5 if half-mass assembly time occurs before burst
	try:
		half_time = brentq(sfh_half_time, 0,14,
	                       args=(sfh_params,0.5),
	                       rtol=1.48e-08, maxiter=100)
	except ValueError:
		
		# make error only pop up once
		import warnings
		warnings.simplefilter('once', UserWarning)

		# big problem
		warnings.warn("You've passed SFH parameters that don't allow t_half to be calculated. Check for bugs.", UserWarning)
		half_time = np.nan

	return tuniv-half_time

def load_truths(truthname,objname,sample_results, sps=None, calc_prob = True):

	'''
	loads truths
	generates plotvalues
	'''

	# load truths
	with open(truthname, 'r') as f:
		hdr = f.readline().split()[1:]
	truth = np.loadtxt(truthname, comments = '#',
					   dtype = np.dtype([(n, np.float) for n in hdr]))

	truths = np.array([x for x in truth[int(objname)-1]])
	parnames = np.array(hdr)

	#### define extra parameters ####
	mass = truths[np.array([True if 'mass' in x else False for x in parnames])]

	# SFH parameters
	if sample_results is not None:
		sfh_params = find_sfh_params(sample_results['model'],truths,sample_results['obs'],sps)
		deltat=0.1
		sfr_10  = np.log10(calculate_sfr(sfh_params,0.01))
		ssfr_10 = np.log10(10**sfr_10 / mass)
		sfr_100  = np.log10(calculate_sfr(sfh_params,deltat))
		ssfr_100 = np.log10(calculate_sfr(sfh_params,deltat) / mass)
		halftime = halfmass_assembly_time(sfh_params,sfh_params['tage'])
		if calc_prob == True:
			lnprob   = test_likelihood(sps=sps,model=sample_results['model'], obs=sample_results['obs'],thetas=truths)

		modelout = measure_emline_lum(sps, thetas = truths,
	 								  model=sample_results['model'], obs = sample_results['obs'],
									  saveplot=False,measure_ir=True)
		emnames = np.array(modelout['emlines'].keys())
		emflux = np.array([modelout['emlines'][line]['flux'] for line in emnames])
		absnames = np.array(modelout['abslines'].keys())
		abseqw = np.array([modelout['abslines'][line]['eqw'] for line in absnames])
		dn4000 = modelout['dn4000']

	else:
		sfh_params = None
		sfr_100 = None
		ssfr_100 = None
		halftime = None
		lnprob   = None
    

	#### parameter conversions for plotting
	plot_truths = truths+0.0
	for kk in xrange(len(parnames)):
		if parnames[kk] == 'mass':
			plot_truths[kk] = np.log10(plot_truths[kk])

	truths_dict = {'parnames':parnames,
				   'truths':truths,
				   'plot_truths':plot_truths,
				   'extra_parnames':np.array(['sfr_10','ssfr_10','sfr_100','ssfr_100','half_time']),
				   'extra_truths':np.array([sfr_10,ssfr_10,sfr_100,ssfr_100,halftime]),
				   'sfh_params': sfh_params,
				   'emnames':emnames,
				   'emflux':emflux,
				   'absnames':absnames,
				   'abseqw':abseqw,
				   'dn4000':dn4000}
	
	if calc_prob == True:
		truths_dict['truthprob'] = lnprob

	return truths_dict

def running_median(x,y,nbins=10,avg=False):

	bins = np.linspace(x.min(),x.max(), nbins)
	delta = bins[1]-bins[0]
	idx  = np.digitize(x,bins)
	if avg == False:
		running_median = np.array([np.median(y[idx-1==k]) for k in range(nbins)])
	else:
		running_median = np.array([np.mean(y[idx-1==k]) for k in range(nbins)])
	bins = bins+delta/2.

	# remove empty
	empty = np.isnan(running_median) == 1
	running_median[empty] = 0

	return bins,running_median

def generate_basenames(runname):

	filebase=[]
	parm=[]
	ancilname='COSMOS_testsamp.dat'

	if runname == 'testsed_nonoise':

		id_list = os.getenv('APPS')+"/threedhst_bsfh/data/testsed_nonoise.ids"
		ids = np.loadtxt(id_list, dtype='|S20')
		ngals = len(ids)

		basename = "testsed_nonoise"
		parm_basename = "testsed_nonoise_params"
		ancilname=None

		for jj in xrange(ngals):
			filebase.append(os.getenv('APPS')+"/threedhst_bsfh/results/"+runname+'/'+basename+'_'+ids[jj])
			parm.append(os.getenv('APPS')+"/threedhst_bsfh/parameter_files/"+runname+'/'+parm_basename+'_'+str(jj+1)+'.py')	

	elif runname == 'brownseds':

		id_list = os.getenv('APPS')+'/threedhst_bsfh/data/brownseds_data/photometry/namelist.txt'
		ids = np.loadtxt(id_list, dtype='|S20',delimiter=',')
		ngals = len(ids)

		basename = "brownseds"
		parm_basename = basename+"_params"
		ancilname=None

		for jj in xrange(ngals):
			filebase.append(os.getenv('APPS')+"/threedhst_bsfh/results/"+runname+'/'+basename+'_'+ids[jj])
			parm.append(os.getenv('APPS')+"/threedhst_bsfh/parameter_files/"+runname+'/'+parm_basename+'_'+str(jj+1)+'.py')	

	elif runname == 'brownseds_tightbc':

		id_list = os.getenv('APPS')+'/threedhst_bsfh/data/brownseds_data/photometry/namelist.txt'
		ids = np.loadtxt(id_list, dtype='|S20',delimiter=',')
		ngals = len(ids)

		basename = runname
		parm_basename = basename+"_params"
		ancilname=None

		for jj in xrange(ngals):
			filebase.append(os.getenv('APPS')+"/threedhst_bsfh/results/"+runname+'/'+basename+'_'+ids[jj])
			parm.append(os.getenv('APPS')+"/threedhst_bsfh/parameter_files/"+runname+'/'+parm_basename+'_'+str(jj+1)+'.py')	


	elif 'simha' in runname:

		id_list = os.getenv('APPS')+"/threedhst_bsfh/data/testsed_simha.ids"
		ids = np.loadtxt(id_list, dtype='|S20')
		ngals = len(ids)

		parm_basename = runname+"_params"
		ancilname=None

		for jj in xrange(ngals):
			filebase.append(os.getenv('APPS')+"/threedhst_bsfh/results/"+runname+'/'+runname+'_'+ids[jj])
			parm.append(os.getenv('APPS')+"/threedhst_bsfh/parameter_files/"+runname+'/'+parm_basename+'_'+str(jj+1)+'.py')	

	elif 'virgo' in runname:

		id_list = os.getenv('APPS')+"/threedhst_bsfh/data/virgo/names.txt"
		ids = np.loadtxt(id_list, dtype='|S20')
		ngals = len(ids)

		parm_basename = runname+"_params"
		ancilname=None

		for jj in xrange(ngals):
			filebase.append(os.getenv('APPS')+"/threedhst_bsfh/results/"+runname+'/'+runname+'_'+ids[jj])
			parm.append(os.getenv('APPS')+"/threedhst_bsfh/parameter_files/"+runname+'/'+parm_basename+'_'+str(jj+1)+'.py')	


	else:

		id_list = os.getenv('APPS')+"/threedhst_bsfh/data/"+runname+".ids"
		ids = np.loadtxt(id_list, dtype='|S20')
		ngals = len(ids)

		parm_basename = runname+"_params"
		ancilname=None

		for jj in xrange(ngals):
			filebase.append(os.getenv('APPS')+"/threedhst_bsfh/results/"+runname+'/'+runname+'_'+ids[jj])
			parm.append(os.getenv('APPS')+"/threedhst_bsfh/parameter_files/"+runname+'/'+parm_basename+'_'+str(jj+1)+'.py')	

	return filebase,parm,ancilname

def chop_chain(chain):
	'''
	simple placeholder
	will someday replace with a test for convergence to determine where to chop
	JRL 1/5/15

	... haha
	JRL 6/8/15
	'''
	nchop=1.66

	if len(chain.shape) == 3:
		flatchain = chain[:,int(chain.shape[1]/nchop):,:]
		flatchain = flatchain.reshape(flatchain.shape[0] * flatchain.shape[1],
		                              flatchain.shape[2])
	else:
		flatchain = chain[:,int(chain.shape[1]/nchop):]
		flatchain = flatchain.reshape(flatchain.shape[0] * flatchain.shape[1])

	return flatchain


def return_mwave_custom(filters):

	"""
	returns effective wavelength based on filter names
	"""

	loc = os.getenv('APPS')+'/threedhst_bsfh/filters/'
	key_str = 'filter_keys_threedhst.txt'
	lameff_str = 'lameff_threedhst.txt'
	
	lameff = np.loadtxt(loc+lameff_str)
	keys = np.loadtxt(loc+key_str, dtype='S20',usecols=[1])
	keys = keys.tolist()
	keys = np.array([keys.lower() for keys in keys], dtype='S20')
	
	lameff_return = [[lameff[keys == filters[i]]][0] for i in range(len(filters))]
	lameff_return = [item for sublist in lameff_return for item in sublist]
	assert len(filters) == len(lameff_return), "Filter name is incorrect"

	return lameff_return

def load_moustakas_data(objnames = None):

	'''
	specifically written to load optical emission line fluxes, of the "radial strip" variety
	this corresponds to the aperture used in the Brown sample

	if we pass a list of object names, return a sorted, matched list
	otherwise return everything

	returns in units of 10^-15^erg/s/cm^2
	'''

	#### load data
	# arcane vizier formatting means I'm using astropy tables here
	from astropy.io import ascii
	foldername = os.getenv('APPS')+'/threedhst_bsfh/data/Moustakas+10/'
	filename = 'table3.dat'
	readme = 'ReadMe'
	table = ascii.read(foldername+filename, readme=foldername+readme)

	#### filter to only radial strips
	accept = table['Spectrum'] == 'Radial Strip'
	table = table[accept.data]

	#####
	if objnames is not None:
		outtable = []
		for name in objnames:
			match = table['Name'] == name
			if np.sum(match.data) == 0:
				outtable.append(None)
				continue
			else:
				outtable.append(table[match.data])
	else:
		outtable = table

	return outtable

def load_moustakas_newdat(objnames = None):

	'''
	access (new) Moustakas line fluxes, from email in Jan 2016

	if we pass a list of object names, return a sorted, matched list
	otherwise return everything

	returns in units of erg/s/cm^2
	'''

	#### load data
	from astropy.io import fits
	filename = os.getenv('APPS')+'/threedhst_bsfh/data/Moustakas_new/atlas_specdata_solar_drift_v1.1.fits'
	hdulist = fits.open(filename)

	##### match
	if objnames is not None:
		outtable = []
		objnames = np.core.defchararray.replace(objnames, ' ', '')  # strip spaces
		for name in objnames:
			match = hdulist[1].data['GALAXY'] == name
			if np.sum(match.data) == 0:
				outtable.append(None)
				continue
			else:
				outtable.append(hdulist[1].data[match])
	else:
		outtable = hdulist.data

	return outtable

def asym_errors(center, up, down, log=False):

	if log:
		errup = np.log10(up)-np.log10(center)
		errdown = np.log10(center)-np.log10(down)
		errarray = [errdown,errup]
	else:
		errarray = [center-down,up-center]

	return errarray

def return_n_outliers(x,y,objnames,noutliers,
	                  outfolder='/Users/joel/code/python/threedhst_bsfh/plots/brownseds_tightbc/outliers',
	                  cp_files=None):

	'''
	for observed parameter x versus predicted parameter y, will choose the top N_OUTLIERS (in an absolute sense)
	puts all diagnostic plots for these outliers in a folder
	and adds a text file with the x+y values and the object name
	'''

	residual = np.abs(x - y)
	indices = np.argpartition(residual, -noutliers)[-noutliers:]
	indices = indices[np.argsort(residual[indices])]

	# create the destination folder
	# and put diagnostic plots there
	# also create a text file 
	# x is observed, y is model
	if cp_files is not None:
		### create destinations if they don't exist
		destination = outfolder+'/'+cp_files
		if not os.path.isdir(outfolder):
			os.makedirs(outfolder)
		if not os.path.isdir(destination):
			os.makedirs(destination)
		### make sure they are empty
		os.system('rm '+destination+'/*')

		# move plots there
		to_move = objnames[indices]
		xout = x[indices]
		yout = y[indices]
		for obj in to_move:
			# plots made by threedhst_diag
			threedhst_diag_loc = "/".join(outfolder.split('/')[:-1])
			os.system('cp '+threedhst_diag_loc+'/*'+obj.replace(' ','*')+'* '+destination)
			# old magphys sed comparison
			mag_sed_loc = threedhst_diag_loc + '/magphys/sed_residuals'
			os.system('cp '+mag_sed_loc+'/*'+obj.replace(' ','*')+'* '+destination)
			# emission line fits
			mag_sed_loc = threedhst_diag_loc + '/magphys/line_fits'
			os.system('cp '+mag_sed_loc+'/*'+obj.replace(' ','*')+'* '+destination)
			# RGB
			#rgb_loc = '/Users/joel/code/python/threedhst_bsfh/data/brownseds_data/rgb'
			#os.system('cp '+rgb_loc+'/*'+obj.replace(' ','*')+'* '+destination)

		with open(destination+'/list.txt', 'w') as f:
			f.write('# name observed model \n')
			for ii,obj in enumerate(to_move):
				writeout=str(obj)+' '+"{:.2f}".format(xout[ii])+' '+"{:.2f}".format(yout[ii])
				f.write(writeout)
				f.write('\n')

def gas_met(nii_ha,oiii_hb):
	# return gas-phase metallicity based on NII / Halpha ratio
	# 
	# uses the following relationships from Maiolino et al. 2008:
	# 		x = log (Z/Zsol) = 12 + log(O/H) - 8.69, Allende Priete et al. 2001
	# 		c0: -0.7732 c1: 1.2357  c2: -0.2811 c3: -0.7201 c4:-0.3330  [NII/Halpha]
	#		c0: 0.1549  c1: -1.5031 c2: -0.9790 c3: -0.0297 [OIII 5007 / Hbeta] 
	# 		log R = c0 + c1 x + c2 x**2 + c3 x**3 + c4 x**4


	#### find solutions
	c1 = [-0.7732 - np.log10(nii_ha), 1.2357, -0.2811, -0.7201, -0.3330]
	c2 = [0.1549 - np.log10(oiii_hb), -1.5031, -0.9790, -0.0297]
	'''
	solutions = np.polynomial.polynomial.polyroots(coeffs)

	### only keep reasonable solutions
	keep = (np.real(solutions) > -2.0) & (np.real(solutions) < 0.4)
	answer = np.unique(np.real(solutions[keep]))
	if np.unique(np.real(solutions[keep])).shape[0] != 1:
		print 1/0 # we have an unreasonable solution, or multiple solutions
	'''
	#### dummy metallicity array, NII / Ha
	# used to determine where to look with OIII + Hb
	# since that is double-valued
	z_solution = np.linspace(-3.0,0.5,500)
	zeros_guess = c1[0] + c1[1]*z_solution + c1[2]*z_solution**2 + c1[3]*z_solution**3 + c1[4]*z_solution**4
	closest_niiha = z_solution[np.abs(zeros_guess).argmin()]

	#### dummy metallicity array, OIII / Hb
	inflection_point = 7.9-8.69 # estimated from Fig 5 of Maiolino+08
	if closest_niiha < inflection_point:
		z_solution = np.linspace(-4.0,inflection_point,500)
	else:
		z_solution = np.linspace(inflection_point,0.7,500)

	zeros_guess = c2[0] + c2[1]*z_solution + c2[2]*z_solution**2 + c2[3]*z_solution**3
	closest = z_solution[np.abs(zeros_guess).argmin()]

	return closest
	#return (closest+closest_niiha)/2.

def equalize_axes(ax, x,y, dynrange=0.1, line_of_equality=True, log=False, axlims=None):
	
	''' 
	sets up an equal x and y range that encompasses all of the data
	if line_of_equality, add a diagonal line of equality 
	dynrange represents the percent of the data range above and below which
	the plot limits are set
	'''

	if log:
		dynx, dyny = (np.nanmin(x)*0.5, np.nanmin(y)*0.5) 
	else:
		dynx, dyny = (np.nanmax(x)-np.nanmin(x))*dynrange,\
	                 (np.nanmax(y)-np.nanmin(y))*dynrange
	
	if axlims is None:
		if np.nanmin(x)-dynx > np.nanmin(y)-dyny:
			min = np.nanmin(y)-dyny
		else:
			min = np.nanmin(x)-dynx
		if np.nanmax(x)+dynx > np.nanmax(y)+dyny:
			max = np.nanmax(x)+dynx
		else:
			max = np.nanmax(y)+dyny
	else:
		min = axlims[0]
		max = axlims[1]

	ax.set_xlim(min,max)
	ax.set_ylim(min,max)

	if line_of_equality:
		ax.plot([min,max],[min,max],linestyle='--',color='0.1',alpha=0.8)
	return ax

def integral_average(x,y,x0,x1):
	'''
	to do a definite integral over a given x,y array
	you have to redefine the x,y array to only exist over
	the relevant range
	'''	

	xarr_new = np.linspace(x0, x1, 40)
	bad = xarr_new == 0.0
	if np.sum(bad) > 0:
		xarr_new[bad]=1e-10
	intfnc = interp1d(x,y, bounds_error = False, fill_value = 0)
	yarr_new = intfnc(xarr_new)

	from scipy import integrate
	I1 = integrate.simps(yarr_new, xarr_new) / (x1 - x0)

	return I1


def create_prosp_filename(filebase):

	# find most recent output file
	# with the objname
	folder = "/".join(filebase.split('/')[:-1])
	filename = filebase.split("/")[-1]
	files = [f for f in os.listdir(folder) if "_".join(f.split('_')[:-2]) == filename]	
	times = [f.split('_')[-2] for f in files]

	# if we found no files, skip this object
	if len(times) == 0:
		print 'Failed to find any files to extract times in ' + folder + ' of form ' + filename
		return 0

	# load results
	mcmc_filename=filebase+'_'+max(times)+"_mcmc"
	model_filename=filebase+'_'+max(times)+"_model"

	return mcmc_filename,model_filename

def load_prospector_data(filebase):

	from bsfh import read_results

	mcmc_filename, model_filename = create_prosp_filename(filebase)
	sample_results, powell_results, model = read_results.read_pickles(mcmc_filename, model_file=model_filename,inmod=None)

	return sample_results, powell_results, model


def load_zp_offsets(field):

	filename = os.getenv('APPS')+'/threedhst_bsfh/data/zp_offsets_tbl11_skel14.txt'
	with open(filename, 'r') as f:
		for jj in range(1): hdr = f.readline().split()
	dtype = [np.dtype((str, 35)),np.dtype((str, 35)),np.float,np.float]
	dat = np.loadtxt(filename, comments = '#',dtype = np.dtype([(hdr[n+1], dtype[n]) for n in xrange(len(hdr)-1)]))

	if field is not None:
		good = dat['Field'] == field
		if np.sum(good) == 0:
			print 'Not an acceptable field name! Returning None'
			return None
		else:
			dat = dat[good]

	return dat

def load_ancil_data(filename,objnum):

	'''
	loads ancillary plotting information
	'''
	
	with open(filename, 'r') as f:
		for jj in range(1): hdr = f.readline().split()
	dat = np.loadtxt(filename, comments = '#',dtype = np.dtype([(n, np.float) for n in hdr[1:]]))

	if objnum:
		objdat = dat[dat['id'] == float(objnum)]
		return objdat

	return dat

def load_mips_data(filename,objnum=None):
	
	with open(filename, 'r') as f:
		for jj in range(1): hdr = f.readline().split()
	dat = np.loadtxt(filename, comments = '#',dtype = np.dtype([(n, np.float) for n in hdr[1:]]))
	
	if objnum is not None:
		objdat = dat[dat['id'] == float(objnum)]
		return objdat

	return dat

def load_obs_3dhst(filename, objnum, mips=None, min_error = None, abs_error=False,zperr=False):
	"""
	Load 3D-HST photometry file, return photometry for a particular object.
	min_error: set the minimum photometric uncertainty to be some fraction
	of the flux. if not set, use default errors.
	"""
	obs ={}

	with open(filename, 'r') as f:
		hdr = f.readline().split()
	dat = np.loadtxt(filename, comments = '#',
					 dtype = np.dtype([(n, np.float) for n in hdr[1:]]))
	obj_ind = np.where(dat['id'] == int(objnum))[0][0]
	
	# extract fluxes+uncertainties for all objects and all filters
	flux_fields = [f for f in dat.dtype.names if f[0:2] == 'f_']
	unc_fields = [f for f in dat.dtype.names if f[0:2] == 'e_']
	filters = [f[2:] for f in flux_fields]

	# extract fluxes for particular object, converting from record array to numpy array
	flux = dat[flux_fields].view(float).reshape(len(dat),-1)[obj_ind]
	unc  = dat[unc_fields].view(float).reshape(len(dat),-1)[obj_ind]

	# define all outputs
	wave_effective = np.array(return_mwave_custom(filters))
	phot_mask = np.logical_or(np.logical_or((flux != unc),(flux > 0)),flux != -99.0)
	maggies = flux/(10**10)
	maggies_unc = unc/(10**10)

	# set minimum photometric error
	if min_error is not None:
		if abs_error:
			maggies_unc = min_error*maggies
		else:
			under = maggies_unc < min_error*maggies
			maggies_unc[under] = min_error*maggies[under]
	
	if zperr is True:
		zp_offsets = load_zp_offsets(None)
		band_names = np.array([x['Band'].lower()+'_'+x['Field'].lower() for x in zp_offsets])
		
		for kk in xrange(len(filters)):
			match = band_names == filters[kk]
			if np.sum(match) > 0:
				maggies_unc[kk] = ( (maggies_unc[kk]**2) + (maggies[kk]*(1-zp_offsets[match]['Flux-Correction'][0]))**2 ) **0.5

	# sort outputs based on effective wavelength
	points = zip(wave_effective,filters,phot_mask,maggies,maggies_unc)
	sorted_points = sorted(points)

	# build output dictionary
	obs['wave_effective'] = np.array([point[0] for point in sorted_points])
	obs['filters'] = np.array([point[1] for point in sorted_points])
	obs['phot_mask'] =  np.array([point[2] for point in sorted_points])
	obs['maggies'] = np.array([point[3] for point in sorted_points])
	obs['maggies_unc'] =  np.array([point[4] for point in sorted_points])
	obs['wavelength'] = None
	obs['spectrum'] = None

	return obs

def av_to_dust2(av):

	# FSPS
	# dust2: opacity at 5500 
	# e.g., tau1 = (lam/5500)**dust_index)*dust1, and flux = flux*np.exp(-tautot)
	# 
	# Calzetti
	# A_V = Rv (A_B - A_V)
	# A_V = Rv*A_B / (1+Rv)
	# Rv = 4.05
	# 0.63um < lambda < 2.20um
	#k = 2.659(-1.857+1.040/lam) + Rv
	# 0.12um < lambda < 0.63um
	#k = 2.659*(-2.156+1.509/lam - 0.198/(lam**2)+0.11/(lam**3))+Rv
	# F(lambda) = Fobs(lambda)*10**(0.4*E(B-V)*k)

	# eqn: 10**(0.4*E(B-V)*k) = np.exp(-tau), solve for opacity(Av)

	# http://webast.ast.obs-mip.fr/hyperz/hyperz_manual1/node10.html
	# A_5500 = k(lambda)/Rv * Av

	# first, calculate Calzetti extinction at 5500 Angstroms
	lam = 5500/1e4   # in microns
	Rv   = 4.05
	klam = 2.659*(-2.156+1.509/(lam)-0.198/(lam)**2+0.011/(lam)**3)+Rv
	A_lam = klam/Rv*av

	# now convert from extinction (fobs = fint * 10 ** -0.4*extinction)
	# into opacity (fobs = fint * exp(-opacity)) [i.e, dust2]
	#tau = -np.log(10**(-0.4*A_5500))
	#return av
	#tau = av/1.086
	tau = av/1.17
	return tau

	#return av

def return_fast_sed(fastname,objname, sps=None, obs=None, dustem = False):

	'''
	give the fast parameters straight from the FAST out file
	return observables with best-fit FAST parameters, main difference hopefully being stellar population models
	'''

	# load fast parameters
	fast, fields = load_fast_3dhst(fastname, objname)
	fields = np.array(fields)

	# load fast model
	param_file = os.getenv('APPS')+'/threedhst_bsfh/parameter_files/fast_mimic/fast_mimic.py'
	model = model_setup.load_model(param_file)
	parnames = np.array(model.theta_labels())

	# feed parameters into model
	model.params['zred']                   = np.array(fast[fields == 'z'])
	model.initial_theta[parnames=='tage']  = np.clip(np.array((10**fast[fields == 'lage'])/1e9),0.101,10000)
	model.initial_theta[parnames=='tau']   = np.array((10**fast[fields == 'ltau'])/1e9)
	model.initial_theta[parnames=='dust2'] = np.array(av_to_dust2(fast[fields == 'Av']))
	model.initial_theta[parnames=='mass']  = np.array(10**fast[fields == 'lmass'])

	print 'z,tage,tau,dust2,mass'
	print model.params['zred'],model.initial_theta[parnames=='tage'],model.initial_theta[parnames=='tau'],model.initial_theta[parnames=='dust2'],model.initial_theta[parnames=='mass']

	# get dust emission, if desired
	if dustem:
		model.params['add_dust_emission']  = np.array(True)
		model.params['add_agb_dust_model'] = np.array(True)


	spec, mags,sm = model.mean_model(model.initial_theta, obs, sps=sps, norm_spec=True)
	w = sps.wavelengths
	return spec,mags,w,fast,fields


def load_fast_3dhst(filename, objnum):
	"""
	Load FAST output for a particular object
	Returns a dictionary of inputs for BSFH
	"""

	# filter through header junk, load data
	with open(filename, 'r') as f:
		for jj in range(1): hdr = f.readline().split()
	dat = np.loadtxt(filename, comments = '#',dtype = np.dtype([(n, np.float) for n in hdr[1:]]))

	# extract field names, search for ID, pull out object info
	fields = [f for f in dat.dtype.names]
	
	
	if objnum is None:
		values = dat[fields].view(float).reshape(len(dat),-1)
	else:
		values = dat[fields].view(float).reshape(len(dat),-1)
		id_ind = fields.index('id')
		obj_ind = [int(x[id_ind]) for x in dat].index(int(objnum))
		values = values[obj_ind]

	return values, fields

def integrate_mag(spec_lam,spectra,filter, z=None, alt_file='/Users/joel/code/python/threedhst_bsfh/filters/allfilters_threedhst.dat'):

	'''
	borrowed from calc_ml
	given a filter name and spectrum, calculate magnitude/luminosity in filter (see alt_file for filter names)
	INPUT: 
		SPEC_LAM: must be in angstroms. this will NOT BE corrected for reddening even if redshift is specified. this
		allows you to calculate magnitudes in rest or observed frame.
		SPECTRA: must be in Lsun/Hz (FSPS standard). if redshift is specified, the normalization will be taken care of.
	OUTPUT:
		MAG: comes out as absolute magnitude
		LUMINOSITY: comes out in erg/s
			NOTE: if redshift is specified, INSTEAD RETURN apparent magnitude and flux [erg/s/cm^2]
	'''

	if type(filter) == str:
		resp_lam, res = load_filter_response(filter, 
		                                 	 alt_file=alt_file)
	else:
		resp_lam = filter[0][0]
		res      = filter[1][0]

	# calculate effective width
	#dlam = (resp_lam[1:]-resp_lam[:-1])/2.
	#width = np.array([dlam[ii]+dlam[ii+1] for ii in xrange(len(resp_lam)-2)])
	#width = np.concatenate((np.atleast_1d(dlam[0]*2),width,np.atleast_1d(dlam[-1]*2)))
	#effective_width = np.sum(res*width)
	
	# physical units, in CGS, from sps_vars.f90 in the SPS code
	pc2cm = 3.08568E18
	lsun  = 3.839E33
	c     = 2.99E10

	# interpolate filter response onto spectral lambda array
	# when interpolating, set response outside wavelength range to be zero.
	response_interp_function = interp1d(resp_lam,res, bounds_error = False, fill_value = 0)
	resp_interp = response_interp_function(spec_lam)
	
	# integrate spectrum over filter response
	# first calculate luminosity: convert to flambda (factor of c/lam^2, with c converted to AA/s)
	# then integrate over flambda [Lsun/AA] to get Lsun
	spec_flam = spectra*(c*1e8/(spec_lam**2))
	luminosity = simps(spec_flam*resp_interp,spec_lam)
	
	# now calculate luminosity density [erg/s/Hz] in filter
	# this involves normalizing the filter response by integrating over wavelength
	norm = simps(resp_interp/spec_lam,spec_lam)
	luminosity_density = simps(spectra*(resp_interp/norm)/spec_lam,spec_lam)

	# if redshift is specified, convert to flux and apparent magnitude
	if z is not None:
		dfactor = (WMAP9.luminosity_distance(z).value*1e5)**(-2)*(1+z)
		luminosity = luminosity*dfactor
		luminosity_density = luminosity_density*dfactor

	# convert luminosity density to flux density
	# the units of the spectra are Lsun/Hz; convert to
	# erg/s/cm^2/Hz, at 10pc for absolute mags
	flux_density = luminosity_density*lsun/(4.0*np.pi*(pc2cm*10)**2)
	luminosity   = luminosity*lsun

	# convert flux density to magnitudes in AB system
	mag = -2.5*np.log10(flux_density)-48.60

	#print 'maggies: {0}'.format(10**(-0.4*mag)*1e10)
	return mag, luminosity

def return_full_sfh(t, sfh_params):

	'''
	returns full SFH given a time vector [in Gyr] and a
	set of SFH parameters
	'''

	deltat=0.0001

	# calculate new time vector such that
	# the spacing from tage back to zero
	# is identical for each SFH model
	tcalc = t-sfh_params['tage']
	tcalc = tcalc[tcalc < 0]*-1

	intsfr = np.zeros(len(t))
	for mm in xrange(len(tcalc)): 
		intsfr[mm] = calculate_sfr(sfh_params, deltat, tcalc = tcalc[mm])

	return intsfr

def calculate_sfr(sfh_params, timescale, tcalc = None, 
	              minsfr = None, maxsfr = None):

	'''
	standardized SFR calculator. returns SFR averaged over timescale.

	SFH_PARAMS: standard input
	TIMESCALE: timescale over which to calculate SFR. timescale must be in Gyr.
	TCALC: at what point in the SFH do we want the SFR? If not specified, TCALC is set to sfh_params['tage']
	MINSFR: minimum returned SFR. if not specified, minimum is 0.01% of average SFR over lifetime
	MAXSFR: maximum returned SFR. if not specified, maximum is infinite.

	returns in [Msun/yr]

	'''

	if tcalc is None:
		tcalc = sfh_params['tage']

	sfr=integrate_sfh(tcalc-timescale,
		              tcalc,
		              sfh_params)*np.sum(sfh_params['mformed'])/(timescale*1e9)

	if minsfr is None:
		minsfr = np.sum(sfh_params['mformed']) / (sfh_params['tage']*1e9*10000)

	if maxsfr is None:
		maxsfr = np.inf

	sfr = np.clip(sfr, minsfr, maxsfr)

	return sfr

def integrate_delayed_tau(t1,t2,sfh):

	return (np.exp(-t1/sfh['tau'])*(1+t1/sfh['tau']) - \
	       np.exp(-t2/sfh['tau'])*(1+t2/sfh['tau']))*sfh['tau']**2

def integrate_linramp(t1,t2,sfh):

	# integration constant: SFR(sf_trunc-sf_start)
	cs = (sfh['sf_trunc']-sfh['sf_start'])*(np.exp(-(sfh['sf_trunc']-sfh['sf_start'])/sfh['tau']))

	# enforce positive SFRs
	# by limiting integration to where SFR > 0
	t_zero_cross = -1.0/sfh['sf_slope'] + sfh['sf_trunc']
	if t_zero_cross > sfh['sf_trunc']-sfh['sf_start']:
		t1           = np.clip(t1,sfh['sf_trunc']-sfh['sf_start'],t_zero_cross)
		t2           = np.clip(t2,sfh['sf_trunc']-sfh['sf_start'],t_zero_cross)

	# initial integral: SFR = SFR[t=t_trunc] * [1 + s(t-t_trunc)]
	intsfr = cs*(t2-t1)*(1-sfh['sf_trunc']*sfh['sf_slope']) + cs*sfh['sf_slope']*0.5*((t2+sfh['sf_start'])**2-(t1+sfh['sf_start'])**2)

	return intsfr

def integrate_sfh(t1,t2,sfh_params):
	
	'''
	integrate a delayed tau SFH from t1 to t2
	sfh = dictionary of SFH parameters
	'''

	# copy over so we don't overwrite values
	# put tau into linear units
	sfh = sfh_params.copy()
	sfh['tau'] = 10**sfh['tau']

	# here is our coordinate transformation to match fsps
	t1 = t1-sfh['sf_start']
	t2 = t2-sfh['sf_start']

	# match dimensions, if two-tau model
	ndim = len(np.atleast_1d(sfh['mass']))
	if len(np.atleast_1d(t2)) != ndim:
		t2 = np.zeros(ndim)+t2
	if len(np.atleast_1d(t1)) != ndim:
		t1 = np.zeros(ndim)+t1

	# redefine sf_trunc, if not being used for sfh=5 purposes
	if sfh['sf_trunc'] == 0.0:
		sfh['sf_trunc'] = sfh['tage']

	# if we're outside of the time boundaries, clip to boundary values
	# this only affects integrals which would have been questionable in the first place
	t1 = np.clip(t1,0,sfh['tage']-sfh['sf_start'])
	t2 = np.clip(t2,0,sfh['tage']-sfh['sf_start'])

  	# if we're using delayed tau
  	if (sfh['sfh'] == 1) or (sfh['sfh'] == 4):

  			# add tau model
			intsfr =  integrate_delayed_tau(t1,t2,sfh)
			norm =    1.0- np.exp(-(sfh['sf_trunc']-sfh['sf_start'])/sfh['tau'])*(1+(sfh['sf_trunc']-sfh['sf_start'])/sfh['tau'])
			intsfr = intsfr/(norm*sfh['tau']**2)

	# else, add lin-ramp
	elif (sfh['sfh'] == 5):

		# by-hand calculation
		norm1 = integrate_delayed_tau(0,sfh['sf_trunc']-sfh['sf_start'],sfh)
		norm2 = integrate_linramp(sfh['sf_trunc']-sfh['sf_start'],sfh['tage']-sfh['sf_start'],sfh)

		if (t1 < sfh['sf_trunc']-sfh['sf_start']) and \
		   (t2 < sfh['sf_trunc']-sfh['sf_start']):
			intsfr = integrate_delayed_tau(t1,t2,sfh) / (norm1+norm2)
		elif (t1 > sfh['sf_trunc']-sfh['sf_start']) and \
		     (t2 > sfh['sf_trunc']-sfh['sf_start']):
			intsfr = integrate_linramp(t1,t2,sfh) / (norm1+norm2)
		else:
			intsfr = (integrate_delayed_tau(t1,sfh['sf_trunc']-sfh['sf_start'],sfh) + \
				      integrate_linramp(sfh['sf_trunc']-sfh['sf_start'],t2,sfh)) / \
                      (norm1+norm2)

	else:
		print 'no such SFH implemented'
		print 1/0

	# return sum of SFR components
	tot_sfr = np.sum(intsfr*sfh['mass'])/np.sum(sfh['mass'])
	return tot_sfr

def measure_emline_lum(sps, model = None, obs = None, thetas = None, 
	                   measure_ir = False, savestr = False, saveplot=True,
	                   spec=None, abslines=True):
	
	'''
	takes spec(on)-spec(off) to measure emission line luminosity
	sideband is defined for each emission line after visually 
	inspecting the spectral sampling density around each line

	if we pass spec, then avoid the first model call

	flux comes out in Lsun
	'''
	out = {}

	# get spectrum
	if model:

		# save redshift
		z      = model.params.get('zred', np.array(0.0))
		model.params['zred'] = np.array(0.0)

		# nebon
		spec_nebon,mags_nebon,sm = model.mean_model(thetas, obs, sps=sps)

		# neboff
		model.params['add_neb_emission'] = np.array(False)
		model.params['add_neb_continuum'] = np.array(False)
		spec_neboff,mags_neboff,sm = model.mean_model(thetas, obs, sps=sps)
		w = sps.wavelengths
		model.params['add_neb_emission'] = np.array(True)
		model.params['add_neb_continuum'] = np.array(True)

		# subtract, switch to flam
		factor = 3e18 / w**2
		spec = (spec_nebon-spec_neboff) *factor
		spec_nebon *= factor
		spec_neboff *= factor

		model.params['zred'] = z

	else:
		if spec is None:
			print 'okay, you need to give me something here'
			print 1/0
		w = sps.wavelengths


	##### measure absorption lines and Dn4000
	out['dn4000'] = measure_Dn4000(w,spec_nebon)
	if abslines:
		smooth_spec = smooth_spectrum(w,spec_neboff,200.0,minlam=3e3,maxlam=8e3)
		out['abslines'] = measure_abslines(w,smooth_spec) # comes out in Lsun and rest-frame EQW

	##### measure emission lines
	out['emlines'] = measure_emlines(w,spec,smooth_spec)

	if measure_ir:

		# set up filters
		mips_index = [i for i, s in enumerate(obs['filters']) if 'mips' in s]
		if np.sum(mips_index) == 0:
			mips_index = [i for i, s in enumerate(obs['filters']) if 'MIPS' in s]

		lir = return_lir(w,spec, z=None, alt_file=None) # comes out in ergs/s
		lir /= 3.846e33 #  convert to Lsun

		# if no MIPS flux...
		try:
			mips = mags_nebon[mips_index][0]*1e10 # comes out in maggies, convert to flux such that AB zeropoint is 25 mags
		except IndexError:
			mips = np.nan

		out['lir'] = lir
		out['mips'] = mips
	
	return out

def measure_Dn4000(lam,flux,ax=None):

	# D4000, defined as average flux ratio between
	# [4050,4250] and [3750,3950] (Bruzual 1983; Hamilton 1985)

	# ratio of continuua, measured as defined below 
	# blue: 3850-3950 . . . 4000-4100 (Balogh 1999)

	blue = (lam > 3850) & (lam < 3950)
	red  = (lam > 4000) & (lam < 4100)
	dn4000 = np.mean(flux[red])/np.mean(flux[blue])

	if ax is not None:
		ax.plot(lam,flux,color='black',drawstyle='steps-mid')
		ax.plot(lam[blue],flux[blue],color='blue',drawstyle='steps-mid')
		ax.plot(lam[red],flux[red],color='red',drawstyle='steps-mid')

		ax.text(0.96,0.05, 'D$_{n}$(4000)='+"{:.2f}".format(dn4000), transform = ax.transAxes,horizontalalignment='right')

		ax.set_xlim(3800,4150)
		plt_lam_idx = (lam > 3800) & (lam < 4150)
		minplot = np.min(flux[plt_lam_idx])*0.9
		maxplot = np.max(flux[plt_lam_idx])*1.1
		ax.set_ylim(minplot,maxplot)

	return dn4000

def measure_emlines(lam,flux_emline, flux_abs):

	'''
	Nelan et al. (2005)
	Halpha wide: 6515-6540, 6554-6575, 6575-6585
	Halpha narrow: 6515-6540, 6554-6568, 6568-6575

	Worthey et al. 1994
	Hbeta: 4827.875-4847.875, 4847.875-4876.625, 4876.625-4891

	Worthey et al. 1997
	hdelta wide: 4041.6-4079.75, 4083.5-4122.25, 4128.5-4161.0
	hdelta narrow: 4057.25-4088.5, 4091-4112.25, 4114.75-4137.25

	WIDENED HALPHA AND HBETA BECAUSE OF SMOOTHING
	'''

	out = {}

    # define emission lines
	lines = np.array(['[OII]','Hdelta','Hbeta','[OIII]1','[OIII]2','Halpha','[NII]','[SII]'])
	wavelength = np.array([3728,4102.0,4861.33,4959,5007,6562,6583,6732.71])
	sideband   = [(3723,3736),(4090,4110),(4857,4868),(4954,4968),(5001,5015),(6556,6573),(6573,6593),(6710,6728)]

	up   = [(3742.,3750.),(4124.25,4151.00),(4894.625,4910.000),(4970.,4979.),(5025.,5035.),(6590.,6610.),(6590.,6610.),(6730.,6740.)]
	down = [(3713.,3720.),(4041.6,4081.5),(4817.875,4835.875),(4946.,4955.),(4985.,4995.),(6515.,6540.),(6515.,6540.),(6700.,6710.)]

	#fig, ax = plt.subplots(2,4, figsize = (25,12))
	#ax = np.ravel(ax)

	##### measure emission line flux + EQW
	for jj in xrange(len(lines)):
		dic = {}
		dic['flux'], dic['eqw'] = measure_em(lam,flux_emline,flux_abs,wavelength[jj],sideband[jj],up[jj],down[jj],ax=None)
		out[lines[jj]] = dic

	#plt.savefig('test.png',dpi=150)
	#os.system('open test.png')
	#plt.savefig(os.getenv('APPS')+'/threedhst_bsfh/plots/testem/'+savestr+'.png',dpi=150)
	#plt.close()

	return out


def measure_em(w,spec,spec_abs,wavelength,sideband,up, down, ax=None,savestr=None):

	#### emission line flux
	integrate_lam = (w > sideband[0]) & (w < sideband[1])
	baseline      = np.min(spec[integrate_lam])
	emline_flux = np.trapz(spec[integrate_lam]-baseline, w[integrate_lam])

	# plot emission lines
	if False:
		ax.plot(w,spec,'ro',linestyle='-')
		ax.plot(w[integrate_lam], spec[integrate_lam], 'bo',linestyle=' ')

		plotlines=['[SIII]','[NII]','Halpha','[SII]']
		plotlam  =np.array([6312,6583,6563,6725])
		for kk in xrange(len(plotlam)):
			ax.vlines(plotlam[kk],plt.ylim()[0],plt.ylim()[1],color='0.5',linestyle='--')
			ax.text(plotlam[kk],(plt.ylim()[0]+plt.ylim()[1])/2.*(1.0-kk/6.0),plotlines[kk])
		
		ax.set_xlim(wavelength-40,wavelength+40)
		ax.set_ylim(-np.max(spec[integrate_lam])*0.2,np.max(spec[integrate_lam])*1.2)

	#### measure continuum
	# identify average flux, average wavelength
	low_cont = (w > down[0]) & (w < down[1])
	high_cont = (w > up[0]) & (w < up[1])

	low_flux = np.mean(spec_abs[low_cont])
	high_flux = np.mean(spec_abs[high_cont])

	low_lam = np.mean(down)
	high_lam = np.mean(up)

	# draw straight line between midpoints
	m = (high_flux-low_flux)/(high_lam-low_lam)
	b = high_flux - m*high_lam
	contflux = m*np.mean(sideband)+b
	emline_eqw = emline_flux / contflux

	##### plot if necessary
	if ax is not None:
		ax.plot(w,spec_abs,color='black', drawstyle='steps-mid')
		ax.plot([low_lam,high_lam],[contflux,contflux],color='red')

		ax.set_xlim(np.min(down)-50,np.max(up)+50)
		plt_lam_idx = (w > down[0]) & (w< up[1])
		ax.set_ylim(np.min(spec_abs[plt_lam_idx])*0.96,np.max(spec_abs[plt_lam_idx])*1.04)


	return emline_flux, emline_eqw

def measure_abslines(lam,flux,plot=False, alt_plot=False):

	'''
	Nelan et al. (2005)
	Halpha wide: 6515-6540, 6554-6575, 6575-6585
	Halpha narrow: 6515-6540, 6554-6568, 6568-6575

	Worthey et al. 1994
	Hbeta: 4827.875-4847.875, 4847.875-4876.625, 4876.625-4891

	Worthey et al. 1997
	hdelta wide: 4041.6-4079.75, 4083.5-4122.25, 4128.5-4161.0
	hdelta narrow: 4057.25-4088.5, 4091-4112.25, 4114.75-4137.25

	WIDENED HALPHA AND HBETA BECAUSE OF SMOOTHING
	'''

	out = {}

	# define lines and indexes
	lines = np.array(['halpha_wide', 'halpha_narrow',
	                  'hbeta',
	                  'hdelta_wide', 'hdelta_narrow'])

	# improved hdelta narrow
	index = [(6540.,6586.),(6542.,6584.),(4842.875,4884.625),(4081.5,4124.25),(4095.0,4113.75)]
	up =    [(6590.,6610.),(6585.,6595.),(4894.625,4910.000),(4124.25,4151.00),(4113.75,4130.25)]
	down =  [(6515.,6540.),(6515.,6540.),(4817.875,4835.875),(4041.6,4081.5),(4072.25,4094.50)]

	# if we want to plot but don't have fig + ax set up, set them up
	# else if we have them, unpack them
	if plot == True:
		fig, ax = plt.subplots(2,3, figsize = (18.75,12))
		ax = np.ravel(ax)
	elif plot is not False:
		fig, ax = plot[0],plot[1]

	# measure the absorption flux
	for ii in xrange(len(lines)):
		dic = {}

		#### only alt_plot for hdelta!
		if alt_plot & (ii <=2):
			continue

		if plot: 
			dic['flux'], dic['eqw'], dic['lam'] = measure_idx(lam,flux,index[ii],up[ii],down[ii],ax=ax[ii],alt_plot=alt_plot)
		else: 
			dic['flux'], dic['eqw'], dic['lam'] = measure_idx(lam,flux,index[ii],up[ii],down[ii],ax=None)
		out[lines[ii]] = dic

	if plot is not False:
		out['ax'] = ax
		out['fig'] = fig

	return out

def measure_idx(lam,flux,index,up,down,ax=None,alt_plot=False):

	'''
	measures absorption depths
	'''

	continuum_line_color='red'
	observed_flux_color = 'black'
	line_index_color = 'blue'
	continuum_index_color = 'cyan'
	eqw_yloc = 0.05
	alpha = 1.0

	if alt_plot:
		eqw_yloc = 0.1
		line_index_color = 'green'
		alpha = 0.5


	##### identify average flux, average wavelength
	low_cont = (lam > down[0]) & (lam < down[1])
	high_cont = (lam > up[0]) & (lam < up[1])
	abs_idx = (lam > index[0]) & (lam < index[1])

	low_flux = np.mean(flux[low_cont])
	high_flux = np.mean(flux[high_cont])

	low_lam = np.mean(down)
	high_lam = np.mean(up)

	##### draw straight line between midpoints
	# y = mx + b
	# m = (y2 - y1) / (x2 - x1)
	# b = y0 - mx0
	m = (high_flux-low_flux)/(high_lam-low_lam)
	b = high_flux - m*high_lam

	##### integrate the flux and the straight line, take the difference
	yline = m*lam[abs_idx]+b
	absflux = np.trapz(yline,lam[abs_idx]) - np.trapz(flux[abs_idx], lam[abs_idx])

	lamcont = np.mean(lam[abs_idx])
	abseqw = absflux/(m*lamcont+b)

	##### plot if necessary
	if ax is not None:
		ax.plot(lam,flux,color=observed_flux_color, drawstyle='steps-mid',alpha=alpha)
		ax.plot(lam[abs_idx],yline,color=continuum_line_color,alpha=alpha)
		ax.plot(lam[abs_idx],flux[abs_idx],color=line_index_color,alpha=alpha)
		ax.plot(lam[low_cont],flux[low_cont],color=continuum_index_color,alpha=alpha)
		ax.plot(lam[high_cont],flux[high_cont],color=continuum_index_color,alpha=alpha)

		ax.text(0.96,eqw_yloc, 'EQW='+"{:.2f}".format(abseqw), transform = ax.transAxes,horizontalalignment='right',color=line_index_color)
		ax.set_xlim(np.min(down)-50,np.max(up)+50)

		plt_lam_idx = (lam > down[0]) & (lam < up[1])
		
		if alt_plot:
			minplot = np.min([ np.min(flux[plt_lam_idx])*0.9, ax.get_ylim()[0] ])
			maxplot = np.max([ np.max(flux[plt_lam_idx])*1.1, ax.get_ylim()[1] ])
		else:
			minplot = np.min(flux[plt_lam_idx])*0.9
			maxplot = np.max(flux[plt_lam_idx])*1.1
		ax.set_ylim(minplot,maxplot)

	return absflux, abseqw, lamcont








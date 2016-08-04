import numpy as np
from astropy.cosmology import WMAP9 as cosmo
from astropy.modeling import core, fitting, Parameter, functional_models
import threed_dutils
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import leastsq
from astropy import constants
from matplotlib.ticker import MaxNLocator
from scipy.integrate import simps

c_kms = 2.99e5
maxfev = 2000
dpi=150

class tLinear1D(core.Fittable1DModel):

	slope_low = Parameter(default=0)
	intercept_low = Parameter(default=0)
	slope_mid = Parameter(default=0)
	intercept_mid = Parameter(default=0)
	slope_high = Parameter(default=0)
	intercept_high = Parameter(default=0)
	linear = True

	@staticmethod
	def evaluate(x, slope_low, intercept_low, slope_mid, intercept_mid, slope_high, intercept_high):
		"""One dimensional Line model function"""

		out = np.zeros_like(x)
		low = np.array(x) < 4000
		mid = (np.array(x) > 4000) & (np.array(x) < 5600)
		high = np.array(x) > 5600
		out[low] = slope_low*x[low]+intercept_low
		out[mid] = slope_mid*x[mid]+intercept_mid
		out[high] = slope_high*x[high]+intercept_high

		return out

	@staticmethod
	def fit_deriv(x, slope_low, intercept_low, slope_mid, intercept_mid, slope_high, intercept_high):
		"""One dimensional Line model derivative with respect to parameters"""

		low = np.array(x) < 4000
		mid = (np.array(x) > 4000) & (np.array(x) < 5600)
		high = np.array(x) > 5600
		
		d_lslope = np.zeros_like(x)
		d_lint = np.zeros_like(x)
		d_mslope = np.zeros_like(x)
		d_mint = np.zeros_like(x)
		d_hslope = np.zeros_like(x)
		d_hint = np.zeros_like(x)
		
		d_lslope[low] = x[low]
		d_mslope[mid] = x[mid]
		d_hslope[high] = x[high]
		d_lint[low] = np.ones_like(x[low])
		d_mint[mid] = np.ones_like(x[mid])
		d_hint[high] = np.ones_like(x[high])

		return [d_lslope, d_lint, d_mslope, d_mint, d_hslope, d_hint]


class bLinear1D(core.Fittable1DModel):

	slope_low = Parameter(default=0)
	intercept_low = Parameter(default=0)
	slope_high = Parameter(default=0)
	intercept_high = Parameter(default=0)
	linear = True

	@staticmethod
	def evaluate(x, slope_low, intercept_low, slope_high, intercept_high):
		"""One dimensional Line model function"""

		out = np.zeros_like(x)
		low = np.array(x) < 5600
		high = np.array(x) > 5600
		out[low] = slope_low*x[low]+intercept_low
		out[high] = slope_high*x[high]+intercept_high

		return out

	@staticmethod
	def fit_deriv(x, slope_low, intercept_low, slope_high, intercept_high):
		"""One dimensional Line model derivative with respect to parameters"""

		low = np.array(x) < 5600
		high = np.array(x) > 5600
		
		d_lslope = np.zeros_like(x)
		d_lint = np.zeros_like(x)
		d_hslope = np.zeros_like(x)
		d_hint = np.zeros_like(x)
		
		d_lslope[low] = x[low]
		d_hslope[high] = x[high]
		d_lint[low] = np.ones_like(x[low])
		d_hint[high] = np.ones_like(x[high])

		return [d_lslope, d_lint, d_hslope, d_hint]


class jLorentz1D(core.Fittable1DModel):
	"""
	redefined to add a constant
	"""
	amplitude = Parameter(default=1)
	x_0 = Parameter(default=0)
	fwhm = Parameter(default=1)
	constant = Parameter(default=0.0)

	@staticmethod
	def evaluate(x, amplitude, x_0, fwhm, constant):
		"""One dimensional Lorentzian model function"""

		return (amplitude * ((fwhm / 2.) ** 2) / ((x - x_0) ** 2 +
		                                          (fwhm / 2.) ** 2)) + constant

	@staticmethod
	def fit_deriv(x, amplitude, x_0, fwhm, constant):
		"""One dimensional Lorentzian model derivative with respect to parameters"""

		d_amplitude = fwhm ** 2 / (fwhm ** 2 + (x - x_0) ** 2)
		d_x_0 = (amplitude * d_amplitude * (2 * x - 2 * x_0) /
		         (fwhm ** 2 + (x - x_0) ** 2))
		d_fwhm = 2 * amplitude * d_amplitude / fwhm * (1 - d_amplitude)
		d_constant = np.zeros(len(x))+1.0

		return [d_amplitude, d_x_0, d_fwhm, d_constant]

class Voigt1D(core.Fittable1DModel):
	x_0 = Parameter(default=0)
	amplitude_L = Parameter(default=1)
	fwhm_L = Parameter(default=2/np.pi)
	fwhm_G = Parameter(default=np.log(2))

	_abcd = np.array([
	    [-1.2150, -1.3509, -1.2150, -1.3509],  # A
	    [1.2359, 0.3786, -1.2359, -0.3786],    # B
	    [-0.3085, 0.5906, -0.3085, 0.5906],    # C
	    [0.0210, -1.1858, -0.0210, 1.1858]])   # D

	@classmethod
	def evaluate(cls, x, x_0, amplitude_L, fwhm_L, fwhm_G):

		A, B, C, D = cls._abcd
		sqrt_ln2 = np.sqrt(np.log(2))
		X = (x - x_0) * 2 * sqrt_ln2 / fwhm_G
		X = np.atleast_1d(X)[..., np.newaxis]
		Y = fwhm_L * sqrt_ln2 / fwhm_G
		Y = np.atleast_1d(Y)[..., np.newaxis]

		V = np.sum((C * (Y - A) + D * (X - B))/(((Y - A) ** 2 + (X - B) ** 2)), axis=-1)

		return (fwhm_L * amplitude_L * np.sqrt(np.pi) * sqrt_ln2 / fwhm_G) * V

	@classmethod
	def fit_deriv(cls, x, x_0, amplitude_L, fwhm_L, fwhm_G):

		A, B, C, D = cls._abcd
		sqrt_ln2 = np.sqrt(np.log(2))
		X = (x - x_0) * 2 * sqrt_ln2 / fwhm_G
		X = np.atleast_1d(X)[:, np.newaxis]
		Y = fwhm_L * sqrt_ln2 / fwhm_G
		Y = np.atleast_1d(Y)[:, np.newaxis]
		cnt = fwhm_L * amplitude_L * np.sqrt(np.pi) * sqrt_ln2 / fwhm_G

		alpha = C * (Y - A) + D * (X - B)
		beta = (Y - A) ** 2 + (X - B) ** 2
		V = np.sum((alpha / beta), axis=-1)
		dVdx = np.sum((D/beta - 2 * (X - B) * alpha / np.square(beta)), axis=-1)
		dVdy = np.sum((C/beta - 2 * (Y - A) * alpha / np.square(beta)), axis=-1)

		dyda = [- cnt * dVdx * 2 * sqrt_ln2 / fwhm_G,
				cnt * V / amplitude_L,
		        cnt * (V / fwhm_L + dVdy * sqrt_ln2 / fwhm_G),
		        -cnt * (V + (sqrt_ln2 / fwhm_G) * (2 * (x - x_0) * dVdx + fwhm_L * dVdy)) / fwhm_G]
		return dyda

def bootstrap(obslam, obsflux, model, fitter, noise, line_lam, 
	          flux_flag=True, nboot=100):

	# count lines
	nlines = len(np.atleast_1d(line_lam))

	# desired output
	params = np.zeros(shape=(len(model.parameters),nboot))
	flux   = np.zeros(shape=(nboot,nlines))

	# random data sets are generated and fitted
	for i in xrange(nboot):
		randomDelta = np.random.normal(0., 1.,len(obsflux))
		randomFlux = obsflux + randomDelta*noise
		fit = fitter(model, obslam, randomFlux,maxiter=1000)
		params[:,i] = fit.parameters
		#model.parameters = fit.parameters

		# calculate emission line flux + rest EQW
		if flux_flag:
			for j in xrange(nlines): flux[i,j] = getattr(fit, 'amplitude_'+str(j)).value*np.sqrt(2*np.pi*getattr(fit, 'stddev_'+str(j)).value**2) * constants.L_sun.cgs.value

	# now get median + percentiles
	medianpar = np.percentile(params, 50,axis=1)
	fit.parameters = medianpar

	# we want errors for flux and eqw
	fluxout = np.array([np.percentile(flux, 50,axis=0),np.percentile(flux, 84,axis=0),np.percentile(flux, 16,axis=0)])

	return fit,fluxout,flux

def absline_model(lam):
	'''
	return model for region + set of lambdas
	might need to complicate by setting custom del_lam, fwhm_max, etc for different regions
	'''

	### how much do we allow centroid to vary?
	lam = np.atleast_1d(lam)

	for ii in xrange(len(lam)):
		if ii == 0:
			voi = Voigt1D(amplitude_L=-5e6, x_0=lam[ii], fwhm_L=5.0, fwhm_G=5.0)
		else:
			voi += Voigt1D(amplitude_L=-5e6, x_0=lam[ii], fwhm_L=5.0, fwhm_G=5.0)

	#voi += tLinear1D(intercept_low=5e6, intercept_mid=5e6, intercept_high=5e6)
	voi += functional_models.Linear1D(intercept=5e6)

	return voi

def absobs_model(lams):

	lams = np.atleast_1d(lams)

	#### ADD ALL MODELS FIRST
	for ii in xrange(len(lams)):
		if ii == 0:
			model = functional_models.Gaussian1D(amplitude=-5e5, mean=lams[ii], stddev=3.0)
		else: 
			model += functional_models.Gaussian1D(amplitude=-5e5, mean=lams[ii], stddev=3.0)

	#### NOW ADD LINEAR COMPONENT
	model += functional_models.Linear1D(intercept=1e7)

	return model

def tiedfunc_oii(g1):
	amp = 0.35 * g1.amplitude_6
	return amp

def tiedfunc_oiii(g1):
	amp = 2.98 * g1.amplitude_0
	return amp

def tiedfunc_nii(g1):
	amp = 2.93 * g1.amplitude_3
	return amp

def tiedfunc_sig(g1):
	return g1.stddev_0

def loii_2(g1):
	zadj = g1.mean_0 / 4958.92 - 1
	return (1+zadj) *  3728.8

def loii_1(g1):
	zadj = g1.mean_0 / 4958.92 - 1
	return (1+zadj) *  3726.1

def loiii_2(g1):
	zadj = g1.mean_0 / 4958.92 - 1
	return (1+zadj) *  5006.84

def lhbeta(g1):
	zadj = g1.mean_0 / 4958.92 - 1
	return (1+zadj) *  4861.33

def lnii_1(g1):
	zadj = g1.mean_0 / 4958.92 - 1
	return (1+zadj) *  6548.03

def lnii_2(g1):
	zadj = g1.mean_0 / 4958.92 - 1
	return (1+zadj) *  6583.41

def lhalpha(g1):
	zadj = g1.mean_0 / 4958.92 - 1
	return (1+zadj) *  6562.80

def soii(g1):
	return g1.stddev_6

def soiii(g1):
	return g1.stddev_0

def snii(g1):
	return g1.stddev_3

def sig_ret(model):
	n=0
	sig=[]
	while True:
		try:
			sig.append(getattr(model, 'stddev_'+str(n)).value)
		except:
			break
		n+=1

	return sig

def umbrella_model(lams, amp_tied, lam_tied, sig_tied,continuum_6400):
	'''
	return model for [OIII], Hbeta, [NII], Halpha, + constant
	centers are all tied to redshift parameter
	'''

	#### EQW initial
	# OIII 1, OIII 2, Hbeta, NII 1, NII 2, Halpha
	eqw_init = np.array([3.,9.,4.,1.,3.,12.,3.,3.])*4	
	stddev_init = 3.5

	#### ADD ALL MODELS FIRST
	for ii in xrange(len(lams)):
		amp_init = continuum_6400*eqw_init[ii] / (np.sqrt(2.*np.pi*stddev_init**2))
		if ii == 0:
			model = functional_models.Gaussian1D(amplitude=amp_init, mean=lams[ii], stddev=stddev_init)
		else: 
			model += functional_models.Gaussian1D(amplitude=amp_init, mean=lams[ii], stddev=stddev_init)

	# add slope + constant
	model += tLinear1D(intercept_low=continuum_6400,intercept_mid=continuum_6400,intercept_high=continuum_6400)

	#### NOW TIE THEM TOGETHER
	for ii in xrange(len(lams)):
		# position and widths
		if ii != 0:
			#getattr(model, 'stddev_'+str(ii)).tied = tiedfunc_sig
			getattr(model, 'mean_'+str(ii)).tied = lam_tied[ii]

		# amplitudes, if necessary
		if amp_tied[ii] is not None:
			getattr(model, 'amplitude_'+str(ii)).tied = amp_tied[ii]

		# sigmas, if necessary
		if sig_tied[ii] is not None:
			getattr(model, 'stddev_'+str(ii)).tied = sig_tied[ii]

		getattr(model, 'stddev_'+str(ii)).max = 10


	return model

def measure(sample_results, obs_spec, magphys, sps, sigsmooth=None):
	
	'''
	measure rest-frame emission line luminosities using two different continuum models, MAGPHYS and Prospector

	ALGORITHM:
	convert to rest-frame spectrum in proper units
	
	for each model:
		fetch Voigt model for Halpha, Hbeta
		smooth by [200 km/s, 200 km/s]
		fit Voigt + slope + constant

	fit Gaussians + slope + constant + redshift to two regions around [OIII],Hbeta and [NII], Halpha
	for each model:

		normalize and redshift (if needed) absorption to match emission model
		smooth absorption spectrum with measured width of emission lines [may need to add multiplicative constant]
		subtract continuum from spectrum
		mask emission lines
		fit gaussian to remaining residuals
		the width is saved as the measured error

	take the MINIMUM error from both continuum models as the error in the spectrum

	for each model:

		bootstrap errors, median
		plot continuum, continuum + emission, observations

	return: LUMINOSITY, FLUX, ERRORS, LINE NAME, LINE LAMBDA
		luminosity comes out in ergs / s
		flux comes out in ergs / s / cm^2

	to add: wavelength-dependent smoothing
	'''

	##### FIRST, CHECK RESOLUTION IN REGION OF HALPHA
	##### IF IT'S JUNK, THEN RETURN NOTHING!
	idx = (np.abs(obs_spec['rest_lam'] - 6563)).argmin()
	dellam = obs_spec['rest_lam'][idx+1] - obs_spec['rest_lam'][idx]
	if dellam > 14:
		print 'too low resolution, not measuring fluxes for '+sample_results['run_params']['objname']
		return None

	#### smoothing (prospector, magphys) in km/s
	smooth       = 200

	#### output names
	objname_short = sample_results['run_params']['objname'].replace(' ','_')
	base = '/Users/joel/code/python/threedhst_bsfh/plots/brownseds_np/magphys/line_fits/'
	out_em = base+objname_short+'_em_prosp.png'
	out_abs = base+objname_short+'_abs_prosp.png'
	out_absobs = base+objname_short+'_absobs.png'

    #### define emission lines to measure
    # we do this in sets of lines, which are unpacked at the end
	emline = np.array(['[OIII] 4959','[OIII] 5007',r'H$\beta$','[NII] 6549', '[NII] 6583', r'H$\alpha$','[OII] 3726','[OII] 3728'])
	em_wave = np.array([4958.92,5006.84,4861.33,6548.03,6583.41,6562.80,3726.1,3728.8])
	em_bbox = [(4700,5100),(6350,6680),(3600,3800)]
	amp_tied = [None, tiedfunc_oiii, None, None, tiedfunc_nii, None, None, tiedfunc_oii]
	lam_tied = [None, loiii_2, lhbeta, lnii_1, lnii_2, lhalpha,loii_1, loii_2]
	sig_tied = [None, soiii, None, None, snii, None, None, soii]
	nline = len(emline)

	#### now define sets of absorption lines to measure
	abslines = np.array(['halpha_wide', 'halpha_narrow', 'hbeta', 'hdelta_wide', 'hdelta_narrow'])
	nabs = len(abslines)             

	#### mapping between em_bbox and abslines!
	mod_abs_mapping = [np.where(abslines == 'hbeta')[0][0],np.where(abslines == 'halpha_wide')[0][0]]

	#### put all spectra in proper units (Lsun/AA, rest wavelength in Angstroms)
	# first, get rest-frame Prospector spectrum w/o emission lines
	# with z=0.0, it arrives in Lsun/Hz, and rest lambda
	# save redshift, restore at end
	z = sample_results['model'].params.get('zred', np.array(0.0))
	sample_results['model'].params['zred'] = np.array(0.0)
	prospflux_em,mags_em,sm = sample_results['model'].mean_model(sample_results['bfit']['maxprob_params'], sample_results['obs'], sps=sps)
	sample_results['model'].params['add_neb_emission'] = np.array(False)
	sample_results['model'].params['add_neb_continuum'] = np.array(False)
	prospflux,mags,sm = sample_results['model'].mean_model(sample_results['bfit']['maxprob_params'], sample_results['obs'], sps=sps)
	wav = sps.wavelengths
	sample_results['model'].params['add_neb_emission'] = np.array(2)
	sample_results['model'].params['add_neb_continuum'] = np.array(True)
	sample_results['model'].params['zred'] = z
	factor = 3e18 / wav**2
	prospflux *= factor
	prospflux_em *= factor

	# observed spectra arrive in Lsun/cm^2/AA
	# convert distance factor
	pc = 3.085677581467192e18  # cm
	dfactor = 4*np.pi*(pc*cosmo.luminosity_distance(magphys['metadata']['redshift']).value *
                    1e6)**2 * (1+magphys['metadata']['redshift'])
	obsflux = obs_spec['flux_lsun']*dfactor
	obslam = obs_spec['rest_lam']

	#### set up model iterables
	model_flux   = prospflux
	model_lam    = wav
	model_name   = 'Prospector'

	##### define fitter
	fitter = fitting.LevMarLSQFitter()

	#############################
	##### MODEL ABSORPTION ######
	#############################
	# this is now done entirely to 
	# calculate good model continuua
	# EQWs and fluxes are saved b/c we can 
	# calculate continuum fluxes with these
	absmods = []
	mod_abs_flux = np.zeros(shape=(nabs,2))
	mod_abs_eqw = np.zeros(shape=(nabs,2))
	mod_abs_lamcont = np.zeros(shape=(nabs,2))

	#### smooth and measure absorption lines
	flux_smooth = threed_dutils.smooth_spectrum(model_lam,model_flux,smooth,minlam=3e3,maxlam=3e8)
	out = threed_dutils.measure_abslines(model_lam,flux_smooth,plot=True)
	for kk in xrange(nabs): mod_abs_flux[kk,0] = out[abslines[kk]]['flux']
	for kk in xrange(nabs): mod_abs_eqw[kk,0] = out[abslines[kk]]['eqw']
	for kk in xrange(nabs): mod_abs_lamcont[kk,0] = out[abslines[kk]]['lam']

	flux_smooth = threed_dutils.smooth_spectrum(model_lam,prospflux_em,smooth,minlam=3e3,maxlam=3e8)
	out = threed_dutils.measure_abslines(model_lam,flux_smooth,plot=[out['fig'],out['ax']],alt_plot=True)
	for kk in xrange(nabs): mod_abs_flux[kk,1] = out.get(abslines[kk],{}).get('flux',0.0)
	for kk in xrange(nabs): mod_abs_eqw[kk,1] = out.get(abslines[kk],{}).get('eqw',0.0)
	for kk in xrange(nabs): mod_abs_lamcont[kk,1] = out.get(abslines[kk],{}).get('lam',0.0)

	#######################
	##### Dn(4000) ########
	#######################
	dn4000_mod = threed_dutils.measure_Dn4000(model_lam,flux_smooth,ax=out['ax'][5])

	out['fig'].tight_layout()
	out['fig'].savefig(out_abs, dpi=dpi)
	plt.close()

	#######################
	##### NOISE ###########
	#######################
	#### define emission line model and fitting region
	# use continuum_6400 to set up a good guess for starting positions
	continuum_6400 = obsflux[(np.abs(obslam-6400)).argmin()]
	emmod =  umbrella_model(em_wave, amp_tied, lam_tied, sig_tied, continuum_6400=continuum_6400)
	p_idx = np.zeros_like(obslam,dtype=bool)
	for bbox in em_bbox:p_idx[(obslam > bbox[0]) & (obslam < bbox[1])] = True

	### initial fit to emission
	# use for normalization purposes
	gauss_fit,_,_ = bootstrap(obslam[p_idx], obsflux[p_idx], emmod, fitter, np.min(obsflux[p_idx])*0.001, [1], 
	                          flux_flag=False, nboot=1)
	
	### find proper smoothing
	# by taking two highest EQW lines, and using the width
	# this assumes no scaling w/ wavelength, and that gas ~ stellar velocity dispersion
	tmpflux = np.zeros(nline)
	for j in xrange(nline):tmpflux[j] = getattr(gauss_fit, 'amplitude_'+str(j)).value*np.sqrt(2*np.pi*getattr(gauss_fit, 'stddev_'+str(j)).value**2)
	mlow = em_wave < 4000
	mmid = (em_wave >= 4000) & (em_wave < 5600)
	mhigh = em_wave >= 5600

	emnorm =  np.concatenate((gauss_fit.intercept_low_8+gauss_fit.slope_low_8*em_wave[mlow],
							  gauss_fit.intercept_mid_8+gauss_fit.slope_mid_8*em_wave[mmid],
							  gauss_fit.intercept_high_8+gauss_fit.slope_high_8*em_wave[mhigh]))
	tmpeqw = tmpflux / emnorm

	# take two highest EQW lines
	high_eqw = tmpeqw.argsort()[-2:][::-1]
	sigma_spec = np.mean(np.array(sig_ret(gauss_fit))[high_eqw]/em_wave[high_eqw])*c_kms
	sigma_spec = np.clip(sigma_spec,10.0,300.0)

	test_plot = False
	if test_plot:
		fig, ax = plt.subplots(1,1)
		plt.plot(obslam[p_idx],obsflux[p_idx], color='black')
		plt.plot(obslam[p_idx],gauss_fit(obslam[p_idx]),color='red')
		plt.show()
		print 1/0

	#### now interact with model
	# normalize continuum model to observations
	# subtract and fit Gaussian to residuals ---> constant noise
	# calculate and apply normalization factor
	# add a little padding for interpolation later
	model_norm = model_flux
	for kk in xrange(len(em_bbox)):
		m_idx = (model_lam > em_bbox[kk][0]-30) & (model_lam < em_bbox[kk][1]+30)

		#### here we have a bizarre, handmade mapping between
		#### mod_abs_lamcont and the ordered em_bbox vector.
		if kk == 2:
			blue = (model_lam > 3652.) & (model_lam < 3802.)
			norm_factor = np.mean(obsflux[blue])/np.mean(prospflux_em[blue])
			model_norm[m_idx] = model_flux[m_idx]*norm_factor
		else:
			mod_abs_idx = mod_abs_mapping[kk]
			close_fit = np.abs(model_lam-mod_abs_lamcont[mod_abs_idx,0]).argmin()
			norm_factor = gauss_fit[8](model_lam[close_fit])/(mod_abs_flux[mod_abs_idx,0]/mod_abs_eqw[mod_abs_idx,0])
			model_norm[m_idx] = model_flux[m_idx]*norm_factor

	#### adjust model wavelengths for the slight difference with published redshifts
	zadj = gauss_fit.mean_0 / 4958.92 - 1
	model_newlam = (1+zadj)*model_lam

	#### absorption model versus emission model
	test_plot = False
	if test_plot:
		p_idx_mod = ((model_newlam > em_bbox[0][0]) & (model_newlam < em_bbox[0][1])) | \
		            ((model_newlam > em_bbox[1][0]) & (model_newlam < em_bbox[1][1])) | \
		            ((model_newlam > em_bbox[2][0]) & (model_newlam < em_bbox[2][1]))
		fig, ax = plt.subplots(1,1)
		plt.plot(model_newlam[p_idx_mod], model_flux[p_idx_mod], color='black')
		plt.plot(obslam[p_idx], gauss_fit(obslam[p_idx]), color='red')
		plt.show()
		print 1/0

	#### interpolate model onto observations
	flux_interp = interp1d(model_newlam, model_norm, bounds_error = False, fill_value = 0)
	modflux = flux_interp(obslam[p_idx])

	#### smooth model to match observations
	smoothed_absflux = threed_dutils.smooth_spectrum(obslam[p_idx], modflux, sigma_spec)

	#### subtract model from observations
	residuals = obsflux[p_idx] - modflux

	#### mask emission lines
	masklam = 30
	mask = np.ones_like(obslam[p_idx],dtype=bool)
	for lam in em_wave: mask[(obslam[p_idx] > lam-masklam) & (obslam[p_idx] < lam+masklam)] = 0

	#### fit Gaussian to histogram of (model-obs) (clear structure visible sometimes...)
	hist, bin_edges = np.histogram(residuals[mask], density=True)
	ngauss_init = functional_models.Gaussian1D(mean=0.0,stddev=1e5,amplitude=1e-6)
	ngauss_init.mean.fixed = True
	ngauss_init.stddev.max = np.max(np.abs(bin_edges))*0.6
	noise_gauss,_,_ = bootstrap((bin_edges[1:]+bin_edges[:-1])/2., hist, ngauss_init, fitter, np.min(hist)*0.001, [1], 
                  flux_flag=False, nboot=1)

	##### TEST PLOT
	test_plot = False
	# (obs , model)
	if test_plot:
		fig, ax = plt.subplots(1,1)
		plt.plot(obslam[p_idx], obsflux[p_idx], color='black', drawstyle='steps-mid')
		plt.plot(obslam[p_idx], flux_interp(obslam[p_idx]), color='red')
		plt.show()
	# (obs - model)
	if test_plot:
		fig, ax = plt.subplots(1,1)
		plt.plot(obslam[p_idx], residuals, color='black', drawstyle='steps-mid')
		plt.show()

	# Gaussian fit to residuals in (obs - model)
	if test_plot:
		fig, ax = plt.subplots(1,1)
		plt.plot((bin_edges[1:]+bin_edges[:-1])/2.,hist, color='black')
		plt.plot((bin_edges[1:]+bin_edges[:-1])/2.,noise_gauss((bin_edges[1:]+bin_edges[:-1])/2.),color='red')
		plt.show()
		print 1/0

	#### measure noise!
	continuum = gauss_fit[8](em_wave)
	for ii, em in enumerate(emline): print 'measured '+em+' noise: '"{:.4f}".format(np.abs(noise_gauss.stddev.value)/continuum[ii])

	emline_noise = np.abs(noise_gauss.stddev.value)

	###############################################
	##### MEASURE FLUXES FROM OBSERVATIONS ########
	###############################################
	# get errors in parameters by bootstrapping
	# use lower of two error estimates
	# flux come out of bootstrap as (nlines,[median,errup,errdown])

	#### now measure emission line fluxes
	bfit_mod, emline_flux, emline_chain = bootstrap(obslam[p_idx],residuals,emmod,fitter,emline_noise,em_wave)

	#############################
	#### PLOT ALL THE THINGS ####
	#############################
	# set up figure
	# loop over models at some point? including above fit?
	fig, axarr = plt.subplots(1, 3, figsize = (25,5))
	for ii, bbox in enumerate(em_bbox):

		p_idx_em = ((obslam[p_idx] > bbox[0]) & (obslam[p_idx] < bbox[1]))

		# observations
		axarr[ii].plot(obslam[p_idx][p_idx_em],obsflux[p_idx][p_idx_em],color='black',drawstyle='steps-mid')
		# emission model + continuum model
		axarr[ii].plot(obslam[p_idx][p_idx_em],bfit_mod(obslam[p_idx][p_idx_em])+smoothed_absflux[p_idx_em],color='red')
		# continuum model + zeroth-order emission continuum
		axarr[ii].plot(obslam[p_idx][p_idx_em],smoothed_absflux[p_idx_em]+bfit_mod[8](obslam[p_idx][p_idx_em]),color='#1E90FF')

		axarr[ii].set_ylabel(r'flux [L$_{\odot}/\AA$]')
		axarr[ii].set_xlabel(r'$\lambda$ [$\AA$]')

		axarr[ii].xaxis.set_major_locator(MaxNLocator(5))
		axarr[ii].yaxis.get_major_formatter().set_powerlimits((0, 1))

		e_idx = (em_wave > bbox[0]) & (em_wave < bbox[1])
		nplot=0
		for kk in xrange(len(emline)):
			if e_idx[kk]:
				fmt = "{{0:{0}}}".format(".2e").format
				emline_str = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
				emline_str = emline_str.format(fmt(emline_flux[0,kk]), 
					                           fmt(emline_flux[1,kk]-emline_flux[0,kk]), 
					                           fmt(emline_flux[0,kk]-emline_flux[2,kk])) + ' erg/s, '
				axarr[ii].text(0.03, 0.93-0.085*nplot, 
					           emline[kk]+': '+emline_str,
					           fontsize=16, transform = axarr[ii].transAxes)
				nplot+=1

		axarr[ii].text(0.98, 0.93, 'em+abs model', fontsize=16, transform = axarr[ii].transAxes,ha='right',color='red')
		axarr[ii].text(0.98, 0.845, 'abs model', fontsize=16, transform = axarr[ii].transAxes,ha='right',color='#1E90FF')
		axarr[ii].text(0.98, 0.76, 'observations', fontsize=16, transform = axarr[ii].transAxes,ha='right')

	plt.tight_layout()
	plt.savefig(out_em, dpi = 300)
	plt.close()

	##############################################
	##### MEASURE OBSERVED ABSORPTION LINES ######
	##############################################
	# use redshift from emission line fit
	# currently, NO spectral smoothing; inspect to see if it's important for HDELTA ONLY
	zadj = bfit_mod.mean_0.value / 4958.92 - 1

	if sigma_spec < 200:
		to_convolve = (200.**2 - sigma_spec**2)**0.5
		obsflux = threed_dutils.smooth_spectrum(obslam/(1+zadj),obsflux,to_convolve,minlam=3e3,maxlam=3e8)

	#### bootstrap
	nboot = 100
	tobs_abs_flux,tobs_abs_eqw = [np.zeros(shape=(nabs,nboot)) for i in xrange(2)]

	for i in xrange(nboot):
		randomDelta = np.random.normal(0.,1.,len(obsflux))
		randomFlux = obsflux + randomDelta*emline_noise
		out = threed_dutils.measure_abslines(obslam/(1+zadj),randomFlux,plot=False)

		for kk in xrange(nabs): tobs_abs_flux[kk,i] = out[abslines[kk]]['flux']
		for kk in xrange(nabs): tobs_abs_eqw[kk,i] = out[abslines[kk]]['eqw']

	### we want errors for flux and eqw
	obs_abs_flux,obs_abs_eqw = [np.zeros(shape=(nabs,3)) for i in xrange(2)]

	for kk in xrange(nabs): obs_abs_flux[kk,:] = np.array([np.percentile(tobs_abs_flux[kk,:],50),
		                                                   np.percentile(tobs_abs_flux[kk,:],84),
		                                                   np.percentile(tobs_abs_flux[kk,:],16)])

	for kk in xrange(nabs): obs_abs_eqw[kk,:] =  np.array([np.percentile(tobs_abs_eqw[kk,:],50), 
		                                                   np.percentile(tobs_abs_eqw[kk,:],84), 
		                                                   np.percentile(tobs_abs_eqw[kk,:],16)])

	# bestfit, for plotting purposes
	out = threed_dutils.measure_abslines(obslam/(1+zadj),obsflux,plot=True)
	obs_lam_cont = np.zeros(nabs)
	for kk in xrange(nabs): obs_lam_cont[kk] = out[abslines[kk]]['lam']

	dn4000_obs = threed_dutils.measure_Dn4000(obslam,obsflux,ax=out['ax'][5])

	plt.tight_layout()
	plt.savefig(out_absobs, dpi=dpi)
	plt.close()

	out = {}
	# SAVE OBSERVATIONS
	obs = {}

	obs['lum'] = emline_flux[0,:]
	obs['lum_errup'] = emline_flux[1,:]
	obs['lum_errdown'] = emline_flux[2,:]

	obs['flux'] = emline_flux[0,:]  / dfactor / (1+magphys['metadata']['redshift'])
	obs['flux_errup'] = emline_flux[1,:]  / dfactor / (1+magphys['metadata']['redshift'])
	obs['flux_errdown'] = emline_flux[2,:]  / dfactor / (1+magphys['metadata']['redshift'])
	obs['flux_chain'] = emline_chain

	obs['dn4000'] = dn4000_obs
	obs['balmer_lum'] = obs_abs_flux
	obs['balmer_eqw_rest'] = obs_abs_eqw
	obs['balmer_flux'] = obs_abs_flux / dfactor / (1+magphys['metadata']['redshift'])
	obs['balmer_names'] = abslines
	obs['balmer_eqw_rest_chain'] = tobs_abs_eqw

	obs['continuum_obs'] = obs_abs_flux[:,0] / obs_abs_eqw[:,0] 
	obs['continuum_lam'] = obs_lam_cont

	out['obs'] = obs

	# SAVE MODEL
	mod = {}

	mod['balmer_lum'] = mod_abs_flux[:,0]
	mod['balmer_flux'] = mod_abs_flux[:,0] / dfactor / (1+magphys['metadata']['redshift'])
	mod['balmer_eqw_rest'] = mod_abs_eqw[:,0]
	mod['balmer_names'] = abslines

	mod['balmer_lum_addem'] = mod_abs_flux[:,1]
	mod['balmer_flux_addem'] = mod_abs_flux[:,1] / dfactor / (1+magphys['metadata']['redshift'])
	mod['balmer_eqw_rest_addem'] = mod_abs_eqw[:,1]

	mod['continuum_mod'] = mod_abs_flux[ii,:] / mod_abs_eqw[ii,:]
	mod['continuum_lam'] = mod_abs_lamcont[ii,:]

	mod['Dn4000'] = dn4000_mod

	out['mod'] = mod 

	out['em_name'] = emline
	out['em_lam'] = em_wave
	out['sigsmooth'] = sigma_spec

	return out
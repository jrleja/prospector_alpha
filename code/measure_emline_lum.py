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
		low = np.array(x) < 4300
		mid = (np.array(x) > 4300) & (np.array(x) < 5600)
		high = np.array(x) > 5600
		out[low] = slope_low*x[low]+intercept_low
		out[mid] = slope_mid*x[mid]+intercept_mid
		out[high] = slope_high*x[high]+intercept_high

		return out

	@staticmethod
	def fit_deriv(x, slope_low, intercept_low, slope_mid, intercept_mid, slope_high, intercept_high):
		"""One dimensional Line model derivative with respect to parameters"""

		low = np.array(x) < 4300
		mid = (np.array(x) > 4300) & (np.array(x) < 5600)
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
			for j in xrange(nlines):
				flux[i,j] = getattr(fit, 'amplitude_'+str(j)).value*np.sqrt(2*np.pi*getattr(fit, 'stddev_'+str(j)).value**2) * constants.L_sun.cgs.value

	# now get median + percentiles
	medianpar = np.percentile(params, 50,axis=1)
	fit.parameters = medianpar

	# we want errors for flux and eqw
	fluxout = np.array([np.percentile(flux, 50,axis=0),np.percentile(flux, 84,axis=0),np.percentile(flux, 16,axis=0)])

	return fit,fluxout

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

def tiedfunc_oiii(g1):
	amp = 2.98 * g1.amplitude_0
	return amp

def tiedfunc_nii(g1):
	amp = 2.93 * g1.amplitude_3
	return amp

def tiedfunc_sig(g1):
	return g1.stddev_0

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

def umbrella_model(lams, amp_tied, lam_tied, sig_tied):
	'''
	return model for [OIII], Hbeta, [NII], Halpha, + constant
	centers are all tied to redshift parameter
	'''

	#### ADD ALL MODELS FIRST
	for ii in xrange(len(lams)):
		if ii == 0:
			model = functional_models.Gaussian1D(amplitude=1e7, mean=lams[ii], stddev=5.0)
		else: 
			model += functional_models.Gaussian1D(amplitude=1e7, mean=lams[ii], stddev=5.0)

	# add slope + constant
	model += bLinear1D(intercept_low=1e7,intercept_high=1e7)

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

		getattr(model, 'stddev_'+str(ii)).max = 25

	return model

def measure_Dn4000(lam,flux):

	# D4000, defined as average flux ratio between
	# [4050,4250] and [3750,3950] (Bruzual 1983; Hamilton 1985)

	# ratio of continuua, measured as defined below 
	# blue: 3850-3950 . . . 4000-4100 (Balogh 1999)

	blue = (lam > 3850) & (lam < 3950)
	red  = (lam > 4000) & (lam < 4100)

	return np.mean(flux[red])/np.mean(flux[blue])

def measure(sample_results, obs_spec, magphys, sps, sigsmooth=None):
	
	'''
	measure rest-frame emission line luminosities using two different continuum models, MAGPHYS and Prospectr

	ALGORITHM:
	convert to rest-frame spectrum in proper units
	
	for each model:
		fetch Voigt model for Halpha, Hbeta
		smooth by [220 km/s, 50 km/s]
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

	#### smoothing (prospectr, magphys) in km/s
	smooth       = [200,200]

	#### output names
	base = '/Users/joel/code/python/threedhst_bsfh/plots/brownseds/magphys/line_fits/'
	out_em = [base+sample_results['run_params']['objname']+'_em_prosp.png',base+sample_results['run_params']['objname']+'_em_mag.png']
	out_abs = [base+sample_results['run_params']['objname']+'_abs_prosp.png',base+sample_results['run_params']['objname']+'_abs_mag.png']
	out_absobs = base+sample_results['run_params']['objname']+'_absobs.png'


    #### define emission lines to measure
    # we do this in sets of lines, which are unpacked at the end
	emline = np.array(['[OIII] 4959','[OIII] 5007',r'H$\beta$','[NII] 6549', '[NII] 6583', r'H$\alpha$'])
	em_wave = np.array([4958.92,5006.84,4861.33,6548.03,6583.41,6562.80,])
	em_bbox = [(4700,5100),(6350,6680)]
	amp_tied = [None, tiedfunc_oiii, None, None, tiedfunc_nii, None]
	lam_tied = [None, loiii_2, lhbeta, lnii_1, lnii_2, lhalpha]
	sig_tied = [None, soiii, None, None, snii, None]
	nline = len(emline)

	#### now define sets of absorption lines to measure
	absline = np.array([r'H$\delta$',r'H$\beta$',r'H$\alpha$'])
	abs_wave = np.array([4101.74,4861.33,6562.80])
	abs_bbox  = [(4020,4200),(4550,5100),(6350,6680)]
	nabs = len(abs_wave)             

	#### put all spectra in proper units (Lsun/AA, rest wavelength in Angstroms)
	# first, get rest-frame Prospectr spectrum w/o emission lines
	# with z=0.0, it arrives in Lsun/Hz, and rest lambda
	# save redshift, restore at end
	z = sample_results['model'].params.get('zred', np.array(0.0))
	sample_results['model'].params['zred'] = np.array(0.0)
	sample_results['model'].params['add_neb_emission'] = np.array(False)
	sample_results['model'].params['add_neb_continuum'] = np.array(False)
	prospflux,mags,sm = sample_results['model'].mean_model(sample_results['bfit']['maxprob_params'], sample_results['obs'], sps=sps)
	wav = sps.wavelengths
	sample_results['model'].params['add_neb_emission'] = np.array(2)
	sample_results['model'].params['add_neb_continuum'] = np.array(True)
	sample_results['model'].params['zred'] = z
	factor = 3e18 / wav**2
	prospflux *= factor

	# MAGPHYS spectra arrive in Lsun/AA
	# and observed lambda
	maglam = magphys['model']['lam']/(1+sample_results['model'].params.get('zred',0.0))
	magflux = magphys['model']['speclum']

	# observed spectra arrive in Lsun/cm^2/AA
	# convert distance factor
	pc = 3.085677581467192e18  # cm
	dfactor = 4*np.pi*(pc*cosmo.luminosity_distance(magphys['metadata']['redshift']).value *
                    1e6)**2 * (1+magphys['metadata']['redshift'])
	obsflux = obs_spec['flux_lsun']*dfactor
	obslam = obs_spec['rest_lam']

	#### set up model iterables
	model_fluxes = [prospflux,magflux]
	model_lams   = [wav,maglam]
	model_names  = ['Prospectr', 'MAGPHYS']
	nmodel       = len(model_names)

	##### define fitter
	fitter = fitting.LevMarLSQFitter()

	#######################
	##### ABSORPTION ######
	#######################
	# Voigt profile fails to fit the Balmer absorption cores
	# so smooth it with some kernel
	absmods = []
	mod_abs_flux = np.zeros(shape=(nmodel,nabs))
	mod_abs_eqw = np.zeros(shape=(nmodel,nabs))
	for jj in xrange(nmodel):
		fig, axarr = plt.subplots(1, 3, figsize = (20,5))
		tmp = []
		for kk in xrange(nabs):
			absmod = absobs_model(abs_wave[kk])

			p_idx = (model_lams[jj] > abs_bbox[kk][0]) & (model_lams[jj] < abs_bbox[kk][1])
			fit_lam = model_lams[jj][p_idx]
			fit_dat = model_fluxes[jj][p_idx]
			fit_smooth = threed_dutils.smooth_spectrum(fit_lam,fit_dat,smooth[jj])

			abs_fit,_ = bootstrap(fit_lam, fit_smooth, absmod, fitter, np.min(fit_smooth)*0.001, [1], 
		                          flux_flag=True, nboot=10)

			# Voigt profile is too complex to integrate analytically
			# so we're going in numerically
			#I1 = simps(abs_fit(fit_lam), fit_lam)
			#I2 = simps(abs_fit[1](fit_lam), fit_lam)
			#mod_abs_flux[jj,kk] = -(I2 - I1) * constants.L_sun.cgs.value
			mod_abs_flux[jj,kk] = np.squeeze(_[0])
			mod_abs_eqw[jj,kk] = mod_abs_flux[jj,kk] / abs_fit[1](abs_wave[kk]) / constants.L_sun.cgs.value

			tmp.append(abs_fit)

			axarr[kk].plot(fit_lam,fit_smooth,color='black',drawstyle='steps-mid')
			axarr[kk].plot(fit_lam,abs_fit(fit_lam),color='red')

			axarr[kk].set_ylabel(r'flux [L$_{\odot}/\AA$]')
			axarr[kk].set_xlabel(r'$\lambda$ [$\AA$]')

			axarr[kk].xaxis.set_major_locator(MaxNLocator(5))
			axarr[kk].yaxis.get_major_formatter().set_powerlimits((0, 1))

			axarr[kk].text(0.03, 0.07, 
				           absline[kk]+': '+"{:.2e}".format(mod_abs_flux[jj,kk])+' erg/s, ' + "{:.2f}".format(mod_abs_eqw[jj,kk])+r' $\AA$',
				           fontsize=16, transform = axarr[kk].transAxes)

			axarr[kk].text(0.03, 0.155, model_names[jj], fontsize=16, transform = axarr[kk].transAxes)
			axarr[kk].text(0.98, 0.07, 'model', fontsize=16, transform = axarr[kk].transAxes,ha='right',color='black')
			axarr[kk].text(0.98, 0.155, 'abs fit', fontsize=16, transform = axarr[kk].transAxes,ha='right',color='red')

		absmods.append(tmp)
		plt.tight_layout()
		plt.savefig(out_abs[jj], dpi = 300)
		plt.close()

	#######################
	##### Dn(4000) ########
	#######################
	dn4000_obs = measure_Dn4000(obslam,obsflux)
	dn4000_mod = []
	for flux,lam in zip(model_fluxes,model_lams):
		dn4000_mod.append(measure_Dn4000(lam,flux))
	dn4000_mod = np.array(dn4000_mod)


	#######################
	##### NOISE ###########
	#######################
	# set up emission line outputs
	# flux and resteqw are lists because we'll define these locally for each set
	emline_noise = np.zeros(nmodel)
	emline_flux = []
	residuals   = []
	eqw_percs = []
	smoothed_absflux = []

	#### define model, fitting region
	emmod =  umbrella_model(em_wave, amp_tied, lam_tied, sig_tied)
	p_idx = np.zeros_like(obslam,dtype=bool)
	for bbox in em_bbox:p_idx[(obslam > bbox[0]) & (obslam < bbox[1])] = True

	### initial fit to emission
	gauss_fit,_ = bootstrap(obslam[p_idx], obsflux[p_idx], emmod, fitter, np.min(obsflux[p_idx])*0.001, [1], 
	                          flux_flag=False, nboot=10)
	
	### find proper smoothing
	tmpflux = np.zeros(nline)
	for j in xrange(nline):tmpflux[j] = getattr(gauss_fit, 'amplitude_'+str(j)).value*np.sqrt(2*np.pi*getattr(gauss_fit, 'stddev_'+str(j)).value**2)
	mlow = em_wave < 5600
	emnorm =  np.concatenate((gauss_fit.intercept_low_6+gauss_fit.slope_low_6*em_wave[mlow],
							  gauss_fit.intercept_high_6+gauss_fit.slope_high_6*em_wave[~mlow]))
	tmpeqw = tmpflux / emnorm

	# take two highest EQW lines
	high_eqw = tmpeqw.argsort()[-2:][::-1]
	sigma_spec = np.mean(np.array(sig_ret(gauss_fit))[high_eqw]/em_wave[high_eqw])*c_kms

	test_plot = False
	if test_plot:
		fig, ax = plt.subplots(1,1)
		plt.plot(obslam[p_idx],obsflux[p_idx], color='black')
		plt.plot(obslam[p_idx],gauss_fit(obslam[p_idx]),color='red')
		plt.show()
		print 1/0

	#### now interact with model
	for jj in xrange(nmodel):

		model_norm = model_fluxes[jj]

		# add a little padding for interpolation later
		for kk in xrange(len(em_bbox)):
			m_idx = (model_lams[jj] > em_bbox[kk][0]-30) & (model_lams[jj] < em_bbox[kk][1]+30)
			model_norm[m_idx] = model_fluxes[jj][m_idx]*gauss_fit[6](model_lams[jj][m_idx])/absmods[jj][kk+1][1](model_lams[jj][m_idx])

		zadj = gauss_fit.mean_0 / 4958.92 - 1
		model_newlam = (1+zadj)*model_lams[jj]

		# absorption model versus emission model
		test_plot = False
		if test_plot:
			p_idx_mod = ((model_newlam > em_bbox[0][0]) & (model_newlam < em_bbox[0][1])) | ((model_newlam > em_bbox[1][0]) & (model_newlam < em_bbox[1][1]))
			fig, ax = plt.subplots(1,1)
			plt.plot(model_newlam[p_idx_mod], model_norm[p_idx_mod], color='black')
			plt.plot(obslam[p_idx], gauss_fit(obslam[p_idx]), color='red')
			plt.show()

		#### interpolate
		flux_interp = interp1d(model_newlam, model_norm, bounds_error = False, fill_value = 0)
		modflux = flux_interp(obslam[p_idx])

		#### smooth
		modflux = threed_dutils.smooth_spectrum(obslam[p_idx], modflux, sigma_spec)

		#### subtract
		resid = obsflux[p_idx] - modflux

		smoothed_absflux.append(modflux)
		residuals.append(resid)

		#### mask emission lines
		masklam = 30
		mask = np.ones_like(obslam[p_idx],dtype=bool)
		for lam in em_wave: mask[(obslam[p_idx] > lam-masklam) & (obslam[p_idx] < lam+masklam)] = 0

		#### fit Gaussian to residuals (clear structure visible sometimes...)
		hist, bin_edges = np.histogram(resid[mask], density=True)
		ngauss_init = functional_models.Gaussian1D(mean=0.0,stddev=1e5,amplitude=1e-6)
		ngauss_init.mean.fixed = True
		ngauss_init.stddev.max = np.max(np.abs(bin_edges))*0.6
		noise_gauss,_ = bootstrap((bin_edges[1:]+bin_edges[:-1])/2., hist, ngauss_init, fitter, np.min(hist)*0.001, [1], 
                      flux_flag=False, nboot=10)

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
			plt.plot(obslam[p_idx], resid, color='black', drawstyle='steps-mid')
			plt.show()

		# Gaussian fit to residuals in (obs - model)
		if test_plot:
			fig, ax = plt.subplots(1,1)
			plt.plot((bin_edges[1:]+bin_edges[:-1])/2.,hist, color='black')
			plt.plot((bin_edges[1:]+bin_edges[:-1])/2.,noise_gauss((bin_edges[1:]+bin_edges[:-1])/2.),color='red')
			plt.show()
			print 1/0

		#### measure noise!
		mlow = em_wave < 5600
		continuum = gauss_fit[6](em_wave)
		for ii, em in enumerate(emline): print 'measured '+em+' noise: '"{:.4f}".format(np.abs(noise_gauss.stddev.value)/continuum[ii])

		emline_noise[jj] = np.abs(noise_gauss.stddev.value)

	#########################################
	##### MEASURED FROM OBSERVATIONS ########
	#########################################
	# get errors in parameters by bootstrapping
	# use lower of two error estimates
	# flux come out of bootstrap as (nlines,[median,errup,errdown])
	tnoise = np.min(emline_noise)

	#### now interact with model
	for jj in xrange(nmodel):

		bfit_mod, emline_flux_local = bootstrap(obslam[p_idx],residuals[jj],emmod,fitter,tnoise,em_wave)
		emline_flux.append(emline_flux_local)

		#### measure fluxes, EQW
		# estimate continuum
		low_idx = (obslam[p_idx] > 5030) & (obslam[p_idx] < 5100)
		cont_low = np.mean(bfit_mod[6](obslam[p_idx][low_idx]) + smoothed_absflux[jj][low_idx])*constants.L_sun.cgs.value

		high_idx = (obslam[p_idx] > 6400) & (obslam[p_idx] < 6500)
		cont_high = np.mean(bfit_mod[6](obslam[p_idx][high_idx]) + smoothed_absflux[jj][high_idx])*constants.L_sun.cgs.value

		eqw = np.concatenate((emline_flux_local[:,:3] / cont_low, emline_flux_local[:,3:] / cont_high),axis=1)
		eqw_percs.append(eqw)

		#############################
		#### PLOT ALL THE THINGS ####
		#############################
		# set up figure
		# loop over models at some point? including above fit?
		fig, axarr = plt.subplots(1, 2, figsize = (20,5))
		for ii, bbox in enumerate(em_bbox):

			p_idx_em = ((obslam[p_idx] > bbox[0]) & (obslam[p_idx] < bbox[1]))

			axarr[ii].plot(obslam[p_idx][p_idx_em],obsflux[p_idx][p_idx_em],color='black',drawstyle='steps-mid')
			axarr[ii].plot(obslam[p_idx][p_idx_em],bfit_mod(obslam[p_idx][p_idx_em])+smoothed_absflux[jj][p_idx_em],color='red')

			axarr[ii].plot(obslam[p_idx][p_idx_em],smoothed_absflux[jj][p_idx_em]+bfit_mod[6](obslam[p_idx][p_idx_em]),color='#1E90FF')

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
					emline_str = emline_str.format(fmt(emline_flux_local[0,kk]), 
						                           fmt(emline_flux_local[1,kk]-emline_flux_local[0,kk]), 
						                           fmt(emline_flux_local[0,kk]-emline_flux_local[2,kk])) + ' erg/s, '
					axarr[ii].text(0.03, 0.93-0.085*nplot, 
						           emline[kk]+': '+emline_str,
						           fontsize=16, transform = axarr[ii].transAxes)
					nplot+=1

			axarr[ii].text(0.03, 0.93-0.085*nplot, model_names[jj], fontsize=16, transform = axarr[ii].transAxes)
			axarr[ii].text(0.98, 0.93, 'em+abs model', fontsize=16, transform = axarr[ii].transAxes,ha='right',color='red')
			axarr[ii].text(0.98, 0.845, 'abs model', fontsize=16, transform = axarr[ii].transAxes,ha='right',color='#1E90FF')
			axarr[ii].text(0.98, 0.76, 'observations', fontsize=16, transform = axarr[ii].transAxes,ha='right')


		plt.tight_layout()
		plt.savefig(out_em[jj], dpi = 300)
		plt.close()


	##### MEASURE OBSERVED ABSORPTION LINES
	# currently only hdelta
	absobs = absobs_model(abs_wave[0])
	p_idx = (obslam > abs_bbox[0][0]) & (obslam < abs_bbox[0][1])
	bfit_abs_mod, abs_flux = bootstrap(obslam[p_idx],obsflux[p_idx],absobs,fitter,tnoise,abs_wave[0])

	absobs.parameters = bfit_abs_mod.parameters
	absobs_eqw =  abs_flux / absobs[1](abs_wave[0]) / constants.L_sun.cgs.value

	# plot
	fig, ax = plt.subplots(1, 1, figsize = (10,10))
	ax.plot(obslam[p_idx],obsflux[p_idx],color='black',drawstyle='steps-mid')
	ax.plot(obslam[p_idx],bfit_abs_mod(obslam[p_idx]),color='red')

	ax.set_ylabel(r'flux [L$_{\odot}/\AA$]')
	ax.set_xlabel(r'$\lambda$ [$\AA$]')

	ax.xaxis.set_major_locator(MaxNLocator(5))
	ax.yaxis.get_major_formatter().set_powerlimits((0, 1))

	fmt = "{{0:{0}}}".format(".2e").format
	absline_str = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
	absline_str = absline_str.format(fmt(abs_flux[0][0]), 
		                             fmt(abs_flux[0][0]-abs_flux[1][0]), 
		                             fmt(abs_flux[2][0]-abs_flux[0][0])) + ' erg/s, '
	ax.text(0.03, 0.07, 
		    absline[0]+': '+absline_str + "{:.2f}".format(absobs_eqw[0][0]) + r' $\AA$',
		    fontsize=16, transform = ax.transAxes)

	ax.text(0.98, 0.11, 'abs model', fontsize=16, transform = ax.transAxes,ha='right',color='red')
	ax.text(0.98, 0.07, 'observations', fontsize=16, transform = ax.transAxes,ha='right')

	plt.savefig(out_absobs, dpi = 300)
	plt.close()

	out = {}
	# save fluxes
	for ii, mod in enumerate(model_names):
		temp = {}

		temp['lum'] = emline_flux[ii][0,:]
		temp['lum_errup'] = emline_flux[ii][1,:]
		temp['lum_errdown'] = emline_flux[ii][2,:]

		temp['flux'] = emline_flux[ii][0,:]  / dfactor / (1+magphys['metadata']['redshift'])
		temp['flux_errup'] = emline_flux[ii][1,:]  / dfactor / (1+magphys['metadata']['redshift'])
		temp['flux_errdown'] = emline_flux[ii][2,:]  / dfactor / (1+magphys['metadata']['redshift'])

		temp['eqw_rest'] = eqw_percs[ii]
		temp['Dn4000'] = dn4000_mod[ii]

		temp['hdelta_lum'] = mod_abs_flux[ii,:]
		temp['hdelta_flux'] = mod_abs_flux[ii,:] / dfactor / (1+magphys['metadata']['redshift'])
		temp['hdelta_eqw_rest'] = mod_abs_eqw[ii,:]

		out[mod] = temp

	obs = {}
	obs['dn4000'] = dn4000_obs
	obs['hdelta_eqw'] = absobs_eqw
	obs['hdelta_lum'] = abs_flux[0][0]
	obs['hdelta_lum_errup'] = abs_flux[1][0]
	obs['hdelta_lum_errdown'] = abs_flux[2][0]

	obs['hdelta_flux'] = abs_flux[0][0] / dfactor / (1+magphys['metadata']['redshift'])
	obs['hdelta_flux_errup'] = abs_flux[1][0]/ dfactor / (1+magphys['metadata']['redshift'])
	obs['hdelta_flux_errdown'] = abs_flux[2][0]/ dfactor / (1+magphys['metadata']['redshift'])

	out['obs'] = obs 
	out['em_name'] = emline
	out['em_lam'] = em_wave
	out['sigsmooth'] = sigma_spec

	return out
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
maxfev = 300

class jGaussian1D(core.Fittable1DModel):
	"""
	redefined to add a constant
	"""

	amplitude = Parameter(default=1)
	mean = Parameter(default=0)
	stddev = Parameter(default=1)
	constant = Parameter(default=0)
	slope = Parameter(default=0)
	_bounding_box = 'auto'

	def bounding_box_default(self, factor=5.5):
		x0 = self.mean.value
		dx = factor * self.stddev

		return (x0 - dx, x0 + dx)

	@staticmethod
	def evaluate(x, amplitude, mean, stddev, constant,slope):
		"""
		Gaussian1D model function.
		"""
		return amplitude * np.exp(- 0.5 * (x - mean) ** 2 / stddev ** 2) + constant + slope*x

	@staticmethod
	def fit_deriv(x, amplitude, mean, stddev, constant,slope):
		"""
		Gaussian1D model function derivatives.
		"""

		d_amplitude = np.exp(-0.5 / stddev ** 2 * (x - mean) ** 2)
		d_mean = amplitude * d_amplitude * (x - mean) / stddev ** 2
		d_stddev = amplitude * d_amplitude * (x - mean) ** 2 / stddev ** 3
		d_constant = np.zeros_like(x)
		d_slope = x
		return [d_amplitude, d_mean, d_stddev, d_constant, d_slope]

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

class jVoigt1D(core.Fittable1DModel):
	x_0 = Parameter(default=0)
	amplitude_L = Parameter(default=1)
	fwhm_L = Parameter(default=2/np.pi)
	fwhm_G = Parameter(default=np.log(2))
	constant = Parameter(default=0)
	slope = Parameter(default=0)

	_abcd = np.array([
	    [-1.2150, -1.3509, -1.2150, -1.3509],  # A
	    [1.2359, 0.3786, -1.2359, -0.3786],    # B
	    [-0.3085, 0.5906, -0.3085, 0.5906],    # C
	    [0.0210, -1.1858, -0.0210, 1.1858]])   # D

	@classmethod
	def evaluate(cls, x, x_0, amplitude_L, fwhm_L, fwhm_G,constant,slope):

		A, B, C, D = cls._abcd
		sqrt_ln2 = np.sqrt(np.log(2))
		X = (x - x_0) * 2 * sqrt_ln2 / fwhm_G
		X = np.atleast_1d(X)[..., np.newaxis]
		Y = fwhm_L * sqrt_ln2 / fwhm_G
		Y = np.atleast_1d(Y)[..., np.newaxis]

		V = np.sum((C * (Y - A) + D * (X - B))/(((Y - A) ** 2 + (X - B) ** 2)), axis=-1)

		return (fwhm_L * amplitude_L * np.sqrt(np.pi) * sqrt_ln2 / fwhm_G) * V + constant + slope*x

	@classmethod
	def fit_deriv(cls, x, x_0, amplitude_L, fwhm_L, fwhm_G,constant,slope):

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
		        -cnt * (V + (sqrt_ln2 / fwhm_G) * (2 * (x - x_0) * dVdx + fwhm_L * dVdy)) / fwhm_G,
		        np.ones_like(x),
		        x]
		return dyda

def bootstrap(obslam, obsflux, model, fitter, noise, line_lam):

	# number of bootstrap fits
	nboot = 300

	# count lines
	nlines = len(line_lam)

	# desired output
	params = np.zeros(shape=(len(model.parameters),nboot))
	flux   = np.zeros(shape=(nboot,nlines))
	rest_eqw = np.zeros(shape=(nboot,nlines))

	# random data sets are generated and fitted
	for i in xrange(nboot):
		randomDelta = np.random.normal(0., 1.,len(obsflux))
		randomFlux = obsflux + randomDelta*noise
		fit = fitter(model, obslam, randomFlux,maxiter=maxfev)
		params[:,i] = fit.parameters

		# calculate emission line flux + rest EQW
		for j in xrange(nlines):
			flux[i,j] = getattr(fit, 'amplitude_'+str(j)).value*np.sqrt(2*np.pi*getattr(fit, 'stddev_'+str(j)).value**2) * constants.L_sun.cgs.value
			rest_eqw[i,j] = flux[i,j] / constants.L_sun.cgs.value / (fit.constant_0 + fit.slope_0*line_lam[j])

	# now get median + percentiles
	medianpar = np.percentile(params, 50,axis=1)

	# we want errors for flux and eqw
	fluxout = np.array([np.percentile(flux, 50,axis=0),np.percentile(flux, 84,axis=0),np.percentile(flux, 16,axis=0)])
	eqwout = np.array([np.percentile(rest_eqw, 50,axis=0),np.percentile(rest_eqw, 84,axis=0),np.percentile(rest_eqw, 16,axis=0)])

	return medianpar,fluxout,eqwout

class observify():

	def __init__(self,xfit,yfit, eline_lam, fitrange):
		self.mask = self.create_mask(xfit, eline_lam, fitrange)
		self.xfit = xfit[self.mask]
		self.yfit = yfit[self.mask]

	def observify(self, pars, x, y):
		# pars is the following:
		# [y-intercept, slope, redshift, sigma]

		# first redshift
		xred = (1 + pars[2]) * x

		# now normalize
		yflux = y + (pars[0] + pars[1]*xred)

		# now smooth
		ynew = threed_dutils.smooth_spectrum(xred,yflux,pars[3])

		return xred, ynew

	def fit_observify(self, pars, x, y):

		# interpolate onto observations, return residuals
		xint,yint=self.observify(pars,x,y)
		flux_interp = interp1d(xint,yint, bounds_error = False, fill_value = 0)
		return self.yfit - flux_interp(self.xfit)

	def create_mask(self, lam, emline_lam, bounds):

			#### apply boundary region
			mask = np.ones_like(lam,dtype=bool)
			masked = (lam < bounds[0]) | (lam > bounds[1])
			mask[masked] = 0

			#### mask emission lines
			masklam = 20
			for elam in emline_lam: mask[(lam > elam-masklam) & (lam < elam+masklam)] = 0

			return mask


def absline_model(lam):
	'''
	return model for region + set of lambdas
	might need to complicate by setting custom del_lam, fwhm_max, etc for different regions
	'''

	### how much do we allow centroid to vary?
	del_lam = 2.0

	### what is the maximum line width in Angstroms?
	fwhm_max = 8.0

	### build model
	voi = jVoigt1D(amplitude_L=-5e6, x_0=lam, fwhm_L=5.0, fwhm_G=5.0, constant=5e6)
	voi.x_0.max = lam+del_lam
	voi.x_0.min = lam-del_lam
	#voi.fwhm_L.max = fwhm_max
	#voi.fwhm_G.max = fwhm_max

	return voi

def halpha_tied_to_nii_1(g1):
	return g1.mean_0+14

def nii_2_tied_to_nii_1(g1):
	return g1.mean_0+34

def oiii_2_tied_to_oiii_1(g1):
	return g1.mean_0+48

def hbeta_tied_to_oiii_1(g1):
	return g1.mean_0-97.67

def tiedfunc_oiii(g1):
	amp = 2.98 * g1.amplitude_0
	return amp

def tiedfunc_nii(g1):
	amp = 2.93 * g1.amplitude_0
	return amp

def tiedfunc_sig(g1):
	return g1.stddev_0

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

def emline_model(lams, emtied,emtied_lam):
	'''
	return model for region + set of lambdas
	might need to complicate by setting custom del_lam, stddev_max, etc for different regions
	'''

	### how much do we allow centroid to vary?
	#del_lam = 20.0

	### what is the maximum line width in Angstroms?
	stddev_max = 40

	### build model
	gauss_init = None
	for lam in lams:
		gtemp = jGaussian1D(amplitude=1e7, mean=lam, stddev=5.0, constant=0.2e7)
		#gtemp.mean.max = lam+del_lam
		#gtemp.mean.min = lam-del_lam
		gtemp.stddev.max = stddev_max

		#### sum models
		# remove constants if we have one
		if gauss_init is None:
			gauss_init = gtemp
		else:
			gtemp.constant = 0.0
			gtemp.slope = 0.0
			gtemp.constant.fixed = True
			gtemp.slope.fixed = True
			gauss_init += gtemp

	# if it exists, tie the second amplitude to the first one
	if emtied is not None:
		gauss_init.amplitude_1.tied = emtied
		gauss_init.stddev_1.tied = tiedfunc_sig

	if emtied_lam is not None:
		gauss_init.mean_1.tied = emtied_lam[0]
		gauss_init.mean_2.tied = emtied_lam[1]

	return gauss_init

def measure(sample_results, obs_spec, magphys, sps, sigsmooth=None):
	
	'''
	measure rest-frame emission line luminosities
	and absorption strengths for two different models:
		(1) MAGPHYS
		(2) PROSPECTR

	ALGORITHM:
	convert to rest-frame spectrum in proper units
	
	setup plot
	for each set of absorption lines:
		for each model:
			fetch Voigt model
			smooth by [220 km/s, 50 km/s]
			fit Voigt
			plot
	save plot

	setup plot
	for each set of emission lines:

		fetch emission model (function)
		fit observed emission region (inital fit for normalization + width of lines)
	
		loop to measure noise in spectra
		for each model spectra:
			normalize model spectra to observations
				additive: calculate linear continuum for both, calculate difference, add to model
			smooth model, using avg(SIGMA) from all measured emission lines
			subtract model from observations
			
			mask emission lines, measure residuals
			fit gaussian to distribution of residuals to get noise
			cap noise at 1% < (noise/flux) < 20%
		
		bootstrap errors + best fit to emission line region using provided noise (only use Prospectr noise, not that it matters)

		plot fits

	return: LUMINOSITY, FLUX, ERRORS, LINE NAME, LINE LAMBDA
		luminosity comes out in ergs / s
		flux comes out in ergs / s / cm^2

	ways to improve:
		-- do some emission lines need separate component?
		--simultaneous fitting of nearby emission lines, such as [NII] doublet + Halpha

	'''

	##### FIRST, CHECK RESOLUTION IN REGION OF HALPHA
	##### IF IT'S JUNK, THEN RETURN NOTHING!
	idx = (np.abs(obs_spec['rest_lam'] - 6563)).argmin()
	dellam = obs_spec['rest_lam'][idx+1] - obs_spec['rest_lam'][idx]
	if dellam > 14:
		print 'too low resolution, not measuring fluxes for '+sample_results['run_params']['objname']
		return None

	#### min / max noise measured in spectrum
	maxnoise = np.inf
	minnoise = 0.01

	#### smoothing (prospectr, magphys) in km/s
	smooth       = [200,50]

	#### output names
	base = '/Users/joel/code/python/threedhst_bsfh/plots/brownseds/magphys/line_fits/'
	out_em = base+sample_results['run_params']['objname']+'_em.png'
	out_abs = base+sample_results['run_params']['objname']+'_abs.png'

    #### define emission lines to measure
    # we do this in sets of lines, which are unpacked at the end
	emline = np.array([['[OIII] 4959','[OIII] 5007',r'H$\beta$'],
		               ['[NII] 6549', '[NII] 6583', r'H$\alpha$']])
	em_wave = np.array([[4959,5007,4861.33],
		                [6549,6583,6563,]])
	em_bbox  = [(4700,5400),(6350,6680)]
	emtied = [tiedfunc_oiii, tiedfunc_nii]
	emtied_lam = [[oiii_2_tied_to_oiii_1,hbeta_tied_to_oiii_1],[nii_2_tied_to_nii_1,halpha_tied_to_nii_1]]
	nset_em = emline.shape[0]

	#### now define sets of absorption lines to measure
	absline = np.array([[r'H$\beta$'],
		                [r'H$\alpha$']])
	abs_wave = np.array([[4861.33],\
		                 [6563]])
	abs_bbox  = [(4550,5100),(6300,6840)]
	nset_abs = absline.shape[0]               

	#### put all spectra in proper units (Lsun/AA, rest wavelength in Angstroms)
	# first, get rest-frame Prospectr spectrum w/o emission lines
	# with z=0.0, it arrives in Lsun/Hz, and rest lambda
	# save redshift, restore at end
	z      = sample_results['model'].params.get('zred', np.array(0.0))
	sample_results['model'].params['zred'] = np.array(0.0)
	sample_results['model'].params['add_neb_emission'] = np.array(False)
	sample_results['model'].params['add_neb_continuum'] = np.array(False)
	prospflux,mags,sm = sample_results['model'].mean_model(sample_results['quantiles']['maxprob_params'], sample_results['obs'], sps=sps)
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
                    1e6)**2 / (1+magphys['metadata']['redshift'])
	obsflux = obs_spec['flux_lsun']*dfactor
	obslam = obs_spec['rest_lam']

	##### Prepare for loop over absorption lines
	# set up model iterables + outputs
	model_fluxes = [prospflux,magflux]
	model_lams   = [wav,maglam]
	model_names  = ['Prospectr', 'MAGPHYS']
	nmodel       = len(model_names)

	absline_flux = np.zeros(shape=(nset_abs,nmodel))
	absline_eqw = np.zeros(shape=(nset_abs,nmodel))
	absline_constant = np.zeros(shape=(nset_abs,nmodel))
	absline_slope = np.zeros(shape=(nset_abs,nmodel))

	##### define fitter
	fitter = fitting.LevMarLSQFitter()

	######################
	##### OBSERVIFY ######
	######################
	fitrange = [np.min(em_bbox)-300, np.max(em_bbox)]
	fitrange = [5000,6000]
	ob = observify(obslam,obsflux,np.ravel(em_wave),fitrange)

	obsfits = []
	fig, ax = plt.subplots(1, 1, figsize = (10,10))

	for ii, mflux in enumerate(model_fluxes):

		mod_mask = ob.create_mask(model_lams[ii], np.ravel(em_wave), fitrange)
		x = model_lams[ii][mod_mask]
		y = mflux[mod_mask]

		# \vec{x}_init
		# y-intercept, slope, additional redshift, sigma
		obs_init = [0.0, 0.0, 0.0, 500]
		try:
			obspars, cov_x,fit_info,msg,_ = leastsq(ob.fit_observify, np.array(obs_init), args=(x,y), maxfev=maxfev,full_output=True)
		except TypeError as e:
			print e
			print 1/0
		obsfits.append(obspars)

		fig, ax = plt.subplots(1, 1, figsize = (10,5))
		ax.plot(ob.xfit, ob.yfit,color='black')
		ax.plot(x, y,color='red')
		xp, yp = ob.observify(obspars,x,y)
		ax.plot(xp,yp,color='blue')
		ax.set_xlim(5100,5300)
		plt.show()
		print 1/0



	#### set up figure
	fig, axarr = plt.subplots(nset_abs, nmodel, figsize = (5*nset_abs, 5*nmodel))

	
	#######################
	##### ABSORPTION ######
	#######################
	# Voigt profile fails to fit the Balmer absorption cores
	# smoothed with 150 km/s resolution
	for ii in xrange(nset_abs):
		absmod = absline_model(abs_wave[ii][0])
		for jj in xrange(nmodel):
			p_idx = (model_lams[jj] > abs_bbox[ii][0]) & (model_lams[jj] < abs_bbox[ii][1])
			fit_lam = model_lams[jj][p_idx]
			fit_dat = model_fluxes[jj][p_idx]
			fit_smooth = threed_dutils.smooth_spectrum(fit_lam,fit_dat,smooth[jj])
			abs_fit = fitter(absmod, fit_lam, fit_smooth, maxiter=maxfev)
			
			# Voigt profile is too complex to integrate analytically
			# so we're going in numerically
			I1 = simps(abs_fit(fit_lam), fit_lam)
			I2 = simps(np.ones_like(fit_lam)+abs_fit.constant.value+abs_fit.slope.value*fit_lam, fit_lam)
			absline_flux[ii,jj] = -(I2 - I1) * constants.L_sun.cgs.value
			absline_eqw[ii,jj] = absline_flux[ii,jj]/(abs_fit.constant.value + abs_fit.slope.value*abs_wave[ii][0]) / constants.L_sun.cgs.value
			absline_constant[ii,jj] = abs_fit.constant.value
			absline_slope[ii,jj] = abs_fit.slope.value


			#### plot 			
			axarr[ii,jj].plot(fit_lam,fit_smooth,color='black')
			axarr[ii,jj].plot(fit_lam,abs_fit(fit_lam),color='red')
			axarr[ii,jj].set_ylabel(r'model [L$_{\odot}/\AA$]')
			axarr[ii,jj].set_xlabel(r'$\lambda$ [$\AA$]')
			axarr[ii,jj].text(0.04, 0.935, absline[ii][0],fontsize=16, transform = axarr[ii,jj].transAxes)
			axarr[ii,jj].text(0.04, 0.87, r'flux='+"{:.2e}".format(absline_flux[ii,jj])+' erg/s',fontsize=16, transform = axarr[ii,jj].transAxes)
			axarr[ii,jj].text(0.04, 0.805, r'EQW='+"{:.2f}".format(absline_eqw[ii,jj])+r' $\AA$',fontsize=16, transform = axarr[ii,jj].transAxes)
			axarr[ii,jj].text(0.975, 0.938, model_names[jj],fontsize=16, transform = axarr[ii,jj].transAxes,ha='right')

			axarr[ii,jj].xaxis.set_major_locator(MaxNLocator(5))
			axarr[ii,jj].yaxis.get_major_formatter().set_powerlimits((0, 1))
			axarr[ii,jj].set_ylim(np.min(fit_smooth)*0.9,np.max(fit_smooth)*1.3)
	
	plt.tight_layout()
	plt.savefig(out_abs, dpi = 300)
	plt.close()


	#######################
	##### EMISSION ########
	#######################
	# set up emission line outputs
	# flux and resteqw are lists because we'll define these locally for each set
	emline_noise = np.zeros(shape=(nset_em,nmodel))
	emline_flux = []
	emline_rest_eqw = []

	##### set up figure
	fig, axarr = plt.subplots(nset_em, 1, figsize = (5*nset_em,10))

	for ii in xrange(nset_em):

		#### define model, region to fit to
		emmod = emline_model(em_wave[ii],emtied[ii],emtied_lam[ii])
		p_idx = (obslam > em_bbox[ii][0]) & (obslam < em_bbox[ii][1])

		### initial fit to emission
		gauss_fit = fitter(emmod, obslam[p_idx], obsflux[p_idx],maxiter= maxfev)

		test_plot = False
		if test_plot:
			fig, ax = plt.subplots(1,1)
			plt.plot(obslam[p_idx],obsflux[p_idx], color='black')
			plt.plot(obslam[p_idx],gauss_fit(obslam[p_idx]),color='red')
			plt.show()
			print 1/0

		#### now interact with model
		for jj in xrange(nmodel):
			
			#### normalize model spectra to observations
			norm =  (gauss_fit.constant_0+gauss_fit.slope_0*model_lams[jj]) - (absline_constant[ii,jj]+absline_slope[ii,jj]*model_lams[jj])
			model_norm = model_fluxes[jj] + norm

			#### smooth spectrum
			sigma_spec = np.mean(sig_ret(gauss_fit)/em_wave[ii])*c_kms
			smoothed = threed_dutils.smooth_spectrum(model_lams[jj], model_norm, sigma_spec)

			#### interpolate and subtract
			flux_interp = interp1d(model_lams[jj],smoothed, bounds_error = False, fill_value = 0)
			resid = obsflux[p_idx] - flux_interp(obslam[p_idx])

			if jj == 0:
				prosp_sub = resid

			#### mask emission lines
			masklam = 50
			mask = np.ones_like(obslam[p_idx],dtype=bool)
			for lam in em_wave[ii]: mask[(obslam[p_idx] > lam-masklam) & (obslam[p_idx] < lam+masklam)] = 0

			#### fit Gaussian to residuals (clear structure visible sometimes...)
			hist, bin_edges = np.histogram(resid[mask], density=True)
			ngauss_init = jGaussian1D(mean=0.0,constant=0.0,slope=0.0,stddev=1e5,amplitude=1e-6)
			ngauss_init.slope.fixed = True
			ngauss_init.constant.fixed = True
			ngauss_init.mean.fixed = True
			ngauss_init.stddev.max = np.max(np.abs(bin_edges))*0.6
			noise_gauss = fitter(ngauss_init,(bin_edges[1:]+bin_edges[:-1])/2.,hist, maxiter=maxfev)

			##### TEST PLOT
			test_plot = False
			# (obs , model)
			if test_plot:
				fig, ax = plt.subplots(1,1)
				plt.plot(obslam[p_idx], obsflux[p_idx], color='black', drawstyle='steps-mid')
				plt.plot(obslam[p_idx], flux_interp(obslam[p_idx]), color='red')
				plt.show()
			# Gaussian fit to (obs - model)
			if test_plot:
				fig, ax = plt.subplots(1,1)
				plt.plot((bin_edges[1:]+bin_edges[:-1])/2.,hist, color='black')
				plt.plot((bin_edges[1:]+bin_edges[:-1])/2.,noise_gauss((bin_edges[1:]+bin_edges[:-1])/2.),color='red')
				plt.show()
				print 1/0




			#### measure noise!
			lam_noise = em_wave[ii].mean()
			flux_noise = (gauss_fit.constant_0+gauss_fit.slope_0*lam_noise)
			print 'measured noise: ' + "{:.4f}".format(np.abs(noise_gauss.stddev.value)/flux_noise)
			if np.abs(noise_gauss.stddev.value)/flux_noise > 100:
				print 1/0

			emline_noise[ii,jj] = np.clip(np.abs(noise_gauss.stddev.value),minnoise*flux_noise,maxnoise*flux_noise)

		#### now get errors in parameters by bootstrapping
		# use lower of two error estimates
		# eqw and flux come out of bootstrap as (nlines,[median,errup,errdown])
		tnoise = np.min(emline_noise[ii,:])
		median_param, emline_flux_local, emline_eqw_local = bootstrap(obslam[p_idx],prosp_sub,emmod,fitter,tnoise,em_wave[ii])
		emline_flux.append(emline_flux_local)
		emline_rest_eqw.append(emline_eqw_local)

		#### create median model
		emmod.parameters = median_param

		#### plot 
		axarr[ii].plot(obslam[p_idx],prosp_sub,color='black',drawstyle='steps-mid')
		axarr[ii].plot(obslam[p_idx],emmod(obslam[p_idx]),color='red')

		axarr[ii].set_ylabel(r'flux [L$_{\odot}/\AA$]')
		axarr[ii].set_xlabel(r'$\lambda$ [$\AA$]')
		for kk in xrange(len(emline[ii])):
			fmt = "{{0:{0}}}".format(".2e").format
			emline_str = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
			emline_str = emline_str.format(fmt(emline_flux_local[0,kk]), 
				                           fmt(emline_flux_local[1,kk]-emline_flux_local[0,kk]), 
				                           fmt(emline_flux_local[0,kk]-emline_flux_local[2,kk])) + ' erg/s, '
			axarr[ii].text(0.04, 0.935-0.085*kk, 
				           emline[ii][kk]+': '+emline_str+"{:.2f}".format(emline_eqw_local[0,kk])+r' $\AA$',
				           fontsize=16, transform = axarr[ii].transAxes)

		axarr[ii].xaxis.set_major_locator(MaxNLocator(5))
		axarr[ii].yaxis.get_major_formatter().set_powerlimits((0, 1))

	plt.tight_layout()
	plt.savefig(out_em, dpi = 300)
	plt.close()

	out = {}
	##### save emission line fluxes
	for kk in xrange(len(em_wave)): 

		out['lum'] = np.concatenate((out.get('lum',[]),emline_flux[kk][0,:]))
		out['lum_errup'] = np.concatenate((out.get('lum_errup',[]),emline_flux[kk][1,:]))
		out['lum_errdown'] = np.concatenate((out.get('lum_errdown',[]),emline_flux[kk][2,:]))

		out['flux'] = np.concatenate((out.get('flux',[]),emline_flux[kk][0,:]  / dfactor / (1+magphys['metadata']['redshift'])))
		out['flux_errup'] = np.concatenate((out.get('flux_errup',[]),emline_flux[kk][1,:]  / dfactor / (1+magphys['metadata']['redshift'])))
		out['flux_errdown'] = np.concatenate((out.get('flux_errdown',[]),emline_flux[kk][2,:]  / dfactor / (1+magphys['metadata']['redshift'])))

		out['rest_eqw'] = np.concatenate((out.get('rest_eqw',[]),emline_rest_eqw[kk][0,:]))
		out['rest_eqw_errup'] = np.concatenate((out.get('rest_eqw_errup',[]),emline_rest_eqw[kk][1,:]))
		out['rest_eqw_errdown'] = np.concatenate((out.get('rest_eqw_errdown',[]),emline_rest_eqw[kk][2,:]))

		out['em_name'] = np.concatenate((out.get('em_name',[]),emline[kk,:]))
		out['em_lam'] = np.concatenate((out.get('em_lam',[]),em_wave[kk,:]))

	##### save absorption line fluxes
	# 	absline_flux(eqw) = np.zeros(shape=(nset_abs,nmodel))
	for kk in xrange(nset_abs):

		out['abs_name'] = np.concatenate((out.get('abs_name',[]),absline[kk,:]))
		out['abs_lam'] = np.concatenate((out.get('abs_lam',[]),abs_wave[kk,:]))
		# fill in the rest here
		# note that there is one measurement per model
		out['abs_lum'] = np.concatenate((out.get('abs_lum',[]),absline_flux[kk,:]))
		out['abs_flux'] = np.concatenate((out.get('abs_flux',[]),absline_flux[kk,:]/ dfactor / (1+magphys['metadata']['redshift'])))
		out['abs_eqw'] = np.concatenate((out.get('abs_eqw',[]),absline_eqw[kk,:]))

	return out
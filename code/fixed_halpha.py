import numpy as np
import threed_dutils
import os
from bsfh import model_setup
import matplotlib as mpl
import random
import pickle
import matplotlib.pyplot as plt
import math
import triangle
import magphys_plot_pref
from astropy import constants
random.seed(69)

#### define parameter file
param_file = os.getenv('APPS') + '/threedhst_bsfh/parameter_files/brownseds_tightbc/brownseds_tightbc_params_1.py'

#### load test model, build sps, build important variables ####
model = model_setup.load_model(param_file)
obs   = model_setup.load_obs(param_file)
obs['filters'] = None # don't generate photometry, for speed
sps = threed_dutils.setup_sps()
parnames = np.array(model.theta_labels())

#### load relationship between CLOUDY/empirical Halpha and metallicity
outloc = '/Users/joel/code/python/threedhst_bsfh/data/pickles/ha_ratio.pickle'
with open(outloc, "rb") as f:
	coeff=pickle.load(f)
fit = np.poly1d(coeff)

class jLogFormatter(mpl.ticker.LogFormatter):
	'''
	this changes the format from exponential to floating point.
	'''

	def __call__(self, x, pos=None):
		"""Return the format for tick val *x* at position *pos*"""
		vmin, vmax = self.axis.get_view_interval()
		d = abs(vmax - vmin)
		b = self._base
		if x == 0.0:
			return '0'
		sign = np.sign(x)
		# only label the decades
		fx = math.log(abs(x)) / math.log(b)
		isDecade = mpl.ticker.is_close_to_int(fx)
		if not isDecade and self.labelOnlyBase:
			s = ''
		elif x > 10000:
			s = '{0:.3g}'.format(x)
		elif x < 1:
			s = '{0:.3g}'.format(x)
		else:
			s = self.pprint_val(x, d)
		if sign == -1:
			s = '-%s' % s
		return self.fix_minus(s)

#### format those log plots! 
minorFormatter = jLogFormatter(base=10, labelOnlyBase=False)
majorFormatter = jLogFormatter(base=10, labelOnlyBase=True)

def random_theta_draw(theta):
	'''
	draw a random set of thetas in the SFH, metallicity, and dust parameters
	'''

	to_draw = ['tage','logtau','delt_trunc','sf_tanslope','dust2','dust1','dust_index','logzsol']

	for par in to_draw:
		idx = np.where(parnames == par)[0][0]

		# set custom boundaries on dust parameters
		if par == 'dust2':
			min,max = (0.0,0.5)
		elif par == 'dust1':
			dust2 = theta[parnames == 'dust2']
			min,max = (0.5*dust2,2.0*dust2)
		elif par == 'dust_index':
			min,max = (-0.4,-1.0)
		else:
			min,max = model.theta_bounds()[idx] # everything else
		theta[idx] = random.random()*(max-min)+min

	return theta

def apply_metallicity_correction(ha_lum,logzsol):
	'''
	placeholder. must fit the CLOUDY / Kennicutt curve as function of logzsol, 
	apply correction here.
	'''
	ratio = 10**fit(logzsol)
	return ha_lum*ratio

def halpha_draw(theta0, ha_lum_fixed, delta, ndraw, thetas_start=None):
	'''
	INPUTS:
		theta0: the initial thetas
		ha_lum_fixed: the desired Halpha luminosity (after dust extinction), in Lsun
		delta: the tolerance on Halpha draws, in fractional numbers
		ndraw: the desired number of draws

	OUTPUT:
		Numpy vector of dimensions (nparams,ndraw)
		containing a number of SEDs that produce the desired Halpha luminosity within the desired tolerance
	'''

	out_pars = np.zeros(shape=(theta0.shape[0],0))
	out_spec = np.zeros(shape=(sps.wavelengths.shape[0],0))

	ntest=0
	while out_pars.shape[1] < ndraw:

		# draw random dust, SFH, metallicity properties
		theta = random_theta_draw(theta0)

		#### if we have them, use previous guesses
		if thetas_start is not None:
			if thetas_start['pars'].shape[1] != 0:
				theta = thetas_start['pars'][:,0]
				thetas_start['pars'] = thetas_start['pars'][:,1:]

		# calculate SFR(10 Myr)
		spec,mags,sm = model.mean_model(theta, obs, sps=sps)
		sfh_params = threed_dutils.find_sfh_params(model,theta,obs,sps,sm=sm)
		sfr_10     = threed_dutils.calculate_sfr(sfh_params, 0.01, minsfr=-np.inf, maxsfr=np.inf)

		# calculate synthetic Halpha, in Lsun
		ha_lum = threed_dutils.synthetic_halpha(sfr_10,theta[parnames=='dust1'],theta[parnames=='dust2'],
			                                    -1.0,theta[parnames=='dust_index'],kriek=True) / constants.L_sun.cgs.value

		# metallicity correction to CLOUDY scale
		ha_lum_adj = apply_metallicity_correction(ha_lum,theta[parnames=='logzsol'])[0]

		# check and add if necessary
		ratio = np.abs(ha_lum_adj/ha_lum_fixed - 1)
		if ratio < delta:
			out_pars = np.concatenate((out_pars,theta[:,None]),axis=1)
			out_spec = np.concatenate((out_spec,spec[:,None]),axis=1)
			print out_pars.shape[1]

		ntest+=1

	print 'total of {0} combinations tested'.format(ntest)

	return out_pars,out_spec


def main(redraw_thetas=True,pass_guesses=False):
	'''
	main driver, to create plot
		redraw_thetas: if true, will redraw parameters based on draw parameters defined in the IF statement
		pass_guesses: if true, during the redraw_thetas phase, will load previous save file and start with
		              those guesses
	'''

	#### name draws
	colors = ['red','green','blue']
	names = ['5e6','5e7','5e8'] # 3e7 is SFR = 1 Msun / yr, dust1 = 0.3, dust2 = 0.15
	smoothing = 10000 # km/s smooth
	norm_idx = np.abs(sps.wavelengths/1e4-1).argmin() # where to normalize the SEDs

	#### load or create halpha draws
	outpickle = '/Users/joel/code/python/threedhst_bsfh/data/pickles/ha_fixed.pickle'
	if redraw_thetas:
		ndraw = 200
		delta = 0.05
		thetas = {}

		for name in names:
			temp = {}

			# pass guesses
			if pass_guesses:
				with open(outpickle, "rb") as f:
					load_thetas = pickle.load(f)
				load_thetas = load_thetas[name]
			else:
				load_thetas = None

			temp['pars'],temp['spec'] = halpha_draw(model.initial_theta,float(name),delta,ndraw,thetas_start=load_thetas)
			thetas[name] = temp

		pickle.dump(thetas,open(outpickle, "wb"))
	else:
		with open(outpickle, "rb") as f:
			thetas=pickle.load(f)
		try:
			ndraw = thetas.values()[0].values()[0].shape[1] # this finds the length of the first item in the dictionary of dictionaries...
		except IndexError as e:
			ndraw = thetas.values()[0].values()[0].shape[0]

	#### Open figure
	fig = plt.figure(figsize=(18, 8))
	ax = [fig.add_axes([0.05, 0.1, 0.4, 0.85]),
		  fig.add_axes([0.48, 0.1, 0.4, 0.85])]
	cmap_ax = fig.add_axes([0.94,0.05,0.05,0.9])
	xlim = (10**-1.25,10**2.96)
	ylim = (6.5,9.3)


	#### plot spectra and relationship
	c = 3e18   # angstroms per second
	to_fnu = c/sps.wavelengths

	for ii, name in enumerate(names):

		'''
		#### try plotting all spectra first (maybe percentile of spectra ?)
		for kk in xrange(ndraw):
			ax.plot(sps.wavelengths/1e4,np.log10(thetas[name]['spec'][:,kk]*to_fnu),color=colors[ii])
		'''

		#### percentile of spectra
		spec_perc = np.zeros(shape=(sps.wavelengths.shape[0],3))
		for kk in xrange(spec_perc.shape[0]):
			spec_perc[kk,:] = triangle.quantile(thetas[name]['spec'][kk,:], [0.5, 0.84, 0.16])*to_fnu[kk]

		#### smooth and log spectra
		for nn in xrange(3): spec_perc[:,nn] = np.log10(threed_dutils.smooth_spectrum(sps.wavelengths,spec_perc[:,nn],smoothing))

		#### normalize at 10,000 angstroms
		spec_perc -= spec_perc[norm_idx,0]-8.5

		#### plot spectra
		ax[0].fill_between(sps.wavelengths/1e4, spec_perc[:,1], spec_perc[:,2], 
			                  color=colors[ii],alpha=0.15)
		ax[0].plot(sps.wavelengths/1e4,spec_perc[:,0],color=colors[ii])

		#### second panel plots
		# if SFR + halpha extinction data doesn't exist, make it then save it
		if 'sfr_10' not in thetas[name]:
			sfr_10, ha_ext = np.zeros(ndraw),np.zeros(ndraw)
			for kk in xrange(ndraw):
				theta = thetas[name]['pars'][:,kk]
				sfh_params = threed_dutils.find_sfh_params(model,theta,obs,sps)
				sfr_10[kk] = threed_dutils.calculate_sfr(sfh_params, 0.01, minsfr=-np.inf, maxsfr=np.inf)

				ha_ext[kk] = threed_dutils.charlot_and_fall_extinction(6563.0,
					                                                   theta[parnames=='dust1'],
					                                                   theta[parnames=='dust2'],
					                                                   -1.0,
					                                                   theta[parnames=='dust_index'],
					                                                   kriek=True)
				ha_ext[kk] = 1./ha_ext[kk]

			thetas[name]['sfr_10'] = sfr_10
			thetas[name]['ha_ext'] = ha_ext
			pickle.dump(thetas,open(outpickle, "wb"))


		'''
		ax[1].plot(thetas[name]['ha_ext'],thetas[name]['sfr_10'],'o',color=colors[ii],alpha=0.5,linestyle=' ')
		'''

	#### plot sample of SEDs at fixed halpha luminosity
	#### sample evenly in halpha extinction
	vmin = np.min(thetas[name]['ha_ext'])
	vmax = np.max(thetas[name]['ha_ext'])

	nplot = 11
	nplot = 12
	name = names[1]
	nsample = ndraw / nplot
	#sorted_by_ha_ext = np.argsort(thetas[name]['ha_ext'])
	ha_ext_delta = (vmax-vmin)/nplot
	ha_ext_start = vmin + ha_ext_delta/2.

	#### create colormap
	cmap = mpl.cm.plasma
	cmap = mpl.cm.rainbow
	norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
	scalarMap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

	#### prep and plot each spectrum
	for nn in xrange(nplot):
		# idx = sorted_by_ha_ext[nn*nsample+np.floor(nsample/2.)]
		idx = np.abs(thetas[name]['ha_ext']-(ha_ext_start+ha_ext_delta*nn)).argmin()
		spec_to_plot = thetas[name]['spec'][:,idx]*to_fnu

		#### smooth and log spectra
		spec_to_plot = np.log10(threed_dutils.smooth_spectrum(sps.wavelengths,spec_to_plot,smoothing))

		#### normalize at 10,000 angstroms
		spec_to_plot -= spec_to_plot[norm_idx]-8.5

		#### get color
		color = scalarMap.to_rgba(thetas[name]['ha_ext'][idx])

		#### plot sed
		ax[1].plot(sps.wavelengths/1e4,spec_to_plot,color=color,
			       lw=1.5,alpha=0.9)

	#### colorbar
	cb1 = mpl.colorbar.ColorbarBase(cmap_ax, cmap=cmap,
                                	norm=norm,
                                	orientation='vertical')
	cb1.set_label(r'F$_{\mathrm{intrinsic}}$/F$_{\mathrm{extincted}}$ (6563 $\AA$)')
	cb1.ax.yaxis.set_ticks_position('left')
	cb1.ax.yaxis.set_label_position('left')

	#### First plot labels and scales
	ax[0].set_ylabel(r'log($\nu$ f$_{\nu}$) [arbitrary]')
	ax[0].set_xlabel(r'wavelength [microns]')
	ax[0].set_xlim(xlim)
	ax[0].set_ylim(ylim)
	ax[0].set_xscale('log',nonposx='clip',subsx=(1,2,5))
	ax[0].xaxis.set_minor_formatter(minorFormatter)
	ax[0].xaxis.set_major_formatter(majorFormatter)

	# text
	yt = 0.945
	xt = 0.98
	deltay = 0.015
	ax[0].text(xt,yt,r'H$_{\alpha}$ luminosity',weight='bold',transform = ax[0].transAxes,ha='right')
	for ii, name in enumerate(names):
		ax[0].text(xt,yt-(ii+1)*0.05,name+r' L$_{\odot}$',
			       color=colors[ii],transform = ax[0].transAxes,
			       ha='right')

	#### second plot labels and scales
	ax[1].set_xlabel(r'wavelength [microns]')
	ax[1].set_xlim(xlim)
	ax[1].set_ylim(ylim)
	ax[1].set_xscale('log',nonposx='clip',subsx=(1,2,5))
	ax[1].xaxis.set_minor_formatter(minorFormatter)
	ax[1].xaxis.set_major_formatter(majorFormatter)
	ax[1].text(xt,yt,r'SED variation at fixed H$_{\alpha}$ luminosity',
		       weight='bold',transform = ax[1].transAxes,ha='right')

	'''
	ax[1].set_xlabel(r'F$_{intrinsic}$/F$_{extincted}$ (6563 $\AA$)')
	ax[1].set_ylabel(r'SFR (10 Myr, M$_{\odot}$/yr)')
	ax[1].set_xlim(10**0.0,10**0.52)
	ax[1].set_ylim(10**-1.5,10**1.6)
	ax[1].set_xscale('log',nonposx='clip',subsx=(1,1.5,2,2.5,3.0,3.5))
	ax[1].xaxis.set_minor_formatter(minorFormatter)
	ax[1].xaxis.set_major_formatter(majorFormatter)
	ax[1].set_yscale('log',nonposy='clip',subsy=(1,2,5))
	ax[1].yaxis.set_minor_formatter(minorFormatter)
	ax[1].yaxis.set_major_formatter(majorFormatter)
	'''

	plt.savefig('/Users/joel/my_papers/prospector_brown/figures/fixed_ha.png',dpi=150)
	plt.close()

	print 1/0

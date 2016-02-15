import numpy as np
import threed_dutils
import os
from bsfh import model_setup
import matplotlib as mpl
import random
import pickle
import matplotlib.pyplot as plt
import math
import corner
import magphys_plot_pref
from astropy import constants
random.seed(69)

#### define parameter file
param_file = os.getenv('APPS') + '/threedhst_bsfh/parameter_files/brownseds_tightbc/brownseds_tightbc_params_1.py'

#### load test model, build sps, build important variables ####
model = model_setup.load_model(param_file)
model.params['zred'] = np.atleast_1d(0.0)
obs   = model_setup.load_obs(param_file)
obs['filters'] = None # don't generate photometry, for speed
sps = threed_dutils.setup_sps()
parnames = np.array(model.theta_labels())

#### load relationship between CLOUDY/empirical Halpha and metallicity
outloc = '/Users/joel/code/python/threedhst_bsfh/data/pickles/ha_ratio.pickle'
with open(outloc, "rb") as f:
	coeff=pickle.load(f)
fit = np.poly1d(coeff)

#### global plotting parameters
fontsize = 20

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

def halpha_draw(theta0, delta, ndraw, thetas_start=None, fixed_lbol=None, ha_lum_fixed=None):
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
	out_luv = np.zeros(0)
	out_lir = np.zeros(0)
	out_ha = np.zeros(0)

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

		# calculate LIR, LUV
		luv = threed_dutils.return_luv(sps.wavelengths,spec) / 3.846e33
		lir = threed_dutils.return_lir(sps.wavelengths,spec) / 3.846e33

		# check Halpha and, if necessary, LIR to see if it's within tolerance
		# if so, add parameters to the pile
		if ha_lum_fixed is not None:
			ratio = np.abs(ha_lum_adj/ha_lum_fixed - 1)
			if ratio < delta:
				
				out_pars = np.concatenate((out_pars,theta[:,None]),axis=1)
				out_spec = np.concatenate((out_spec,spec[:,None]),axis=1)
				out_lir = np.concatenate((out_lir,np.atleast_1d(lir)))
				out_luv = np.concatenate((out_luv,np.atleast_1d(luv)))
				out_ha = np.concatenate((out_ha,np.atleast_1d(ha_lum_adj)))
				print out_pars.shape[1]
		elif fixed_lbol is not None:
			lbol = luv + lir
			lbol_ratio = np.abs(lbol/fixed_lbol - 1)
			if (lbol_ratio < delta) & (sfr_10 > 0.0):
				out_pars = np.concatenate((out_pars,theta[:,None]),axis=1)
				out_spec = np.concatenate((out_spec,spec[:,None]),axis=1)
				out_lir = np.concatenate((out_lir,np.atleast_1d(lir)))
				out_luv = np.concatenate((out_luv,np.atleast_1d(luv)))
				out_ha = np.concatenate((out_ha,np.atleast_1d(ha_lum_adj)))
				print out_pars.shape[1]

		ntest+=1

	print 'total of {0} combinations tested'.format(ntest)

	return out_pars,out_spec,out_lir,out_luv,out_ha

def label_sed(sedax,ylim):

	#### L_IR, L_UV, halpha labels
	lw = 5
	alpha = 1.0
	color = '0.5'
	zorder = 2
	length = 0.05
	for ax in sedax:
		# L_IR
		yloc = 7.6
		ax.plot([8,1000],[yloc,yloc], lw=lw, alpha=alpha, color=color, zorder=zorder)
		ax.plot([8,8],[yloc-length,yloc+length], lw=lw, alpha=alpha, color=color, zorder=zorder)
		ax.plot([1e3,1e3],[yloc-length,yloc+length], lw=lw, alpha=alpha, color=color, zorder=zorder)
		ax.text(90,yloc+0.05,r'L$_{\mathrm{IR}}$',weight='bold',fontsize=fontsize,ha='center', zorder=zorder)
		
		# L_UV
		ax.plot([.1216,.3000],[yloc,yloc], lw=lw, alpha=alpha, color=color, zorder=zorder)
		ax.plot([.1216,.1216],[yloc-length,yloc+length], lw=lw, alpha=alpha, color=color, zorder=zorder)
		ax.plot([.3000,.3000],[yloc-length,yloc+length], lw=lw, alpha=alpha, color=color, zorder=zorder)
		ax.text(.19,yloc+0.05,r'L$_{\mathrm{UV}}$',weight='bold',fontsize=fontsize,ha='center', zorder=zorder)

		# Halpha
		ax.plot([.6563,.6563],[ylim[0],ylim[1]],linestyle='--',color='0.2', zorder=zorder)
		ax.text(.6000,10.65,r'H$_{\mathrm{\alpha}}$',weight='bold', ha='right',fontsize=fontsize, zorder=zorder)

	return sedax

def main(redraw_thetas=True,pass_guesses=False,redraw_lbol_thetas=False):
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
	lw = 2

	#### load or create fixed halpha draws
	outpickle_fixedha = '/Users/joel/code/python/threedhst_bsfh/data/pickles/ha_fixed.pickle'
	if redraw_thetas:
		ndraw = 200
		delta = 0.05
		thetas = {}

		for name in names:
			temp = {}

			# pass guesses
			if pass_guesses:
				with open(outpickle_fixedha, "rb") as f:
					load_thetas = pickle.load(f)
				load_thetas = load_thetas[name]
			else:
				load_thetas = None

			temp['pars'],temp['spec'],temp['lir'],temp['luv'],temp['ha'] = \
			halpha_draw(model.initial_theta,delta,ndraw,
				        thetas_start=load_thetas,
				        ha_lum_fixed=float(name))
			thetas[name] = temp

		pickle.dump(thetas,open(outpickle_fixedha, "wb"))
	else:
		with open(outpickle_fixedha, "rb") as f:
			thetas=pickle.load(f)
		try:
			ndraw = thetas.values()[0].values()[0].shape[1] # this finds the length of the first item in the dictionary of dictionaries...
		except IndexError as e:
			ndraw = thetas.values()[0].values()[0].shape[0]

	#### load or create lbol draws
	outpickle_fixedlbol = '/Users/joel/code/python/threedhst_bsfh/data/pickles/lbol_fixed.pickle'
	if redraw_lbol_thetas:
		ndraw = 50
		delta = 0.008
		thetas_lbol = {}
		name = names[1]

		if pass_guesses:
			with open(outpickle_fixedlbol, "rb") as f:
				load_thetas = pickle.load(f)
		else:
			load_thetas = None

		thetas_lbol['pars'],thetas_lbol['spec'],thetas_lbol['lir'],thetas_lbol['luv'],thetas_lbol['ha'] = \
		halpha_draw(model.initial_theta,delta,ndraw,
			        thetas_start=load_thetas,
			        fixed_lbol=1.61e10) # from the median of halpha_lum=5e7
			
		pickle.dump(thetas_lbol,open(outpickle_fixedlbol, "wb"))
	else:
		with open(outpickle_fixedlbol, "rb") as f:
			thetas_lbol=pickle.load(f)
		ndraw = thetas_lbol.values()[0].shape[0]

	#### Open figure
	sedfig = plt.figure(figsize=(18, 8))
	sedax = [sedfig.add_axes([0.05, 0.1, 0.4, 0.85]),
		     sedfig.add_axes([0.494, 0.1, 0.4, 0.85])]
	cmap_ax = sedfig.add_axes([0.948,0.05,0.05,0.9])

	relfig, relax = plt.subplots(ncols=1, nrows=1, figsize=(8,8))

	xlim_sed = (10**-1.25,10**3.05)
	ylim_sed = (7.5,11.3)
	xlim_rel = (-0.55,0.05)

	#### L_IR, L_UV, Halpha labels
	sedax = label_sed(sedax,ylim_sed)

	#### plot spectra and relationship
	c = 3e18   # angstroms per second
	to_fnu = c/sps.wavelengths

	for ii, name in enumerate(names):

		#### percentile of spectra
		spec_perc = np.zeros(shape=(sps.wavelengths.shape[0],3))
		for kk in xrange(spec_perc.shape[0]):
			spec_perc[kk,:] = corner.quantile(thetas[name]['spec'][kk,:], [0.5, 0.84, 0.16])*to_fnu[kk]

		#### smooth and log spectra
		for nn in xrange(3): spec_perc[:,nn] = np.log10(threed_dutils.smooth_spectrum(sps.wavelengths,spec_perc[:,nn],smoothing))

		#### normalize at 10,000 angstroms
		# spec_perc -= spec_perc[norm_idx,0]-8.5

		#### plot spectra
		sedax[0].plot(sps.wavelengths/1e4,spec_perc[:,0],color=colors[ii],lw=lw,zorder=1)

		#### information for additional plots
		# if SFR + halpha extinction data doesn't exist, make it then save it
		if 'sfr_10' not in thetas[name]:
			sfr_10, ha_ext, luv, lir = [np.zeros(ndraw) for i in xrange(4)]
			for kk in xrange(ndraw):
				theta = thetas[name]['pars'][:,kk]
				spec,mags,sm = model.mean_model(theta, obs, sps=sps)
				sfh_params = threed_dutils.find_sfh_params(model,theta,obs,sps,sm=sm)
				sfr_10[kk] = threed_dutils.calculate_sfr(sfh_params, 0.01, minsfr=-np.inf, maxsfr=np.inf)

				ha_ext[kk] = threed_dutils.charlot_and_fall_extinction(6563.0,
					                                                   theta[parnames=='dust1'],
					                                                   theta[parnames=='dust2'],
					                                                   -1.0,
					                                                   theta[parnames=='dust_index'],
					                                                   kriek=True)
				ha_ext[kk] = 1./ha_ext[kk]

				luv[kk] = threed_dutils.return_luv(sps.wavelengths,spec) / 3.846e33
				lir[kk] = threed_dutils.return_lir(sps.wavelengths,spec) / 3.846e33

			thetas[name]['sfr_10'] = sfr_10
			thetas[name]['ha_ext'] = ha_ext
			thetas[name]['luv'] = luv
			thetas[name]['lir'] = lir
			pickle.dump(thetas,open(outpickle_fixedha, "wb"))

		#### relationship plot
		xplot = thetas[name]['lir']/thetas[name]['luv']
		xplot = np.log10(1./thetas[name]['ha_ext'])
		yplot = np.log10(thetas[name]['lir']+thetas[name]['luv'])
		
		'''
		if name == '5e6':
			slope = (8.5-8.4) / ((-0.2)-(-0.1))
			b = 8.5 - slope*(-0.2)
			bad = yplot > (slope*xplot+b)
			xplot = xplot[~bad]
			yplot = yplot[~bad]
		if name == '5e7':
			bad = (xplot < 2) & (yplot > 9.25)
			xplot = xplot[~bad]
			yplot = yplot[~bad]
		'''
		relax.plot(xplot, yplot, 'o', color=colors[ii])

	#### plot sample of SEDs at fixeed
	#### sample evenly in halpha extinction
	#name = names[1]
	#cquant = thetas[name]['ha_ext']
	#cquant = np.log10(thetas[name]['lir']/thetas[name]['luv'])
	cquant = np.log10(thetas_lbol['ha'])
	vmin = np.min(cquant)
	vmax = np.max(cquant)

	nplot = 6
	nsample = ndraw / nplot
	delta = (vmax-vmin)/nplot
	start = vmin+delta/2.

	#### create colormap
	cmap = mpl.colors.LinearSegmentedColormap( 'rainbow_r', mpl.cm.revcmap(mpl.cm.rainbow._segmentdata))
	norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
	scalarMap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

	#### prep and plot each spectrum
	for nn in xrange(nplot):
		idx = np.abs(cquant-(start+delta*nn)).argmin()
		spec_to_plot = thetas_lbol['spec'][:,idx]*to_fnu

		#### smooth and log spectra
		spec_to_plot = np.log10(threed_dutils.smooth_spectrum(sps.wavelengths,spec_to_plot,smoothing))

		#### normalize at 10,000 angstroms
		#spec_to_plot -= spec_to_plot[norm_idx]-8.5

		#### get color
		color = scalarMap.to_rgba(cquant[idx])

		#### plot sed
		sedax[1].plot(sps.wavelengths/1e4,spec_to_plot,color=color,
			          lw=lw,alpha=0.9,zorder=1)

	#### colorbar
	cb1 = mpl.colorbar.ColorbarBase(cmap_ax, cmap=cmap,
                                	norm=norm,
                                	orientation='vertical')
	cb1.set_label(r'F$_{\mathrm{intrinsic}}$/F$_{\mathrm{extincted}}$ (6563 $\AA$)')
	cb1.set_label(r'log(L$_{\mathrm{IR}}$ / L$_{\mathrm{UV}}$)')
	cb1.set_label(r'log(H$_{\alpha}$ luminosity)')
	cb1.ax.yaxis.set_ticks_position('left')
	cb1.ax.yaxis.set_label_position('left')

	#### First plot labels and scales
	sedax[0].set_ylabel(r'log($\nu$ f$_{\nu}$)')
	sedax[0].set_xlabel(r'wavelength [microns]')
	sedax[0].set_xlim(xlim_sed)
	sedax[0].set_ylim(ylim_sed)
	sedax[0].set_xscale('log',nonposx='clip',subsx=(1,2,4))
	sedax[0].xaxis.set_minor_formatter(minorFormatter)
	sedax[0].xaxis.set_major_formatter(majorFormatter)

	# text
	yt = 0.945
	xt = 0.98
	deltay = 0.015
	sedax[0].text(xt,yt,r'H$_{\mathbf{\alpha}}$ luminosity',weight='bold',transform = sedax[0].transAxes,ha='right',fontsize=fontsize)
	for ii, name in enumerate(names):
		sedax[0].text(xt,yt-(ii+1)*0.05,name+r' L$_{\odot}$',
			       color=colors[ii],transform = sedax[0].transAxes,
			       ha='right',fontsize=fontsize)

	#### second plot labels and scales
	#sedax[1].set_ylabel(r'log($\nu$ f$_{\nu}$)')
	sedax[1].set_xlabel(r'wavelength [microns]')
	sedax[1].set_xlim(xlim_sed)
	sedax[1].set_ylim(ylim_sed)
	sedax[1].set_xscale('log',nonposx='clip',subsx=(1,2,4))
	sedax[1].xaxis.set_minor_formatter(minorFormatter)
	sedax[1].xaxis.set_major_formatter(majorFormatter)
	sedax[1].text(xt,yt,r'SED variation at fixed L$_{\mathbf{IR}}$ + L$_{\mathbf{UV}}$',
		       weight='bold',transform = sedax[1].transAxes,ha='right',fontsize=fontsize)

	#### relationship labels
	relax.set_ylabel(r'log(L$_{\mathrm{IR}}$ + L$_{\mathrm{UV}}$)')
	relax.set_xlabel(r'L$_{\mathrm{IR}}$ / L$_{\mathrm{UV}}$')
	relax.set_xlabel(r'log(F$_{\mathrm{extincted}}$/F$_{\mathrm{intrinsic}}$) (6563 $\AA$)')
	relax.set_xlim(xlim_rel)

	sedfig.savefig('/Users/joel/my_papers/prospector_brown/figures/fixed_ha.png',dpi=150)
	relfig.savefig('/Users/joel/my_papers/prospector_brown/figures/fixed_ha_relationship.png',dpi=150)
	plt.close()

	print 1/0

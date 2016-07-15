import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
import numpy as np
import threed_dutils
import copy, math
from scipy.special import erf
from astropy import constants

#### constants
dpi = 75
c = 3e18   # angstroms per second

#### plot preferences
plt.ioff() # don't pop up a window for each plot
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
mpl.rcParams['font.sans-serif']='Geneva'

mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble']=[r"\usepackage{bm}"]
mpl.rcParams.update({'font.size': 30})

#### parameter plot style
colors = ['blue', 'black', 'red']
alpha = 0.8
lw = 3

#### define model
import brownseds_tightbc_params as pfile
sps = pfile.load_sps(**pfile.run_params)
model = pfile.load_model(**pfile.run_params)
obs = pfile.load_obs(**pfile.run_params)

#### set initial theta
# 'mass', 'tage', 'logtau', 
# 'dust2', 'logzsol', 'dust_index', 
# 'delt_trunc', 'sf_tanslope', 'dust1', 
# 'duste_qpah', 'duste_gamma', 'duste_umin'
labels = model.theta_labels()
model.initial_theta = np.array([10, 1.1, -0.5,
                                0.0, -1.0, 0.0,
                                0.75, -1.2, 0.0,
                                3.0, 1e-1, 10.0])

#### set up starting model
model.params['add_dust_emission'] = np.atleast_1d(False)
model.params['add_neb_emission'] = np.atleast_1d(False)
model.params['zred'] = np.atleast_1d(0.0)
model.params['peraa'] = np.atleast_1d(True)

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

def make_ticklabels_invisible(ax,showx=False):
	if showx:
	    for tl in ax.get_yticklabels():tl.set_visible(False)
	else:
	    for tl in ax.get_xticklabels() + ax.get_yticklabels():
	        tl.set_visible(False)

def sfh_xplot(ax,par,par_idx,first=False):

	#### sandwich 'regular' spectra in between examples
	if first:
		par = np.atleast_1d(model.initial_theta[par_idx])
		pcolor = ['black']
	else:
		par = [par[0],model.initial_theta[par_idx],par[1]]
		pcolor = copy.copy(colors)

	#### time (DEFINE DYNAMICALLY FOR TAGE)
	if par_idx == 1:
		t = np.linspace(0,np.max(par),100)
	else:
		t = np.linspace(0,1.1,100)

	#### calculate and plot different SFHs
	theta = copy.copy(model.initial_theta)
	for ii,p in enumerate(par):

		theta[par_idx] = p
		sfh_pars = threed_dutils.find_sfh_params(model,theta,obs,sps)
		sfh = threed_dutils.return_full_sfh(t, sfh_pars)

		ax.plot(t[::-1]*10,np.log10(sfh[::-1]),color=pcolor[ii],lw=lw,alpha=alpha)

	#### limits and labels
	#ax.set_xlim(3.3,6.6)
	ax.set_ylim(-0.5,1.7)
	ax.set_xlabel(r'lookback time [Gyr]')
	ax.set_ylabel(r'log(SFR [M$_{\odot}$/yr])')

	ax.set_xlim(ax.get_xlim()[1],ax.get_xlim()[0])

def plot_sed(ax,par_idx,par=None,txtlabel=None,fmt="{:.2e}",longwave=False,first=False):

	#### constants
	pc = 3.085677581467192e18  # in cm
	dfactor_10pc = 4*np.pi*(10*pc)**2

	#### sandwich 'regular' spectra in between examples
	if first:
		par = np.atleast_1d(model.initial_theta[par_idx])
		pcolor = [first]
	else:
		par = [par[0],model.initial_theta[par_idx],par[1]]
		pcolor = copy.copy(colors)


	#### loop, plot first plot
	theta = copy.copy(model.initial_theta)
	spec_sav = []
	for ii,p in enumerate(par):
		theta[par_idx] = p

		# spectrum comes out in erg/s/cm^2/AA at 10 pc
		spec,mags,_ = model.mean_model(theta, obs, sps=sps)
		spec *= sps.csp.wavelengths
		spec *= dfactor_10pc / constants.L_sun.cgs.value

		spec = threed_dutils.smooth_spectrum(sps.csp.wavelengths,spec,200,minlam=3e3,maxlam=1e4)

		ax.plot(sps.csp.wavelengths/1e4,np.log10(spec),color=pcolor[ii],lw=lw,alpha=alpha)
		spec_sav.append(spec)

	#### limits and labels [main plot]
	if longwave:
		xlim = (10**-1.05,10**2.6)
	else:
		xlim = (10**-1.05,10)
	ylim = (-6.0+15,-3.9+15)
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	ax.set_xlabel(r'wavelength [microns]')
	ax.set_ylabel(r'log(flux density [$\nu$f$_{\nu}$])')
	ax.set_xscale('log',nonposx='clip',subsx=(1,2,5))
	ax.xaxis.set_minor_formatter(minorFormatter)
	ax.xaxis.set_major_formatter(majorFormatter)
	make_ticklabels_invisible(ax,showx=True)

def plot_sed_dustshape(ax,longwave=False):

	#### constants
	pc = 3.085677581467192e18  # in cm
	dfactor_10pc = 4*np.pi*(10*pc)**2

	duste_qpah = [0.1,3,8]
	duste_umin = [20,8,2]
	duste_gamma = [0.5,0.1,0.0]

	qpah_idx = labels.index('duste_qpah')
	umin_idx = labels.index('duste_umin')
	gamma_idx = labels.index('duste_gamma')

	#### loop, plot first plot
	theta = copy.copy(model.initial_theta)
	spec_sav = []
	for ii,p in enumerate(duste_qpah):

		theta[qpah_idx] = duste_qpah[ii]
		theta[umin_idx] = duste_umin[ii]
		theta[gamma_idx] = duste_gamma[ii]

		# spectrum comes out in erg/s/cm^2/AA at 10 pc
		spec,mags,_ = model.mean_model(theta, obs, sps=sps)
		spec *= sps.csp.wavelengths
		spec *= dfactor_10pc / constants.L_sun.cgs.value

		spec = threed_dutils.smooth_spectrum(sps.csp.wavelengths,spec,200,minlam=3e3,maxlam=1e4)

		ax.plot(sps.csp.wavelengths/1e4,np.log10(spec),color=colors[ii],lw=lw,alpha=alpha)
		spec_sav.append(spec)

	#### limits and labels [main plot]
	if longwave:
		xlim = (10**-1.05,10**2.6)
	else:
		xlim = (10**-1.05,10)
	ylim = (-6.0+15,-4.5+15)
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	ax.set_xlabel(r'wavelength [microns]')
	ax.set_ylabel(r'log(flux density [$\nu$f$_{\nu}$])')
	ax.set_xscale('log',nonposx='clip',subsx=(1,2,5))
	ax.xaxis.set_minor_formatter(minorFormatter)
	ax.xaxis.set_major_formatter(majorFormatter)
	make_ticklabels_invisible(ax,showx=True)

def main_plot():

	figsize = (14,7)

	#### PLOT 1 DEFAULT
	fig, ax = plt.subplots(1, 1, figsize = (8,8))
	idx = labels.index('sf_tanslope')
	par = [-1.4,1.4]
	sfh_xplot(ax,par,idx,first=True)
	plt.tight_layout()
	fig.savefig('sfh_variation_first.png',dpi=150)
	plt.close()

	fig, ax = plt.subplots(1, 1, figsize = figsize)
	plot_sed(ax, idx, par=par,longwave=False,first='black')
	plt.tight_layout()
	fig.savefig('sfh_sed_first.png',dpi=150)
	plt.close()

	#### PLOT 2 AGE
	fig, ax = plt.subplots(1, 1, figsize = (8,8))
	idx = labels.index('sf_tanslope')
	par = [0.4,-1.4]
	sfh_xplot(ax,par,idx)
	plt.tight_layout()
	fig.savefig('sfh_variation.png',dpi=150)
	plt.close()

	fig, ax = plt.subplots(1, 1, figsize = figsize)
	plot_sed(ax, idx, par=par,longwave=False)
	plt.tight_layout()
	fig.savefig('sfh_sed.png',dpi=150)
	plt.close()

	#### PLOT 3 METALLICITY
	idx = labels.index('logzsol')
	par = [-1.8,0.1]
	fig, ax = plt.subplots(1, 1, figsize = figsize)
	plot_sed(ax,idx,par=par)
	plt.tight_layout()
	fig.savefig('met_sed.png',dpi=150)
	plt.close()

	#### PLOT 4 DUST ATTENUATION
	idx = labels.index('dust2')
	par = [0.3,0.6]
	fig, ax = plt.subplots(1, 1, figsize = figsize)
	plot_sed(ax,idx,par=par)
	plt.tight_layout()
	fig.savefig('dust_sed.png',dpi=150)
	plt.close()

	#### PLOT 5 DUST EMISSION
	model.params['add_dust_emission'] = np.array(True)
	idx = labels.index('dust2')
	par = [0.3,0.6]
	fig, ax = plt.subplots(1, 1, figsize = figsize)
	plot_sed(ax,idx,par=par,longwave=True)
	plt.tight_layout()
	fig.savefig('dust_emission_sed.png',dpi=150)
	plt.close()
	model.initial_theta[idx] = 0.3

	#### PLOT 6 VARY DUST COMPOSITION & GEOMETRY & INCIDENT STARLIGHT
	fig, ax = plt.subplots(1, 1, figsize = figsize)
	plot_sed_dustshape(ax,longwave=True)
	plt.tight_layout()
	fig.savefig('dust_shape_sed.png',dpi=150)
	plt.close()

	#### PLOT 7 NEBULAR EMISSION
	idx = labels.index('sf_tanslope')
	par = [-1.4,1.4]
	fig, ax = plt.subplots(1, 1, figsize = figsize)
	model.params['add_neb_emission'] = np.atleast_1d(True)
	plot_sed(ax, idx, par=par,longwave=False,first='blue')
	model.params['add_neb_emission'] = np.atleast_1d(False)
	plot_sed(ax, idx, par=par,longwave=False,first='black')
	plt.tight_layout()
	fig.savefig('neb_sed.png',dpi=150)
	plt.close()

	import os
	os.system('open *.png')
	print 1/0



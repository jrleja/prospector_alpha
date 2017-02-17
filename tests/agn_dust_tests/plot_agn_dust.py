import numpy as np
import threed_dutils
import matplotlib.pyplot as plt
import magphys_plot_pref
import time

import matplotlib.cm as cmx
import matplotlib.colors as colors
import copy
import matplotlib as mpl
import math
import brownseds_agn_params as nparams

#### LOAD ALL OF THE THINGS
sps = nparams.load_sps(**nparams.run_params)
model = nparams.load_model(**nparams.run_params)
model.params['zred'] = 0.0
obs = nparams.load_obs(**nparams.run_params)
sps.update(**model.params)

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

def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='plasma') 
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color

def plot_agn_fraction(no_dust = False):

	#### open figure
	itheta = copy.deepcopy(model.initial_theta)
	fig, ax = plt.subplots(1,2,figsize=(10,5))

	#### find indexes
	pnames = np.array(model.theta_labels())
	didx = pnames == 'fagn'
	tidx = pnames == 'agn_tau'

	##### turn off dust
	if no_dust:
		d1_idx = pnames == 'dust1'
		d2_idx = pnames == 'dust2'
		itheta[d1_idx] = 0.0
		itheta[d2_idx] = 0.0
		outname = 'AGN_parameters_nodust.png'
	else:
		outname = 'AGN_parameters.png'

	#### set up parameters
	nsamp = 5
	to_samp = [r'f$_{\mathrm{AGN}}$',r'$\tau_{\mathrm{AGN}}$']
	samp_pars = [np.linspace(0.0,2.0,nsamp),np.linspace(5,55,nsamp)] 
	idx = [didx,tidx]
	model.initial_theta[didx] = 0.3
	model.initial_theta[tidx] = 20

	### define wavelength regime + conversions
	to_plot = np.array([0.3,100])
	plt_idx = (sps.wavelengths > to_plot[0]*1e4) & (sps.wavelengths < to_plot[1]*1e4)
	c = 3e18   # angstroms per second
	conversion = c/sps.wavelengths[plt_idx]

	#### generate spectra
	for k,par in enumerate(to_samp):
		nsamp = len(samp_pars[k])
		cmap = get_cmap(nsamp)
		for i in xrange(nsamp):
			#sps.ssp.params.dirtiness = 1
			itheta[idx[k]] = samp_pars[k][i]
			t1 = time.time()
			spec,mags,sm = model.mean_model(itheta, obs, sps=sps)
			t2 = time.time()
			ax[k].plot(sps.wavelengths[plt_idx]/1e4,np.log(spec[plt_idx]*conversion),color=cmap(i),alpha=0.7,lw=2.5,label="{:.1f}".format(samp_pars[k][i]))
		itheta[idx[k]] = model.initial_theta[idx[k]]

	#### legend + labels
	ax[0].legend(loc=4,prop={'size':12},title='f$_{\mathrm{AGN}}$')
	ax[1].legend(loc=4,prop={'size':12},title=to_samp[1])
	ax[0].text(0.05,0.075,r'$\tau_{\mathrm{AGN}}$=20',transform=ax[0].transAxes,fontsize=12)
	ax[1].text(0.05,0.075,r'f$_{\mathrm{AGN}}$=0.3',transform=ax[1].transAxes,fontsize=12)

	for a in ax:
		a.get_legend().get_title().set_fontsize('16')
		a.set_xscale('log',nonposx='clip',subsx=(1,3))
		a.xaxis.set_minor_formatter(minorFormatter)
		a.xaxis.set_major_formatter(majorFormatter)
		a.set_xlim(to_plot)
		a.set_ylim(47,54)

		a.set_xlabel(r'wavelength [$\mu$m]')
		a.set_ylabel(r'log($\nu$f$_{\nu}$)')

	plt.tight_layout()
	fig.savefig(outname,dpi=150)
	plt.close()
	import os
	os.system('open *.png')

if __name__ == "__main__":
	plot_agn_fraction()
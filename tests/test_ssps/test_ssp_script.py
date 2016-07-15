import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import numpy as np
import np_mocks_smooth_params as nparams
import magphys_plot_pref
from prospect.models import model_setup
import os
import matplotlib as mpl
import math
import sys

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


def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv') 
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color


#### format those log plots! 
minorFormatter = jLogFormatter(base=10, labelOnlyBase=False)
majorFormatter = jLogFormatter(base=10, labelOnlyBase=True)

#### nsamp
nsamp = 15
cmap = get_cmap(nsamp)

### load shit
param_file = '/Users/joel/code/python/threedhst_bsfh/parameter_files/np_mocks_smooth/np_mocks_smooth_params.py'
run_params = model_setup.get_run_params(param_file=param_file)
sps = model_setup.load_sps(**run_params)

#### CALCULATE SPECTRA
print sps.ssp.params['dust1'],sps.ssp.params['dust2']
sps.ssp.params['dust1'] = 0.0
sps.ssp.params['dust2'] = 0.0
wave, ssp_spectra = sps.ssp.get_spectrum(tage=0, peraa=False)
wave /= 1e4
xlim = (0.1,3)
plt_lam = (wave > xlim[0]) & (wave < xlim[1])
factor = 3e18 / sps.wavelengths[plt_lam]

#### PLOT SPECTRA
fig, ax = plt.subplots(1,1, figsize=(10,10))

nskip = 4
for ii in xrange(nsamp): ax.plot(wave[plt_lam],np.log10(ssp_spectra[ii*nskip,plt_lam]*factor)-0.3*ii,label=sps.logage[ii*nskip],color=cmap(ii),lw=2,alpha=0.5)

ax.set_xlabel(r'wavelength [$\mu$m]')
ax.set_ylabel(r'log($\nu$f$_{\nu}$)')
ax.legend(loc=1,prop={'size':7},title='log(age/yr)')
ax.set_xlim(xlim)
ax.set_ylim(-2,7)
ax.set_xscale('log',nonposx='clip',subsx=(2,4,7))
ax.xaxis.set_minor_formatter(minorFormatter)
ax.xaxis.set_major_formatter(majorFormatter)

outfig='ssp_test.png'
plt.savefig(outfig,dpi=150)
plt.close()
os.system('open '+outfig)
print 1/0
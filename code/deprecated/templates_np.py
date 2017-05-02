import matplotlib.pyplot as plt
import numpy as np
import np_mocks_params as nparams
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

#### format those log plots! 
minorFormatter = jLogFormatter(base=10, labelOnlyBase=False)
majorFormatter = jLogFormatter(base=10, labelOnlyBase=True)

def main():

	#### information
	param_file = '/Users/joel/code/python/threedhst_bsfh/parameter_files/np_mocks/np_mocks_params.py'
	labels = ['0 Myr-100 Myr','100 Myr-300 Myr', '300 Myr-1 Gyr','1 Gyr-3 Gyr','3 Gyr-6 Gyr','6 Gyr-13.6 Gyr']
	y_offsets = [0.68,0.1,-0.2,-0.35,-1.1,-1.45]
	x_offsets = [-.03,-0.02,-.05,-.09,.3,0.2]
	idx = 480
	colors = ['#9400D3','#31A9B8','#375E97','#3F681C','#FF9100','#FF420E']
	outfig = '/Users/joel/my_papers/prospector_brown/figures/nonparametric_templates.png'
	ylim = (0,7.5)


	#### LOAD SPS
	run_params = model_setup.get_run_params(param_file=param_file)
	sps = model_setup.load_sps(**run_params)
	model = model_setup.load_model(**run_params)
	model.params['dust1'] = np.array(0.0)
	model.params['dust2'] = np.array(0.0)
	obs = model_setup.load_obs(**run_params)
	_,_,_ = model.mean_model(model.initial_theta, obs, sps=sps)

	#### GENERATE SPECTRA
	sps.ssp.params['dust1'] = 0.0
	sps.ssp.params['dust2'] = 0.0
	wave, ssp_spectra = sps.ssp.get_spectrum(tage=0, peraa=False)
	wave /= 1e4
	xlim = (0.1,4)
	plt_lam = (wave > xlim[0]) & (wave < xlim[1])

	ssp_spectra = np.vstack([ssp_spectra[0,:], ssp_spectra])
	weights = sps._bin_weights # dimension of [NBIN, NSPEC]

	fig, ax = plt.subplots(1,1, figsize=(8,8))
	for ii in xrange(weights.shape[0]):
		spectrum = np.dot(weights[ii,:], ssp_spectra) / weights[ii,:].sum()
		factor = 3e18 / sps.wavelengths[plt_lam]
		ax.plot(wave[plt_lam],np.log10(spectrum[plt_lam]*factor),lw=1.5,color=colors[ii])
		ax.text(wave[plt_lam][idx]+x_offsets[ii],np.log10(spectrum[plt_lam][idx]*factor[idx])+y_offsets[ii], labels[ii],
			    color=colors[ii], ha='right')


	ax.set_xlabel(r'wavelength [$\mu$m]')
	ax.set_ylabel(r'log($\nu$f$_{\nu}$)')
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	ax.set_xscale('log',nonposx='clip',subsx=(2,4,7))
	ax.xaxis.set_minor_formatter(minorFormatter)
	ax.xaxis.set_major_formatter(majorFormatter)

	plt.savefig(outfig,dpi=150)
	plt.close()
	os.system('open '+outfig)

if __name__ == "__main__":
	sys.exit(main())
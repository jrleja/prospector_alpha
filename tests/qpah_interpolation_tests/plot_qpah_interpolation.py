from prospect.models import model_setup
import os, prosp_dutils
import numpy as np
import matplotlib.pyplot as plt
from magphys_plot_pref import jLogFormatter
import matplotlib.cm as cmx
import matplotlib.colors as colors

minorFormatter = jLogFormatter(base=10, labelOnlyBase=False)
majorFormatter = jLogFormatter(base=10, labelOnlyBase=True)

def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='jet') 
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color

def plot_qpah(no_dust = False):

	### custom parameter file
	### where mass is a six-parameter thing
	param_file = 'brownseds_np_params.py'

	run_params = model_setup.get_run_params(param_file=param_file)
	sps = model_setup.load_sps(**run_params)
	model = model_setup.load_model(**run_params)
	obs = model_setup.load_obs(**run_params)
	thetas = model.initial_theta

	#### set up parameters
	i1, i2 = model.theta_index['duste_qpah']
	nsamp = 11
	qpah = np.linspace(0.1,7,nsamp)
	spec, phot = [], []
	for q in qpah:
		thetas[i1:i2] = q
		specx, photx, x = model.mean_model(thetas, obs, sps=sps)
		spec.append(specx)
		phot.append(photx)

	### plot
	fig, ax = plt.subplots(1,1, figsize=(10, 9))
	cmap = get_cmap(nsamp)
	for i in xrange(nsamp): ax.plot(sps.wavelengths/1e4, np.log10(spec[i]),color=cmap(i), label=qpah[i],lw=1.5)
	ax.axhline(0, linestyle='--', color='0.1')
	ax.legend(loc=2,prop={'size':12},title=r'Q$_{\mathrm{PAH}}$')

	ax.set_xlim(1,30)
	ax.set_ylim(-8,-4)
	ax.set_xscale('log',nonposx='clip',subsx=(2,5))
	ax.xaxis.set_minor_formatter(minorFormatter)
	ax.xaxis.set_major_formatter(majorFormatter)

	ax.set_ylabel(r'log(flux)')
	ax.set_xlabel('wavelength [microns]')
	plt.tight_layout()
	plt.savefig('qpah_interpolation.png',dpi=150)
	print 1/0

if __name__ == "__main__":
	plot_qpah()
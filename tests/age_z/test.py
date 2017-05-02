from prospect.models import model_setup
import os, prosp_dutils
import numpy as np
import matplotlib.pyplot as plt
from magphys_plot_pref import jLogFormatter
import matplotlib.cm as cmx
import matplotlib.colors as colors

minorFormatter = jLogFormatter(base=10, labelOnlyBase=False)
majorFormatter = jLogFormatter(base=10, labelOnlyBase=True)

def return_declining_sfh(model,mformed, tau=1e9):

	### SFR = Ae^(-t/tau)
	### constant of integration
	agebins = 10**model.params['agebins']
	t1, t2 = agebins.min(), agebins.max()
	A = mformed / (tau * (np.exp(-t1/tau) - np.exp(-t2/tau)))

	### calculate the distribution of mass
	agebins_newt = agebins.max()-agebins
	outmass = np.zeros(shape=agebins.shape[0])
	for i in xrange(agebins.shape[0]): outmass[i] = tau * A * (np.exp(-agebins_newt[i,1]/tau)-np.exp(-agebins_newt[i,0]/tau))

	return outmass

def return_zt(model):
	''' 
	follow a simple closed box model, assume constant SFH

	following http://www.astro.rug.nl/~ahelmi/galaxies_course/class_VII/class_VII-chem.pdf
	Z(t) = Z(0) - p * ln(Mg(t)/Mg(0)) = gas metallicity as a function of time
	Z(0) = initial gas metallicity == 0
	Mg(t) = gas mass as a function of time = (assumption) c - ft
	--> Z(t) = -p ln(1-xt)
	p = yield = 0.5Zsun for local calculation
	assume that Z(13.6) = Zsun
	this means that x ~ (1-e^-2)/13.6 ~ 0.06, for t in Gyr
	''' 
	agebins = (10**model.params['agebins'])/1e9
	agebins_newt = agebins.max()-agebins

	constant = (1-np.exp(-2))/agebins_newt.mean(axis=1)[0]

	met = np.zeros(agebins.shape[0])
	for i in xrange(agebins.shape[0]): met[i] = -0.5*np.log(1-constant*agebins_newt.mean(axis=1)[i])

	return met

def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='gist_rainbow') 
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color


def compare_sed():

	### custom parameter file
	### where mass is a six-parameter thing
	param_file = 'brownseds_np_params.py'

	run_params = model_setup.get_run_params(param_file=param_file)
	sps = model_setup.load_sps(**run_params)
	model = model_setup.load_model(**run_params)
	obs = model_setup.load_obs(**run_params)
	thetas = model.initial_theta

	### create star formation history and metallicity history
	mformed = 1e10
	m_constant = return_declining_sfh(model,mformed, tau=1e12)
	met = return_zt(model)
	#met = np.ones_like(met)
	thetas[model.theta_labels().index('logzsol')] = 0.0

	### simple
	i1, i2 = model.theta_index['mass']
	thetas[i1:i2] = m_constant
	nsamp = 21
	met_comp = np.linspace(-1,0,nsamp)
	spec, phot = [], []
	for m in met_comp:
		thetas[model.theta_labels().index('logzsol')] = m
		specx, photx, x = model.mean_model(thetas, obs, sps=sps)
		spec.append(specx)
		phot.append(photx)

	### complex
	for i in xrange(met.shape[0]):

		# zero out all masses
		thetas[i1:i2] = np.zeros_like(m_constant)

		# fill in masses and metallicities
		thetas[model.theta_labels().index('logzsol')] = np.log10(met[i])
		thetas[i1+i] = m_constant[i]

		# generate spectrum, add to existing spectrum
		sps.ssp.params.dirtiness = 1
		specx, photx, x = model.mean_model(thetas, obs, sps=sps)

		if i == 0:
			spec_agez, phot_agez = specx, photx
		else:
			spec_agez += specx
			phot_agez += photx

	### plot
	fig, ax = plt.subplots(1,1, figsize=(8, 7))
	cmap = get_cmap(nsamp)
	for i in xrange(0,nsamp,4): ax.plot(sps.wavelengths/1e4, np.log10(spec[i] / spec_agez),color=cmap(i), label=met_comp[i])
	ax.axhline(0, linestyle='--', color='0.1')
	ax.legend(loc=4,prop={'size':20},title=r'log(Z/Z$_{\odot}$) [fixed]')

	ax.set_xlim(0.1,10)
	ax.set_xscale('log',nonposx='clip',subsx=(2,5))
	ax.xaxis.set_minor_formatter(minorFormatter)
	ax.xaxis.set_major_formatter(majorFormatter)

	ax.set_ylabel(r'log(f$_{\mathrm{Z}_{\mathrm{fixed}}}$ / f$_{\mathrm{Z(t)}}$)')
	ax.set_xlabel('wavelength [microns]')
	ax.set_ylim(-0.4,0.4)
	plt.tight_layout()
	plt.show()
	print 1/0


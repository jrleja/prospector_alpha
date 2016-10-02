import numpy as np
import os
from sedpy.observate import getSED,load_filters

lightspeed = 2.998e18  # AA/s

def load_data():

	loc = os.getenv('SPS_HOME')+'/dust/Nenkova08_y010_torusg_n10_q2.0.dat'

	hdr = ['blank','lambda',5,10,20,30,40,60,80,100,150]
	dat = np.loadtxt(loc, comments = '#', delimiter='   ',skiprows=4,
                     dtype = {'names':([str(n) for n in hdr]),\
                              'formats':(np.concatenate((np.atleast_1d('S4'),np.repeat('float64',10))))})

	return dat

def plot():

	import matplotlib.pyplot as plt
	import magphys_plot_pref

	minorFormatter = magphys_plot_pref.jLogFormatter(base=10, labelOnlyBase=False)
	majorFormatter = magphys_plot_pref.jLogFormatter(base=10, labelOnlyBase=True)

	dat = load_data()

	fig, ax = plt.subplots(1,1, figsize=(8, 8))

	for name in dat.dtype.names:

		if name == 'blank' or name == 'lambda':
			continue

		ax.plot(dat['lambda']/1e4,dat[name]*3e18/dat['lambda'],label=name,alpha=0.8,lw=2)

	ax.legend(title=r'$\tau_{\mathrm{V}}$',loc=3)
	ax.set_ylabel(r'$\nu$f$_{\nu}$')
	ax.set_xlabel(r'wavelength [micron]')

	ax.set_xscale('log',nonposx='clip',subsx=(1,2,4))
	ax.xaxis.set_minor_formatter(minorFormatter)
	ax.xaxis.set_major_formatter(majorFormatter)

	ax.set_yscale('log',nonposy='clip',subsy=([1]))
	ax.yaxis.set_minor_formatter(minorFormatter)
	ax.yaxis.set_major_formatter(majorFormatter)

	ax.set_xlim(0.1,1000)
	ax.set_ylim(1e-8,1)

	plt.show()



def observe(fnames):

	#  units: lambda (A), flux: fnu normalized to unity
	dat = load_data()
	filters = load_filters(fnames)

	out = {}
	for name in dat.dtype.names:
		if name == 'blank' or name == 'lambda':
			continue

		# sourcewave: Spectrum wavelength (in AA), ndarray of shape (nwave).
		# sourceflux: Associated flux (assumed to be in erg/s/cm^2/AA), ndarray of shape (nsource,nwave).
	    # filterlist: List of filter objects, of length nfilt.
		# array of AB broadband magnitudes, of shape (nsource, nfilter).
		out[name] = getSED(dat['lambda'], (lightspeed/dat['lambda']**2)*dat[name], filters)

	return out

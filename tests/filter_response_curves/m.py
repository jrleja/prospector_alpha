import fsps, prospect
import sedpy.observate
import matplotlib.pyplot as pl
import numpy as np
import os


def load_ssc_curve(name):

	dat = np.loadtxt(name, comments = '#',
					 dtype = {'names':(['lambda','transmission']),
					         'formats':('f16','f16')})
	return dat

sps = fsps.StellarPopulation(zcontinuous=1, compute_vega_mags=False)
w, s = sps.get_spectrum(tage=1.0, peraa=True)

'''
flist=sedpy.observate.load_filters(['galex_FUV', 'bessell_V'])
sedpy_mags = sedpy.observate.getSED(w, s*prospect.sources.to_cgs, filterlist=flist)
fsps_mags = sps.get_mags(tage=1.0, bands=['galex_fuv', 'v'])
print(sedpy_mags - fsps_mags)
'''

# setup plot
fig, axes = pl.subplots(ncols=2, nrows=2, figsize=(12,12))
nbins = 100
alpha = 0.5
lw = 2
color = 'blue'
histtype = 'step'

ax = np.ravel(axes)

# setup filters
ssc_filters = ['irac1','irac2','irac3','irac4']
fsps_filters = ['irac_1','irac_2','irac_3','irac_4']
sedpy_filters = ['spitzer_irac_ch1','spitzer_irac_ch2','spitzer_irac_ch3','spitzer_irac_ch4']
name = ['IRAC1','IRAC2','IRAC3','IRAC4']

for ii in xrange(len(fsps_filters)):
	firac = fsps.get_filter(fsps_filters[ii])
	sirac = sedpy.observate.Filter(sedpy_filters[ii])
	ax[ii].plot(sirac.wavelength/1e4, sirac.transmission / sirac.transmission.max(), label='sedpy',lw=1.5)
	ax[ii].plot(firac.transmission[0]/1e4, firac.transmission[1] /  firac.transmission[1].max(), label='FSPS',lw=1.5)
	
	nirac = load_ssc_curve(ssc_filters[ii]+'.txt')
	ax[ii].plot(nirac['lambda'], nirac['transmission'] / nirac['transmission'].max(), label='new',lw=1.5)


	if ii == 0:
		ax[ii].legend(frameon=False)

	ax[ii].set_xlabel(r'wavelength [$\mu$m]')
	ax[ii].set_ylabel(r'filter response '+name[ii])

outfile = 'irac_response_comparison.png'
pl.savefig(outfile,dpi=150)
os.system('open '+outfile)
pl.close()
print 1/0

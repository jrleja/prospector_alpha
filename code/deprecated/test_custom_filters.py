import numpy as np
import fsps
import matplotlib.pyplot as pl
from bsfh.model_setup import custom_filter_dict

keypath = '/Users/joel/code/python/download/python-fsps/fsps/data/'
fd = custom_filter_dict(keypath+'filter_keys_threedhst.txt')
fsps.filters.FILTERS = fd

sps = fsps.StellarPopulation(compute_vega_mags=False)
bands = sorted(fd.keys(), key = lambda x: fd[x].index) 
wave, spec = sps.get_spectrum(tage=1, zmet=4)
mags = sps.get_mags(tage=1, zmet=4, bands = bands)
wave_eff, mvega, msun = sps.filter_data()

lsun, pc = 3.846e33, 3.085677581467192e18 
pl.plot(wave, spec, color='b')
pl.plot(wave_eff, 10**(-0.4 * (mags+48.6)),
        linestyle='', marker='o', color='red', markersize=5)
pl.xlim(1e3, 3e4)
pl.yscale('log')
pl.xlabel(r'wavelength ($\AA$)')
pl.ylabel(r'L$_\odot/$Hz')
pl.show()

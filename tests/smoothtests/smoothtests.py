import numpy as np
import fsps, threed_dutils
import matplotlib.pyplot as plt

# change to a folder name, to save figures
outname = ''

# set up sps
sps = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1)
sps.params['tage'] = 5
w = sps.wavelengths

# calculate unsmoothed spectrum
w, spec = sps.get_spectrum(tage=sps.params['tage'], peraa=True)

# calculate smoothed
sigsmooth = 450 # km/s
sps.params['sigma_smooth'] = sigsmooth
w, spec_smooth = sps.get_spectrum(tage=sps.params['tage'], peraa=True)

plt.plot(w,spec,color='blue')
plt.plot(w,spec_smooth,color='red')
plt.xlim(2000,4000)



spec_joelsmooth = threed_dutils.smooth_spectrum(w,spec,sigsmooth)

plt.plot(w,spec_joelsmooth,color='green')
#plt.plot(w,(spec_smooth-spec)/spec_smooth,color='red')
plt.xlim(2000,4000)

plt.show()

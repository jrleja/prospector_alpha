import numpy as np
import fsps

# setup obs, sps
sps = fsps.StellarPopulation(zcontinuous=0, compute_vega_mags=False)
sps.params['sfh'] = 1.0
sps.params['dust_type'] = 4
sps.params['tage'] = 1.0
sps.params['dust2'] = np.atleast_1d(0.05)

mags = sps.get_mags(tage=sps.params['tage'])
print mags

sps.params['tage'] = 0.999999
mags = sps.get_mags(tage=sps.params['tage'])
print mags

sps.params['tage'] = 1.000001
mags = sps.get_mags(tage=sps.params['tage'])
print mags
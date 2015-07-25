import numpy as np
import fsps, pickle
import matplotlib.pyplot as plt

# change to a folder name, to save figures
outname = ''

# set up sps
sps = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1)
w = sps.wavelengths

sps.params['sfh'] = 4 #composite
sps.params['imf_type'] = 0 # Salpeter
#sps.params['tage'] = 14.0
sps.params['sf_start'] = 0.0
sps.params['tau']  = 1.0
sps.params['dust2'] = 0.0

if len(sps.zlegend) > 10:

    sps.params['zmet'] = 22
    sps.params['logzsol'] = 0.0

else:

    sps.params['zmet'] = 4
    sps.params['logzsol'] = 0.0

# range in sf_start
#sf_start = np.array([0.0, 3.0, 6.0, 9.0, 12.0, 13.0, 13.5, 13.8])
tage = np.array([0.0, 3.0, 6.0, 9.0, 12.0, 13.0, 13.5, 13.8])+0.2
outspec,outsm,outlbol = [],[],[]

#for start in sf_start:
for age in tage:

    sps.params['tage'] = age
    w, spec = sps.get_spectrum(tage=sps.params['tage'], peraa=True)

    outsm.append(sps.stellar_mass)
    outspec.append(spec)
    outlbol.append(sps.log_lbol)

out = {'outspec': outspec, 'wavelength': w, 'tage': tage, 'sm': outsm, \
       'lbol': outlbol}

pickle.dump(out, open('time_res_incr=8.pickle','wb'))
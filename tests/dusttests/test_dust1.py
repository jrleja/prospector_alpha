import numpy as np
from astropy import constants
import fsps, threed_dutils
import matplotlib.pyplot as plt
from scipy.special import gamma, gammainc

# change to a folder name, to save figures
outname = ''

def integrate_sfh_ben(t1, t2, tage, tau, sf_start, tburst=0, fburst=0):

    normalized_times = (np.array([t1, t2, tage]) - sf_start) / tau
    mass = gammainc(2, normalized_times)
    intsfr = (mass[1] - mass[0]) / mass[2]

    return intsfr

sps = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1)
w = sps.wavelengths

# sfh parameters
sps.params['sfh'] = 1 #composite
sps.params['imf_type'] = 0 # Salpeter
sps.params['const'] = 1 # all constant
sps.params['tage'] = 0.4 # Gyr
sps.params['tau']  = 1.0

# dust parameters
sps.params['dust_type'] = 0 # Charlot & Fall
sps.params['dust_tesc'] = 8.7 # stars at tage < 0.5 Gyr affected by dust1
sps.params['dust2'] = 0.0 # no diffuse dust

if len(sps.zlegend) > 10:
    # We are using BaSeL with it's poorly resolved grid,
    # so we need to broaden the lines to see them

    sps.params['sigma_smooth'] = 1e3
    halpha = (w > 6500) & (w < 6650)
    sps.params['zmet'] = 22
    sps.params['logzsol'] = 0.0

else:
    # we are using MILES with better optical resolution

    sps.params['sigma_smooth'] = 0.0
    halpha = (w > 6556) & (w < 6573)
    sps.params['zmet'] = 4
    sps.params['logzsol'] = 0.0

# get the nebular free spectrum
ns = 40
ks = np.zeros(ns)
cloudy = np.zeros(ns)
gu = np.random.uniform(-4, 1, ns)
gz = np.random.uniform(-2, 0.2, ns)

# what parameters do we use?
params = [{'name':'dust2','range':(0,4)}
          ]

deltat=0.1
for par in params:

    # draw from flat prob distribution, N_samp - 1 ( add default to samp later )
    # save initial, restore later
    sample = np.random.uniform(par['range'][0],par['range'][1],ns-1)
    default = sps.params[par['name']]

    # create figure
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i,samp in enumerate(sample):

        # vary parameter
        sps.params[par['name']] = samp
        sps.params['add_neb_emission'] = True
        sps.params['add_neb_continuum'] = True
        w, nebspec = sps.get_spectrum(tage=sps.params['tage'], peraa=True)
        
        plt.plot(np.log10(w),np.log10(nebspec))

    print 1/0
    plt.close()

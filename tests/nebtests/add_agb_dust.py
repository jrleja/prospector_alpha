import numpy as np
from astropy import constants
import fsps, prosp_dutils,pylab
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

sps.params['sfh'] = 4 #composite
sps.params['imf_type'] = 0 # Salpeter
sps.params['smooth_velocity'] = True
sps.params['tage'] = 13.0
sps.params['tau']  = 1.0
sps.params['dust2'] = 0.35
sps.params['logzsol'] = 0.0

k98 = 7.9e-42 #(M_sun/yr) / (erg/s)

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
sps.params['add_dust_emission'] = True
ns = 8
ks = np.zeros(ns)
cloudy = np.zeros(ns)


# what parameters do we use?
params = [{'name':'tau','range':(0.3,9)},
          {'name':'sf_start','range':(0.0,12.0)},
          ]

deltat=0.1
for par in params:

    # sample from parameter
    sample = np.linspace(par['range'][0],par['range'][1],ns)

    # save default
    default = sps.params[par['name']]

    # create figure
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # cycle colors
    cm = pylab.get_cmap('cool')
    plt.rcParams['axes.color_cycle'] = [cm(1.*i/ns) for i in range(ns)] 

    for i,samp in enumerate(sample):

        # vary parameter
        sps.params[par['name']] = samp

        # lineflux in cgs
        x = prosp_dutils.measure_emline_lum(sps)
        cloudy[i] = constants.L_sun.cgs.value * x['emline_flux'][x['emline_name'] == 'Halpha']

        sps.params['add_agb_dust_model'] = False
        sps.params['add_dust_emission'] = False
        w, spec = sps.get_spectrum(tage=sps.params['tage'], peraa=True)
        sps.params['add_agb_dust_model'] = True
        sps.params['add_dust_emission'] = False
        w, nebspec = sps.get_spectrum(tage=sps.params['tage'], peraa=True)
        
        ax.plot(w/1e4,(nebspec-spec)/spec,
                label = "{:10.2f}".format(samp))

    # restore sps default
    sps.params[par['name']] = default

    # plot
    ax.legend(loc=0,prop={'size':6},
                              frameon=False,
                              title=par['name'])
    ax.set_ylabel(r'[f$_{AGB-ON}$-f$_{AGB-OFF}$]/f$_{AGB-OFF}$')
    ax.set_xlabel(r'$\lambda$ [$\mu m$]')
    ax.set_xlim(1.0,12.0)
    ax.set_ylim(-0.1,3.0)
    plt.savefig(outname+par['name']+'_agbtest.png',dpi=300)
    plt.close()


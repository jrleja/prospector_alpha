import numpy as np
from astropy import constants
import fsps
import matplotlib.pyplot as plt

# change to a folder name, to save figures
outname = ''

def integrate_sfh(t1,t2,tage,tau,sf_start,tburst,fburst):
    
    '''
    integrate a delayed tau SFH between t1 and t2
    '''
    
    # if we're outside of the boundaries, return boundary values
    if t2 < sf_start:
        return 0.0
    elif t2 > tage:
        return 1.0

    intsfr = (np.exp(-t1/tau)*(1+t1/tau) - 
              np.exp(-t2/tau)*(1+t2/tau))
    norm=(1.0-fburst)/(np.exp(-sf_start/tau)*(sf_start/tau+1)-
              np.exp(-tage    /tau)*(tage    /tau+1))
    intsfr=intsfr*norm
    
    # if burst occured in this time
    if (t2 > tburst) & (t1 < tburst):
        intsfr=intsfr+fburst

    return intsfr

sps = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=0)
w = sps.wavelengths

sps.params['sfh'] = 4 #composite
sps.params['imf_type'] = 0 # Salpeter
sps.params['smooth_velocity'] = True
sps.params['tage'] = 13.0
sps.params['tau']  = 1.0
sps.params['dust2'] = 0.0

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
sps.params['add_neb_emission'] = False
w, spec = sps.get_spectrum(tage=sps.params['tage'], peraa=True)

sps.params['add_neb_emission'] = True
sps.params['add_neb_continuum'] = True

ns = 40
ks = np.zeros(ns)
cloudy = np.zeros(ns)
gu = np.random.uniform(-4, 1, ns)
gz = np.random.uniform(-2, 0.2, ns)

# what parameters do we use?
params = [{'name':'gas_logu','range':(-4,1)},
          {'name':'gas_logz','range':(-2,0.2)},
          {'name':'tau','range':(0.3,5)},
          {'name':'sf_start','range':(0.0,12.9)}
          ]

deltat=0.1
for par in params:
    
    # draw from flat prob distribution, N_samp - 1 ( add default to samp later )
    # save initial, restore later
    sample = np.random.uniform(par['range'][0],par['range'][1],ns-1)
    default = sps.params[par['name']]

    # add default value to sample
    sample = np.concatenate((np.atleast_1d(default),sample))

    # initialize output
    sfr_100     = np.zeros(ns)
    cloudy      = np.zeros(ns)

    for i,samp in enumerate(sample):
        sps.params[par['name']] = samp
        w, nebspec = sps.get_spectrum(tage=sps.params['tage'], peraa=True)

        sfr_100[i] = integrate_sfh(sps.params['tage']-deltat,
                                   sps.params['tage'],
                                   sps.params['tage'],
                                   sps.params['tau'],
                                   sps.params['sf_start'],
                                   sps.params['tburst'],
                                   sps.params['fburst'])/(deltat*1e9)
        # lineflux in cgs
        cloudy[i] = constants.L_sun.cgs.value * np.trapz((nebspec - spec)[halpha], w[halpha])
    
    # restore sps default
    sps.params[par['name']] = default

    # create figure
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # this sets any line flux < 0
    # to half of the largest line flux above 0
    min_flux = np.min(cloudy[cloudy > 0])
    below_min = cloudy < min_flux
    if np.sum(below_min) > 0:
        cloudy[below_min] = min_flux*0.5

    # plot
    ax.plot(sample,np.log10(sfr_100/cloudy), 'bo', alpha=0.5,linestyle=' ')
    ax.plot(sample[0],np.log10(sfr_100[0]/cloudy[0]), 'ro',linestyle=' ')
    plt.axhline(np.log10(k98),linestyle='--',color='black')
    ax.set_ylabel(r'SFR/H$\alpha$-CLOUDY')
    ax.set_xlabel(par['name'])

    # statistics
    log_of_mean = np.log10(np.mean(sfr_100/cloudy))-np.log10(k98)
    ax.text(0.03, 0.03,"mean offset ="+"{:.2f}".format(log_of_mean)+" dex",transform = ax.transAxes)

    plt.savefig(outname+par['name']+'_nebtest.png',dpi=300)
    plt.close()

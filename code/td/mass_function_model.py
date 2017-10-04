import numpy as np
from scipy.integrate import simps
from astropy.cosmology import WMAP9
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, interp2d
import hickle


dloc = '/Users/joel/code/python/prospector_alpha/plots/td/fast_plots/data/masscomp.h5'
with open(dloc, "r") as f:
    mcorr = hickle.load(f)
mass_fnc = interp2d(mcorr['fast_mass'], mcorr['z'], mcorr['log_mprosp_mfast'], kind='linear')

dloc = '/Users/joel/code/python/prospector_alpha/plots/td/fast_plots/data/ssfrcomp.h5'
with open(dloc, "r") as f:
    ssfrcorr = hickle.load(f)
ssfr_fnc = interp2d(ssfrcorr['log_sfruvir_mfast'], ssfrcorr['z'], ssfrcorr['log_sfrprosp_sfruvir'], kind='linear')

def sf_fraction(z,logm):
    """this returns the fraction of star-forming galaxies as a function of 
    mass and redshift
    """
    spars_qu = mf_parameters(z,qu=True)
    spars_sf = mf_parameters(z,sf=True)

    return mf_phi(spars_sf,logm) / (mf_phi(spars_sf,logm) + mf_phi(spars_qu,logm))

def sfr_ms(z,logm):
    """ returns the SFR of the star-forming sequence from Whitaker+14
    as a function of mass and redshift
    note that this is only valid over 0.5 < z < 2.5.
    we use the broken power law form (as opposed to the quadratic form)
    """

    # sanitize logm
    sfr_out = np.zeros_like(logm)
    logm = np.atleast_2d(logm).T

    # kick us out if we're doing something bad
    if (z < 0.5) | (z > 2.5):
        print "we're outside the allowed redshift range. intentionally barfing."
        print 1/0

    # parameters from whitaker+14
    zwhit = np.atleast_2d([0.75, 1.25, 1.75, 2.25])
    alow = np.array([0.94,0.99,1.04,0.91])
    ahigh = np.array([0.14,0.51,0.62,0.67])
    b = np.array([1.11, 1.31, 1.49, 1.62])

    # generate SFR(M) at all redshifts 
    log_sfr = alow*(logm - 10.2) + b
    high = (logm > 10.2).squeeze()
    log_sfr[high] = ahigh*(logm[high] - 10.2) + b

    # interpolate to proper redshift
    for i in range(sfr_out.shape[0]): 
        tmp = interp1d(zwhit.squeeze(), log_sfr[i,:],fill_value='extrapolate')
        sfr_out[i] = 10**tmp(z)

    return sfr_out

def sfr_ms12(z,logm):
    """ returns the SFR of the star-forming sequence from Whitaker+12
    as a function of mass and redshift
    note that this is only valid over 0.5 < z < 2.5.
    """
    alpha = 0.70 - 0.13*z
    beta = 0.38 + 1.14*z - 0.19*z**2
    log_sfr = alpha*(logm - 10.5) + beta
    return 10**log_sfr

def mf_phi(spars,logm):
    """ returns the number density of galaxies, per dex per Mpc^-3
    requires spars input dictionary from return_mf_par
    """
    phi = np.log(10)*np.exp(-10**(logm-spars['log_mstar'])) * \
          (10**spars['log_phi1']*10**((spars['alpha1']+1)*(logm-spars['log_mstar'])) + \
           10**spars['log_phi2']*10**((spars['alpha2']+1)*(logm-spars['log_mstar'])))

    return phi

def mf_parameters(z,sf=False,qu=False):
    """ returns double schechter parameters describing the redshift evolution of the 
    Tomczak+14 mass function over 0.01 < z < 2.5, as published in Leja+15
    keywords are for quiescent, star-forming, or total"""
    if qu:
        spars = {
                'log_phi1': -2.51 - 0.33*z - 0.07*z**2,
                'log_phi2': -3.54 - 2.31*z + 0.73*z**2,
                'log_mstar': 10.70,
                'alpha1': -0.1,
                'alpha2': -1.69
                }
    elif sf:
        spars = {
                'log_phi1': -2.88 + 0.11*z - 0.31*z**2,
                'log_phi2': -3.48 + 0.07*z - 0.11*z**2,
                'log_mstar': 10.67-0.02*z+0.10*z**2,
                'alpha1': -0.97,
                'alpha2': -1.58
                }
    else:
        spars = {
                'log_phi1': -2.46 + 0.07*z - 0.28*z**2, 
                'log_phi2': -3.11 - 0.18*z - 0.03*z**2,
                'log_mstar': 10.72 - 0.13*z + 0.11*z**2, 
                'alpha1': -0.39,
                'alpha2': -1.53
                }

    return spars

def sfrd(z,logm_min=9, logm_max=13, dm=0.01, use_whit12=False, 
         apply_pcorrections=False,**kwargs):
    """ calculates SFRD from analytic star-forming mass function
    plus the Whitaker+14 star-forming sequence
    """

    # generate stellar mass array
    # this represents the CENTER of each `bin`
    logm = np.arange(logm_min,logm_max,dm)+(dm/2.)

    # first, get the star-forming sequence
    # also apply corrections to SFR if necessary
    if use_whit12:
        sfr = sfr_ms12(z,logm)
    else:
        sfr = sfr_ms(z,logm)
    # input: log(sfr[UVIR]/M[fast]), z. output: log(sfr[prosp]/sfr[uvir])
    if apply_pcorrections:
        sfr += ssfr_fnc(np.log10(sfr/10**logm),z)

    # multiply n(M) by SFR(M) to get SFR / Mpc^-3 / dex
    spars = mf_parameters(z,sf=True)
    phi_sf = sfr * mf_phi(spars,logm)

    # integrate over stellar mass to get SFRD
    sfrd = simps(phi_sf,dx=dm)

    return sfrd

def drho_dt(z, logm_min=9, logm_max=13, dm=0.01, dz=0.001, 
           massloss_correction=False, apply_pcorrections=False, **kwargs):
    """ calculates d(rho)/dt from evolution of stellar mass function
    this is a numerical approximation in redshift AND mass
    """

    # generate stellar mass array
    # this represents the CENTER of each `bin`
    logm = np.arange(logm_min,logm_max,dm)+(dm/2.)

    # generate delta(phi[z])
    # if we apply the Prospector M(M) corrections, do that here
    spars, spars_dz = mf_parameters(z), mf_parameters(z+dz)
    phi_dz = 10**mf_phi(spars,logm) - 10**mf_phi(spars_dz,logm)
    if apply_pcorrections:
        out = load_pcorrections()
        logm += mass_fnc(logm,z)

    # now get mass_formed
    mass_formed = simps(phi_dz*10**logm,dx=dm)

    # divide by delta t + mass-loss to get sfrd
    delta_t = (WMAP9.age(z).value - WMAP9.age(z+dz).value)*1e9
    sfrd = (mass_formed/delta_t)

    if massloss_correction:
        sfrd = sfrd / 0.64

    return sfrd

def plot_sfrd(logm_min=9,logm_max=13,dm=0.01,use_whit12=False,
              massloss_correction=False, apply_pcorrections=False):
    """ compare SFRD from mass function(z) versus observed SFR
    """

    # generate z-array + numerical options
    dz = 0.1 # for stability...
    zrange = np.arange(0.5, 2.5, dz)
    opts = {
            'logm_min': logm_min,
            'logm_max': logm_max,
            'dm': dm,
            'use_whit12': use_whit12,
            'massloss_correction': massloss_correction,
            'apply_pcorrections': apply_pcorrections
           }

    # calculate both
    sf_sfrd, mf_sfrd = [], []
    for z in zrange:
        sf_sfrd += [sfrd(z,**opts)]
        mf_sfrd += [drho_dt(z, dz=dz, **opts)]

    # plot both
    fig, ax = plt.subplots(1,1, figsize=(4, 4))
    red, blue = '#FF3D0D', '#1C86EE'
    popts = {
             'linewidth': 3,
             'alpha': 0.9
            }
    ax.plot(zrange, np.log10(sf_sfrd), color=blue, label='star formation', **popts)
    ax.plot(zrange, np.log10(mf_sfrd), color=red, label='mass function',**popts)

    ax.set_xlabel('redshift')
    ax.set_ylabel(r'log(SFRD) [M$_{\odot}$/yr]')

    ax.legend(loc=4, prop={'size':12},
              scatterpoints=1,fancybox=True)

    plt.tight_layout()
    plt.show()

import numpy as np
from scipy.integrate import simps
from astropy.cosmology import WMAP9
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, interp2d, RectBivariateSpline
import hickle, os


dloc = '/Users/joel/code/python/prospector_alpha/plots/td_new/fast_plots/data/masscomp.h5'
with open(dloc, "r") as f:
    mcorr = hickle.load(f)
mass_fnc = RectBivariateSpline(mcorr['fast_mass'], mcorr['z'], mcorr['log_mprosp_mfast'],kx=1,ky=1)

dloc = '/Users/joel/code/python/prospector_alpha/plots/td_new/fast_plots/data/sfrcomp.h5'
with open(dloc, "r") as f:
    sfrcorr = hickle.load(f)
sfr_fnc = RectBivariateSpline(sfrcorr['mass'], sfrcorr['z'], sfrcorr['sfr'],kx=1,ky=1)

dloc = '/Users/joel/code/python/prospector_alpha/plots/td_new/fast_plots/data/sfrcomp_uvir.h5'
with open(dloc, "r") as f:
    sfrcorr_uvir = hickle.load(f)
sfr_fnc_uvir = RectBivariateSpline(sfrcorr_uvir['mass'], sfrcorr_uvir['z'], sfrcorr_uvir['sfr'],kx=1,ky=1)

# sfr_fnc = RectBivariateSpline(sfrcorr['mass'], sfrcorr['z'], sfrcorr['sfr_corr'],kx=1,ky=1)

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
    if (z < 0.5).any() | (z > 2.5).any():
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
         apply_pcorrections=False,use_avg=False,**kwargs):
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
        #sfr = sfr_ms(z,logm)
        sfr = sfr_fnc_uvir(logm,z).squeeze()

    if apply_pcorrections:
        sfr = sfr_fnc(logm,z).squeeze()

    # multiply n(M) by SFR(M) to get SFR / Mpc^-3 / dex
    numdens = mf_phi(mf_parameters(z),logm)
    starforming_function = sfr * numdens

    #starforming_numdens = mf_phi(mf_parameters(z,sf=True),logm)
    #starforming_function = sfr * starforming_numdens

    # integrate over stellar mass to get SFRD
    sfrd = simps(starforming_function,dx=dm)

    # plots to investigate interpolation
    '''
    if (z > 0.95) & (z < 1.05) & (apply_pcorrections):

        fig1, ax1 = plt.subplots(1,1, figsize=(5, 5))
        """
        idx = np.array(sfrcorr['z']) == 1.25
        ax1.plot(np.array(sfrcorr['mass'])[idx],np.array(sfrcorr['sfr'])[idx],'o',linestyle=' ',color='red')
        idx = np.array(sfrcorr['z']) == 0.75
        ax1.plot(np.array(sfrcorr['mass'])[idx],np.array(sfrcorr['sfr'])[idx],'o',linestyle=' ',color='blue')
        """
        ax1.plot(logm,sfr_fnc(logm,1.25),linestyle='-',color='red')
        ax1.plot(logm,sfr_fnc(logm,0.75),linestyle='-',color='blue')
        ax1.plot(logm,sfr_fnc(logm,1.0),linestyle='-',color='k')
        fig1.show()
        
        fig2, ax2 = plt.subplots(1,1, figsize=(5, 5))
        ax2.plot(logm, np.log10(mf_phi(mf_parameters(1.0,sf=True),logm)),color='black')
        ax2.plot(logm, np.log10(mf_phi(mf_parameters(0.75,sf=True),logm)),color='blue')
        ax2.plot(logm, np.log10(mf_phi(mf_parameters(1.25,sf=True),logm)),color='red')
        fig2.show()

        fig3, ax3 = plt.subplots(1,1, figsize=(5, 5))
        for zpl, col in zip([0.75,1.0,1.25],['blue','black','red']):
            sf = sfr_fnc(logm,zpl).squeeze() * mf_phi(mf_parameters(zpl,sf=True),logm)
            ax3.plot(logm, sf,color=col)
        fig3.show()
        print 1/0
    '''
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
        logm += mass_fnc(logm,z)

    # now get mass_formed
    mass_formed = simps(phi_dz*10**logm,dx=dm)

    # divide by delta t + mass-loss to get sfrd
    delta_t = (WMAP9.age(z).value - WMAP9.age(z+dz).value)*1e9
    sfrd = (mass_formed/delta_t)

    if massloss_correction:
        sfrd = sfrd / 0.64

    return sfrd

def zfourge_param_rhostar(z, massloss_correction=False, apply_pcorrections=False, logm_interp=11, **opts):
    """using equation (5) in Tomczak et al. 2014 instead of calculating directly from evolution of mass function
    this removes the "bump" at z~1.5 which is likely erroneous!
    applies for 9 < log(M) < 13
    """

    # this is the function we'll need
    def zfourge_rhostar(z):
        a, b = -0.33, 8.75
        return (a*(1+z)+b)

    # turn into rhodot
    nz, dz = len(z), 0.005
    rhodot = np.zeros(nz)
    for i in range(nz):
        upz, downz = z[i]+dz/2., z[i]-dz/2.
        logrho_down = zfourge_rhostar(downz)
        logrho_up = zfourge_rhostar(upz)
        if apply_pcorrections:
            logm_increase = mass_fnc(logm_interp,z[i])[0]
            logrho_down += logm_increase
            logrho_up += logm_increase
        delta_rho = 10**logrho_down - 10**logrho_up
        delta_t = (WMAP9.age(downz).value - WMAP9.age(upz).value)*1e9
        rhodot[i] = delta_rho/delta_t

    if massloss_correction:
        rhodot /= 0.64

    return np.log10(rhodot)

def plot_sfrd(logm_min=9.,logm_max=12,dm=0.01,use_whit12=False,
              massloss_correction=True):
    """ compare SFRD from mass function(z) versus observed SFR
    """

    # generate z-array + numerical options
    dz = 0.1
    zrange = np.arange(0.75, 2.25, dz)
    opts = {
            'logm_min': logm_min,
            'logm_max': logm_max,
            'dm': dm,
            'use_whit12': use_whit12,
            'massloss_correction': massloss_correction,
            'use_avg': True
           }

    # calculate both
    mf_sfrd = zfourge_param_rhostar(zrange, **opts)
    mf_sfrd_prosp = zfourge_param_rhostar(zrange, apply_pcorrections = True, **opts)
    sf_sfrd, sf_sfrd_prosp,  = [], []
    for z in zrange:
        sf_sfrd += [sfrd(z,**opts)]
        sf_sfrd_prosp += [sfrd(z,apply_pcorrections=True,**opts)]
        # mf_sfrd += [drho_dt(z, dz=dz, **opts)]
        # mf_sfrd_prosp += [drho_dt(z,apply_pcorrections=True,dz=dz, **opts)]

    # plot options
    old_color, new_color = '0.3','#FF3D0D'
    mass_linestyle, sfr_linestyle = '-', '--'
    ylim = (-1.6,-0.75)
    xlim = (0., 2.7)
    popts = {
             'linewidth': 3,
             'alpha': 0.9
            }

    # Plot1: change in SFRD
    fig, ax = plt.subplots(1,1, figsize=(4.4, 4))
    ax.plot(zrange, mf_sfrd, mass_linestyle, color=old_color, label='FAST mass',**popts)
    ax.plot(zrange, mf_sfrd_prosp, mass_linestyle, color=new_color, label='Prospector mass',**popts)
    ax.plot(zrange, np.log10(sf_sfrd), sfr_linestyle, color=old_color, label='UV+IR SFR', **popts)
    ax.plot(zrange, np.log10(sf_sfrd_prosp), sfr_linestyle, color=new_color, label='Prospector SFR', **popts)

    # Madau+15
    zrange_madau = np.arange(xlim[0], xlim[1], dz)
    phi_madau = 0.015*(1+zrange_madau)**2.7 / (1+((1+zrange_madau)/2.9)**5.6)
    phi_madau = np.log10(phi_madau) - 0.24 # salpeter correction

    # labels, legends, and add Madau
    ax.set_xlabel('redshift')
    ax.set_ylabel(r'log(SFRD) [M$_{\odot}$ yr$^{-1}$ Mpc$^{-3}$]')
    ax.plot(zrange_madau, phi_madau, ':', color='purple', label='Madau+14\ncompilation', zorder=-1,**popts)
    ax.legend(loc=2, prop={'size':8.5}, scatterpoints=1,fancybox=True)

    # limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    outname = '/Users/joel/code/python/prospector_alpha/plots/td_new/fast_plots/madau_plot.png'
    plt.tight_layout()
    plt.savefig(outname,dpi=200)
    plt.close()
    os.system('open '+outname)
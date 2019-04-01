import numpy as np
from scipy.integrate import simps
from astropy.cosmology import WMAP9
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, interp2d, RectBivariateSpline
import hickle, os, glob
from astropy.io import ascii
import csfh
from scipy.ndimage.filters import gaussian_filter1d as smooth

def bpl_eqn(logm,b,a1,a2):
    idx = (logm > 10.2)
    logsfr = a2*(logm-10.2)+b
    logsfr[idx] = a1*(logm[idx]-10.2)+b
    return 10**logsfr

# delta m
dloc = '/Users/joel/code/python/prospector_alpha/plots/td_delta/fast_plots/data/masscomp.h5'
with open(dloc, "r") as f:
    mcorr = hickle.load(f)
mass_fnc = RectBivariateSpline(mcorr['fast_mass'], mcorr['z'], mcorr['log_mprosp_mfast'],kx=1,ky=1)

dloc = '/Users/joel/code/python/prospector_alpha/plots/td_delta/fast_plots/data/sfrcomp.h5'
with open(dloc, "r") as f:
    sfrcorr = hickle.load(f)
sfr_fnc = RectBivariateSpline(sfrcorr['mass'], sfrcorr['z'], sfrcorr['sfr'],kx=1,ky=1)

# average delta(SFR)
dloc = '/Users/joel/code/python/prospector_alpha/plots/td_delta/fast_plots/data/sfrcomp_uvir.h5'
with open(dloc, "r") as f:
    sfrcorr_uvir = hickle.load(f)
sfr_fnc_uvir = RectBivariateSpline(sfrcorr_uvir['mass'], sfrcorr_uvir['z'], sfrcorr_uvir['sfr'],kx=1,ky=1)

# fixed slope fit to conditional SFR
dloc = '/Users/joel/code/python/prospector_alpha/plots/td_delta/fast_plots/data/conditional_sfr_fit.h5'
with open(dloc, "r") as f:
    fits = hickle.load(f)

logm = np.linspace(6., 11.5, 200)
z = np.array([0.75,1.25,1.75,2.25])
sfr_prosp = np.array([bpl_eqn(logm,fits['b'][i],fits['a1'][i],fits['a2'][i]).tolist() for i in range(len(z))]).T
sfr_uvir = np.array([bpl_eqn(logm,fits['b_uvir'][i],fits['a1_uvir'][i],fits['a2_uvir'][i]).tolist() for i in range(len(z))]).T

prosp_fit_fixed_fnc = RectBivariateSpline(logm, z, sfr_prosp,kx=1,ky=1)
uvir_fit_fixed_fnc = RectBivariateSpline(logm, z, sfr_uvir,kx=1,ky=1)

# free slope fit to conditional SFR
dloc = '/Users/joel/code/python/prospector_alpha/plots/td_delta/fast_plots/data/conditional_sfr_fit_nofix.h5'
with open(dloc, "r") as f:
    fits = hickle.load(f)

logm = np.linspace(6., 11.5, 200)
z = np.array([0.75,1.25,1.75,2.25])
sfr_prosp = np.array([bpl_eqn(logm,fits['b'][i],fits['a1'][i],fits['a2'][i]).tolist() for i in range(len(z))]).T
sfr_uvir = np.array([bpl_eqn(logm,fits['b_uvir'][i],fits['a1_uvir'][i],fits['a2_uvir'][i]).tolist() for i in range(len(z))]).T

prosp_fit_fnc = RectBivariateSpline(logm, z, sfr_prosp,kx=1,ky=1)
uvir_fit_fnc = RectBivariateSpline(logm, z, sfr_uvir,kx=1,ky=1)

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

def mf_phi(spars,logm, cumulative=False):
    """ returns the number density of galaxies, per dex per Mpc^-3
    requires spars input dictionary from return_mf_par
    """
    phi = np.log(10)*np.exp(-10**(logm-spars['log_mstar'])) * \
          (10**spars['log_phi1']*10**((spars['alpha1']+1)*(logm-spars['log_mstar'])) + \
           10**spars['log_phi2']*10**((spars['alpha2']+1)*(logm-spars['log_mstar'])))

    if cumulative:
        phi = np.cumsum(phi[::-1])[::-1]

    return phi

def load_zfourge_mf():
    """ load data from Tomczak+14
    """
    
    loc='/Users/joel/code/IDLWorkspace82/ND_analytic/data/Table1_total-SMF.dat'
    data = ascii.read(loc)
    
    # original table changed. updated values:
    data_local=np.array([[-1.37,-1.53,-1.71,-1.86,-2.03,-2.01,-2.10,-2.17,-2.24,-2.31,-2.41,-2.53,-2.91,-3.46,-99],
                        [0.06,0.06,0.07,0.07,0.08,0.07,0.07,0.08,0.08,0.08,0.08,0.09,0.11,0.14,-99],
                        [0.07,0.07,0.08,0.08,0.09,0.08,0.09,0.10,0.10,0.09,0.10,0.11,0.15,0.18,-99],
                        [-99,-1.53,-1.60,-1.76,-1.86,-2.00,-2.12,-2.21,-2.25,-2.35,-2.45,-2.55,-2.82,-3.32,-99],
                        [-99,0.06,0.05,0.06,0.06,0.06,0.07,0.06,0.06,0.07,0.07,0.08,0.09,0.10,-99],
                        [-99,0.07,0.06,0.06,0.07,0.07,0.08,0.07,0.08,0.08,0.09,0.09,0.11,0.13,-99],
                        [-99,-99,-1.70,-1.86,-2.01,-2.10,-2.23,-2.39,-2.45,-2.45,-2.52,-2.59,-2.93,-3.47,-99],
                        [-99,-99,0.05,0.05,0.06,0.06,0.06,0.07,0.07,0.07,0.08,0.08,0.10,0.11,-99],
                        [-99,-99,0.06,0.06,0.06,0.07,0.07,0.08,0.09,0.09,0.09,0.10,0.13,0.15,-99],
                        [-99,-99,-99,-1.99,-2.14,-2.24,-2.29,-2.48,-2.59,-2.73,-2.64,-2.72,-3.01,-3.62,-99],
                        [-99,-99,-99,0.06,0.06,0.06,0.06,0.07,0.08,0.08,0.07,0.08,0.10,0.11,-99],
                        [-99,-99,-99,0.06,0.07,0.07,0.07,0.08,0.09,0.10,0.09,0.10,0.12,0.15,-99],
                        [-99,-99,-99,-2.02,-2.14,-2.28,-2.46,-2.53,-2.61,-2.68,-2.71,-2.84,-3.12,-3.65,-4.99],
                        [-99,-99,-99,0.06,0.06,0.06,0.07,0.07,0.08,0.08,0.08,0.08,0.10,0.12,0.30],
                        [-99,-99,-99,0.07,0.07,0.07,0.08,0.08,0.09,0.09,0.09,0.10,0.13,0.16,0.41],
                        [-99,-99,-99,-99,-2.20,-2.31,-2.41,-2.54,-2.67,-2.76,-2.87,-3.03,-3.13,-3.56,-4.27],
                        [-99,-99,-99,-99,0.05,0.05,0.05,0.06,0.06,0.06,0.07,0.08,0.08,0.10,0.12],
                        [-99,-99,-99,-99,0.06,0.06,0.06,0.06,0.07,0.07,0.08,0.09,0.10,0.13,0.15],
                        [-99,-99,-99,-99,-99,-2.53,-2.50,-2.63,-2.74,-2.91,-3.07,-3.35,-3.54,-3.89,-4.41],
                        [-99,-99,-99,-99,-99,0.06,0.06,0.06,0.07,0.08,0.09,0.10,0.12,0.12,0.14],
                        [-99,-99,-99,-99,-99,0.07,0.07,0.07,0.08,0.09,0.10,0.13,0.16,0.17,0.19],
                        [-99,-99,-99,-99,-99,-99,-2.65,-2.78,-3.02,-3.21,-3.35,-3.74,-4.00,-4.14,-4.73],
                        [-99,-99,-99,-99,-99,-99,0.06,0.07,0.08,0.09,0.10,0.13,0.18,0.17,0.31],
                        [-99,-99,-99,-99,-99,-99,0.07,0.08,0.09,0.10,0.13,0.17,0.25,0.28,2.00]])
    column_names = ['logphi0','eup0','elo0','logphi1','eup1','elo1','logphi2','eup2','elo2','logphi3','eup3','elo3','logphi4','eup4','elo4','logphi5','eup5','elo5']
    
    for n in range(len(column_names)): data[column_names[n]] = data_local[n,:]

    # add the last bits
    data['logphi6'] = data_local[-6,:]
    data['eup6'] = data_local[-5,:]
    data['elo6'] = data_local[-4,:]
    data['logphi7'] = data_local[-3,:]
    data['eup7'] = data_local[-2,:]
    data['elo7'] = data_local[-1,:]
    
    zdict = {'z_up': [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0],
             'z_low': [0.2, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]}

    return data, zdict

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

def measure_sfrd(z,logm_min=9, logm_max=13, dm=0.01,
         apply_pcorrections=False,use_avg=False,fixed=False,**kwargs):
    """ calculates SFRD from analytic mass function
    plus the Whitaker+14 star-forming sequence
    """

    # generate stellar mass array
    # this represents the CENTER of each `bin`
    logm = np.arange(logm_min,logm_max,dm)+(dm/2.)

    # first, get <SFR(M)>
    # use Prospector SFRs or UV+IR SFRs
    #sfr = sfr_ms(z,logm)
    if use_avg:
        sfr = sfr_fnc_uvir(logm,z).squeeze()
        if apply_pcorrections:
            sfr = sfr_fnc(logm,z).squeeze()
    elif fixed:
        sfr = uvir_fit_fixed_fnc(logm,z).squeeze()
        if apply_pcorrections:
            sfr = prosp_fit_fixed_fnc(logm,z).squeeze()
    else:
        sfr = uvir_fit_fnc(logm,z).squeeze()
        if apply_pcorrections:
            sfr = prosp_fit_fnc(logm,z).squeeze()

    # multiply n(M) by SFR(M) to get SFR / Mpc^-3 / dex
    numdens = mf_phi(mf_parameters(z),logm)
    starforming_function = sfr * numdens

    # integrate over stellar mass to get SFRD
    sfrd = simps(starforming_function,dx=dm)

    return sfrd, starforming_function, logm

def drho_dt(z, logm_min=9, logm_max=12, dm=0.01, dz=0.0001, 
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
    phi_dz = mf_phi(spars,logm) - mf_phi(spars_dz,logm)
    if apply_pcorrections:
        logm += mass_fnc(logm,z)[:,0]

    # now get mass_formed
    mass_formed = simps(phi_dz*10**logm,dx=dm)

    # divide by delta t + mass-loss to get sfrd
    delta_t = (WMAP9.age(z).value - WMAP9.age(z+dz).value)*1e9
    sfrd = (mass_formed/delta_t)

    if massloss_correction:
        sfrd = sfrd / 0.64

    return sfrd

def zfourge_rhostar(z):
    """eqn 5 in Tomczak+14
    """
    a, b = -0.33, 8.75
    return (a*(1+z)+b)

def zfourge_param_rhostar(z, massloss_correction=False, apply_pcorrections=False, **opts):
    """using equation (5) in Tomczak et al. 2014 instead of calculating directly from evolution of mass function
    this removes the "bump" at z~1.5 which is likely erroneous!
    applies for 9 < log(M) < 13
    """

    # turn into rhodot
    nz, dz = len(z), 0.005
    rhodot = np.zeros(nz)
    for i in range(nz):
        upz, downz = z[i]+dz/2., z[i]-dz/2.
        logrho_down = zfourge_rhostar(downz)
        logrho_up = zfourge_rhostar(upz)
        if apply_pcorrections:
            logm_increase = np.log10(drho_dt(z[i],massloss_correction=massloss_correction,apply_pcorrections=True,**opts))-\
                            np.log10(drho_dt(z[i],massloss_correction=massloss_correction,apply_pcorrections=False,**opts))
            logrho_down += logm_increase
            logrho_up += logm_increase
        delta_rho = 10**logrho_down - 10**logrho_up
        delta_t = (WMAP9.age(downz).value - WMAP9.age(upz).value)*1e9
        rhodot[i] = delta_rho/delta_t

    if massloss_correction:
        rhodot /= 0.64

    return np.log10(rhodot)

def generate_evolving_mcut(z_in,logm_at_zstart=9.):

    # generate mass vector, accurate to 0.01 dex
    logm = np.arange(logm_at_zstart,12,0.01)

    # figure out appropriate number density
    numdens = mf_phi(mf_parameters(z_in.max()),logm,cumulative=True)
    nd_target = np.interp(logm_at_zstart,logm,numdens)
    logm_min = []
    for z in z_in:
        numdens = mf_phi(mf_parameters(z),logm,cumulative=True)


        logm_min += [np.interp(nd_target,numdens[::-1],logm[::-1])]

    return logm_min

def mass_vs_sfrd(true_values=False,mloss=0.64):

    # physics
    zplot = [0.75, 1.5, 2.25]
    logm = np.arange(8,11,0.2)
    nlogm = logm.shape[0]

    # plotting options
    fig, axes = plt.subplots(1,3,figsize=(7,3))
    fs = 13
    ylim = (0,2)
    #ylim = (-3,-0.8)
    colors = ['blue','red']

    for i, z in enumerate(zplot):

        ax = axes[i]
        drho, sfrd = np.zeros(nlogm), np.zeros(nlogm)
        true_values = True
        for j, mass in enumerate(logm):
            drho[j] = np.log10(10**csfh.behroozi_rhodot(z,mass,true_mass=true_values)/mloss)
            dat = csfh.behroozi_sfrd(z,mass,true_mass=true_values,true_sfr=true_values)
            sfrd[j] = dat['sfrd_med']
        ax.plot(logm,sfrd/10**drho,lw=2,color=colors[0], label="'True'")
        #ax.plot(logm,np.log10(sfrd),lw=2,color=colors[0], label="'sfrd'")
        #ax.plot(logm,drho,lw=2,color=colors[1], label="'drho'")

        true_values = False
        for j, mass in enumerate(logm):
            drho[j] = np.log10(10**csfh.behroozi_rhodot(z,mass,true_mass=true_values)/mloss)
            dat = csfh.behroozi_sfrd(z,mass,true_mass=true_values,true_sfr=true_values)
            sfrd[j] = dat['sfrd_med']
        ax.plot(logm,sfrd/10**drho,lw=2,color=colors[1],label="'Observed'")

        ax.set_xlabel('minimum logM',fontsize=fs)
        ax.set_title('z='+str(z),fontsize=fs,weight='semibold')
        ax.set_ylim(ylim)
        ax.axhline(1.0,linestyle='--',color='grey')

        if (i > 0):
            for tl in ax.get_yticklabels():tl.set_visible(False)

    axes[0].legend()
    axes[0].set_ylabel(r'$\dot{\rho}_{\mathrm{SFR}}/\dot{\rho}_{\mathrm{mass}}$',fontsize=fs)

    plt.tight_layout()
    plt.show()
    print 1/0

def plot_sfrd_new(logm_min=9.,logm_max=13,dm=0.01,
                  massloss_correction=True,use_avg=True,fixed=False,
                  mcmc=False):
    """ compare SFRD from mass function(z) versus observed SFR
    """

    # I/O
    print 'beware: this uses a HACK which makes any lower mass except 10^9 break!'
    outfolder = '/Users/joel/code/python/prospector_alpha/plots/td_delta/fast_plots/'
    outname = outfolder+'delta_sfrd.png'
    outname2 = outfolder+'abs_sfrd.pdf'

    # plot options
    old_color, new_color, compilation_color = '#6f41cf','#FF3D0D', '0.55'
    compilation_alpha = 0.3
    mass_linestyle, sfr_linestyle = '--', '-'
    ylim = (-0.55,0.55)
    xlim = (0.5, 2.5)
    fs = 13
    popts = {
             'linewidth': 3,
             'alpha': 0.9
            }
    masslim = r'logM$_{\mathrm{FAST}}$ > '+str(int(logm_min))


    mloss_corr = 1.
    if massloss_correction: mloss_corr = 0.64

    # redshift array
    dz = 0.05
    zrange = np.arange(0.75, 2.25+dz, dz)
    opts = {
            'logm_min': logm_min,
            'logm_max': logm_max,
            'dm': dm,
            'massloss_correction': massloss_correction,
            'use_avg': use_avg,
            'fixed': fixed
           }

    # calculate SFRD
    sfrd_uvir, sfrd_prosp = [], [] 
    bratio, bsfrd, bdrho, bsfrd_all, bdrho_all = [], [], [], [], []
    for i, z in enumerate(zrange):

        # calculate SFRD from star formation for both UV+IR SFRs and Prospector
        sfrd, rhosfr, logm = measure_sfrd(z,**opts)
        sfrd_uvir += [sfrd]
        sfrd, rhosfr, logm = measure_sfrd(z,apply_pcorrections=True,**opts)
        sfrd_prosp += [sfrd]

        # calculate expected ratio of mass to SFR
        mdiff, sfdiff = csfh.behroozi_offsets(z)
        bdrho += [float(10**csfh.behroozi_rhodot(z,logm_min-mdiff,true_mass=True)/mloss_corr)]
        dat = csfh.behroozi_sfrd(z,logm_min-mdiff,true_mass=True,true_sfr=True)
        bsfrd += [float(dat['sfrd_med'])]
        bratio += [np.log10(bsfrd[-1]/bdrho[-1])]

        bdrho_all += [float(10**csfh.behroozi_rhodot(z,8.,true_mass=True)/mloss_corr)]
        dat = csfh.behroozi_sfrd(z,8.,true_mass=True,true_sfr=True)
        bsfrd_all += [float(dat['sfrd_med'])]

    # calculate SFRD from change in stellar mass density
    mf_sfrd = zfourge_param_rhostar(zrange, **opts)
    mf_sfrd_prosp = zfourge_param_rhostar(zrange, apply_pcorrections=True, **opts)

    # Plot1: change in SFRD
    labels = [r'FAST dM/dt +'+'\n'+r'SFR$_{\mathrm{UV+IR}}$','Prospector']
    labels = ['old models',r'Prospector-$\alpha$']
    fig, ax = plt.subplots(1,1, figsize=(4,3.7))
    ax.plot(zrange, np.log10(sfrd_uvir)-mf_sfrd, '-', color=old_color, label=labels[0],**popts)
    ax.plot(zrange, np.log10(sfrd_prosp)-mf_sfrd_prosp, '-', color=new_color, label=labels[1],lw=2,**popts)
    bratio_smooth = smooth(bratio,8)
    ax.plot(zrange, bratio_smooth, '--', color=compilation_color, zorder=-1)
    ax.axhline(0.0,linestyle=':',color='0.3')

    # limits, labels, and add compilation
    ax.set_xlabel('redshift',fontsize=fs)
    ax.legend(loc=4, prop={'size':8.5}, scatterpoints=1,fancybox=True)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # special treatment of legend + yaxis ticks & labels
    ax.set_ylabel(r'log($\rho_{\mathrm{SFR}}/\dot{\rho}_{\mathrm{mass}}$)',fontsize=fs)

    plt.tight_layout()
    plt.savefig(outname,dpi=250)
    plt.close()

    # Plot2: SFRD, multiplied by UM fractions
    fig, ax = plt.subplots(1,2, figsize=(8,3.7))
    ylim = (-1.55,-0.8)
    labels = [r'FAST dM/dt +'+'\n'+r'SFR$_{\mathrm{UV+IR}}$','Prospector']
    labels = ['old models',r'Prospector-$\alpha$']

    drho_to_all = smooth(np.array(bdrho_all)/np.array(bdrho),8)
    sfrd_to_all = smooth(np.array(bsfrd_all)/np.array(bsfrd),8)
    #drho_to_all = np.ones_like(drho_to_all)
    #sfrd_to_all = np.ones_like(sfrd_to_all)
    ax[0].plot(zrange, np.log10(sfrd_uvir*sfrd_to_all), '-', color=old_color, label='old SFR',**popts)
    ax[0].plot(zrange, np.log10((10**mf_sfrd)*drho_to_all), '--', color=old_color, label='old dM/dt',**popts)
    ax[1].plot(zrange, np.log10(sfrd_prosp*sfrd_to_all), '-', color=new_color, label='Prospector SFR',lw=2,**popts)
    ax[1].plot(zrange, np.log10((10**mf_sfrd_prosp)*drho_to_all), '--', color=new_color, label='Prospector dM/dt',lw=2,**popts)

    # limits, labels, and add compilation
    for a in ax: 
        a.set_xlabel('redshift',fontsize=fs)
        a.legend(loc=4, prop={'size':8.5}, scatterpoints=1,fancybox=True)
        a.set_xlim(xlim)
        a.set_ylim(ylim)
    ax[0].set_ylabel(r'log(SFRD) [M$_{\odot}$ yr$^{-1}$ Mpc$^{-3}$]',fontsize=fs)
    for tl in ax[1].get_yticklabels():tl.set_visible(False)
    #ax[0].text(0.05,0.95,'log(M$_*$/M$_{\odot}$)>'+)

    plt.tight_layout()
    plt.savefig(outname2,dpi=250)
    plt.close()
    os.system('open '+outname2)

    print 1/0

def plot_sfrd(logm_min=9.,logm_max=12,dm=0.01,
              massloss_correction=True,use_avg=True,fixed=False,
              true_mass=False, true_sfr=False, mcmc=False, ndens=True):
    """ compare SFRD from mass function(z) versus observed SFR
    """

    # I/O
    outfolder = '/Users/joel/code/python/prospector_alpha/plots/td_delta/fast_plots/'
    outname1 = outfolder+'sfrd.png'
    outname2 = outfolder+'sfrd_fixed.png'
    outname3 = outfolder+'behroozi18.png'

    # plot options
    old_color, new_color, compilation_color = '#6f41cf','#FF3D0D', '0.55'
    compilation_alpha = 0.3
    mass_linestyle, sfr_linestyle = '--', '-'
    ylim = (-1.55,-0.45)
    xlim = (0.5, 2.5)
    fs = 9
    popts = {
             'linewidth': 3,
             'alpha': 0.9
            }

    # labels and keys
    masslim = r', logM$_{\mathrm{FAST}}$>'+str(int(logm_min))
    if ndens:
        masslim = r', constant $n$'
    bkey, blabel = 'obs_csfr', 'UM CSFR'
    if true_sfr:
        bkey, blabel = 'true_csfr', 'UM true CSFR'

    # generate z-array + numerical options
    dz = 0.1
    zrange = np.arange(0.75, 2.35, dz)
    opts = {
            'logm_min': logm_min,
            'logm_max': logm_max,
            'dm': dm,
            'massloss_correction': massloss_correction,
            'use_avg': use_avg,
            'fixed': fixed
           }

    # calculate stellar mass density
    # if we're doing evolving mass cut, do it the slow way directly from the MF
    # otherwise, use Tomczak+14 eqn 5 (logm_min=9 ONLY)
    mmin = generate_evolving_mcut(zrange)
    if ndens:
        mf_sfrd, mf_sfrd_prosp = [], []
        for i, z in enumerate(zrange):
            #opts['logm_min'] = mmin[i]
            mf_sfrd += [np.log10(drho_dt(z, **opts))]
            mf_sfrd_prosp += [np.log10(drho_dt(z, apply_pcorrections=True, **opts))]
    else:
        mf_sfrd = zfourge_param_rhostar(zrange, **opts)
        mf_sfrd_prosp = zfourge_param_rhostar(zrange, apply_pcorrections=True, **opts)

    # calculate star formation rate density
    sfrd_uvir, sfrd_prosp, rhosfr_uvir, rhosfr_prosp  = [], [], [], []
    for i, z in enumerate(zrange):

        if ndens:
            opts['logm_min'] = mmin[i]
        sfrd, rhosfr, logm = measure_sfrd(z,**opts)
        sfrd_uvir += [sfrd]
        rhosfr_uvir += [rhosfr]

        sfrd, rhosfr, logm = measure_sfrd(z,apply_pcorrections=True,**opts)
        sfrd_prosp += [sfrd]
        rhosfr_prosp += [rhosfr]

    # Plot1: change in SFRD
    fig, ax = plt.subplots(1,2, figsize=(7, 3.5))
    ax = ax.ravel()
    ax[0].plot(zrange, mf_sfrd, mass_linestyle, color=old_color, label='FAST dM/dt'+masslim,**popts)
    ax[0].plot(zrange, np.log10(sfrd_uvir), sfr_linestyle, color=old_color, label='UV+IR SFR'+masslim, **popts)
    ax[1].plot(zrange, mf_sfrd_prosp, mass_linestyle, color=new_color, label='Prospector dM/dt'+masslim,**popts)
    ax[1].plot(zrange, np.log10(sfrd_prosp), sfr_linestyle, color=new_color, label='Prospector SFR'+masslim, **popts)

    # add Behroozi SFRD
    # first do the partial SFRD above the mass limit
    """
    names = glob.glob('/Users/joel/data/UniverseMachine/data/sfhs/*sm*dat')
    scales = np.unique([name.split('_a')[1].split('.dat')[0] for name in names])
    zred_sfh = 1./scales.astype(float) - 1
    idx_b_sfh = (zred_sfh > 0.75) & (zred_sfh < 2.25)
    zred_sfh = zred_sfh[idx_b_sfh]
    nzred = zred_sfh.shape[0]

    bsfrd = np.zeros(nzred)
    if mcmc:
        bsfrd = np.zeros(shape=(3,zred_sfh.shape[0]))
    for i, z in enumerate(zred_sfh):
        out = csfh.behroozi_weighted_sfrd(z,logm_min,true_mass=true_mass,true_sfr=true_sfr,mcmc=mcmc)
        med, eup, edo, zred = out['sfrd_med'], out['sfrd_errup'], out['sfrd_errdown'], out['sfrd_z']
        idx = np.where(med != 0)[0].min()
        if mcmc:
            bsfrd[:,i] = np.log10([med[idx], eup[idx], edo[idx]])
        else:
            bsfrd[i] = np.log10(med[idx])
    for a in ax[:2]:
        a.plot(zred_sfh, np.atleast_2d(bsfrd)[0,:], '-', color=compilation_color, zorder=-1,linewidth=popts['linewidth']*0.5,alpha=popts['alpha'])
        a.text(1.55,-1.105,r'log(M$_*$/M$_{\odot}$)>'+str(int(logm_min)),fontsize=fs,rotation=-8,weight='semibold',color=str(float(compilation_color)-0.2))
        if mcmc:
            a.fill_between(zred_sfh, bsfrd[2,:], bsfrd[1,:], color=compilation_color, 
                           alpha=compilation_alpha,zorder=-1)

    # now do the total SFRD from Behroozi+18
    param_bsfrd = csfh.load_behroozi_sfrd()
    zred_beh = 1./param_bsfrd['scale'] - 1
    for a in ax[:2]: 
        a.text(1.55,-0.82,r'all galaxies',fontsize=fs,weight='semibold',color=str(float(compilation_color)-0.2))
        a.plot(zred_beh[idx_b_sfh],np.log10(param_bsfrd[bkey])[idx_b_sfh],'-',color=compilation_color,label=blabel,linewidth=popts['linewidth']*0.5,alpha=popts['alpha'])
        a.fill_between(zred_beh[idx_b_sfh], 
                       np.log10(param_bsfrd[bkey]-param_bsfrd[bkey+'_edo'])[idx_b_sfh], 
                       np.log10(param_bsfrd[bkey]+param_bsfrd[bkey+'_eup'])[idx_b_sfh], 
                       color=compilation_color, alpha=compilation_alpha,zorder=-1)
    """
    # limits, labels, and add compilation
    for a in ax[:2]: 
        a.set_xlabel('redshift')
        a.legend(loc=2, prop={'size':8.5}, scatterpoints=1,fancybox=True)
        a.set_xlim(xlim)
        a.set_ylim(ylim)

    # special treatment of legend + yaxis ticks & labels
    ax[0].set_ylabel(r'log(SFRD) [M$_{\odot}$ yr$^{-1}$ Mpc$^{-3}$]')
    for tl in ax[1].get_yticklabels():tl.set_visible(False)

    ax[0].set_title('3D-HST catalog parameters',fontsize=fs*1.4,weight='semibold')
    ax[1].set_title('Prospector parameters',fontsize=fs*1.4,weight='semibold')

    """
    # now plot mass function
    a = ax[2]
    linestyle = [':','-.']
    zred = [1,2]
    labels = ['z={0}'.format(z) for z in zred]
    for i, z in enumerate([1,2]):
        numdens = mf_phi(mf_parameters(z),logm)
        a.plot(logm,np.log10(numdens),color='k',lw=2,label=labels[i],linestyle=linestyle[i])
        a.set_xlim(9,11.5)
    a.set_title(r'FAST mass function',fontsize=fs*1.4,weight='semibold')
    a.legend(loc=3, prop={'size':8.5}, scatterpoints=1,fancybox=True)
    a.set_ylim(-5,-1.9)
    a.set_xlabel(r'log(M/M$_{\odot}$)')
    a.set_ylabel(r'log($\phi$/Mpc$^{-3}$)')

    # and average SFR
    logm = np.arange(logm_min,logm_max,dm)+(dm/2.)
    a = ax[3]
    labels = ['(z={0})'.format(z) for z in zred]
    for i, z in enumerate([1,2]):
        sfr_uvir = sfr_fnc_uvir(logm,z).squeeze()
        sfr_prosp = sfr_fnc(logm,z).squeeze()

        a.plot(logm,np.log10(sfr_uvir),color=old_color,lw=2,label='Prospector '+labels[i],linestyle=linestyle[i])
        a.plot(logm,np.log10(sfr_prosp),color=new_color,lw=2,label=r'SFR$_{\mathrm{UV+IR}}$ '+labels[i],linestyle=linestyle[i])

        a.set_xlim(9,11.5)
        a.set_ylim(0,2.5)
    a.set_title(r'<SFR>(M$_\mathrm{FAST}}$)',fontsize=fs*1.4,weight='semibold')
    a.legend(loc=2, prop={'size':8.5}, scatterpoints=1,fancybox=True)
    a.set_xlabel(r'log(M/M$_{\odot}$)')
    a.set_ylabel(r'log(SFR/(M$_{\odot}$/yr))')
    """

    plt.tight_layout(w_pad=0.03)
    plt.savefig(outname1,dpi=250)
    plt.close()
    #os.system('open '+outname1)

    # Plot2: rhosfr
    fig, ax = plt.subplots(2, 2, figsize = (6.5,6.5))
    fig.subplots_adjust(hspace=0.0,wspace=0.0)
    ax = np.ravel(ax)
    fs = 12
    ylim = (0, 7)

    # redshift bins, dm
    zbins = np.linspace(0.5,2.5,5)
    nbins = len(zbins)-1
    zlabels = ['$'+"{0:.1f}".format(zbins[i])+'<z<'+"{0:.1f}".format(zbins[i+1])+'$' for i in range(nbins)]
    delm = logm[1]-logm[0]

    # loop
    for i in range(nbins):

        idx = np.abs(zrange-(zbins[i]+zbins[i+1])/2.).argmin()

        ax[i].plot(logm, rhosfr_uvir[idx]/delm, '-', color=old_color, **popts)
        ax[i].plot(logm, rhosfr_prosp[idx]/delm, '-', color=new_color, **popts)

        # labels
        ax[i].set_ylim(ylim)
        ax[i].text(0.97,0.93,zlabels[i],ha='right',fontsize=fs,transform=ax[i].transAxes)
        if i > 1:
            ax[i].set_xlabel('log(M/M$_{\odot}$) [FAST]',fontsize=fs)
            for tl in ax[i].get_xticklabels():tl.set_fontsize(fs)
        else:
            for tl in ax[i].get_xticklabels():tl.set_visible(False)
        if (i % 2 == 0):
            ax[i].set_ylabel(r'$\rho_{\mathrm{SFR}}$ [M$_{\odot}$ yr$^{-1}$ Mpc$^{-3}$ dex$^{-1}$]',fontsize=fs)
            for tl in ax[i].get_yticklabels():tl.set_fontsize(fs)
        else:
            for tl in ax[i].get_yticklabels():tl.set_visible(False)
    
    plt.savefig(outname2,dpi=200)
    plt.close()

    # Plot3: deltaSFR
    dm_bz17, dsfr_bz17 = csfh.behroozi_offsets(zrange)
    fig, ax = plt.subplots(1, 3, figsize = (9,3))
    lopts = {'linestyle':'-','lw':4,'alpha':1}
    bcolor, pcolor = '#66c2a5','#fc8d62'#, 'blue'
    ylims = 0.5

    ax[0].plot(zrange,-dm_bz17,color=bcolor,label='Behroozi+18',**lopts)
    ax[1].plot(zrange,dsfr_bz17,color=bcolor,**lopts)
    ax[2].plot(zrange,dsfr_bz17-dm_bz17,color=bcolor,**lopts)

    dsfr = np.log10(np.array(sfrd_uvir)/np.array(sfrd_prosp))
    dm = np.array(mf_sfrd) - np.array(mf_sfrd_prosp)
    ax[0].plot(zrange,-dm,color=pcolor,label='Prospector',**lopts)
    ax[1].plot(zrange,dsfr,color=pcolor,**lopts)
    ax[2].plot(zrange,dsfr-dm,color=pcolor,**lopts)

    for a in ax:
        a.set_ylim(-ylims,ylims)
        a.axhline(0.0,linestyle='--',color='k')
        a.set_xlabel('redshift')
    ax[0].set_ylabel('log(M$_{\mathrm{B18,Prosp}}$/M$_{\mathrm{obs}}$)')
    ax[1].set_ylabel('log(SFR$_{\mathrm{obs}}$/SFR$_{\mathrm{B18,Prosp}}$)')
    ax[2].set_ylabel('log(SFRD$_{\mathrm{obs}}$/SFRD$_{\mathrm{B18,Prosp}}$)')
    ax[0].legend(fancybox=True)


    plt.tight_layout()
    plt.savefig(outname3,dpi=200)
    plt.close()


    print 1/0

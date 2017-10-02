import numpy as np
import prosp_dutils, prospector_io, copy
import os
import td_massive_params as pfile
from astropy import constants
from astropy.cosmology import WMAP9
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os

plt.ioff()
pc = 3.085677581467192e18  # in cm
dfactor_10pc = 4*np.pi*(10*pc)**2
to_ergs = 3631e-23

dale_helou_txt = '/Users/joel/code/python/prospector_alpha/data/Wuyts08_conversion.txt'
with open(dale_helou_txt, 'r') as f: 
    for i in range(8): f.readline()
    hdr = f.readline().split()[1:]
conversion = np.genfromtxt(dale_helou_txt, comments = '#', dtype = np.dtype([(n, np.float) for n in hdr]))

outdir = '/Users/joel/code/python/prospector_alpha/tests/irsed/'

def dl07_to_dh02():

    '''
    This calculates UV + IR SFRs based on the Dale & Helou IR templates & MIPS flux
    '''

    ## build model components
    run_params = pfile.run_params
    sps = pfile.load_sps(**run_params)
    model = pfile.load_model(**run_params)
    obs = pfile.load_obs(**run_params)
    mips_idx = [f.name for f in obs['filters']].index('mips_24um_aegis')
    model.params['nebemlineinspec'] = True # yeah this is necessary right now

    ## generate default model
    theta = copy.copy(model.initial_theta)
    pnames = model.theta_labels()
    theta[pnames.index('duste_gamma')] = 0.01
    theta[pnames.index('duste_umin')] = 1.0
    qpah_grid = np.linspace(0,10,101)

    ### generate Prospector L_IR
    # this should remain constant with redshift
    # input angstroms, Lsun/Hz. output in erg/s, convert to Lsun
    model.params['zred'] = 0.0
    spec,mags,sm = model.mean_model(theta, obs, sps=sps) # everything in maggies
    spec *= dfactor_10pc / constants.L_sun.cgs.value * to_ergs
    lir = prosp_dutils.return_lir(sps.wavelengths,spec)/constants.L_sun.cgs.value

    ## range of redshifts
    zrange = np.linspace(0.1,3,50)

    mips_to_lir_dh, mips_to_lir_prosp, qpah_min = [], [], []
    for i, z in enumerate(zrange):
        print i
        #### generate observed MIPS flux
        model.params['zred'] = z
        obs_mips_mjy, lir_mips = [], []
        for qpah in qpah_grid:
            theta[pnames.index('duste_qpah')] = qpah
            spec,mags,sm = model.mean_model(theta, obs, sps=sps) # everything in maggies
            obs_mips_mjy.append(mags[mips_idx] * 3631 * 1e3)
 
            ### goes in in milliJy, comes out in Lsun
            lir_mips.append(mips_to_lir(obs_mips_mjy[-1],z))

        minidx = np.abs(lir_mips-lir).argmin()

        ### output in milliJy / Lsun
        mips_to_lir_dh.append(lir_mips[minidx]/obs_mips_mjy[minidx])
        mips_to_lir_prosp.append(lir/obs_mips_mjy[minidx])
        qpah_min.append(qpah_grid[minidx])

    ### quick plotting routine
    fig, ax = plt.subplots(1,3, figsize = (15,5))
    xplot = zrange
    yplot = [np.array(mips_to_lir_dh),np.array(mips_to_lir_prosp)]
    color = ['blue','green']
    label = ['Dale & Helou 02', 'DL07 (Prospector)']

    for i, yp in enumerate(yplot): ax[0].plot(xplot, np.log10(yp), 'o',alpha=0.6,linestyle='-',color=color[i],lw=2,label=label[i])
    ax[1].plot(xplot, np.log10(yplot[0])-np.log10(yplot[1]), 'o',alpha=0.6,linestyle='-',color=color[i],lw=2,label=label[i])
    ax[2].plot(xplot, qpah_min, 'o', alpha=0.6, linestyle='-', color='0.3',lw=2)

    for a in ax: a.set_xlabel('redshift')
    ax[0].set_ylabel('log(LIR / MIPS flux)')
    ax[1].set_ylabel(r'$\Delta$ (DH02 - DL07)')
    ax[2].set_ylabel(r'Q$_{\mathrm{PAH}}$(min)')

    ax[0].legend(loc=0, fontsize=12, prop={'size':12}, frameon=True,numpoints=1)
    ax[1].axhline(0, linestyle='--', color='0.5',lw=2,zorder=-1)
    ax[2].set_ylim(qpah_grid[0]-0.5,qpah_grid[-1]+0.5)

    plt.tight_layout()
    plt.savefig(outdir+'qpah_with_zred.png', dpi=150)
    plt.close()

    np.savetxt(outdir+'qpah.txt', np.transpose([zrange,qpah_min]), fmt='%1.4f', header='z qpah')

def mips_to_lir(mips_flux,z):

    '''
    input flux must be in mJy
    output is in Lsun
    L_IR [Lsun] = fac_<band>(redshift) * flux [milliJy]
    '''
    
    # if we're at higher redshift, interpolate
    # it decrease error due to redshift relative to rest-frame template (good)
    # but adds nonlinear error due to distances (bad)
    # else, scale the nearest conversion factor by the 
    # ratio of luminosity distances, since nonlinear error due to distances will dominate
    if z > 0.1:
        intfnc = interp1d(conversion['Redshift'],conversion['fac_MIPS24um'], bounds_error = True, fill_value = 0)
        fac = intfnc(z)
    else:
        near_idx = np.abs(conversion['Redshift']-z).argmin()
        lumdist_ratio = (WMAP9.luminosity_distance(z).value / WMAP9.luminosity_distance(conversion['Redshift'][near_idx]).value)**2
        zfac_ratio = (1.+conversion['Redshift'][near_idx]) / (1.+z)
        fac = conversion['fac_MIPS24um'][near_idx]*lumdist_ratio*zfac_ratio

    return fac*mips_flux
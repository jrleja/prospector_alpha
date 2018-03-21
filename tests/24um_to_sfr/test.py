import test_params as pfile
import copy, prosp_dutils
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import WMAP9

# initialize model
try:
    mod2
except NameError:
    #mod = pfile.load_model(**pfile.run_params)
    mod2 = pfile.load_model(zred=6,**pfile.run_params)
    sps = pfile.load_sps(**pfile.run_params)
    obs = pfile.load_obs(**pfile.run_params)
    #_, _, _ = mod.mean_model(mod.initial_theta, obs, sps=sps)
    _, _, _ = mod2.mean_model(mod2.initial_theta, obs, sps=sps)

def ascii_write(mags,v1,v2,v1_name,v2_name,outname):

    names = [v1_name]+[v2_name]+[f.name for f in obs['filters']]
    format = {name: '{:1.2f}' for name in names}
    
    nv1, nv2 = v1.shape[0], v2.shape[0]
    odat = np.zeros(shape=(nv1*nv2,mags.shape[2]+2))
    for i in range(nv1):
        for j in range(nv2):
            idx = i*nv2+j
            odat[idx,2:] = np.log10(mags[i,j,:])
            odat[idx,1] = v2[j]
            odat[idx,0] = v1[i]

    np.savetxt(outname, odat, fmt='%1.4f', header=' '.join(names))

def starforming_galaxy():

    thetas = copy.copy(mod2.initial_theta)
    thetas[mod2.theta_index['fagn']] = 0.0
    thetas[mod2.theta_index['dust2']] = prosp_dutils.av_to_dust2(1)
    mod2.params['nebemlineinspec'] = np.array([True])

    zred_grid = np.array([1,2,3])
    nwave, nred = sps.wavelengths.shape[0], len(zred_grid)
    spec = np.zeros(shape=(nwave,nred))
    flux_conversion = 3631*1e-23

    colors = ['#e31a1c', '#ff7f00','#33a02c','#1f78b4','#6a3d9a']
    for i in range(nred):
        mod2.params['zred'] = zred_grid[i]
        spec[:,i],_,_ = mod2.mean_model(thetas, obs, sps=sps)
        idx = sps.wavelengths > 1000
        plt.plot(np.log10(sps.wavelengths[idx]),np.log10(spec[idx,i]*flux_conversion),label='z={0}'.format(zred_grid[i]),color=colors[i])

    plt.xlabel('log($\lambda_{\mathrm{rest}}$) [$\AA$]')
    plt.ylabel(r'f$_{\nu}$ [cgs]')
    plt.legend()
    plt.tight_layout()
    plt.savefig('starforming_sed.png',dpi=200)
    plt.close()

    out = np.concatenate((sps.wavelengths[:,None],np.log10(spec)),axis=1)
    np.savetxt('starforming_sed.dat', out, fmt='%1.4f', header='restframe wavelength, log(z=1 spectrum), log(z=2 spectrum), log(z=3 spectrum)')

def sfr_flux_relationship():

    # initialize
    thetas = copy.copy(mod.initial_theta)
    thetas[mod.theta_index['fagn']] = 0.0
    thetas[mod.theta_index['dust2']] = prosp_dutils.av_to_dust2(20)

    # grids and constants
    ngrid_sfr, ngrid_zred = 31, 30
    sfr_grid = np.linspace(1,1000,ngrid_sfr)
    zred_grid = np.linspace(0.1,3,ngrid_zred)

    mags = np.zeros(shape=(ngrid_zred,ngrid_sfr,len(obs['filters'])))
    wlam = np.array([f.wave_effective for f in obs['filters']])
    flux_conversion = 3631*1e-23
    tuniv = WMAP9.age(mod.params['zred']).value * 1e9

    # loop
    for j in range(ngrid_zred):
        # set zred 
        mod.params['zred'] = zred_grid[j]
        for i in range(ngrid_sfr):
            # set SFR value
            thetas[0] = np.log10(sfr_grid[i]*tuniv)
            _, mags[j,i,:], _ = mod.mean_model(thetas, obs, sps=sps)

    # convert to cgs
    mags *= flux_conversion

    # plot values
    colors = ['#e31a1c', '#ff7f00','#33a02c','#1f78b4','#6a3d9a']
    labels = ['A$_V$=1','A$_V$=2','A$_V$=3']
    labels = ['10$\mu$m','15$\mu$m','18$\mu$m','21$\mu$m','24$\mu$m']
    fig, ax = plt.subplots(1,2, figsize=(7, 3.5))

    zred_idx = np.abs(zred_grid-2).argmin()
    sfr_idx = np.abs(sfr_grid-100).argmin()
    for j in range(5): 
        ax[0].plot(np.log10(sfr_grid),np.log10(mags[zred_idx,:,j]),'o-',color=colors[j],label=labels[j],lw=1.4,ms=2.4)
        ax[1].plot(zred_grid,np.log10(mags[:,sfr_idx,j]),'o-',color=colors[j],label=labels[j],lw=1.4,ms=2.4)
    ax[0].legend()
    ax[1].legend()
    ax[0].set_title('z=2')
    ax[1].set_title('SFR=100 M$_{\odot}$/yr')

    for a in ax: a.set_ylabel(r'log(f$_{\nu}$) [cgs]')
    ax[0].set_xlabel(r'log(SFR) [M$_{\odot}$/yr]')
    ax[1].set_xlabel(r'redshift')

    plt.tight_layout()
    plt.savefig('flux_sfr_redshift.png',dpi=200)
    plt.close()

    ascii_write(mags,zred_grid,sfr_grid,'redshift','SFR','flux_sfr_redshift.dat')

    # loop over fagn
    thetas[mod.theta_index['agn_tau']] = 10
    fagn_grid = np.array([0.0,0.05,0.5])
    mags = np.zeros(shape=(ngrid_zred,len(fagn_grid),len(obs['filters'])))

    # loop
    for j in range(len(fagn_grid)):
        # set fagn
        thetas[mod.theta_index['fagn']] = fagn_grid[j]
        for i in range(ngrid_zred):
            # set zred
            mod.params['zred'] = zred_grid[i]
            _, mags[i,j,:], _ = mod.mean_model(thetas, obs, sps=sps)

    # convert to cgs
    mags *= flux_conversion

    # plot values
    fig, ax = plt.subplots(1,1, figsize=(4, 4))

    for j in range(5): ax.plot(zred_grid,np.log10(mags[:,0,j]/mags[:,-1,j]),'o-',color=colors[j],label=labels[j],lw=1.4,ms=2.4)
    ax.set_ylim(-1,0)
    ax.legend()
    ax.set_ylabel(r'log(f$_{\nu}$/f$_{\nu,AGN}$)')
    ax.set_xlabel(r'redshift')

    plt.tight_layout()
    plt.savefig('agn.png',dpi=200)
    plt.close()

    ascii_write(mags,zred_grid,fagn_grid,'redshift','fagn','flux_redshift_agn.dat')

    # loop over PAH
    thetas[mod.theta_index['fagn']] = 0.0
    qpah_grid = np.array([0.0,2,7])
    mags = np.zeros(shape=(ngrid_zred,len(qpah_grid),len(obs['filters'])))

    # loop
    for j in range(len(qpah_grid)):
        # set qph
        mod.params['duste_qpah'] = qpah_grid[j]
        for i in range(ngrid_zred):
            # set zred
            mod.params['zred'] = zred_grid[i]            
            _, mags[i,j,:], _ = mod.mean_model(thetas, obs, sps=sps)

    # convert to cgs
    mags *= flux_conversion

    # plot values
    fig, ax = plt.subplots(1,1, figsize=(4, 4))

    for j in range(5): ax.plot(zred_grid,np.log10(mags[:,0,j]/mags[:,-1,j]),'o-',color=colors[j],label=labels[j],lw=1.4,ms=2.4)
    ax.set_ylim(-1.5,0.5)
    ax.legend()
    ax.set_ylabel(r'log(f$_{\nu,lowPAH}$/f$_{\nu,highPAH}$)')
    ax.set_xlabel(r'redshift')
    ax.axhline(0, linestyle='--', color='k',lw=1,zorder=-1)

    plt.tight_layout()
    plt.savefig('qpah.png',dpi=200)
    plt.close()

    ascii_write(mags,zred_grid,qpah_grid,'redshift','qpah','flux_redshift_qpah.dat')
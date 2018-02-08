import vis_params as pfile
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from prosp_dutils import smooth_spectrum


def do_all():

    model = pfile.load_model(**pfile.run_params)
    sps = pfile.load_sps(**pfile.run_params)
    obs = pfile.load_obs(**pfile.run_params)

    analyze(model,sps,obs)

def analyze(model,sps,obs):
    """this needs to:
    (a) measure the depth of metallicity-sensitive absorption line
        Fe 4668? this is 
    (b) measure the depth of age-sensitive absorption line
        H-gamma (F), from Worthey+94
    (b) measure IR colors
    """

    # load Lick indices from 
    # http://astro.wsu.edu/worthey/html/system.html
    # resolution is given as FWHM = 8-11 Angstrom
    # R = (c / delta(v)) ~ (lambda/FWHM)  --> delta_v = c * (FWHM/lambda) ~ 650
    # alternatively, fwhm   = sigma*2.35482/ckms/dlstep, where dlstep = 0.00023586852983115136  --> delta_v = 300 (this seems more reasonable)
    # Fe5270 from Conroy & Gunn 2010b (Fig 9, http://iopscience.iop.org/article/10.1088/0004-637X/712/2/833/pdf)
    lick = ascii.read('/Users/joel/code/python/prospector_alpha/data/lick_indices.dat')
    sigma = 100
    
    # pull out spectral parameters and IDs
    lam = sps.wavelengths
    names = ['Fe5270','Hgamma_F','Mg_b']
    depths = {}
    for name in lick['name']:
        idx = lick['name'] == name
        depths[name] = {
                        'main':(lick['band_lower'][idx][0], lick['band_upper'][idx][0]),
                        'blue':(lick['blue_lower'][idx][0], lick['blue_upper'][idx][0]),
                        'red':(lick['red_lower'][idx][0], lick['red_upper'][idx][0]),
                        'ew': []
                        }

    logzsol = np.linspace(-1.98,0.19,50)

    # begin loop
    theta = model.initial_theta.copy()
    ir_color, feh5270, mgb, hgamma = [], [], [], []
    for i, logz in enumerate(logzsol):

        # smooth to appropriate resolution, convert to flambda
        theta[model.theta_index['logzsol']] = logz
        spec, mags,sm = model.mean_model(theta, obs, sps=sps)
        flux = smooth_spectrum(lam,spec,sigma,minlam=3500,maxlam=6000) * (3e18 / lam**2)

        # measure
        for key in depths.keys():

            # identify average flux, average wavelength
            low_cont = (lam > depths[key]['blue'][0]) & (lam < depths[key]['blue'][1])
            high_cont = (lam > depths[key]['red'][0]) & (lam < depths[key]['red'][1])
            abs_idx = (lam > depths[key]['main'][0]) & (lam < depths[key]['main'][1])

            low_flux = np.mean(flux[low_cont])
            high_flux = np.mean(flux[high_cont])

            low_lam = np.mean(depths[key]['blue'])
            high_lam = np.mean(depths[key]['red'])

            # draw straight line between midpoints
            m = (high_flux-low_flux)/(high_lam-low_lam)
            b = high_flux - m*high_lam

            # integrate the flux and the straight line, take the difference
            yline = m*lam[abs_idx]+b
            absflux = np.trapz(yline,lam[abs_idx]) - np.trapz(flux[abs_idx], lam[abs_idx])

            lamcont = np.mean(lam[abs_idx])
            abseqw = absflux/(m*lamcont+b)

            if False:
                fig, ax = plt.subplots(1,1, figsize=(5, 5))
                ax.plot(lam,flux,color=observed_flux_color, drawstyle='steps-mid',alpha=alpha)
                ax.plot(lam[abs_idx],yline,color=continuum_line_color,alpha=alpha)
                ax.plot(lam[abs_idx],flux[abs_idx],color=line_index_color,alpha=alpha)
                ax.plot(lam[low_cont],flux[low_cont],color=continuum_index_color,alpha=alpha)
                ax.plot(lam[high_cont],flux[high_cont],color=continuum_index_color,alpha=alpha)

                ax.text(0.96,eqw_yloc, 'EQW='+"{:.2f}".format(abseqw), transform = ax.transAxes,horizontalalignment='right',color=line_index_color)
                ax.set_xlim(np.min(depths[key]['blue'])-50,np.max(depths[key]['red'])+50)

                plt_lam_idx = (lam > depths[key]['blue'][0]) & (lam < depths[key]['red'][1])
                minplot = np.min(flux[plt_lam_idx])*0.9
                maxplot = np.max(flux[plt_lam_idx])*1.1
                ax.set_ylim(minplot,maxplot)
                plt.show()
                print 1/0

            depths[key]['ew'] += [abseqw]


    fig1, ax1 = plt.subplots(5,5, figsize=(12, 12))
    fig2, ax2 = plt.subplots(5,5, figsize=(12, 12))
    ax1 = ax1.ravel()
    ax2 = ax2.ravel()
    for i,key in enumerate(depths.keys()):
        ax1.plot(logzsol,depths[key]['ew'],'o',ms=3)
        ax2.plot(age,depths[key]['ew'],'o',ms=3)
        ax1.set_xlabel(key)
        ax2.set_xlabel(key)

    fig1.tight_layout()
    fig2.tight_layout()
    fig1.savefig('logzsol.png',dpi=150)
    fig2.savefig('age.png',dpi=150)
    plt.close()
    print 1/0
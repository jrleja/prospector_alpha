import numpy as np
from astropy.io import fits
import os, prosp_dutils, hickle
from prospector_io import load_prospector_data
from astropy import constants
from astropy import units as u
import matplotlib.pyplot as plt

plotopts = {
         'fmt':'o',
         'ecolor':'k',
         'capthick':0.4,
         'elinewidth':0.4,
         'alpha':0.5,
         'ms':0.0,
         'zorder':-2
        } 

def make_master_catalog():
    """makes master catalog for selecting objects
    CATALOG 1 (FOR ME): Objname, RA, DEC, PA, box dimensions, expected Br gamma luminosity, recessional velocity, 
    nearby OH lines + strengths, sSFR, metallicity, (nearby bright stars)?, (non-sidereal tracking?)
    CATALOG 2 (CSV): Objname, RA, DEC, equinox (J2000), non-sidereal tracking rate
        -- this is 2' per 10 minutes in direction perpendicular to PA, in [d(RA), d(DEC)]
    
    order by brightness, for galaxies with no strong OH lines
    """


def oh_lines(lam,v,dv=500):
    """ data from Oliva+15 http://adsabs.harvard.edu/abs/2015A%26A...581A..47O
    return list of lines + strengths for a given lambda (Angstrom), velocity, and velocity width (km/s)
    strengths are normalized such that strongest line is 10^4
    """

    dloc = '/Users/joel/data/triplespec/oh_lines/'

    d1 = np.loadtxt(dloc+'table1.dat', comments = '#', delimiter=' ', 
                    dtype = {'formats':('f16','S40','f16','S40','f16'),'names':('wav1','nam1','wav2','name2','strength')})
    d2 = np.loadtxt(dloc+'table2.dat', comments = '#', dtype = {'formats':('f16','S40','f16'),'names':('wav','nam','strength')})

    dlam = (dv/3e5)*lam

    w1 = np.abs(d1['wav1']-lam) < dlam
    w2 = np.abs(d2['wav']-lam) < dlam

    lamlist = d1['wav1'][w1].tolist() + d2['wav'][w2].tolist()
    strengthlist = d1['strength'][w1].tolist() + d2['strength'][w2].tolist()

    return lamlist, strengthlist

def make_plots(dat,outfolder=None,errs=True):

    # set it up
    fig, ax = plt.subplots(1,3, figsize=(16, 5))
    fs = 18 # font size
    ms = [8,7] # marker size
    alpha = [0.9,0.65]
    zorder = [1,-1]
    sfr_min = 0.01
    colors = ['#FF420E', '#545454','#31A9B8']
    plotlines = ['Br gamma 21657','H alpha 6563']
    plotlabels = [r'Br-$\gamma$ 21657',r'H$\alpha$ 6563']

    # make plots
    for i, line in enumerate(plotlines):
        xplot_low, xplot, xplot_high = np.percentile(dat[line]['ew'],[16,50,84],axis=1)  
        yplot_low, yplot, yplot_high = np.percentile(dat[line]['flux'],[16,50,84],axis=1)
        xplot_err = prosp_dutils.asym_errors(xplot,xplot_high,xplot_low)
        yplot_err = prosp_dutils.asym_errors(yplot,yplot_high,yplot_low)
        if errs:
            ax[0].errorbar(xplot, yplot, xerr=xplot_err, yerr=yplot_err,
                           **plotopts)
        ax[0].plot(xplot, yplot, 'o', linestyle=' ', 
                label=plotlabels[i], alpha=alpha[i], markeredgecolor='k',
                color=colors[i],ms=ms[i],zorder=zorder[i])
    
    # plot geometry
    ax[0].set_xlabel(r'equivalent width [\AA]',fontsize=fs)
    ax[0].set_ylabel(r'flux [erg/s/cm$^{2}$]',fontsize=fs)
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].xaxis.set_tick_params(labelsize=fs)
    ax[0].yaxis.set_tick_params(labelsize=fs)
    ax[0].legend(prop={'size':fs*0.8},frameon=False)

    # make plots
    xplot_low, xplot, xplot_high = np.percentile(dat['sfr'],[16,50,84],axis=1)
    idx = xplot > sfr_min  
    xplot_low, xplot, xplot_high = xplot_low[idx], xplot[idx], xplot_high[idx]
    xplot_err = prosp_dutils.asym_errors(xplot,xplot_high,xplot_low)
    for i, line in enumerate(plotlines):
        yplot_low, yplot, yplot_high = np.percentile(np.log10(dat[line]['lum'][idx]),[16,50,84],axis=1)
        yplot_err = prosp_dutils.asym_errors(yplot,yplot_high,yplot_low)
        if errs:
            ax[1].errorbar(xplot, yplot, xerr=xplot_err, yerr=yplot_err,
                           **plotopts)
        ax[1].plot(xplot, yplot, 'o', linestyle=' ', 
                label=plotlabels[i], alpha=alpha[i], markeredgecolor='k',
                color=colors[i],ms=ms[i])
    
    # plot geometry
    ax[1].set_xlabel(r'SFR [M$_{\odot}$/yr]',fontsize=fs)
    ax[1].set_xscale('log')
    ax[1].set_ylabel(r'log(L/L$_{\odot}$) [dust attenuated]',fontsize=fs)
    ax[1].xaxis.set_tick_params(labelsize=fs)
    ax[1].yaxis.set_tick_params(labelsize=fs)
    ax[1].legend(prop={'size':fs*0.8},frameon=False)

    # make plots
    yplot_low, yplot, yplot_high = np.percentile(np.log10(np.array(dat[plotlines[0]]['flux']) / np.array(dat[plotlines[1]]['flux']))[idx],[16,50,84],axis=1)
    yplot_err = prosp_dutils.asym_errors(yplot,yplot_high,yplot_low)
    if errs:
        ax[2].errorbar(xplot, yplot, xerr=xplot_err, yerr=yplot_err,
                       **plotopts)
    ax[2].plot(xplot, yplot, 'o', linestyle=' ', 
            label=line, alpha=alpha[i], markeredgecolor='k',
            color=colors[1],ms=ms[i],zorder=zorder[i])
    
    # plot geometry
    ax[2].set_xlabel(r'SFR [M$_{\odot}$/yr]',fontsize=fs)
    ax[2].set_xscale('log')
    ax[2].set_ylabel(r'log(F$_{\mathrm{Br-}\gamma}$/F$_{\mathrm{H}\alpha}$)',fontsize=fs)
    ax[2].xaxis.set_tick_params(labelsize=fs)
    ax[2].yaxis.set_tick_params(labelsize=fs)
    ax[2].set_ylim(-2.2,-1.15)

    ax[2].axhline(yplot.min(), linestyle='--', color='k',lw=2,zorder=-3)
    ax[2].text(np.median(xplot)*1.5,yplot.min()+0.015,'atomic ratio',fontsize=14,weight='semibold')

    ax[2].arrow(sfr_min*2.5, -1.6, 0.0, 0.1,
                head_width=0.01, width=0.002,color='#FF3D0D')
    ax[2].text(sfr_min*2.5,-1.65,'dust',color='#FF3D0D',ha='center',fontsize=14)

    plt.tight_layout()
    fig.savefig(outfolder+'brgamma_halpha.png',dpi=150)
    plt.close()

def do_all(runname='brownseds_agn', regenerate=False,errs=True):
    # I/O folder
    outfolder = os.getenv('APPS')+'/prospector_alpha/plots/'+runname+'/bracket_gamma/'
    if not os.path.isdir(outfolder):
        os.makedirs(outfolder)
    dat = get_galaxy_properties(regenerate=regenerate,runname=runname,outfolder=outfolder)
    make_plots(dat,outfolder=outfolder,errs=errs)

def get_galaxy_properties(runname='brownseds_agn', regenerate=False, outfolder=None):
    """Loads output, runs post-processing.
    Measure luminosity, fluxes, EW for (Br_gamma, Paschen_alpha, H_alpha)
    Measure mass, SFR, sSFR
    currently best-fit only!
    """

    # skip processing if we don't need new data
    filename = outfolder+'bracket_gamma.hickle'
    if os.path.isfile(filename) and regenerate == False:
        with open(filename, "r") as f:
            outdict = hickle.load(f)
        return outdict

    # build output dictionary
    outlines = ['Br gamma 21657','Pa alpha 18752','H alpha 6563']
    outpars = ['stellar_mass', 'ssfr', 'sfr','logzsol']
    outdict = {'names':[]}
    ngal, nsamp = 129, 300
    for par in outpars: outdict[par] = np.zeros(shape=(ngal,nsamp))
    for line in outlines:
        outdict[line] = {}
        outdict[line]['lum'] = np.zeros(shape=(ngal,nsamp))
        outdict[line]['flux'] = np.zeros(shape=(ngal,nsamp))
        outdict[line]['ew'] = np.zeros(shape=(ngal,nsamp))

    # interface with FSPS lines
    loc = os.getenv('SPS_HOME')+'/data/emlines_info.dat'
    dat = np.loadtxt(loc, delimiter=',', dtype = {'names':('lam','name'),'formats':('f16','S40')})
    line_idx = [dat['name'].tolist().index(line) for line in outlines]

    # iterate over sample
    basenames, _, _ = prosp_dutils.generate_basenames(runname)
    for i, name in enumerate(basenames):
        sample_results, powell_results, model, eout = load_prospector_data(name,hdf5=True,load_extra_output=True)
        outdict['names'].append(str(sample_results['run_params']['objname']))

        # instatiate sps, make fake obs dict
        if i == 0:
            import brownseds_agn_params
            sps = brownseds_agn_params.load_sps(**sample_results['run_params'])
            fake_obs = {'maggies': None, 
                        'phot_mask': None,
                        'wavelength': None, 
                        'filters': []}

        # generate model
        for k in xrange(nsamp):

            spec,mags,sm = model.mean_model(eout['quantiles']['sample_chain'][k,:] , fake_obs, sps=sps)
            mformed = float(10**eout['quantiles']['sample_chain'][k,0])
            
            for j, line in enumerate(outlines):

                # line luminosity (Lsun / [solar mass formed])
                llum = sps.get_nebline_luminosity[line_idx[j]] * mformed
                lumdist = model.params['lumdist'][0] * u.Mpc.to(u.cm)

                # flux [erg / s / cm^2]
                lflux = llum * constants.L_sun.cgs.value / (4*np.pi*lumdist**2)

                # continuum [erg / s / cm^2 / AA]
                spec_in_units = spec * 3631 * 1e-23 * (3e18/sps.wavelengths**2)
                if line == 'H alpha 6563':
                    idx = (sps.wavelengths > 6400) & (sps.wavelengths < 6700)
                    continuum = np.median(spec_in_units[idx])
                else:
                    continuum = np.interp(dat['lam'][line_idx[j]], sps.wavelengths, spec_in_units)
                ew = lflux / continuum

                outdict[line]['lum'][i,k] = float(llum)
                outdict[line]['flux'][i,k] = float(lflux)
                outdict[line]['ew'][i,k] = float(ew)

            # save galaxy parameters
            outdict['stellar_mass'][i,k] = np.log10(mformed * sm)
            outdict['logzsol'][i,k] = float(eout['quantiles']['sample_chain'][k,eout['quantiles']['parnames'] == 'logzsol'])
            outdict['sfr'][i,k] = float(eout['extras']['flatchain'][k,eout['extras']['parnames'] == 'sfr_100'])
            outdict['ssfr'][i,k] = float(eout['extras']['flatchain'][k,eout['extras']['parnames'] == 'ssfr_100'])
        print i

    hickle.dump(outdict,open(filename, "w"))
    return outdict

def locations():
    """prints RA, DEC of galaxies in the Brown sample
    """

    ### locations
    datname = os.getenv('APPS')+'/prospector_alpha/data/brownseds_data/photometry/table1.fits'
    herschname = os.getenv('APPS')+'/prospector_alpha/data/brownseds_data/photometry/kingfish.brownapertures.flux.fits'
    photname = os.getenv('APPS')+'/prospector_alpha/data/brownseds_data/photometry/table3.txt'

    ### open
    herschel = fits.open(herschname)
    hdulist = fits.open(datname)

    match=0
    for i, name in enumerate(herschel[1].data['Name']):

        idx = hdulist[1].data['Name'].lower() .replace(' ','') == name
        if herschel[1].data['pacs70'][i] == 0:
            continue

        print name + ' RA:' + str(hdulist[1].data['RAh'][i]) + 'h ' + str(hdulist[1].data['RAm'][i])\
                   + 'm, Dec: ' + str(hdulist[1].data['DEd'][i]) + 'd ' + str(hdulist[1].data['DEm'][i])+'m'
        match +=1

    print match












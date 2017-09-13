import numpy as np
import matplotlib.pyplot as plt
import os, hickle, td_io, prosp_dutils
from brown_io import load_prospector_extra
from astropy.cosmology import WMAP9
from astropy import units as u

plt.ioff()

plotopts = {
         'fmt':'o',
         'ecolor':'k',
         'capthick':0.4,
         'elinewidth':0.4,
         'alpha':0.5,
         'ms':0.0,
         'zorder':-2
        } 
def collate_data(runname, filename=None, regenerate=False, nsamp=100):
    
    ### if it's already made, load it and give it back
    # else, start with the making!
    if os.path.isfile(filename) and regenerate == False:
        with open(filename, "r") as f:
            outdict=hickle.load(f)
            return outdict

    ### define output containers
    out = {'objname':[]}
    entries = ['ha_ew_obs', 'ha_ew_mod', 'ha_flux_mod', 'ha_flux_obs']
    qvals = ['q50', 'q16', 'q84']
    for e in entries:
        out[e] = {}
        if 'obs' in e:
            out[e]['val'] = []
            out[e]['err'] = []
        else:
            for q in qvals: out[e][q] = []

    ### load up ancillary dataset
    basenames, _, _ = prosp_dutils.generate_basenames(runname)
    field = [name.split('/')[-1].split('_')[0] for name in basenames]

    ancil = []
    allfields = np.unique(field).tolist()
    for f in allfields:
        ancil.append(td_io.load_ancil_data(runname,f))

    for i, name in enumerate(basenames):

        #### load input
        # make sure all files exist
        try:
            prosp = load_prospector_extra(name)
            print name.split('/')[-1]+' loaded.'
        except:
            continue
        out['objname'].append(name.split('/')[-1])
        objfield = out['objname'][-1].split('_')[0]
        objnumber = int(out['objname'][-1].split('_')[1])

        # fill out model data
        # comes out in rest-frame EW and Lsun
        hidx = prosp['model_emline']['emnames'] == 'Halpha'
        for q in qvals: out['ha_ew_mod'][q].append(prosp['model_emline']['eqw'][q][hidx][0])
        for q in qvals: out['ha_flux_mod'][q].append(prosp['model_emline']['flux'][q][hidx][0])

        # fill out observed data
        # comes out in rest-frame EW and (10**-17 ergs / s / cm**2) ?
        fidx = allfields.index(objfield)
        oidx = ancil[fidx]['phot_id'] == objnumber

        zred = ancil[fidx]['z_max_grism'][oidx][0]
        lumdist = WMAP9.luminosity_distance(zred).value
        dfactor = 4*np.pi*(u.Mpc.to(u.cm) * lumdist)**2

        out['ha_flux_obs']['val'].append(ancil[fidx]['Ha_FLUX'][oidx][0] * 1e-17 * dfactor / 3.846e33)
        out['ha_flux_obs']['err'].append(ancil[fidx]['Ha_FLUX_ERR'][oidx][0] * 1e-17 * dfactor / 3.846e33)
        out['ha_ew_obs']['val'].append(ancil[fidx]['Ha_EQW'][oidx][0]/(1+zred))
        out['ha_ew_obs']['err'].append(ancil[fidx]['Ha_EQW_ERR'][oidx][0]/(1+zred))

    for key in out.keys():
        if type(out[key]) == dict:
            for key2 in out[key].keys(): out[key][key2] = np.array(out[key][key2])
        else:
            out[key] = np.array(out[key])

    ### dump files and return
    hickle.dump(out,open(filename, "w"))
    return out

def do_all(runname='td_ha', outfolder=None,**opts):

    if outfolder is None:
        outfolder = os.getenv('APPS') + '/prospector_alpha/plots/'+runname+'/fast_plots/'
        if not os.path.isdir(outfolder):
            os.makedirs(outfolder)
            os.makedirs(outfolder+'data/')

    data = collate_data(runname,filename=outfolder+'data/hacomp.h5',**opts)

    plot(data,outfolder)

def plot(data, outfolder):

    # set it up
    fig, ax = plt.subplots(1,2, figsize=(10.5, 5))
    fs = 18 # font size
    ms = 8
    alpha = 0.9
    color = '#545454'

    # make plots
    xplot = data['ha_flux_obs']['val']
    yplot = data['ha_flux_mod']['q50']
    xplot_err = data['ha_flux_obs']['err']
    yplot_err = prosp_dutils.asym_errors(yplot,data['ha_flux_mod']['q84'],data['ha_flux_mod']['q16'])
    ax[0].errorbar(xplot, yplot, xerr=xplot_err, yerr=yplot_err,
                   **plotopts)
    ax[0].plot(xplot, yplot, 'o', linestyle=' ', 
               alpha=alpha, markeredgecolor='k',
               color=color,ms=ms)
    
    ax[0].set_xlabel(r'observed L(H$\alpha$) [L$_{\odot}$]',fontsize=fs)
    ax[0].set_ylabel(r'model L(H$\alpha$) [L$_{\odot}$]',fontsize=fs)
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].xaxis.set_tick_params(labelsize=fs)
    ax[0].yaxis.set_tick_params(labelsize=fs)

    ## line of equality + range
    min, max = np.min([xplot.min(),yplot.min()])*0.5, np.max([xplot.max(),yplot.max()])*2
    ax[0].axis((min,max,min,max))
    ax[0].plot([min,max],[min,max],linestyle='--',color='0.1',alpha=0.8)

    # offset and scatter
    off,scat = prosp_dutils.offset_and_scatter(np.log10(xplot),np.log10(yplot),biweight=True)
    ax[0].text(0.02,0.92,'offset='+"{:.2f}".format(off)+' dex',fontsize=fs,transform=ax[0].transAxes)
    ax[0].text(0.02,0.865,'scatter='+"{:.2f}".format(scat)+' dex',fontsize=fs,transform=ax[0].transAxes)

    # make plots
    xplot = data['ha_ew_obs']['val']
    yplot = data['ha_ew_mod']['q50']
    xplot_err = data['ha_ew_obs']['err']
    yplot_err = prosp_dutils.asym_errors(yplot,data['ha_ew_mod']['q84'],data['ha_ew_mod']['q16'])
    ax[1].errorbar(xplot, yplot, xerr=xplot_err, yerr=yplot_err,
                   **plotopts)
    ax[1].plot(xplot, yplot, 'o', linestyle=' ', 
               alpha=alpha, markeredgecolor='k',
               color=color,ms=ms)
    
    ax[1].set_xlabel(r'observed EW(H$\alpha$)',fontsize=fs)
    ax[1].set_ylabel(r'model EW(H$\alpha$)',fontsize=fs)
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].xaxis.set_tick_params(labelsize=fs)
    ax[1].yaxis.set_tick_params(labelsize=fs)

    ## line of equality + range
    min, max = np.min([xplot.min(),yplot.min()])*0.5, np.max([xplot.max(),yplot.max()])*2
    ax[1].axis((min,max,min,max))
    ax[1].plot([min,max],[min,max],linestyle='--',color='0.1',alpha=0.8)

   # offset and scatter
    off,scat = prosp_dutils.offset_and_scatter(np.log10(xplot),np.log10(yplot),biweight=True)
    ax[1].text(0.02,0.92,'offset='+"{:.2f}".format(off)+' dex',fontsize=fs,transform=ax[1].transAxes)
    ax[1].text(0.02,0.865,'scatter='+"{:.2f}".format(scat)+' dex',fontsize=fs,transform=ax[1].transAxes)

    plt.tight_layout()
    plt.savefig(outfolder+'ha_comp.png',dpi=150)





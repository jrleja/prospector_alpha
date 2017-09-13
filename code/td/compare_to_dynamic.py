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

    ### define output containers and constants
    G = 4.302e-3 # pc Msun**-1 (km/s)**2
    out = {'objname':[]}
    entries = ['prosp_smass', 'rachel_smass', 'fast_smass', 'dyn_mass', 'dyn_mass_sersic']
    qvals = ['q50', 'q16', 'q84']
    for e in entries:
        if ('prosp' in e) or ('dyn' in e):
            out[e] = {}
            for q in qvals: out[e][q] = []
        else:
            if 'smass' in e: # FAST doesn't give errors on mass..
                out[e] = []
            else:
                out[e] = {}
                out[e]['val'] = []
                out[e]['err'] = []


    ### load up ancillary dataset
    basenames, _, _ = prosp_dutils.generate_basenames(runname)
    field = [name.split('/')[-1].split('_')[0] for name in basenames]

    ancil, fast = [], []
    allfields = np.unique(field).tolist()
    for f in allfields:
        ancil.append(td_io.load_ancil_data(runname,f))
        fast.append(td_io.load_fast(runname,f))

    for i, name in enumerate(basenames):

        #### load input
        # make sure all files exist
        try:
            prosp = load_prospector_extra(name)
            print name.split('/')[-1]+' loaded.'
        except:
            continue
        out['objname'].append(name.split('/')[-1])
        field = out['objname'][-1].split('_')[0]
        number = out['objname'][-1].split('_')[1]

        # model data
        idx = prosp['extras']['parnames'] == 'stellar_mass'
        out['prosp_smass']['q50'].append(np.log10(prosp['extras']['q50'][idx])[0])
        out['prosp_smass']['q84'].append(np.log10(prosp['extras']['q84'][idx])[0])
        out['prosp_smass']['q16'].append(np.log10(prosp['extras']['q16'][idx])[0])

        # fast masses
        fidx = allfields.index(field)
        idx = ancil[fidx]['phot_id'] == float(number)
        out['rachel_smass'].append(ancil[fidx][idx]['logM'][0])
        out['fast_smass'].append(fast[fidx][idx]['lmass'][0])

        # rachel dynamical masses
        # from bezanson 2014, eqn 13+14
        # sigmaRe in km/s, Re in kpc
        sigmaRe = ancil[fidx][idx]['sigmaRe'][0]
        e_sigmaRe = ancil[fidx][idx]['e_sigmaRe'][0]
        Re      = ancil[fidx][idx]['Re'][0]*1e3
        nserc   = ancil[fidx][idx]['n'][0]

        k              = 5.0
        mdyn_cnst      = k*Re*sigmaRe**2/G
        mdyn_cnst_err  = 2*k*Re*sigmaRe*e_sigmaRe/G
        out['dyn_mass']['q50'].append(np.log10(mdyn_cnst))
        out['dyn_mass']['q84'].append(np.log10(mdyn_cnst+mdyn_cnst_err))
        out['dyn_mass']['q16'].append(np.log10(mdyn_cnst-mdyn_cnst_err))

        k              = 8.87 - 0.831*nserc + 0.0241*nserc**2
        mdyn_serc      = k*Re*sigmaRe**2/G
        mdyn_serc_err  = 2*k*Re*sigmaRe*e_sigmaRe/G
        out['dyn_mass_sersic']['q50'].append(np.log10(mdyn_serc))
        out['dyn_mass_sersic']['q84'].append(np.log10(mdyn_serc+mdyn_serc_err))
        out['dyn_mass_sersic']['q16'].append(np.log10(mdyn_serc-mdyn_serc_err))

    for key in out.keys():
        if type(out[key]) == dict:
            for key2 in out[key].keys(): out[key][key2] = np.array(out[key][key2])
        else:
            out[key] = np.array(out[key])

    ### dump files and return
    hickle.dump(out,open(filename, "w"))
    return out

def do_all(runname='td_dynamic', outfolder=None,**opts):

    if outfolder is None:
        outfolder = os.getenv('APPS') + '/prospector_alpha/plots/'+runname+'/fast_plots/'
        if not os.path.isdir(outfolder):
            os.makedirs(outfolder)
            os.makedirs(outfolder+'data/')

    data = collate_data(runname,filename=outfolder+'data/dyncomp.h5',**opts)

    plot(data,outfolder)

def plot(data, outfolder):

    # set it up
    fig, ax = plt.subplots(1,2, figsize=(10.5, 5))
    fs = 18 # font size
    ms = 8
    alpha = 0.9
    color = '#545454'

    # make plots
    xplot = data['rachel_smass']
    yplot = data['dyn_mass_sersic']['q50']
    yplot_err = prosp_dutils.asym_errors(yplot,data['dyn_mass_sersic']['q84'],data['dyn_mass_sersic']['q16'])
    ax[0].errorbar(xplot, yplot, yerr=yplot_err,
                   **plotopts)
    ax[0].plot(xplot, yplot, 'o', linestyle=' ', 
               alpha=alpha, markeredgecolor='k',
               color=color,ms=ms)
    
    ax[0].set_xlabel(r'log(M$_{\mathrm{FAST}}$/M$_{\odot}$)',fontsize=fs)
    ax[0].set_ylabel(r'log(M$_{\mathrm{dyn}}$/M$_{\odot}$)',fontsize=fs)
    ax[0].xaxis.set_tick_params(labelsize=fs)
    ax[0].yaxis.set_tick_params(labelsize=fs)

    ## line of equality + range
    min, max = np.min([xplot.min(),yplot.min()])*0.98, np.max([xplot.max(),yplot.max()])*1.02
    ax[0].axis((min,max,min,max))
    ax[0].plot([min,max],[min,max],linestyle='--',color='0.1',alpha=0.8)

    # offset and scatter
    off,scat = prosp_dutils.offset_and_scatter(xplot,yplot,biweight=True)
    ax[0].text(0.98,0.11,'offset='+"{:.2f}".format(off)+' dex',fontsize=fs,transform=ax[0].transAxes,ha='right')
    ax[0].text(0.98,0.055,'scatter='+"{:.2f}".format(scat)+' dex',fontsize=fs,transform=ax[0].transAxes,ha='right')

    # make plots
    xplot = data['prosp_smass']['q50']
    xplot_err = prosp_dutils.asym_errors(xplot,data['prosp_smass']['q84'],data['prosp_smass']['q16'])
    ax[1].errorbar(xplot, yplot, xerr=xplot_err, yerr=yplot_err,
                   **plotopts)
    ax[1].plot(xplot, yplot, 'o', linestyle=' ', 
               alpha=alpha, markeredgecolor='k',
               color=color,ms=ms)
    
    ax[1].set_xlabel(r'log(M$_{\mathrm{Prospector}}$/M$_{\odot}$)',fontsize=fs)
    ax[1].set_ylabel(r'log(M$_{\mathrm{dyn}}$/M$_{\odot}$)',fontsize=fs)
    ax[1].xaxis.set_tick_params(labelsize=fs)
    ax[1].yaxis.set_tick_params(labelsize=fs)

    ## line of equality + range
    min, max = np.min([xplot.min(),yplot.min()])*0.98, np.max([xplot.max(),yplot.max()])*1.02
    ax[1].axis((min,max,min,max))
    ax[1].plot([min,max],[min,max],linestyle='--',color='0.1',alpha=0.8)

   # offset and scatter
    off,scat = prosp_dutils.offset_and_scatter(xplot,yplot,biweight=True)
    ax[1].text(0.98,0.11,'offset='+"{:.2f}".format(off)+' dex',fontsize=fs,transform=ax[1].transAxes,ha='right')
    ax[1].text(0.98,0.055,'scatter='+"{:.2f}".format(scat)+' dex',fontsize=fs,transform=ax[1].transAxes,ha='right')

    plt.tight_layout()
    plt.savefig(outfolder+'dyn_comp.png',dpi=150)
    print 1/0
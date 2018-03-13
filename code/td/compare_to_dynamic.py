import numpy as np
import matplotlib.pyplot as plt
import os, hickle, td_io, prosp_dutils
from prospector_io import load_prospector_extra
from astropy.cosmology import WMAP9
from astropy import units as u

plt.ioff()

plotopts = {
         'fmt':'o',
         'ecolor':'k',
         'capthick':1,
         'elinewidth':1,
         'alpha':0.6,
         'ms':0.0,
         'zorder':-2
        } 
def collate_data(runname, filename=None, regenerate=False, nsamp=100, **opts):
    
    ### if it's already made, load it and give it back
    # else, start with the making!
    if os.path.isfile(filename) and regenerate == False:
        with open(filename, "r") as f:
            outdict=hickle.load(f)
            return outdict

    ### define output containers and constants
    G = 4.302e-3 # pc Msun**-1 (km/s)**2
    out = {'objname':[]}
    entries = ['prosp_smass', 'rachel_smass', 'rachel_z', 'fast_smass', 'fast_z', 'dyn_mass', 'dyn_mass_sersic']
    qvals = ['q50', 'q16', 'q84']
    for e in entries:
        if ('prosp' in e) or ('dyn' in e):
            out[e] = {}
            for q in qvals: out[e][q] = []
        else:
            if ('smass') in e or 'z' in e: # FAST doesn't give errors on mass..
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
        out['prosp_smass']['q50'].append(np.log10(prosp['extras']['stellar_mass']['q50']))
        out['prosp_smass']['q84'].append(np.log10(prosp['extras']['stellar_mass']['q84']))
        out['prosp_smass']['q16'].append(np.log10(prosp['extras']['stellar_mass']['q16']))

        # fast masses
        fidx = allfields.index(field)
        idx = ancil[fidx]['phot_id'] == float(number)
        out['rachel_smass'].append(ancil[fidx][idx]['logM'][0])
        out['rachel_z'] += [ancil[fidx][idx]['z_bez'][0]]
        out['fast_smass'].append(fast[fidx][idx]['lmass'][0])
        out['fast_z'] += [fast[fidx][idx]['z'][0]]

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

    plot(data,outfolder,**opts)

def plot(data, outfolder, use_rachel_smass=False,**popts):

    # set it up
    fig, ax = plt.subplots(1,3, figsize=(15, 5))
    fs = 18 # font size
    ms = 8
    alpha = 0.9
    color = '#545454'
    lim = (9.8,11.7)

    # physics
    fidx = 'fast_smass'
    smass_str = '3dhst'
    if use_rachel_smass:
        fidx = 'rachel_smass'
        smass_str = 'rachel'
    didx = 'dyn_mass'
    didx = 'dyn_mass_sersic'

    # light editing (mismatch? what's happening?)
    # this is a BAD MATCH
    # 3D-HST has a v. different redshift which makes the mass much higher
    # this passes through the matching algorithm
    # solution: require dz? not sure. try it out.
    idx = (data['prosp_smass']['q50'] > 10)

    # make plots
    xplot = data[fidx][idx]
    yplot = data[didx]['q50'][idx]
    yplot_err = prosp_dutils.asym_errors(yplot,data[didx]['q84'][idx],data[didx]['q16'][idx])
    ax[0].errorbar(xplot, yplot, yerr=yplot_err,
                   **plotopts)
    ax[0].plot(xplot, yplot, 'o', linestyle=' ', 
               alpha=alpha, markeredgecolor='k',
               color=color,ms=ms)
    
    ax[0].set_xlabel(r'log(M$_{\mathrm{FAST}}$/M$_{\odot}$)',fontsize=fs)
    ax[0].set_ylabel(r'log(M$_{\mathrm{dyn}}$/M$_{\odot}$)',fontsize=fs)

    ## line of equality + range
    ax[0].axis((lim[0],lim[1],lim[0],lim[1]))
    ax[0].plot([lim[0],lim[1]],[lim[0],lim[1]],linestyle='--',color='0.1',alpha=0.8)

    # offset and scatter
    off,scat = prosp_dutils.offset_and_scatter(xplot,yplot,biweight=True)
    ax[0].text(0.98,0.11,'offset='+"{:.2f}".format(off)+' dex',fontsize=fs,transform=ax[0].transAxes,ha='right')
    ax[0].text(0.98,0.055,'scatter='+"{:.2f}".format(scat)+' dex',fontsize=fs,transform=ax[0].transAxes,ha='right')

    # make plots
    xplot = data['prosp_smass']['q50'][idx]
    xplot_err = prosp_dutils.asym_errors(xplot,data['prosp_smass']['q84'][idx],data['prosp_smass']['q16'][idx])
    ax[1].errorbar(xplot, yplot, xerr=xplot_err, yerr=yplot_err,
                   **plotopts)
    ax[1].plot(xplot, yplot, 'o', linestyle=' ', 
               alpha=alpha, markeredgecolor='k',
               color=color,ms=ms)
    
    ax[1].set_xlabel(r'log(M$_{\mathrm{Prospector}}$/M$_{\odot}$)',fontsize=fs)
    ax[1].set_ylabel(r'log(M$_{\mathrm{dyn}}$/M$_{\odot}$)',fontsize=fs)

    ## line of equality + range
    ax[1].axis((lim[0],lim[1],lim[0],lim[1]))
    ax[1].plot([lim[0],lim[1]],[lim[0],lim[1]],linestyle='--',color='0.1',alpha=0.8)

    # offset and scatter
    off,scat = prosp_dutils.offset_and_scatter(xplot,yplot,biweight=True)
    ax[1].text(0.98,0.11,'offset='+"{:.2f}".format(off)+' dex',fontsize=fs,transform=ax[1].transAxes,ha='right')
    ax[1].text(0.98,0.055,'scatter='+"{:.2f}".format(scat)+' dex',fontsize=fs,transform=ax[1].transAxes,ha='right')

    # and the finale
    xplot = data[didx]['q50'][idx] - data[fidx][idx]
    yplot = data['prosp_smass']['q50'][idx] - data[fidx][idx]
    ax[2].errorbar(xplot, yplot, xerr=yplot_err, yerr=xplot_err,
                   **plotopts)
    ax[2].plot(xplot, yplot, 'o', linestyle=' ', 
               alpha=alpha, markeredgecolor='k',
               color=color,ms=ms)
    ax[2].set_xlabel('log(M$_{\mathrm{dyn}}$/M$_{\mathrm{FAST}}$)',fontsize=fs)
    ax[2].set_ylabel('log(M$_{\mathrm{Prosp}}$/M$_{\mathrm{FAST}}$)',fontsize=fs)

    lim = (-0.2,1)
    ax[2].plot([lim[0],lim[1]],[lim[0],lim[1]],linestyle='--',color='0.1',alpha=0.8)
    ax[2].plot([lim[0],lim[1]],[0,0],linestyle=':',color='0.1',alpha=0.8)
    ax[2].plot([0,0],[lim[0],lim[1]],linestyle=':',color='0.1',alpha=0.8)
    ax[2].axis((lim[0],lim[1],lim[0],lim[1]))

    # globals
    for a in ax:
        a.xaxis.set_tick_params(labelsize=fs)
        a.yaxis.set_tick_params(labelsize=fs)


    plt.tight_layout()
    plt.savefig(outfolder+'dyn_comp'+smass_str+'.png',dpi=150)
    print 1/0

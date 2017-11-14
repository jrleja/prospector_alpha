import numpy as np
import matplotlib.pyplot as plt
import os, hickle, td_io, prosp_dutils
from prospector_io import load_prospector_extra
from astropy.cosmology import WMAP9
from astropy import units as u
from scipy.interpolate import interp2d

plt.ioff()

def nii_ha_ratio():
    """generates interpolator for the NII / Ha ratio
    """
    # table 1 of Faisst et al. 2017
    zred = np.linspace(0.0,2.6,14)
    mass = np.linspace(8.5,11.1,14)
    dat = np.loadtxt(os.getenv('APPS')+'/prospector_alpha/data/faisst_et_al_2017_tbl1.txt',delimiter=',')

    # interpolate
    func = interp2d(mass, zred, dat, kind='cubic')
    return func

def collate_data(runname, filename=None, regenerate=False, **opts):
    
    ### if it's already made, load it and give it back
    # else, start with the making!
    if os.path.isfile(filename) and (regenerate == False):
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
    nii_ha_fnc = nii_ha_ratio()

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
        if prosp is None:
            continue
        out['objname'].append(name.split('/')[-1])
        objfield = out['objname'][-1].split('_')[0]
        objnumber = int(out['objname'][-1].split('_')[1])

        # fill in model data
        # comes out in rest-frame EW and Lsun
        for q in qvals: out['ha_ew_mod'][q].append(prosp['obs']['elines']['H alpha 6563']['ew'][q])
        for q in qvals: out['ha_flux_mod'][q].append(prosp['obs']['elines']['H alpha 6563']['flux'][q])

        # fill in observed data
        # comes out in observed-frame EW and (10**-17 ergs / s / cm**2)
        fidx = allfields.index(objfield)
        oidx = ancil[fidx]['phot_id'] == objnumber

        # account for NII / Halpha ratio, distance
        zred = ancil[fidx]['z_best'][oidx][0]
        mass = np.log10(prosp['extras']['stellar_mass']['q50'])
        nii_correction = float(1-nii_ha_fnc(mass,zred))
        lumdist = WMAP9.luminosity_distance(zred).value
        dfactor = 4*np.pi*(u.Mpc.to(u.cm) * lumdist)**2

        # fill in and march on
        out['ha_flux_obs']['val'].append(ancil[fidx]['Ha_FLUX'][oidx][0] * 1e-17 * dfactor / 3.828e33 * nii_correction)
        out['ha_flux_obs']['err'].append(ancil[fidx]['Ha_FLUX_ERR'][oidx][0] * 1e-17 * dfactor / 3.828e33 * nii_correction)
        out['ha_ew_obs']['val'].append(ancil[fidx]['Ha_EQW'][oidx][0]/(1+zred) * nii_correction)
        out['ha_ew_obs']['err'].append(ancil[fidx]['Ha_EQW_ERR'][oidx][0]/(1+zred) * nii_correction)

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
    symopts = {'ms':1.2,'alpha':0.6,'color':'#545454','linestyle':' '}
    ebaropts = {'fmt':'o', 'ecolor':'k', 'capthick':0.1, 'elinewidth':0.1, 'alpha':0.3, 'ms':0.0, 'zorder':-2} 
    sn_limit = 5
    sn_mod_limit = 5

    # make cuts
    sn = data['ha_flux_obs']['val'] / data['ha_flux_obs']['err']
    sn_mod = data['ha_flux_mod']['q50'] / ((data['ha_flux_mod']['q84'] - data['ha_flux_mod']['q16'])/2.)
    sn_ew = data['ha_ew_obs']['val'] / data['ha_ew_obs']['err']
    sn_mod_ew = data['ha_ew_mod']['q50'] / ((data['ha_ew_mod']['q84'] - data['ha_ew_mod']['q16'])/2.)
    idx = (sn > sn_limit) & np.isfinite(sn) & \
          (sn_mod > sn_mod_limit) & \
          (sn_ew > sn_limit) & np.isfinite(sn_ew) & \
          (sn_mod_ew > sn_mod_limit)

    # grab data
    xplot = data['ha_flux_obs']['val'][idx]
    yplot = data['ha_flux_mod']['q50'][idx]
    xplot_err = data['ha_flux_obs']['err'][idx]
    yplot_err = prosp_dutils.asym_errors(yplot,data['ha_flux_mod']['q84'][idx],data['ha_flux_mod']['q16'][idx])

    # make plots
    ax[0].errorbar(xplot, yplot, xerr=xplot_err, yerr=yplot_err, **ebaropts)
    ax[0].plot(xplot, yplot, 'o', **symopts)
    
    ax[0].set_xlabel(r'observed L(H$\alpha$) [L$_{\odot}$]',fontsize=fs)
    ax[0].set_ylabel(r'model L(H$\alpha$) [L$_{\odot}$]',fontsize=fs)
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].xaxis.set_tick_params(labelsize=fs)
    ax[0].yaxis.set_tick_params(labelsize=fs)

    ## line of equality + range
    min, max = np.min([xplot.min(),yplot.min()])*0.5, np.max([xplot.max(),yplot.max()])*2
    min, max = 6e6,5e9
    ax[0].axis((min,max,min,max))
    ax[0].plot([min,max],[min,max],linestyle='--',color='0.1',alpha=0.8)

    # offset and scatter
    off,scat = prosp_dutils.offset_and_scatter(np.log10(xplot),np.log10(yplot),biweight=True)
    ax[0].text(0.02,0.92,'offset='+"{:.2f}".format(off)+' dex',fontsize=fs,transform=ax[0].transAxes)
    ax[0].text(0.02,0.865,'scatter='+"{:.2f}".format(scat)+' dex',fontsize=fs,transform=ax[0].transAxes)

    # grab data
    xplot = data['ha_ew_obs']['val'][idx]
    yplot = data['ha_ew_mod']['q50'][idx]
    xplot_err = data['ha_ew_obs']['err'][idx]
    yplot_err = prosp_dutils.asym_errors(yplot,data['ha_ew_mod']['q84'][idx],data['ha_ew_mod']['q16'][idx])

    # make plots
    ax[1].errorbar(xplot, yplot, xerr=xplot_err, yerr=yplot_err, **ebaropts)
    ax[1].plot(xplot, yplot, 'o', **symopts)
    
    ax[1].set_xlabel(r'observed EW(H$\alpha$)',fontsize=fs)
    ax[1].set_ylabel(r'model EW(H$\alpha$)',fontsize=fs)
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].xaxis.set_tick_params(labelsize=fs)
    ax[1].yaxis.set_tick_params(labelsize=fs)

    ## line of equality + range
    min, max = np.min([xplot.min(),yplot.min()])*0.5, np.max([xplot.max(),yplot.max()])*2
    min, max = 8,1e3
    ax[1].axis((min,max,min,max))
    ax[1].plot([min,max],[min,max],linestyle='--',color='0.1',alpha=0.8)

    # offset and scatter
    off,scat = prosp_dutils.offset_and_scatter(np.log10(xplot),np.log10(yplot),biweight=True)
    ax[1].text(0.02,0.92,'offset='+"{:.2f}".format(off)+' dex',fontsize=fs,transform=ax[1].transAxes)
    ax[1].text(0.02,0.865,'scatter='+"{:.2f}".format(scat)+' dex',fontsize=fs,transform=ax[1].transAxes)

    plt.tight_layout()
    plt.savefig(outfolder+'ha_comp.png',dpi=150)





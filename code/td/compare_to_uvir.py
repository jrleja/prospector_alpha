import numpy as np
import matplotlib.pyplot as plt
import os
import hickle
import td_io
from brown_io import load_prospector_extra
import magphys_plot_pref
from astropy.cosmology import WMAP9
import td_massive_params as pfile
from prospect.models import sedmodel
import prosp_dutils
import copy
import matplotlib.cm as cmx
import matplotlib.colors as colors



plt.ioff()

minorFormatter = magphys_plot_pref.jLogFormatter(base=10, labelOnlyBase=False)
majorFormatter = magphys_plot_pref.jLogFormatter(base=10, labelOnlyBase=True)
popts = {'fmt':'o', 'capthick':1.5,'elinewidth':1.5,'ms':9,'alpha':0.8,'color':'0.3'}
red = '#FF3D0D'
dpi = 120
cmap = 'cool'

def collate_data(runname, filename=None, regenerate=False):
    
    ### if it's already made, load it and give it back
    # else, start with the making!
    if os.path.isfile(filename) and regenerate == False:
        with open(filename, "r") as f:
            outdict=hickle.load(f)
            return outdict

    ### define output containers
    sfr_uvir_obs, sfr_uvir_prosp = [], []
    sfr_prosp, sfr_prosp_up, sfr_prosp_do = [], [], []
    
    sfr_bins = {}
    nbins = 6
    for i in range(nbins): 
        sfr_bins['bin'+str(i+1)] = []
    sfr_bins['time'] = []

    ### fill output containers
    basenames, _, _ = prosp_dutils.generate_basenames(runname)
    field = [name.split('/')[-1].split('_')[0] for name in basenames]

    uvirlist = []
    allfields = np.unique(field).tolist()
    for f in allfields:
        uvirlist.append(td_io.load_mips_data(f))
    for i, name in enumerate(basenames):

        #### load input
        print 'loading '+name.split('/')[-1]
        ### make sure all files exist
        try:
            prosp = load_prospector_extra(name)
        except:
            print name.split('/')[-1]+' failed to load. skipping.'
            continue

        ### calculuate UV+IR SFRs from model, as fraction of each stellar population
        run_params = pfile.run_params

        ### we only want these once, they're heavy
        if i == 0:
            sps = pfile.load_sps(**run_params)
            obs = pfile.load_obs(**run_params)
            objnames = np.genfromtxt(run_params['datdir']+run_params['runname']+'.ids',dtype=[('objnames', '|S40')])
        
            sfr_idx = prosp['extras']['parnames'] == 'sfr_100'

        #### clear out mass dependency in model
        # this is clever... I think?
        run_params['objname'] = objnames['objnames'][i]
        oldmodel = pfile.load_model(**run_params)
        model_params = copy.deepcopy(oldmodel.config_list)
        for j in range(len(model_params)):
            if model_params[j]['name'] == 'mass':
                model_params[j].pop('depends_on', None)
        model = sedmodel.SedModel(model_params)

        ### grab thetas, propagate, and get original UVIR SFR
        theta = prosp['bfit']['maxprob_params']
        oldmodel.set_parameters(theta)
        mass = oldmodel.params['mass']
        luv, lir = prosp['bfit']['luv'], prosp['bfit']['lir']
        sfr_uvir_prosp.append(prosp_dutils.sfr_uvir(lir,luv))

        ### calculate in each bin
        model.params['mass'] = np.zeros_like(mass)
        for j in range(nbins):
            model.params['mass'][j] = mass[j]
            out = prosp_dutils.measure_restframe_properties(sps, model = model, obs = obs, thetas = theta,
                                                            measure_ir = True, measure_luv = True)
            sfr_bins['bin'+str(j+1)].append(prosp_dutils.sfr_uvir(out['lir'],out['luv']))
            model.params['mass'][j] = 0.0

        sfr_bins['time'].append((10**model.params['agebins']).mean(axis=1))

        ### now the easy stuff
        sfr_prosp.append(prosp['extras']['q50'][sfr_idx])
        sfr_prosp_up.append(prosp['extras']['q84'][sfr_idx])
        sfr_prosp_do.append(prosp['extras']['q16'][sfr_idx])

        ### now UV+IR SFRs
        # find correct field, find ID match
        uvir = uvirlist[allfields.index(field[i])]
        u_idx = uvir['id'] == int(name.split('_')[-1])
        sfr_uvir_obs.append(uvir['sfr'][u_idx][0])


    ### turn everything into numpy arrays
    for i in range(nbins): sfr_bins['bin'+str(i+1)] = np.array(sfr_bins['bin'+str(i+1)])

    out = {
           'sfr_bins': sfr_bins,
           'sfr_uvir_obs': np.array(sfr_uvir_obs),
           'sfr_uvir_prosp': np.array(sfr_uvir_prosp),
           'sfr_prosp': np.array(sfr_prosp),
           'sfr_prosp_up': np.array(sfr_prosp_up),
           'sfr_prosp_do': np.array(sfr_prosp_do)
          }

    ### dump files and return
    hickle.dump(out,open(filename, "w"))
    return out

def do_all(runname='td_massive', outfolder=None,**opts):

    if outfolder is None:
        outfolder = os.getenv('APPS') + '/threedhst_bsfh/plots/'+runname+'/fast_plots/'
        if not os.path.isdir(outfolder):
            os.makedirs(outfolder)
            os.makedirs(outfolder+'data/')

    data = collate_data(runname,filename=outfolder+'data/uvircomp.h5',**opts)

    uvir_comparison(data,outfolder)

def uvir_comparison(data, outfolder):
    '''
    x-axis is log time (bins)
    y-axis is SFR_BIN / SFR_UVIR_PROSP ?
    color-code each line by SFR_PROSP / SFR_UVIR_PROSP ?
    make sure SFR_PROSP and SFR_UVIR_PROSP show same dependencies! 
    '''

    fig, ax = plt.subplots(1, 1, figsize = (7.5,6))
    fig2, ax2 = plt.subplots(1, 1, figsize = (6.5,6))

    #### First plot
    # clip at 1, for clarity (what's up with the four galaxies above 1?)
    ratio = np.clip(data['sfr_prosp'].squeeze() / data['sfr_uvir_prosp'], -np.inf, 1)
    sfr = 

    color_norm  = colors.Normalize(vmin=ratio.min(), vmax=ratio.max())
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap=cmap) 
    pltbins = []
    for i, time in enumerate(data['sfr_bins']['time']):

        tmp = []
        for j in xrange(6): tmp.append(data['sfr_bins']['bin'+str(j+1)][i])
        to_add = np.array(tmp) / data['sfr_uvir_prosp'][i]
        to_add = to_add / to_add.sum() # ensure it sums to one (THIS NEEDS TO BE INVESTIGATED)
        pltbins.append(to_add)

        ax.plot(time/1e9, pltbins[-1], 'o', color=scalar_map.to_rgba(ratio[i]), alpha=0.4, ms=4)

    ratio_bins = np.linspace(0,1,5)
    tmax = np.max(data['sfr_bins']['time'])/1e9
    tbins = np.array([0.01, 0.1, 0.3, 0.8, 1.8, 3.3, tmax])
    for i in xrange(ratio_bins.shape[0]-1):
        in_bin = (ratio > ratio_bins[i]) & (ratio <= ratio_bins[i+1])

        timeavg, sfravg = prosp_dutils.running_median(np.array(data['sfr_bins']['time'])[in_bin].flatten()/1e9,
                                                      np.array(pltbins)[in_bin].flatten(),
                                                      bins=tbins,avg=True)

        ax.plot(timeavg, sfravg, '-', color=scalar_map.to_rgba((ratio_bins[i]+ratio_bins[i+1])/2.), lw=4, zorder=1)


    pts = ax.scatter(ratio, ratio, s=0.0, cmap=cmap, c=ratio, vmin=ratio.min(), vmax=ratio.max())
    ax.set_xscale('log',nonposx='clip',subsx=(1,3))
    ax.xaxis.set_minor_formatter(minorFormatter)
    ax.xaxis.set_major_formatter(majorFormatter)

    ax.set_ylabel('SFR$_{\mathrm{UV+IR,bin}}$/SFR$_{\mathrm{UV+IR,total}}$')
    ax.set_xlabel('time [Gyr]')

    ax.set_xlim(0.03, tmax)
    ax.set_ylim(0,1)

    #### label and add colorbar
    cb = fig.colorbar(pts, ax=ax, aspect=10)
    cb.set_label(r'SFR$_{\mathrm{Prosp}}$/SFR$_{\mathrm{UV+IR,PROSP}}$')
    cb.solids.set_rasterized(True)
    cb.solids.set_edgecolor("face")

    #### second plot
    good = data['sfr_uvir_obs'] > 0

    ax2.plot(data['sfr_uvir_obs'][good], data['sfr_uvir_prosp'][good], 'o', color='0.4')
    min, max = data['sfr_uvir_obs'][good].min()*0.8, data['sfr_uvir_obs'][good].max()*1.2
    ax2.plot([min, max], [min, max], '--', color='0.4')
    ax2.axis([min,max,min,max])

    ax2.set_ylabel('SFR$_{\mathrm{UV+IR,PROSP}}$')
    ax2.set_xlabel('SFR$_{\mathrm{UV+IR,KATE}}$')

    ax2.set_xscale('log',nonposx='clip',subsx=(1,3))
    ax2.xaxis.set_minor_formatter(minorFormatter)
    ax2.xaxis.set_major_formatter(majorFormatter)

    ax2.set_yscale('log',nonposx='clip',subsy=(1,3))
    ax2.yaxis.set_minor_formatter(minorFormatter)
    ax2.yaxis.set_major_formatter(majorFormatter)

    fig.tight_layout()
    fig2.tight_layout()

    fig.savefig(outfolder+'uvir_bin_comparison.png',dpi=dpi)
    fig2.savefig(outfolder+'prosp_uvir_to_obs_uvir.png',dpi=dpi)

    plt.close()
    print 1/0

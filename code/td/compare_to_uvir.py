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
cmap = 'gist_rainbow'

def collate_data(runname, filename=None, regenerate=False, nsamp=100):
    
    ### if it's already made, load it and give it back
    # else, start with the making!
    if os.path.isfile(filename) and regenerate == False:
        with open(filename, "r") as f:
            outdict=hickle.load(f)
            return outdict

    ### define output containers
    sfr_uvir_obs, objname = [], []
    sfr_uvir_prosp, sfr_uvir_prosp_up, sfr_uvir_prosp_do, sfr_uvir_prosp_chain = [], [], [], []
    sfr_prosp, sfr_prosp_up, sfr_prosp_do, sfr_prosp_chain = [], [], [], []
    ssfr_prosp, ssfr_prosp_up, ssfr_prosp_do, ssfr_prosp_chain = [], [], [], []

    sfr_bins = {}
    nbins = 6
    for i in range(nbins): 
        sfr_bins['bin'+str(i+1)] = []
        sfr_bins['bin'+str(i+1)+'_up'] = []
        sfr_bins['bin'+str(i+1)+'_do'] = []
        sfr_bins['bin'+str(i+1)+'_chain'] = []

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
        # make sure all files exist
        try:
            prosp = load_prospector_extra(name)
        except:
            print name.split('/')[-1]+' failed to load. skipping.'
            continue
        objname.append(name.split('/')[-1])
        print 'loading '+objname[-1]

        ### calculuate UV+IR SFRs from model, as fraction of each stellar population
        run_params = pfile.run_params

        ### we only want these once, they're heavy
        if i == 0:
            sps = pfile.load_sps(**run_params)
            obs = pfile.load_obs(**run_params)
            objnames = np.genfromtxt(run_params['datdir']+run_params['runname']+'.ids',dtype=[('objnames', '|S40')])
        
            sfr_idx = prosp['extras']['parnames'] == 'sfr_100'
            ssfr_idx = prosp['extras']['parnames'] == 'ssfr_100'

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
        tsfr_uvir_prosp = []
        bins = np.empty(shape=(nsamp,nbins))
        for k in xrange(nsamp):
            
            ### if we only do it once, use best-fit
            ### otherwise, sample from posterior
            if nsamp == 1:
                theta = prosp['bfit']['maxprob_params']
            else:
                theta = prosp['quantiles']['sample_chain'][k+1,:] # skip the best-fit
            oldmodel.set_parameters(theta)
            mass = oldmodel.params['mass']

            out = prosp_dutils.measure_restframe_properties(sps, model = oldmodel, obs = obs, thetas = theta,
                                                            measure_ir = True, measure_luv = True)
            tsfr_uvir_prosp.append(prosp_dutils.sfr_uvir(out['lir'],out['luv']))

            ### calculate in each bin
            model.params['mass'] = np.zeros_like(mass)
            for j in range(nbins):
                model.params['mass'][j] = mass[j]
                out = prosp_dutils.measure_restframe_properties(sps, model = model, obs = obs, thetas = theta,
                                                                measure_ir = True, measure_luv = True)
                bins[k,j] = prosp_dutils.sfr_uvir(out['lir'],out['luv'])
                model.params['mass'][j] = 0.0

        #### append everything
        mid, up, down = np.percentile(tsfr_uvir_prosp,[50,84,16])
        sfr_uvir_prosp.append(mid)
        sfr_uvir_prosp_up.append(up)
        sfr_uvir_prosp_do.append(down)
        sfr_uvir_prosp_chain.append(tsfr_uvir_prosp)
        for j in range(nbins):
            mid, up, down = np.percentile(bins[:,j],[50,84,16])
            sfr_bins['bin'+str(j+1)].append(mid)
            sfr_bins['bin'+str(j+1)+'_up'].append(up)
            sfr_bins['bin'+str(j+1)+'_do'].append(down)
            sfr_bins['bin'+str(j+1)+'_chain'].append(bins[:,j])

        sfr_bins['time'].append((10**model.params['agebins']).mean(axis=1))

        ### now the easy stuff
        sfr_prosp.append(prosp['extras']['q50'][sfr_idx][0])
        sfr_prosp_up.append(prosp['extras']['q84'][sfr_idx][0])
        sfr_prosp_do.append(prosp['extras']['q16'][sfr_idx][0])
        sfr_prosp_chain.append(prosp['extras']['flatchain'][:,sfr_idx].squeeze())
        ssfr_prosp.append(prosp['extras']['q50'][ssfr_idx][0])
        ssfr_prosp_up.append(prosp['extras']['q84'][ssfr_idx][0])
        ssfr_prosp_do.append(prosp['extras']['q16'][ssfr_idx][0])
        ssfr_prosp_chain.append(prosp['extras']['flatchain'][:,ssfr_idx].squeeze())

        ### now UV+IR SFRs
        # find correct field, find ID match
        uvir = uvirlist[allfields.index(field[i])]
        u_idx = uvir['id'] == int(name.split('_')[-1])
        sfr_uvir_obs.append(uvir['sfr'][u_idx][0])


    ### turn everything into numpy arrays
    for i in range(nbins): 
        sfr_bins['bin'+str(i+1)] = np.array(sfr_bins['bin'+str(i+1)])
        sfr_bins['bin'+str(i+1)+'_up'] = np.array(sfr_bins['bin'+str(i+1)+'_up'])
        sfr_bins['bin'+str(i+1)+'_do'] = np.array(sfr_bins['bin'+str(i+1)+'_do'])

    out = {
           'sfr_bins': sfr_bins,
           'sfr_uvir_obs': np.array(sfr_uvir_obs),
           'sfr_uvir_prosp': np.array(sfr_uvir_prosp),
           'sfr_uvir_prosp_up': np.array(sfr_uvir_prosp_up),
           'sfr_uvir_prosp_do': np.array(sfr_uvir_prosp_do),
           'sfr_uvir_prosp_chain': sfr_uvir_prosp_chain,
           'sfr_prosp': np.array(sfr_prosp),
           'sfr_prosp_up': np.array(sfr_prosp_up),
           'sfr_prosp_do': np.array(sfr_prosp_do),
           'sfr_prosp_chain': sfr_prosp_chain,
           'ssfr_prosp': np.array(ssfr_prosp),
           'ssfr_prosp_up': np.array(ssfr_prosp_up),
           'ssfr_prosp_do': np.array(ssfr_prosp_do),
           'ssfr_prosp_chain': ssfr_prosp_chain,
           'objname': np.array(objname)
          }

    ### dump files and return
    hickle.dump(out,open(filename, "w"))
    return out

def do_all(runname='td_massive', outfolder=None,**opts):

    if outfolder is None:
        outfolder = os.getenv('APPS') + '/prospector_alpha/plots/'+runname+'/fast_plots/'
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

    fig, ax = plt.subplots(1, 1, figsize = (7.5,6)) # fractional UVIR SFR in each time bin + running averages
    fig2, ax2 = plt.subplots(1, 1, figsize = (6.5,6)) # Prospector UVIR SFR versus Kate's UVIR SFR
    fig3, ax3 = plt.subplots(1, 1, figsize = (7.5,6)) # Fractional UVIR SFR in 100 Myr versus (SED SFR - UVIR SFR)

    #### Plot individual points
    ssfr_prosp = np.log10(data['sfr_prosp'].squeeze())

    color_norm  = colors.Normalize(vmin=ssfr_prosp.min(), vmax=ssfr_prosp.max())
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap=cmap) 
    pltbins = []
    for i, time in enumerate(data['sfr_bins']['time']):

        tmp = []
        for j in xrange(6): tmp.append(data['sfr_bins']['bin'+str(j+1)][i])
        to_add = np.array(tmp) / data['sfr_uvir_prosp'][i]
        pltbins.append(to_add)

        ax.plot(time/1e9, pltbins[-1], 'o', color=scalar_map.to_rgba(ssfr_prosp[i]), alpha=0.4, ms=4)

    ### plot running averages
    bins = np.linspace(ssfr_prosp.min(),ssfr_prosp.max(),5)
    tmax = np.max(data['sfr_bins']['time'])/1e9
    tbins = np.array([0.01, 0.1, 0.3, 0.8, 1.8, 3.3, tmax])
    for i in xrange(bins.shape[0]-1):
        in_bin = (ssfr_prosp > bins[i]) & (ssfr_prosp <= bins[i+1])

        timeavg, sfravg = prosp_dutils.running_median(np.array(data['sfr_bins']['time'])[in_bin].flatten()/1e9,
                                                      np.array(pltbins)[in_bin].flatten(),
                                                      bins=tbins,avg=True)

        ax.plot(timeavg, sfravg, '-', color=scalar_map.to_rgba((bins[i]+bins[i+1])/2.), lw=4, zorder=1)

    ### FAKE SCATTER PLOT for the color map (sure there's a better way to do this but this is a one-liner)
    pts = ax.scatter(ssfr_prosp, ssfr_prosp, s=0.0, cmap=cmap, c=ssfr_prosp, vmin=ssfr_prosp.min(), vmax=ssfr_prosp.max())
    ax.set_xscale('log',nonposx='clip',subsx=(1,3))
    ax.xaxis.set_minor_formatter(minorFormatter)
    ax.xaxis.set_major_formatter(majorFormatter)

    ax.set_ylabel('SFR$_{\mathrm{UV+IR,bin}}$/SFR$_{\mathrm{UV+IR,total}}$')
    ax.set_xlabel('time [Gyr]')

    ax.set_xlim(0.03, tmax)
    ax.set_ylim(0,1)

    #### label and add colorbar
    cb = fig.colorbar(pts, ax=ax, aspect=10)
    cb.set_label(r'log(SFR$_{\mathrm{Prosp}}$)')
    cb.solids.set_rasterized(True)
    cb.solids.set_edgecolor("face")

    #### Plot Prospector UV+IR SFR versus the observed UV+IR SFR
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

    ##### Plot (UV+IR SFR / Prosp SFR)
    uvir_frac, uvir_frac_up, uvir_frac_do = [], [], []
    ratio, ratio_up, ratio_do = [], [], []
    for i, bchain in enumerate(data['sfr_bins']['bin1_chain']):
        sfr_uvir_prosp_chain = np.array([float(x) for x in data['sfr_uvir_prosp_chain'][i]])
        mid, up, do = np.percentile(bchain / sfr_uvir_prosp_chain, [50,84,16])
        uvir_frac.append(mid)
        uvir_frac_up.append(up)
        uvir_frac_do.append(do)

        mid, up, do = np.percentile(data['sfr_prosp_chain'][i][1:sfr_uvir_prosp_chain.shape[0]+1] / sfr_uvir_prosp_chain, [50,84,16])
        ratio.append(mid)
        ratio_up.append(up)
        ratio_do.append(do)

    x = uvir_frac
    xerr = prosp_dutils.asym_errors(np.array(uvir_frac), np.array(uvir_frac_up), np.array(uvir_frac_do))
    y = ratio
    yerr = prosp_dutils.asym_errors(np.array(ratio), np.array(ratio_up), np.array(ratio_do))

    #ax3.errorbar(x,y, yerr=yerr, xerr=xerr, fmt='o', ecolor='0.2', capthick=0.6,elinewidth=0.6,ms=0.0,alpha=0.5,zorder=-5)
    ax3.axis([0,1,0,1.2])

    #### add colored points and colorbar
    ssfr_prosp = np.log10(data['ssfr_prosp'])
    pts = ax3.scatter(x,y, s=50, cmap=cmap, c=ssfr_prosp, vmin=ssfr_prosp.min(), vmax=ssfr_prosp.max())
    cb = fig3.colorbar(pts, ax=ax3, aspect=10)
    cb.set_label(r'log(sSFR$_{\mathrm{SED}}$)')
    cb.solids.set_rasterized(True)
    cb.solids.set_edgecolor("face")

    ### label
    ax3.set_ylabel('SFR$_{\mathrm{SED}}$/SFR$_{\mathrm{UV+IR}}$')
    ax3.set_xlabel('fraction of heating from young stars')

    #### clean up
    fig.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()

    fig.savefig(outfolder+'uvir_bin_comparison.png',dpi=dpi)
    fig2.savefig(outfolder+'prosp_uvir_to_obs_uvir.png',dpi=dpi)
    fig3.savefig(outfolder+'uvir_frac.png',dpi=dpi)

    plt.close()
    print 1/0
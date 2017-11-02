import numpy as np
import matplotlib.pyplot as plt
import os, hickle, td_io, prosp_dutils
from matplotlib.ticker import FormatStrFormatter
from prospector_io import load_prospector_data
from astropy.cosmology import WMAP9
from dynesty.plotting import _quantile as weighted_quantile
from fix_ir_sed import mips_to_lir
import copy

plt.ioff()

red = '#FF3D0D'
dpi = 160

minlogssfr = -15
minssfr = 10**minlogssfr
minsfr = 0.0001

nbin_min = 5

def collate_data(runname, runname_fast, filename=None, regenerate=False, lir_from_mips=False):
    
    ### if it's already made, load it and give it back
    # else, start with the making!
    if os.path.isfile(filename) and regenerate == False:
        with open(filename, "r") as f:
            outdict=hickle.load(f)
            return outdict

    ### define output containers
    parlabels = [r'log(M$_{\mathrm{stellar}}$/M$_{\odot}$)', 'SFR [M$_{\odot}$/yr]',
                 r'$\tau_{\mathrm{diffuse}}$', r'log(sSFR) [yr$^-1$]',
                 r"t$_{\mathrm{half-mass}}$ [Gyr]", r'log(Z/Z$_{\odot}$)', r'Q$_{\mathrm{PAH}}$',
                 r'f$_{\mathrm{AGN}}$', 'dust index']
    fnames = ['stellar_mass','sfr_100','dust2','ssfr_100','half_time'] # take for fastmimic too
    pnames = fnames+['massmet_2', 'duste_qpah', 'fagn', 'dust_index'] # already calculated in Prospector
    enames = ['model_uvir_sfr', 'model_uvir_ssfr', 'sfr_ratio','model_uvir_truelir_sfr', 'model_uvir_truelir_ssfr', 'sfr_ratio_truelir'] # must calculate here

    outprosp, outprosp_fast, outfast, outlabels = {},{'bfit':{}},{},{}
    sfr_100_uvir, sfr_100_uv, sfr_100_ir = [], [], []
    phot_chi, phot_percentile, phot_obslam, phot_restlam, phot_fname = [], [], [], [], []
    outfast['z'] = []

    for i,par in enumerate(pnames+enames):
        
        ### look for it in FAST
        if par in fnames:
            outfast[par] = []

        ### if it's in FAST, it's in Prospector-FAST
        if par in outfast.keys():
            outprosp_fast[par] = {q:[] for q in ['q50','q84','q16']}
            outprosp_fast['bfit'][par] = []

        ### it's always in Prospector
        outprosp[par] = {}
        outprosp[par]['q50'],outprosp[par]['q84'],outprosp[par]['q16'] = [],[],[]
    
    # fill output containers
    basenames, _, _ = prosp_dutils.generate_basenames(runname)
    if runname_fast is not None:
        basenames_fast, _, _ = prosp_dutils.generate_basenames(runname_fast)
    field = [name.split('/')[-1].split('_')[0] for name in basenames]

    fastlist, uvirlist = [], []
    allfields = np.unique(field).tolist()
    for f in allfields:
        fastlist.append(td_io.load_fast(runname,f))
        uvirlist.append(td_io.load_ancil_data(runname,f))

    for i, name in enumerate(basenames):

        print 'loading '+name.split('/')[-1]

        ### make sure all files exist
        try:
            res, _, model, prosp = load_prospector_data(name)
            if runname_fast is not None:
                fres, _, fmodel, prosp_fast = load_prospector_data(basenames_fast[i])
            else:
                prosp_fast = None
        except:
            print name.split('/')[-1]+' failed to load. skipping.'
            continue

        if (prosp is None) or (model is None) or ((prosp_fast is None) & (runname_fast is not None)):
            continue

        ### prospector first
        for par in pnames:
            if par in prosp['thetas'].keys():
                loc = 'thetas'
            elif par in prosp['extras'].keys():
                loc = 'extras'
            else:
                continue
            for q in ['q16','q50','q84']:
                x = prosp[loc][par][q]
                if par == 'stellar_mass' or par == 'ssfr_100':
                    x = np.log10(x)
                outprosp[par][q].append(x)
        
        # input flux must be in mJy
        midx = np.array(['mips' in u for u in res['obs']['filternames']],dtype=bool)
        mips_flux = prosp['obs']['mags'][:,midx].squeeze() * 3631 * 1e3
        lir = mips_to_lir(mips_flux, res['model'].params['zred'][0])
        uvir_chain = prosp_dutils.sfr_uvir(lir,prosp['extras']['luv']['chain'])

        truelir = prosp['extras']['lir']['chain']
        uvir_truelir_chain = prosp_dutils.sfr_uvir(truelir,prosp['extras']['luv']['chain'])

        for q in ['q16','q50','q84']: 
            outprosp['model_uvir_truelir_sfr'][q] += [weighted_quantile(uvir_truelir_chain, np.array([float(q[1:])/100.]),weights=prosp['weights'])[0]]
            outprosp['model_uvir_truelir_ssfr'][q] += [weighted_quantile(uvir_truelir_chain/prosp['extras']['stellar_mass']['chain'], 
                                                               np.array([float(q[1:])/100.]),weights=prosp['weights'])[0]]
            outprosp['sfr_ratio_truelir'][q] += [weighted_quantile(np.log10(prosp['extras']['sfr_100']['chain']/uvir_truelir_chain), 
                                                         np.array([float(q[1:])/100.]),weights=prosp['weights'])[0]]
            outprosp['model_uvir_sfr'][q] += [weighted_quantile(uvir_chain, np.array([float(q[1:])/100.]),weights=prosp['weights'])[0]]
            outprosp['model_uvir_ssfr'][q] += [weighted_quantile(uvir_chain/prosp['extras']['stellar_mass']['chain'], 
                                                               np.array([float(q[1:])/100.]),weights=prosp['weights'])[0]]
            outprosp['sfr_ratio'][q] += [weighted_quantile(np.log10(prosp['extras']['sfr_100']['chain']/uvir_chain), 
                                                         np.array([float(q[1:])/100.]),weights=prosp['weights'])[0]]

        # prospector-fast
        if runname_fast is not None:
            for par in pnames:

                # switch to tell between 'thetas', 'extras', and 'NOT THERE'
                if par in prosp_fast['thetas'].keys():
                    loc = 'thetas'
                elif par in prosp_fast['extras'].keys():
                    loc = 'extras'
                else:
                    continue

                # fill up quantiles
                for q in ['q16','q50','q84']:
                    x = prosp_fast[loc][par][q]
                    if par == 'stellar_mass' or par == 'ssfr_100':
                        x = np.log10(x)
                    try:
                        outprosp_fast[par][q].append(x)
                    except KeyError:
                        continue
        
                # best-fit!
                # if it's a theta, must generate best-fit
                if loc == 'thetas':
                    amax = prosp_fast['sample_idx'][0]
                    idx = fres['model'].theta_labels().index(par)
                    x = fres['chain'][amax,idx]
                else:
                    x = prosp_fast[loc][par]['chain'][0]
                if par == 'stellar_mass' or par == 'ssfr_100':
                    x = np.log10(x)
                try:
                    outprosp_fast['bfit'][par].append(x)
                except KeyError:
                    continue

        ### now FAST, UV+IR SFRs
        # find correct field, find ID match
        fidx = allfields.index(field[i])
        fast = fastlist[fidx]
        uvir = uvirlist[fidx]
        f_idx = fast['id'] == int(name.split('_')[-1])
        u_idx = uvir['phot_id'] == int(name.split('_')[-1])

        # fill it up
        outfast['stellar_mass'] += [fast['lmass'][f_idx][0]]
        outfast['dust2'] += [prosp_dutils.av_to_dust2(fast['Av'][f_idx][0])]
        outfast['sfr_100'] += [10**fast['lsfr'][f_idx][0]]
        outfast['half_time'] += [prosp_dutils.exp_decl_sfh_half_time(10**fast['lage'][f_idx][0],10**fast['ltau'][f_idx][0])/1e9]
        outfast['z'] += [fast['z'][f_idx][0]]
        sfr_100_uvir += [uvir['sfr'][u_idx][0]]
        sfr_100_ir += [uvir['sfr_IR'][u_idx][0]]
        sfr_100_uv += [uvir['sfr_UV'][u_idx][0]]

        # photometry chi, etc.
        mask = res['obs']['phot_mask']
        phot_percentile += ((res['obs']['maggies'][mask] - prosp['obs']['mags'][0,mask]) / res['obs']['maggies'][mask]).tolist()
        phot_chi += ((res['obs']['maggies'][mask] - prosp['obs']['mags'][0,mask]) / res['obs']['maggies_unc'][mask]).tolist()
        phot_obslam += (res['obs']['wave_effective'][mask]/1e4).tolist()
        phot_restlam += (res['obs']['wave_effective'][mask]/1e4/(1+outfast['z'][-1])).tolist()
        phot_fname += [str(fname) for fname in np.array(res['obs']['filternames'])[mask]]

    ### turn everything into numpy arrays
    for k1 in outprosp.keys():
        for k2 in outprosp[k1].keys():
            outprosp[k1][k2] = np.array(outprosp[k1][k2])
    for k1 in outprosp_fast.keys():
        for k2 in outprosp_fast[k1].keys():
            outprosp_fast[k1][k2] = np.array(outprosp_fast[k1][k2])
    for key in outfast: outfast[key] = np.array(outfast[key])

    out = {
           'fast':outfast,
           'prosp':outprosp,
           'prosp_fast': outprosp_fast,
           'labels':np.array(parlabels),
           'pnames':np.array(pnames),
           'uvir_sfr': np.array(sfr_100_uvir),
           'ir_sfr': np.array(sfr_100_ir),
           'uv_sfr': np.array(sfr_100_uv),
           'phot_chi': np.array(phot_chi),
           'phot_percentile': np.array(phot_percentile),
           'phot_obslam': np.array(phot_obslam),
           'phot_restlam': np.array(phot_restlam),
           'phot_fname': np.array(phot_fname)
          }

    ### dump files and return
    hickle.dump(out,open(filename, "w"))
    return out

def do_all(runname='td_massive', runname_fast='fast_mimic',outfolder=None,**opts):

    if outfolder is None:
        outfolder = os.getenv('APPS') + '/prospector_alpha/plots/'+runname+'/fast_plots/'
        if not os.path.isdir(outfolder):
            os.makedirs(outfolder)
            os.makedirs(outfolder+'data/')

    data = collate_data(runname,runname_fast,filename=outfolder+'data/fastcomp.h5',**opts)

    popts = {'fmt':'o', 'capthick':1.5,'elinewidth':1.5,'alpha':0.8,'color':'0.3','ms':5,'markeredgecolor':'k'} # for small samples
    if len(data['uvir_sfr']) > 400:
        popts = {'fmt':'o', 'capthick':.15,'elinewidth':.15,'alpha':0.35,'color':'0.3','ms':2, 'errorevery': 5000}

    phot_residuals(data,outfolder,popts)

    # if we have FAST-mimic runs, do a thorough comparison
    # else just do Prospector-FAST
    if runname_fast is not None:
        fast_comparison(data['fast'],data['prosp_fast'],data['labels'],data['pnames'],
                        outfolder+'fast_to_fastmimic_comparison.png',popts,plabel='FAST-mimic')
        fast_comparison(data['fast'],data['prosp_fast'],data['labels'],data['pnames'],
                        outfolder+'fast_to_fastmimic_bfit_comparison.png',popts,plabel='FAST-mimic',bfit=True)
        fast_comparison(data['prosp_fast'],data['prosp'],data['labels'],data['pnames'],
                        outfolder+'fastmimic_to_palpha_comparison.png',popts,flabel='FAST-mimic')
        fast_comparison(data['fast'],data['prosp'],data['labels'],data['pnames'],
                        outfolder+'fast_to_palpha_comparison.png',popts)
    else:
        fast_comparison(data['fast'],data['prosp'],data['labels'],data['pnames'],
                        outfolder+'fast_to_palpha_comparison.png',popts)

    mass_metallicity_relationship(data, outfolder+'massmet_vs_z.png', popts)
    deltam_with_redshift(data['fast'], data['prosp'], data['fast']['z'], outfolder+'deltam_vs_z.png', filename=outfolder+'data/masscomp.h5')
    prospector_versus_z(data,outfolder+'prospector_versus_z.png',popts)

    # full UV+IR comparison
    uvir_comparison(data,outfolder+'sfr_uvir_comparison', popts, ssfr=False)
    uvir_comparison(data,outfolder+'sfr_uvir_comparison_model',  popts, model_uvir = True, ssfr=False)
    uvir_comparison(data,outfolder+'sfr_uvir_truelir_comparison_model',  popts, model_uvir = 'true_LIR', ssfr=False)
    uvir_comparison(data,outfolder+'ssfr_uvir_comparison', popts, filename=outfolder+'data/ssfrcomp.h5', ssfr=True)
    uvir_comparison(data,outfolder+'ssfr_uvir_comparison_model',  popts, model_uvir = True, ssfr=True)
    uvir_comparison(data,outfolder+'ssfr_uvir_truelir_comparison_model',  popts, model_uvir = 'true_LIR', ssfr=True)

def fast_comparison(fast,prosp,parlabels,pnames,outname,popts,flabel='FAST',plabel='Prospector',bfit=False):
    
    fig, axes = plt.subplots(2, 2, figsize = (6,6))
    axes = np.ravel(axes)

    for i,par in enumerate(['stellar_mass','sfr_100','dust2','half_time']):

        ### clip SFRs
        if par[:3] == 'sfr':
            minimum = minsfr
        else:
            minimum = -np.inf

        # grab for FAST
        # switch for FAST vs non-FAST outputs
        try:
            xfast = np.clip(fast[par]['q50'],minimum,np.inf)
            xfast_up = np.clip(fast[par]['q84'],minimum,np.inf)
            xfast_down = np.clip(fast[par]['q16'],minimum,np.inf)
            xerr = prosp_dutils.asym_errors(xfast, xfast_up, xfast_down, log=False)
        except:
            xfast = np.clip(fast[par],minimum,np.inf)
            xerr = None

        if bfit:
            yprosp = np.clip(prosp['bfit'][par],minimum,np.inf)
            yerr = None
        else:
            yprosp = np.clip(prosp[par]['q50'],minimum,np.inf)
            yprosp_up = np.clip(prosp[par]['q84'],minimum,np.inf)
            yprosp_down = np.clip(prosp[par]['q16'],minimum,np.inf)
            yerr = prosp_dutils.asym_errors(yprosp, yprosp_up, yprosp_down, log=False)

        ### plot
        axes[i].errorbar(xfast,yprosp,xerr=xerr,yerr=yerr,**popts)

        ### if we have some enforced minimum, don't include in scatter calculation
        if ((xfast == xfast.min()).sum()-1 != 0) | ((yprosp == yprosp.min()).sum()-1 != 0):
            good = (xfast != xfast.min()) & (yprosp != yprosp.min())
        else:
            good = np.ones_like(xfast,dtype=bool)

        ## log axes & range
        if par[:3] == 'sfr' or par == 'half_time': 
            axes[i] = prosp_dutils.equalize_axes(axes[i], np.log10(xfast), np.log10(yprosp), dynrange=0.1, line_of_equality=True, log_in_linear=True)
            off,scat = prosp_dutils.offset_and_scatter(np.log10(xfast[good]),np.log10(yprosp[good]),biweight=True)     
            axes[i].set_xscale('log',nonposx='clip',subsx=([1]))
            axes[i].set_yscale('log',nonposy='clip',subsy=([1]))
            axes[i].xaxis.set_minor_formatter(FormatStrFormatter('%2.4g'))
            axes[i].xaxis.set_major_formatter(FormatStrFormatter('%2.4g'))
            axes[i].yaxis.set_minor_formatter(FormatStrFormatter('%2.4g'))
            axes[i].yaxis.set_major_formatter(FormatStrFormatter('%2.4g'))
            axes[i].tick_params('both', pad=3.5, size=3.5, width=1.0, which='both')

        elif 'mass' in par:
            axes[i].set_xlim(8.5,12)
            axes[i].set_ylim(8.5,12)
            axes[i].plot([8.5,12],[8.5,12],linestyle='--',color='0.1',alpha=0.8)
            off,scat = prosp_dutils.offset_and_scatter(xfast[good],yprosp[good],biweight=True)
        else:
            axes[i] = prosp_dutils.equalize_axes(axes[i], xfast, yprosp, dynrange=0.1, line_of_equality=True)
            off,scat = prosp_dutils.offset_and_scatter(xfast[good],yprosp[good],biweight=True)
        
        if par == 'dust2':
            scatunits = ''
        else:
            scatunits = ' dex'

        ### labels
        axes[i].text(0.985,0.09, 'offset='+"{:.2f}".format(off)+scatunits,
                     transform = axes[i].transAxes,horizontalalignment='right')
        axes[i].text(0.985,0.03, 'biweight scatter='+"{:.2f}".format(scat)+scatunits,
                     transform = axes[i].transAxes,horizontalalignment='right')
        axes[i].set_xlabel(flabel + ' ' + parlabels[pnames==par][0])
        axes[i].set_ylabel(plabel + ' ' +  parlabels[pnames==par][0])

    plt.tight_layout()
    plt.savefig(outname,dpi=dpi)
    plt.close()

def prospector_versus_z(data,outname,popts):
    
    fig, axes = plt.subplots(2, 3, figsize = (10,6.66))
    axes = np.ravel(axes)

    toplot = ['fagn','sfr_100','ssfr_100','half_time','massmet_2','dust2']
    for i,par in enumerate(toplot):
        zred = data['fast']['z']
        yprosp, yprosp_up, yprosp_down = data['prosp'][par]['q50'], data['prosp'][par]['q84'], data['prosp'][par]['q16']

        ### clip SFRs
        if par[:3] == 'sfr':
            minimum = minsfr
        elif par == 'ssfr_100':
            minimum = minlogssfr
        else:
            minimum = -np.inf
        yprosp = np.clip(yprosp,minimum,np.inf)
        yprosp_up = np.clip(yprosp_up,minimum,np.inf)
        yprosp_down = np.clip(yprosp_down,minimum,np.inf)

        yerr = prosp_dutils.asym_errors(yprosp, yprosp_up, yprosp_down, log=False)
        axes[i].errorbar(zred,yprosp,yerr=yerr,**popts)
        axes[i].set_xlabel('redshift')
        axes[i].set_ylabel('Prospector '+data['labels'][data['pnames'] == par][0])

        # add tuniv
        if par == 'half_time':
            n = 50
            zred = np.linspace(zred.min(),zred.max(),n)
            tuniv = WMAP9.age(zred).value
            axes[i].plot(zred,tuniv,'--',lw=2,zorder=-1, color=red)
            axes[i].text(zred[n/2]*1.1,tuniv[n/2]*1.1, r't$_{\mathrm{univ}}$',rotation=-50,color=red,weight='bold')

        # logscale
        if par == 'sfr_100' or par == 'fagn':
            axes[i].set_yscale('log',nonposx='clip',subsy=([3]))
            axes[i].yaxis.set_minor_formatter(FormatStrFormatter('%2.4g'))
            axes[i].yaxis.set_major_formatter(FormatStrFormatter('%2.4g'))
    plt.tight_layout()
    plt.savefig(outname,dpi=dpi)
    plt.close()

def phot_residuals(data,outfolder,popts_orig):
    """ two plots: 
    2(chi versus percentiles) by 5 (fields) [obs-frame]
    2(chi versus percentiles) by 1 [rest-frame]
    """
    
    # pull out field & filter names
    fields = np.array([f.split('_')[-1] for f in data['phot_fname']])
    field_names = np.unique(fields)
    filters = np.array([f.split('_')[0] for f in data['phot_fname']])

    # plot stuff
    popts = copy.deepcopy(popts_orig)
    popts['ms'] = 1
    popts['zorder'] = -5
    medopts = {'marker':'o','alpha':0.95,'color':'red','ms': 7,'mec':'k','zorder':5}
    fontsize = 16
    ylim_chi = (-4,4)
    ylim_percentile = (-0.25,0.25)

    # residuals, by field and observed-frame filter
    fig, ax = plt.subplots(5,2, figsize=(8,18))
    for i,field in enumerate(field_names):

        # determine field and filter names
        fidx = fields == field
        fnames = np.unique(filters[fidx])

        # in each field, plot chi and percentile
        ax[i,0].errorbar(data['phot_obslam'][fidx], data['phot_chi'][fidx], **popts)
        ax[i,1].errorbar(data['phot_obslam'][fidx], data['phot_percentile'][fidx], **popts)

        # plot the median for these
        for filter in fnames:
            fmatch = filters[fidx] == filter
            lam = data['phot_obslam'][fidx][fmatch][0]
            ax[i,0].plot(lam, np.median(data['phot_chi'][fidx][fmatch]), **medopts)
            ax[i,1].plot(lam, np.median(data['phot_percentile'][fidx][fmatch]), **medopts)            

        # labels
        for a in ax[i,:]: 
            a.set_xlabel(r'observed wavelength ($\mu$m)',fontsize=fontsize)
            a.set_xscale('log',nonposx='clip',subsx=(2,4))
            a.xaxis.set_minor_formatter(FormatStrFormatter('%2.4g'))
            a.xaxis.set_major_formatter(FormatStrFormatter('%2.4g'))
            a.tick_params('both', pad=3.5, size=3.5, width=1.0, which='both',labelsize=fontsize)
            a.axhline(0, linestyle='--', color='k',lw=2,zorder=3)
            a.text(0.98,0.92,field,fontsize=fontsize,transform=a.transAxes,ha='right')

        ax[i,0].set_ylabel('(f$_{\mathrm{obs}}$-f$_{\mathrm{model}}$)/$\sigma_{\mathrm{obs}}$',fontsize=fontsize)
        ax[i,1].set_ylabel('(f$_{\mathrm{obs}}$-f$_{\mathrm{model}}$)/f$_{\mathrm{obs}}$',fontsize=fontsize)

        ax[i,0].set_ylim(ylim_chi)
        ax[i,1].set_ylim(ylim_percentile)

    plt.tight_layout()
    plt.savefig(outfolder+'residual_by_field.png',dpi=dpi)
    plt.close()
 
    # residuals by rest-frame wavelength
    fig, ax = plt.subplots(1,2, figsize=(10,5))

    ax[0].errorbar(data['phot_restlam'], data['phot_chi'], **popts)
    ax[1].errorbar(data['phot_restlam'], data['phot_percentile'], **popts)

    # plot the median for these
    for filter in fnames:
        x, y, bincount = prosp_dutils.running_median(np.log10(data['phot_restlam']),data['phot_chi'],avg=False,return_bincount=True,nbins=20)
        x, y = x[bincount > nbin_min], y[bincount > nbin_min]
        ax[0].plot(10**x,y, **medopts)

        x, y, bincount = prosp_dutils.running_median(np.log10(data['phot_restlam']),data['phot_percentile'],avg=False,return_bincount=True,nbins=20)
        x, y = x[bincount > nbin_min], y[bincount > nbin_min]
        ax[1].plot(10**x,y, **medopts)

    # labels & scale
    for a in ax: 
        a.set_xlabel(r'rest-frame wavelength ($\mu$m)',fontsize=fontsize)
        a.set_xscale('log',nonposx='clip',subsx=(2,4))
        a.xaxis.set_minor_formatter(FormatStrFormatter('%2.4g'))
        a.xaxis.set_major_formatter(FormatStrFormatter('%2.4g'))
        a.tick_params('both', pad=3.5, size=3.5, width=1.0, which='both',labelsize=fontsize)
        a.axhline(0, linestyle='--', color='k',lw=2,zorder=3)

    ax[0].set_ylabel('(f$_{\mathrm{obs}}$-f$_{\mathrm{model}}$)/$\sigma_{\mathrm{obs}}$',fontsize=fontsize)
    ax[1].set_ylabel('(f$_{\mathrm{obs}}$-f$_{\mathrm{model}}$)/f$_{\mathrm{obs}}$',fontsize=fontsize)

    ax[0].set_ylim(ylim_chi)
    ax[1].set_ylim(ylim_percentile)

    plt.tight_layout()
    plt.savefig(outfolder+'residual_restframe.png',dpi=dpi)
    plt.close()


def mass_metallicity_relationship(data,outname,popts):
    
    # plot information
    fig, axes = plt.subplots(2, 3, figsize = (9,6))
    fig.subplots_adjust(wspace=0.0,hspace=0.0)
    axes = np.ravel(axes)
    xlim = (9,11.5)
    ylim = (-2,0.5)

    # redshift bins + colors
    zbins = np.linspace(0.5,3,6)
    nbins = len(zbins)-1
    cmap = prosp_dutils.get_cmap(nbins,cmap='plasma')

    # z=0 mass-met
    massmet = np.loadtxt(os.getenv('APPS')+'/prospector_alpha/data/gallazzi_05_massmet.txt')
    for i in range(nbins):

        # the main show
        idx = (data['fast']['z'] > zbins[i]) & (data['fast']['z'] <= zbins[i+1])
        xm, xm_up, xm_down = [data['prosp']['stellar_mass'][q][idx] for q in ['q50','q84','q16']]
        yz, yz_up, yz_down = [data['prosp']['massmet_2'][q][idx] for q in ['q50','q84','q16']]
        xerr = prosp_dutils.asym_errors(xm, xm_up, xm_down)
        yerr = prosp_dutils.asym_errors(yz, yz_up, yz_down)
        axes[i+1].errorbar(xm,yz,yerr=yerr,xerr=xerr,**popts)

        # labels
        if i > 1:
            axes[i+1].set_xlabel('log(M/M$_{\odot}$)')
        else:
            for tl in axes[i+1].get_xticklabels():tl.set_visible(False)
        if i == 2:
            axes[i+1].set_ylabel('log(Z/Z$_{\odot}$)')
            axes[0].set_ylabel('log(Z/Z$_{\odot}$)')
            for tl in axes[0].get_xticklabels():tl.set_visible(False)
        else:
            for tl in axes[i+1].get_yticklabels():tl.set_visible(False)

        # z=0 relationship
        lw = 1.5
        color = '0.5'
        axes[i+1].plot(massmet[:,0], massmet[:,1], color=color, lw=lw, linestyle='--', zorder=-1, label='Gallazzi et al. 2005')
        axes[i+1].plot(massmet[:,0],massmet[:,2], color=color, lw=lw, zorder=-1)
        axes[i+1].plot(massmet[:,0],massmet[:,3], color=color, lw=lw, zorder=-1)
        if i == 0:
            axes[0].plot(massmet[:,0], massmet[:,1], color=color, lw=lw, linestyle='--', zorder=-1, label='Gallazzi et al. 2005')
            axes[0].plot(massmet[:,0],massmet[:,2], color=color, lw=lw, zorder=-1)
            axes[0].plot(massmet[:,0],massmet[:,3], color=color, lw=lw, zorder=-1)


        # running median
        nbin_min = 5
        x, y, bincount = prosp_dutils.running_median(xm,yz,avg=False,return_bincount=True)
        x, y = x[bincount > nbin_min], y[bincount > nbin_min]
        axes[i+1].plot(x, y, color=cmap(i),lw=3,alpha=0.95,label="{0:.1f}".format(zbins[i])+'<z<'+"{0:.1f}".format(zbins[i+1]))
        axes[0].plot(x, y, color=cmap(i),lw=3,alpha=0.95,label="{0:.1f}".format(zbins[i])+'<z<'+"{0:.1f}".format(zbins[i+1]))

    for a in axes:
        a.set_xlim(xlim)
        a.set_ylim(ylim)

    axes[0].legend(loc=4, prop={'size':8},
                   scatterpoints=1,fancybox=True)

    plt.savefig(outname,dpi=dpi)
    plt.close()

def deltam_with_redshift(fast, prosp, z, outname, filename=None):

    # plot
    fig, ax = plt.subplots(1, 1, figsize = (3.5,3.5))

    # define quantities
    par = 'stellar_mass'
    mfast = fast[par]
    mprosp = prosp[par]['q50']
    if type(mfast) == dict:
        mfast = mfast['q50']
    delta_m = mprosp-mfast

    # binning
    zbins = np.linspace(0.5,3,6)
    nbins = len(zbins)-1
    cmap = prosp_dutils.get_cmap(nbins,cmap='plasma')
    xmed, ymed, zmed = [], [], []
    for i in range(nbins):
        inbin = (z > zbins[i]) & (z <= zbins[i+1])
        if inbin.sum() == 0:
            continue
        x,y = mfast[inbin], delta_m[inbin]

        x, y, bincount = prosp_dutils.running_median(x,y,avg=False,return_bincount=True)
        x, y = x[bincount > nbin_min], y[bincount > nbin_min]
        ax.plot(x, y, 'o-', color=cmap(i),lw=2,alpha=0.95,label="{0:.1f}".format(zbins[i])+'<z<'+"{0:.1f}".format(zbins[i+1]))
        xmed += x.tolist()
        ymed += y.tolist()
        zmed += [(zbins[i]+zbins[i+1])/2.]*len(y)

    ax.axhline(0, linestyle='--', color='red', lw=2,zorder=-1)
    ax.set_xlabel(r'log(M$_{\mathrm{FAST}}$/M$_{\odot}$)')
    ax.set_ylabel(r'log(M$_{\mathrm{Prosp}}$/M$_{\mathrm{FAST}}$)')
    ax.legend(prop={'size':8}, scatterpoints=1,fancybox=True)
    ax.set_ylim(-0.5,0.5)

    if filename is not None:
        out = {'fast_mass': xmed, 'log_mprosp_mfast': ymed, 'z': zmed}
        hickle.dump(out,open(filename, "w"))

    plt.tight_layout()
    plt.savefig(outname+'.png',dpi=dpi)
    plt.close()


def uvir_comparison(data, outname, popts, model_uvir = False, ssfr=False, filename=None):
    """ plot sSFR_prosp versus sSFR_UVIR against a variety of variables
    what drives the differences?
    model_uvir: instead of using LIR + LUV from observations + templates,
    calculate directly from Prospector model
    """

    # define flag where a UV+IR SFR is observed
    good = (data['uv_sfr'] > -1) & (data['ir_sfr'] > -1)

    # load up Prospector data
    if ssfr:
        prosp, prosp_up, prosp_down = [np.clip(10**data['prosp']['ssfr_100'][q][good],minssfr,np.inf) for q in ['q50','q84','q16']]
        prosp_label = 'sSFR$_{\mathrm{Prosp}}$'
    else:
        prosp, prosp_up, prosp_down = [np.clip(data['prosp']['sfr_100'][q][good],minsfr,np.inf) for q in ['q50','q84','q16']]
        prosp_label = 'SFR$_{\mathrm{Prosp}}$'
    prosp_err = prosp_dutils.asym_errors(prosp, prosp_up, prosp_down, log=False)

    try:
        qpah, qpah_up, qpah_down = [data['prosp']['duste_qpah'][q][good] for q in ['q50','q84','q16']]
        qpah_err = prosp_dutils.asym_errors(qpah, qpah_up, qpah_down, log=False)
    except:
        qpah, qpah_err = None, None
    fagn, fagn_up, fagn_down = [data['prosp']['fagn'][q][good] for q in ['q50','q84','q16']]
    fagn_err = prosp_dutils.asym_errors(fagn, fagn_up, fagn_down, log=False)

    # calculate UV+IR values, and ratios
    # different if we use model or observed UVIR
    # and if we use sSFR or SFR
    if model_uvir:

        if model_uvir == 'true_LIR':
            lbl = 'model_uvir_truelir'
            ratio_lbl = 'sfr_ratio_truelir'
            append_lbl = ' [True L$_{\mathrm{IR}}$]'
        else:
            lbl = 'model_uvir'
            ratio_lbl = 'sfr_ratio'
            append_lbl = ' [L$_{\mathrm{IR}}$ from MIPS]'

        if ssfr:
            uvir = np.clip(data['prosp'][lbl+'_ssfr']['q50'][good],minssfr,np.inf)
            uvir_err = prosp_dutils.asym_errors(uvir, 
                                                np.clip(data['prosp'][lbl+'_ssfr']['q84'][good],minssfr,np.inf), 
                                                np.clip(data['prosp'][lbl+'_ssfr']['q16'][good],minssfr,np.inf))
            ratio, ratio_err = np.log10(prosp/uvir), None
            uvir_label = 'sSFR$_{\mathrm{UV+IR,mod}}$'+append_lbl
        else:
            uvir = np.clip(data['prosp'][lbl+'_sfr']['q50'][good],minsfr,np.inf)
            uvir_err = prosp_dutils.asym_errors(uvir,
                                                np.clip(data['prosp'][lbl+'_sfr']['q84'][good],minsfr,np.inf), 
                                                np.clip(data['prosp'][lbl+'_sfr']['q16'][good],minsfr,np.inf))
            ratio, ratio_up, ratio_down = [data['prosp'][ratio_lbl]['q50'][good] for q in ['q50','q84','q16']]
            ratio_err = prosp_dutils.asym_errors(ratio, ratio_up, ratio_down)
            uvir_label = 'SFR$_{\mathrm{UV+IR,mod}}$'+append_lbl
    else:
        if ssfr:
            #uvir = np.clip(data['uvir_sfr'][good]/10**data['fast']['stellar_mass'][good],minssfr,np.inf)
            uvir = np.clip(data['uvir_sfr'][good]/10**data['prosp']['stellar_mass']['q50'][good],minssfr,np.inf)
            uvir_err = None     
            ratio, ratio_err = np.log10(prosp/uvir), None
            uvir_label = 'SFR$_{\mathrm{UV+IR,obs}}$/M$_{\mathrm{prosp}}$'
        else:
            uvir = np.clip(data['uvir_sfr'][good],minsfr,np.inf)
            uvir_err = None
            ratio, ratio_up, ratio_down = np.log10(prosp/uvir), np.log10(prosp_up/uvir), np.log10(prosp_down/uvir)
            ratio_err = prosp_dutils.asym_errors(ratio, ratio_up, ratio_down)
            uvir_label = 'SFR$_{\mathrm{UV+IR,obs}}$'

    # remove clips from scatter calculations
    no_min = (uvir != minssfr) & (prosp != minssfr)

    # plot geometry
    fig, ax = plt.subplots(1, 2, figsize = (8,4))
    ax = np.ravel(ax)

    # sSFR_prosp versus sSFR_uvir
    ax[0].errorbar(uvir, prosp, xerr=uvir_err, yerr=prosp_err, **popts)

    ax[0].set_xlabel(uvir_label)
    ax[0].set_ylabel(prosp_label)

    ax[0].set_yscale('log')
    ax[0].set_xscale('log')

    off,scat = prosp_dutils.offset_and_scatter(np.log10(uvir[no_min]),np.log10(prosp[no_min]),biweight=True)
    ax[0].text(0.05,0.94, 'offset=' + "{:.2f}".format(off) + ' dex', transform = ax[0].transAxes)
    ax[0].text(0.05,0.89, 'biweight scatter=' + "{:.2f}".format(scat) + ' dex', transform = ax[0].transAxes)

    min, max = np.min([prosp.min(),uvir.min()]), np.max([prosp.max(),uvir.max()])
    ax[0].axis([min,max,min,max])
    ax[0].plot([min,max],[min,max],'--', color='red', zorder=2)

    # sSFR_prosp versus (SFR_prosp / SFR_uvir) in redshift bins
    zbins = np.linspace(0.5,3,6)
    nbins = len(zbins)-1
    cmap = prosp_dutils.get_cmap(nbins,cmap='plasma')
    xmed, ymed, zmed = [], [], []
    for i in range(nbins):
        inbin = (data['fast']['z'][good] > zbins[i]) & (data['fast']['z'][good] <= zbins[i+1])
        if inbin.sum() == 0:
            continue
        
        if ssfr == True:
            bins = np.linspace(-11,-8,6)
        else:
            bins = None

        x,y = np.log10(uvir[inbin]), ratio[inbin]
        x, y, bincount = prosp_dutils.running_median(x,y,avg=False,return_bincount=True, bins=bins)
        x, y = x[bincount > nbin_min], y[bincount > nbin_min]
        ax[1].plot(x, y, 'o-', color=cmap(i),lw=2,alpha=0.95,label="{0:.1f}".format(zbins[i])+'<z<'+"{0:.1f}".format(zbins[i+1]))
        xmed += x.tolist()
        ymed += y.tolist()
        zmed += [(zbins[i]+zbins[i+1])/2.]*len(y)

    ax[1].axhline(0, linestyle='--', color='red', lw=2,zorder=-1)
    ax[1].set_xlabel('log('+prosp_label+')')
    ax[1].set_ylabel(r'log('+prosp_label+'/'+uvir_label+')')
    ax[1].legend(prop={'size':8}, scatterpoints=1,fancybox=True)
    ax[1].set_ylim(-1.5,1.5)

    plt.tight_layout()
    plt.savefig(outname+'.png',dpi=dpi)
    plt.close()

    if filename is not None:
        out = {'log_ssfruvir_mfast': xmed, 'log_ssfrprosp_ssfruvir': ymed, 'z': zmed}
        hickle.dump(out,open(filename, "w"))

    # there is some ~complex if-logic in case Q_PAH is not a variable here
    if ratio is not None:
        if qpah is not None:
            fig, ax = plt.subplots(1, 2, figsize = (8,4))
            ax[1].errorbar(qpah, ratio, xerr=qpah_err, yerr=ratio_err, **popts)
            ax[1].set_ylim(-1,1)
            ax[1].set_xlabel(data['labels'][data['pnames'] == 'duste_qpah'][0])
        else:
            fig, ax = plt.subplots(1,1, figsize = (4,4))
            ax = [ax] # so indexing works
        ax[0].errorbar(fagn, ratio, xerr=fagn_err, yerr=ratio_err, **popts)
        for i, a in enumerate(ax):
            a.set_xscale('log', subsx=(1,3))
            a.xaxis.set_minor_formatter(FormatStrFormatter('%2.4g'))
            a.xaxis.set_major_formatter(FormatStrFormatter('%2.4g'))
            for tl in a.get_xticklabels():tl.set_visible(False)
            a.tick_params('both', pad=3.5, size=3.5, width=1.0, which='both')

            a.set_ylabel(r'log('+prosp_label+'/'+uvir_label+')')
            a.axhline(0, linestyle='--', color='red',lw=2,zorder=-1)
        
        ax[0].set_xlabel(data['labels'][data['pnames'] == 'fagn'][0])
        ax[0].set_ylim(-3,3)

        plt.tight_layout()
        plt.savefig(outname+'_par_variation.png',dpi=dpi)
        plt.close()





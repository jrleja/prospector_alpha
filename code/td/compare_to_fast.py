import numpy as np
import matplotlib.pyplot as plt
import os, hickle, td_io
from prosp_dutils import generate_basenames,av_to_dust2,asym_errors,equalize_axes,offset_and_scatter,exp_decl_sfh_half_time,sfr_uvir
from matplotlib.ticker import FormatStrFormatter
from prospector_io import load_prospector_extra
from astropy.cosmology import WMAP9
from dynesty.plotting import _quantile as weighted_quantile
plt.ioff()

popts = {'fmt':'o', 'capthick':1.5,'elinewidth':1.5,'ms':9,'alpha':0.8,'color':'0.3','markeredgecolor':'k'}
red = '#FF3D0D'
dpi = 120

minlogssfr = -14
minssfr = 1e-14
minsfr = 0.001

def collate_data(runname, runname_fast, filename=None, regenerate=False):
    
    ### if it's already made, load it and give it back
    # else, start with the making!
    if os.path.isfile(filename) and regenerate == False:
        with open(filename, "r") as f:
            outdict=hickle.load(f)
            return outdict

    ### define output containers
    parlabels = [r'log(M$_{\mathrm{stellar}}$/M$_{\odot}$)', 'SFR [M$_{\odot}$/yr]',
                 r'diffuse dust optical depth', r'log(sSFR) [yr$^-1$]',
                 r"t$_{\mathrm{half-mass}}$ [Gyr]", r'log(Z/Z$_{\odot}$)', r'Q$_{\mathrm{PAH}}$',
                 r'f$_{\mathrm{AGN}}$', 'dust index']
    fnames = ['stellar_mass','sfr_100','dust2','ssfr_100','half_time'] # take for fastmimic too
    pnames = fnames+['massmet_2', 'duste_qpah', 'fagn', 'dust_index'] # already calculated in Prospector
    enames = ['model_uvir_sfr', 'model_uvir_ssfr', 'sfr_ratio'] # must calculate here
    outprosp, outprosp_fast, outfast, outlabels = {},{},{},{}
    sfr_100_uvir, sfr_100_uv, sfr_100_ir = [], [], []
    outfast['z'] = []
    for i,par in enumerate(pnames+enames):
        
        ### look for it in FAST
        if par in fnames:
            outfast[par] = []

        ### if it's in FAST, it's in Prospector-FAST
        if par in outfast.keys():
            outprosp_fast[par] = {}
            outprosp_fast[par]['q50'],outprosp_fast[par]['q84'],outprosp_fast[par]['q16'] = [],[],[]

        ### it's always in Prospector
        outprosp[par] = {}
        outprosp[par]['q50'],outprosp[par]['q84'],outprosp[par]['q16'] = [],[],[]
    
    ### fill output containers
    basenames, _, _ = generate_basenames(runname)
    if runname_fast is not None:
        basenames_fast, _, _ = generate_basenames(runname_fast)
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
            prosp = load_prospector_extra(name)
            if runname_fast is not None:
                prosp_fast = load_prospector_extra(basenames_fast[i])
        except:
            print name.split('/')[-1]+' failed to load. skipping.'
            continue

        ### prospector first
        for par in pnames:
            
            ### switch to tell between 'thetas' and 'extras'
            loc = 'thetas'
            if par in prosp['extras'].keys():
                loc = 'extras'
            
            ### fill it up
            for q in ['q16','q50','q84']:
                x = prosp[loc][par][q]
                if par == 'stellar_mass' or par == 'ssfr_100':
                    x = np.log10(x)
                outprosp[par][q].append(x)
        
        # a little extra
        uvir_chain = sfr_uvir(prosp['extras']['lir']['chain'],prosp['extras']['luv']['chain'])
        for q in ['q16','q50','q84']: 
            outprosp['model_uvir_sfr'][q] += [weighted_quantile(uvir_chain, np.array([float(q[1:])/100.]),weights=prosp['weights'])[0]]
            outprosp['model_uvir_ssfr'][q] += [weighted_quantile(uvir_chain/prosp['extras']['stellar_mass']['chain'], 
                                                               np.array([float(q[1:])/100.]),weights=prosp['weights'])[0]]
            outprosp['sfr_ratio'][q] += [weighted_quantile(np.log10(uvir_chain/prosp['extras']['sfr_100']['chain']), 
                                                         np.array([float(q[1:])/100.]),weights=prosp['weights'])[0]]

        ### prospector-fast
        if runname_fast is not None:
            for par in pnames:

                ### switch to tell between 'thetas', 'extras', and 'NOT THERE'
                if par in prosp_fast['thetas'].keys():
                    loc = 'thetas'
                elif par in prosp_fast['extras'].keys():
                    loc = 'extras'
                else:
                    continue

                ### fill it up
                for q in ['q16','q50','q84']:
                    x = prosp_fast[loc][par][q]
                    if par == 'stellar_mass' or par == 'ssfr_100':
                        x = np.log10(x)
                    outprosp_fast[par][q].append(x)
        
        ### now FAST, UV+IR SFRs
        # find correct field, find ID match
        fidx = allfields.index(field[i])
        fast = fastlist[fidx]
        uvir = uvirlist[fidx]
        f_idx = fast['id'] == int(name.split('_')[-1])
        u_idx = uvir['phot_id'] == int(name.split('_')[-1])

        # fill it up
        outfast['stellar_mass'] += [fast['lmass'][f_idx][0]]
        outfast['dust2'] += [av_to_dust2(fast['Av'][f_idx][0])]
        outfast['sfr_100'] += [10**fast['lsfr'][f_idx][0]]
        outfast['half_time'] += [exp_decl_sfh_half_time(10**fast['lage'][f_idx][0],10**fast['ltau'][f_idx][0])/1e9]
        outfast['z'] += [fast['z'][f_idx][0]]
        sfr_100_uvir += [uvir['sfr'][u_idx][0]]
        sfr_100_ir += [uvir['sfr_IR'][u_idx][0]]
        sfr_100_uv += [uvir['sfr_UV'][u_idx][0]]

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
           'uv_sfr': np.array(sfr_100_uv)
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

    # if we have FAST-mimic runs, do a thorough comparison
    # else just do Prospector-FAST
    if runname_fast is not None:
        fast_comparison(data['fast'],data['prosp_fast'],data['labels'],data['pnames'],outfolder+'fast_to_fastmimic_comparison.png')
        fast_comparison(data['prosp_fast'],data['prosp'],data['labels'],data['pnames'],outfolder+'fastmimic_to_palpha_comparison.png')
    else:
        fast_comparison(data['fast'],data['prosp'],data['labels'],data['pnames'],outfolder+'fast_to_palpha_comparison.png')

    prospector_versus_z(data,outfolder+'prospector_versus_z.png')
    uvir_comparison(data,outfolder+'uvir_comparison')
    uvir_comparison(data,outfolder+'uvir_comparison_model', model_uvir = True)

def fast_comparison(fast,prosp,parlabels,pnames,outname):
    
    fig, axes = plt.subplots(2, 2, figsize = (6,6))
    axes = np.ravel(axes)

    for i,par in enumerate(['stellar_mass','sfr_100','dust2','half_time']):

        ### clip SFRs
        if par[:3] == 'sfr':
            minimum = minsfr
        else:
            minimum = -np.inf

        ### grab data
        try:
            xfast = np.clip(fast[par]['q50'],minimum,np.inf)
            xfast_up = np.clip(fast[par]['q84'],minimum,np.inf)
            xfast_down = np.clip(fast[par]['q16'],minimum,np.inf)
            xerr = asym_errors(xfast, xfast_up, xfast_down, log=False)
        except:
            xfast = np.clip(fast[par],minimum,np.inf)
            xerr = None

        yprosp = np.clip(prosp[par]['q50'],minimum,np.inf)
        yprosp_up = np.clip(prosp[par]['q84'],minimum,np.inf)
        yprosp_down = np.clip(prosp[par]['q16'],minimum,np.inf)
        yerr = asym_errors(yprosp, yprosp_up, yprosp_down, log=False)

        ### plot
        axes[i].errorbar(xfast,yprosp,xerr=xerr,yerr=yerr,**popts)

        ### if we have some enforced minimum, don't include in scatter calculation
        if ((xfast == xfast.min()).sum()-1 != 0) | ((yprosp == yprosp.min()).sum()-1 != 0):
            good = (xfast != xfast.min()) & (yprosp != yprosp.min())
        else:
            good = np.ones_like(xfast,dtype=bool)

        ## log axes & range
        if par[:3] == 'sfr' or par == 'half_time': 
            axes[i] = equalize_axes(axes[i], np.log10(xfast), np.log10(yprosp), dynrange=0.1, line_of_equality=True, log_in_linear=True)
            off,scat = offset_and_scatter(np.log10(xfast[good]),np.log10(yprosp[good]),biweight=True)     
            axes[i].set_xscale('log',nonposx='clip',subsx=([3]))
            axes[i].set_yscale('log',nonposy='clip',subsy=([3]))
            axes[i].xaxis.set_minor_formatter(FormatStrFormatter('%2.3g'))
            axes[i].xaxis.set_major_formatter(FormatStrFormatter('%2.3g'))
            axes[i].yaxis.set_minor_formatter(FormatStrFormatter('%2.3g'))
            axes[i].yaxis.set_major_formatter(FormatStrFormatter('%2.3g'))
            axes[i].tick_params('both', pad=3.5, size=3.5, width=1.0, which='both')

        else:
            axes[i] = equalize_axes(axes[i], xfast, yprosp, dynrange=0.1, line_of_equality=True)
            off,scat = offset_and_scatter(xfast[good],yprosp[good],biweight=True)
        
        if par == 'dust2':
            scatunits = ''
        else:
            scatunits = ' dex'

        ### labels
        axes[i].text(0.95,0.12, 'offset='+"{:.2f}".format(off)+scatunits,
                     transform = axes[i].transAxes,horizontalalignment='right')
        axes[i].text(0.95,0.06, 'biweight scatter='+"{:.2f}".format(scat)+scatunits,
                     transform = axes[i].transAxes,horizontalalignment='right')
        axes[i].set_xlabel('FAST '+ parlabels[pnames==par][0])
        axes[i].set_ylabel('Prospector '+  parlabels[pnames==par][0])

    plt.tight_layout()
    plt.savefig(outname,dpi=dpi)
    plt.close()

def prospector_versus_z(data,outname):
    
    fig, axes = plt.subplots(2, 3, figsize = (10,6.66))
    axes = np.ravel(axes)

    toplot = ['stellar_mass','sfr_100','ssfr_100','half_time','massmet_2','dust2']
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

        yerr = asym_errors(yprosp, yprosp_up, yprosp_down, log=False)
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
        if par == 'sfr_100':
            axes[i].set_yscale('log',nonposx='clip',subsy=([3]))
            axes[i].yaxis.set_minor_formatter(FormatStrFormatter('%2.3g'))
            axes[i].yaxis.set_major_formatter(FormatStrFormatter('%2.3g'))
    plt.tight_layout()
    plt.savefig(outname,dpi=dpi)
    plt.close()

def uvir_comparison(data, outname, model_uvir = False):
    """ plot sSFR_prosp versus sSFR_UVIR against a variety of variables
    what drives the differences?
    model_uvir: instead of using LIR + LUV from observations + templates,
    calculate directly from Prospector model
    """

    # load up data
    sfr_prosp, sfr_prosp_up, sfr_prosp_down = data['prosp']['sfr_100']['q50'], data['prosp']['sfr_100']['q84'], data['prosp']['sfr_100']['q16']
    ssfr_prosp, ssfr_prosp_up, ssfr_prosp_down = 10**data['prosp']['ssfr_100']['q50'], 10**data['prosp']['ssfr_100']['q84'], 10**data['prosp']['ssfr_100']['q16']

    logzsol = data['prosp']['massmet_2']['q50']
    halftime = data['prosp']['half_time']['q50']

    qpah, qpah_up, qpah_down = data['prosp']['duste_qpah']['q50'], data['prosp']['duste_qpah']['q84'], data['prosp']['duste_qpah']['q16']
    fagn, fagn_up, fagn_down = data['prosp']['fagn']['q50'], data['prosp']['fagn']['q84'], data['prosp']['fagn']['q16']

    # clip to minimum
    sfr_prosp = np.clip(sfr_prosp,minssfr,np.inf)
    sfr_prosp_up = np.clip(sfr_prosp_up,minssfr,np.inf)
    sfr_prosp_down = np.clip(sfr_prosp_down,minssfr,np.inf)

    ssfr_prosp = np.clip(ssfr_prosp,minssfr,np.inf)
    ssfr_prosp_up = np.clip(ssfr_prosp_up,minssfr,np.inf)
    ssfr_prosp_down = np.clip(ssfr_prosp_down,minssfr,np.inf)

    # calculate UV_IR SFRs, and ratios
    # different if we use model or observed UVIR SFRs
    if model_uvir == False:
        ssfr_uvir = np.clip(data['uvir_sfr']/10**data['prosp']['stellar_mass']['q50'],minssfr,np.inf)
        sfr_uvir = np.clip(data['uvir_sfr'],minsfr,np.inf)
        ssfr_uvir_err, sfr_uvir_err = None, None
        sfr_ratio, sfr_ratio_up, sfr_ratio_down = np.log10(sfr_prosp/sfr_uvir), np.log10(sfr_prosp_up/sfr_uvir), np.log10(sfr_prosp_down/sfr_uvir)
        ssfr_label = 'sSFR$_{\mathrm{UVIR,obs}}$'
    else:
        sfr_uvir = data['prosp']['model_uvir_sfr']['q50']
        sfr_uvir_err = asym_errors(sfr_uvir,data['prosp']['model_uvir_sfr']['q84'], data['prosp']['model_uvir_sfr']['q16'])
        ssfr_uvir = data['prosp']['model_uvir_ssfr']['q50']
        ssfr_uvir_err = asym_errors(ssfr_uvir,data['prosp']['model_uvir_ssfr']['q84'], data['prosp']['model_uvir_ssfr']['q16'])
        sfr_ratio, sfr_ratio_up, sfr_ratio_down = data['prosp']['sfr_ratio']['q50'], data['prosp']['sfr_ratio']['q84'], data['prosp']['sfr_ratio']['q16']
        ssfr_label = 'sSFR$_{\mathrm{UVIR,mod}}$'

    # errors
    # also define flag where minimum was enforced 
    # these aren't used in offset/scatter calculations
    good = (sfr_uvir != minsfr) & (sfr_prosp != minsfr)
    good = (ssfr_uvir != minssfr) & (ssfr_prosp != minssfr)
    sfr_err = asym_errors(sfr_prosp, sfr_prosp_up, sfr_prosp_down, log=False)
    ssfr_err = asym_errors(ssfr_prosp, ssfr_prosp_up, ssfr_prosp_down, log=False)
    sfr_ratio_err = asym_errors(sfr_ratio[good], sfr_ratio_up[good], sfr_ratio_down[good], log=False)
    qpah_err = asym_errors(qpah[good], qpah_up[good], qpah_down[good], log=False)
    fagn_err = asym_errors(fagn[good], fagn_up[good], fagn_down[good], log=False)

    # plot geometry
    if qpah.sum() != 0:
        fig, ax = plt.subplots(2, 2, figsize = (7,6.5))
    else:
        fig, ax = plt.subplots(1,2,figsize=(12.5,6))
    ax = np.ravel(ax)

    # sSFR_prosp versus sSFR_uvir
    ax[0].errorbar(ssfr_uvir, ssfr_prosp, xerr=ssfr_uvir_err, yerr=ssfr_err, **popts)

    ax[0].set_xlabel(ssfr_label)
    ax[0].set_ylabel('sSFR$_{\mathrm{Prosp}}$')

    ax[0].set_yscale('log')
    ax[0].set_xscale('log')

    off,scat = offset_and_scatter(np.log10(ssfr_uvir[good]),np.log10(ssfr_prosp[good]),biweight=True)
    ax[0].text(0.05,0.94, 'offset=' + "{:.2f}".format(off) + ' dex', transform = ax[0].transAxes)
    ax[0].text(0.05,0.89, 'biweight scatter=' + "{:.2f}".format(scat) + ' dex', transform = ax[0].transAxes)
    min, max = np.min(ax[0].axis())*0.5, np.max(ax[0].axis())*2
    ax[0].axis([min,max,min,max])
    ax[0].plot([min,max],[min,max],'--', color='red', zorder=2)

    # sSFR_prosp versus (sSFR_prosp / sSFR_uvir)
    ax[1].errorbar(ssfr_prosp[good], np.log10(ssfr_prosp[good]/ssfr_uvir[good]),
                   xerr=[ssfr_err[0][good],ssfr_err[1][good]], yerr=sfr_ratio_err, **popts)
    pts = ax[1].scatter(ssfr_prosp[good], np.log10(ssfr_prosp[good]/ssfr_uvir[good]), marker='o', c=halftime[good],
                  cmap=plt.cm.plasma,s=75,zorder=10, edgecolors='k')
    cbar = fig.colorbar(pts, ax=ax[1])
    cbar.set_label(r'half-mass time [Gyr]')

    ax[1].set_xlabel('sSFR$_{\mathrm{Prosp}}$')
    ax[1].set_ylabel('log(sSFR$_{\mathrm{Prosp}}$/' + ssfr_label + ')',labelpad=2)
    ax[1].axhline(0, linestyle='--', color='red',lw=2,zorder=-1)

    ax[1].set_xscale('log')

    # if we are also fitting for (fagn, qpah), throw them in!
    if ax.shape[0] > 2:
        ax[2].errorbar(qpah[good], sfr_ratio[good], xerr=qpah_err, yerr=sfr_ratio_err, **popts)
        ax[3].errorbar(fagn[good], sfr_ratio[good], xerr=fagn_err, yerr=sfr_ratio_err, **popts)
        subsx = [(1,3),([1])]
        for i, a in enumerate(ax[2:]):
            a.set_xscale('log', subsx=subsx[i])
            a.xaxis.set_minor_formatter(FormatStrFormatter('%2.2g'))
            a.xaxis.set_major_formatter(FormatStrFormatter('%2.2g'))
            for tl in a.get_xticklabels():tl.set_visible(False)
            a.tick_params('both', pad=3.5, size=3.5, width=1.0, which='both')

            a.set_ylabel(r'log(SFR$_{\mathrm{Prosp}}$/'+ssfr_label)
            a.axhline(0, linestyle='--', color='red',lw=2,zorder=-1)
        ax[2].set_xlabel(data['labels'][data['pnames'] == 'duste_qpah'][0])
        ax[3].set_xlabel(data['labels'][data['pnames'] == 'fagn'][0])

    plt.tight_layout()
    plt.savefig(outname+'.png',dpi=dpi)
    plt.close()







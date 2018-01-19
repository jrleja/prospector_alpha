import numpy as np
import matplotlib.pyplot as plt
import os, hickle, td_io, prosp_dutils, copy
from prospector_io import load_prospector_data, load_prospector_extra
from astropy.cosmology import WMAP9
from prospect.models import sedmodel
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from dynesty.plotting import _quantile as weighted_quantile
from fix_ir_sed import mips_to_lir

plt.ioff()

popts = {'fmt':'o', 'capthick':1.5,'elinewidth':1.5,'ms':9,'alpha':0.8,'color':'0.3'}
red = '#FF3D0D'
dpi = 160 # plot resolution
cmap = 'cool'
minsfr, minssfr = 0.01, 1e-13
ms, s = 0.5, 1 # symbol sizes
alpha, alpha_scat = 0.1, 0.3

def collate_data(runname, filename=None, regenerate=False, nobj=None, **opts):
    """must rewrite this such that NO CHAINS are saved!
    want to return AGN_FRAC and OLD_FRAC
    which is (SFR_AGN) / (SFR_UVIR_LIR_PROSP), (SFR_OLD) / (SFR_UVIR_LIR_PROSP)
    also compare (SFR_UVIR_PROSP) to (SFR_PROSP)
    """

    ### if it's already made, load it and give it back
    # else, start with the making!
    if os.path.isfile(filename) and regenerate == False:
        with open(filename, "r") as f:
            outdict=hickle.load(f)

        return outdict

    ### define output containers
    out = {}
    qvals = ['q50','q84','q16']
    for par in ['sfr_uvir_truelir_prosp', 'sfr_uvir_lirfrommips_prosp', 'sfr_prosp', 'ssfr_prosp','avg_age', \
                'logzsol', 'fagn','agn_heating_fraction','old_star_heating_fraction', 'young_star_heating_fraction']:
        out[par] = {}
        for q in qvals: out[par][q] = []
    for par in ['sfr_uvir_obs','objname']: out[par] = []

    ### fill output containers
    basenames, _, _ = prosp_dutils.generate_basenames(runname)
    if nobj is not None:
        basenames = basenames[:nobj]
    field = [name.split('/')[-1].split('_')[0] for name in basenames]

    ### load necessary information
    uvirlist = []
    allfields = np.unique(field).tolist()
    for f in allfields:
        uvirlist.append(td_io.load_mips_data(f))

    ### iterate over items
    for i, name in enumerate(basenames):

        # load output from fit
        try:
            res, _, model, prosp = load_prospector_data(name)
        except:
            print name.split('/')[-1]+' failed to load. skipping.'
            continue
        if prosp is None:
            continue

        out['objname'] += [name.split('/')[-1]]
        print 'loaded ' + out['objname'][-1]

        # Variable model parameters

        # Variable + nonvariable model parameters ('extras')
        for q in qvals: 
            out['fagn'][q] += [prosp['thetas']['fagn'][q]]
            out['logzsol'][q] += [prosp['thetas']['massmet_2'][q]]
            out['sfr_prosp'][q] += [prosp['extras']['sfr_100'][q]]
            out['ssfr_prosp'][q] += [prosp['extras']['ssfr_100'][q]]
            out['avg_age'][q] += [prosp['extras']['avg_age'][q]]

        # observed UV+IR SFRs
        # find correct field, find ID match
        u_idx = uvirlist[allfields.index(field[i])]['id'].astype(int) == int(out['objname'][-1].split('_')[-1])
        out['sfr_uvir_obs'] += [uvirlist[allfields.index(field[i])]['sfr'][u_idx][0]]

        # pull out relevant quantities
        # "AGN" luminosities are old+young stars, NO AGN
        # "young" luminosities are young stars, NO AGN OR OLD STARS
        lir_agn, luv_agn = prosp['extras']['lir_agn']['chain'], prosp['extras']['luv_agn']['chain']
        lir_tot, luv_tot = prosp['extras']['lir']['chain'], prosp['extras']['luv']['chain']
        lir_young, luv_young = prosp['extras']['lir_young']['chain'], prosp['extras']['luv_young']['chain']

        # Ratios and calculations
        # here we calculate SFR(UV+IR) with LIR + LUV from:
        # (a) the 'true' L_IR and L_UV in the model
        # (b) true L_IR with only heating from AGN
        # (c) true L_IR with only heating from old stars
        # (c) true L_IR with only heating from young stars
        sfr_uvir_truelir_prosp = prosp_dutils.sfr_uvir(lir_tot,luv_tot)
        agn_heating_fraction = 1-prosp_dutils.sfr_uvir(lir_agn,luv_agn) / sfr_uvir_truelir_prosp
        young_star_heating_fraction = prosp_dutils.sfr_uvir(lir_young, luv_young) / sfr_uvir_truelir_prosp
        old_star_heating_fraction = 1- young_star_heating_fraction - agn_heating_fraction

        # we also calculate SFR (UV+IR) using L_IR estimated from MIPS flux
        # MIPS flux needs some careful treatment
        # input for this LIR must be in janskies
        midx = np.array(['mips' in u for u in res['obs']['filternames']],dtype=bool)
        mips_flux = prosp['obs']['mags'][:,midx].squeeze() * 3631 * 1e3
        lir = mips_to_lir(mips_flux, res['model'].params['zred'][0])
        sfr_uvir_lirfrommips_prosp = prosp_dutils.sfr_uvir(lir,prosp['extras']['luv']['chain'])

        # turn all of these into percentiles
        pars = ['agn_heating_fraction', 'old_star_heating_fraction', 'young_star_heating_fraction', \
                'sfr_uvir_truelir_prosp','sfr_uvir_lirfrommips_prosp']
        vals = [agn_heating_fraction, old_star_heating_fraction, young_star_heating_fraction, \
                sfr_uvir_truelir_prosp, sfr_uvir_lirfrommips_prosp]
        for p, v in zip(pars,vals):
            mid, up, down = weighted_quantile(v, np.array([0.5, 0.84, 0.16]), weights=res['weights'][prosp['sample_idx']])
            out[p]['q50'] += [mid]
            out[p]['q84'] += [up]
            out[p]['q16'] += [down]

    for key in out.keys():
        if type(out[key]) == dict:
            for key2 in out[key].keys(): out[key][key2] = np.array(out[key][key2])
        else:
            out[key] = np.array(out[key])

    ### dump files and return
    hickle.dump(out,open(filename, "w"))
    return out

def do_all(runname='td_huge', outfolder=None,**opts):

    if outfolder is None:
        outfolder = os.getenv('APPS') + '/prospector_alpha/plots/'+runname+'/fast_plots/'
        if not os.path.isdir(outfolder):
            os.makedirs(outfolder)
            os.makedirs(outfolder+'data/')

    data = collate_data(runname,filename=outfolder+'data/uvircomp.h5',**opts)

    plot_uvir_comparison(data,outfolder)
    plot_heating_sources(data,outfolder)

def plot_uvir_comparison(data, outfolder):

    fig, ax = plt.subplots(1, 1, figsize = (4.5,4.5)) # Prospector UVIR SFR versus Kate's UVIR SFR

    #### Plot Prospector UV+IR SFR versus the observed UV+IR SFR
    good = data['sfr_uvir_obs'] > 0
    xdat = np.clip(data['sfr_uvir_obs'][good],minsfr,np.inf)
    ydat = np.clip(data['sfr_uvir_lirfrommips_prosp']['q50'][good],minsfr,np.inf)

    ax.plot(xdat, ydat, 'o', color='0.4',markeredgecolor='k',ms=ms, alpha=alpha)
    min, max = xdat.min()*0.8, ydat.max()*1.2
    ax.plot([min, max], [min, max], '--', color='0.4',zorder=-1)
    ax.axis([min,max,min,max])

    ax.set_xlabel('SFR$_{\mathrm{UV+IR,classic}}$')
    ax.set_ylabel('SFR$_{\mathrm{UV+IR,Prospector}}$')

    ax.set_xscale('log',subsx=[3])
    ax.xaxis.set_minor_formatter(FormatStrFormatter('%2.4g'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%2.4g'))
    ax.tick_params('both', pad=3.5, size=8.0, width=1.0, which='both')

    ax.set_yscale('log',subsy=[3])
    ax.yaxis.set_minor_formatter(FormatStrFormatter('%2.4g'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%2.4g'))

    # save and exit
    plt.savefig(outfolder+'prosp_uvir_to_obs_uvir.png',dpi=dpi)
    plt.tight_layout()
    plt.close()

def plot_heating_sources(data, outfolder, color_by_fagn=False, color_by_logzsol=True):

    fig, ax = plt.subplots(1, 1, figsize = (5,4))
    fig2, ax2 = plt.subplots(1, 3, figsize = (9,3))

    # first plot SFR ratio versus heating by young stars
    # grab quantities
    x = data['young_star_heating_fraction']['q50']
    y = data['sfr_prosp']['q50'] / data['sfr_uvir_truelir_prosp']['q50']

    # add colored points and colorbar
    ax.axis([0,1,0,1.2])
    if color_by_fagn:
        colorvar = np.log10(data['fagn']['q50'])
        colorlabel = r'log(f$_{\mathrm{AGN}}$)'
        ax2_outstring = '_fagn'
    elif color_by_logzsol:
        colorvar = data['logzsol']['q50']
        colorlabel = r'log(Z/Z$_{\odot}$)'
        ax2_outstring = '_logzsol'
    else:
        colorvar = np.log10(np.clip(data['ssfr_prosp']['q50'],minssfr,np.inf))
        colorlabel = r'log(sSFR$_{\mathrm{SED}}$)'
        ax2_outstring = '_ssfr'

    pts = ax.scatter(x, y, s=s, cmap=cmap, c=colorvar, vmin=colorvar.min(), vmax=colorvar.max(),alpha=alpha_scat)
    cb = fig.colorbar(pts, ax=ax, aspect=10)
    cb.set_label(colorlabel)
    cb.solids.set_rasterized(True)
    cb.solids.set_edgecolor("face")

    # labels
    ax.set_ylabel('SFR$_{\mathrm{Prosp}}$/SFR$_{\mathrm{UV+IR}}$')
    ax.set_xlabel('fraction of heating from young stars')

    # next plot the sources of the heating
    y1, y2, y3 = [data[x+'_heating_fraction']['q50'] for x in 'young_star', 'old_star', 'agn']
    x = np.log10(np.clip(data['ssfr_prosp']['q50'],minssfr,np.inf))
    for i,p in enumerate([y1,y2,y3]): ax2[i].plot(x, p, 'o', color='0.4',markeredgecolor='k',ms=ms,alpha=alpha)

    # labels and limits
    for a in ax2: 
        a.set_xlabel('log(sSFR$_{\mathrm{Prosp}}$)')
        a.set_ylim(0,1)
    ax2[0].set_ylabel('fraction of energy from young stars')
    ax2[1].set_ylabel('fraction of energy from old stars')
    ax2[2].set_ylabel('fraction of energy from AGN')

    # clean up
    fig.tight_layout()
    fig2.tight_layout()

    fig.savefig(outfolder+'uvir_frac'+ax2_outstring+'.png',dpi=dpi)
    fig2.savefig(outfolder+'heating_fraction_against_ssfr.png',dpi=dpi)
    plt.close()
    plt.close()
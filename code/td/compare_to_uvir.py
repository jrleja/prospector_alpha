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
dpi = 160
cmap = 'cool'
minsfr = 0.01

def collate_data(runname, filename=None, regenerate=False, nsamp=100, add_fagn=True, lir_from_mips=True):
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
    for par in ['sfr_uvir_truelir_prosp', 'sfr_uvir_lirfrommips_prosp', 'sfr_prosp', 'ssfr_prosp', 'logzsol', 'fagn','agn_heating_fraction','old_star_heating_fraction']:
        out[par] = {}
        for q in qvals: out[par][q] = []
    for par in ['sfr_uvir_obs','objname']: out[par] = []

    ### fill output containers
    basenames, _, _ = prosp_dutils.generate_basenames(runname)
    field = [name.split('/')[-1].split('_')[0] for name in basenames]

    ### load necessary information
    uvirlist = []
    allfields = np.unique(field).tolist()
    for f in allfields:
        uvirlist.append(td_io.load_mips_data(f))

    ### iterate over items
    for i, name in enumerate(basenames[:100]):

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
            out['fagn'][q] = prosp['thetas']['fagn'][q] 
            out['logzsol'][q] = prosp['thetas']['massmet_2'][q] 
            out['sfr_prosp'][q] = prosp['extras']['sfr_100'][q]
            out['ssfr_prosp'][q] = prosp['extras']['ssfr_100'][q]

        # observed UV+IR SFRs
        # find correct field, find ID match
        u_idx = uvirlist[allfields.index(field[i])]['id'].astype(int) == int(out['objname'][-1].split('_')[-1])
        out['sfr_uvir_obs'] += [uvirlist[allfields.index(field[i])]['sfr'][u_idx][0]]

        # Ratios and calculations
        # here we calculate SFR(UV+IR) with LIR + LUV from:
        # (a) the 'true' L_IR and L_UV in the model,
        # (b) L_IR estimated from MIPS flux,
        # (c) true L_IR with no AGN contribution,
        # (d) true L_IR with only heating from young stars
        sfr_uvir_truelir_prosp = prosp_dutils.sfr_uvir(prosp['extras']['lir']['chain'],prosp['extras']['luv']['chain'])
        agn_heating_fraction = 1-prosp_dutils.sfr_uvir(prosp['extras']['lir_agn']['chain'],prosp['extras']['luv_agn']['chain']) / sfr_uvir_truelir_prosp
        old_star_heating_fraction = 1-prosp_dutils.sfr_uvir(prosp['extras']['lir_young']['chain'],prosp['extras']['luv_young']['chain']) / sfr_uvir_truelir_prosp

        # MIPS flux needs some careful treatment
        # input for this LIR must be in janskies
        midx = np.array(['mips' in u for u in res['obs']['filternames']],dtype=bool)
        mips_flux = prosp['obs']['mags'][:,midx].squeeze() * 3631 * 1e3
        lir = mips_to_lir(mips_flux, res['model'].params['zred'][0])
        sfr_uvir_lirfrommips_prosp = prosp_dutils.sfr_uvir(lir,prosp['extras']['luv']['chain'])

        # turn all of these into percentiles
        pars = ['agn_heating_fraction', 'old_star_heating_fraction','sfr_uvir_truelir_prosp','sfr_uvir_lirfrommips_prosp']
        vals = [agn_heating_fraction, old_star_heating_fraction, sfr_uvir_truelir_prosp, sfr_uvir_lirfrommips_prosp]
        for p, v in zip(pars,vals):
            mid, up, down = weighted_quantile(v, np.array([0.5, 0.84, 0.16]), weights=res['weights'][prosp['sample_idx']])
            out[p]['q50'] += [mid]
            out[p]['q50'] += [up]
            out[p]['q50'] += [down]

    for key in out.keys():
        if type(out[key]) == dict:
            for key2 in out[key].keys(): out[key][key2] = np.array(out[key][key2])
        else:
            out[key] = np.array(out[key])

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

def uvir_comparison(data, outfolder,color_by_fagn=False,color_by_logzsol=True):
    '''
    x-axis is log time (bins)
    y-axis is SFR_BIN / SFR_UVIR_PROSP ?
    color-code each line by SFR_PROSP / SFR_UVIR_PROSP ?
    make sure SFR_PROSP and SFR_UVIR_PROSP show same dependencies! 
    '''

    fig, ax = plt.subplots(1, 1, figsize = (4.5,4.5)) # Prospector UVIR SFR versus Kate's UVIR SFR
    fig2, ax2 = plt.subplots(1, 1, figsize = (5,4)) # Fractional UVIR SFR in 100 Myr versus (SED SFR - UVIR SFR)
    fig3, ax3 = plt.subplots(1, 2, figsize = (8,4)) # Fractional UVIR SFR in 100 Myr versus (SED SFR - UVIR SFR)
    ms, s = 1.5, 3

    #### Plot Prospector UV+IR SFR versus the observed UV+IR SFR
    good = data['sfr_uvir_obs'] > 0
    xdat = np.clip(data['sfr_uvir_obs'][good],minsfr,np.inf)
    ydat = np.clip(data['sfr_uvir_prosp'][good],minsfr,np.inf)

    ax.plot(xdat, ydat, 'o', color='0.4',markeredgecolor='k',ms=ms)
    min, max = xdat.min()*0.8, ydat.max()*1.2
    ax.plot([min, max], [min, max], '--', color='0.4',zorder=-1)
    ax.axis([min,max,min,max])

    ax.set_ylabel('SFR$_{\mathrm{UV+IR,PROSP}}$')
    ax.set_xlabel('SFR$_{\mathrm{UV+IR,KATE}}$')

    ax.set_xscale('log',subsx=[3])
    ax.xaxis.set_minor_formatter(FormatStrFormatter('%2.4g'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%2.4g'))
    ax.tick_params('both', pad=3.5, size=8.0, width=1.0, which='both')

    ax.set_yscale('log',subsy=[3])
    ax.yaxis.set_minor_formatter(FormatStrFormatter('%2.4g'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%2.4g'))

    ##### Plot (UV+IR SFR / Prosp SFR)
    """ we want to save AGN_FRAC and OLD_FRAC
    which is (SFR_AGN) / (SFR_UVIR_LIR_PROSP), (SFR_OLD) / (SFR_UVIR_LIR_PROSP)
    also compare (SFR_UVIR_PROSP) to (SFR_PROSP)
    """
    uvir_frac, uvir_frac_up, uvir_frac_do = [], [], []
    agn_frac, agn_frac_up, agn_frac_do = [], [], []
    old_frac, old_frac_up, old_frac_do = [], [], []
    ratio, ratio_up, ratio_do = [], [], []
    for i, bchain in enumerate(data['sfr_source']['young_stars_chain']):

        # first is for heating ratios, calculated self-consistently with Prospector LIR
        # second is the 'observed SFR_IR+UV', i.e. with DH02 templates
        sfr_uvir_prosp_for_comp_with_obs = np.array([float(x) for x in data['sfr_uvir_prosp_chain'][i]])
        sfr_uvir_prosp_for_heating = np.array([float(x) for x in data['sfr_uvir_lir_prosp_chain'][i]])

        mid, up, do = np.percentile(np.array(bchain) / sfr_uvir_prosp_for_heating, [50,84,16])
        uvir_frac.append(mid)
        uvir_frac_up.append(up)
        uvir_frac_do.append(do)

        mid, up, do = np.percentile(np.array(data['sfr_source']['agn_chain'][i]) /  sfr_uvir_prosp_for_heating, [50,84,16])
        agn_frac.append(mid)
        agn_frac_up.append(up)
        agn_frac_do.append(do)

        ofrac = (sfr_uvir_prosp_for_heating - np.array(data['sfr_source']['agn_chain'][i]) - np.array(bchain)) /  sfr_uvir_prosp_for_heating
        mid, up, do = np.percentile(ofrac, [50,84,16])
        old_frac.append(mid)
        old_frac_up.append(up)
        old_frac_do.append(do)

        mid, up, do = np.percentile(data['sfr_prosp_chain'][i] / sfr_uvir_prosp_for_heating, [50,84,16])
        ratio.append(mid)
        ratio_up.append(up)
        ratio_do.append(do)

    ### grab quantities
    x = uvir_frac
    xerr = prosp_dutils.asym_errors(np.array(uvir_frac), np.array(uvir_frac_up), np.array(uvir_frac_do))
    y = ratio
    yerr = prosp_dutils.asym_errors(np.array(ratio), np.array(ratio_up), np.array(ratio_do))

    #### add colored points and colorbar
    ax2.axis([0,1,0,1.2])
    if color_by_fagn:
        colorvar = np.log10(data['fagn'])
        colorlabel = r'log(f$_{\mathrm{AGN}}$)'
        ax2_outstring = '_fagn'
    elif color_by_logzsol:
        colorvar = data['logzsol'].squeeze()
        colorlabel = r'log(Z/Z$_{\odot}$)'
        ax2_outstring = '_logzsol'
    else:
        colorvar = np.log10(data['ssfr_prosp'])
        colorlabel = r'log(sSFR$_{\mathrm{SED}}$)'
        ax2_outstring = '_ssfr'

    pts = ax2.scatter(x,y, s=s, cmap=cmap, c=colorvar, vmin=colorvar.min(), vmax=colorvar.max())
    #ax2.errorbar(x,y, yerr=yerr, xerr=xerr, fmt='o', ecolor='0.2', capthick=0.6,elinewidth=0.6,ms=0.0,alpha=0.5,zorder=-5)
    cb = fig2.colorbar(pts, ax=ax2, aspect=10)
    cb.set_label(colorlabel)
    cb.solids.set_rasterized(True)
    cb.solids.set_edgecolor("face")

    ### label
    ax2.set_ylabel('SFR$_{\mathrm{Prosp}}$/SFR$_{\mathrm{UV+IR}}$')
    ax2.set_xlabel('fraction of dust heating from young stars')

    ### grab quantities for last plot
    y1, y2 = agn_frac, old_frac
    yerr1, yerr2 = prosp_dutils.asym_errors(np.array(agn_frac), np.array(agn_frac_up), np.array(agn_frac_do)), \
                   prosp_dutils.asym_errors(np.array(agn_frac), np.array(agn_frac_up), np.array(agn_frac_do))
    x = np.log10(data['ssfr_prosp'])
    xerr = prosp_dutils.asym_errors(y, np.log10(data['ssfr_prosp_up']), np.log10(data['ssfr_prosp_do']))

    #### add colored points and colorbar
    ax3[0].plot(x, y1, 'o', color='0.4',markeredgecolor='k',ms=ms)
    ax3[1].plot(x, y2, 'o', color='0.4',markeredgecolor='k',ms=ms)

    ### label
    for a in ax3: 
        a.set_xlabel('log(sSFR$_{\mathrm{Prosp}}$)')
        a.set_ylim(0,1)
    ax3[0].set_ylabel('fraction of dust heating from AGN')
    ax3[1].set_ylabel('fraction of dust heating from old stars')

    #### clean up
    fig.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()

    fig.savefig(outfolder+'prosp_uvir_to_obs_uvir.png',dpi=dpi)
    fig2.savefig(outfolder+'uvir_frac'+ax2_outstring+'.png',dpi=dpi)
    fig3.savefig(outfolder+'other_heating_sources.png',dpi=dpi)

    plt.close()
    print 1/0

import numpy as np
import matplotlib.pyplot as plt
import os, hickle, td_io, prosp_dutils, copy
from prospector_io import load_prospector_data, load_prospector_extra
from astropy.cosmology import WMAP9
import td_params as pfile
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
    
    ### if it's already made, load it and give it back
    # else, start with the making!
    if os.path.isfile(filename) and regenerate == False:
        with open(filename, "r") as f:
            outdict=hickle.load(f)

        return outdict

    # load obs dict and sps object
    # we only want these once, they're heavy
    run_params = pfile.run_params
    sps = pfile.load_sps(**run_params)
    obs = pfile.load_obs(**run_params)
    objnames = np.genfromtxt(run_params['datdir']+run_params['runname']+'.ids',dtype=[('objnames', '|S40')])

    ### define output containers
    sfr_uvir_obs, fagn, objname = [], [], []
    sfr_uvir_prosp, sfr_uvir_prosp_up, sfr_uvir_prosp_do, sfr_uvir_prosp_chain = [], [], [], []
    sfr_uvir_lir_prosp, sfr_uvir_lir_prosp_up, sfr_uvir_lir_prosp_do, sfr_uvir_lir_prosp_chain = [], [], [], []
    sfr_prosp, sfr_prosp_up, sfr_prosp_do, sfr_prosp_chain = [], [], [], []
    ssfr_prosp, ssfr_prosp_up, ssfr_prosp_do, ssfr_prosp_chain = [], [], [], []
    logzsol = []

    sfr_source = {}
    sfr_source_names = ['young_stars','agn']
    for i,source in enumerate(sfr_source_names): 
        sfr_source[source] = []
        sfr_source[source+'_up'] = []
        sfr_source[source+'_do'] = []
        sfr_source[source+'_chain'] = []

    ### fill output containers
    basenames, _, _ = prosp_dutils.generate_basenames(runname)
    field = [name.split('/')[-1].split('_')[0] for name in basenames]

    uvirlist = []
    allfields = np.unique(field).tolist()
    for f in allfields:
        uvirlist.append(td_io.load_mips_data(f))
    basenames = basenames[:100]
    for i, name in enumerate(basenames):

        # load output from fit
        try:
            res, _, model, prosp = load_prospector_data(name)
        except:
            print name.split('/')[-1]+' failed to load. skipping.'
            continue
        if prosp is None:
            continue

        objname.append(name.split('/')[-1])
        print 'loaded '+objname[-1]

        # generate two new models
        # one has all of the 'depends_on' functions stripped
        # so we can update masses without having them overridden by the z_fraction variables
        run_params['objname'] = objnames['objnames'][i]
        oldmodel = pfile.load_model(**run_params)
        model_params = copy.deepcopy(oldmodel.config_list)
        for j in range(len(model_params)):
            if model_params[j]['name'] == 'mass':
                model_params[j].pop('depends_on', None)
        model = sedmodel.SedModel(model_params)

        # if we only sample once, use best-fit
        # otherwise, take `nsamp` samples from posterior
        if nsamp == 1:
            sample_idx = [res['lnprobability'].argmax()]
        else:
            weights = res['weights'][prosp['sample_idx']]
            sample_idx = np.random.choice(prosp['sample_idx'], size=nsamp, p=weights/weights.sum(), replace=False)

        # here we calculate SFR(UV+IR) with LIR + LUV from:
        # (a) the 'true' L_IR and L_UV,
        # (b) L_IR estimated from MIPS flux,
        # (a) true L_IR with no AGN contribution,
        # (b) true L_IR with only heating from young stars
        true_sfr_uvir_prosp, model_lir_sfr_uvir, sidx_map = [], [], []
        for source in sfr_source_names: sfr_source[source+'_chain'].append([])
        for k, sidx in enumerate(sample_idx):
            
            # calculate thetas, bookkeeping
            sidx_map.append(np.where(prosp['sample_idx'] == sidx)[0][0])
            theta = copy.copy(res['chain'][sidx])

            # pass thetas to model with dependencies
            # then pull out the masses and f_agn
            oldmodel.set_parameters(theta)
            mass = copy.copy(oldmodel.params['mass'])
            fagn_mod = copy.copy(oldmodel.params['fagn'])

            # calculate UV+IR SFR using new thetas
            out = prosp_dutils.measure_restframe_properties(sps, model = oldmodel, thetas = theta, measure_ir = True, measure_luv = True)
            model_lir_sfr_uvir.append(prosp_dutils.sfr_uvir(out['lir'],out['luv']))

            # also calculate UV+IR SFR a second way, by extrapolating from the MIPS flux
            # input for this LIR must be in janskies
            midx = np.array(['mips' in u for u in res['obs']['filternames']],dtype=bool)
            mips_flux = prosp['obs']['mags'][sidx_map[-1],midx].squeeze() * 3631 * 1e3
            lir = mips_to_lir(mips_flux, res['model'].params['zred'][0])
            true_sfr_uvir_prosp.append(prosp_dutils.sfr_uvir(lir,out['luv']))

            # calculate AGN contribution, young star contribution to LUV, LIR
            for source in sfr_source_names:
                if source == 'agn':
                    theta[model.theta_index['fagn']] = 0.0
                    out = prosp_dutils.measure_restframe_properties(sps, model = model, thetas = theta, measure_ir = True, measure_luv = True)
                    model.params['fagn'] = fagn_mod
                    sfr_source[source+'_chain'][-1] += [model_lir_sfr_uvir[-1] - prosp_dutils.sfr_uvir(out['lir'],out['luv'])]

                if source == 'young_stars':
                    theta[model.theta_index['fagn']] = 0.0
                    model.params['mass'] = np.zeros_like(mass)
                    model.params['mass'][0] = mass[0]
                    if model.params['agebins'][0][1] != 8:
                        model.params['mass'][1] = mass[1]
                    out = prosp_dutils.measure_restframe_properties(sps, model = model, thetas = theta, measure_ir = True, measure_luv = True)
                    model.params['mass'] = mass
                    sfr_source[source+'_chain'][-1] += [prosp_dutils.sfr_uvir(out['lir'],out['luv'])]


        #### append everything
        if nsamp > 1:
            mid, up, down = weighted_quantile(true_sfr_uvir_prosp, np.array([0.5, 0.84, 0.16]), weights=res['weights'][sample_idx])
        else:
            mid, up, down = true_sfr_uvir_prosp[-1], true_sfr_uvir_prosp[-1], true_sfr_uvir_prosp[-1]
        sfr_uvir_prosp.append(mid)
        sfr_uvir_prosp_up.append(up)
        sfr_uvir_prosp_do.append(down)
        sfr_uvir_prosp_chain.append(true_sfr_uvir_prosp)

        if nsamp > 1:
            mid, up, down = weighted_quantile(model_lir_sfr_uvir, np.array([0.5, 0.84, 0.16]), weights=res['weights'][sample_idx])
        else:
            mid, up, down = model_lir_sfr_uvir[-1], model_lir_sfr_uvir[-1], model_lir_sfr_uvir[-1]
        sfr_uvir_lir_prosp.append(mid)
        sfr_uvir_lir_prosp_up.append(up)
        sfr_uvir_lir_prosp_do.append(down)
        sfr_uvir_lir_prosp_chain.append(model_lir_sfr_uvir)

        for source in sfr_source_names:
            if nsamp > 1:
                mid, up, down = weighted_quantile(sfr_source[source+'_chain'][-1], np.array([0.5, 0.84, 0.16]), weights=res['weights'][sample_idx])
            else:
                mid, up, down = sfr_source[source+'_chain'][-1], sfr_source[source+'_chain'][-1], sfr_source[source+'_chain'][-1]
            sfr_source[source].append(mid)
            sfr_source[source+'_up'].append(up)
            sfr_source[source+'_do'].append(down)

        ### now the easy stuff
        sfr_prosp.append(prosp['extras']['sfr_100']['q50'])
        sfr_prosp_up.append(prosp['extras']['sfr_100']['q84'])
        sfr_prosp_do.append(prosp['extras']['sfr_100']['q16'])
        sfr_prosp_chain.append(prosp['extras']['sfr_100']['chain'][np.array(sidx_map)].squeeze())
        ssfr_prosp.append(prosp['extras']['ssfr_100']['q50'])
        ssfr_prosp_up.append(prosp['extras']['ssfr_100']['q84'])
        ssfr_prosp_do.append(prosp['extras']['ssfr_100']['q16'])
        ssfr_prosp_chain.append(prosp['extras']['ssfr_100']['chain'][np.array(sidx_map)].squeeze())

        if nsamp > 1:
            fagn.append(weighted_quantile(res['chain'][sample_idx,model.theta_labels().index('fagn')],np.array([0.5]),weights=res['weights'][sample_idx]))
            logzsol.append(weighted_quantile(res['chain'][sample_idx,model.theta_labels().index('massmet_2')],np.array([0.5]),weights=res['weights'][sample_idx]))
        else:
            fagn.append(res['chain'][sample_idx,model.theta_labels().index('fagn')])
            logzsol.append(res['chain'][sample_idx,model.theta_labels().index('massmet_2')])


        ### now UV+IR SFRs
        # find correct field, find ID match
        uvir = uvirlist[allfields.index(field[i])]
        u_idx = uvir['id'] == int(name.split('_')[-1])
        sfr_uvir_obs.append(uvir['sfr'][u_idx][0])

    ### turn everything into numpy arrays
    for source in sfr_source_names:
        sfr_source[source] = np.array(sfr_source[source])
        sfr_source[source+'_up'] = np.array(sfr_source[source+'_up'])
        sfr_source[source+'_do'] = np.array(sfr_source[source+'_do'])

    out = {
           'sfr_source': sfr_source,
           'sfr_uvir_obs': np.array(sfr_uvir_obs),
           'sfr_uvir_prosp': np.array(sfr_uvir_prosp),
           'sfr_uvir_prosp_up': np.array(sfr_uvir_prosp_up),
           'sfr_uvir_prosp_do': np.array(sfr_uvir_prosp_do),
           'sfr_uvir_prosp_chain': sfr_uvir_prosp_chain,
           'sfr_uvir_lir_prosp': np.array(sfr_uvir_lir_prosp),
           'sfr_uvir_lir_prosp_up': np.array(sfr_uvir_lir_prosp_up),
           'sfr_uvir_lir_prosp_do': np.array(sfr_uvir_lir_prosp_do),
           'sfr_uvir_lir_prosp_chain': sfr_uvir_lir_prosp_chain,
           'sfr_prosp': np.array(sfr_prosp),
           'sfr_prosp_up': np.array(sfr_prosp_up),
           'sfr_prosp_do': np.array(sfr_prosp_do),
           'sfr_prosp_chain': sfr_prosp_chain,
           'ssfr_prosp': np.array(ssfr_prosp),
           'ssfr_prosp_up': np.array(ssfr_prosp_up),
           'ssfr_prosp_do': np.array(ssfr_prosp_do),
           'ssfr_prosp_chain': ssfr_prosp_chain,
           'fagn': np.array(fagn),
           'logzsol': np.array(logzsol),
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

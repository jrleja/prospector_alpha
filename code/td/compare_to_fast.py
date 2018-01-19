import numpy as np
import matplotlib.pyplot as plt
import os, hickle, td_io, prosp_dutils
from matplotlib.ticker import FormatStrFormatter
from prospector_io import load_prospector_data
from astropy.cosmology import WMAP9
from dynesty.plotting import _quantile as weighted_quantile
from fix_ir_sed import mips_to_lir
import copy
from scipy.stats import spearmanr, mode
from stack_td_sfh import sfr_ms
import td_huge_params as pfile

plt.ioff()

red = '#FF3D0D'
dpi = 160

minlogssfr = -15
minssfr = 10**minlogssfr
minsfr = 0.0001

nbin_min = 5

def sfr_ratio_for_fast(tau,tage):
    """ get ratio of instantaneous SFR to 100 Myr SFR
    """
    tau, tage = float(tau), float(tage)
    norm = tau*(1-np.exp(-tage/tau))
    sfr_inst = np.exp(-tage/tau)/norm
    sfr_100myr = integrate_exp_tau(tage-0.1,tage,tau,tage) / (0.1/tage)

    return sfr_100myr/sfr_inst

def integrate_exp_tau(t1,t2,tau,tage):
    """ integrate exponentially declining function
    """
    # write down both halves of integral
    integrand = tau*(np.exp(-t1/tau)-np.exp(-t2/tau))
    norm = tau*(1-np.exp(-tage/tau))

    return integrand/norm

def calc_uvj_flag(uvj, return_dflag = True):
    """calculate a UVJ flag as Whitaker et al. 2012
    U-V > 0.8(V-J) + 0.7, U-V > 1.3, V-J < 1.5
    1 = star-forming, 2 = dusty star-forming, 3 = quiescent
    """

    uvjmag = 25 - 2.5*np.log10(uvj)
    
    u_v = uvjmag[:,0]-uvjmag[:,1]
    v_j = uvjmag[:,1]-uvjmag[:,2]  
    
    # initialize flag to 3, for quiescent
    uvj_flag = np.repeat(3,uvjmag.shape[0])
    
    # star-forming
    sfing = (u_v < 1.3) | (v_j >= 1.5)
    uvj_flag[sfing] = 1
    sfing = (v_j >= 0.75) & (v_j <= 1.5) & (u_v <= 0.8*v_j+0.7)
    uvj_flag[sfing] = 1
    
    # dusty star-formers
    dusty_sf = (uvj_flag == 1) & (u_v >= 1.3)
    uvj_flag[dusty_sf] = 2

    # dust flag: if True, has very little dust
    if return_dflag:
        dflag= (u_v < (-1.25*v_j+2.875))
        return uvj_flag, dflag

    return uvj_flag

def collate_data(runname, runname_fast, filename=None, regenerate=False, calc_dmips=False, nobj=None, **kwargs):
    """
    regenerate, boolean: 
        if true, always load individual files to re-create data
        else will check for existence of data file

    calc_dmips, boolean:
        if true, calculate the difference in IR fluxes due to nebemlineinspec keyword 
        only set if necessary, this means instantiating an SPS and doing a model call every loop

    nobj, int:
        only load X number of objects. useful for re-making catalog for testing purposes.
    """
    
    sps = None

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
                 r'f$_{\mathrm{AGN}}$', 'dust index', r'f$_{\mathrm{AGN,MIR}}$']
    fnames = ['stellar_mass','sfr_100','dust2','ssfr_100','avg_age','half_time'] # take for fastmimic too
    pnames = fnames+['massmet_2', 'duste_qpah', 'fagn', 'dust_index', 'fmir'] # already calculated in Prospector
    enames = ['model_uvir_sfr', 'model_uvir_ssfr', 'sfr_ratio','model_uvir_truelir_sfr', 'model_uvir_truelir_ssfr', 'sfr_ratio_truelir'] # must calculate here

    outprosp, outprosp_fast, outfast, outlabels = {},{'bfit':{}},{},{}
    sfr_100_uvir, sfr_100_uv, sfr_100_ir, objname = [], [], [], []
    phot_chi, phot_percentile, phot_obslam, phot_restlam, phot_fname, dmips = [], [], [], [], [], []
    outfast['z'] = []
    outfast['uvj'], outfast['uvj_prosp'], outfast['uvj_dust_prosp'], outfast['uv'], outfast['vj'] = [], [], [], [], []

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
    if nobj is not None:
        basenames = basenames[:nobj]
    if runname_fast is not None:
        basenames_fast, _, _ = prosp_dutils.generate_basenames(runname_fast)
    field = [name.split('/')[-1].split('_')[0] for name in basenames]

    fastlist, uvirlist, adatlist = [], [], []
    allfields = np.unique(field).tolist()
    for f in allfields:
        fastlist.append(td_io.load_fast(runname,f))
        uvirlist.append(td_io.load_ancil_data(runname,f))
        adatlist.append(td_io.load_ancil_data(runname,f))

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

        # we need SPS object to calculate difference with best-fit
        midx = np.array(['mips' in u for u in res['obs']['filternames']],dtype=bool)
        if calc_dmips:
            if sps is None:
                sps = pfile.load_sps(**res['run_params'])

            # calculate difference in best-fit MIPS value
            bfit_theta = res['chain'][prosp['sample_idx'][0],:]
            model.params['nebemlineinspec'] = np.atleast_1d(False)
            _,mag_em,_ = model.mean_model(bfit_theta, res['obs'], sps=sps)

            dmips += ((prosp['obs']['mags'][0,midx] - mag_em[midx]) / prosp['obs']['mags'][0,midx]).tolist()
            print dmips[-1]
        else:
            dmips += [-1]

        # object name
        objname.append(name.split('/')[-1])

        # prospector first
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

        ### now FAST, UV+IR SFRs, UVJ
        # find correct field, find ID match
        fidx = allfields.index(field[i])
        fast = fastlist[fidx]
        uvir = uvirlist[fidx]
        f_idx = fast['id'] == int(name.split('_')[-1])
        u_idx = uvir['phot_id'] == int(name.split('_')[-1])

        # UVJ
        adat = adatlist[fidx]
        aidx = adat['phot_id'] == int(objname[-1].split('_')[-1])
        outfast['uvj'] += [adat['uvj'][aidx][0]]

        try:
            uvj_flag, uvj_dust_flag = calc_uvj_flag(prosp['obs']['uvj'])
            outfast['uvj_prosp'] += mode(uvj_flag)[0].tolist()
            outfast['uvj_dust_prosp'] += mode(uvj_dust_flag)[0].tolist()
            uv = 2.5*np.log10(prosp['obs']['uvj'][:,1]/prosp['obs']['uvj'][:,0]) 
            vj = 2.5*np.log10(prosp['obs']['uvj'][:,2]/prosp['obs']['uvj'][:,1])
            outfast['uv'] += weighted_quantile(uv, 0.5, weights=prosp['weights'])
            outfast['vj'] += weighted_quantile(vj, 0.5, weights=prosp['weights'])
        except KeyError:
            outfast['uvj_prosp'] += [-1]
            outfast['uvj_dust_prosp'] += [-1]
            outfast['uv'] += [-1]
            outfast['vj'] += [-1]

        # fill it up
        outfast['stellar_mass'] += [fast['lmass'][f_idx][0]]
        outfast['dust2'] += [prosp_dutils.av_to_dust2(fast['Av'][f_idx][0])]
        sfr_ratio = sfr_ratio_for_fast(10**(fast['ltau'][f_idx]-9),10**(fast['lage'][f_idx]-9))
        outfast['sfr_100'] += [(10**fast['lsfr'][f_idx][0])*sfr_ratio]
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
           'objname':objname,
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
           'phot_fname': np.array(phot_fname),
           'dmips': np.array(dmips)
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

    # load all data 
    fastfile = outfolder+'data/fastcomp.h5'
    data = collate_data(runname,runname_fast,filename=fastfile,**opts)

    # different plot options based on sample size
    popts = {'fmt':'o', 'capthick':1.5,'elinewidth':1.5,'alpha':0.8,'color':'0.3','ms':5,'markeredgecolor':'k'} # for small samples
    if len(data['uvir_sfr']) > 400:
        popts = {'fmt':'o', 'capthick':.15,'elinewidth':.15,'alpha':0.35,'color':'0.3','ms':2, 'errorevery': 5000}
    if len(data['uvir_sfr']) > 4000:
        popts = {'fmt':'o', 'capthick':.05,'elinewidth':.05,'alpha':0.2,'color':'0.3','ms':0.5, 'errorevery': 5000}

    # UVJ star-forming sequence
    """
    idx = (data['fast']['uvj'] < 3) # to make it look like Kate's selection
    star_forming_sequence(np.log10(data['prosp']['sfr_100']['q50'][idx]),
                          data['prosp']['stellar_mass']['q50'][idx],
                          data['fast']['z'][idx],
                          outfolder+'uvj_starforming.png',popts,
                          xlabel='[Prospector]', ylabel='[Prospector]',priors=True,ssfr_min=-np.inf)
    idx = (data['fast']['uvj'] == 3) # to make it look like Kate's selection
    star_forming_sequence(np.log10(data['prosp']['sfr_100']['q50'][idx]),
                          data['prosp']['stellar_mass']['q50'][idx],
                          data['fast']['z'][idx],
                          outfolder+'uvj_quiescent.png',popts,
                          xlabel='[Prospector]', ylabel='[Prospector]',priors=True,ssfr_min=-np.inf)
    """
    # star-forming sequence
    idx = (data['uvir_sfr'] > 0) & (data['fast']['uvj_prosp'] < 3) # to make it look like Kate's selection
    star_forming_sequence(np.log10(data['uvir_sfr'][idx]),
                          data['fast']['stellar_mass'][idx],
                          data['fast']['z'][idx],
                          outfolder+'star_forming_sequence_uvir.png',popts,
                          xlabel='[FAST]', ylabel='[UV+IR]')

    star_forming_sequence(np.log10(data['prosp']['sfr_100']['q50'][idx]),
                          data['fast']['stellar_mass'][idx],
                          data['fast']['z'][idx],
                          outfolder+'star_forming_sequence_prospector.png',popts,
                          xlabel='[FAST]', ylabel='[Prospector]',outfile=outfolder+'data/sfrcomp.h5',
                          correct_prosp=np.log10(data['uvir_sfr'])[idx],correct_prosp_mass=data['prosp']['stellar_mass']['q50'][idx])

    star_forming_sequence(np.log10(data['prosp']['sfr_100']['q50'][idx]),
                          data['prosp']['stellar_mass']['q50'][idx],
                          data['fast']['z'][idx],
                          outfolder+'star_forming_sequence_pure_prospector.png',popts,
                          xlabel='[Prospector]', ylabel='[Prospector]',priors=True,correct_prosp=np.log10(data['uvir_sfr'])[idx])

    idx = (data['fast']['uvj_prosp'] == 3) # only quiescent
    star_forming_sequence(np.log10(data['prosp']['sfr_100']['q50'][idx]),
                          data['prosp']['stellar_mass']['q50'][idx],
                          data['fast']['z'][idx],
                          outfolder+'star_forming_sequence_quiescent_prospector.png',popts,
                          xlabel='[Prospector]', ylabel='quiescent ',priors=True)


    # if we have FAST-mimic runs, do a thorough comparison
    # else just do Prospector-FAST
    data['labels'] = np.array(data['labels'].tolist() + [r'f$_{\mathrm{AGN,MIR}}$']) # hack, remove once this is rerun
    fast_comparison(data['fast'],data['prosp'],data['labels'],data['pnames'],
                    outfolder+'fast_to_palpha_comparison.png',popts)
    delta_age_versus_delta_mass(data['fast'],data['prosp'],
                    outfolder+'deltat_deltam_fast_to_palpha.png',popts)
    if runname_fast is not None:
        fast_comparison(data['fast'],data['prosp_fast'],data['labels'],data['pnames'],
                        outfolder+'fast_to_fastmimic_comparison.png',popts,plabel='FAST-mimic')
        fast_comparison(data['fast'],data['prosp_fast'],data['labels'],data['pnames'],
                        outfolder+'fast_to_fastmimic_bfit_comparison.png',popts,plabel='FAST-mimic',bfit=True)
        fast_comparison(data['prosp_fast'],data['prosp'],data['labels'],data['pnames'],
                        outfolder+'fastmimic_to_palpha_comparison.png',popts,flabel='FAST-mimic')
        delta_age_versus_delta_mass(data['fast'],data['prosp_fast'],
                        outfolder+'deltat_deltam_fast_to_fastmimic.png',popts)
        delta_age_versus_delta_mass(data['prosp_fast'],data['prosp'],
                        outfolder+'deltat_deltam_fastmimic_to_palpha.png',popts)        
        delta_age_versus_delta_mass(data['fast'],data['prosp'],
                        outfolder+'deltat_deltam_fast_to_palpha.png',popts)  

    phot_residuals(data,outfolder,popts)
    mass_metallicity_relationship(data, outfolder+'massmet_vs_z.png', popts)
    deltam_with_redshift(data['fast'], data['prosp'], data['fast']['z'], outfolder+'deltam_vs_z.png', filename=outfolder+'data/masscomp.h5')
    prospector_versus_z(data,outfolder+'prospector_versus_z.png',popts)
    sfr_mass_density_comparison(data,outfolder=outfolder)

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
            good = (xfast != xfast.min()) & (yprosp != yprosp.min()) & np.isfinite(xfast) & np.isfinite(yprosp)
        else:
            good = np.isfinite(xfast) & np.isfinite(yprosp)

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

def delta_age_versus_delta_mass(fast,prosp,outname,popts):

    # try some y-variables
    if type(fast['half_time']) == type({}):
        mfast = fast['stellar_mass']['q50']
        tfast = fast['half_time']['q50']
        dfast = fast['dust2']['q50']
    else:
        tfast = fast['half_time']
        dfast = fast['dust2']
        mfast = fast['stellar_mass']
    params = [np.log10(prosp['half_time']['q50']/tfast),
              prosp['dust2']['q50'] - dfast]
    ylabels = [r'log(t$_{\mathrm{Prosp}}$/t$_{\mathrm{FAST}}$)',
               r'$\tau_{\mathrm{diffuse,Prosp}}-\tau_{\mathrm{diffuse,FAST}}$']
    ylims = [(-2.5,2.5),
             (-2,2)]
    xlim = (-1.5,1.5)

    if 'massmet_2' in prosp.keys():
        params.append(prosp['massmet_2']['q50'])
        ylabels.append(r'log(Z$_{\mathrm{Prosp}}$/Z$_{\odot}$)')
        ylims.append((-2,2))

    # plot geometry
    ysize = 2.666666
    fig, ax = plt.subplots(1, len(params), figsize = (ysize*len(params),ysize+0.2))
    ax = ax.ravel()
    medopts = {'marker':' ','alpha':0.95,'color':'red','ms': 7,'mec':'k','zorder':5}

    # x variable
    delta_mass = prosp['stellar_mass']['q50'] - mfast

    for i, par in enumerate(params):
        ax[i].errorbar(delta_mass,par,**popts)
        ax[i].set_xlabel(r'log(M$_{\mathrm{Prosp}}$/M$_{\mathrm{FAST}}$)')
        ax[i].set_ylabel(ylabels[i])

        ax[i].set_xlim(xlim)
        ax[i].set_ylim(ylims[i])
        ax[i].text(0.02,0.9,r'$\rho_{\mathrm{S}}$='+'{:1.2f}'.format(spearmanr(delta_mass,par)[0]),transform=ax[i].transAxes)

        ax[i].axhline(0, linestyle='--', color='k',lw=1,zorder=10)
        ax[i].axvline(0, linestyle='--', color='k',lw=1,zorder=10)

        # running median
        in_plot = (delta_mass > xlim[0]) & (delta_mass < xlim[1])
        x, y, bincount = prosp_dutils.running_median(delta_mass[in_plot],par[in_plot],avg=True,return_bincount=True,nbins=20)
        x, y = x[bincount > nbin_min], y[bincount > nbin_min]
        ax[i].plot(x,y, **medopts)

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
        axes[i].set_ylabel('Prospector ' + data['labels'][data['pnames'] == par][0])

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
    ylim_percentile = (-0.65,0.65)

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

def star_forming_sequence(sfr,mass,zred,outname,popts,xlabel=None,ylabel=None,outfile=None,priors=False,
                          correct_prosp=None,correct_prosp_mass=None, ssfr_min = -np.inf):
    """ Plot star-forming sequence for whatever SFR + mass combination is input
    impossible to replicate the Whitaker+14 work without pre-selection with UVJ cuts
    we don't have UVJ cuts (THOUGH WE CAN GET THEM IF DESIRED)
    instead, use a sSFR cut
    """

    # set redshift binning + figsize
    zbins = np.linspace(0.5,2.5,5)
    fig, ax = plt.subplots(2,2,figsize=(6,6))
    ax = np.ravel(ax)
    medopts = {'marker':' ','alpha':0.85,'color':'red','zorder':5,'lw':1.5}
    corrected_opts = {'marker':' ','alpha':0.85,'color':'orange','zorder':5,'lw':1.5}
    
    # min, max for data + model
    logm_min, logm_max = 8.5, 11.5
    logsfr_min, logsfr_max = -4,3.3
    ssfr_max = -8 # from Prospector physics
    mbins = np.linspace(logm_min,logm_max,14)
    prior_opts = {'linestyle':'--','color':'k','zorder':5,'lw':1.5}

    # whitaker+14 information
    zwhit = np.array([0.75, 1.25, 1.75, 2.25])
    logm_whit = np.linspace(logm_min,logm_max,50)
    whitopts = {'color':'blue','alpha':0.85,'lw':1.5,'zorder':5}

    # let's go!
    mass_save, sfr_save, sfr_corrected_save, z_save = [], [], [], []
    for i in range(len(zbins)-1):

        # the data
        in_bin = (zred > zbins[i]) & \
                 (zred <= zbins[i+1]) & \
                 (np.isfinite(sfr)) & \
                 (np.log10(10**sfr/10**mass) > ssfr_min)
        ax[i].errorbar(mass[in_bin], sfr[in_bin], **popts)

        # the data-driven relationship
        x, y, bincount = prosp_dutils.running_median(mass[in_bin], 10**sfr[in_bin],bins=mbins,avg=True,return_bincount=True)
        ax[i].errorbar(x, np.log10(y), **medopts)
        mass_save += x.tolist()
        sfr_save.append(y.tolist())
        z_save += np.repeat(zwhit[i],len(x)).tolist()

        # correct for Prospector sSFR ceiling
        if correct_prosp is not None:
            # if our mass vector is not Prospector mass, find it in keywords
            if correct_prosp_mass is None:
                pmass = mass[in_bin]
            else:
                pmass = correct_prosp_mass[in_bin]

            # take anything within 0.3 dex of the prior limit
            idx_atlimit = (sfr[in_bin] - pmass) > ssfr_max-0.3
            sfr_new, sfr_corr = sfr[in_bin], correct_prosp[in_bin]
            sfr_new[idx_atlimit] = sfr_corr[idx_atlimit]
            x, y, bincount = prosp_dutils.running_median(mass[in_bin], 10**sfr_new,bins=mbins,avg=True,return_bincount=True)
            ax[i].errorbar(x, np.log10(y), **corrected_opts)
            sfr_corrected_save.append(y.tolist())

        # the old model
        sfr_whit = sfr_ms(zwhit[i],logm_whit)
        ax[i].plot(logm_whit, sfr_whit, **whitopts)

        # the labels
        ax[i].set_xlabel(r'log(M/M$_{\odot}$) ' + xlabel)
        ax[i].set_ylabel(r'log(SFR/M$_{\odot}$ yr$^{-1}$) ' + ylabel)
        ax[i].text(0.02, 0.93, "{0:.1f}".format(zbins[i])+'<z<'+"{0:.1f}".format(zbins[i+1]),transform=ax[i].transAxes)

        # the ranges
        ax[i].set_xlim(logm_min,logm_max)
        ax[i].set_ylim(logsfr_min,logsfr_max)

        # the guiderails
        if np.isfinite(ssfr_min):
            ax[i].plot([logm_min,logm_max],[ssfr_min+logm_min,ssfr_min+logm_max],**prior_opts)
            ax[i].text(9.5,ssfr_min+9.6,'sSFR cut',rotation=30,fontsize=8)
        if priors:
            ax[i].text(logm_min+0.3,ssfr_max+logm_min+1.5,'Prospector prior',rotation=30,fontsize=8)
            ax[i].plot([logm_min,logm_max],[ssfr_max+logm_min,ssfr_max+logm_max],**prior_opts)

    ax[0].text(0.02,0.87,'Whitaker+14',color=whitopts['color'],transform=ax[0].transAxes)
    # flip the geometry
    if correct_prosp is not None:
        ax[0].text(0.02,0.81,'<data>(corr)',color=corrected_opts['color'],transform=ax[0].transAxes)
        ax[0].text(0.02,0.75,'<data>',color=medopts['color'],transform=ax[0].transAxes)
    else:
        ax[0].text(0.02,0.81,'<data>',color=medopts['color'],transform=ax[0].transAxes)

    # save
    if outfile is not None:
        out = {'mass':x,'sfr':np.dstack((sfr_save)).squeeze(),'sfr_corr':None,'z':zwhit}
        if correct_prosp is not None:
            out['sfr_corr'] = np.dstack((sfr_corrected_save)).squeeze()
        hickle.dump(out,open(outfile, "w"))

    plt.tight_layout()
    plt.savefig(outname,dpi=150)
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

def sfr_mass_density_comparison(data, outfolder=None):

    ### physics choices
    zbins = [(0.5,1.),(1.,1.5),(1.5,2.),(2.,2.5),(2.5,3.0)]
    mass_options = {
                    'xdata': data['fast']['stellar_mass'],
                    'ydata': data['prosp']['stellar_mass']['q50'],
                    'min': 8,
                    'max': 12,
                    'ylabel': r'N*M$_{\mathrm{stellar}}$',
                    'xlabel': r'log(M$_*$/M$_{\odot}$)',
                    'norm': 1e13,
                    'name': 'rhomass_fast_comparison.png',
                    'rho_label': r'$\rho_{\mathrm{Prosp}}/\rho_{\mathrm{FAST}}$=',
                    'ysource': 'FAST'
                   }

    sfr_options = {
                   'xdata': np.log10(np.clip(data['uvir_sfr'],0.001,np.inf)),
                   'ydata': np.log10(np.clip(data['prosp']['sfr_100']['q50'],0.001,np.inf)),
                   'min': -3,
                   'max': 5,
                   'ylabel': r'N*SFR',
                   'xlabel': r'log(SFR) [M$_{\odot}$/yr]',
                   'norm': 1e4,
                   'name': 'rhosfr_uvir_comparison.png',
                   'rho_label': r'$\rho_{\mathrm{Prosp}}/\rho_{\mathrm{UV+IR}}$=',
                   'ysource': 'UV+IR'
                   }

    ### plot choices
    nbins = 20
    histopts = {'drawstyle':'steps-mid','alpha':1.0, 'lw':1, 'linestyle': '-'}
    fontopts = {'fontsize':10}
    pcolor = '#1C86EE'
    oldcolor = '#FF3D0D'

    # plot mass + mass density distribution
    for opt in [mass_options,sfr_options]:
        fig, ax = plt.subplots(5,2,figsize=(5,9))
        for i,zbin in enumerate(zbins):

            # masses & indexes
            idx = (data['fast']['z'] > zbin[0]) & \
                  (data['fast']['z'] < zbin[1]) & \
                  (opt['xdata'] > opt['min']) & \
                  (opt['xdata'] < opt['max']) & \
                  (opt['ydata'] > opt['min']) & \
                  (opt['ydata'] < opt['max'])

            master_dat = opt['xdata'][idx]
            sample_dat = opt['ydata'][idx]

            # mass histograms. get bins from master histogram
            hist_master, bins = np.histogram(master_dat,bins=nbins,density=False)
            hist_sample, bins = np.histogram(sample_dat,bins=bins,density=False)
            bins_mid = (bins[1:]+bins[:-1])/2.

            # mass distribution
            ax[i,0].plot(bins_mid,hist_master,color=oldcolor, **histopts)
            ax[i,0].plot(bins_mid,hist_sample,color=pcolor, **histopts)

            # mass density distribution
            ax[i,1].plot(bins_mid,hist_master*10**bins_mid/opt['norm'],color=oldcolor, **histopts)
            ax[i,1].plot(bins_mid,hist_sample*10**bins_mid/opt['norm'],color=pcolor, **histopts)

            # axis labels
            ax[i,0].set_ylabel(r'N')
            ax[i,1].set_ylabel(opt['ylabel'])
            for a in ax[i,:]: a.set_ylim(a.get_ylim()[0],a.get_ylim()[1]*1.1)

            # text labels
            rhofrac = (10**sample_dat).sum() / (10**master_dat).sum()
            ax[i,0].text(0.98, 0.91,'{:1.1f} < z < {:1.1f}'.format(zbin[0],zbin[1]), transform=ax[i,0].transAxes,ha='right',**fontopts)
            ax[i,1].text(0.02, 0.91,opt['rho_label']+'{:1.2f}'.format(rhofrac),
                         transform=ax[i,1].transAxes,ha='left',**fontopts)

        # only bottom axes
        ax[0,0].text(0.98, 0.82, 'Prospector', transform=ax[0,0].transAxes,ha='right',color=pcolor,**fontopts)
        ax[0,0].text(0.98, 0.73, opt['ysource'], transform=ax[0,0].transAxes,ha='right',color=oldcolor,**fontopts)

        for a in ax[-1,:]: a.set_xlabel(opt['xlabel'])

        fig.tight_layout()
        fig.subplots_adjust(wspace=0.4,hspace=0.0)
        fig.savefig(outfolder+opt['name'],dpi=150)



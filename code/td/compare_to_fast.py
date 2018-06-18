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
from astropy.io import ascii
from astropy.table import Table
from matplotlib.ticker import MaxNLocator
from scipy.optimize import curve_fit

plt.ioff()

red = '#FF3D0D'
dpi = 160
cmap = ['#0202d6','#31A9B8','#FF9100','#FF420E']

minlogssfr = -15
minssfr = 10**minlogssfr
minsfr = 0.0001

nbin_min = 10

def get_cmap(N):

    import matplotlib.cm as cmx
    import matplotlib.colors as colors

    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='rainbow') 
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color

def sfr_ratio_for_fast(tau,tage,t2=0.1):
    """ get ratio of instantaneous SFR to t2 Gyr SFR
    """
    tau, tage = float(tau), float(tage)
    norm = tau*(1-np.exp(-tage/tau))
    sfr_inst = np.exp(-tage/tau)/norm
    sfr_avg = integrate_exp_tau(tage-t2,tage,tau,tage) / (t2/tage)

    return sfr_avg/sfr_inst

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

def collate_data(runname, runname_fast, runname_sample=None, filename=None, filename_grid=None, regenerate=False, nobj=None, **kwargs):
    """
    regenerate, boolean: 
        if true, always load individual files to re-create data
        else will check for existence of data file

    nobj, int:
        only load X number of objects. useful for re-making catalog for testing purposes.
    """
    
    ### if it's already made, load it and give it back
    # else, start with the making!
    if os.path.isfile(filename) and regenerate == False:
        with open(filename, "r") as f:
            outdict = hickle.load(f)
        try:
            with open(filename_grid, "r") as f:
                outg = hickle.load(f)
        except:
            outg = None
        return outdict, outg

    ### define output containers
    parlabels = [r'log(M$_{\mathrm{stellar}}$/M$_{\odot}$)', 'SFR [M$_{\odot}$/yr]',
                 r'$\tau_{\mathrm{diffuse}}$', r'log(sSFR) [yr$^-1$]',
                 r"t$_{\mathrm{avg}}$ [Gyr]", r"t$_{\mathrm{half-mass}}$ [Gyr]",
                  'SFR [M$_{\odot}$/yr] (30)', r'log(sSFR) [yr$^-1$] (30)',
                 r'log(Z/Z$_{\odot}$)', r'Q$_{\mathrm{PAH}}$',
                 r'f$_{\mathrm{AGN}}$', 'dust index', r'f$_{\mathrm{AGN,MIR}}$']
    fnames = ['stellar_mass','sfr_100','dust2','ssfr_100','avg_age','half_time', 'sfr_30','ssfr_30'] # take for fastmimic too
    pnames = fnames+['massmet_2', 'duste_qpah', 'fagn', 'dust_index', 'fmir'] # already calculated in Prospector
    enames = ['model_uvir_sfr', 'model_uvir_ssfr', 'sfr_ratio','model_uvir_truelir_sfr', 'model_uvir_truelir_ssfr', 'sfr_ratio_truelir'] # must calculate here

    outprosp, outprosp_fast, outfast, outlabels = {},{'bfit':{}},{},{}
    sfr_100_uvir, sfr_100_uv, sfr_100_ir, objname = [], [], [], []
    phot_chi, phot_percentile, phot_sn, phot_mag, phot_obslam, phot_restlam, phot_fname = [], [], [], [], [], [], []
    outfast['z'] = []
    outfast['uvj'], outfast['uvj_prosp'], outfast['uvj_dust_prosp'], outfast['uv'], outfast['vj'] = [], [], [], [], []
    logpar = ['stellar_mass', 'ssfr_30', 'ssfr_100']

    ### define grids
    ngrid_ssfr, ngrid_fast, ngrid_sfr = 100, 32, 40
    delssfr_lim = (-3,1)
    ssfr_lim = (-11,-8)
    delm_lim = (-1.,1.)    # minimum age is 15 Myr = log(-1.82/Gyr), maximum is tuniv(z=0.5) = 8.65
    logm_lim = (8.5,12.)
    logsfr_lim = (-2.5,3)
    outg = {}
    outg['grids'] = {
                    'ssfr_delssfr': [],
                    'logm_delm': [],
                    'logm_logsfr': [],
                    'logm_loguvirsfr': [],
                    'logsfr': np.linspace(logsfr_lim[0],logsfr_lim[1],ngrid_sfr+1),
                    'logm': np.linspace(logm_lim[0],logm_lim[1],ngrid_sfr+1),
                    'logm_fast': np.linspace(logm_lim[0],logm_lim[1],ngrid_fast+1),
                    'delm': np.linspace(delm_lim[0],delm_lim[1],ngrid_fast+1),
                    'ssfr': np.linspace(ssfr_lim[0],ssfr_lim[1],ngrid_ssfr+1),
                    'delssfr': np.linspace(delssfr_lim[0],delssfr_lim[1],ngrid_ssfr+1)
                   }

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
    if runname_sample is None:
        runname_sample = runname
    for f in allfields:
        fastlist.append(td_io.load_fast(runname_sample,f))
        uvirlist.append(td_io.load_ancil_data(runname_sample,f))
        adatlist.append(td_io.load_ancil_data(runname_sample,f))

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

        # object name
        objname.append(name.split('/')[-1])

        # prospector first
        for par in pnames:
            loc = 'thetas'
            if par in prosp['extras'].keys():
                loc = 'extras'
            for q in ['q16','q50','q84']:
                try:
                    x = prosp[loc][par][q]
                except:
                    x = -99
                if par in logpar:
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
            outprosp['sfr_ratio_truelir'][q] += [weighted_quantile(np.log10(prosp['extras']['sfr_30']['chain']/uvir_truelir_chain), 
                                                         np.array([float(q[1:])/100.]),weights=prosp['weights'])[0]]
            outprosp['model_uvir_sfr'][q] += [weighted_quantile(uvir_chain, np.array([float(q[1:])/100.]),weights=prosp['weights'])[0]]
            outprosp['model_uvir_ssfr'][q] += [weighted_quantile(uvir_chain/prosp['extras']['stellar_mass']['chain'], 
                                                               np.array([float(q[1:])/100.]),weights=prosp['weights'])[0]]
            outprosp['sfr_ratio'][q] += [weighted_quantile(np.log10(prosp['extras']['sfr_30']['chain']/uvir_chain), 
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
                    if par in logpar:
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
                if par in logpar:
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
        sfr_ratio = sfr_ratio_for_fast(10**(fast['ltau'][f_idx]-9),10**(fast['lage'][f_idx]-9),t2=0.03)
        outfast['sfr_30'] += [(10**fast['lsfr'][f_idx][0])*sfr_ratio]
        outfast['half_time'] += [prosp_dutils.exp_decl_sfh_half_time(10**fast['lage'][f_idx][0],10**fast['ltau'][f_idx][0])/1e9]

        outfast['avg_age'] += [prosp_dutils.exp_decl_sfh_avg_age(10**fast['lage'][f_idx][0],10**fast['ltau'][f_idx][0])/1e9]
        outfast['z'] += [float(model.params['zred'])]
        sfr_100_uvir += [uvir['sfr'][u_idx][0]]
        sfr_100_ir += [uvir['sfr_IR'][u_idx][0]]
        sfr_100_uv += [uvir['sfr_UV'][u_idx][0]]

        # photometry chi, etc.
        mask = res['obs']['phot_mask']
        phot_percentile += ((res['obs']['maggies'][mask] - prosp['obs']['mags'][0,mask]) / np.abs(res['obs']['maggies'][mask])).tolist()
        phot_chi += ((res['obs']['maggies'][mask] - prosp['obs']['mags'][0,mask]) / res['obs']['maggies_unc'][mask]).tolist()
        phot_sn += (res['obs']['maggies'][mask] / res['obs']['maggies_unc'][mask]).tolist()
        phot_mag += (-2.5*np.log10(res['obs']['maggies'][mask])).tolist()
        phot_obslam += (res['obs']['wave_effective'][mask]/1e4).tolist()
        phot_restlam += (res['obs']['wave_effective'][mask]/1e4/(1+outfast['z'][-1])).tolist()
        phot_fname += [str(fname) for fname in np.array(res['obs']['filternames'])[mask]]

        # grid calculations
        # these are clipped to the edges in the Y-direction (otherwise they're counted as NaNs!)
        # add in 0.1 dex Gaussian error to FAST masses for smoothing purposes. only do this if we're 
        # inside the mass boundaries to begin with, else galaxies below the mass limit dominate!
        fsm, nchain = outfast['stellar_mass'][-1], prosp['extras']['stellar_mass']['chain'].shape[0]
        if fsm < outg['grids']['logm_fast'][0]:
            logm_fast = np.repeat(fsm,nchain)
        else:
            logm_fast = np.random.normal(loc=fsm, scale=0.1, size=nchain)
        logm = np.log10(prosp['extras']['stellar_mass']['chain'])
        delm = np.clip(np.log10(prosp['extras']['stellar_mass']['chain']/10**logm_fast),delm_lim[0],delm_lim[1])
        ssfr = np.log10(prosp['extras']['ssfr_30']['chain'])
        delssfr = np.clip(np.log10(prosp['extras']['sfr_30']['chain']/uvir_chain),delssfr_lim[0],delssfr_lim[1])
        logsfr = np.clip(np.log10(prosp['extras']['sfr_30']['chain']),logsfr_lim[0],logsfr_lim[1])
        logsfr_p_uvir = np.clip(np.log10(uvir_chain),logsfr_lim[0],logsfr_lim[1])
        g1,_,_ = np.histogram2d(logm_fast,delm, normed=True,weights=prosp['weights'],
                                bins=[outg['grids']['logm_fast'],outg['grids']['delm']])
        g2,_,_ = np.histogram2d(ssfr,delssfr,normed=True,weights=prosp['weights'], 
                                bins=[outg['grids']['ssfr'],outg['grids']['delssfr']])
        g3,_,_ = np.histogram2d(logm_fast,logsfr,normed=True,weights=prosp['weights'], 
                                bins=[outg['grids']['logm'],outg['grids']['logsfr']])
        g4,_,_ = np.histogram2d(logm_fast,logsfr_p_uvir,normed=True,weights=prosp['weights'], 
                                bins=[outg['grids']['logm'],outg['grids']['logsfr']])

        outg['grids']['logm_delm'] += [g1]
        outg['grids']['ssfr_delssfr'] += [g2]
        outg['grids']['logm_logsfr'] += [g3]
        outg['grids']['logm_loguvirsfr'] += [g4]

    ### turn everything into numpy arrays
    for k1 in outprosp.keys():
        for k2 in outprosp[k1].keys():
            outprosp[k1][k2] = np.array(outprosp[k1][k2])
    for k1 in outprosp_fast.keys():
        for k2 in outprosp_fast[k1].keys():
            outprosp_fast[k1][k2] = np.array(outprosp_fast[k1][k2])
    for key in outfast: outfast[key] = np.array(outfast[key])

    # and for outg
    for k1 in outg.keys():
        if type(outg[k1]) == dict:
            for k2 in outg[k1].keys(): outg[k1][k2] = np.array(outg[k1][k2])
        else:
            outg[k1] = np.array(outg[k1])

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
           'phot_mag': np.array(phot_mag),
           'phot_sn': np.array(phot_sn),
           'phot_obslam': np.array(phot_obslam),
           'phot_restlam': np.array(phot_restlam),
           'phot_fname': np.array(phot_fname)
          }

    ### dump files and return
    hickle.dump(out,open(filename, "w"))
    hickle.dump(outg,open(filename_grid, "w"))

    return out, outg

def do_all(runname='td_new', runname_fast=None,outfolder=None,**opts):

    if outfolder is None:
        outfolder = os.getenv('APPS') + '/prospector_alpha/plots/'+runname+'/fast_plots/'
        outtable = os.getenv('APPS') + '/prospector_alpha/plots/'+runname+'/fast_plots/tables/'
        if not os.path.isdir(outfolder):
            os.makedirs(outfolder)
            os.makedirs(outfolder+'data/')

    # load all data 
    fastfile = outfolder+'data/fastcomp.h5'
    gridfile = outfolder+'data/m_sfr_grid.h5'
    data, datag = collate_data(runname,runname_fast,filename=fastfile,filename_grid=gridfile,**opts)

    # different plot options based on sample size
    popts = {'fmt':'o', 'capthick':1.5,'elinewidth':1.5,'alpha':0.8,'color':'0.3','ms':5,'markeredgecolor':'k'} # for small samples
    if len(data['uvir_sfr']) > 400:
        popts = {'fmt':'o', 'capthick':.15,'elinewidth':.15,'alpha':0.35,'color':'0.3','ms':2, 'errorevery': 5000}
    if len(data['uvir_sfr']) > 4000:
        popts = {'fmt':'o', 'capthick':.05,'elinewidth':.05,'alpha':0.2,'color':'0.3','ms':0.5, 'errorevery': 5000}

    phot_residuals_by_flux(data,outfolder,popts)

    sfr_m_grid(data, datag, outfolder+'conditional_sfr_m.png',outfile=outfolder+'data/conditional_sfr_fit.h5')
    sfr_m_grid(data, datag, outfolder+'conditional_sfr_m_nofix.png',fix=False,outfile=outfolder+'data/conditional_sfr_fit_nofix.h5')
    dm_dsfr_grid(data, datag, outfolder, outtable)
    deltam_with_redshift(data['fast'], data['prosp'], data['fast']['z'], outfolder+'deltam_vs_z.png', filename=outfolder+'data/masscomp.h5')

    # mass_met_age_z(data, outfolder, outtable, popts) # this is now deprecated
    deltam_spearman(data['fast'],data['prosp'],
                    outfolder+'deltam_spearman_fast_to_palpha.png',popts)

    # star-forming sequence.
    idx = (data['uvir_sfr'] > 0) & (data['fast']['uvj_prosp'] < 3) # to make it look like Kate's selection
    zred = data['fast']['z']
    uvir_sfr, prosp_sfr = np.log10(data['uvir_sfr']), np.log10(data['prosp']['sfr_30']['q50'])
    fast_mass, prosp_mass = data['fast']['stellar_mass'], data['prosp']['stellar_mass']['q50']

    star_forming_sequence(uvir_sfr[idx], fast_mass[idx], zred[idx],
                          outfolder+'star_forming_sequence_uvir.png',popts,
                          xlabel='[FAST]', ylabel='[UV+IR]')

    star_forming_sequence(prosp_sfr[idx], fast_mass[idx], zred[idx],
                          outfolder+'star_forming_sequence_prospector.png',popts,
                          xlabel='[FAST]', ylabel='[Prospector]')

    star_forming_sequence(prosp_sfr, fast_mass, zred,
                          outfolder+'star_forming_sequence_allgals_prospector.png',popts,
                          xlabel='[FAST]', ylabel='[Prospector]',outfile=outfolder+'data/sfrcomp.h5',
                          plt_whit=np.log10(data['prosp']['model_uvir_sfr']['q50']))

    star_forming_sequence(np.log10(data['prosp']['model_uvir_sfr']['q50']), fast_mass, zred,
                          outfolder+'star_forming_sequence_uvir_allgals_prospector.png',popts,
                          xlabel='[FAST]', ylabel=r'[UV+IR$_{\mathrm{model}}$]',outfile=outfolder+'data/sfrcomp_uvir.h5')

    uvir_comparison(data,outfolder+'ssfr_uvir_comparison', popts, filename=outfolder+'data/ssfrcomp.h5', ssfr=True)

    # if we have FAST-mimic runs, do a thorough comparison
    # else just do Prospector-FAST
    fast_comparison(data['fast'],data['prosp'],data['labels'],data['pnames'],
                    outfolder+'fast_to_palpha_comparison.png',popts)

    if runname_fast is not None:
        fast_comparison(data['fast'],data['prosp_fast'],data['labels'],data['pnames'],
                        outfolder+'fast_to_fastmimic_comparison.png',popts,plabel='FAST-mimic')
        fast_comparison(data['fast'],data['prosp_fast'],data['labels'],data['pnames'],
                        outfolder+'fast_to_fastmimic_bfit_comparison.png',popts,plabel='FAST-mimic',bfit=True)
        fast_comparison(data['prosp_fast'],data['prosp'],data['labels'],data['pnames'],
                        outfolder+'fastmimic_to_palpha_comparison.png',popts,flabel='FAST-mimic')
        deltam_spearman(data['fast'],data['prosp_fast'],
                        outfolder+'deltam_spearman_fast_to_fastmimic.png',popts)
        deltam_spearman(data['prosp_fast'],data['prosp'],
                        outfolder+'deltam_spearman_fastmimic_to_palpha.png',popts)        
        deltam_spearman(data['fast'],data['prosp'],
                        outfolder+'deltam_spearman_fast_to_palpha.png',popts)  

    phot_residuals(data,outfolder,popts)
    prospector_versus_z(data,outfolder+'prospector_versus_z.png',popts)
    sfr_mass_density_comparison(data,outfolder=outfolder)
    print 1/0
    # full UV+IR comparison
    #uvir_comparison(data,outfolder+'sfr_uvir_comparison', popts, ssfr=False)
    #uvir_comparison(data,outfolder+'sfr_uvir_comparison_model',  popts, model_uvir = True, ssfr=False)
    #uvir_comparison(data,outfolder+'sfr_uvir_truelir_comparison_model',  popts, model_uvir = 'true_LIR', ssfr=False)
    #uvir_comparison(data,outfolder+'ssfr_uvir_comparison_model',  popts, model_uvir = True, ssfr=True)
    #uvir_comparison(data,outfolder+'ssfr_uvir_truelir_comparison_model',  popts, model_uvir = 'true_LIR', ssfr=True)

def sfr_ratio(data):
    """quick hack to investigate sfr(100) / sfr(30)
    conclusion: sfr(30) is what we should be using everywhere!
    """

    nsafe = 5700
    ssfr30 = np.log10(data['prosp']['ssfr_30']['q50'][:nsafe])
    ssfr100 = data['prosp']['ssfr_100']['q50'][:nsafe]
    ssfruvir = np.log10(data['prosp']['model_uvir_ssfr']['q50'][:nsafe])

    # ssfr(30) versus ssfr(100)
    fig, ax = plt.subplots(1,4, figsize=(15, 4))
    minmax = (-11,-8)

    x, y = prosp_dutils.running_median(ssfr100,ssfr30,nbins=10)
    ax[0].plot(x,y,color=red,lw=4,zorder=5)
    ax[0].plot(ssfr100,ssfr30,'o',ms=2,alpha=0.7,color='0.3')
    ax[0].set_xlabel('ssfr [100 Myr]')
    ax[0].set_ylabel('ssfr [30 Myr]')

    # ssfr UVIR versus ssfr(100)
    x, y = prosp_dutils.running_median(ssfr100,ssfruvir,nbins=10)
    ax[1].plot(x,y,color=red,lw=4,zorder=5)
    ax[1].plot(ssfr100,ssfruvir,'o',ms=2,alpha=0.7,color='0.3')
    ax[1].set_xlabel('ssfr [100 Myr]')
    ax[1].set_ylabel('ssfr [UVIR]')

    # ssfr UVIR versus ssfr(30)
    x, y = prosp_dutils.running_median(ssfr30,ssfruvir,nbins=10)
    ax[2].plot(x,y,color=red,lw=4,zorder=5)
    ax[2].plot(ssfr30,ssfruvir,'o',ms=2,alpha=0.7,color='0.3')
    ax[2].set_xlabel('ssfr [30 Myr]')
    ax[2].set_ylabel('ssfr [UVIR]')

    # ssfr UVIR versus ratio
    x, y = prosp_dutils.running_median(ssfruvir,ssfr30-ssfr100,nbins=10)
    ax[3].plot(x,y,color=red,lw=4,zorder=5)
    ax[3].plot(ssfruvir,ssfr30-ssfr100,'o',ms=2,alpha=0.7,color='0.3')
    ax[3].set_xlabel('ssfr [UVIR]')
    ax[3].set_ylabel('log(ssfr30/ssfr100)')

    for a in ax[:3]:
        a.set_xlim(minmax)
        a.set_ylim(minmax)
        a.plot(minmax,minmax,linestyle='--',color='blue',zorder=3)
    ax[3].set_xlim(minmax)
    ax[3].set_ylim(-2,2)

    plt.tight_layout()
    plt.show()

def fast_comparison(fast,prosp,parlabels,pnames,outname,popts,flabel='FAST',plabel='Prospector',bfit=False):
    
    fig, axes = plt.subplots(2, 2, figsize = (6,6))
    axes = np.ravel(axes)

    for i,par in enumerate(['stellar_mass','sfr_30','dust2','avg_age']):

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
        if par[:3] == 'sfr' or par == 'avg_age': 
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

def sfr_m_grid(data,datag,outname,fix=True,outfile=None):
    """plots the median of the CONDITIONAL PDF for star formation rate(M) for ALL GALAXIES
    """

    # plot information
    fs, tick_fs, lw, ms = 11, 9.5, 3, 3.8
    #mcomplete = np.array([8.3,8.88,9.67,9.86])
    #mcomplete_new = np.array([8.45,8.96,9.29,9.45])

    opt = {
               'xlim': (8.5,11.5),
               'ylim': (-2,3),
               'xtitle': 'log(M$_{\mathrm{FAST}}$/M$_{\odot}$)',
               'ytitle': 'log(SFR) [M$_{\odot}$/yr]',
               'ytitle2': r'$\sigma$ [dex]',
               'xpar': data['fast']['stellar_mass'],
               'ypar1': data['prosp']['sfr_30'],
               'ypar2': data['prosp']['model_uvir_sfr'],
               'grid1': datag['grids']['logm_logsfr'],
               'grid2': datag['grids']['logm_loguvirsfr'],
               'xbins': datag['grids']['logm'],
               'ybins': datag['grids']['logsfr'],
               'xt': 0.98, 'yt': 0.05, 'ha': 'right'
               }
    uvir_color = '#dd1c77'

    # redshift bins + colors
    zbins = np.linspace(0.5,2.5,5)
    nbins = len(zbins)-1
    zlabels = ['$'+"{0:.1f}".format(zbins[i])+'<z<'+"{0:.1f}".format(zbins[i+1])+'$' for i in range(nbins)]
    from plot_sample_selection import mass_completeness
    mcomplete = mass_completeness((zbins[1:]+zbins[:-1])/2.)

    # plot geometry
    fig, ax = plt.subplots(2, 2, figsize = (10.5,6.5))
    fig.subplots_adjust(right=0.985,left=0.49,hspace=0.065,wspace=0.065,top=0.95,bottom=0.1)
    ax = np.ravel(ax)
    bigax = fig.add_axes([0.09, 0.275, 0.31, 0.5])

    # grid information
    ngrid = opt['xbins'].shape[0]
    dx, dy = opt['xbins'][1] - opt['xbins'][0], opt['ybins'][1] - opt['ybins'][0]
    xmid = (opt['xbins'][:-1] + opt['xbins'][1:])*0.5
    ymid = (opt['ybins'][:-1] + opt['ybins'][1:])*0.5

    # out variables
    a1, a2, b, a1_uvir, a2_uvir, b_uvir = [], [], [], [], [], []

    for i in range(nbins):

        # conditional PDF in our redshift bin
        idx = (data['fast']['z'] > zbins[i]) & (data['fast']['z'] <= zbins[i+1])
        grid1 = sum(np.nan_to_num(opt['grid1'][idx,:,:]))
        grid2 = sum(np.nan_to_num(opt['grid2'][idx,:,:]))

        # conditional PDF
        xidx = (opt['xbins']+dx > opt['xlim'][0]) & (opt['xbins']-dx <= opt['xlim'][1])
        yidx = (opt['ybins']+dy > opt['ylim'][0]) & (opt['ybins']-dy <= opt['ylim'][1])

        # index to include only galaxies in ylim, xlim
        plotgrid1 = grid1[xidx[1:] & xidx[:-1], :]
        plotgrid1 = plotgrid1[:,yidx[1:] & yidx[:-1]].T
        plotgrid1 = plotgrid1 / plotgrid1.max(axis=0)[None,:]

        plotgrid2 = grid2[xidx[1:] & xidx[:-1], :]

        # plot the PDF
        X, Y = np.meshgrid(opt['xbins'][xidx], opt['ybins'][yidx])
        ax[i].pcolormesh(X, Y, plotgrid1, cmap='Greys',label=zlabels[i],alpha=0.5,edgecolor='None')

        # calculate percentiles of the conditional PDF
        # and errors
        ngrid = grid1.shape[0]
        errs_1 = (opt['ypar1']['q84'][idx] - opt['ypar1']['q16'][idx])/2.
        errs_2 = (opt['ypar2']['q84'][idx] - opt['ypar2']['q16'][idx])/2.
        q50_1, q84_1, q16_1, err_1, q50_2, q84_2, q16_2, err_2, avg_1, avg_2 = [np.empty(ngrid) for j in range(10)] 
        for j in range(ngrid):
            q50_1[j], q84_1[j], q16_1[j] = weighted_quantile(ymid, np.array([0.5, 0.84, 0.16]), weights=grid1[j,:])
            q50_2[j], q84_2[j], q16_2[j] = weighted_quantile(ymid, np.array([0.5, 0.84, 0.16]), weights=grid2[j,:])

            avg_1[j], avg_2[j] = np.nan, np.nan
            if grid1[j,:].sum() > 0:
                avg_1[j] = np.log10(np.average(10**ymid, weights=grid1[j,:]))
            if grid2[j,:].sum() > 0:
                avg_2[j] = np.log10(np.average(10**ymid, weights=grid2[j,:]))

            in_grid = (opt['xpar'][idx] > opt['xbins'][j]) & (opt['xpar'][idx] <= opt['xbins'][j+1])
            err_1[j] = np.median(errs_1[in_grid])
            err_2[j] = np.median(errs_2[in_grid])

        # plot percentiles
        #ax[i].plot(xmid, avg_1, color=cmap[i], lw=2, linestyle='-', zorder=6,label='Prospector')
        #bigax.plot(xmid, avg_1, color=cmap[i], lw=3.0, linestyle='-', zorder=6,label=zlabels[i])
        #ax[i].plot(xmid, avg_2, color=uvir_color, lw=2, linestyle='-.', zorder=6,label='UV+IR')

        # fit above the mass-completeness limit
        def fit_eqn(logm,b,a1,a2):
            idx = (logm > 10.2)
            logsfr = a2*(logm-10.2)+b
            logsfr[idx] = a1*(logm[idx]-10.2)+b
            return 10**logsfr

        # for z=2-2.5
        def fit_eqn_fixedslope(logm,b,a1):
            idx = (logm > 10.2)
            logsfr = 0.87*(logm-10.2)+b
            logsfr[idx] = a1*(logm[idx]-10.2)+b
            return 10**logsfr
        eqn = fit_eqn
        if (i == 3) & (fix):
            eqn = fit_eqn_fixedslope
            a2.append(0.87)

        for n, (lab, col, grd,sty) in enumerate(zip(['Prospector','UV+IR'],[cmap[i],uvir_color],[grid1,grid2],['-','-.'])):
            idx_cmp = (xmid > mcomplete[i])
            xf, yf = np.meshgrid(xmid[idx_cmp],ymid)
            xf, yf, weights = xf.flatten(), 10**yf.flatten(), grd[idx_cmp,:].T.flatten()
            gidx = weights > 0
            popts, pcov = curve_fit(eqn,xf[gidx],yf[gidx],sigma=1./weights[gidx])
            ax[i].plot(xmid,np.log10(eqn(xmid,*popts)),lw=2,color=col,label=lab,linestyle=sty)

            if n == 0:
                bigax.plot(xmid, np.log10(eqn(xmid,*popts)), color=col, lw=3.0, label=zlabels[i])
                a1 += [popts[1]]
                b += [popts[0]]
                try:
                    a2 += [popts[2]]
                except:
                    a2 += [a2[-1]]
            else:
                a1_uvir += [popts[1]]
                b_uvir += [popts[0]]
                try:
                    a2_uvir += [popts[2]]
                except:
                    a2_uvir += [a2[-1]]

        # redshift label
        ax[i].text(opt['xt'],opt['yt'],zlabels[i],ha=opt['ha'],fontsize=fs,transform=ax[i].transAxes)

        # labels
        if i > 1:
            ax[i].set_xlabel(opt['xtitle'],fontsize=fs)
            bigax.set_xlabel(opt['xtitle'],fontsize=fs*1.3)
            for tl in ax[i].get_xticklabels():tl.set_fontsize(fs)
            for tl in bigax.get_xticklabels():tl.set_fontsize(fs*1.3)
            ax[i].xaxis.set_major_locator(MaxNLocator(4))
            bigax.xaxis.set_major_locator(MaxNLocator(4))
        else:
            for tl in ax[i].get_xticklabels():tl.set_visible(False)
        if (i % 2 == 0):
            ax[i].set_ylabel(opt['ytitle'],fontsize=fs)
            bigax.set_ylabel(opt['ytitle'],fontsize=fs*1.3)
            for tl in ax[i].get_yticklabels():tl.set_fontsize(fs)
            for tl in bigax.get_yticklabels():tl.set_fontsize(fs*1.3)
        else:
            for tl in ax[i].get_yticklabels():tl.set_visible(False)

    # labels, limits
    for a in ax.tolist()+[bigax]: 
        a.set_xlim(opt['xlim'])
        a.set_ylim(opt['ylim'])

    # turn off the bottom y-label
    ax[0].legend(loc=2,prop={'size':fs*0.9}, scatterpoints=1,fancybox=True)

    # finish off bigax plotting and labels
    bigax.axhline(0, linestyle='-.', color='0.1', lw=2,zorder=10)
    bigax.legend(loc=2, prop={'size':fs*0.9}, scatterpoints=1,fancybox=True,ncol=1)

    plt.savefig(outname,dpi=dpi)
    plt.close()

    # save
    if outfile is not None:
        out = {'a1': a1, 'a2': a2, 'b': b, 'a1_uvir': a1_uvir,'a2_uvir': a2_uvir, 'b_uvir': b_uvir}
        hickle.dump(out,open(outfile, "w"))

def dm_dsfr_grid(data,datag,outfolder,outtable,normalize=True):

    # plot information
    fs, tick_fs, lw, ms = 11, 9.5, 3, 3.8
    dmopts = {
               'xlim': (8.5,11.5),
               'ylim': (-0.7,0.7),
               'xtitle': 'log(M$_{\mathrm{FAST}}$/M$_{\odot}$)',
               'ytitle': 'log(M$_{\mathrm{Prospector}}$/M$_{\mathrm{FAST}}$)',
               'ytitle2': r'$\sigma$ [dex]',
               'xpar': data['fast']['stellar_mass'],
               'ypar': {q: data['prosp']['stellar_mass'][q] - data['fast']['stellar_mass'] for q in ['q50','q84','q16']},
               'grid': datag['grids']['logm_delm'],
               'xbins': datag['grids']['logm_fast'],
               'ybins': datag['grids']['delm'],
               'outname': outfolder+'conditional_dm_v_z.png',
               'table_out': outtable+'conditional_dm.dat',
               'xt': 0.97, 'yt': 0.04, 'ha': 'right',
               'legend_loc': 4, 'legend_ncol': 1,
               'dens_power': 1., 'xticks': [9,10,11]
               }

    dsfropts = {
               'xlim': (-10.8,-8),
               'ylim': (-2,0.999),
               'xtitle': 'log(sSFR$_{\mathrm{Prospector}}$/yr$^{-1}$)',
               'ytitle': 'log(SFR$_{\mathrm{Prospector}}$/SFR$_{\mathrm{UV+IR}}$)',
               'ytitle2': r'$\sigma$ [dex]',
               'xpar': data['prosp']['ssfr_30']['q50'],
               'ypar': data['prosp']['sfr_ratio'],
               'grid': datag['grids']['ssfr_delssfr'],
               'xbins': datag['grids']['ssfr'],
               'ybins': datag['grids']['delssfr'],
               'outname': outfolder+'conditional_dsfr_vs_z.png',
               'table_out': outtable+'conditional_dsfr.dat',
               'xt': 0.97, 'yt': 0.04, 'ha': 'right',
               'legend_loc': 4, 'legend_ncol': 1,
               'dens_power': 0.5, 'xticks': [-11,-10,-9,-8]
               }

    # redshift bins + colors
    zbins = np.linspace(0.5,2.5,5)
    nbins = len(zbins)-1
    zlabels = ['$'+"{0:.1f}".format(zbins[i])+'<z<'+"{0:.1f}".format(zbins[i+1])+'$' for i in range(nbins)]

    # loop over options
    for opt in [dmopts,dsfropts]:

        fig, a1 = plt.subplots(2, 2, figsize = (10.5,6.5))
        fig.subplots_adjust(right=0.985,left=0.49,hspace=0.065,wspace=0.065,top=0.95,bottom=0.1)
        a1 = np.ravel(a1)
        bigax = fig.add_axes([0.09, 0.275, 0.31, 0.5])

        # grid information
        ngrid = opt['xbins'].shape[0]
        dx, dy = opt['xbins'][1] - opt['xbins'][0], opt['ybins'][1] - opt['ybins'][0]
        xmid = (opt['xbins'][:-1] + opt['xbins'][1:])*0.5
        ymid = (opt['ybins'][:-1] + opt['ybins'][1:])*0.5

        for i in range(nbins):

            # conditional PDF in our redshift bin
            idx = (data['fast']['z'] > zbins[i]) & (data['fast']['z'] <= zbins[i+1])
            grid = sum(np.nan_to_num(opt['grid'][idx,:,:]))

            # conditional PDF
            xidx = (opt['xbins']+dx > opt['xlim'][0]) & (opt['xbins']-dx <= opt['xlim'][1])
            yidx = (opt['ybins']+dy > opt['ylim'][0]) & (opt['ybins']-dy <= opt['ylim'][1])
            plotgrid = grid[xidx[1:] & xidx[:-1], :]
            plotgrid = plotgrid[:,yidx[1:] & yidx[:-1]].T
            if normalize:
                plotgrid = plotgrid / plotgrid.max(axis=0)[None,:]
            X, Y = np.meshgrid(opt['xbins'][xidx], opt['ybins'][yidx])
            a1[i].pcolormesh(X, Y, plotgrid, cmap='Greys',label=zlabels[i],alpha=0.5,edgecolor='None')

            # calculate percentiles of the conditional PDF
            # and errors
            ngrid = grid.shape[0]
            errs = (opt['ypar']['q84'][idx] - opt['ypar']['q16'][idx])/2.
            q50, q84, q16, err, median, mdown, mup = [np.empty(ngrid) for j in range(7)] 
            for j in range(ngrid):
                q50[j], q84[j], q16[j] = weighted_quantile(ymid, np.array([0.5, 0.84, 0.16]), weights=grid[j,:])
                in_grid = (opt['xpar'][idx] > opt['xbins'][j]) & (opt['xpar'][idx] <= opt['xbins'][j+1])
                err[j] = np.median(errs[in_grid])
                median[j] = np.median(opt['ypar']['q50'][idx][in_grid])
                mdown[j] = np.median(opt['ypar']['q84'][idx][in_grid])
                mup[j] = np.median(opt['ypar']['q16'][idx][in_grid])

            # plot percentiles
            a1[i].plot(xmid, q50, color=cmap[i], lw=2, linestyle='-', zorder=6,label='Prospector')
            a1[i].plot(xmid, q84, color=cmap[i], lw=2, linestyle='--', zorder=6)
            a1[i].plot(xmid, q16, color=cmap[i], lw=2, linestyle='--', zorder=6)
            bigax.plot(xmid, q50, color=cmap[i], lw=3.0, linestyle='-', zorder=6,label=zlabels[i])

            # redshift label
            a1[i].text(opt['xt'],opt['yt'],zlabels[i],ha=opt['ha'],fontsize=fs,transform=a1[i].transAxes)
            a1[i].axhline(0, linestyle='-.', color='0.1', lw=2,zorder=10)

            # labels
            if i > 1:
                a1[i].set_xlabel(opt['xtitle'],fontsize=fs)
                bigax.set_xlabel(opt['xtitle'],fontsize=fs*1.3)
                for tl in a1[i].get_xticklabels():tl.set_fontsize(fs)
                for tl in bigax.get_xticklabels():tl.set_fontsize(fs*1.3)
                a1[i].xaxis.set_ticks(opt['xticks'])
                bigax.xaxis.set_ticks(opt['xticks'])
            else:
                for tl in a1[i].get_xticklabels():tl.set_visible(False)
            if (i % 2 == 0):
                a1[i].set_ylabel(opt['ytitle'],fontsize=fs)
                bigax.set_ylabel(opt['ytitle'],fontsize=fs*1.3)
                for tl in a1[i].get_yticklabels():tl.set_fontsize(fs)
                for tl in bigax.get_yticklabels():tl.set_fontsize(fs*1.3)
            else:
                for tl in a1[i].get_yticklabels():tl.set_visible(False)

        # labels, limits
        for a in a1.tolist()+[bigax]: 
            a.set_xlim(opt['xlim'])
            a.set_ylim(opt['ylim'])

        # turn off the bottom xtick label, ytick label
        a1[0].yaxis.get_major_ticks()[0].label1.set_visible(False)

        # finish off bigax plotting and labels
        bigax.axhline(0, linestyle='-.', color='0.1', lw=2,zorder=10)
        bigax.legend(loc=opt['legend_loc'], prop={'size':fs*0.9},
                       scatterpoints=1,fancybox=True,ncol=opt['legend_ncol'])

        plt.savefig(opt['outname'],dpi=dpi)
        plt.close()

def deltam_spearman(fast,prosp,outname,popts):

    # try some y-variables
    if type(fast['avg_age']) == type({}):
        mfast = fast['stellar_mass']['q50']
        tfast = fast['avg_age']['q50']
        dfast = fast['dust2']['q50']
    else:
        tfast = fast['avg_age']
        dfast = fast['dust2']
        mfast = fast['stellar_mass']
    params = [np.log10(prosp['half_time']['q50']/tfast),
              prosp['dust2']['q50'] - dfast]
    xlabels = [r'log(t$_{\mathrm{Prospector}}$/t$_{\mathrm{FAST}}$)',
               r'$\tau_{\mathrm{diffuse,Prospector}}-\tau_{\mathrm{diffuse,FAST}}$']
    xlims = [(-3,3),
             (-2,2)]
    ylim = (-1.5,1.5)

    if 'massmet_2' in prosp.keys():
        params.append(prosp['massmet_2']['q50'])
        xlabels.append(r'log(Z$_{\mathrm{Prospector}}$/Z$_{\odot}$)')
        xlims.append((-2,0.3))

    # plot geometry
    ysize = 2.75
    fig, ax = plt.subplots(1, len(params), figsize = (ysize*len(params)+0.1,ysize))
    ax = ax.ravel()
    medopts = {'marker':' ','alpha':0.95,'color':'red','ms': 7,'mec':'k','zorder':5}

    # x variable
    delta_mass = prosp['stellar_mass']['q50'] - mfast

    for i, par in enumerate(params):
        ax[i].errorbar(par,delta_mass,**popts)
        ax[i].set_ylabel(r'log(M$_{\mathrm{Prospector}}$/M$_{\mathrm{FAST}}$)')
        ax[i].set_xlabel(xlabels[i])

        ax[i].set_xlim(xlims[i])
        ax[i].set_ylim(ylim)
        ax[i].text(0.02,0.05,r'$\rho_{\mathrm{S}}$='+'{:1.2f}'.format(spearmanr(delta_mass,par)[0]),transform=ax[i].transAxes)

        ax[i].axhline(0, linestyle='--', color='k',lw=1,zorder=10)
        ax[i].axvline(0, linestyle='--', color='k',lw=1,zorder=10)

        # running median
        in_plot = (delta_mass > ylim[0]) & (delta_mass < ylim[1])
        x, y, bincount = prosp_dutils.running_median(par[in_plot],delta_mass[in_plot],avg=False,return_bincount=True,nbins=20)
        x, y = x[bincount > nbin_min], y[bincount > nbin_min]
        ax[i].plot(x,y, **medopts)

    plt.tight_layout()
    plt.savefig(outname,dpi=dpi)
    plt.close()

def prospector_versus_z(data,outname,popts):
    
    fig, axes = plt.subplots(2, 3, figsize = (10,6.66))
    axes = np.ravel(axes)

    toplot = ['fagn','sfr_30','ssfr_30','avg_age','massmet_2','dust2']
    for i,par in enumerate(toplot):
        zred = data['fast']['z']
        yprosp, yprosp_up, yprosp_down = data['prosp'][par]['q50'], data['prosp'][par]['q84'], data['prosp'][par]['q16']

        ### clip SFRs
        if par[:3] == 'sfr':
            minimum = minsfr
        elif par == 'ssfr_30':
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
        if par == 'avg_age':
            n = 50
            zred = np.linspace(zred.min(),zred.max(),n)
            tuniv = WMAP9.age(zred).value
            axes[i].plot(zred,tuniv,'--',lw=2,zorder=-1, color=red)
            axes[i].text(zred[n/2]*1.1,tuniv[n/2]*1.1, r't$_{\mathrm{univ}}$',rotation=-50,color=red,weight='bold')

        # logscale
        if par == 'sfr_30' or par == 'fagn':
            axes[i].set_yscale('log',nonposx='clip',subsy=([3]))
            axes[i].yaxis.set_minor_formatter(FormatStrFormatter('%2.4g'))
            axes[i].yaxis.set_major_formatter(FormatStrFormatter('%2.4g'))
    plt.tight_layout()
    plt.savefig(outname,dpi=dpi)
    plt.close()

def phot_residuals_by_flux(data,outfolder,popts_orig):
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
    medopts = {'marker':' ','alpha':0.5,'zorder':5,'linestyle':'-','lw':0.8}
    fontsize = 16
    fs = 10
    ylim_chi = (-4,4)
    ylim_percentile = (-0.65,0.65)

    # residuals by SN
    fig, ax = plt.subplots(5,2, figsize=(8,18))
    for i,field in enumerate(field_names):

        # determine field and filter names
        fidx = fields == field
        fnames = np.unique(filters[fidx])

        # determine wavelength for filters so we can sort
        lam = []
        fnames = np.array(['irac1','irac2','irac3','irac4','mips'])
        for filter in fnames:
            fmatch = filters[fidx] == filter
            lam += [data['phot_obslam'][fidx][fmatch][0]]
        argsort = np.array(lam).argsort()

        # plot the median for these
        colormap = get_cmap(len(fnames))
        for j,filter in enumerate(fnames[argsort]):
            fmatch = filters[fidx] == filter
            sn = np.abs(data['phot_sn'])[fidx][fmatch]

            x,y,bincount = prosp_dutils.running_median(sn,data['phot_chi'][fidx][fmatch],return_bincount=True,nbins=20)
            x,y = x[bincount > 10], y[bincount > 10]
            ax[i,0].plot(x,y, color=colormap(j), **medopts)

            x,y,bincount = prosp_dutils.running_median(sn,data['phot_percentile'][fidx][fmatch],return_bincount=True,nbins=20)
            x,y = x[bincount > 10], y[bincount > 10]
            ax[i,1].plot(x,y, color=colormap(j), **medopts)

            ax[i,0].text(0.02,0.93-j*0.035,filter,color=colormap(j),fontsize=fs,transform=ax[i,0].transAxes)
        ax[i,0].text(0.98,0.93,field,color=colormap(j),fontsize=fs,transform=ax[i,0].transAxes,ha='right')

        ax[i,0].set_ylabel('(f$_{\mathrm{obs}}$-f$_{\mathrm{model}}$)/$\sigma_{\mathrm{obs}}$',fontsize=fontsize)
        ax[i,1].set_ylabel('(f$_{\mathrm{obs}}$-f$_{\mathrm{model}}$)/f$_{\mathrm{obs}}$',fontsize=fontsize)

        ax[i,0].set_xlabel('S/N',fontsize=fontsize)
        ax[i,1].set_xlabel('S/N',fontsize=fontsize)


        ax[i,0].set_ylim(ylim_chi)
        ax[i,1].set_ylim(ylim_percentile)

    plt.tight_layout()
    plt.savefig(outfolder+'residual_by_sn.png',dpi=dpi)
    plt.close()
 
    # residuals by magnitude
    fig, ax = plt.subplots(5,2, figsize=(8,18))
    for i,field in enumerate(field_names):

        # determine field and filter names
        fidx = fields == field
        fnames = np.unique(filters[fidx])
        fnames = np.array(['irac1','irac2','irac3','irac4','mips'])

        # determine wavelength for filters so we can sort
        lam = []
        for filter in fnames:
            fmatch = filters[fidx] == filter
            lam += [data['phot_obslam'][fidx][fmatch][0]]
        argsort = np.array(lam).argsort()

        # plot the median for these
        colormap = get_cmap(len(fnames))
        for j,filter in enumerate(fnames[argsort]):
            fmatch = filters[fidx] == filter
            mag = np.abs(data['phot_mag'])[fidx][fmatch]
            gidx = np.isfinite(mag)

            x,y,bincount = prosp_dutils.running_median(mag[gidx],data['phot_chi'][fidx][fmatch][gidx],return_bincount=True,nbins=20)
            x,y = x[bincount > 10], y[bincount > 10]
            ax[i,0].plot(x,y, color=colormap(j), **medopts)

            x,y,bincount = prosp_dutils.running_median(mag[gidx],data['phot_percentile'][fidx][fmatch][gidx],return_bincount=True,nbins=20)
            x,y = x[bincount > 10], y[bincount > 10]
            ax[i,1].plot(x,y, color=colormap(j), **medopts)

            ax[i,0].text(0.02,0.93-j*0.035,filter,color=colormap(j),fontsize=fs,transform=ax[i,0].transAxes)

        ax[i,0].text(0.98,0.93,field,color=colormap(j),fontsize=fs,transform=ax[i,0].transAxes,ha='right')

        ax[i,0].set_ylabel('(f$_{\mathrm{obs}}$-f$_{\mathrm{model}}$)/$\sigma_{\mathrm{obs}}$',fontsize=fontsize)
        ax[i,1].set_ylabel('(f$_{\mathrm{obs}}$-f$_{\mathrm{model}}$)/f$_{\mathrm{obs}}$',fontsize=fontsize)

        ax[i,0].set_xlabel('magnitude',fontsize=fontsize)
        ax[i,1].set_xlabel('magnitude',fontsize=fontsize)


        ax[i,0].set_ylim(ylim_chi)
        ax[i,1].set_ylim(ylim_percentile)

    plt.tight_layout()
    plt.savefig(outfolder+'residual_by_magnitude.png',dpi=dpi)
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

def mass_met_age_z(data,outfolder,outtable,popts):
    """this is now replaced by conditionals! 2/26/18
    """

    # plot information
    fs, lw, ms = 12, 3, 5.5
    metopts = {
               'xlim': (9,11.49),
               'ylim': (-1.98,0.5),
               'sdss': np.loadtxt(os.getenv('APPS')+'/prospector_alpha/data/gallazzi_05_massmet.txt'),
               'nmassbins': 10,
               'xtitle': 'log(M/M$_{\odot}$)',
               'ytitle': 'log(Z/Z$_{\odot}$)',
               'xpar': data['prosp']['stellar_mass'],
               'xpar_fast': None,
               'ypar': data['prosp']['massmet_2'],
               'ypar_fast': None,
               'outname': outfolder+'massmet_vs_z.png',
               'table_out': outtable+'massmet.dat',
               'xt': 0.98, 'yt': 0.05, 'ha': 'right',
               'legend_loc': 4, 'legend_ncol': 1
               }
    
    ageopts = {
               'xlim': (9,11.49),
               'ylim': (0.05,13),
               'sdss': np.loadtxt(os.getenv('APPS')+'/prospector_alpha/data/gallazzi_05_age.txt'),
               'nmassbins': 10,
               'xtitle': 'log(M/M$_{\odot}$)',
               'ytitle': '<stellar age>/Gyr',
               'xpar': data['prosp']['stellar_mass'],
               'xpar_fast': data['fast']['stellar_mass'],
               'ypar': data['prosp']['avg_age'],
               'ypar_fast': data['fast']['avg_age'],
               'outname': outfolder+'age_vs_z.png',
               'table_out': outtable+'massage.dat',
               'xt': 0.98, 'yt': 0.05, 'ha': 'right',
               'legend_loc': 4, 'legend_ncol': 1
               }
    ageopts['sdss'][:,1:] = 10**ageopts['sdss'][:,1:]/1e9

    # redshift bins + colors
    zbins = np.linspace(0.5,2.5,5)
    nbins = len(zbins)-1
    zlabels = ['$'+"{0:.1f}".format(zbins[i])+'<z<'+"{0:.1f}".format(zbins[i+1])+'$' for i in range(nbins)]

    # loop over options
    for opt in [metopts,ageopts]:

        # plot geometry
        fig, axes = plt.subplots(2, 2, figsize = (10.5,6.5))
        fig.subplots_adjust(right=0.985,left=0.49,hspace=0.065,wspace=0.065,top=0.95,bottom=0.1)
        axes = np.ravel(axes)
        bigax = fig.add_axes([0.09, 0.275, 0.31, 0.5])

        # massbins
        massbins = np.linspace(opt['xlim'][0],opt['xlim'][1],opt['nmassbins']+1)
        q84_t, q50_t, q16_t, x_t = [[] for i in range(4)]
        for i in range(nbins):

            # the main show
            idx = (data['fast']['z'] > zbins[i]) & (data['fast']['z'] <= zbins[i+1])
            xm, xm_up, xm_down = [opt['xpar'][q][idx] for q in ['q50','q84','q16']]
            yz, yz_up, yz_down = [opt['ypar'][q][idx] for q in ['q50','q84','q16']]

            xerr = prosp_dutils.asym_errors(xm, xm_up, xm_down)
            yerr = prosp_dutils.asym_errors(yz, yz_up, yz_down)
            axes[i].errorbar(xm,yz,yerr=yerr,xerr=xerr,**popts)

            # z=0 relationship
            # for age, only put it on the summary plot
            lw_z0, color = 1.5, '0.5'
            if ('age' not in opt['table_out']):
                axes[i].plot(opt['sdss'][:,0], opt['sdss'][:,1], color=color, lw=lw_z0, zorder=-1, label='Gallazzi+05')
                axes[i].plot(opt['sdss'][:,0],opt['sdss'][:,2], color=color, lw=lw_z0, linestyle='--', zorder=-1)
                axes[i].plot(opt['sdss'][:,0],opt['sdss'][:,3], color=color, lw=lw_z0, linestyle='--', zorder=-1)
            else:
                xm_fast, yz_fast = opt['xpar_fast'][idx], opt['ypar_fast'][idx]
                x, y, bincount = prosp_dutils.running_median(xm_fast,yz_fast,avg=False,return_bincount=True,
                                                             weights=np.ones_like(xm_fast),bins=massbins)
                yerr = prosp_dutils.asym_errors(y[:,0], y[:,1], y[:,2])
                axes[i].errorbar(x, y[:,0], yerr=yerr, color=cmap[i], marker='o',lw=lw*0.5,label=zlabels[i],zorder=5,ms=ms,linestyle='--')
            if i == 0:
                bigax.plot(opt['sdss'][:,0], opt['sdss'][:,1], color=color, lw=lw_z0, zorder=-1, label='Gallazzi+05')
                bigax.plot(opt['sdss'][:,0],opt['sdss'][:,2], color=color, lw=lw_z0, linestyle='--', zorder=1)
                bigax.plot(opt['sdss'][:,0],opt['sdss'][:,3], color=color, lw=lw_z0, linestyle='--', zorder=1)

            # running median
            weights = ((xm_up-xm_down)/2.)**(-2)
            x, y, bincount = prosp_dutils.running_median(xm,yz,avg=False,return_bincount=True,weights=weights,bins=massbins)
            x, y = x[bincount > nbin_min], y[bincount > nbin_min]
            yerr = prosp_dutils.asym_errors(y[:,0], y[:,1], y[:,2])
            q84_t += y[:,1].tolist()
            q16_t += y[:,2].tolist()
            q50_t += y[:,0].tolist()
            x_t += x.tolist()
            axes[i].errorbar(x, y[:,0], yerr=yerr, color=cmap[i],marker='o',lw=lw,label=zlabels[i],zorder=5,ms=ms)
            bigax.plot(x, y[:,0], color=cmap[i],marker='o',lw=lw,label=zlabels[i],zorder=5,ms=ms*1.5)

            # redshift label
            axes[i].text(opt['xt'],opt['yt'],zlabels[i],ha=opt['ha'], fontsize=fs,transform=axes[i].transAxes)

            # max tuniv label
            if 'age' in opt['ytitle']:
                tuniv = WMAP9.age(zbins[i]).value
                axes[i].text(9.02,tuniv*1.11,'t$_{\mathrm{univ}}$',ha='left', fontsize=7.5,color='red')
                axes[i].axhline(tuniv,linestyle=':',color='red',lw=1,zorder=-1,alpha=0.9)
                
                # scales
                subsy, tickfs = ([]), 0.0
                if (i % 2 == 0):
                    subsy, tickfs = (1,3,6), fs
                axes[i].set_yscale('log', subsy=subsy)
                axes[i].yaxis.set_minor_formatter(FormatStrFormatter('%2.4g'))
                axes[i].yaxis.set_major_formatter(FormatStrFormatter('%2.4g'))
                for tl in axes[i].get_yticklabels():tl.set_visible(False)
                axes[i].tick_params('both', pad=3.5, labelsize=tickfs,size=3.5, width=1.0, which='both')

            # labels
            if i > 1:
                axes[i].set_xlabel(opt['xtitle'],fontsize=fs)
                bigax.set_xlabel(opt['xtitle'],fontsize=fs*1.3)
                for tl in axes[i].get_xticklabels():tl.set_fontsize(fs)
                for tl in bigax.get_xticklabels():tl.set_fontsize(fs*1.3)
            else:
                for tl in axes[i].get_xticklabels():tl.set_visible(False)
            if (i % 2 == 0):
                axes[i].set_ylabel(opt['ytitle'],fontsize=fs)
                bigax.set_ylabel('median '+ opt['ytitle'],fontsize=fs*1.3)
                for tl in axes[i].get_yticklabels():tl.set_fontsize(fs)
                for tl in bigax.get_yticklabels():tl.set_fontsize(fs*1.3)
            else:
                for tl in axes[i].get_yticklabels():tl.set_visible(False)

        # legends, scales, limits
        for a in [bigax]+axes.tolist():
            a.set_xlim(opt['xlim'])
            a.set_ylim(opt['ylim'])

        if 'age' in opt['ytitle']:
            bigax.set_yscale('log', subsy=(1,3,6))
            bigax.yaxis.set_minor_formatter(FormatStrFormatter('%2.4g'))
            bigax.yaxis.set_major_formatter(FormatStrFormatter('%2.4g'))
            for tl in bigax.get_yticklabels():tl.set_visible(False)
            bigax.tick_params('both', pad=3.5, labelsize=fs*1.3,size=3.5, width=1.0, which='both')

        bigax.legend(loc=opt['legend_loc'], prop={'size':fs*0.9},
                       scatterpoints=1,fancybox=True,ncol=opt['legend_ncol'])

        # save the table
        zlabs = []
        for n in range(nbins): zlabs += [zlabels[n]]*opt['nmassbins']
        odat = Table([zlabs, np.array(x_t), np.array(q50_t), np.array(q84_t), np.array(q16_t)], 
                      names=['z','log(M/M$_{\odot}$)', 'P50', 'P84', 'P16'])
        formats = {name: '%1.2f' for name in odat.columns.keys()}
        formats['z'] = str
        ascii.write(odat, opt['table_out'], format='aastex',overwrite=True,formats=formats)

        plt.savefig(opt['outname'],dpi=dpi)
        plt.close()

def star_forming_sequence(sfr,mass,zred,outname,popts,xlabel=None,ylabel=None,outfile=None,priors=False,
                          plt_whit=True):
    """ Plot <SFR(M)> for input mass, SFR
    """

    # Figure geometry
    fig, ax = plt.subplots(2, 2, figsize = (10.35,6))
    fig.subplots_adjust(right=0.985,left=0.5,hspace=0.0,wspace=0.0)
    bigax = fig.add_axes([0.09, 0.25, 0.31, 0.5])
    ax = np.ravel(ax)

    # colors and limits
    medopts = {'marker':' ','alpha':0.85,'zorder':5,'lw':3}
    corrected_opts = {'marker':' ','alpha':0.85,'color':'orange','zorder':5,'lw':1.5}
    fs, lw, ms = 12, 3, 5.5
    xlim = (8.5, 11.5)
    ylim = (0.01,900)
    xtitle = r'log(M/M$_{\odot}$) ' + xlabel
    ytitle = r'SFR/M$_{\odot}$ yr$^{-1}$ ' + ylabel

    # mass and redshift bins
    nbins = 4
    zbins = np.linspace(0.5,2.5,nbins+1)
    zlabels = ["{0:.1f}".format(zbins[i])+'<z<'+"{0:.1f}".format(zbins[i+1]) for i in range(nbins)]
    mbins = np.linspace(xlim[0],xlim[1],12)
    sfr = 10**sfr

    # Prospector prior
    ssfr_max = -8 # from Prospector physics
    prior_opts = {'linestyle':'--','color':'k','zorder':5,'lw':1.5}

    # Whitaker+14 SFR-M relationship
    zwhit = np.array([0.75, 1.25, 1.75, 2.25])
    logm_whit = np.linspace(xlim[0],xlim[1],50)
    whitopts = {'color':'#dd1c77','alpha':0.85,'lw':1.5,'zorder':5,'linestyle':'-.'}

    # let's go!
    mass_save, sfr_save = [], []
    for i in range(len(zbins)-1):

        # the data
        in_bin = (zred > zbins[i]) & \
                 (zred <= zbins[i+1]) & \
                 (np.isfinite(sfr))
        ax[i].errorbar(mass[in_bin], sfr[in_bin], **popts)

        # calculate and plot running average
        x, y = prosp_dutils.running_median(mass[in_bin], sfr[in_bin], bins=mbins,avg=True)
        ax[i].errorbar(x, y, color=cmap[i],**medopts)
        bigax.errorbar(x, y, color=cmap[i], label=zlabels[i], ms=ms, **medopts)

        # scales
        subsy, tickfs = ([]), 0.0
        if (i % 2 == 0):
            subsy, tickfs = ([1]), fs
        ax[i].set_yscale('log', subsy=subsy)
        ax[i].yaxis.set_minor_formatter(FormatStrFormatter('%2.4g'))
        ax[i].yaxis.set_major_formatter(FormatStrFormatter('%2.4g'))
        for tl in ax[i].get_yticklabels():tl.set_visible(False)
        ax[i].tick_params('both', pad=3.5, labelsize=tickfs,size=3.5, width=1.0, which='both')

        # labels
        if i > 1:
            ax[i].set_xlabel(xtitle,fontsize=fs)
            bigax.set_xlabel(xtitle,fontsize=fs*1.3)
            for tl in ax[i].get_xticklabels():tl.set_fontsize(fs)
            ax[i].xaxis.set_major_locator(MaxNLocator(3))
        else:
            for tl in ax[i].get_xticklabels():tl.set_visible(False)
        if (i % 2 == 0):
            ax[i].set_ylabel(ytitle,fontsize=fs)
            bigax.set_ylabel(ytitle,fontsize=fs*1.3)
            for tl in ax[i].get_yticklabels():tl.set_fontsize(fs)
        else:
            for tl in ax[i].get_yticklabels():tl.set_visible(False)

        # the Whitaker+14 model
        # if it's an array, it's UV+IR SFRs. plot it separately.
        if type(plt_whit) == np.ndarray:
            nx, ny = prosp_dutils.running_median(mass[in_bin], 10**plt_whit[in_bin], bins=mbins,avg=True)
            ax[i].plot(nx,ny, **whitopts)
            ax[0].text(0.02,0.93,'UV+IR SFRs',color=whitopts['color'],transform=ax[0].transAxes)
        elif plt_whit:
            sfr_whit = 10**sfr_ms(zwhit[i],logm_whit)
            ax[i].plot(logm_whit, sfr_whit, **whitopts)
            ax[0].text(0.02,0.93,'Whitaker+14',color=whitopts['color'],transform=ax[0].transAxes)


        # zlabel
        ax[i].text(0.98, 0.05, zlabels[i],transform=ax[i].transAxes,zorder=1,ha='right')

        # show prior?
        if priors:
            ax[i].text(xlim[0]+0.3,ssfr_max+xlim[0]+1.5,'Prospector prior',rotation=30,fontsize=8)
            ax[i].plot([xlim[0],xlim[1]],[ssfr_max+xlim[0],ssfr_max+xlim[1]],**prior_opts)

        # save masses and star formation rates
        mass_save += x.tolist()
        sfr_save += [y.tolist()]

    # limits
    for a in ax.tolist()+[bigax]:
        a.set_xlim(xlim)
        a.set_ylim(ylim)
    
    # change bigax to logscale
    bigax.set_yscale('log', subsy=([1,4]))
    bigax.yaxis.set_minor_formatter(FormatStrFormatter('%2.4g'))
    bigax.yaxis.set_major_formatter(FormatStrFormatter('%2.4g'))
    for tl in bigax.get_yticklabels():tl.set_visible(False)
    bigax.tick_params('both', pad=3.5, labelsize=fs*1.3,size=3.5, width=1.0, which='both')

    # add legend
    bigax.legend(loc=4, prop={'size':fs*0.9},
                   scatterpoints=1,fancybox=True,ncol=2)

    # save data
    if outfile is not None:
        out = {'mass':x,'sfr':np.array(sfr_save).T,'z':zwhit}
        hickle.dump(out,open(outfile, "w"))

    plt.savefig(outname,dpi=150)
    plt.close()

def deltam_with_redshift(fast, prosp, z, outname, filename=None):

    # plot
    fig, ax = plt.subplots(1, 1, figsize = (3.5,3.5))
    lw = 2

    # define quantities
    par = 'stellar_mass'
    mfast = fast[par]
    mprosp, mprosp_up, mprosp_down = prosp[par]['q50'], prosp[par]['q84'], prosp[par]['q16']
    if type(mfast) == dict:
        mfast = mfast['q50']
    delta_m = mprosp-mfast

    # mass binning
    m_shift = 0.025
    nmassbins, xlim = 10, (8.8,11.5)
    massbins = np.linspace(xlim[0],xlim[1],nmassbins+1)

    # redshift binning
    zbins = np.linspace(0.5,2.5,5)
    nbins = len(zbins)-1
    xmed, ymed, zmed = [], [], []
    for i in range(nbins):
        inbin = (z > zbins[i]) & (z <= zbins[i+1])
        x,y = mfast[inbin], delta_m[inbin]

        # implement a floor for mass error: a handful of runs have borked
        # and have errors @ 1e-5 dex!
        mass_errs = np.clip((mprosp_up[inbin]-mprosp_down[inbin])/2., 0.05, np.inf)
        weights = mass_errs**(-2)
        weights = np.ones_like(mass_errs)

        # calculate weighted running median
        x, y, bincount = prosp_dutils.running_median(x,y,avg=False,return_bincount=True,weights=weights,bins=massbins)
        idx = bincount > nbin_min
        yerr = prosp_dutils.asym_errors(y[idx,0], y[idx,1], y[idx,2])

        ax.errorbar(x[idx]-m_shift*(i-1.5), y[idx,0], yerr=yerr, marker='o', linestyle='-', color=cmap[i],lw=lw,
                    label="{0:.1f}".format(zbins[i])+'<z<'+"{0:.1f}".format(zbins[i+1]), alpha=0.7,elinewidth=lw*0.5,capthick=lw*0.5)
        ymed += [y[:,0].tolist()]
        zmed += [(zbins[i]+zbins[i+1])/2.]

    ax.axhline(0, linestyle='--', color='0.4', lw=2,zorder=-1)
    ax.set_xlabel(r'log(M$_{\mathrm{FAST}}$/M$_{\odot}$)')
    ax.set_ylabel(r'log(M$_{\mathrm{Prospector}}$/M$_{\mathrm{FAST}}$)')
    ax.legend(prop={'size':8}, scatterpoints=1,fancybox=True,loc=4)
    ax.set_xlim(xlim)
    ax.set_ylim(-0.5,0.5)

    if filename is not None:
        out = {'fast_mass': (massbins[1:]+massbins[:-1])/2., 'log_mprosp_mfast': np.array(ymed).T, 'z': np.array(zmed)}
        hickle.dump(out,open(filename, "w"))

    plt.tight_layout()
    plt.savefig(outname,dpi=200)
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
        prosp, prosp_up, prosp_down = [np.clip(10**data['prosp']['ssfr_30'][q][good],minssfr,np.inf) for q in ['q50','q84','q16']]
        prosp_label = 'sSFR$_{\mathrm{Prosp}}$'
    else:
        prosp, prosp_up, prosp_down = [np.clip(data['prosp']['sfr_30'][q][good],minsfr,np.inf) for q in ['q50','q84','q16']]
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
    fig, ax = plt.subplots(1, 2, figsize = (7,3.5))
    ax = np.ravel(ax)
    lw = 2 


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
    ssfr_shift = 0.04
    zbins = np.linspace(0.5,2.5,5)
    nbins = len(zbins)-1

    xmed, ymed, zmed = [], [], []
    for i in range(nbins):
        inbin = (data['fast']['z'][good] > zbins[i]) & (data['fast']['z'][good] <= zbins[i+1])
        if inbin.sum() == 0:
            continue
        
        if ssfr == True:
            bins = np.linspace(-12,-8,8)
        else:
            bins = None
        
        # minimum error of 0.1 dex in sSFRs, maximum error 1 dex
        #ssfr_err = np.clip((np.log10(prosp_up)-np.log10(prosp_down))[inbin]/2., 0.1, 0.3)
        #weights = ssfr_err**(-2.)
        weights = np.ones(inbin.sum(),dtype=float)

        x,y = np.log10(uvir[inbin]), ratio[inbin]
        x, y, bincount = prosp_dutils.running_median(x,y,avg=False,return_bincount=True, bins=bins,weights=weights)
        x, y = x[bincount > nbin_min], y[bincount > nbin_min]
        yerr = prosp_dutils.asym_errors(y[:,0], y[:,1], y[:,2])
        ax[1].errorbar(x+(ssfr_shift*(i-1.5)), y[:,0], yerr=yerr, marker='o', color=cmap[i],lw=lw,alpha=0.6,
                       label="{0:.1f}".format(zbins[i])+'<z<'+"{0:.1f}".format(zbins[i+1]),elinewidth=lw*0.5,capthick=lw*0.5)
        xmed += x.tolist()
        ymed += y[:,0].tolist()
        zmed += [(zbins[i]+zbins[i+1])/2.]*len(y)

    ax[1].axhline(0, linestyle='--', color='0.5', lw=lw,zorder=-1)
    ax[1].set_xlabel('log('+prosp_label+')')
    ax[1].set_ylabel(r'log(SFR$_{\mathrm{Prospector}}$/SFR$_{\mathrm{UV+IR,mod}}$)')
    ax[1].legend(prop={'size':8}, scatterpoints=1,fancybox=True)
    ax[1].set_ylim(-2,2)

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
                   'ydata': np.log10(np.clip(data['prosp']['sfr_30']['q50'],0.001,np.inf)),
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



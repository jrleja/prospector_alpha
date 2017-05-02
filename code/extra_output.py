from prospect.models import model_setup
from prospect.io import read_results
import os, prosp_dutils, hickle, sys, time
import numpy as np
from copy import copy
from astropy import constants

try:
    import prosp_diagnostic_plots
except IOError:
    pass

def maxprob_model(sample_results,sps):

    ### grab maximum probability, plus the thetas that gave it
    maxprob = sample_results['flatprob'].max()
    maxtheta = sample_results['flatchain'][sample_results['flatprob'].argmax()]

    ### ensure that maxprob stored is the same as calculated now 
    current_maxprob = prosp_dutils.test_likelihood(sps,
                                                    sample_results['model'],
                                                    sample_results['obs'],
                                                    maxtheta,
                                                    sample_results['run_params']['param_file'])
    
    print 'Best-fit lnprob currently: {0}'.format(current_maxprob)
    print 'Best-fit lnprob during sampling: {0}'.format(maxprob)

    return maxtheta, maxprob

def measure_spire_phot(sample_results, flatchain, sps):

    '''
    for plots for kim-vy tran on 10/26/16
    '''

    from sedpy.observate import load_filters
    filters = ['herschel_spire_250','herschel_spire_350','herschel_spire_500']
    obs = {'filters': load_filters(filters), 'wavelength': None}

    ncalc = flatchain.shape[0]
    mags = np.zeros(shape=(len(filters),ncalc))
    for i in xrange(ncalc):
        _,mags[:,i],sm = sample_results['model'].mean_model(flatchain[i,:], obs, sps=sps)
        print i

    sample_results['hphot'] = {}
    sample_results['hphot']['mags'] = mags
    sample_results['hphot']['name'] = np.array(filters)

    return sample_results

def sample_flatchain(chain, lnprob, parnames, ir_priors=True, include_maxlnprob=True, nsamp=2000):

    '''
    CURRENTLY UNDER DEVELOPMENT
    goal: sample the flatchain in a smart way
    '''

    ##### use randomized, flattened chain for posterior draws
    # don't allow draws which are outside the priors
    good = np.isfinite(lnprob) == True

    ### cut in IR priors
    if ir_priors:
        gamma_idx = parnames == 'duste_gamma'
        umin_idx = parnames == 'duste_umin'
        qpah_idx = parnames == 'duste_qpah'
        gamma_prior = 0.15
        umin_prior = 15
        qpah_prior = 7
        in_priors = (chain[:,gamma_idx] < gamma_prior) & (chain[:,umin_idx] < umin_prior) & (chain[:,qpah_idx] < qpah_prior) & good[:,None]
        if in_priors.sum() < 2*nsamp:
            print 'Insufficient number of samples within the IR priors! Not applying IR priors.'
        else:
            good = in_priors

    sample_idx = np.random.choice(np.where(good)[0],nsamp)

    ### include maxlnprob?
    if include_maxlnprob:
        sample_idx[0] = lnprob.argmax()

    return sample_idx

def set_sfh_time_vector(sample_results,ncalc):

    # if parameterized, calculate linearly in 100 steps from t=0 to t=tage
    # if nonparameterized, calculate at bin edges.
    if 'tage' in sample_results['model'].theta_labels():
        nt = 100
        idx = np.array(sample_results['model'].theta_labels()) == 'tage'
        maxtime = np.max(sample_results['flatchain'][:ncalc,idx])
        t = np.linspace(0,maxtime,num=nt)
    elif 'agebins' in sample_results['model'].params:
        in_years = 10**sample_results['model'].params['agebins']/1e9
        t = np.concatenate((np.ravel(in_years)*0.9999, np.ravel(in_years)*1.001))
        t.sort()
        t = t[1:-1] # remove older than oldest bin, younger than youngest bin
        t = np.clip(t,1e-3,np.inf) # nothing younger than 1 Myr!
    else:
        sys.exit('ERROR: not sure how to set up the time array here!')
    return t

def calc_extra_quantities(sample_results, ncalc=3000, **kwargs):

    '''' 
    CALCULATED QUANTITIES
    model nebular emission line strength
    model star formation history parameters (ssfr,sfr,half-mass time)
    '''

    # different options for what to calculate
    # speedup is measured in runtime, where runtime = ncalc * model_call
    opts = {
            'restframe_optical_photometry': False, # currently deprecated! but framework exists in restframe_optical_properties
            'ir_priors': True, # no cost
            'measure_spectral_features': True, # cost = 2 runtimes
            'mags_nodust': False # cost = 1 runtime
            }
    if kwargs:
        for key in kwargs.keys(): opts[key] = kwargs[key]

    parnames = np.array(sample_results['model'].theta_labels())

    ##### describe number of components in Prospector model [legacy]
    sample_results['ncomp'] = np.sum(['mass' in x for x in sample_results['model'].theta_labels()])

    ##### array indexes over which to sample the flatchain
    sample_idx = sample_flatchain(sample_results['flatchain'], sample_results['flatprob'], 
                                  parnames, ir_priors=opts['ir_priors'], include_maxlnprob=True, nsamp=ncalc)

    ##### initialize output arrays for SFH + emission line posterior draws
    half_time,sfr_10,sfr_100,ssfr_100,stellar_mass,emp_ha,lir,luv,lmir,lbol, \
    bdec_calc,ext_5500,dn4000,ssfr_10,xray_lum = [np.zeros(shape=(ncalc)) for i in range(15)]
    if 'fagn' in parnames:
        l_agn, fmir = [np.zeros(shape=(ncalc)) for i in range(2)]

    ##### information for empirical emission line calculation ######
    d1_idx = parnames == 'dust1'
    d2_idx = parnames == 'dust2'
    didx = parnames == 'dust_index'

    ##### set up time vector for full SFHs
    t = set_sfh_time_vector(sample_results,ncalc)
    intsfr = np.zeros(shape=(t.shape[0],ncalc))

    ##### initialize sps, calculate maxprob
    # also confirm probability calculations are consistent with fit
    sps = model_setup.load_sps(**sample_results['run_params'])
    maxthetas, maxprob = maxprob_model(sample_results,sps)

    ##### set up model flux vectors
    mags = np.zeros(shape=(len(sample_results['obs']['filters']),ncalc))
    try:
        wavelengths = sps.wavelengths
    except AttributeError:
        wavelengths = sps.csp.wavelengths
    spec = np.zeros(shape=(wavelengths.shape[0],ncalc))

    ##### modify nebular status to ensure emission line production
    # don't cache, and turn on
    if sample_results['model'].params['add_neb_emission'] == 2:
        sample_results['model'].params['add_neb_emission'] = np.array([True])
    sample_results['model'].params['nebemlineinspec'] = np.array([True])

    ######## posterior sampling #########
    for jj,idx in enumerate(sample_idx):
        t1 = time.time()

        ##### model call, to set parameters
        thetas = copy(sample_results['flatchain'][idx])
        spec[:,jj],mags[:,jj],sm = sample_results['model'].mean_model(thetas, sample_results['obs'], sps=sps)

        ##### if we don't use these parameters, set them to defaults
        dust1 = thetas[d1_idx]
        if dust1.shape[0] == 0:
            dust1 = sps.csp.params['dust1']
        dust_idx = thetas[didx]
        if dust_idx.shape[0] == 0:
            dust_idx = sps.csp.params['dust_index']

        ##### extract sfh parameters
        # pass stellar mass to avoid extra model call
        sfh_params = prosp_dutils.find_sfh_params(sample_results['model'],thetas,
                                                   sample_results['obs'],sps,sm=sm)

        ##### calculate SFH
        intsfr[:,jj] = prosp_dutils.return_full_sfh(t, sfh_params)

        ##### solve for half-mass assembly time
        # this is half-time in the sense of integral of SFR, i.e.
        # mass loss is NOT taken into account.
        half_time[jj] = prosp_dutils.halfmass_assembly_time(sfh_params)

        ##### calculate time-averaged SFR
        sfr_10[jj]   = prosp_dutils.calculate_sfr(sfh_params, 0.01, minsfr=-np.inf, maxsfr=np.inf)
        sfr_100[jj]  = prosp_dutils.calculate_sfr(sfh_params, 0.1,  minsfr=-np.inf, maxsfr=np.inf)

        ##### calculate mass, sSFR
        stellar_mass[jj] = sfh_params['mass']
        ssfr_10[jj] = sfr_10[jj] / stellar_mass[jj]
        ssfr_100[jj] = sfr_100[jj] / stellar_mass[jj]

        ##### calculate L_AGN if necessary
        if 'fagn' in parnames:
            l_agn[jj] = prosp_dutils.measure_agn_luminosity(thetas[parnames=='fagn'],sps,sfh_params['mformed'])
        xray_lum[jj] = prosp_dutils.estimate_xray_lum(sfr_100[jj])

        ##### empirical halpha HERE
        emp_ha[jj] = prosp_dutils.synthetic_halpha(sfr_10[jj],dust1,
                                                    thetas[d2_idx],-1.0,
                                                    dust_idx,
                                                    kriek = (sample_results['model'].params['dust_type'] == 4)[0])

        ##### dust extinction at 5500 angstroms
        ext_5500[jj] = dust1 + thetas[d2_idx]

        ##### empirical Balmer decrement
        bdec_calc[jj] = prosp_dutils.calc_balmer_dec(dust1, thetas[d2_idx], -1.0, 
                                                      dust_idx,
                                                      kriek = (sample_results['model'].params['dust_type'] == 4)[0])

        ##### lbol
        lbol[jj] = prosp_dutils.measure_lbol(sps,sfh_params['mformed'])

        ##### spectral quantities (emission line flux, Balmer decrement, Hdelta absorption, Dn4000)
        ##### and magnitudes (LIR, LUV)
        if opts['measure_spectral_features']:
            modelout = prosp_dutils.measure_restframe_properties(sps, thetas = thetas,
                                                        model=sample_results['model'], obs = sample_results['obs'],
                                                        measure_ir=True, measure_luv=True, measure_mir=True, 
                                                        emlines=True, abslines=True, restframe_optical_photometry = False)
            #### initialize arrays
            if jj == 0:
                emnames = np.array(modelout['emlines'].keys())
                nline = len(emnames)
                emflux, emeqw = [np.empty(shape=(ncalc,nline)) for i in xrange(2)]

                absnames = np.array(modelout['abslines'].keys())
                nabs = len(absnames)
                absflux, abseqw = [np.empty(shape=(ncalc,nabs)) for i in xrange(2)]

            absflux[jj,:] = np.array([modelout['abslines'][line]['flux'] for line in absnames])
            abseqw[jj,:] = np.array([modelout['abslines'][line]['eqw'] for line in absnames])
            emflux[jj,:] = np.array([modelout['emlines'][line]['flux'] for line in emnames])
            emeqw[jj,:] = np.array([modelout['emlines'][line]['eqw'] for line in emnames])

            lir[jj]        = modelout['lir']
            luv[jj]        = modelout['luv']
            lmir[jj]       = modelout['lmir']
            dn4000[jj]     = modelout['dn4000']

            if 'fagn' in parnames:
                nagn_thetas = copy(thetas)
                nagn_thetas[parnames == 'fagn'] = 0.0
                modelout = prosp_dutils.measure_restframe_properties(sps, thetas=nagn_thetas,
                                            model=sample_results['model'], obs=sample_results['obs'],
                                            measure_mir=True)
                fmir[jj] = (lmir[jj]-modelout['lmir'])/lmir[jj]

        #### no dust
        if opts['mags_nodust']:
            if jj == 0:
                mags_nodust = np.zeros(shape=(len(sample_results['obs']['filters']),ncalc))

            nd_thetas = copy(thetas)
            nd_thetas[d1_idx] = np.array([0.0])
            nd_thetas[d2_idx] = np.array([0.0])
            _,mags_nodust[:,jj],sm = sample_results['model'].mean_model(nd_thetas, sample_results['obs'], sps=sps)

        print('loop {0} took {1}s'.format(jj,time.time() - t1))

    ##### CALCULATE Q16,Q50,Q84 FOR MODEL PARAMETERS
    ntheta = len(sample_results['initial_theta'])
    q_16, q_50, q_84 = (np.zeros(ntheta)+np.nan for i in range(3))
    for kk in xrange(ntheta): q_16[kk], q_50[kk], q_84[kk] = np.percentile(sample_results['flatchain'][sample_idx][:,kk], [16.0, 50.0, 84.0])
    
    #### QUANTILE OUTPUTS #
    quantiles = {'sample_chain': sample_results['flatchain'][sample_idx],
                 'parnames': parnames,
                 'q16':q_16,
                 'q50':q_50,
                 'q84':q_84}
    extra_output['quantiles'] = quantiles

    ##### CALCULATE Q16,Q50,Q84 FOR EXTRA PARAMETERS
    extra_output = {}
    extra_flatchain = np.dstack((half_time, sfr_10, sfr_100, ssfr_10, ssfr_100, stellar_mass, emp_ha, bdec_calc, ext_5500, xray_lum,lbol))[0]
    if 'fagn' in parnames:
        extra_flatchain = np.append(extra_flatchain, np.hstack((l_agn[:,None], fmir[:,None])), axis=1)
    nextra = extra_flatchain.shape[1]
    q_16e, q_50e, q_84e = (np.zeros(nextra)+np.nan for i in range(3))
    for kk in xrange(nextra): q_16e[kk], q_50e[kk], q_84e[kk] = np.percentile(extra_flatchain[:,kk], [16.0, 50.0, 84.0])

    #### EXTRA PARAMETER OUTPUTS 
    extras = {'flatchain': extra_flatchain,
              'parnames': np.array(['half_time','sfr_10','sfr_100','ssfr_10','ssfr_100','stellar_mass','emp_ha','bdec_calc','total_ext5500', 'xray_lum','lbol']),
              'q16': q_16e,
              'q50': q_50e,
              'q84': q_84e,
              'sfh': intsfr,
              't_sfh': t}
    if 'fagn' in parnames:
        extras['parnames'] = np.append(extras['parnames'],np.array(['l_agn','fmir']))
    extra_output['extras'] = extras

    #### OBSERVABLES
    observables = {'spec': spec,
                   'mags': mags,
                   'lam_obs': wavelengths}
    extra_output['observables'] = observables

    #### BEST-FITS
    bfit      = {'maxprob_params':maxthetas,
                 'maxprob':maxprob,
                 'emp_ha': emp_ha[0],
                 'sfh': intsfr[:,0],
                 'half_time': half_time[0],
                 'sfr_10': sfr_10[0],
                 'sfr_100':sfr_100[0],
                 'bdec_calc':bdec_calc[0],
                 'lbol': lbol[0],
                 'spec':spec[:,0],
                 'mags':mags[:,0]}
    extra_output['bfit'] = bfit

    ##### filters with no dust
    if opts['mags_nodust']:
        extra_output['bfit']['mags_nodust'] = mags_nodust[:,0]
        extra_output['observables']['mags_nodust'] = mags_nodust

    ##### spectral features
    if opts['measure_spectral_features']:

        ##### FORMAT EMLINE OUTPUT 
        q_16flux, q_50flux, q_84flux, q_16eqw, q_50eqw, q_84eqw = (np.zeros(nline)+np.nan for i in range(6))
        for kk in xrange(nline): q_16flux[kk], q_50flux[kk], q_84flux[kk] = np.percentile(emflux[:,kk], [16.0, 50.0, 84.0])
        for kk in xrange(nline): q_16eqw[kk], q_50eqw[kk], q_84eqw[kk] = np.percentile(emeqw[:,kk], [16.0, 50.0, 84.0])
        emline_info = {}
        emline_info['eqw'] = {'chain':emeqw,
                            'q16':q_16eqw,
                            'q50':q_50eqw,
                            'q84':q_84eqw}
        emline_info['flux'] = {'chain':emflux,
                            'q16':q_16flux,
                            'q50':q_50flux,
                            'q84':q_84flux}
        emline_info['emnames'] = emnames
        extra_output['model_emline'] = emline_info

        ##### SPECTRAL QUANTITIES
        q_16flux, q_50flux, q_84flux, q_16eqw, q_50eqw, q_84eqw = (np.zeros(nabs) for i in range(6))
        for kk in xrange(nabs): q_16flux[kk], q_50flux[kk], q_84flux[kk] = np.percentile(absflux[:,kk], [16.0, 50.0, 84.0])
        for kk in xrange(nabs): q_16eqw[kk], q_50eqw[kk], q_84eqw[kk] = np.percentile(abseqw[:,kk], [16.0, 50.0, 84.0])
        q_16dn, q_50dn, q_84dn = np.percentile(dn4000, [16.0, 50.0, 84.0])
        
        spec_info = {}
        spec_info['dn4000'] = {'chain':dn4000,
                               'q16':q_16dn,
                               'q50':q_50dn,
                               'q84':q_84dn}
        spec_info['eqw'] = {'chain':abseqw,
                            'q16':q_16eqw,
                            'q50':q_50eqw,
                            'q84':q_84eqw}
        spec_info['flux'] = {'chain':absflux,
                            'q16':q_16flux,
                            'q50':q_50flux,
                            'q84':q_84flux}
        spec_info['absnames'] = absnames
        extra_output['spec_info'] = spec_info

        ### LUV + LIR
        extra_output['observables']['L_IR'] = lir
        extra_output['observables']['L_UV'] = luv
        extra_output['observables']['L_MIR'] = lmir

        ### bfits
        extra_output['bfit']['lir'] = lir[0]
        extra_output['bfit']['luv'] = luv[0]
        extra_output['bfit']['lmir'] = lmir[0]
        extra_output['bfit']['halpha_flux'] = emflux[0,emnames == 'Halpha']
        extra_output['bfit']['hbeta_flux'] = emflux[0,emnames == 'Hbeta']   
        extra_output['bfit']['hdelta_flux'] = emflux[0,emnames == 'Hdelta']
        extra_output['bfit']['halpha_abs'] = absflux[0,absnames == 'halpha_wide']
        extra_output['bfit']['hbeta_abs'] = absflux[0,absnames == 'hbeta']
        extra_output['bfit']['hdelta_abs'] = absflux[0,absnames == 'hdelta_wide']
        extra_output['bfit']['dn4000'] = dn4000[0]
    
    return extra_output

def update_all(runname, **kwargs):
    '''
    change some parameters, need to update the post-processing?
    run this!
    '''
    filebase, parm_basename, ancilname=prosp_dutils.generate_basenames(runname)
    for param in parm_basename:
        post_processing(param, **kwargs)

def post_processing(param_name, **kwargs):

    '''
    Driver. Loads output, runs post-processing routine.
    '''

    from brown_io import load_prospector_data, create_prosp_filename

    # I/O
    parmfile = model_setup.import_module_from_file(param_name)
    outname = parmfile.run_params['outfile']
    outfolder = os.getenv('APPS')+'/threedhst_bsfh/plots/'+outname.split('/')[-2]+'/'

    # check for output folder, create if necessary
    if not os.path.isdir(outfolder):
        os.makedirs(outfolder)

    try:
        sample_results, powell_results, model, _ = load_prospector_data(outname,hdf5=True,load_extra_output=False)
    except AttributeError:
        print 'Failed to load chain for '+sample_results['run_params']['objname']+'. Returning.'
        return

    print 'Performing post-processing on ' + sample_results['run_params']['objname']

    ### create flatchain, run post-processing
    sample_results['flatchain'] = prosp_dutils.chop_chain(sample_results['chain'],**sample_results['run_params'])
    sample_results['flatprob'] = prosp_dutils.chop_chain(sample_results['lnprobability'],**sample_results['run_params'])
    extra_output = calc_extra_quantities(sample_results,**kwargs)
    
    ### create post-processing name, dump info
    mcmc_filename, model_filename, extra_filename = create_prosp_filename(outname)
    hickle.dump(extra_output,open(extra_filename, "w"))

    ### MAKE PLOTS HERE
    try:
        prosp_diagnostic_plots.make_all_plots(sample_results=sample_results,extra_output=extra_output,
                                      filebase=outname,outfolder=outfolder,param_name=param_name+'.py')
    except NameError:
        print "Unable to make plots for "+sample_results['run_params']['objname']+" due to import error. Passing."
        pass

if __name__ == "__main__":
    
    # note that this only processes booleans!
    kwargs = {}
    for arg in sys.argv:
        if arg[:2] == '--':
            split = arg[2:].split('=')
            kwargs[split[0]] = split[1].lower() in ['true', 't', 'yes']

    post_processing(sys.argv[1],**kwargs)


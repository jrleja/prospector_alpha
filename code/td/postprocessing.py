from prospect.models import model_setup
import os, prosp_dutils, hickle, sys, time
import numpy as np
import argparse
from copy import deepcopy
from prospector_io import load_prospector_data, create_prosp_filename
import prosp_dynesty_plots
from dynesty.plotting import _quantile as weighted_quantile
from prospect.models import sedmodel

def set_sfh_time_vector(theta,model):
    """if parameterized, calculate linearly in 100 steps from t=0 to t=tage
    if nonparameterized, calculate at bin edges.
    """

    model.set_parameters(theta)
    if 'tage' in model.theta_labels():
        nt = 100
        tage = theta[model.theta_index['tage']]
        t = np.linspace(0,tage,num=nt)
    elif 'agebins' in model.params:
        in_years = 10**model.params['agebins']/1e9
        t = np.concatenate((np.ravel(in_years)*0.9999, np.ravel(in_years)*1.001))
        t.sort()
        t = t[1:-1] # remove older than oldest bin, younger than youngest bin
        t = np.clip(t,1e-3,np.inf) # nothing younger than 1 Myr!
        t = np.unique(t)
    else:
        sys.exit('ERROR: not sure how to set up the time array here!')
    return t

def calc_extra_quantities(res, sps, obs, ncalc=3000, shorten_spec=True, measure_abslines=False,
                          measure_herschel=False,**kwargs):
    """calculate extra quantities: star formation history, stellar mass, spectra, photometry, etc
    shorten_spec: if on, return only the 50th / 84th / 16th percentiles. else return all spectra.
    """

    # calculate maxprob
    # and ensure that maxprob stored is the same as calculated now 
    # don't recalculate lnprobability after we fix MassMet
    res['lnprobability'] = res['lnlikelihood'] + res['model'].prior_product(res['chain'])
    amax = res['lnprobability'].argmax()
    current_maxprob = prosp_dutils.test_likelihood(sps, res['model'], res['obs'], 
                                                   res['chain'][amax], 
                                                   res['run_params']['param_file'])
    print 'Best-fit lnprob currently: {0}'.format(float(current_maxprob))
    print 'Best-fit lnprob during sampling: {0}'.format(res['lnprobability'][amax])

    # randomly choose from the chain, weighted by dynesty weights
    # make sure the first one is the maximum probability model (so we're cheating a bit!)
    # don't do replacement, we can use weights to rebuild PDFs
    nsample = res['chain'].shape[0]
    sample_idx = np.random.choice(np.arange(nsample), size=ncalc, p=res['weights'], replace=False)
    if amax in sample_idx:
        sample_idx[sample_idx == amax] = sample_idx[0]
    sample_idx[0] = amax
    print "we are measuring {0}% of the weights".format(res['weights'][sample_idx].sum()/res['weights'].sum()*100)

    # compact creation of outputs
    eout = {
            'thetas':{},
            'extras':{},
            'sfh':{},
            'obs':{},
            'sample_idx':sample_idx,
            'weights':res['weights'][sample_idx],
            'zred': float(res['model'].params['zred'])
            }
    fmt = {'chain':np.zeros(shape=ncalc),'q50':0.0,'q84':0.0,'q16':0.0}

    # thetas
    parnames = res['model'].theta_labels()
    for i, p in enumerate(parnames):  
        q50, q16, q84 = weighted_quantile(res['chain'][:,i], np.array([0.5, 0.16, 0.84]), weights=res['weights'])
        eout['thetas'][p] = {'q50': q50, 'q16': q16, 'q84': q84}

    # extras
    extra_parnames = ['avg_age','lwa_rband','lwa_lbol','half_time','sfr_100','ssfr_100','ssfr_50','sfr_50',\
                      'stellar_mass','lir','luv','lmir','lbol','luv_young','lir_young']
    if 'fagn' in parnames:
        extra_parnames += ['l_agn', 'fmir', 'luv_agn', 'lir_agn']
    for p in extra_parnames: eout['extras'][p] = deepcopy(fmt)

    # sfh
    tvec = set_sfh_time_vector(res['model'].initial_theta,res['model'])
    eout['sfh']['t'] = np.zeros(shape=(ncalc,tvec.shape[0]))
    eout['sfh']['sfh'] = np.zeros(shape=(ncalc,tvec.shape[0]))

    # observables
    eout['obs']['spec'] = np.zeros(shape=(ncalc,sps.wavelengths.shape[0]))
    eout['obs']['mags'] = np.zeros(shape=(ncalc,len(res['obs']['filters'])))
    eout['obs']['uvj'] = np.zeros(shape=(ncalc,3))
    eout['obs']['lam_obs'] = sps.wavelengths
    elines = ['H beta 4861', 'H alpha 6563','Br gamma 21657','Pa alpha 18752']
    eout['obs']['elines'] = {key: {'ew': deepcopy(fmt), 'flux': deepcopy(fmt)} for key in elines}
    eout['obs']['dn4000'] = deepcopy(fmt)
    res['model'].params['nebemlineinspec'] = True
    if measure_abslines:
        abslines = ['halpha_wide', 'halpha_narrow', 'hbeta', 'hdelta_wide', 'hdelta_narrow']
        eout['obs']['abslines'] = {key+'_ew': deepcopy(fmt) for key in abslines}

    if measure_herschel:
        eout['obs']['herschel'] = {'mags':np.zeros(shape=(ncalc,5))}
        from sedpy.observate import load_filters
        filters = ['herschel_pacs_100','herschel_pacs_160','herschel_spire_250','herschel_spire_350','herschel_spire_500']
        fobs = {'filters': load_filters(filters), 'wavelength': None}

    # generate model w/o dependencies for young star contribution
    model_params = deepcopy(res['model'].config_list)
    for j in range(len(model_params)):
        if model_params[j]['name'] == 'mass':
            print model_params[j]['name']
            model_params[j].pop('depends_on', None)
    nodep_model = sedmodel.SedModel(model_params)

    # sample in the posterior
    for jj,sidx in enumerate(sample_idx):

        # bookkeepping
        t1 = time.time()

        # model call
        thetas = res['chain'][sidx,:]
        eout['obs']['spec'][jj,:],eout['obs']['mags'][jj,:],sm = res['model'].mean_model(thetas, res['obs'], sps=sps)

        # calculate SFH-based quantities
        sfh_params = prosp_dutils.find_sfh_params(res['model'],thetas,res['obs'],sps,sm=sm)
        eout['extras']['stellar_mass']['chain'][jj] = sfh_params['mass']
        eout['sfh']['t'][jj,:] = set_sfh_time_vector(thetas,res['model'])
        eout['sfh']['sfh'][jj,:] = prosp_dutils.return_full_sfh(eout['sfh']['t'][jj,:], sfh_params)
        eout['extras']['half_time']['chain'][jj] = prosp_dutils.halfmass_assembly_time(sfh_params)
        eout['extras']['sfr_100']['chain'][jj] = prosp_dutils.calculate_sfr(sfh_params, 0.1,  minsfr=-np.inf, maxsfr=np.inf)
        eout['extras']['ssfr_100']['chain'][jj] = eout['extras']['sfr_100']['chain'][jj].squeeze() / eout['extras']['stellar_mass']['chain'][jj].squeeze()
        eout['extras']['sfr_50']['chain'][jj] = prosp_dutils.calculate_sfr(sfh_params, 0.05,  minsfr=-np.inf, maxsfr=np.inf)
        eout['extras']['ssfr_50']['chain'][jj] = eout['extras']['sfr_50']['chain'][jj].squeeze() / eout['extras']['stellar_mass']['chain'][jj].squeeze()

        # calculate AGN parameters if necessary
        if 'fagn' in parnames:
            eout['extras']['l_agn']['chain'][jj] = prosp_dutils.measure_agn_luminosity(thetas[parnames.index('fagn')],sps,sfh_params['mformed'])

        # lbol
        eout['extras']['lbol']['chain'][jj] = prosp_dutils.measure_lbol(sps,sfh_params['mformed'])

        # measure from rest-frame spectrum
        t2 = time.time()
        props = prosp_dutils.measure_restframe_properties(sps, thetas = thetas, model=res['model'], measure_uvj=True, abslines=measure_abslines, 
                                                          measure_ir=True, measure_luv=True, measure_mir=True, emlines=elines)
        eout['extras']['lir']['chain'][jj] = props['lir']
        eout['extras']['luv']['chain'][jj] = props['luv']
        eout['extras']['lmir']['chain'][jj] = props['lmir']
        eout['obs']['dn4000']['chain'][jj] = props['dn4000']
        eout['obs']['uvj'][jj,:] = props['uvj']

        for e in elines: 
            eout['obs']['elines'][e]['flux']['chain'][jj] = props['emlines'][e]['flux']
            eout['obs']['elines'][e]['ew']['chain'][jj] = props['emlines'][e]['eqw']

        if measure_abslines:
            for a in abslines: eout['obs']['abslines'][a+'_ew']['chain'][jj] = props['abslines'][a]['eqw']

        nagn_thetas = deepcopy(thetas)
        if 'fagn' in parnames:
            nagn_thetas[parnames.index('fagn')] = 0.0
            props = prosp_dutils.measure_restframe_properties(sps, thetas=nagn_thetas, model=res['model'], 
                                                              measure_mir=True,measure_ir = True, measure_luv = True)
            eout['extras']['fmir']['chain'][jj] = (eout['extras']['lmir']['chain'][jj]-props['lmir'])/eout['extras']['lmir']['chain'][jj]
            eout['extras']['luv_agn']['chain'][jj] = props['luv']
            eout['extras']['lir_agn']['chain'][jj] = props['lir']

        # isolate young star contribution
        nodep_model.params['mass'] = np.zeros_like(res['model'].params['mass'])
        nodep_model.params['mass'][0] = res['model'].params['mass'][0]
        out = prosp_dutils.measure_restframe_properties(sps, model = nodep_model, thetas = nagn_thetas, measure_ir = True, measure_luv = True)
        eout['extras']['luv_young']['chain'][jj] = out['luv']
        eout['extras']['lir_young']['chain'][jj] = out['lir']

        # ages
        eout['extras']['avg_age']['chain'][jj], eout['extras']['lwa_lbol']['chain'][jj], \
        eout['extras']['lwa_rband']['chain'][jj] = prosp_dutils.all_ages(thetas,res['model'],sps)

        if measure_herschel:
            _,eout['obs']['herschel']['mags'][jj,:],__ = res['model'].mean_model(thetas, fobs, sps=sps)

        t3 = time.time()
        print('loop {0} took {1}s ({2}s for absorption+emission)'.format(jj,t3 - t1,t3 - t2))

    # calculate percentiles from chain
    for p in eout['extras'].keys():
        q50, q16, q84 = weighted_quantile(eout['extras'][p]['chain'], np.array([0.5, 0.16, 0.84]), weights=eout['weights'])
        for q,qstr in zip([q50,q16,q84],['q50','q16','q84']): eout['extras'][p][qstr] = q
    
    q50, q16, q84 = weighted_quantile(eout['obs']['dn4000']['chain'], np.array([0.5, 0.16, 0.84]), weights=eout['weights'])
    for q,qstr in zip([q50,q16,q84],['q50','q16','q84']): eout['obs']['dn4000'][qstr] = q

    for key1 in eout['obs']['elines'].keys():
        for key2 in ['ew','flux']:
            q50, q16, q84 = weighted_quantile(eout['obs']['elines'][key1][key2]['chain'], np.array([0.5, 0.16, 0.84]), weights=eout['weights'])
            for q,qstr in zip([q50,q16,q84],['q50','q16','q84']): eout['obs']['elines'][key1][key2][qstr] = q

    if measure_abslines:
        for key in eout['obs']['abslines'].keys():
            q50, q16, q84 = weighted_quantile(eout['obs']['abslines'][key]['chain'], np.array([0.5, 0.16, 0.84]), weights=eout['weights'])
            for q,qstr in zip([q50,q16,q84],['q50','q16','q84']): eout['obs']['abslines'][key][qstr] = q

    if shorten_spec:
        spec_pdf = np.zeros(shape=(len(sps.wavelengths),3))
        for jj in xrange(spec_pdf.shape[0]): spec_pdf[jj,:] = weighted_quantile(eout['obs']['spec'][:,jj], np.array([0.5, 0.16, 0.84]), weights=eout['weights'])
        eout['obs']['spec'] = {'q50':spec_pdf[:,0],'q16':spec_pdf[:,1],'q84':spec_pdf[:,2]}

    return eout

def post_processing(param_name, objname=None, runname = None, overwrite=True, obj_outfile=None,
                    plot_outfolder=None, plot=True, **kwargs):
    """Driver. Loads output, runs post-processing routine.
    overwrite=False will return immediately if post-processing file already exists.
    if runname is specified, we can pass in parameter file for run A with outputs at location runname
    kwargs are passed to calc_extra_quantities
    """

    # bookkeeping: where are we coming from and where are we going?
    pfile = model_setup.import_module_from_file(param_name)
    run_outfile = pfile.run_params['outfile']

    if obj_outfile is None:
        if runname is None:
            runname = run_outfile.split('/')[-2]
            obj_outfile = "/".join(run_outfile.split('/')[:-1]) + '/' + objname
        else:
            obj_outfile = "/".join(run_outfile.split('/')[:-2]) + '/' + runname + '/' + objname

        # account for unique td_huge storage situation
        if (runname == 'td_huge') | (runname == 'td_new'):
            field = obj_outfile.split('/')[-1].split('_')[0]
            obj_outfile = "/".join(obj_outfile.split('/')[:-1])+'/'+field+'/'+obj_outfile.split('/')[-1]  

    if plot_outfolder is None:
        plot_outfolder = os.getenv('APPS')+'/prospector_alpha/plots/'+runname+'/'

    # check for output folder, create if necessary
    if not os.path.isdir(plot_outfolder):
        os.makedirs(plot_outfolder)

    # I/O
    res, powell_results, model, eout = load_prospector_data(obj_outfile,hdf5=True,load_extra_output=True)
    if res is None:
        print 'there are no sampling results! returning.'
        return
    if (not overwrite) & (eout is not None):
        print 'post-processing file already exists! returning.'
        return

    if res['model'] is None:
        res['model'] = pfile.load_model(**res['run_params'])

    # make filenames local...
    print 'Performing post-processing on ' + objname
    for key in res['run_params']:
        if type(res['run_params'][key]) == unicode:
            if 'prospector_alpha' in res['run_params'][key]:
                res['run_params'][key] = os.getenv('APPS')+'/prospector_alpha'+res['run_params'][key].split('prospector_alpha')[-1]
    sps = pfile.load_sps(**res['run_params'])
    obs = res['obs']

    # sample from chain
    extra_output = calc_extra_quantities(res,sps,obs,**kwargs)
    
    # create post-processing name, dump info
    _, _, extra_filename = create_prosp_filename(obj_outfile)
    hickle.dump(extra_output,open(extra_filename, "w"))

    # make standard plots
    if plot:
        prosp_dynesty_plots.make_all_plots(filebase=obj_outfile,outfolder=plot_outfolder)


def do_all(param_name=None,runname=None,**kwargs):
    ids = np.genfromtxt('/Users/joel/code/python/prospector_alpha/data/3dhst/'+runname+'.ids',dtype=str)
    for id in ids:
        post_processing(param_name, objname=id, **kwargs)
        
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

if __name__ == "__main__":

    ### don't create keyword if not passed in!
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('parfile', type=str)
    parser.add_argument('--objname')
    parser.add_argument('--ncalc',type=int)
    parser.add_argument('--overwrite',type=str2bool)
    parser.add_argument('--shorten_spec',type=str2bool)
    parser.add_argument('--runname', type=str)
    parser.add_argument('--plot',type=str2bool)
    parser.add_argument('--measure_herschel',type=str2bool)

    args = vars(parser.parse_args())
    kwargs = {}
    for key in args.keys(): kwargs[key] = args[key]

    print kwargs
    post_processing(kwargs['parfile'],**kwargs)


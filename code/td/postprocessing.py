from prospect.models import model_setup
import os, prosp_dutils, hickle, sys, time
import numpy as np
import argparse
from copy import deepcopy
from prospector_io import load_prospector_data, create_prosp_filename
import prosp_dynesty_plots
from dynesty.plotting import _quantile as weighted_quantile

def set_sfh_time_vector(res,ncalc):
    """if parameterized, calculate linearly in 100 steps from t=0 to t=tage
    if nonparameterized, calculate at bin edges.
    """
    if 'tage' in res['model'].theta_labels():
        nt = 100
        idx = np.array(res['model'].theta_labels()) == 'tage'
        maxtime = np.max(res['chain'][:ncalc,idx])
        t = np.linspace(0,maxtime,num=nt)
    elif 'agebins' in res['model'].params:
        in_years = 10**res['model'].params['agebins']/1e9
        t = np.concatenate((np.ravel(in_years)*0.9999, np.ravel(in_years)*1.001))
        t.sort()
        t = t[1:-1] # remove older than oldest bin, younger than youngest bin
        t = np.clip(t,1e-3,np.inf) # nothing younger than 1 Myr!
    else:
        sys.exit('ERROR: not sure how to set up the time array here!')
    return t

def calc_extra_quantities(res, sps, obs, ncalc=3000, 
                          **kwargs):
    """calculate extra quantities: star formation history, stellar mass, spectra, photometry, etc
    """

    # calculate maxprob
    # and ensure that maxprob stored is the same as calculated now 
    # don't recalculate lnprobability after we fix MassMet
    res['lnprobability'] = res['lnlikelihood'] + res['model'].prior_product(res['chain'])
    amax = res['lnprobability'].argmax()
    current_maxprob = prosp_dutils.test_likelihood(sps, res['model'], res['obs'], 
                                                   res['chain'][amax], 
                                                   res['run_params']['param_file'])
    print 'Best-fit lnprob currently: {0}'.format(current_maxprob[0])
    print 'Best-fit lnprob during sampling: {0}'.format(res['lnprobability'].argmax())

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
            'weights':res['weights'][sample_idx]
            }
    fmt = {'chain':np.zeros(shape=ncalc),'q50':0.0,'q84':0.0,'q16':0.0}

    # thetas
    parnames = res['model'].theta_labels()
    for i, p in enumerate(parnames):  
        q50, q16, q84 = weighted_quantile(res['chain'][:,i], np.array([0.5, 0.16, 0.84]), weights=res['weights'])
        eout['thetas'][p] = {'q50': q50, 'q16': q16, 'q84': q84}

    # extras
    extra_parnames = ['half_time','sfr_100','ssfr_100','stellar_mass','lir','luv','lmir','lbol']
    if 'fagn' in parnames:
        extra_parnames += ['l_agn', 'fmir']
    for p in extra_parnames: eout['extras'][p] = deepcopy(fmt)

    # sfh
    eout['sfh']['t'] = set_sfh_time_vector(res,ncalc)
    eout['sfh']['sfh'] = np.zeros(shape=(ncalc,eout['sfh']['t'].shape[0]))

    # observables
    eout['obs']['spec'] = np.zeros(shape=(ncalc,sps.wavelengths.shape[0]))
    eout['obs']['mags'] = np.zeros(shape=(ncalc,len(res['obs']['filters'])))
    eout['obs']['lam_obs'] = sps.wavelengths
    elines = ['H beta 4861', 'H alpha 6563']
    eout['obs']['elines'] = {key: {'ew': deepcopy(fmt), 'flux': deepcopy(fmt)} for key in elines}
    eout['obs']['dn4000'] = deepcopy(fmt)
    res['model'].params['nebemlineinspec'] = True

    # sample in the posterior
    for jj,sidx in enumerate(sample_idx):

        # bookkeepping
        t1 = time.time()

        # model call
        thetas = res['chain'][sidx,:]
        eout['obs']['spec'][jj,:],eout['obs']['mags'][jj,:],sm = res['model'].mean_model(thetas, res['obs'], sps=sps)

        # calculate SFH-based quantities
        sfh_params = prosp_dutils.find_sfh_params(res['model'],thetas,
                                                  res['obs'],sps,sm=sm)
        eout['extras']['stellar_mass']['chain'][jj] = sfh_params['mass']
        eout['sfh']['sfh'][jj,:] = prosp_dutils.return_full_sfh(eout['sfh']['t'], sfh_params)
        eout['extras']['half_time']['chain'][jj] = prosp_dutils.halfmass_assembly_time(sfh_params)
        eout['extras']['sfr_100']['chain'][jj] = prosp_dutils.calculate_sfr(sfh_params, 0.1,  minsfr=-np.inf, maxsfr=np.inf)
        eout['extras']['ssfr_100']['chain'][jj] = eout['extras']['sfr_100']['chain'][jj].squeeze() / eout['extras']['stellar_mass']['chain'][jj].squeeze()

        # calculate AGN parameters if necessary
        if 'fagn' in parnames:
            eout['extras']['l_agn']['chain'][jj] = prosp_dutils.measure_agn_luminosity(thetas[parnames.index('fagn')],sps,sfh_params['mformed'])

        # lbol
        eout['extras']['lbol']['chain'][jj] = prosp_dutils.measure_lbol(sps,sfh_params['mformed'])

        # measure from rest-frame spectrum
        t2 = time.time()
        props = prosp_dutils.measure_restframe_properties(sps, thetas = thetas,
                                                          model=res['model'],
                                                          measure_ir=True, measure_luv=True, measure_mir=True, 
                                                          emlines=elines)
        eout['extras']['lir']['chain'][jj] = props['lir']
        eout['extras']['luv']['chain'][jj] = props['luv']
        eout['extras']['lmir']['chain'][jj] = props['lmir']
        eout['obs']['dn4000']['chain'][jj] = props['dn4000']
        for e in elines: 
            eout['obs']['elines'][e]['flux']['chain'][jj] = props['emlines'][e]['flux']
            eout['obs']['elines'][e]['ew']['chain'][jj] = props['emlines'][e]['eqw']

        if 'fagn' in parnames:
            nagn_thetas = deepcopy(thetas)
            nagn_thetas[parnames.index('fagn')] = 0.0
            props = prosp_dutils.measure_restframe_properties(sps, thetas=nagn_thetas, model=res['model'], measure_mir=True)
            eout['extras']['fmir']['chain'][jj] = (eout['extras']['lmir']['chain'][jj]-props['lmir'])/eout['extras']['lmir']['chain'][jj]

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

    return eout

def post_processing(param_name, objname=None, overwrite=True, **kwargs):
    """Driver. Loads output, runs post-processing routine.
    overwrite=False will return immediately if post-processing file already exists.
    kwargs are passed to calc_extra_quantities
    """

    # bookkeeping: where are we coming from and where are we going?
    pfile = model_setup.import_module_from_file(param_name)
    run_outfile = pfile.run_params['outfile']
    obj_outfile = "/".join(run_outfile.split('/')[:-1]) + '/' + objname
    run_name = run_outfile.split('/')[-2]
    plot_outfolder = os.getenv('APPS')+'/prospector_alpha/plots/'+run_name+'/'

    # check for output folder, create if necessary
    if not os.path.isdir(plot_outfolder):
        os.makedirs(plot_outfolder)

    # I/O
    res, powell_results, model, eout = load_prospector_data(obj_outfile,hdf5=True,load_extra_output=False)
    if res is None:
        print 'there are no sampling results! returning.'
        return
    if (eout is not None) & (not overwrite):
        print 'post-processing file already exists! returning.'
        return

    # make filenames local...
    print 'Performing post-processing on ' + objname
    for key in res['run_params']:
        if type(res['run_params'][key]) == unicode:
            if 'prospector_alpha' in res['run_params'][key]:
                res['run_params'][key] = os.getenv('APPS')+'/prospector_alpha'+res['run_params'][key].split('prospector_alpha')[-1]
    sps = pfile.load_sps(**res['run_params'])
    obs = pfile.load_obs(**res['run_params'])

    # sample from chain
    extra_output = calc_extra_quantities(res,sps,obs,**kwargs)
    
    # create post-processing name, dump info
    _, _, extra_filename = create_prosp_filename(obj_outfile)
    hickle.dump(extra_output,open(extra_filename, "w"))

    # make standard plots
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

    args = vars(parser.parse_args())
    kwargs = {}
    for key in args.keys(): kwargs[key] = args[key]

    print kwargs
    post_processing(kwargs['parfile'],**kwargs)


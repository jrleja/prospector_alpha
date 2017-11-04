import numpy as np
from scipy.optimize import brentq
from prosp_dutils import integrate_sfh, chop_chain, find_sfh_params
from prospector_io import load_prospector_data
from prospect.models import model_setup
import os, copy

def sfh_half_time(x,sfh_params,c):

    '''
    wrapper for use with halfmass assembly time
    '''
    # check for nonparametric
    sf_start = sfh_params['sf_start']
    if sfh_params['sf_start'].shape[0] == 0:
        sf_start = 0.0
    return integrate_sfh(sf_start,x,sfh_params)-c

def halfmass_assembly_time(sfh_params,c=0.5):

    try:
        half_time = brentq(sfh_half_time, 0, 14, args=(sfh_params,c), rtol=1.48e-08, maxiter=1000)
    except ValueError:
        # big problem
        warnings.warn("You've passed SFH parameters that don't allow t_half to be calculated. Check for bugs.", UserWarning)
        half_time = np.nan

    # define age of galaxy
    tgal = sfh_params['tage']
    if tgal.shape[0] == 0:
        tgal = np.max(10**sfh_params['agebins']/1e9)

    return tgal-half_time

def sample_posterior(param_name=None):

    # I/O
    paramfile = model_setup.import_module_from_file(param_name)
    outname = paramfile.run_params['outfile']
    sample_results, powell_results, model, eout = load_prospector_data(outname,hdf5=True,load_extra_output=True)

    # create useful quantities
    sample_results['flatchain'] = chop_chain(sample_results['chain'],**sample_results['run_params'])
    sample_results['flatprob'] = chop_chain(sample_results['lnprobability'],**sample_results['run_params'])
    sps = paramfile.load_sps(**sample_results['run_params'])
    obs = paramfile.load_obs(**sample_results['run_params'])

    # sample from posterior
    nsamp = 3000
    good = np.isfinite( sample_results['flatprob']) == True
    sample_idx = np.random.choice(np.where(good)[0],nsamp)

    # define outputs
    mfrac = np.linspace(0,0.95,20)
    mfrac_out = np.zeros(shape=(nsamp,mfrac.shape[0]))
    for jj,idx in enumerate(sample_idx):
        print jj
        ##### model call, to set parameters
        thetas = copy.copy(sample_results['flatchain'][idx])
        spec,mags,sm = sample_results['model'].mean_model(thetas, sample_results['obs'], sps=sps)

        ##### extract sfh parameters
        sfh_params = find_sfh_params(sample_results['model'],thetas,
                                     sample_results['obs'],sps,sm=sm)

        for ii,m in enumerate(mfrac): mfrac_out[jj,ii] = halfmass_assembly_time(sfh_params,c=m)

    # fixing negatives
    mfrac_out = np.clip(mfrac_out, 0.0, np.inf)
    # write out
    out = np.percentile(mfrac_out, [50,84,16],axis=0)
    with open('out.txt', 'w') as f:
        f.write('# mass_fraction median_time err_up err_down\n')
        for ii in range(out.shape[1]):
            f.write("{:.2f}".format(mfrac[ii]) + ' '+"{:.3f}".format(out[0,ii]) + ' '+"{:.3f}".format(out[1,ii]) + ' '+"{:.3f}".format(out[2,ii]) + ' ')
            f.write('\n')



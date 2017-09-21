import numpy as np
import os
from prospect.models import priors, sedmodel
from prospect.sources import CSPBasis
from sedpy import observate
from astropy.cosmology import WMAP9
from td_io import load_zp_offsets
from scipy.stats import truncnorm
from astropy.io import ascii

lsun = 3.846e33
pc = 3.085677581467192e18  # in cm

lightspeed = 2.998e18  # AA/s
to_cgs = lsun/(4.0 * np.pi * (pc*10)**2)
jansky_mks = 1e-26

#############
# RUN_PARAMS
#############
APPS = os.getenv('APPS')
run_params = {'verbose':True,
              'debug': False,
              'outfile': APPS+'/prospector_alpha/results/td_fastmimic/GOODSS_47721',
              'nofork': True,
              # dynesty params
              'nested_bound': 'multi', # bounding method
              'nested_sample': 'rwalk', # sampling method
              'nested_bootstrap': 10, # how much bootstrapping? tone down a bit
              'nested_nlive_batch': 200, # size of live point "batches"
              'nested_nlive_init': 100, # number of initial live points
              'nested_weight_kwargs': {'pfrac': 1.0}, # weight posterior over evidence by 100%
              'nested_dlogz_init': 0.01,
              # Model info
              'zcontinuous': 2,
              'compute_vega_mags': False,
              'interp_type': 'logarithmic',
              # Data info (phot = .cat, dat = .dat, fast = .fout)
              'datdir':APPS+'/prospector_alpha/data/3dhst/',
              'runname': 'td',
              'objname':'GOODSS_47721'
              }
############
# OBS
#############
def load_obs(objname=None, datdir=None, runname=None, err_floor=0.05, zperr=True, **extras):
    """
    objname: number of object in the 3D-HST COSMOS photometric catalog
    err_floor: the fractional error floor (0.05 = 5% floor)
    zp_err: inflate the errors by the zeropoint offsets from Skelton+14
    """

    ### open file, load data
    photname = datdir + objname.split('_')[0] + '_' + runname + '.cat'
    with open(photname, 'r') as f:
        hdr = f.readline().split()
    dtype = np.dtype([(hdr[1],'S20')] + [(n, np.float) for n in hdr[2:]])
    dat = np.loadtxt(photname, comments = '#', delimiter=' ',
                     dtype = dtype)

    ### extract filters, fluxes, errors for object
    # from ReadMe: "All fluxes are normalized to an AB zeropoint of 25, such that: magAB = 25.0-2.5*log10(flux)
    obj_idx = (dat['id'] == objname.split('_')[-1])
    filters = np.array([f[2:] for f in dat.dtype.names if f[0:2] == 'f_'])
    flux = np.squeeze([dat[obj_idx]['f_'+f] for f in filters])
    unc = np.squeeze([dat[obj_idx]['e_'+f] for f in filters])

    ### define photometric mask, convert to maggies
    phot_mask = (flux != unc) & (flux != -99.0) & (unc > 0)
    mips_idx = np.array(['mips_24um' in f for f in filters],dtype=bool)
    phot_mask[mips_idx] = False # NO DUST EMISSION FOR FAST!
    maggies = flux/(1e10)
    maggies_unc = unc/(1e10)

    ### implement error floor
    maggies_unc = np.clip(maggies_unc, maggies*err_floor, np.inf)

    ### inflate errors by zeropoint offsets from Table 11, Skelton+14
    # ~5% to ~20% effect
    if zperr:
        zp_offsets = load_zp_offsets(None)
        band_names = np.array([x['Band'].lower()+'_'+x['Field'].lower() for x in zp_offsets])
        for ii,f in enumerate(filters):
            match = band_names == f
            if match.sum():
                maggies_unc[ii] = ( (maggies_unc[ii]**2) + (maggies[ii]*(1-zp_offsets[match]['Flux-Correction'][0]))**2 ) **0.5

    ### if we have super negative flux, then mask it !
    ### where super negative is <0 with 95% certainty
    neg = (maggies < 0) & (np.abs(maggies/maggies_unc) > 2)
    phot_mask[neg] = False

    ### build output dictionary
    obs = {}
    obs['filters'] = observate.load_filters(filters)
    obs['wave_effective'] = np.array([filt.wave_effective for filt in obs['filters']])
    obs['phot_mask'] = phot_mask
    obs['maggies'] = maggies
    obs['maggies_unc'] =  maggies_unc
    obs['wavelength'] = None
    obs['spectrum'] = None
    obs['logify_spectrum'] = False

    return obs

##########################
# TRANSFORMATION FUNCTIONS
##########################
def load_gp(**extras):
    return None, None

def transform_logmass_to_mass(mass=None, logmass=None, **extras):
    return 10**logmass

def transform_logtau_to_tau(tau=None, logtau=None, **extras):
    return 10**logtau

#############
# MODEL_PARAMS
#############
model_params = []

###### BASIC PARAMETERS ##########
model_params.append({'name': 'zred', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
                        'units': '',
                        'prior': priors.TopHat(mini=0.0, maxi=4.0)})

model_params.append({'name': 'add_igm_absorption', 'N': 1,
                        'isfree': False,
                        'init': 1,
                        'units': None,
                        'prior_function': None,
                        'prior_args': None})

model_params.append({'name': 'add_agb_dust_model', 'N': 1,
                        'isfree': False,
                        'init': False,
                        'units': None,
                        'prior_function': None,
                        'prior_args': None})
                        
model_params.append({'name': 'logmass', 'N': 1,
                        'isfree': True,
                        'init': 10.0,
                        'units': 'Msun',
                        'prior': priors.TopHat(mini=5.0, maxi=13.0)})

model_params.append({'name': 'mass', 'N': 1,
                        'isfree': False,
                        'init': 1e10,
                        'depends_on': transform_logmass_to_mass,
                        'units': 'Msun',
                        'prior': priors.TopHat(mini=1e5, maxi=1e13)})

model_params.append({'name': 'logzsol', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
                        'init_disp': 0.4,
                        'log_param': True,
                        'units': r'$\log (Z/Z_\odot)$',
                        'prior': priors.TopHat(mini=-1.98, maxi=0.19)})
                        
###### SFH   ########
model_params.append({'name': 'sfh', 'N': 1,
                        'isfree': False,
                        'init': 1,
                        'units': 'type',
                        'prior_function_name': None,
                        'prior_args': None})

model_params.append({'name': 'tau', 'N': 1,
                        'isfree': False,
                        'init': 10,
                        'depends_on': transform_logtau_to_tau,
                        'init_disp': 0.5,
                        'units': 'Gyr',
                        'prior': priors.TopHat(mini=0.03, maxi=100)})

model_params.append({'name': 'logtau', 'N': 1,
                        'isfree': True,
                        'init': 1,
                        'init_disp': 0.5,
                        'units': 'Gyr',
                        'prior': priors.TopHat(mini=-1.52, maxi=2.0)})

model_params.append({'name': 'tage', 'N': 1,
                        'isfree': True,
                        'init': 1.0,
                        'units': 'Gyr',
                        'prior': priors.TopHat(mini=0.01, maxi=14.0)})

########    IMF  ##############
model_params.append({'name': 'imf_type', 'N': 1,
                             'isfree': False,
                             'init': 1, #1 = chabrier
                             'units': None,
                             'prior_function_name': None,
                             'prior_args': None})

######## Dust Absorption ##############
model_params.append({'name': 'dust_type', 'N': 1,
                        'isfree': False,
                        'init': 2,
                        'units': 'index',
                        'prior_function_name': None,
                        'prior_args': None})

model_params.append({'name': 'dust2', 'N': 1,
                        'isfree': True,
                        'init': 0.0,
                        'init_disp': 0.2,
                        'units': '',
                        'prior': priors.TopHat(mini=0.0, maxi=4.0)})

###### Dust Emission ##############
model_params.append({'name': 'add_dust_emission', 'N': 1,
                        'isfree': False,
                        'init': 0,
                        'units': None,
                        'prior_function': None,
                        'prior_args': None})

###### Nebular Emission ###########
model_params.append({'name': 'add_neb_emission', 'N': 1,
                        'isfree': False,
                        'init': False,
                        'units': r'log Z/Z_\odot',
                        'prior_function_name': None,
                        'prior_args': None})


####### Calibration ##########
model_params.append({'name': 'peraa', 'N': 1,
                     'isfree': False,
                     'init': False})

model_params.append({'name': 'mass_units', 'N': 1,
                     'isfree': False,
                     'init': 'mstar'})

#### resort list of parameters 
#### so that major ones are fit first
parnames = [m['name'] for m in model_params]
fit_order = ['logmass','tage', 'tau', 'dust2']
tparams = [model_params[parnames.index(i)] for i in fit_order]
for param in model_params: 
    if param['name'] not in fit_order:
        tparams.append(param)
model_params = tparams

###### Redefine SPS ######
def load_sps(**extras):
    sps = CSPBasis(**extras)
    return sps

def load_model(objname=None, datdir=None, runname=None, agelims=[], **extras):

    # all we need is the redshift
    datname = datdir + objname.split('_')[0] + '_' + runname + '.dat'
    dat = ascii.read(datname)
    idx = dat['phot_id'] == int(objname.split('_')[-1])
    zred = float(dat['z_best'][idx])

    # set tage_max, fix redshift
    n = [p['name'] for p in model_params]
    tuniv = WMAP9.age(zred).value
    model_params[n.index('tage')]['prior'].update(maxi=tuniv)
    model_params[n.index('zred')]['init'] = zred

    return sedmodel.SedModel(model_params)

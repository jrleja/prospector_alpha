import numpy as np
import os
from prospect.models import priors, sedmodel
from prospect.sources import CSPBasis
from sedpy import observate
from astropy.cosmology import WMAP9
from td_io import load_zp_offsets
tophat = priors.tophat
logarithmic = priors.logarithmic
APPS = os.getenv('APPS')

#############
# RUN_PARAMS
#############

run_params = {'verbose':True,
              'debug': False,
              'outfile': os.getenv('APPS')+'/threedhst_bsfh/results/td_massive/td_massive',
              'nofork': True,
              # Optimizer params
              'ftol':0.5e-5, 
              'maxfev':5000,
              # MCMC params
              'nwalkers':310,
              'nburn':[100,150], 
              'niter': 800,
              'interval': 0.2,
              # Model info
              'zcontinuous': 2,
              'compute_vega_mags': False,
              'initial_disp':0.1,
              'interp_type': 'logarithmic',
              'agelims': [0.0,8.0,8.5,9.0,9.5,9.8,10.0],
              # Data info
              'photname':APPS+'/threedhst_bsfh/data/3dhst/COSMOS_td_massive.cat',
              'datname':APPS+'/threedhst_bsfh/data/3dhst/COSMOS_td_massive.dat',
              'fastname':APPS+'/threedhst_bsfh/data/3dhst/COSMOS_td_massive.fout',
              'objname':'26720',
              }
run_params['outfile'] = run_params['outfile']+'_'+run_params['objname']

############
# OBS
#############
def load_obs(photname, objname, err_floor=0.05, zperr=True, **extras):

    ''' 
    photname: photometric file location
    objname: number of object in the 3D-HST COSMOS photometric catalog
    err_floor: the fractional error floor (0.05 = 5% floor)
    zp_err: inflate the errors by the zeropoint offsets from Skelton+14
    '''


    ### open file, load data
    with open(photname, 'r') as f:
        hdr = f.readline().split()
    dtype = np.dtype([(hdr[1],'S20')] + [(n, np.float) for n in hdr[2:]])
    dat = np.loadtxt(photname, comments = '#', delimiter=' ',
                     dtype = dtype)

    ### extract filters, fluxes, errors for object
    # from ReadMe: "All fluxes are normalized to an AB zeropoint of 25, such that: magAB = 25.0-2.5*log10(flux)
    obj_idx = (dat['id'] == objname)
    filters = np.array([f[2:] for f in dat.dtype.names if f[0:2] == 'f_'])
    flux = np.squeeze([dat[obj_idx]['f_'+f] for f in filters])
    unc = np.squeeze([dat[obj_idx]['e_'+f] for f in filters])

    ### add correction to MIPS magnitudes (only MIPS 24 right now!)
    # due to weird MIPS filter conventions
    dAB_mips_corr = np.array([-0.03542,-0.07669,-0.03807]) # 24, 70, 160, in AB magnitudes
    dflux = 10**(-dAB_mips_corr/2.5)

    mips_idx = np.array(['mips_24um' in f for f in filters],dtype=bool)
    flux[mips_idx] *= dflux[0]
    unc[mips_idx] *= dflux[0]

    ### define photometric mask, convert to maggies
    phot_mask = (flux != unc) & (flux != -99.0)
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
def transform_logmass_to_mass(mass=None, logmass=None, **extras):

    return 10**logmass

def load_gp(**extras):
    return None, None

def add_dust1(dust2=None, **extras):

    return 0.86*dust2

def tie_gas_logz(logzsol=None, **extras):

    return logzsol

#############
# MODEL_PARAMS
#############

model_params = []

###### BASIC PARAMETERS ##########
model_params.append({'name': 'zred', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
                        'units': '',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.0, 'maxi':4.0}})

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
                        'prior_function': priors.tophat,
                        'prior_args':{'mini':1.0, 'maxi':14.0}})

model_params.append({'name': 'mass', 'N': 1,
                        'isfree': False,
                        'init': 1e10,
                        'depends_on': transform_logmass_to_mass,
                        'units': 'Msun',
                        'prior_function': priors.tophat,
                        'prior_args':{'mini':1e1, 'maxi':1e14}})

model_params.append({'name': 'logzsol', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
                        'init_disp': 0.4,
                        'log_param': True,
                        'units': r'$\log (Z/Z_\odot)$',
                        'prior_function': tophat,
                        'prior_args': {'mini':-2.0, 'maxi':0.19}})
                        
###### SFH   ########
model_params.append({'name': 'sfh', 'N': 1,
                        'isfree': False,
                        'init': 1,
                        'units': 'type',
                        'prior_function_name': None,
                        'prior_args': None})

model_params.append({'name': 'tau', 'N': 1,
                        'isfree': True,
                        'init': 10,
                        'init_disp': 0.5,
                        'units': 'Gyr',
                        'prior_function':tophat,
                        'prior_args': {'mini':0.1,
                                       'maxi':100.0}})

model_params.append({'name': 'tage', 'N': 1,
                        'isfree': True,
                        'init': 1.0,
                        'units': 'Gyr',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.01, 'maxi':14.0}})

model_params.append({'name': 'tburst', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
                        'init_disp': 1.0,
                        'units': '',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.0, 'maxi':10.0}})

model_params.append({'name': 'fburst', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
                        'init_disp': 0.5,
                        'units': '',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.0, 'maxi':0.2}})

model_params.append({'name': 'fconst', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
                        'units': '',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.0, 'maxi':1.0}})

model_params.append({'name': 'sf_start', 'N': 1,
                        'isfree': False,
                        'reinit': True,
                        'init': 0.0,
                        'units': 'Gyr',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.0,
                                       'maxi':14.0}})

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
                        'prior_function': tophat,
                        'prior_args': {'mini':0.0,
                                       'maxi':4.0}})

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
                        
model_params.append({'name': 'gas_logz', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
                        'units': r'log Z/Z_\odot',
                        'prior_function': tophat,
                        'prior_args': {'mini':-2.0, 'maxi':0.5}})

model_params.append({'name': 'gas_logu', 'N': 1,
                        'isfree': False,
                        'init': -2.0,
                        'units': '',
                        'prior_function': tophat,
                        'prior_args': {'mini':-4, 'maxi':-1}})


####### Calibration ##########
model_params.append({'name': 'phot_jitter', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
                        'units': 'mags',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.0, 'maxi':0.2}})
model_params.append({'name': 'peraa', 'N': 1,
                     'isfree': False,
                     'init': False})

model_params.append({'name': 'mass_units', 'N': 1,
                     'isfree': False,
                     'init': 'mstar'})

#### resort list of parameters 

#### resort list of parameters 
#### so that major ones are fit first
parnames = [m['name'] for m in model_params]
fit_order = ['logmass','tage', 'tau', 'dust2']
tparams = [model_params[parnames.index(i)] for i in fit_order]
for param in model_params: 
    if param['name'] not in fit_order:
        tparams.append(param)
model_params = tparams

###### REDEFINE MODEL FOR MY OWN NEFARIOUS PURPOSES ######
class BurstyModel(sedmodel.SedModel):

    def prior_product(self, theta):
        """
        Return a scalar which is the ln of the product of the prior
        probabilities for each element of theta.  Requires that the
        prior functions are defined in the theta descriptor.

        :param theta:
            Iterable containing the free model parameter values.

        :returns lnp_prior:
            The log of the product of the prior probabilities for
            these parameter values.
        """  
        lnp_prior = 0

        for k, v in self.theta_index.iteritems():
            start, end = v
            this_prior = np.sum(self._config_dict[k]['prior_function']
                                (theta[start:end], **self._config_dict[k]['prior_args']))

            if (not np.isfinite(this_prior)):
                print('WARNING: ' + k + ' is out of bounds')
            lnp_prior += this_prior
        return lnp_prior

def load_sps(**extras):

    sps = CSPBasis(**extras)
    return sps

def load_model(objname='',datname='',fastname='', agelims=[], **extras):

    ###### REDSHIFT ######
    ### open file, load data
    '''
    with open(datname, 'r') as f:
        hdr = f.readline().split()
    dtype = np.dtype([(hdr[1],'S20')] + [(n, np.float) for n in hdr[2:]])
    dat = np.loadtxt(datname, comments = '#', delimiter=' ',
                     dtype = dtype)
    '''
    with open(fastname, 'r') as f:
        hdr = f.readline().split()
    dtype = np.dtype([(hdr[1],'S20')] + [(n, np.float) for n in hdr[2:]])
    fast = np.loadtxt(fastname, comments = '#', delimiter=' ', dtype = dtype)
    idx = fast['id'] == objname
    zred = fast['z'][idx][0]

    #### INITIAL VALUES
    tau = 10**fast['ltau'][idx][0]/1e9
    tage = 10**fast['lage'][idx][0]/1e9
    logmass = fast['lmass'][idx][0]
    dust2 = fast['Av'][0] / 1.086

    n = [p['name'] for p in model_params]
    model_params[n.index('tau')]['init'] = tau
    model_params[n.index('tage')]['init'] = tage
    model_params[n.index('logmass')]['init'] = logmass
    model_params[n.index('dust2')]['init'] = dust2

    #### CALCULATE TUNIV #####
    tuniv = WMAP9.age(zred).value

    #### INSERT REDSHIFT INTO MODEL PARAMETER DICTIONARY ####
    zind = n.index('zred')
    model_params[zind]['init'] = zred

    #### CREATE MODEL
    model = BurstyModel(model_params)

    return model

model_type = BurstyModel

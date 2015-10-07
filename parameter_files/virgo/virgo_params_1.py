import numpy as np
import os
from bsfh import priors, sedmodel
from astropy.cosmology import WMAP9
tophat = priors.tophat
logarithmic = priors.logarithmic

#############
# RUN_PARAMS
#############

run_params = {'verbose':True,
              'outfile':os.getenv('APPS')+'/threedhst_bsfh/results/virgo/virgo',
              'ftol':0.5e-5, 
              'maxfev':5000,
              'nwalkers':496,
              'nburn':[32,32,64], 
              'niter': 1000,
              'initial_disp':0.1,
              'debug': False,
              'spec': False, 
              'phot':True,
              'photname':os.getenv('APPS')+'/threedhst_bsfh/data/virgo/virgosamp.dat',
              'objname':'cw4293',
              }

############
# OBS
#############

def translate_filters(bfilters, full_list = False):
    '''
    translate filter names to FSPS standard
    this is ALREADY a mess, clean up soon!
    suspect there are smarter routines to do this in python-fsps
    '''

    # this is necessary for my code
    # to calculate effective wavelength
    # in threed_dutils
    translate = {
    'SDSS_u': 'SDSS Camera u Response Function, airmass = 1.3 (June 2001)',
    'SDSS_g': 'SDSS Camera g Response Function, airmass = 1.3 (June 2001)',
    'SDSS_r': 'SDSS Camera r Response Function, airmass = 1.3 (June 2001)',
    'SDSS_i': 'SDSS Camera i Response Function, airmass = 1.3 (June 2001)',
    'SDSS_z': 'SDSS Camera z Response Function, airmass = 1.3 (June 2001)',
    '2MASS_J': '2MASS J filter (total response w/atm)',
    '2MASS_H': '2MASS H filter (total response w/atm)',
    '2MASS_Ks': '2MASS Ks filter (total response w/atm)',
    'IRAC_1': 'IRAC Channel 1',
    'IRAC_2': 'IRAC Channel 2',
    'IRAC_3': 'IRAC Channel 3',
    'IRAC_4': 'IRAC CH4',
    'MIPS_24': 'MIPS 24um',
    'MIPS_70': 'MIPS 70um',
    'MIPS_160': 'MIPS 160um',
    'SPIRE_250': 'Herschel SPIRE 250um',
    'SPIRE_350': 'Herschel SPIRE 350um',
    'SPIRE_500': 'Herschel SPIRE 500um'
    }

    return np.array([translate[f] for f in bfilters])


def load_obs_virgo(photname, objname):
    """
    let's do this
    """
    obs = {}

    # written out by hand from Louise's email
    filters = np.array(['SDSS_u', 'SDSS_g', 'SDSS_r', 'SDSS_i', 'SDSS_z', '2MASS_J', '2MASS_H', '2MASS_Ks', 
                        'IRAC_1', 'IRAC_2', 'IRAC_3', 'IRAC_4', 'MIPS_24', 'MIPS_70', 'MIPS_160', 'SPIRE_250',
                        'SPIRE_350', 'SPIRE_500'])

    # load photometry
    fl_err = []
    for filt in filters: fl_err+=[(filt,'f8')] + [(filt+'_err','f8')]
    names = [('name','S20')] + [('zred','f8')] + fl_err
    dat = np.loadtxt(photname,dtype = names)

    # find object
    if objname is not None:
        idx = dat['name'] == objname
    else:
        idx = np.ones_like(dat['name'],dtype=bool)
    
    # extract fluxes+uncertainties for all objects
    mag_fields = [f for f in dat.dtype.names if (f[-3:] != 'err') and (f != 'name') and (f != 'zred')]
    magunc_fields = [f for f in dat.dtype.names if f[-3:] == 'err']

    # extract fluxes for particular object
    flux = np.array([np.squeeze(dat[f][idx]) for f in mag_fields])
    flux_err = np.array([np.squeeze(dat[f][idx]) for f in magunc_fields])

    # phot mask
    phot_mask = ~np.isclose(flux, -0.999, atol=1e-05)

    # convert from Jy to maggies
    flux = flux / 3631.
    flux_err = flux_err / 3631

    # load wave_effective
    from translate_filter import calc_lameff_for_fsps
    wave_effective = calc_lameff_for_fsps(translate_filters(filters))

    # build output dictionary
    obs['wave_effective'] = wave_effective
    obs['filters'] = filters
    obs['phot_mask'] = phot_mask
    obs['maggies'] = flux
    obs['maggies_unc'] =  flux_err
    obs['wavelength'] = None
    obs['spectrum'] = None
    obs['zred'] = dat['zred'][idx]

    return obs

obs = load_obs_virgo(run_params['photname'], run_params['objname'])

#############
# MODEL_PARAMS
#############

def transform_sftanslope_to_sfslope(sf_slope=None,sf_tanslope=None,**extras):

    return np.tan(sf_tanslope)

def transform_delt_to_sftrunc(tage=None, delt_trunc=None, **extras):

    return tage*delt_trunc

def transform_logtau_to_tau(tau=None, logtau=None, **extras):

    return 10**logtau

def add_dust1(dust2=None, **extras):

    return 0.86*dust2

#### SET SFH PRIORS #####
###### REDSHIFT ######
zred =  obs['zred']

#### TUNIV #####
tuniv = WMAP9.age(zred).value
run_params['tuniv']       = tuniv

#### TAGE #####
tage_maxi = tuniv
tage_init = 1.1
tage_mini  = 0.11      # FSPS standard

model_params = []

param_template = {'name':'', 'N':1, 'isfree': False,
                  'init':0.0, 'units':'', 'label':'',
                  'prior_function_name': None, 'prior_args': None}

###### BASIC PARAMETERS ##########
model_params.append({'name': 'zred', 'N': 1,
                        'isfree': False,
                        'init': zred,
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
                        'init': True,
                        'units': None,
                        'prior_function': None,
                        'prior_args': None})
                        
model_params.append({'name': 'mass', 'N': 1,
                        'isfree': True,
                        'init': 1e10,
                        'init_disp': 0.25,
                        'units': r'M_\odot',
                        'prior_function': tophat,
                        'prior_args': {'mini':1e5,'maxi':1e14}})

model_params.append({'name': 'pmetals', 'N': 1,
                        'isfree': False,
                        'init': -99,
                        'units': '',
                        'prior_function': None,
                        'prior_args': {'mini':-3, 'maxi':-1}})

model_params.append({'name': 'logzsol', 'N': 1,
                        'isfree': True,
                        'init': -0.5,
                        'init_disp': 0.15,
                        'units': r'$\log (Z/Z_\odot)$',
                        'prior_function': tophat,
                        'prior_args': {'mini':-1.98, 'maxi':0.19}})
                        
###### SFH   ########
model_params.append({'name': 'sfh', 'N': 1,
                        'isfree': False,
                        'init': 5,
                        'units': 'type',
                        'prior_function_name': None,
                        'prior_args': None})

model_params.append({'name': 'logtau', 'N': 1,
                        'isfree': True,
                        'init': 0.0,
                        'init_disp': 0.25,
                        'units': 'log(Gyr)',
                        'prior_function': tophat,
                        'prior_args': {'mini':-1.0,
                                       'maxi':2.0}})

model_params.append({'name': 'tau', 'N': 1,
                        'isfree': False,
                        'init': 1.0,
                        'depends_on': transform_logtau_to_tau,
                        'units': 'Gyr',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.1,
                                       'maxi':100}})

model_params.append({'name': 'tage', 'N': 1,
                        'isfree': True,
                        'init': tage_init,
                        'init_disp': 0.25,
                        'units': 'Gyr',
                        'prior_function': tophat,
                        'prior_args': {'mini':tage_mini, 'maxi':tage_maxi}})

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
                        'init': 0.0,
                        'units': 'Gyr',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.0,'maxi':14.0}})

model_params.append({'name': 'delt_trunc', 'N': 1,
                        'isfree': True,
                        'init': 1.0,
                        'init_disp': 0.1,
                        'units': '',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.0, 'maxi': 1.0}})

model_params.append({'name': 'sf_trunc', 'N': 1,
                        'isfree': False,
                        'init': 1.0,
                        'units': '',
                        'depends_on': transform_delt_to_sftrunc,
                        'prior_function': tophat,
                        'prior_args': {'mini':0, 'maxi':16}})

model_params.append({'name': 'sf_tanslope', 'N': 1,
                        'isfree': True,
                        'init': 0.0,
                        'init_disp': 0.25,
                        'units': '',
                        'prior_function': tophat,
                        'prior_args': {'mini':-np.pi/2., 'maxi': np.pi/2}})

model_params.append({'name': 'sf_slope', 'N': 1,
                        'isfree': False,
                        'init': -1.0,
                        'units': '',
                        'depends_on': transform_sftanslope_to_sfslope,
                        'prior_function': tophat,
                        'prior_args': {'mini':-np.inf,'maxi':5.0}})

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
                        'init': 4,
                        'units': 'index',
                        'prior_function_name': None,
                        'prior_args': None})
                        
model_params.append({'name': 'dust1', 'N': 1,
                        'isfree': True,
                        'init': 1.0,
                        'init_disp': 0.8,
                        'units': '',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.0, 'maxi':4.0}})

model_params.append({'name': 'dust2', 'N': 1,
                        'isfree': True,
                        'init': 1.0,
                        'units': '',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.0,'maxi':4.0}})

model_params.append({'name': 'dust_index', 'N': 1,
                        'isfree': True,
                        'init': -0.7,
                        'units': '',
                        'prior_function': priors.normal_clipped,
                        'prior_args': {'mini':-3.0, 'maxi': -0.4,'mean':-0.7,'sigma':0.5}})

model_params.append({'name': 'dust1_index', 'N': 1,
                        'isfree': False,
                        'init': -1.0,
                        'units': '',
                        'prior_function': tophat,
                        'prior_args': {'mini':-1.5, 'maxi':-0.5}})

model_params.append({'name': 'dust_tesc', 'N': 1,
                        'isfree': False,
                        'init': 7.0,
                        'units': 'log(Gyr)',
                        'prior_function_name': None,
                        'prior_args': None})

###### Dust Emission ##############
model_params.append({'name': 'add_dust_emission', 'N': 1,
                        'isfree': False,
                        'init': 1,
                        'units': None,
                        'prior_function': None,
                        'prior_args': None})

model_params.append({'name': 'duste_gamma', 'N': 1,
                        'isfree': True,
                        'init': 0.01,
                        'init_disp': 0.2,
                        'units': None,
                        'prior_function': tophat,
                        'prior_args': {'mini':0.0, 'maxi':1.0}})

model_params.append({'name': 'duste_umin', 'N': 1,
                        'isfree': True,
                        'init': 1.0,
                        'init_disp': 5.0,
                        'units': None,
                        'prior_function': tophat,
                        'prior_args': {'mini':0.1, 'maxi':25.0}})

model_params.append({'name': 'duste_qpah', 'N': 1,
                        'isfree': True,
                        'init': 3.0,
                        'init_disp': 3.0,
                        'units': 'percent',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.0, 'maxi':10.0}})

###### Nebular Emission ###########
model_params.append({'name': 'add_neb_emission', 'N': 1,
                        'isfree': False,
                        'init': 2,
                        'units': r'log Z/Z_\odot',
                        'prior_function_name': None,
                        'prior_args': None})

model_params.append({'name': 'add_neb_continuum', 'N': 1,
                        'isfree': False,
                        'init': True,
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
                        'init_disp': 0.5,
                        'units': 'fractional maggies (mags/1.086)',
                        'prior_function':tophat,
                        'prior_args': {'mini':0.0, 'maxi':0.5}})

#### resort list of parameters 
#### so that major ones are fit first
parnames = [m['name'] for m in model_params]
fit_order = ['mass', 'tage', 'logtau', 'dust2', 'delt_trunc', 'sf_tanslope', 'duste_gamma', 'logzsol', 'dust1', 'dust_index','duste_umin', 'duste_qpah']
tparams = [model_params[parnames.index(i)] for i in fit_order]
for param in model_params: 
    if param['name'] not in fit_order:
        tparams.append(param)
model_params = tparams

# name outfile
run_params['outfile'] = run_params['outfile']+'_'+run_params['objname']


###### REDEFINE MODEL FOR MY OWN NEFARIOUS PURPOSES ######
class BurstyModel(sedmodel.CSPModel):

    def theta_disps(self, thetas, initial_disp=0.1):
        """Get a vector of dispersions for each parameter to use in
        generating sampler balls for emcee's Ensemble sampler.

        :param initial_disp: (default: 0.1)
            The default dispersion to use in case the `init_disp` key
            is not provided in the parameter configuration.  This is
            in units of the parameter, so e.g. 0.1 will result in a
            smpler ball with a dispersion that is 10% of the central
            parameter value.
        """
        disp = np.zeros(self.ndim) + initial_disp
        for par, inds in self.theta_index.iteritems():
            
            # fractional dispersion
            if par == 'mass' or \
               par == 'tage':
                disp[inds[0]:inds[1]] = self._config_dict[par].get('init_disp', initial_disp) * thetas[inds[0]:inds[1]]

            # constant (log) dispersion
            if par == 'logtau' or \
               par == 'metallicity' or \
               par == 'sf_tanslope' or \
               par == 'delt_trunc' or \
               par == 'duste_umin' or \
               par == 'duste_qpah' or \
               par == 'duste_gamma':
                disp[inds[0]:inds[1]] = self._config_dict[par].get('init_disp', initial_disp)

            # fractional dispersion with artificial floor
            if par == 'dust2' or \
               par == 'dust1' or \
               par == 'dust_index':
                disp[inds[0]:inds[1]] = (self._config_dict[par].get('init_disp', initial_disp) * thetas[inds[0]:inds[1]]**2 + \
                                         0.1**2)**0.5
            
        return disp

    def theta_disp_floor(self, thetas):
        """Get a vector of dispersions for each parameter to use as
        a floor for the walker-calculated dispersions.
        """
        disp = np.zeros(self.ndim)
        for par, inds in self.theta_index.iteritems():
            
            # constant 5% floor
            if par == 'mass':
                disp[inds[0]:inds[1]] = 0.05 * thetas[inds[0]:inds[1]]

            # constant 0.05 floor (log space, sf_slope, dust_index)
            if par == 'logzsol':
                disp[inds[0]:inds[1]] = 0.2

            if par == 'logtau':
                disp[inds[0]:inds[1]] = 0.25

            if par == 'sf_tanslope':
                disp[inds[0]:inds[1]] = 0.3

            if par == 'dust2' or \
               par == 'dust_index':
                disp[inds[0]:inds[1]] = 0.15

            if par == 'dust1':
                disp[inds[0]:inds[1]] = 0.4

            if par == 'duste_umin':
                disp[inds[0]:inds[1]] = 4.5

            if par == 'duste_qpah':
                disp[inds[0]:inds[1]] = 3.0

            if par == 'duste_gamma':
                disp[inds[0]:inds[1]] = 0.15

            # 20% floor
            if par == 'tage':
                disp[inds[0]:inds[1]] = 0.2 * thetas[inds[0]:inds[1]]

            if par == 'delt_trunc':
                disp[inds[0]:inds[1]] = 0.1
            
        return disp

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

        # implement uniqueness of outliers
        if 'gp_outlier_locs' in self.theta_index:
            start,end = self.theta_index['gp_outlier_locs']
            outlier_locs = theta[start:end]
            if len(np.unique(np.round(outlier_locs))) != len(outlier_locs):
                return -np.inf

        if 'dust1' in self.theta_index:
            if 'dust2' in self.theta_index:
                start,end = self.theta_index['dust1']
                dust1 = theta[start:end]
                start,end = self.theta_index['dust2']
                dust2 = theta[start:end]
                if dust1/4. > dust2:
                    return -np.inf

        for k, v in self.theta_index.iteritems():
            start, end = v
            lnp_prior += np.sum(self._config_dict[k]['prior_function']
                                (theta[start:end], **self._config_dict[k]['prior_args']))
        return lnp_prior

model_type = BurstyModel


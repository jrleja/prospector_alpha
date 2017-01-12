import numpy as np
import os
from prospect.models import priors, sedmodel
from prospect.sources import FastStepBasis
from sedpy import observate
from astropy.cosmology import WMAP9
tophat = priors.tophat
logarithmic = priors.logarithmic

#############
# RUN_PARAMS
#############

run_params = {'verbose':True,
              'debug': False,
              'outfile': os.getenv('APPS')+'/threedhst_bsfh/results/browndwarfs/browndwarfs',
              'nofork': True,
              # Optimizer params
              'ftol':0.5e-5, 
              'maxfev':5000,
              # MCMC params
              'nwalkers':620,
              'nburn':[150,200,400,600], 
              'niter': 2500,
              'interval': 0.2,
              # Model info
              'zcontinuous': 2,
              'compute_vega_mags': False,
              'initial_disp':0.1,
              'interp_type': 'logarithmic',
              'agelims': [0.0,8.0,8.5,9.0,9.5,9.8,10.0],
              # Data info
              'datname':os.getenv('APPS')+'/threedhst_bsfh/data/browndwarfs.csv',
              'objname':'UGCA 219',
              }
run_params['outfile'] = run_params['outfile']+'_'+run_params['objname']

############
# OBS
#############

def translate_filters(filters):

    trans_dict = {
                  'GALEX_FUV': 'galex_FUV',
                  'Swift_UVW2': 'uvot_w2',
                  'Swift_UVM2': 'uvot_m2',
                  'GALEX_NUV': 'galex_NUV',
                  'Swift_UVM1': np.nan,
                  'Swift_U': np.nan,
                  'SDSS_u': 'sdss_u0',
                  'SDSS_g': 'sdss_g0',
                  'Swift_V': np.nan,
                  'SDSS_r': 'sdss_r0',
                  'SDSS_i': 'sdss_i0',
                  'SDSS_z': 'sdss_z0',
                  '2MASS_J': 'twomass_J',
                  '2MASS_H': 'twomass_H',
                  '2MASS_Ks': 'twomass_Ks',
                  'WISE_W1': 'wise_w1',
                  'WISE_W2': 'wise_w2',
                  'WISE_W3': 'wise_w3',
                  'WISE_W4': 'wise_w4_prime',
                  'IRAC_I1': 'spitzer_irac_ch1',
                  'IRAC_I2': 'spitzer_irac_ch2',
                  'IRAC_I3': 'spitzer_irac_ch3',
                  'IRAC_I4': 'spitzer_irac_ch4',
                  'IRS_PB': np.nan,
                  'IRS_PR': np.nan,
                  'MIPS_M1': 'spitzer_mips_24'
    }

    return [trans_dict[f] for f in filters]

def load_obs(datname='', objname='', **extras):
    """
    let's do this
    """
    obs ={}

    with open(datname, 'r') as f:
        hdr = f.readline()

    hdr = hdr[1:-2].split(',')
    dtype = np.dtype([(hdr[0],'S40')] + [(n, np.float) for n in hdr[1:]])
    dat = np.loadtxt(datname, comments = '#', delimiter=',', dtype = dtype)
    names = np.array([s.strip() for s in dat['Name']])
    obj_ind = np.where(names == objname)[0][0]

    # extract fluxes+uncertainties for all objects and all filters
    mag_fields = [f for f in dat.dtype.names if f[-4:] != '_err' and (f != 'Name') and (f != 'z')]
    magunc_fields = [f for f in dat.dtype.names if f[-4:] == '_err']

    # translate filters, dump those we don't have response curves for
    filters = np.array(translate_filters(mag_fields))
    have_definition = filters != 'nan'
    filters = filters[have_definition]

    # extract fluxes for particular object
    mag = np.array([dat[obj_ind][idx] for idx in mag_fields])[have_definition]
    magunc = np.array([dat[obj_ind][idx] for idx in magunc_fields])[have_definition]
    phot_mask = (mag != 0) & (magunc != 0)

    # add correction to MIPS magnitudes (only MIPS 24 right now!)
    mips_corr = np.array([-0.03542,-0.07669,-0.03807]) # 24, 70, 160
    mag[filters == 'spitzer_mips_24'] += mips_corr[0]

    # then convert to maggies
    flux = 10**((-2./5)*mag)
    unc = magunc*flux/1.086

    # implement error floor
    unc = np.clip(unc, flux*0.05, np.inf)

    # build output dictionary
    obs['filters'] = observate.load_filters(filters)
    obs['wave_effective'] = np.array([filt.wave_effective for filt in obs['filters']])
    obs['phot_mask'] = phot_mask
    obs['maggies'] = flux
    obs['maggies_unc'] =  unc
    obs['wavelength'] = None
    obs['spectrum'] = None
    obs['logify_spectrum'] = False

    return obs
        
######################
# GENERATING FUNCTIONS
######################
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
                        'init': True,
                        'units': None,
                        'prior_function': None,
                        'prior_args': None})

model_params.append({'name': 'pmetals', 'N': 1,
                        'isfree': False,
                        'init': -99,
                        'units': '',
                        'prior_function': None,
                        'prior_args': {'mini':-3, 'maxi':-1}})

model_params.append({'name': 'logzsol', 'N': 1,
                        'isfree': True,
                        'init': -0.5,
                        'init_disp': 0.25,
                        'disp_floor': 0.2,
                        'units': r'$\log (Z/Z_\odot)$',
                        'prior_function': tophat,
                        'prior_args': {'mini':-1.98, 'maxi':0.19}})
                        
###### SFH   ########
model_params.append({'name': 'sfh', 'N':1,
                        'isfree': False,
                        'init': 0,
                        'units': None})

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

model_params.append({'name': 'agebins', 'N': 1,
                        'isfree': False,
                        'init': [],
                        'units': 'log(yr)',
                        'prior_function': priors.tophat,
                        'prior_args':{'mini':0.1, 'maxi':15.0}})

model_params.append({'name': 'sfr_fraction', 'N': 1,
                        'isfree': True,
                        'init': [],
                        'units': 'Msun',
                        'prior_function': priors.tophat,
                        'prior_args':{'mini':0.0, 'maxi':1.0}})

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
                        'disp_floor': 0.5,
                        'units': '',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.0, 'maxi':4.0}})

model_params.append({'name': 'dust2', 'N': 1,
                        'isfree': True,
                        'init': 1.0,
                        'init_disp': 0.25,
                        'disp_floor': 0.15,
                        'units': '',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.0,'maxi':4.0}})

model_params.append({'name': 'dust_index', 'N': 1,
                        'isfree': True,
                        'init': 0.0,
                        'init_disp': 0.25,
                        'disp_floor': 0.15,
                        'units': '',
                        'prior_function': tophat,
                        'prior_args': {'mini':-2.2, 'maxi': 0.4}})

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
                        'disp_floor': 0.15,
                        'units': None,
                        'prior_function': tophat,
                        'prior_args': {'mini':0.0, 'maxi':1.0}})

model_params.append({'name': 'duste_umin', 'N': 1,
                        'isfree': True,
                        'init': 1.0,
                        'init_disp': 5.0,
                        'disp_floor': 4.5,
                        'units': None,
                        'prior_function': tophat,
                        'prior_args': {'mini':0.1, 'maxi':25.0}})

model_params.append({'name': 'duste_qpah', 'N': 1,
                        'isfree': True,
                        'init': 3.0,
                        'init_disp': 3.0,
                        'disp_floor': 3.0,
                        'units': 'percent',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.0, 'maxi':10.0}})

###### Nebular Emission ###########
model_params.append({'name': 'add_neb_emission', 'N': 1,
                        'isfree': False,
                        'init': True,
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
                        'depends_on': tie_gas_logz,
                        'units': r'log Z/Z_\odot',
                        'prior_function': tophat,
                        'prior_args': {'mini':-2.0, 'maxi':0.5}})

model_params.append({'name': 'gas_logu', 'N': 1,
                        'isfree': True,
                        'init': -2.0,
                        'units': '',
                        'prior_function': tophat,
                        'prior_args': {'mini':-4, 'maxi':-1}})

##### AGN dust ##############
model_params.append({'name': 'add_agn_dust', 'N': 1,
                        'isfree': False,
                        'init': True,
                        'units': '',
                        'prior_function_name': None,
                        'prior_args': None})

model_params.append({'name': 'fagn', 'N': 1,
                        'isfree': True,
                        'init': 0.1,
                        'init_disp': 0.2,
                        'disp_floor': 0.1,
                        'units': '',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.0, 'maxi':5.0}})

model_params.append({'name': 'agn_tau', 'N': 1,
                        'isfree': True,
                        'init': 4.0,
                        'init_disp': 5,
                        'disp_floor': 2,
                        'units': '',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.0, 'maxi':40.0}})

####### Calibration ##########
model_params.append({'name': 'phot_jitter', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
                        'init_disp': 0.5,
                        'units': 'fractional maggies (mags/1.086)',
                        'prior_function':tophat,
                        'prior_args': {'mini':0.0, 'maxi':0.5}})

####### Units ##########
model_params.append({'name': 'peraa', 'N': 1,
                     'isfree': False,
                     'init': False})

model_params.append({'name': 'mass_units', 'N': 1,
                     'isfree': False,
                     'init': 'mstar'})

#### resort list of parameters 
#### so that major ones are fit first
parnames = [m['name'] for m in model_params]
fit_order = ['logmass','sfr_fraction','dust2', 'logzsol', 'dust_index', 'dust1', 'duste_qpah', 'duste_gamma', 'duste_umin']
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

        # implement uniqueness of outliers
        if 'gp_outlier_locs' in self.theta_index:
            start,end = self.theta_index['gp_outlier_locs']
            outlier_locs = theta[start:end]
            if len(np.unique(np.round(outlier_locs))) != len(outlier_locs):
                return -np.inf

        # dust1/dust2 ratio
        if 'dust1' in self.theta_index:
            if 'dust2' in self.theta_index:
                start,end = self.theta_index['dust1']
                dust1 = theta[start:end]
                start,end = self.theta_index['dust2']
                dust2 = theta[start:end]
                if dust1/1.5 > dust2:
                    return -np.inf
                '''
                if dust1 < 0.5*dust2:
                    return -np.inf
                '''

        # sum of SFH fractional bins <= 1.0
        if 'sfr_fraction' in self.theta_index:
            start,end = self.theta_index['sfr_fraction']
            sfr_fraction = theta[start:end]
            if np.sum(sfr_fraction) > 1.0:
                return -np.inf

        for k, v in self.theta_index.iteritems():
            start, end = v
            this_prior = np.sum(self._config_dict[k]['prior_function']
                                (theta[start:end], **self._config_dict[k]['prior_args']))

            if (not np.isfinite(this_prior)):
                print('WARNING: ' + k + ' is out of bounds')
            lnp_prior += this_prior
        return lnp_prior

class FracSFH(FastStepBasis):
    
    def get_galaxy_spectrum(self, **params):
        self.update(**params)

        #### here's the custom fractional stuff
        fractions = np.array(self.params['sfr_fraction'])
        bin_fractions = np.append(fractions,(1-np.sum(fractions)))
        time_per_bin = []
        for (t1, t2) in self.params['agebins']: time_per_bin.append(10**t2-10**t1)
        bin_fractions *= np.array(time_per_bin)
        bin_fractions /= bin_fractions.sum()
        
        mass = bin_fractions*self.params['mass']
        mtot = self.params['mass'].sum()

        time, sfr, tmax = self.convert_sfh(self.params['agebins'], mass)
        self.ssp.params["sfh"] = 3 #Hack to avoid rewriting the superclass
        self.ssp.set_tabular_sfh(time, sfr)
        wave, spec = self.ssp.get_spectrum(tage=tmax, peraa=False)

        return wave, spec / mtot, self.ssp.stellar_mass / mtot

def load_sps(**extras):

    sps = FracSFH(**extras)
    return sps

def load_model(objname='',datname='', agelims=[], **extras):

    ###### REDSHIFT ######
    with open(datname, 'r') as f:
        hdr = f.readline()

    hdr = hdr[1:-2].split(',')
    dtype = np.dtype([(hdr[0],'S40')] + [(n, np.float) for n in hdr[1:]])
    dat = np.loadtxt(datname, comments = '#', delimiter=',', dtype = dtype)
    names = np.array([s.strip() for s in dat['Name']])
    obj_ind = np.where(names == objname)[0][0]
    zred = dat['z'][obj_ind]

    #### CALCULATE TUNIV #####
    tuniv = WMAP9.age(zred).value

    #### NONPARAMETRIC SFH ######
    agelims[-1] = np.log10(tuniv*1e9)
    agebins = np.array([agelims[:-1], agelims[1:]])
    ncomp = len(agelims) - 1

    #### ADJUST MODEL PARAMETERS #####
    n = [p['name'] for p in model_params]

    #### SET UP AGEBINS
    model_params[n.index('agebins')]['N'] = ncomp
    model_params[n.index('agebins')]['init'] = agebins.T

    #### FRACTIONAL MASS INITIALIZATION
    # N-1 bins, last is set by x = 1 - np.sum(sfr_fraction)
    model_params[n.index('sfr_fraction')]['N'] = ncomp-1
    model_params[n.index('sfr_fraction')]['prior_args'] = {
                                                           'maxi':np.full(ncomp-1,1.0), 
                                                           'mini':np.full(ncomp-1,0.0),
                                                           # NOTE: ncomp instead of ncomp-1 makes the prior take into account the implicit Nth variable too
                                                          }
    model_params[n.index('sfr_fraction')]['init'] =  np.zeros(ncomp-1)+1./ncomp
    model_params[n.index('sfr_fraction')]['init_disp'] = 0.02

    #### INSERT REDSHIFT INTO MODEL PARAMETER DICTIONARY ####
    zind = n.index('zred')
    model_params[zind]['init'] = zred

    #### CREATE MODEL
    model = BurstyModel(model_params)

    return model

model_type = BurstyModel


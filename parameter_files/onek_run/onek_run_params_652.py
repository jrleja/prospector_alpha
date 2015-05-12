import numpy as np
import fsps,os,threed_dutils
from sedpy import attenuation
from bsfh import priors, sedmodel, elines
from astropy.cosmology import WMAP9
import bsfh.datautils as dutils
tophat = priors.tophat
logarithmic = priors.logarithmic

#############
# RUN_PARAMS
#############

run_params = {'verbose':True,
              'outfile':os.getenv('APPS')+'/threedhst_bsfh/results/onek_run/onek_run',
              'ftol':0.5e-5, 
              'maxfev':5000,
              'nwalkers':496,
              'nburn':[32,32,256,80], 
              'niter': 1800,
              'initial_disp':0.1,
              'edge_trunc':0.3,
              'debug': False,
              'mock': False,
              'logify_spectrum': False,
              'normalize_spectrum': False,
              'set_init_params': None,  # DO NOT SET THIS TO TRUE SINCE TAGE == TUNIV*1.2 (fast,random)
              'min_error': 0.02,
              'abs_error': False,
              'spec': False, 
              'phot':True,
              'photname':os.getenv('APPS')+'/threedhst_bsfh/data/COSMOS_onek.cat',
              'truename':os.getenv('APPS')+'/threedhst_bsfh/data/COSMOS_onek.dat',
              'objname':'498',
              }

############
# OBS
#############

obs = threed_dutils.load_obs_3dhst(run_params['photname'], run_params['objname'],
                                   min_error=run_params['min_error'],
                                   abs_error=run_params['abs_error'])

#############
# MODEL_PARAMS
#############

class BurstyModel(sedmodel.CSPModel):
	
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
        
        # check to make sure tau models are separated
        if 'tau' in self.theta_index:
            start,end = self.theta_index['tau']
            tau = theta[start:end]
            if (tau[0] < 2*tau[1]):
                return -np.inf

        # implement mass ratio
        if 'mass' in self.theta_index:
            start,end = self.theta_index['mass']
            mass = theta[start:end]
            if (mass[1]/mass[0] > 20):
                return -np.inf

        # implement uniqueness of outliers
        if 'gp_outlier_locs' in self.theta_index:
            start,end = self.theta_index['gp_outlier_locs']
            outlier_locs = theta[start:end]
            if len(np.unique(np.round(outlier_locs))) != len(outlier_locs):
                return -np.inf

        for k, v in self.theta_index.iteritems():
            start, end = v
            lnp_prior += np.sum(self._config_dict[k]['prior_function']
                                (theta[start:end], **self._config_dict[k]['prior_args']))
        
        return lnp_prior

model_type = BurstyModel
model_params = []

param_template = {'name':'', 'N':1, 'isfree': False,
                  'init':0.0, 'units':'', 'label':'',
                  'prior_function_name': None, 'prior_args': None}

###### BASIC PARAMETERS ##########
model_params.append({'name': 'zred', 'N': 1,
                        'isfree': False,
                        'init': 1.0,
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
                        
model_params.append({'name': 'mass', 'N': 2,
                        'isfree': True,
                        'init': np.array([1e10, 1e9]),
                        'units': r'M_\odot',
                        'prior_function': tophat,
                        'prior_args': {'mini':np.array([1e7, 1e7]),
                                       'maxi':np.array([1e14, 1e14])}})

model_params.append({'name': 'pmetals', 'N': 1,
                        'isfree': False,
                        'init': -99,
                        'units': '',
                        'prior_function': None,
                        'prior_args': {'mini':-3, 'maxi':-1}})

model_params.append({'name': 'logzsol', 'N': 1,
                        'isfree': True,
                        'init': -0.1,
                        'init_disp': 0.2,
                        'log_param': True,
                        'units': r'$\log (Z/Z_\odot)$',
                        'prior_function': tophat,
                        'prior_args': {'mini':-1, 'maxi':0.19}})
                        
###### SFH   ########
model_params.append({'name': 'sfh', 'N': 1,
                        'isfree': False,
                        'init': 4,
                        'units': 'type',
                        'prior_function_name': None,
                        'prior_args': None})

model_params.append({'name': 'tau', 'N': 2,
                        'isfree': True,
                        'init': np.array([10.0, 1.0]),
                        'init_disp': 0.3,
                        'units': 'Gyr',
                        'prior_function':logarithmic,
                        'prior_args': {'mini':np.array([0.1, 0.1]),
                                       'maxi':np.array([100.0, 100.0])}})

model_params.append({'name': 'tage', 'N': 1,
                        'isfree': False,
                        'init': 14.0,
                        'units': 'Gyr',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.1, 'maxi':14.0}})

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

model_params.append({'name': 'sf_start', 'N': 2,
                        'isfree': True,
                        'reinit': True,
                        'init_disp': 0.3,
                        'init': np.array([1.0, 1.0]),
                        'units': 'Gyr',
                        'prior_function': tophat,
                        'prior_args': {'mini':np.array([0.0, 0.0]),
                                       'maxi':np.array([14.0, 14.0])}})

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
                        'init': 0,
                        'units': 'index',
                        'prior_function_name': None,
                        'prior_args': None})
                        
model_params.append({'name': 'dust1', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
                        'init_disp': 0.5,
                        'units': '',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.0, 'maxi':3.0}})

model_params.append({'name': 'dust2', 'N': 2,
                        'isfree': True,
                        'init': np.array([0.35,0.35]),
                        'reinit': True,
                        'init_disp': 0.3,
                        'units': '',
                        'prior_function': tophat,
                        'prior_args': {'mini':np.array([0.0, 0.0]),
                                       'maxi':np.array([4.0, 4.0])}})

model_params.append({'name': 'dust_index', 'N': 1,
                        'isfree': True,
                        'init': -0.7,
                        'reinit': True,
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
                        'isfree': False,
                        'init': 0.01,
                        'units': None,
                        'prior_function': tophat,
                        'prior_args': {'mini':0.0, 'maxi':1.0}})

model_params.append({'name': 'duste_umin', 'N': 1,
                        'isfree': False,
                        'init': 1.0,
                        'units': None,
                        'prior_function': tophat,
                        'prior_args': {'mini':0.1, 'maxi':25.0}})

model_params.append({'name': 'duste_qpah', 'N': 1,
                        'isfree': False,
                        'init': 3.0,
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
                        'nuisance': 2,
                        'units': 'fractional maggies (mags/1.086)',
                        'prior_function':tophat,
                        'prior_args': {'mini':0.0, 'maxi':0.5}})

# Here we define groups of filters to which we will add additional
# uncertainty above and beyond the stated uncertainty and the
# additional jitter.
gp_filts = np.array([['u_cosmos', 'ia427_cosmos', 'b_cosmos', 'ia464_cosmos',\
                      'ia484_cosmos', 'g_cosmos', 'ia505_cosmos', 'ia527_cosmos',\
                      'v_cosmos', 'ia574_cosmos', 'ia624_cosmos',\
                      'r_cosmos', 'rp_cosmos', 'ia679_cosmos', 'ia709_cosmos',\
                      'ia738_cosmos', 'ip_cosmos', 'i_cosmos', 'ia767_cosmos',\
                      'ia827_cosmos', 'z_cosmos', 'zp_cosmos',\
                      'uvista_y_cosmos', 'j1_cosmos', 'j2_cosmos',\
                      'uvista_j_cosmos', 'j_cosmos', 'j3_cosmos',\
                      'h1_cosmos', 'h_cosmos', 'uvista_h_cosmos',\
                      'h2_cosmos', 'uvista_ks_cosmos', 'ks_cosmos', 'k_cosmos',\
                      ],\
                     ['irac1_cosmos','irac2_cosmos','irac3_cosmos','irac4_cosmos'],\
                     ['f606w_cosmos','f814w_cosmos','f125w_cosmos','f140w_cosmos','f160w_cosmos','mips_24um_cosmos']])
ngpf = gp_filts.shape[0]

model_params.append({'name': 'gp_filter_amps','N': ngpf,
                        'isfree': True,
                        'init': np.zeros(ngpf),
                        'init_disp': 0.5,
                        'nuisance': 1,
                        'reinit': True,
                        'units': 'fractional maggies (mags/1.086)',
                        'prior_function':tophat,
                        'prior_args': {'mini':np.zeros(ngpf), 'maxi':np.zeros(ngpf)+0.4}})

model_params.append({'name': 'gp_filter_locs','N': ngpf,
                        'isfree': False,
                        'init': gp_filts,
                        'init_disp': 0.5,
                        'units': 'filter names or filter_indices',
                        'prior_function':None,
                        'prior_args': None})

##### OUTLIERS #####
noutliers=3
model_params.append({'name': 'gp_outlier_amps','N': noutliers,
                        'isfree': False,
                        'init': np.zeros(noutliers)+100.0,
                        'init_disp': 0.5,
                        'reinit': True,
                        'units': 'fractional maggies (mags/1.086)',
                        'prior_function':tophat,
                        'prior_args': {'mini':np.zeros(noutliers), 'maxi':np.zeros(noutliers)+100.0}})

model_params.append({'name': 'gp_outlier_locs','N': noutliers,
                        'isfree': True,
                        'init': np.linspace(noutliers,noutliers**2,noutliers),
                        'init_disp': 0.5,
                        'nuisance': 1,
                        'units': 'filter_indices',
                        'prior_function':tophat,
                        'prior_args': {'mini':np.zeros(noutliers), 'maxi': np.zeros(noutliers)+np.sum(obs['phot_mask'])-1}})

# name outfile
run_params['outfile'] = run_params['outfile']+'_'+run_params['objname']

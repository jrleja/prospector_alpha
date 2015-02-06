import numpy as np
import fsps,os,threed_dutils
from sedpy import attenuation
from bsfh import priors, sedmodel, elines
from astropy.cosmology import WMAP9
import bsfh.datautils as dutils
tophat = priors.tophat

#############
# RUN_PARAMS
#############

run_params = {'verbose':True,
              'outfile':os.getenv('APPS')+'/threedhst_bsfh/results/dtau_neboff/dtau_neboff',
              'ftol':0.5e-5, 
              'maxfev':5000,
              'nwalkers':248,
              'nburn':[32,64,128], 
              'niter': 2048,
              'initial_disp':0.1,
              'debug': False,
              'mock': False,
              'logify_spectrum': False,
              'normalize_spectrum': False,
              'set_init_params': None,  # DO NOT SET THIS TO TRUE SINCE TAGE == TUNIV*1.2 (fast,random)
              'min_error': 0.02,
              'abs_error': False,
              'spec': False, 
              'phot':True,
              'photname':os.getenv('APPS')+'/threedhst_bsfh/data/COSMOS_testsamp.cat',
              'fastname':os.getenv('APPS')+'/threedhst_bsfh/data/COSMOS_testsamp.fout',
              'ancilname':os.getenv('APPS')+'/threedhst_bsfh/data/COSMOS_testsamp.dat',
              'mipsname':os.getenv('APPS')+'/threedhst_bsfh/data/MIPS/cosmos_3dhst.v4.1.4.sfr',
              'objname':'16718',
              }

############
# OBS
#############

obs = threed_dutils.load_obs_3dhst(run_params['photname'], run_params['objname'],
									mips=run_params['mipsname'], min_error=run_params['min_error'],
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
                        'init': 0,
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

model_params.append({'name': 'logzsol', 'N': 1,
                        'isfree': True,
                        'init': 0.0,
                        'init_disp': 0.75,
                        'reinit': True,
                        'units': r'$\log (Z/Z_\odot)$',
                        'prior_function': tophat,
                        'prior_args': {'mini':-1, 'maxi':0.19}})

model_params.append({'name': 'pmetals', 'N': 1,
                        'isfree': False,
                        'init': 2,
                        'units': '',
                        'prior_function': None,
                        'prior_args': {'mini':-3, 'maxi':-1}})
                        
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
                        'units': 'Gyr',
                        'prior_function':tophat,
                        'prior_args': {'mini':np.array([0.1, 0.1]),
                                       'maxi':np.array([100.0, 100.0])}})

model_params.append({'name': 'tage', 'N': 1,
                        'isfree': False,
                        'init': 1.0,
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
                        'init': np.array([0.0,0.0]),
                        'reinit': True,
                        'init_disp': 0.2,
                        'units': '',
                        'prior_function': tophat,
                        'prior_args': {'mini':np.array([0.0, 0.0]),
                                       'maxi':np.array([4.0, 4.0])}})

model_params.append({'name': 'dust_index', 'N': 1,
                        'isfree': True,
                        'init': -0.7,
                        'reinit': True,
                        'units': '',
                        'prior_function': tophat,
                        'prior_args': {'mini':-3.0, 'maxi': -0.4}})

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
                        'reinit': True,
                        'init': 3.0,
                        'init_disp': 0.5,
                        'units': 'percent',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.0, 'maxi':10.0}})

###### Nebular Emission ###########
model_params.append({'name': 'add_neb_emission', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
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

####### SET INITIAL PARAMETERS ##########
fastvalues,fastfields = threed_dutils.load_fast_3dhst(run_params['fastname'],
                                                      run_params['objname'])
parmlist = [p['name'] for p in model_params]

if run_params['set_init_params'] == 'fast':

	# translate
	fparams = {}
	translate = {#'zred': ('z', lambda x: x),
                 'tau':  ('ltau', lambda x: (10**x)/1e9),
                 #'tage': ('lage', lambda x:  (10**x)/1e9),
                 'dust2':('Av', lambda x: x),
                 'mass': ('lmass', lambda x: (10**x))}
	for k, v in translate.iteritems():		
		fparams[k] = v[1](fastvalues[np.array(fastfields) == v[0]][0])

	for par in model_params:
		if (par['name'] in fparams):
			par['init'] = fparams[par['name']]
if run_params['set_init_params'] == 'random':
	import random
	for ii in xrange(len(model_params)):
		if model_params[ii]['isfree'] == True:
			max = model_params[ii]['prior_args']['maxi']
			min = model_params[ii]['prior_args']['mini']
			model_params[ii]['init'] = random.random()*(max-min)+min

######## LOAD ANCILLARY INFORMATION ########
# name outfiles based on halpha eqw
ancildat = threed_dutils.load_ancil_data(run_params['ancilname'],run_params['objname'])
halpha_eqw_txt = "%04d" % int(ancildat['Ha_EQW_obs'])
run_params['outfile'] = run_params['outfile']+'_'+halpha_eqw_txt+'_'+run_params['objname']

# use zbest, not whatever's in the fast run
zbest = ancildat['z']
model_params[parmlist.index('zred')]['init'] = zbest
			
####### RESET AGE PRIORS TO MATCH AGE OF UNIVERSE ##########
tuniv = WMAP9.age(model_params[0]['init']).value

# set tage
# set max on sf_start
model_params[parmlist.index('tage')]['init'] = tuniv
model_params[parmlist.index('sf_start')]['prior_args']['maxi'] = 0.9*tuniv

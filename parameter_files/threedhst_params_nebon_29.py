import numpy as np
import fsps,os
from sedpy import attenuation
from bsfh import priors, sedmodel, elines
from astropy.cosmology import WMAP9
import bsfh.datautils as dutils
tophat = priors.tophat

#############
# RUN_PARAMS
#############

run_params = {'verbose':True,
              'outfile':os.getenv('APPS')+'/threedhst_bsfh/results/threedhst_nebon',
              'ftol':0.5e-5, 
              'maxfev':5000,
              'nwalkers':124,
              'nburn':[32,64,64], 
              'niter': 4096,
              'initial_disp':0.1,
              'debug': False,
              'mock': False,
              'logify_spectrum': False,
              'normalize_spectrum': False,
              'fast_init_params': False,  # DO NOT SET THIS TO TRUE SINCE TAGE == TUNIV*1.2
              'min_error': 0.02,
              'spec': False, 
              'phot':True,
              'photname':os.getenv('APPS')+'/threedhst_bsfh/data/COSMOS_testsamp.cat',
              'fastname':os.getenv('APPS')+'/threedhst_bsfh/data/COSMOS_testsamp.fout',
              'objname':'235',
              }
run_params['outfile'] = run_params['outfile']+'_'+run_params['objname']

def return_mwave_custom(filters):

	"""
	returns effective wavelength based on filter names
	"""

	loc = os.getenv('APPS')+'/threedhst_bsfh/filters/'
	key_str = 'filter_keys_threedhst.txt'
	lameff_str = 'lameff_threedhst.txt'
	
	lameff = np.loadtxt(loc+lameff_str)
	keys = np.loadtxt(loc+key_str, dtype='S20',usecols=[1])
	keys = keys.tolist()
	keys = np.array([keys.lower() for keys in keys], dtype='S20')
	
	lameff_return = [[lameff[keys == filters[i]]][0] for i in range(len(filters))]
	lameff_return = [item for sublist in lameff_return for item in sublist]
	
	return lameff_return

def load_obs_3dhst(filename, objnum, min_error = None):
	"""
	Load 3D-HST photometry file, return photometry for a particular object.
	min_error: set the minimum photometric uncertainty to be some fraction
	of the flux. if not set, use default errors.
	"""
	obs ={}
	fieldname=filename.split('/')[-1].split('_')[0].upper()
	with open(filename, 'r') as f:
		hdr = f.readline().split()
	dat = np.loadtxt(filename, comments = '#',
					 dtype = np.dtype([(n, np.float) for n in hdr[1:]]))
	obj_ind = np.where(dat['id'] == int(objnum))[0][0]
	
	# extract fluxes+uncertainties for all objects and all filters
	flux_fields = [f for f in dat.dtype.names if f[0:2] == 'f_']
	unc_fields = [f for f in dat.dtype.names if f[0:2] == 'e_']
	filters = [f[2:] for f in flux_fields]

	# extract fluxes for particular object, converting from record array to numpy array
	flux = dat[flux_fields].view(float).reshape(len(dat),-1)[obj_ind]
	unc  = dat[unc_fields].view(float).reshape(len(dat),-1)[obj_ind]

	# define all outputs
	filters = [filter.lower()+'_'+fieldname.lower() for filter in filters]
	wave_effective = np.array(return_mwave_custom(filters))
	phot_mask = np.logical_or((flux != unc),(flux > 0))
	maggies = flux/(10**10)
	maggies_unc = unc/(10**10)
	
	# set minimum photometric error
	if min_error is not None:
		maggies_unc[maggies_unc < min_error*maggies] = min_error*maggies[maggies_unc < min_error*maggies]
	
	# sort outputs based on effective wavelength
	points = zip(wave_effective,filters,phot_mask,maggies,maggies_unc)
	sorted_points = sorted(points)

	# build output dictionary
	obs['wave_effective'] = np.array([point[0] for point in sorted_points])
	obs['filters'] = np.array([point[1] for point in sorted_points])
	obs['phot_mask'] =  np.array([point[2] for point in sorted_points])
	obs['maggies'] = np.array([point[3] for point in sorted_points])
	obs['maggies_unc'] =  np.array([point[4] for point in sorted_points])
	obs['wavelength'] = None
	obs['spectrum'] = None

	return obs

def load_fast_3dhst(filename, objnum):
	"""
	Load FAST output for a particular object
	Returns a dictionary of inputs for BSFH
	"""

	# filter through header junk, load data
	fieldname=filename.split('/')[-1].split('_')[0].upper()
	with open(filename, 'r') as f:
		for jj in range(1): hdr = f.readline().split()
	dat = np.loadtxt(filename, comments = '#',dtype = np.dtype([(n, np.float) for n in hdr[1:]]))

	# extract field names, search for ID, pull out object info
	fields = [f for f in dat.dtype.names]
	id_ind = fields.index('id')
	obj_ind = [int(x[id_ind]) for x in dat].index(int(objnum))
	values = dat[fields].view(float).reshape(len(dat),-1)[obj_ind]

	# translate
	output = {}
	translate = {'zred': ('z', lambda x: x),
                 'tau':  ('ltau', lambda x: (10**x)/1e9),
                 'tage': ('lage', lambda x:  (10**x)/1e9),
                 'dust2':('Av', lambda x: x),
                 'mass': ('lmass', lambda x: (10**x))}
	for k, v in translate.iteritems():		
		output[k] = v[1](values[np.array(fields) == v[0]][0])
	return output

############
# OBS
#############

obs = load_obs_3dhst(run_params['photname'], run_params['objname'], min_error=run_params['min_error'])

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
        
		# check to make sure tburst < tage
		if 'tage' in self.theta_index:
			start,end = self.theta_index['tage']
			tage = theta[start:end]
			if 'tburst' in self.theta_index:
				start,end = self.theta_index['tburst']
				tburst = theta[start:end]
				if tburst > tage:
					return -np.inf
        
		for k, v in self.theta_index.iteritems():
			start, end = v
			#print(k)
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

model_params.append({'name': 'add_dust_emission', 'N': 1,
                        'isfree': False,
                        'init': 0,
                        'units': None,
                        'prior_function': None,
                        'prior_args': None})

model_params.append({'name': 'add_agb_dust_model', 'N': 1,
                        'isfree': False,
                        'init': 0,
                        'units': None,
                        'prior_function': None,
                        'prior_args': None})
                        
model_params.append({'name': 'mass', 'N': 1,
                        'isfree': True,
                        'init': 1e10,
                        'units': r'M_\odot',
                        'prior_function': tophat,
                        'prior_args': {'mini':1e6, 'maxi':1e12}})

model_params.append({'name': 'logzsol', 'N': 1,
                        'isfree': True,
                        'init': 0,
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

model_params.append({'name': 'tau', 'N': 1,
                        'isfree': True,
                        'init': 1.0,
                        'units': 'Gyr',
                        'prior_function':tophat,
                        'prior_args': {'mini':0.1, 'maxi':100}})

model_params.append({'name': 'tage', 'N': 1,
                        'isfree': False,
                        'init': 1.0,
                        'units': 'Gyr',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.1, 'maxi':10.0}})

model_params.append({'name': 'tburst', 'N': 1,
                        'isfree': True,
                        'init': 0.0,
                        'units': '',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.0, 'maxi':10.0}})

model_params.append({'name': 'fburst', 'N': 1,
                        'isfree': True,
                        'init': 0.0,
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
                        'isfree': True,
                        'init': 0.0,
                        'units': 'Gyr',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.0, 'maxi':0.5}})
                        
########    IMF  ##############
model_params.append({'name': 'imf_type', 'N': 1,
                        	 'isfree': False,
                             'init': 1, #1 = chabrier
                       		 'units': None,
                       		 'prior_function_name': None,
                        	 'prior_args': None})

########    Dust ##############
model_params.append({'name': 'dust_type', 'N': 1,
                        'isfree': False,
                        'init': 0,
                        'units': 'index',
                        'prior_function_name': None,
                        'prior_args': None})
                        
model_params.append({'name': 'dust1', 'N': 1,
                        'isfree': True,
                        'init': 0.0,
                        'units': '',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.0, 'maxi':3.0}})

model_params.append({'name': 'dust2', 'N': 1,
                        'isfree': True,
                        'init': 0.35,
                        'units': '',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.0, 'maxi':3.0}})

model_params.append({'name': 'dust_index', 'N': 1,
                        'isfree': True,
                        'init': -0.7,
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
fparams = load_fast_3dhst(run_params['fastname'],
                          run_params['objname'])
parmlist = [p['name'] for p in model_params]
model_params[parmlist.index('zred')]['init'] = fparams['zred']
if run_params['fast_init_params']:
	for par in model_params:
		if (par['name'] in fparams):
			par['init'] = fparams[par['name']]
else:
	import random
	for ii in xrange(len(model_params)):
		if model_params[ii]['isfree'] == True:
			max = model_params[ii]['prior_args']['maxi']
			min = model_params[ii]['prior_args']['mini']
			model_params[ii]['init'] = random.random()*(max-min)+min
			
####### RESET AGE PRIORS TO MATCH AGE OF UNIVERSE ##########
tuniv = WMAP9.age(model_params[0]['init']).value

model_params[parmlist.index('tage')]['init'] = 1.2*tuniv
model_params[parmlist.index('sf_start')]['prior_args']['maxi'] = 0.5*tuniv
model_params[parmlist.index('tburst')]['prior_args']['mini'] = 1.2*tuniv-1
model_params[parmlist.index('tburst')]['prior_args']['maxi'] = 1.2*tuniv
model_params[parmlist.index('tburst')]['init'] = 1.2*tuniv-0.5

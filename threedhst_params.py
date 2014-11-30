import numpy as np
import fsps
from sedpy import attenuation
from bsfh import priors, sedmodel, elines
import bsfh.datautils as dutils
tophat = priors.tophat

#############
# RUN_PARAMS
#############

run_params = {'verbose':True,
              'outfile':'/Users/joel/code/python/threedhst_bsfh/results/threedhst',
              'ftol':0.5e-5, 'maxfev':5000,
              'nwalkers':128,
              'nburn':[32, 32, 64], 'niter':128,
              'initial_disp':0.1,
              'debug': False,
              'mock': False,
              'logify_spectrum': False,
              'normalize_spectrum': False,
              'spec': False, 'phot':True,
              'photname':'/Users/joel/code/python/threedhst_bsfh/data/cosmos_3dhst.v4.1.test.cat',
              'fastname':'/Users/joel/code/python/threedhst_bsfh/data/cosmos_3dhst.v4.1.test.fout',
              'objname':'32206',
              'wlo':3750., 'whi':7200.
              }

def return_mwave_custom(filters):

	loc = '/Users/joel/code/python/threedhst_bsfh/filters/'
	key_str = 'filter_keys_threedhst.txt'
	lameff_str = 'lameff_threedhst.txt'
	
	lameff = np.loadtxt(loc+lameff_str)
	keys = np.loadtxt(loc+key_str, dtype='S20',usecols=[1])
	keys = keys.tolist()
	keys = np.array([keys.lower() for keys in keys], dtype='S20')
	
	lameff_return = [[lameff[keys == filters[i]]][0] for i in range(len(filters))]
	lameff_return = [item for sublist in lameff_return for item in sublist]
	
	return lameff_return

def load_obs_3dhst(filename, objnum):
	"""Load a 3D-HST data file and choose a particular object.
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
	"""Load a 3D-HST data file and choose a particular object.
	"""

	fieldname=filename.split('/')[-1].split('_')[0].upper()
	with open(filename, 'r') as f:
		hdr = f.readline().split()
	dat = np.loadtxt(filename, comments = '#',
					 dtype = np.dtype([(n, np.float) for n in hdr[1:]]))
	obj_ind = np.where(dat['id'] == int(objnum))[0][0]
	
	# extract values and field names
	fields = [f for f in dat.dtype.names]
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

obs = load_obs_3dhst(run_params['photname'], run_params['objname'])

#############
# MODEL_PARAMS
#############
model_type = sedmodel.CSPModel
model_params = []

param_template = {'name':'', 'N':1, 'isfree': False,
                  'init':0.0, 'units':'', 'label':'',
                  'prior_function_name': None, 'prior_args': None}

###### BASIC PARAMETERS ##########
model_params.append({'name': 'zred', 'N': 1,
                        'isfree': False,
                        'init': 0.91,
                        'units': '',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.0, 'maxi':4.0}})

model_params.append({'name': 'add_igm_absorption', 'N': 1,
                        'isfree': False,
                        'init': 1,
                        'units': None,
                        'prior_function': None,
                        'prior_args': None})
                        
model_params.append({'name': 'mass', 'N': 1,
                        'isfree': True,
                        'init': 1e10,
                        'units': r'M_\odot',
                        'prior_function': tophat,
                        'prior_args': {'mini':1e9, 'maxi':1e12}})

model_params.append({'name': 'logzsol', 'N': 1,
                        'isfree': False,
                        'init': 0,
                        'units': r'$\log (Z/Z_\odot)$',
                        'prior_function': tophat,
                        'prior_args': {'mini':-1, 'maxi':0.19}})
                        
###### SFH   ########

model_params.append({'name': 'sfh', 'N': 1,
                        'isfree': False,
                        'init': 1,
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
                        'isfree': True,
                        'init': 1.0,
                        'units': 'Gyr',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.1, 'maxi':10.0}})

model_params.append({'name': 'tburst', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
                        'units': '',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.0, 'maxi':1.3}})

model_params.append({'name': 'fburst', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
                        'units': '',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.0, 'maxi':0.5}})

model_params.append({'name': 'sf_start', 'N': 1,
                        'isfree': False,
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
                        'init': 2,
                        'units': 'index',
                        'prior_function_name': None,
                        'prior_args': None})
model_params.append({'name': 'dust1', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
                        'units': '',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.1, 'maxi':2.0}})

model_params.append({'name': 'dust2', 'N': 1,
                        'isfree': True,
                        'init': 0.35,
                        'units': '',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.0, 'maxi':1.0}})

model_params.append({'name': 'dust_index', 'N': 1,
                        'isfree': False,
                        'init': -0.7,
                        'units': '',
                        'prior_function': tophat,
                        'prior_args': {'mini':-1.5, 'maxi':-0.5}})

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

####### FAST PARAMS ##########
fast_params = True
if fast_params == True:
	fparams = load_fast_3dhst(run_params['fastname'],
                              run_params['objname'])
	for par in model_params:
		if (par['name'] in fparams):
			par['init'] = fparams[par['name']]
	
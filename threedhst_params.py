import numpy as np
import fsps
from sedpy import attenuation
from bsfh import priors, sedmodel, elines
import bsfh.datautils as dutils
tophat = priors.tophat

def load_obs_3dhst(filename, objnum):
	"""Load a 3D-HST data file and choose a particular object.
	"""
	obs ={}
	fieldname=filename.split('/')[-1].split('_')[0].upper()
	with open(filename, 'r') as f:
		hdr = f.readline().split()
	dat = np.loadtxt(filename, comments = '#',
					 dtype = np.dtype([(n, np.float) for n in hdr[1:]]))
	
	flux_fields = [f for f in dat.dtype.names if f[0:2] == 'f_']
	unc_fields = [f for f in dat.dtype.names if f[0:2] == 'e_']
	filters = [f[2:] for f in flux_fields]
	
	# extract fluxes from record array
	flux = dat[flux_fields].view(float).reshape(len(dat),-1)
	unc  = dat[unc_fields].view(float).reshape(len(dat),-1)
    
	obs['filters'] = [flux_field+'_'+fieldname for flux_field in flux_fields]
	obs['phot_mask'] =  np.logical_or((flux != unc),(flux > 0))
	obs['maggies'] = flux/(10**10)
	obs['maggies_unc'] =  flux/(10**10)

	obs['wavelength'] = None

	return obs

#############
# RUN_PARAMS
#############

run_params = {'verbose':True,
              'outfile':'results/mock_3dhst',
              'ftol':0.5e-5, 'maxfev':500,
              'nwalkers':32,
              'nburn':[32, 64, 128], 'niter':256,
              'initial_disp':0.1,
              'mock': False,
              'spec': False, 'phot':True,
              'logify_spectrum':True,
              'normalize_spectrum':True,
              'photname':'/Users/joel/code/python/threedhst_bsfh/data/cosmos_3dhst.v4.1.test.cat',
              'objname':'32206',
              'wlo':3750., 'whi':7200.
              }

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

###### Distance ##########
model_params.append({'name': 'zred', 'N': 1,
                        'isfree': False,
                        'init': 0.7,
                        'units': '',
                        'prior_function_name': 'tophat',
                        'prior_args': {'mini':0.3, 'maxi':1.3}})

###### SFH   ########

model_params.append({'name': 'sfh', 'N': 1,
                        'isfree': False,
                        'init': 4,
                        'units': 'type',
                        'prior_function_name': None,
                        'prior_args': None})

model_params.append({'name': 'mass', 'N': 1,
                        'isfree': True,
                        'init': 1e10,
                        'units': r'M_\odot',
                        'prior_function_name': 'tophat',
                        'prior_args': {'mini':1e9, 'maxi':1e12}})

model_params.append({'name': 'zmet', 'N': 1,
                        'isfree': True,
                        'init': 20,
                        'units': 'index',
                        'prior_function_name': 'tophat',
                        'prior_args': {'mini':1, 'maxi':5}})

model_params.append({'name': 'tau', 'N': 1,
                        'isfree': True,
                        'init': 1.0,
                        'units': 'Gyr',
                        'prior_function_name': 'tophat',
                        'prior_args': {'mini':0.1, 'maxi':100}})

model_params.append({'name': 'tage', 'N': 1,
                        'isfree': True,
                        'init': 1.0,
                        'units': 'Gyr',
                        'prior_function_name': 'tophat',
                        'prior_args': {'mini':0.1, 'maxi':10.0}})

model_params.append({'name': 'tburst', 'N': 1,
                        'isfree': True,
                        'init': 0.0,
                        'units': '',
                        'prior_function_name': 'tophat',
                        'prior_args': {'mini':0.3, 'maxi':1.3}})

model_params.append({'name': 'fburst', 'N': 1,
                        'isfree': True,
                        'init': 0.0,
                        'units': '',
                        'prior_function_name': 'tophat',
                        'prior_args': {'mini':0.0, 'maxi':0.5}})

########    Dust ##############

model_params.append({'name': 'dust_curve', 'N': 1,
                        'isfree': False,
                        'init': attenuation.cardelli,
                        'units': None})

model_params.append({'name': 'dust1', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
                        'units': '',
                        'prior_function_name': 'tophat',
                        'prior_args': {'mini':0.1, 'maxi':2.0}})

model_params.append({'name': 'dust2', 'N': 1,
                        'isfree': True,
                        'init': 0.35,
                        'units': '',
                        'prior_function_name': 'tophat',
                        'prior_args': {'mini':0.0, 'maxi':1.0}})

model_params.append({'name': 'dust_index', 'N': 1,
                        'isfree': False,
                        'init': -0.7,
                        'units': '',
                        'prior_function_name': 'tophat',
                        'prior_args': {'mini':-1.5, 'maxi':-0.5}})

model_params.append({'name': 'dust1_index', 'N': 1,
                        'isfree': False,
                        'init': -1.0,
                        'units': '',
                        'prior_function_name': 'tophat',
                        'prior_args': {'mini':-1.5, 'maxi':-0.5}})

model_params.append({'name': 'dust_tesc', 'N': 1,
                        'isfree': False,
                        'init': 7.0,
                        'units': 'log(Gyr)',
                        'prior_function_name': None,
                        'prior_args': None})

model_params.append({'name': 'dust_type', 'N': 1,
                        'isfree': False,
                        'init': 0,
                        'units': 'index'})

###### Nebular Emission ###########

model_params.append({'name': 'gas_logz', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
                        'units': r'log Z/Z_\odot',
                        'prior_function_name': 'tophat',
                        'prior_args': {'mini':-2.0, 'maxi':0.5}})

model_params.append({'name': 'gas_logu', 'N': 1,
                        'isfree': False,
                        'init': -2.0,
                        'units': '',
                        'prior_function_name': 'tophat',
                        'prior_args': {'mini':-4, 'maxi':-1}})


####### Calibration ##########

model_params.append({'name': 'phot_jitter', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
                        'units': 'mags',
                        'prior_function_name': 'tophat',
                        'prior_args': {'mini':0.0, 'maxi':0.2}})

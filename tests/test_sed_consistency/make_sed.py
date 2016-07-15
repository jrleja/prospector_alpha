import numpy as np
import pickle

'''
from bsfh import model_setup
import threed_dutils

outname = 'old_brownseds.pickle'
# setup model, sps
sps = threed_dutils.setup_sps()
sps.params['tpagb_norm_type'] = 2
model = model_setup.load_model('old_brownseds.py')
obs = model_setup.load_obs('old_brownseds.py')
#model.params['zred']=np.atleast_1d(0.0)
#obs['filters'] = None # turn off filters for now, though should also check these!

# theta = np.array([10**10.7,11.02,0.23,0.57,0.99,1.57,0.02,-1.96,0.1,0.16,0.16,2.18])
theta = np.array([10**10.7,11.02,0.23,0.57,0.8,0.0,0.02,-1.96,0.1,0.16,0.16,2.18])
spec, phot, x = model.mean_model(theta, obs, sps=sps)
w = sps.wavelengths
print 'generated OLD prospector file!'
param_dict = {}
for k, v in sps.params.iteritems(): param_dict[k] = v
print param_dict

'''
from prospect.models import model_setup

outname = 'new_brownseds.pickle'
run_params = model_setup.get_run_params(param_file='new_brownseds.py')
sps = model_setup.load_sps(**run_params)
model = model_setup.load_model(**run_params)
obs = model_setup.load_obs(**run_params)
#model.params['zred']=np.atleast_1d(0.0)

#obs['filters'] = np.array([]) # turn off filters for now, though should also check these!

# theta = np.array([10**10.7,11.02,0.23,0.57,0.99,1.57,0.02,-1.96,0.1,0.16,0.16,2.18])
theta = np.array([10.7,11.02,0.23,0.57,0.8,0.0,0.02,-1.96,0.1,0.16,0.16,2.18])
spec, phot, x = model.mean_model(theta, obs, sps=sps)
w = sps.csp.wavelengths
print 'generated NEW prospector file!'
param_dict = {}
for k, v in sps.csp.params.iteritems(): param_dict[k] = v
#'''
output = {'spec':spec,'phot':phot,'w':w,'filt':obs['filters'],'param_dict':param_dict}
pickle.dump(output,open(outname, "wb"))

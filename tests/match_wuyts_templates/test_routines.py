import fsps,os,time,pylab,threed_dutils
from bsfh import read_results,model_setup
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

c =2.99792458e8


# INTERPOLATED METALLICITY
# setup stellar populations
sps = fsps.StellarPopulation(zcontinuous=1, compute_vega_mags=False)
custom_filter_keys = os.getenv('APPS')+'/threedhst_bsfh/filters/filter_keys_threedhst.txt'
fsps.filters.FILTERS = model_setup.custom_filter_dict(custom_filter_keys)

# load custom model
param_file=os.getenv('APPS')+'/threedhst_bsfh/parameter_files/dtau_intmet/dtau_intmet_params.py'
model = model_setup.load_model(param_file=param_file, sps=sps)
obs = model_setup.load_obs(param_file=param_file)
obs['filters'] = np.array(['mips_24um_cosmos'])

# fuck with tau
#model.initial_theta[3:5] = np.array([1.0,0.1])

# fuck with dust
model.initial_theta[5:7] = 0.1

# set up MIPS + fake L_IR filter
botlam = np.atleast_1d(8e4-1)
toplam = np.atleast_1d(1000e4+1)
edgetrans = np.atleast_1d(0)
lir_filter = [[np.concatenate((botlam,np.linspace(8e4, 1000e4, num=100),toplam))],
              [np.concatenate((edgetrans,np.ones(100),edgetrans))]]

def full_min(x):


	# set model redshift to 0.0
	model.params['zred']= np.atleast_1d(0.00)

	# set params
	model.params['duste_qpah'] = np.atleast_1d(x[0])
	model.params['duste_gamma'] = np.atleast_1d(x[1])
	model.params['duste_umin']  = np.atleast_1d(x[2])

	# calculate L_IR
	spec,mags,w = model.mean_model(model.initial_theta, obs, sps=sps,norm_spec=False)
	_,lir     = threed_dutils.integrate_mag(w,spec,lir_filter, z=None, alt_file=None) # comes out in ergs/s
	lir = lir / 3.826e33 # convert to Lsun

	# set model redshift to match
	# chosen Wuyts template below
	zwuyts = np.array([0.25,0.5,0.75,1.0,1.25,1.50,1.75,2.0])
	wuyts = np.array([5.1163159e+10,3.2618332e+11,7.9534975e+11,1.5062668e+12,3.6834216e+12, 6.6729753e+12,6.3036858e+12,7.7957991e+12])
	resid = 0
	for kk in xrange(len(zwuyts)):

		model.params['zred']= np.atleast_1d(zwuyts[kk])

		# calculate observed mips flux
		spec,mags,w = model.mean_model(model.initial_theta, obs, sps=sps,norm_spec=False)
		mips_flux = mags[0]*3631*1e3 # comes out in maggies, convert to milliJy
		resid += np.abs(wuyts[kk] - lir/mips_flux)/wuyts[kk]

	resid = resid/len(zwuyts)
	print x
	print resid
	return resid

# qpah, gamma, umin
best_qpah = minimize(full_min, [1.44164572,0.04874127,1.01082902],
	                 bounds=[(0,5),(0.0,1.0),(0.1,25.0)])
print best_qpah

# default:
# 1.44164572,  0.04874127,  1.01082902
# taus = taus / 10:
# 0.78883561  0.03584305  0.90337957
# dust = dust / 10
# 1.43405339  0.04493198  1.01025742















	
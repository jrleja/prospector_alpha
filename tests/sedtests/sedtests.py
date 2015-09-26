from bsfh import model_setup
import threed_dutils
import os
import matplotlib.pyplot as plt




#### output names ####
parmfile='/Users/joel/code/python/threedhst_bsfh/parameter_files/brownseds_logzsol/brownseds_logzsol_params_1.py'

#### load model, build sps  ####
model = model_setup.load_model(parmfile)
obs   = model_setup.load_obs(parmfile)
sps   = threed_dutils.setup_sps()

#### load truths #####
truths = threed_dutils.load_truths(os.getenv('APPS')+'/threedhst_bsfh/data/brownseds_logzsol.dat',
	                        '1',None, sps=sps)

spec,maggies,_ = model.mean_model(truths['truths'], obs, sps=sps,norm_spec=False)

print obs['maggies'] - maggies

print 1/0
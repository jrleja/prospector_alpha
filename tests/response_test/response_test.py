import numpy as np
from prospect.models import model_setup

param_file = 'np_mocks_params.py'

run_params = model_setup.get_run_params(param_file=param_file)
sps = model_setup.load_sps(**run_params)
model = model_setup.load_model(**run_params)
obs = model_setup.load_obs(**run_params)

labels = model.theta_labels()
itheta = model.initial_theta

#### generate default spectrum, no dust
itheta[labels.index('dust1')] = 0.0
itheta[labels.index('dust2')] = 0.0
spec, mags, _ = model.mean_model(itheta, obs, sps=sps)

#### generate dusty spectrum
itheta[labels.index('dust2')] = 0.3
spec_dusty, mags_dusty, _ = model.mean_model(itheta,obs,sps=sps)

#### generate dustier spectrum
itheta[labels.index('dust2')] = 1.0
spec_dustier, mags_dustier, _ = model.mean_model(itheta,obs,sps=sps)

#### now fix it
sps.ssp.params.dirtiness = 1
spec_dustier_fixed, mags_dustier_fixed, _ = model.mean_model(itheta,obs,sps=sps)


print (spec-spec_dusty).sum() # should be nonzero, is nonzero
print (spec_dusty-spec_dustier).sum() # should be nonzero, is zero
print (spec_dusty-spec_dustier_fixed).sum() # should be nonzero, is nonzero
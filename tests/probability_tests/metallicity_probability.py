from prospect.models import model_setup
from prospect.likelihood import LikelihoodFunction
import numpy as np


param_file = 'nonparametric_mocks_params_1.py'

run_params = model_setup.get_run_params(param_file)
sps = model_setup.load_sps(**run_params)
gp_spec, gp_phot = model_setup.load_gp(**run_params)
model = model_setup.load_model(**run_params)
obs = model_setup.load_obs(**run_params)

likefn = LikelihoodFunction()

#### THIS IS THE OFFENDING LINE
# something about model.initial_theta is fucking shit up.....
parnames = np.array(model.theta_labels())
model.initial_theta[parnames=='logzsol'] = -0.5
#model.initial_theta[parnames=='logzsol'] = -0.444

mu, phot, x = model.mean_model(model.initial_theta, obs, sps=sps)

tpars = np.array([8.20106849,1.15846959,1.25637895,8.69569366,9.39227547,9.49399115,0.31411183,-0.44468073,-0.06894089,0.19505333,7.06496485,0.54569,7.74711812])
lnp_prior = model.prior_product(tpars)
mu, phot, x = model.mean_model(tpars, obs, sps=sps)
lnp_spec = likefn.lnlike_spec(mu, obs=obs, gp=gp_spec)
lnp_phot = likefn.lnlike_phot(phot, obs=obs, gp=gp_phot)

print lnp_prior+lnp_phot+lnp_spec
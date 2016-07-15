import matplotlib.pyplot as pl
import nonparametric_mocks_params as nparams
import brownseds_tightbc_params as params
import numpy as np

#### NONPARAMETRIC
sps = nparams.load_sps(**nparams.run_params)
model = nparams.load_model(**nparams.run_params)
obs = nparams.load_obs(**nparams.run_params)
sps.update(**model.params)

# only SFH in oldest bin
pnames = model.theta_labels()
for i,name in enumerate(pnames):
	if 'sfh_logmass' in name:
		model.initial_theta[i] = 0.0
	if name=='sfh_logmass_6':
		model.initial_theta[i] = 10.0
print model.initial_theta

# calculate spectrum
nspec, nphot, nmass = model.mean_model(model.initial_theta, obs, sps=sps)


#### PARAMETRIC
sps = params.load_sps(**nparams.run_params)
model = params.load_model(**nparams.run_params)
obs = params.load_obs(**nparams.run_params)

# ~constant SFH from tuniv to now
pnames = np.array(model.theta_labels())
model.initial_theta[pnames=='logtau'] = 2.0
model.initial_theta[pnames=='sf_tanslope'] = -np.pi/2.001
model.initial_theta[pnames=='tage'] = 13.3914871564
model.initial_theta[pnames=='delt_trunc'] = 1-0.236
print model.initial_theta

spec, phot, mass = model.mean_model(model.initial_theta, obs, sps=sps)

good = (sps.csp.wavelengths > 1e3) & (sps.csp.wavelengths < 2e4)
pl.plot(np.log10(sps.csp.wavelengths[good]),nspec[good]/spec[good],color='red')
pl.xlabel(r'log($\lambda$)')
pl.ylabel(r'nonparametric flux / parametric flux')

pl.savefig('parameter_file_test.png',dpi=150)

print nmass,mass
print 1/0


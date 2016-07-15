import matplotlib.pyplot as pl
import nonparametric_mocks_params as nparams
import numpy as np
import os

#### NONPARAMETRIC
sps = nparams.load_sps(**nparams.run_params)
model = nparams.load_model(**nparams.run_params)
obs = nparams.load_obs(**nparams.run_params)
sps.update(**model.params)

pnames = model.theta_labels()
colors = ['#76FF7A', '#1CD3A2', '#1974D2', '#7442C8', '#FC2847', '#FDFC74', '#8E4585', '#FF1DCE']
nbins = model.params['agebins'].shape[0]
model.initial_theta[:nbins-1] = 0.0

fig, ax = pl.subplots(1, 2, figsize = (12, 5))

for i in xrange(1,nbins+1):
	frac_per_bin = 1.0

	for j,name in enumerate(pnames):
		if 'fracmass' in name:
			if int(name[-1]) == i:
				model.initial_theta[j] = frac_per_bin
			else:
				model.initial_theta[j] = 0.0

	print model.initial_theta[:nbins-1]

	spec, phot, mass = model.mean_model(model.initial_theta, obs, sps=sps)

	good = (sps.wavelengths > 1e3) & (sps.wavelengths < 2e4)
	ax[0].plot(np.log10(sps.wavelengths[good]),np.log10(spec[good]),color=colors[i],label=str(i))
	ax[1].plot(np.insert(sps.logage, 0, 5.0), sps.all_ssp_weights/sps.all_ssp_weights.sum(), '-o',color=colors[i])


# calculate spectrum
ax[0].set_xlabel(r'log($\lambda$)')
ax[0].set_ylabel(r'log(f$_{\nu}$)')

#ax[0].legend(loc=1,prop={'size':10})
ax[1].set_xlabel('log(SSP age) [Gyr]')
ax[1].set_ylabel('log(all_ssp_weights/sum(all_ssp_weights))')

fig.savefig('nonparametric_sfh_test.png',dpi=150)
os.system('open nonparametric_sfh_test.png')
print 1/0
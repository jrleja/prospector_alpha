import td_massive_params as pf
from copy import deepcopy
import timeit, time, sys
import numpy as np

sps = pf.load_sps(**pf.run_params)
model = pf.load_model(**pf.run_params)
obs = pf.load_obs(**pf.run_params)

'''
model.params['add_dust_emission'] = np.array([False])
model.params['add_neb_emission'] = np.array([True])
model.params['nebemlineinspec'] = np.array([True])
mu, phot, x = model.mean_model(model.initial_theta,obs=obs,sps=sps)
model.params['add_neb_emission'] = np.array([False])
model.params['nebemlineinspec'] = np.array([False])
mu_noeline, phot_noeline, x = model.mean_model(model.initial_theta,obs=obs,sps=sps)
model.params['add_neb_emission'] = np.array([True])
model.params['nebemlineinspec'] = np.array([False])
mu_seline, phot_seline, x = model.mean_model(model.initial_theta,obs=obs,sps=sps)

xl,xh = 2, 30
idx = (sps.wavelengths/1e4 > xl) & (sps.wavelengths/1e4 < xh)
idx_phot = (obs['wave_effective']/1e4 > xl) & (obs['wave_effective']/1e4 < xh)

plt.plot(np.log10(sps.wavelengths[idx]),np.log10(mu[idx]),lw=2,color='blue')
plt.plot(np.log10(obs['wave_effective'][idx_phot]),np.log10(phot[idx_phot]),'o',linestyle=' ',color='blue',alpha=0.6)
plt.plot(np.log10(sps.wavelengths[idx]),np.log10(mu_noeline[idx]),lw=2,color='red')
plt.plot(np.log10(obs['wave_effective'][idx_phot]),np.log10(phot_noeline[idx_phot]),'o',linestyle=' ',color='red',alpha=0.6)
'''
if __name__ == "__main__":

	### 100 attempts
	ntry = 100
	theta = deepcopy(model.initial_theta)
	logzsol = np.random.uniform(-0.8, -0.2,ntry+1)
	idx = np.array(model.theta_labels()) == 'logzsol'
	time1, time2 = [],[]

	### fast version
	# throw out one of these
	model.params['nebemlineinspec'] = np.array([False])
	model.params['add_neb_emission'] = np.array([True])
	for i in xrange(ntry+1):
		theta[idx] = logzsol[i]
		ts = time.time()
		mu, phot_fake, x = model.mean_model(model.initial_theta,sps=sps,obs=obs)
		ts2 = time.time()
		if i > 0:
			time1.append(ts2-ts)

	### slow version
	model.params['nebemlineinspec'] = np.array([False])
	model.params['add_neb_emission'] = np.array([False])
	for i in xrange(ntry):
		theta[idx] = logzsol[i]
		ts = time.time()
		mu, phot_fake, x = model.mean_model(model.initial_theta,sps=sps,obs=obs)
		ts2 = time.time()
		time2.append(ts2-ts)

	print time1
	print time2
	print np.array(time1).mean()
	print np.array(time2).mean()



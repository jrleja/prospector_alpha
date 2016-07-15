import numpy as np
import matplotlib.pyplot as plt
import magphys_plot_pref
np.random.seed(50)

###### NUMBER OF DRAWS
ndraw_dirichlet = int(1e6)
ndraw_sfh_prior = int(1e6)
nbins = 6

###### GENERATE SFH PARAMETERS
def generate_sfh_pars():
	sfh = np.array([nbins-1])
	while sfh.sum() > 1.0:
		sfh = np.random.random(nbins-1)

	sfh = np.concatenate((sfh,np.array([1-sfh.sum()])))

	return sfh


##### draw dirichlet
dirichlet = np.random.dirichlet(tuple(1. for x in xrange(nbins)),ndraw_dirichlet)
dirichlet_straight = np.ravel(dirichlet)

##### draw sfh
sfh = np.zeros(shape=(nbins,ndraw_sfh_prior))
for ii in xrange(ndraw_sfh_prior): sfh[:,ii] = generate_sfh_pars()

##### histoplot
fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(14,6))
nbins = 100
alpha = 0.5
lw = 2
colors = ['red','blue']
histtype = 'step'

##### all five bins
sfh_direct = np.ravel(sfh[:-1,:])
num, b, p = ax[0].hist(sfh_direct, bins=nbins, range=(0.0,1.0), histtype=histtype, alpha=alpha,lw=lw,color=colors[0],normed=True,log=True)
num, b, p = ax[0].hist(dirichlet_straight, bins=nbins, range=(0.0,1.0), histtype=histtype, alpha=alpha,lw=lw,color=colors[1],normed=True,log=True)

sfh_derived = np.ravel(sfh[-1,:])
num, b, p = ax[1].hist(sfh_derived, bins=nbins, range=(0.0,1.0),histtype=histtype, alpha=alpha,lw=lw,color=colors[0],normed=True,log=True)
num, b, p = ax[1].hist(dirichlet_straight, bins=nbins, range=(0.0,1.0),histtype=histtype, alpha=alpha,lw=lw,color=colors[1],normed=True,log=True)

ax[0].set_ylabel('log(density)')
ax[0].set_xlabel('fractional variables')

ax[1].set_ylabel('log(density)')
ax[1].set_xlabel('1-sum(fractional variables)')

ax[0].text(0.02,0.15, 'SFH distribution', transform = ax[0].transAxes,ha='left',color=colors[0],fontsize='large',weight='bold')
ax[0].text(0.02,0.10, 'Dirichlet distribution', transform = ax[0].transAxes,ha='left',color=colors[1],fontsize='large',weight='bold')

plt.show()
plt.savefig('dirichlet_test.png',dpi=150)
print 1/0

import matplotlib.pyplot as plt
import brownseds_np_params_new as nparams
import magphys_plot_pref
import numpy as np
import os
from astropy.cosmology import WMAP9 as cosmo

def cosmo_distance(zred):
	return cosmo.luminosity_distance(zred).value**(-2) * (1+zred)
def cosmo_distance_extra(zred):
	return cosmo.luminosity_distance(zred).value**(-2) * (1+zred)**2

#### LOAD ALL OF THE THINGS
sps = nparams.load_sps(**nparams.run_params)
model = nparams.load_model(**nparams.run_params)
obs = nparams.load_obs(**nparams.run_params)
sps.update(**model.params)

#### REDSHIFT BINS
zred = np.linspace(0.25,2,8)
zstart_factor = cosmo_distance(zred[0])
zstart_factor_extra = cosmo_distance_extra(zred[0])

mag_z = np.zeros_like(zred)
scaling = np.zeros_like(zred)
scaling_extra = np.zeros_like(zred)
idx = 500
for i,z in enumerate(zred):

	#### magnitudes
	model.params['zred'] = zred[i]
	sp,mags,sm = model.mean_model(model.initial_theta, obs, sps=sps)
	mag_z[i] = sp[idx]

	#### luminosity density scaling with distance
	if i == 0:
		scaling[i] = 1.
		scaling_extra[i] = 1.
	else:
		scaling[i] = cosmo_distance(zred[i]) / zstart_factor
		scaling_extra[i] = cosmo_distance_extra(zred[i]) / zstart_factor_extra

fig, ax = plt.subplots(1, 1, figsize = (9,9))

ax.plot(zred,np.log10(mag_z[0]*scaling),marker=' ',linestyle='dashed',lw=3,alpha=0.8,color='blue')
ax.plot(zred,np.log10(mag_z[0]*scaling_extra),marker=' ',linestyle='dashed',lw=3,alpha=0.8,color='green')
ax.plot(zred,np.log10(mag_z),marker='o',linestyle='None',alpha=0.6,markersize=10,color='red')

ax.text(0.95,0.88, r'f$_{\nu}$ $\propto$ (1+z)/D$_{L}$$^{2}$',transform = ax.transAxes,ha='right',color='blue',fontsize='large')
ax.text(0.95,0.81, r'f$_{\nu}$ $\propto$ (1+z)$^2$/D$_{L}$$^{2}$',transform = ax.transAxes,ha='right',color='green',fontsize='large')
ax.text(0.95,0.95, 'Prospector',transform = ax.transAxes,ha='right',color='red',fontsize='large')

ax.set_xlabel(r'z$_{\mathrm{red}}$')
ax.set_ylabel(r'log(f$_{\nu}$)')

ax.set_xlim(0,2.2)

ax.plot(zred,np.log10(mag_z[0]*scaling),marker=' ',linestyle='dashed',lw=3,alpha=0.8,color='blue')
ax.plot(zred,np.log10(mag_z[0]*scaling_extra),marker=' ',linestyle='dashed',lw=3,alpha=0.8,color='green')
ax.plot(zred,np.log10(mag_z),marker='o',linestyle='None',alpha=0.6,markersize=10,color='red')

ax.text(0.95,0.88, r'f$_{\nu}$ $\propto$ (1+z)/D$_{L}$$^{2}$',transform = ax.transAxes,ha='right',color='blue',fontsize='large')
ax.text(0.95,0.81, r'f$_{\nu}$ $\propto$ (1+z)$^2$/D$_{L}$$^{2}$',transform = ax.transAxes,ha='right',color='green',fontsize='large')
ax.text(0.95,0.95, 'Prospector',transform = ax.transAxes,ha='right',color='red',fontsize='large')

ax.set_xlabel(r'z$_{\mathrm{red}}$')
ax.set_ylabel(r'log(f$_{\nu}$)')

ax.set_xlim(0,2.2)

plt.show()
plt.savefig('flux_vs_z.png',dpi=150)
print 1/0
import numpy as np
import fsps, pickle, prosp_dutils, os
from bsfh import model_setup
import matplotlib.pyplot as plt

# setup model, sps
sps = prosp_dutils.setup_sps()
model = model_setup.load_model('brownseds_params_26.py')
obs = model_setup.load_obs('brownseds_params_26.py')
model.params['add_neb_emission'] = np.array(True)

### find indexes
parnames = np.array(model.theta_labels())
d1_idx = parnames == 'dust1'
d2_idx = parnames == 'dust2'
dind_idx = parnames == 'dust_index'
tage_idx = parnames == 'tage'
trunc_idx = parnames == 'sf_trunc'
slope_idx = parnames == 'sf_slope'

#### set up dust array
sfh_params = prosp_dutils.find_sfh_params(model,model.initial_theta,obs,sps)
nsamp = 45
dust2 = np.linspace(0.0,1.0, nsamp)
tage = 10**np.linspace(-0.9,1.0,nsamp)
trunc = np.linspace(0.01,0.99,nsamp)
gaslogu = np.linspace(-4,-1,nsamp)
gaslogz = np.linspace(-2.0,0.5,nsamp)

###### calculate FSPS quantities ######
# pass parameters to sps object
mod_bdec,bdec,ptau1,ptau2,pdindex = [],[],[],[],[]
thetas = model.initial_theta
thetas[d1_idx] = 0.0
thetas[d2_idx] = 0.0
thetas[slope_idx] = 0.0
thetas[tage_idx] = 1.0
for ii,dd in enumerate(gaslogz):
    #thetas[d1_idx] = dd
    #thetas[trunc_idx] = thetas[tage_idx]*dd
    model.params['gas_logz'] = dd
    model.set_parameters(thetas)

    ##### CLOUDY halpha / hbeta
    modelout = prosp_dutils.measure_emline_lum(sps, thetas = thetas,model=model, obs = obs,saveplot=False, savestr='glogz_'+"{:.2f}".format(dd))
    mod_bdec.append(modelout['emline_flux'][4]/modelout['emline_flux'][1])

    ##### calculated halpha / hbeta
    '''
    ptau1.append(thetas[d1_idx][0])
    ptau2.append(thetas[d2_idx][0])
    pdindex.append(thetas[dind_idx][0])
    bdec.append(prosp_dutils.calc_balmer_dec(ptau1[-1], ptau2[-1], -1.0, pdindex[-1],kriek=True))
    '''

mod_bdec = np.array(mod_bdec)
bdec = np.array(bdec)

# x + y, residual
fig, ax = plt.subplots(1,1,figsize = (8,8))
ax.set_xlabel(r'gas_logz')
ax.set_ylabel('CLOUDY balmer decrement')
ax.plot(gaslogz,mod_bdec,'o',linestyle=' ')
ax.vlines(0.0,ax.get_ylim()[0],ax.get_ylim()[1], linestyle='-',colors='k')

plt.savefig('gas_logz_dustfree.png',dpi=300)
plt.close()

import numpy as np
import fsps, pickle, threed_dutils, os
from bsfh import model_setup
import matplotlib.pyplot as plt

# setup model, sps
sps = threed_dutils.setup_sps()

#### set up dust array
sps.params['sfh'] = 0.0
sps.params['imf_type'] = 1.0
sps.params['zmet'] = 20
sps.params['add_neb_emission'] = True
sps.params['add_neb_continuum'] = True
sps.params['gas_logu'] = -2.0
sps.params['gas_logz'] = 0.0
nsamp = 45
tage = 10**np.linspace(-0.9,1.0,nsamp)
tage = sps.ssp_ages

###### calculate FSPS quantities ######
# pass parameters to sps object
mod_bdec,bdec,ptau1,ptau2,pdindex = [],[],[],[],[]

###### get spec
# nebon
w,spec= sps.get_spectrum(tage=0.0,peraa=False)

# neboff
sps.params['add_neb_emission'] = False
sps.params['add_neb_continuum'] = False
w, spec_neboff = sps.get_spectrum(tage=0.0,peraa=False)

# subtract, switch to flam
factor = 3e18 / w**2
spec = (spec-spec_neboff) *factor

for ii,dd in enumerate(tage):
    sps.params['tage'] = dd


    ##### CLOUDY halpha / hbeta
    modelout = threed_dutils.measure_emline_lum(sps, thetas = None,model=None, obs = None,saveplot=False, 
                                                savestr='tage_'+"{:.2f}".format(dd),spec=spec[ii,:],measure_ir = False)
    mod_bdec.append(modelout['emline_flux'][4]/modelout['emline_flux'][1])

    ##### calculated halpha / hbeta
    '''
    ptau1.append(thetas[d1_idx][0])
    ptau2.append(thetas[d2_idx][0])
    pdindex.append(thetas[dind_idx][0])
    bdec.append(threed_dutils.calc_balmer_dec(ptau1[-1], ptau2[-1], -1.0, pdindex[-1],kriek=True))
    '''

mod_bdec = np.array(mod_bdec)
bdec = np.array(bdec)

# plot
fig, ax = plt.subplots(1,1,figsize = (8,8))
ax.set_xlabel(r'SSP age')
ax.set_ylabel('CLOUDY balmer decrement')
ax.plot(tage,mod_bdec,'o',linestyle=' ')

plt.savefig('ssp_bdec.png',dpi=300)
plt.close()
print 1/0

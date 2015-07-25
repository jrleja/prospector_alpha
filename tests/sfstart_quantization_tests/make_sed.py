import numpy as np
import fsps, pickle, threed_dutils, os
from bsfh import model_setup
import matplotlib.pyplot as plt

# change to a folder name, to save figures
outname = ''

# setup model, sps
sps = threed_dutils.setup_sps()
model = model_setup.load_model('testsed_simha_params_9.py')
initial_theta = np.array([10**8.50,-0.78,10**0.34,11.69,13.31,-1.67,2.94,-1.38])

# make sure parameters are set
model.initial_theta = initial_theta
model.set_parameters(model.initial_theta)

# set up sf_start array
sfh_params = threed_dutils.find_sfh_params(model,model.initial_theta)
tcalc = np.linspace(11.5,11.9, 8)
tcalc = np.append(11.69,tcalc)
print tcalc
c = 3e8

# set up plot
fig, ax = plt.subplots()
colors = ['#76FF7A', '#1CD3A2', '#1974D2', '#7442C8', '#FC2847', '#FDFC74', '#8E4585', '#FF1DCE']

###### calculate FSPS quantities ######
# pass parameters to sps object
for ii,tt in enumerate(tcalc):
    model.params['sf_start'] = np.atleast_1d(tt)
    for k, v in model.params.iteritems():
        if k in sps.params.all_params:
            if k == 'zmet':
                vv = np.abs(v - (np.arange( len(sps.zlegend))+1)).argmin()+1
            elif k == 'dust1':
                # temporary! replace with smarter function soon
                vv = model.params['dust2']*1.86+0.0
            else:
                vv = v.copy()
            sps.params[k] = vv
        if k == 'mass':
            mass = v

    w, spec = sps.get_spectrum(tage=sps.params['tage'], peraa=False)
    yplot   = spec*(c/(w/1e10))
    if ii == 0:
        origspec = yplot
    else:
        ax.plot(np.log10(w),yplot/origspec,linestyle='-',alpha=0.5,color=colors[ii-1],label=tt)

ax.set_xlabel(r'log($\lambda [\AA]$)')
ax.set_ylabel(r'f_{model} / f_{sf_start=11.69}')
ax.legend(loc=1,prop={'size':10},title='sf_start')
ax.axis((3.3,6,0,2))

plt.savefig('sfstart_test.png',dpi=300)
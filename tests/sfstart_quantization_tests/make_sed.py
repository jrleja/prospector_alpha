import numpy as np
import fsps, pickle, threed_dutils, os
from bsfh import model_setup
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors

def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv') 
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color

# change to a folder name, to save figures
outname = ''

# setup model, sps
sps = threed_dutils.setup_sps()
model = model_setup.load_model('brownseds_params_26.py')
obs = model_setup.load_obs('brownseds_params_26.py')
initial_theta=np.array([10**7.01,-1.95,1.34,10.04,0.91,1.24,0.02,0.00,-0.7,0.81,22.62,0.12])
initial_theta=np.array([10**8.44,-1.89,0.62,1.57,0.98,-1.54,0.11,0.01,-1.1,0.89,23.02,0.19])

# make sure parameters are set
model.initial_theta = initial_theta
model.set_parameters(model.initial_theta)

# set up sf_start array
sfh_params = threed_dutils.find_sfh_params(model,model.initial_theta,obs,sps)
nsamp = 200
cmap = get_cmap(nsamp)
tcalc = np.linspace(1.4,1.8, nsamp)
tcalc = np.append(1.57,tcalc)
print tcalc
c = 3e8

# set up plot
fig, ax = plt.subplots()
colors = ['#76FF7A', '#1CD3A2', '#1974D2', '#7442C8', '#FC2847', '#FDFC74', '#8E4585', '#FF1DCE']

###### calculate FSPS quantities ######
# pass parameters to sps object
sfr = np.array([])
for ii,tt in enumerate(tcalc):
    for k, v in model.params.iteritems():
        if k in sps.params.all_params:
            if k == 'zmet':
                vv = np.abs(v - (np.arange( len(sps.zlegend))+1)).argmin()+1
            else:
                vv = v.copy()
            sps.params[k] = vv
        if k == 'mass':
            mass = v
    sps.params['tage'] = np.atleast_1d(tt)
    sps.params['sf_trunc'] = np.atleast_1d(0.98*tt)

    w, spec = sps.get_spectrum(tage=sps.params['tage'], peraa=False)
    yplot   = spec*(c/(w/1e10))
    sfr = np.append(sfr,sps.sfr)
    if ii == 0:
        origspec = yplot
    else:
        ax.plot(np.log10(w),yplot/origspec,linestyle='-',alpha=0.5,color=cmap(ii),label=tt)

fmt = "{:.2f}".format(tcalc[0])
ax.set_xlabel(r'log($\lambda [\AA]$)')
ax.set_ylabel(r'f$_{\mathrm{model}}$ / f$_{\mathrm{tage='+fmt+'}}$')
ax.legend(loc=1,prop={'size':10},title='tage')
ax.axis((3,6.0,0.3,4.0))

plt.savefig('tage_test.png',dpi=300)
plt.close()

fig, ax = plt.subplots()
ax.set_xlabel(r'tage')
ax.set_ylabel(r'sfr')
ax.plot(tcalc[1:],sfr[1:],alpha=0.5,color='blue')
plt.savefig('sfr.png',dpi=300)
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

# make sure parameters are set
model.set_parameters(model.initial_theta)

# set up sf_start array
sfh_params = threed_dutils.find_sfh_params(model,model.initial_theta,obs,sps)
nsamp = 20
cmap = get_cmap(nsamp)
dust2 = np.linspace(0.0,4.0, nsamp)

c = 3e8

# set up plot
fig, ax = plt.subplots()
colors = ['#76FF7A', '#1CD3A2', '#1974D2', '#7442C8', '#FC2847', '#FDFC74', '#8E4585', '#FF1DCE']

###### calculate FSPS quantities ######
# pass parameters to sps object
for ii,tt in enumerate(dust2):
    for k, v in model.params.iteritems():
        if k in sps.params.all_params:
            if k == 'zmet':
                vv = np.abs(v - (np.arange( len(sps.zlegend))+1)).argmin()+1
            else:
                vv = v.copy()
            sps.params[k] = vv
        if k == 'mass':
            mass = v
    sps.params['tage'] = 0.5
    sps.params['dust1'] = np.atleast_1d(tt)
    w, spec = sps.get_spectrum(tage=sps.params['tage'], peraa=False)
    mags = sps.get_mags(tage=sps.params['tage'])
    yplot   = spec*(c/(w/1e10))
    if ii == 0:
        origmags = mags
        origspec = yplot
    else:
        #ax.plot(mags/origmags,linestyle='-',alpha=0.5,color=cmap(ii),label=tt)
        ax.plot(np.log10(w),yplot/origspec,linestyle='-',alpha=0.5,color=cmap(ii),label=tt)

fmt = "{:.2f}".format(dust2[0])
ax.set_xlabel(r'log($\lambda [\AA]$)')
ax.set_ylabel(r'f$_{\mathrm{model}}$ / f$_{\mathrm{dust1='+fmt+'}}$')
ax.legend(loc=2,prop={'size':10},title='tage')
ax.axis((3,6.0,0.3,4.0))
print 1/0
plt.savefig('dust_type_test.png',dpi=300)
plt.close()

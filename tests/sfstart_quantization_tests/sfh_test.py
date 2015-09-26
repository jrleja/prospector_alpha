import numpy as np
import fsps
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

# setup model, sps
sps = fsps.StellarPopulation(zcontinuous=0, compute_vega_mags=False)
sps.params['sfh'] = 5
sps.params['sf_slope'] = np.tan(-1.5)
sps.params['tau'] = 10**0.62

# set up arrays
nsamp = 450
cmap = get_cmap(nsamp)
tcalc = np.linspace(1.4,5.4, nsamp)
c = 3e8

# loop over tage
sfr = np.array([])
for ii,tt in enumerate(tcalc):
    sps.params['tage'] = np.atleast_1d(tt)
    sps.params['sf_trunc'] = sps.params['tage'] * 0.97
    w, spec = sps.get_spectrum(tage=sps.params['tage'], peraa=False)
    sfr = np.append(sfr,sps.sfr)

# plot
fig, ax = plt.subplots()
ax.set_xlabel(r'tage')
ax.set_ylabel(r'sfr')
ax.plot(tcalc[1:],sfr[1:],alpha=0.5,color='blue')
plt.savefig('sfr.png',dpi=300)

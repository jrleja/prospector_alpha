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
sps.params['sf_trunc'] = 10.04 * 0.91
sps.params['sf_slope'] = 1.2863
sps.params['tau'] = 10**1.34

# set up sf_start array
nsamp = 15
cmap = get_cmap(nsamp)
tcalc = np.linspace(8.5,9.2, nsamp)
c = 3e8

# set up plot
fig, ax = plt.subplots()

# loop over tage, using the first element in tage
# as the normalization element
for ii,tt in enumerate(tcalc):
    sps.params['tage'] = np.atleast_1d(tt)

    w, spec = sps.get_spectrum(tage=sps.params['tage'], peraa=False)
    yplot   = spec*(c/(w/1e10))
    if ii == 0:
        origspec = yplot
    else:
        ax.plot(np.log10(w),yplot/origspec,linestyle='-',alpha=0.5,color=cmap(ii),label=tt)

fmt = "{:.2f}".format(tcalc[0])
ax.set_xlabel(r'log($\lambda [\AA]$)')
ax.set_ylabel(r'f$_{\mathrm{model}}$ / f$_{\mathrm{tage='+fmt+'}}$')
ax.legend(loc=1,prop={'size':10},title='tage')
ax.axis((2,4,0.00, 3.0))

plt.savefig('tage_test.png',dpi=300)
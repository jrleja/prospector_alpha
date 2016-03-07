import numpy as np
import threed_dutils
import matplotlib.pyplot as plt
import magphys_plot_pref

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

def plot_optical_depth():

	dust2_index_priors = np.array([-0.4, -3.0])
	lam = np.linspace(1e3,1e4,1000)
	dust2 = 1.0

	#### set up new dust parameters
	nsamp = 12
	dust2_new = np.linspace(-2.0,0.4,nsamp)
	dust2_new = np.linspace(-2.5,0.6,nsamp)
	cmap = get_cmap(nsamp)

	#### open figure
	fig, ax = plt.subplots(1,1,figsize=(8,8))

	#### plot noll version
	for ii,didx in enumerate(dust2_new):
		optical_depth = -np.log(threed_dutils.charlot_and_fall_extinction(lam,0.0,dust2,0.0,didx, kriek=True, nobc=True))
		ax.plot(np.log10(lam),np.log(optical_depth),color=cmap(ii),alpha=0.7,lw=2.5,label="{:.2f}".format(didx))

	#### plot old priors
	for didx in dust2_index_priors:
		optical_depth = -np.log(threed_dutils.charlot_and_fall_extinction(lam,0.0,dust2,0.0,didx, kriek=False, nobc=True))
		ax.plot(np.log10(lam),np.log(optical_depth),color='k',alpha=0.7,lw=4.0)

	#### legend + labels
	ax.legend(loc=1,prop={'size':12},title='Noll+09 dust_index')
	ax.get_legend().get_title().set_fontsize('16')
	ax.set_xlim(3.0,4.4)
	ax.set_ylim(-2.5,6)

	ax.text(4.02,-1.9,'old \n dust_index=-3.0',fontsize=14,weight='bold',multialignment='center')
	ax.text(4.02,-0.4,'old \n dust_index=-0.4',fontsize=14,weight='bold',multialignment='center')


	ax.set_xlabel(r'log10(wavelength) [$\AA$]')
	ax.set_ylabel(r'ln(diffuse optical depth)')

	plt.show()
	print 1/0

if __name__ == "__main__":
	plot_optical_depth()
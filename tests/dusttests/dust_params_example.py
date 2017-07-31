import numpy as np
import fsps, pickle, prosp_dutils, os
from bsfh import model_setup
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import copy

c =2.99792458e8


outplot = 'dustem_params.png'

def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv') 
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color

# setup model, sps
sps = prosp_dutils.setup_sps()
param_file=os.getenv('APPS')+'/prospector_alpha/parameter_files/brownseds_tightbc/brownseds_tightbc_params.py'

model = model_setup.load_model(param_file)
obs = model_setup.load_obs(param_file)
#initial_theta=np.array([10**8.44,-1.89,0.62,1.57,0.98,-1.54,0.11,0.01,-1.1,0.89,23.02,0.19])

# make sure parameters are set
#model.initial_theta = initial_theta
#model.set_parameters(model.initial_theta)
theta = copy.copy(model.initial_theta)

### SET UP PLOT
fig, axes = plt.subplots(1,1,figsize=(7,7))
ax = np.ravel(axes)

#### DEFINE PARAMETERS TO ITERATE OVER
param_iterable= ['duste_qpah']#,'duste_gamma','duste_umin']
parnames = np.array(model.theta_labels())

#### ITERATE OVER PARAMETERS
for nn, par in enumerate(param_iterable):

	#### DEFINE PARAMETER RANGE
	idx = parnames == par
	bounds = np.array(model.theta_bounds())[idx][0]

	#### GENERATE RANGE OF PARAMETERS + COLOR MAP
	nsamp = 6
	cmap = get_cmap(nsamp)
	par_vec = np.linspace(bounds[0],bounds[1],nsamp)

	###### calculate FSPS quantities ######
	for ii,tt in enumerate(par_vec):

		#### SET THETAS
		theta[idx] = tt

		#### GET SPECTRUM
		spec,mags,_ = model.mean_model(theta, obs, sps=sps, norm_spec=False)

		#### PLOT SPECTRUM
		ax[nn].plot(sps.wavelengths/1e4, np.log10(spec),alpha=0.5,color=cmap(ii),label="{:.2f}".format(tt))

	#### RESET THETAS
	theta = copy.copy(model.initial_theta)

	#### CLEAN UP AND FORMAT PLOT
	ax[nn].legend(loc=4,prop={'size':12},title=r'Q$_{\mathrm{PAH}}$',frameon=False)
	ax[nn].get_legend().get_title().set_fontsize('16')
	ax[nn].set_xlabel(r'$\lambda [\mu m]$')
	ax[nn].set_ylabel(r'log(f$_{\nu}$)')
	ax[nn].set_xlim(1,30)
	ax[nn].set_ylim(-6,-3.0)

plt.savefig('dustem_test.png',dpi=300)
plt.close()
os.system('open dustem_test.png')































	
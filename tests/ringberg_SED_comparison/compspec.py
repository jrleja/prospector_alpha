import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d

objnames = ['IC_4051', 'NGC_3198']

def load_prospector_spec(objname):
    dtype = {'names':(['wave','q50','q84','q16']),'formats':(np.repeat('f16',4))}
    dat = np.loadtxt(objname+'_modspec.txt',dtype=dtype)
    return dat

def load_obs_spec(objname):
    file = 'make_brown_data/'+objname+'_spec.txt'
    dtype = {'names':(['wave','spec']),'formats':(np.repeat('f16',2))}
    dat = np.loadtxt(file,dtype=dtype)
    return dat

def compare_spectra(objname):

    ### load data
    mod_dat = [load_prospector_spec(objname)]
    mod_name = ['Prospector']
    mod_color = ['blue']
    obs_dat = load_obs_spec(objname)

    ### create figure
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(2,1, height_ratios=[2,1])
    ax_spec, ax_res = plt.Subplot(fig, gs[0]), plt.Subplot(fig, gs[1])
    fig.add_subplot(ax_spec)
    fig.add_subplot(ax_res)

    ### plot data
    ax_spec.plot(obs_dat['wave'], obs_dat['spec'], color='k', lw=3, zorder=1)

    for i,dat in enumerate(mod_dat):

        ### interpolate data
        med_interp = interp1d(dat['wave'], dat['q50'], bounds_error = False, fill_value = 0)
        up_interp = interp1d(dat['wave'], dat['q84'], bounds_error = False, fill_value = 0)
        do_interp = interp1d(dat['wave'], dat['q16'], bounds_error = False, fill_value = 0)
        med = med_interp(obs_dat['wave'])
        up = up_interp(obs_dat['wave'])
        do = do_interp(obs_dat['wave'])

        ax_spec.plot(obs_dat['wave'], med, color=mod_color[i], label=mod_name[i], lw=2, alpha = 0.8)
        ax_spec.fill_between(obs_dat['wave'],do, up, color=mod_color[i], alpha=0.2)
        ax_res.plot(obs_dat['wave'], (med-obs_dat['spec'])/obs_dat['spec'], color=mod_color[i], lw=2, alpha=0.8)

    ax_res.axvline(1.0, linestyle='--', color='0.2',lw=1.5,zorder=-5)
    ax_res.set_xlabel(r'observed wavelength [$\AA$]')
    ax_res.set_ylabel(r'(mod-obs)/obs')
    ax_spec.set_xlabel(r'flux')

    plt.show()
    print 1/0




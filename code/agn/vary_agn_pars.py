import numpy as np
import prosp_dutils
import matplotlib.pyplot as plt
import magphys_plot_pref
import time

import matplotlib.cm as cmx
import matplotlib.colors as colors
import copy
import matplotlib as mpl
import math
import brownseds_agn_params_1 as nparams
import matplotlib.patheffects as pe
import observe_agn_templates

#### LOAD ALL OF THE THINGS
sps = nparams.load_sps(**nparams.run_params)
model = nparams.load_model(**nparams.run_params)
model.params['zred'] = 0.0
obs = nparams.load_obs(**nparams.run_params)
sps.update(**model.params)

class jLogFormatter(mpl.ticker.LogFormatter):
    '''
    this changes the format from exponential to floating point.
    '''

    def __call__(self, x, pos=None):
        """Return the format for tick val *x* at position *pos*"""
        vmin, vmax = self.axis.get_view_interval()
        d = abs(vmax - vmin)
        b = self._base
        if x == 0.0:
            return '0'
        sign = np.sign(x)
        # only label the decades
        fx = math.log(abs(x)) / math.log(b)
        isDecade = mpl.ticker.is_close_to_int(fx)
        if not isDecade and self.labelOnlyBase:
            s = ''
        elif x > 10000:
            s = '{0:.3g}'.format(x)
        elif x < 1:
            s = '{0:.3g}'.format(x)
        else:
            s = self.pprint_val(x, d)
        if sign == -1:
            s = '-%s' % s
        return self.fix_minus(s)

minorFormatter = jLogFormatter(base=10, labelOnlyBase=False)
majorFormatter = jLogFormatter(base=10, labelOnlyBase=True)

def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='plasma') 
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color

def make_plot():

    #### open figure
    itheta = copy.deepcopy(model.initial_theta)
    fig, ax = plt.subplots(1,3,figsize=(15,5))
    outname = '/Users/joel/code/python/prospector_alpha/plots/brownseds_agn/agn_plots/AGN_parameters.png'

    #### find indexes
    pnames = np.array(model.theta_labels())
    didx = pnames == 'fagn'
    tidx = pnames == 'agn_tau'

    #### set up parameters
    nsamp = 5
    to_samp = [r'f$_{\mathrm{AGN,MIR}}$',r'$\tau_{\mathrm{AGN}}$']
    samp_pars = [[0.0, 0.05, 0.1, 0.2, 0.5],[5,10,20,40,150]] 

    idx = [didx,tidx]
    model.initial_theta[didx] = 0.5
    model.initial_theta[tidx] = 20

    ### define wavelength regime + conversions
    to_plot = np.array([0.5,200])
    plt_idx = (sps.wavelengths > to_plot[0]*1e4) & (sps.wavelengths < to_plot[1]*1e4)
    onemicron = np.abs((sps.wavelengths[plt_idx]/1e4 - 1)).argmin()

    ### add AGN-only templates
    observe_agn_templates.plot(ax[0])

    #### generate spectra
    fmir = []
    for k,par in enumerate(to_samp):
        nsamp = len(samp_pars[k])
        cmap = get_cmap(nsamp)
        for i in xrange(nsamp):
            itheta[idx[k]] = samp_pars[k][i]
            spec,mags,sm = model.mean_model(itheta, obs, sps=sps)

            if k == 0:
                modelout = prosp_dutils.measure_restframe_properties(sps, thetas = itheta,
                                                model=model, obs=obs,
                                                measure_mir=True)
                lmir_agn = modelout['lmir']
                itheta[idx[k]] = 0.0
                modelout = prosp_dutils.measure_restframe_properties(sps, thetas = itheta,
                                        model=model, obs=obs,
                                        measure_mir=True)
                text = (lmir_agn-modelout['lmir'])/lmir_agn
            else:
                text = samp_pars[k][i]

            yplot = spec[plt_idx] * 3e18*3631*1e-23/sps.wavelengths[plt_idx]
            yplot /= yplot[onemicron]

            ax[k+1].plot(sps.wavelengths[plt_idx]/1e4,yplot,
                       color=cmap(i),lw=2.5,label="{:.1f}".format(text),
                       path_effects=[pe.Stroke(linewidth=4.5, foreground='k',alpha=0.7), pe.Normal()],
                       zorder=k)
        itheta[idx[k]] = model.initial_theta[idx[k]]

    #### legend + labels
    ax[1].legend(loc=2,prop={'size':12},title='f$_{\mathrm{AGN,MIR}}$',frameon=False,ncol=2)
    ax[2].legend(loc=2,prop={'size':12},title=to_samp[1],frameon=False)
    ax[1].text(0.971,0.08,r'$\tau_{\mathrm{AGN}}$=20',transform=ax[1].transAxes,fontsize=12,ha='right')
    ax[2].text(0.971,0.08,r'f$_{\mathrm{AGN,MIR}}$=0.8',transform=ax[2].transAxes,fontsize=12,ha='right')

    ax[1].text(0.95,0.91,'AGN+galaxy',weight='semibold',transform=ax[1].transAxes,ha='right',fontsize=16)
    ax[2].text(0.95,0.91,'AGN+galaxy',weight='semibold',transform=ax[2].transAxes,ha='right',fontsize=16)

    for a in ax[1:]:
        a.get_legend().get_title().set_fontsize('16')
        a.set_xscale('log',nonposx='clip',subsx=(1,3))
        a.xaxis.set_minor_formatter(minorFormatter)
        a.xaxis.set_major_formatter(majorFormatter)
        for tl in a.get_xticklabels():tl.set_visible(False)

        a.set_xlim(to_plot)

        a.set_xlabel(r'wavelength [$\mu$m]')
        a.set_ylabel(r'$\nu$f$_{\nu}$ [normalized]')
        a.set_yscale('log',nonposy='clip',subsy=(1,2,4))
        a.xaxis.set_minor_formatter(minorFormatter)
        a.xaxis.set_major_formatter(majorFormatter)
        for tl in a.get_yticklabels():tl.set_visible(False)
        a.set_ylim(a.get_ylim()[0],a.get_ylim()[1]*3)

    plt.tight_layout()
    fig.savefig(outname,dpi=150)
    plt.close()
    
if __name__ == "__main__":
    plot_agn_fraction()
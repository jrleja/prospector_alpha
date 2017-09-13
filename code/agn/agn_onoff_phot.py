import numpy as np
import matplotlib.pyplot as plt
from prospect.io import read_results
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
from prospect.models import model_setup
from magphys_plot_pref import jLogFormatter
from prospector_io import load_prospector_data
from prosp_dutils import asym_errors, smooth_spectrum

plt.ioff() # don't pop up a window for each plot

minorFormatter = jLogFormatter(base=10, labelOnlyBase=False)
majorFormatter = jLogFormatter(base=10, labelOnlyBase=True)

def plot_all(outfolder='/Users/joel/code/python/prospector_alpha/plots/brownseds_agn/agn_plots/'):

    fig_agn, ax_agn = plt.subplots(2,3, figsize=(12, 7))
    fig_both, ax_both = plt.subplots(2,3, figsize=(12, 7))
    fig_agn.subplots_adjust(hspace=0.05)
    fig_both.subplots_adjust(hspace=0.05)
    ax_agn = np.ravel(ax_agn)
    ax_both = np.ravel(ax_both)

    to_plot = ['IRAS 08572+3915','NGC 3690', 'NGC 7674','UGC 08696', 'CGCG 049-057', 'NGC 4552'] # in a specific order

    for i, name in enumerate(to_plot):

        ### load up
        sresults_agn, _, _, eout_agn = load_prospector_data(None, objname=name, runname='brownseds_agn', hdf5=True, load_extra_output=True)
        sresults_noagn, _, _, eout_noagn = load_prospector_data(None, objname=name, runname='brownseds_np', hdf5=True, load_extra_output=True)

        labx, laby, legend, txtupperright = False, False, False, False
        if ((i+3) % 3 == 0):
            laby = True
        if (i > 2):
            labx = True
        if i == 0:
            legend = True
        if i == 5:
            txtupperright = True

        ### plot (only AGN)
        sed_figure(sresults = [sresults_agn], extra_output = [eout_agn], 
                   ax = ax_agn[i], labels=[r'Prospector-$\alpha$'],labx=labx, laby=laby, 
                   legend=legend, objname=name, txtupperright=txtupperright)#, colors=['#1974D2'])
        sed_figure(sresults = [sresults_agn,sresults_noagn], extra_output = [eout_agn,eout_noagn], 
                   ax = ax_both[i], labels=['Model with AGN','Model without AGN'],labx=labx, laby=laby, 
                   legend=legend, objname=name, txtupperright=txtupperright)   

    ### draw some sweet dividing lines
    x1, y1 = 0.35, 0.52 # corner point
    x2, y2 = 0.97, 0.04 # draw lines to?
    lw = 1.0
    line = mpl.lines.Line2D((x1,x2),(y1,y1),transform=fig_agn.transFigure,color='black',lw=lw)
    line2 = mpl.lines.Line2D((x1,x1),(y1,y2),transform=fig_agn.transFigure,color='black',lw=lw)
    fig_agn.lines = line,line2,

    line = mpl.lines.Line2D((x1,x2),(y1,y1),transform=fig_both.transFigure,color='black',lw=lw)
    line2 = mpl.lines.Line2D((x1,x1),(y1,y2),transform=fig_both.transFigure,color='black',lw=lw)
    fig_both.lines = line,line2,

    ### I/O, cleanup
    fig_agn.tight_layout()
    fig_both.tight_layout()

    #fig.subplots_adjust(right=0.85,wspace=0.3,hspace=0.3,left=0.12)
    fig_agn.savefig(outfolder+'phot_agn_only.png',dpi=150)
    fig_both.savefig(outfolder+'phot_both.png',dpi=150)
    plt.close()

def sed_figure(colors = ['#9400D3','#FF420E'], sresults = None, extra_output = None,
               labels = ['spectrum (50th percentile)'], ax=None,
               model_photometry = True, main_color=['black'],
               fir_extra = False, ml_spec=False,transcurves=False,
               objname=None, txtupperright=None,
               ergs_s_cm=True, labx=True, laby=True, legend=False):
    """
    Plot the photometry for the model and data (with error bars), and
    plot residuals
    """

    ms = 5
    alpha = 0.8
    
    ### diagnostic text
    textx = 0.05
    texty = 0.95
    deltay = 0.06

    ### scale
    ax.set_yscale('log',nonposx='clip')
    ax.set_xscale('log',nonposx='clip',subsx=(1,3))
    ax.xaxis.set_minor_formatter(minorFormatter)
    ax.xaxis.set_major_formatter(majorFormatter)

    ### if we have multiple parts, color ancillary data appropriately
    if len(colors) > 1:
        main_color = colors

    #### iterate over things to plot
    for i,sample_results in enumerate(sresults):

        #### grab data for maximum probability model
        wave_eff, obsmags, obsmags_unc, modmags, chi, modspec, modlam = return_sedplot_vars(sample_results,extra_output[i],ergs_s_cm=ergs_s_cm)

        #### plot maximum probability model
        if model_photometry:
            ax.plot(wave_eff, modmags, color=colors[i], 
                      marker='o', ms=ms, linestyle=' ', alpha=alpha, 
                      markeredgewidth=0.7)

        ###### spectra for q50 + 5th, 95th percentile
        w = extra_output[i]['observables']['lam_obs']
        spec_pdf = np.zeros(shape=(len(w),3))

        for jj in xrange(len(w)): spec_pdf[jj,:] = np.percentile(extra_output[i]['observables']['spec'][jj,:],[5.0,50.0,95.0])
        
        sfactor = 3e18/w
        if ergs_s_cm:
            sfactor *= 3631*1e-23

        for z in xrange(3):
            spec_pdf[:,z] = smooth_spectrum(modlam*1e4,spec_pdf[:,z],250,minlam=1e3,maxlam=1e4)

        nz = modspec > 0
        ax.plot(modlam[nz], spec_pdf[nz,1]*sfactor[nz]/(sample_results['model'].params['zred'][0]+1), linestyle='-',
                  color=colors[i], alpha=0.9,zorder=-1,label = labels[i])  

        nz = spec_pdf[:,1] > 0
        ax.fill_between(w*(sample_results['model'].params['zred'][0]+1)/1e4, 
                          spec_pdf[:,0]*sfactor/(sample_results['model'].params['zred'][0]+1), 
                          spec_pdf[:,2]*sfactor/(sample_results['model'].params['zred'][0]+1),
                          color=colors[i],
                          alpha=0.3,zorder=-1)
        ### observations!
        if i == 0:
            xplot = wave_eff
            yplot = obsmags
            yerr = obsmags_unc

            positive_flux = obsmags > 0

            # PLOT OBSERVATIONS + ERRORS 
            ax.errorbar(xplot[positive_flux], yplot[positive_flux], yerr=yerr[positive_flux],
                          color='#545454', marker='o', label='observed', alpha=alpha, linestyle=' ',ms=ms,zorder=0)

        #### calculate and show reduced chi-squared
        chisq = np.sum(chi**2)
        ndof = np.sum(sample_results['obs']['phot_mask'])
        reduced_chisq = chisq/(ndof)

        ax.text(textx, texty-deltay*(i+1), r'$\chi^2$/N$_{\mathrm{phot}}$='+"{:.2f}".format(reduced_chisq),
              fontsize=9, ha='left',transform = ax.transAxes,color=main_color[i])
    
    # label fmir
    fmir_idx = extra_output[0]['extras']['parnames'] == 'fmir'
    fmir = extra_output[0]['extras']['q50'][fmir_idx][0]
    fmir_up = extra_output[0]['extras']['q84'][fmir_idx][0] - fmir
    fmir_do = fmir - extra_output[0]['extras']['q16'][fmir_idx][0]

    fmt = "{{0:{0}}}".format(".2f").format
    title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
    title = title.format(fmt(fmir), fmt(fmir_do), fmt(fmir_up))

    yt1, yt2 = 0.13, 0.05
    if txtupperright:
        yt1, yt2 = 0.9, 0.82

    ax.text(0.95, yt1, objname, fontsize=10, ha='right',transform = ax.transAxes,color='black')
    ax.text(0.95, yt2, "{0} = {1}".format(r'f$_{\mathrm{AGN,MIR}}$', title), fontsize=10, ha='right',transform = ax.transAxes,color='black')

    ### apply plot limits
    ax.set_xlim(0.1, 50)
    ymin, ymax = yplot[positive_flux].min()*0.5, yplot[positive_flux].max()*10
    if legend:
        ymax *=2
    ax.set_ylim(ymin, ymax)

    # legend
    # make sure not to repeat labels
    if legend:
        from collections import OrderedDict
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), 
                    loc=1, prop={'size':7},
                    scatterpoints=1,fancybox=True)
    if laby:
        if ergs_s_cm:
            ax.set_ylabel(r'$\nu f_{\nu}$ [erg/s/cm$^2$]')
        else:
            ax.set_ylabel(r'$\nu f_{\nu}$ [maggie Hz]')
    if labx:
        ax.set_xlabel(r'$\lambda_{\mathrm{obs}}$ [$\mu$m]')
    for tl in ax.get_xticklabels():tl.set_visible(False)

def return_sedplot_vars(sample_results, extra_output, nufnu=True, ergs_s_cm=False):

    '''
    if nufnu == True: return in units of nu * fnu. Else, return maggies.
    '''

    # observational information
    mask = sample_results['obs']['phot_mask']
    wave_eff = sample_results['obs']['wave_effective'][mask]
    obs_maggies = sample_results['obs']['maggies'][mask]
    obs_maggies_unc = sample_results['obs']['maggies_unc'][mask]

    # model information
    spec = extra_output['bfit']['spec']
    mu = extra_output['bfit']['mags'][mask]

    # output units
    if nufnu == True:
        factor = 3e18
        if ergs_s_cm:
            factor *= 3631*1e-23
        mu *= factor/wave_eff
        spec *= factor/extra_output['observables']['lam_obs']
        obs_maggies *= factor/wave_eff
        obs_maggies_unc *= factor/wave_eff

    # here we want to return
    # effective wavelength of photometric bands, observed maggies, observed uncertainty, model maggies, observed_maggies-model_maggies / uncertainties
    # model maggies, observed_maggies-model_maggies/uncertainties
    return wave_eff/1e4, obs_maggies, obs_maggies_unc, mu, \
    (obs_maggies-mu)/obs_maggies_unc, spec/(sample_results['model'].params['zred'][0]+1), \
    (1+sample_results['model'].params['zred'][0])*extra_output['observables']['lam_obs']/1e4


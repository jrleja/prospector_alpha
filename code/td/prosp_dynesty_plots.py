import numpy as np
import matplotlib.pyplot as plt
import os, copy, prosp_dutils
from dynesty import plotting as dyplot
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from prospector_io import load_prospector_data
from scipy.ndimage import gaussian_filter as norm_kde
from matplotlib import gridspec

plt.ioff() # don't pop up a window for each plot

# plotting variables
fs, tick_fs = 28, 22
obs_color = '#545454'

def subcorner(res, eout, parnames, outname=None, maxprob=False):
    """ wrapper around dyplot.cornerplot()
    adds in a star formation history and marginalized parameters
    for some key outputs (stellar mass, SFR, sSFR, half-mass time)
    """

    # write down some keywords
    title_kwargs = {'fontsize':fs*.7}
    label_kwargs = {'fontsize':fs*.7}

    if maxprob:
        truths = res['samples'][eout['sample_idx'][0],:]
    else:
        truths = None 

    # create dynesty plot
    # maximum probability solution in red
    fig, axes = dyplot.cornerplot(res, show_titles=True, labels=parnames, truths=truths,
                                  truth_color='purple',
                                  label_kwargs=label_kwargs, title_kwargs=title_kwargs)
    for ax in axes.ravel():
        ax.xaxis.set_tick_params(labelsize=tick_fs*.7)
        ax.yaxis.set_tick_params(labelsize=tick_fs*.7)
        
    # add SFH plot
    sfh_ax = fig.add_axes([0.72,0.425,0.25,0.25],zorder=32)
    add_sfh_plot([eout], fig,
                 main_color = ['black'],
                 ax_inset=sfh_ax,
                 text_size=4,lw=6)

    # create extra parameters
    axis_size = fig.get_axes()[0].get_position().size
    xs, ys = 0.44, 0.89
    xdelta, ydelta = axis_size[0]*1.6, axis_size[1]*2
    plotloc = 0
    eout_toplot = ['stellar_mass','sfr_100', 'ssfr_100', 'half_time', 'H alpha 6563', 'H alpha/H beta']
    not_log = ['half_time','H alpha/H beta']
    ptitle = [r'log(M$_*$)',r'log(SFR$_{\mathrm{100 Myr}}$)',
              r'log(sSFR$_{\mathrm{100 Myr}}$)',r't$_{\mathrm{half}}$ [Gyr]',
              r'log(EW$_{\mathrm{H \alpha}}$)',r'Balmer decrement']
    for jj, ename in enumerate(eout_toplot):

        # total obfuscated way to add in axis
        ax = fig.add_axes([xs+(jj%3)*xdelta, ys-(jj%2)*ydelta, axis_size[0], axis_size[1]])

        # pull out chain, quantiles
        weights = eout['weights']
        if 'H alpha' not in ename:
            pchain = eout['extras'][ename]['chain']
            qvalues = [eout['extras'][ename]['q16'],
                       eout['extras'][ename]['q50'],
                       eout['extras'][ename]['q84']]
        elif '6563' in ename:
            pchain = eout['obs']['elines'][ename]['ew']['chain']
            qvalues = [eout['obs']['elines'][ename]['ew']['q16'],
                       eout['obs']['elines'][ename]['ew']['q50'],
                       eout['obs']['elines'][ename]['ew']['q84']]
        else:
            pchain = eout['obs']['elines']['H alpha 6563']['flux']['chain'] / eout['obs']['elines']['H beta 4861']['flux']['chain']
            qvalues = dyplot._quantile(pchain,np.array([0.16, 0.50, 0.84]),weights=weights)

        if ename not in not_log:
            pchain = np.log10(pchain)
            qvalues = np.log10(qvalues)

        # complex smoothing routine to match dynesty
        bins = int(round(10. / 0.02))
        n, b = np.histogram(pchain, bins=bins, weights=weights,
                            range=[pchain.min(),pchain.max()])
        n = norm_kde(n, 10.)
        x0 = 0.5 * (b[1:] + b[:-1])
        y0 = n
        ax.fill_between(x0, y0, color='k', alpha = 0.6)

        # plot and show quantiles
        for q in qvalues: ax.axvline(q, ls="dashed", color='red')

        q_m = qvalues[1]-qvalues[0]
        q_p = qvalues[2]-qvalues[1]
        fmt = "{{0:{0}}}".format(".2f").format
        title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
        title = title.format(fmt(float(qvalues[1])), fmt(float(q_m)), fmt(float(q_p)))
        title = "{0}\n={1}".format(ptitle[jj], title)
        ax.set_title(title, va='bottom',**title_kwargs)
        ax.set_xlabel(ptitle[jj],**label_kwargs)

        # set range
        ax.set_xlim(pchain.min(),pchain.max())
        ax.set_ylim(0, 1.1 * np.max(n))
        ax.set_yticklabels([])
        ax.xaxis.set_major_locator(MaxNLocator(5))
        [l.set_rotation(45) for l in ax.get_xticklabels()]
        ax.xaxis.set_tick_params(labelsize=tick_fs*.7)

    fig.savefig('{0}.corner.png'.format(outname))
    plt.close(fig)

def transform_chain(flatchain, model):

    parnames = np.array(model.theta_labels())
    if flatchain.ndim == 1:
        flatchain = np.atleast_2d(flatchain)

    # turn fractional_dust1 into dust1
    if 'dust1_fraction' in model.free_params:
        d1idx, d2idx = model.theta_index['dust1_fraction'], model.theta_index['dust2']
        flatchain[:,d1idx] *= flatchain[:,d2idx]
        parnames[d1idx] = 'dust1'

    # turn z_fraction into sfr_fraction
    if 'z_fraction' in model.free_params:
        zidx = model.theta_index['z_fraction']
        flatchain[:,zidx] = prosp_dutils.transform_zfraction_to_sfrfraction(flatchain[:,zidx]) 
        parnames = np.core.defchararray.replace(parnames,'z_fraction','sfr_fraction')

    # rename mass_met
    if 'massmet' in model.free_params:
        midx = model.theta_index['massmet']
        parnames[midx.start] = 'logmass'
        parnames[midx.start+1] = 'logzsol'

    return flatchain.squeeze(), parnames

def add_sfh_plot(eout,fig,ax_loc=None,
                 main_color=None,tmin=0.01,
                 text_size=1,ax_inset=None,lw=1):
    """add a small SFH plot at ax_loc
    text_size: multiply font size by this, to accomodate larger/smaller figures
    """

    # set up plotting
    if ax_inset is None:
        if fig is None:
            ax_inset = ax_loc
        else:
            ax_inset = fig.add_axes(ax_loc,zorder=32)
    axfontsize=4*text_size

    xmin, ymin = np.inf, np.inf
    xmax, ymax = -np.inf, -np.inf
    for i, eout in enumerate(eout):
        
        #### load SFH
        t = eout['sfh']['t']
        perc = np.zeros(shape=(len(t),3))
        for jj in range(len(t)): perc[jj,:] = np.percentile(eout['sfh']['sfh'][:,jj],[16.0,50.0,84.0])

        #### plot SFH
        ax_inset.plot(t, perc[:,1],'-',color=main_color[i],lw=lw)
        ax_inset.fill_between(t, perc[:,0], perc[:,2], color=main_color[i], alpha=0.3)
        ax_inset.plot(t, perc[:,0],'-',color=main_color[i],alpha=0.3,lw=lw)
        ax_inset.plot(t, perc[:,2],'-',color=main_color[i],alpha=0.3,lw=lw)

        #### update plot ranges
        if 'tage' in eout['thetas'].keys():
            xmin = np.min([xmin,t.min()])
            xmax = np.max([xmax,t.max()])
            ymax = np.max([ymax,perc.max()])
            ymin = ymax*1e-4
        else:
            xmin = np.min([xmin,t.min()])
            xmax = np.max([xmax,t.max()])
            ymin = np.min([ymin,perc.min()])
            ymax = np.max([ymax,perc.max()])

    #### labels, format, scales !
    if tmin:
        xmin = tmin

    axlim_sfh=[xmax, xmin*3, ymin*.7, ymax*1.4]
    ax_inset.axis(axlim_sfh)
    ax_inset.set_ylabel(r'SFR [M$_{\odot}$/yr]',fontsize=axfontsize*3,labelpad=1.5*text_size)
    ax_inset.set_xlabel(r't$_{\mathrm{lookback}}$ [Gyr]',fontsize=axfontsize*3,labelpad=1.5*text_size)
    
    ax_inset.set_xscale('log',subsx=([3]))
    ax_inset.set_yscale('log',subsy=([3]))
    ax_inset.tick_params('both', length=lw*3, width=lw*.6, which='both',labelsize=axfontsize*3)
    for axis in ['top','bottom','left','right']: ax_inset.spines[axis].set_linewidth(lw*.6)

    ax_inset.xaxis.set_major_formatter(FormatStrFormatter('%2.2g'))
    ax_inset.yaxis.set_major_formatter(FormatStrFormatter('%2.2g'))

def return_sedplot_vars(res, eout, nufnu=True, ergs_s_cm=False):

    '''
    if nufnu == True: return in units of nu * fnu. Else, return maggies.
    '''

def sed_figure(outname = None,
               colors = ['#1974D2'], sresults = None, eout = None,
               labels = ['spectrum (50th percentile)'],
               model_photometry = True, main_color=['black'],
               ml_spec=False,transcurves=False,
               ergs_s_cm=True, add_sfh=False,
               **kwargs):
    """Plot the photometry for the model and data (with error bars), and
    plot residuals
        -- nondetections are plotted as downwards-pointing arrows
        -- pass in a list of [res], can iterate over them to plot multiple results
    good complimentary color for the default one is '#FF420E', a light red
    """

    # set up plot
    ms, alpha, fs, ticksize = 5, 0.8, 16, 12
    textx, texty, deltay = 0.02, .95, .04
    fig = plt.figure()
    gs = gridspec.GridSpec(2,1, height_ratios=[3,1])
    gs.update(hspace=0)
    phot, resid = plt.Subplot(fig, gs[0]), plt.Subplot(fig, gs[1])

    # if we have multiple parts, color ancillary data appropriately
    if len(colors) > 1:
        main_color = colors

    # iterate over results to plot
    for i,res in enumerate(sresults):

        # pull out data
        mask = res['obs']['phot_mask']
        phot_wave_eff = res['obs']['wave_effective'][mask]
        obsmags = res['obs']['maggies'][mask]
        obsmags_unc = res['obs']['maggies_unc'][mask]

        # model information
        modspec_bfit = eout[i]['obs']['spec'][0,:]
        modmags_bfit = eout[i]['obs']['mags'][0,mask]
        nspec = modspec_bfit.shape[0]
        spec_pdf = np.zeros(shape=(nspec,3))
        for jj in range(spec_pdf.shape[0]): spec_pdf[jj,:] = np.percentile(eout[i]['obs']['spec'][:,jj],[16.0,50.0,84.0])

        # units
        zred = res['model'].params['zred'][0]
        factor = 3e18
        if ergs_s_cm:
            factor *= 3631*1e-23
        
        # photometry
        modmags_bfit *= factor/phot_wave_eff
        obsmags *= factor/phot_wave_eff
        obsmags_unc *= factor/phot_wave_eff
        photchi = (obsmags-modmags_bfit)/obsmags_unc
        phot_wave_eff /= 1e4

        # spectra
        spec_pdf *= (factor/eout[i]['obs']['lam_obs']/(1+zred)).reshape(nspec,1)
        modspec_bfit *= factor/eout[i]['obs']['lam_obs']/(1+zred)
        modspec_lam = eout[i]['obs']['lam_obs']*(1+zred)/1e4
        
        # plot maximum probability model
        if model_photometry:
            phot.plot(phot_wave_eff, modmags_bfit, color=colors[i], 
                      marker='o', ms=ms, linestyle=' ', label = 'photometry, best-fit', alpha=alpha, 
                      markeredgecolor='k',**kwargs)
        
        resid.plot(phot_wave_eff, photchi, color=colors[i],
                 marker='o', linestyle=' ', label=labels[i], 
                 ms=ms,alpha=alpha,markeredgewidth=0.7,markeredgecolor='k',
                 **kwargs)        

        # model spectra
        yplt = spec_pdf[:,1]
        if ml_spec:
            yplt = modspec_bfit
        pspec = prosp_dutils.smooth_spectrum(modspec_lam*1e4,yplt,200,minlam=1e3,maxlam=1e5)
        nz = pspec > 0
        phot.plot(modspec_lam[nz], pspec[nz], linestyle='-',
                  color=colors[i], alpha=0.9,zorder=-1,label = labels[i],**kwargs)  
        phot.fill_between(modspec_lam[nz], spec_pdf[nz,0], spec_pdf[nz,2],
                          color=colors[i], alpha=0.3,zorder=-1)

        # calculate and show reduced chi-squared
        chisq = np.sum(photchi**2)
        ndof = mask.sum()
        reduced_chisq = chisq/(ndof)

        phot.text(textx, texty-deltay*(i+1), r'best-fit $\chi^2$/N$_{\mathrm{phot}}$='+"{:.2f}".format(reduced_chisq),
                  fontsize=10, ha='left',transform = phot.transAxes,color=main_color[i])

    # plot observations
    pflux = obsmags > 0
    phot.errorbar(phot_wave_eff[pflux], obsmags[pflux], yerr=obsmags_unc[pflux],
                  color=obs_color, marker='o', label='observed', alpha=alpha, linestyle=' ',ms=ms,
                  zorder=0,markeredgecolor='k')

    # limits
    xlim = (phot_wave_eff[pflux].min()*0.5,phot_wave_eff[pflux].max()*3)
    phot.set_xlim(xlim)
    resid.set_xlim(xlim)
    ymin, ymax = obsmags[pflux].min()*0.5, obsmags[pflux].max()*2

    # add transmission curves
    if transcurves:
        dyn = 10**(np.log10(ymin)+(np.log10(ymax)-np.log10(ymin))*0.2)
        for f in res['obs']['filters']: phot.plot(f.wavelength/1e4, f.transmission/f.transmission.max()*dyn+ymin,lw=1.5,color='0.3',alpha=0.7)

    # add in arrows for negative fluxes
    if pflux.sum() != len(obsmags):
        downarrow = [u'\u2193']
        y0 = 10**((np.log10(ymax) - np.log10(ymin))/20.)*ymin[0]
        for x0 in xplot[~positive_flux]: phot.plot(x0, y0, linestyle='none',marker=u'$\u2193$',markersize=16,alpha=alpha,mew=0.0,color=obs_color)
    phot.set_ylim(ymin, ymax)
    resid_ymax = np.abs(resid.get_ylim()).max()
    resid.set_ylim(-resid_ymax,resid_ymax)

    # redshift text
    phot.text(textx, texty, 'z='+"{:.2f}".format(zred),
              fontsize=10, ha='left',transform = phot.transAxes)
    
    # extra line
    resid.axhline(0, linestyle=':', color='grey')
    resid.yaxis.set_major_locator(MaxNLocator(5))

    # legend
    phot.legend(loc=1, prop={'size':8},
                scatterpoints=1,fancybox=True)
                
    # set labels
    resid.set_ylabel( r'$\chi$',fontsize=fs)
    if ergs_s_cm:
        phot.set_ylabel(r'$\nu f_{\nu}$ [erg/s/cm$^2$]',fontsize=fs)
    else:
        phot.set_ylabel(r'$\nu f_{\nu}$ [maggie Hz]',fontsize=fs)
    resid.set_xlabel(r'$\lambda_{\mathrm{obs}}$ [$\mu$m]',fontsize=fs)
    phot.set_yscale('log',nonposx='clip')
    phot.set_xscale('log',nonposx='clip')
    resid.set_xscale('log',nonposx='clip',subsx=(2,5))
    resid.xaxis.set_minor_formatter(FormatStrFormatter('%2.2g'))
    resid.xaxis.set_major_formatter(FormatStrFormatter('%2.2g'))
    resid.tick_params('both', pad=3.5, size=3.5, width=1.0, which='both',labelsize=ticksize)
    phot.tick_params('y', which='major', labelsize=ticksize)

    # add figures
    fig.add_subplot(phot)
    fig.add_subplot(resid)
    
    # set second x-axis (rest-frame wavelength)
    y1, y2=phot.get_ylim()
    x1, x2=phot.get_xlim()
    ax2=phot.twiny()
    ax2.set_xticks(np.arange(0,10,0.2))
    ax2.set_xlim(x1/(1+zred), x2/(1+zred))
    ax2.set_xlabel(r'$\lambda_{\mathrm{rest}}$ [$\mu$m]',fontsize=fs)
    ax2.set_ylim(y1, y2)
    ax2.set_xscale('log',nonposx='clip',subsx=(2,5))
    ax2.xaxis.set_minor_formatter(FormatStrFormatter('%2.2g'))
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%2.2g'))
    ax2.tick_params('both', pad=2.5, size=3.5, width=1.0, which='both',labelsize=ticksize)

    # remove ticks
    phot.set_xticklabels([])
    
    # add SFH 
    if add_sfh:
        sfh_ax = fig.add_axes([0.425,0.385,0.15,0.2],zorder=32)
        add_sfh_plot(eout, fig,
                     main_color = ['black'],
                     ax_inset=sfh_ax,
                     text_size=0.6,lw=1.5)

    if outname is not None:
        fig.savefig(outname, bbox_inches='tight', dpi=180)
        plt.close()

def add_inset_pdf(eout,ax,pars,pnames,text_size=1,lw=1):
    
    axfontsize=4*text_size

    for i, p in enumerate(pars):
        
        if p in eout['quantiles']['parnames']:
            idx = eout['quantiles']['parnames'] == p
            plot = eout['quantiles']['sample_chain'][:,idx]
            qvalues = [eout['quantiles']['q16'][idx],
                       eout['quantiles']['q50'][idx],
                       eout['quantiles']['q84'][idx]]
        else:
            idx = eout['extras']['parnames'] == p
            plot = np.log10(eout['extras']['flatchain'][:,idx])
            qvalues = [np.log10(eout['extras']['q16'][idx]),
                       np.log10(eout['extras']['q50'][idx]),
                       np.log10(eout['extras']['q84'][idx])]

        ### Plot histogram.
        n, b, p = ax[i].hist(plot, bins=50,
                          histtype="step",color='k',
                          range=[np.min(plot),np.max(plot)])

        for q in qvalues: ax[i].axvline(q, ls=':', color='k')

        # display quantiles
        q_m = qvalues[1]-qvalues[0]
        q_p = qvalues[2]-qvalues[1]

        # format quantile display
        fmt = "{{0:{0}}}".format(".2f").format
        title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
        title = title.format(fmt(qvalues[1][0]), fmt(q_m[0]), fmt(q_p[0]))
        ax[i].set_title(title,fontsize=axfontsize*4)
        ax[i].set_xlabel(pnames[i],fontsize=axfontsize*4,labelpad=2*text_size)

        # axes
        # set min/max
        ax[i].set_xlim(np.percentile(plot,0.5),
                    np.percentile(plot,99.5))
        ax[i].set_ylim(0, 1.1 * np.max(n))
        ax[i].set_yticklabels([])
        ax[i].xaxis.set_major_locator(MaxNLocator(4))
        #[l.set_rotation(45) for l in ax[i].get_xticklabels()]
        ax[i].xaxis.set_tick_params(labelsize=axfontsize*3)
        ax[i].yaxis.set_tick_params(labelsize=axfontsize*3)
        ax[i].tick_params('both', length=lw*3, width=lw*.6, which='major')


def make_all_plots(filebase=None,
                   outfolder=os.getenv('APPS')+'/prospector_alpha/plots/',
                   plt_summary=True,
                   plt_trace=True,
                   plt_corner=True,
                   plt_sed=True):
    """Makes basic dynesty diagnostic plots for a single galaxy.
    """

    # make sure the output folder exists
    if not os.path.isdir(outfolder):
        os.makedirs(outfolder)

    # load galaxy output.
    objname = filebase.split('/')[-1]
    try:
        res, powell_results, model, eout = load_prospector_data(filebase, hdf5=True)
    except IOError:
        print 'failed to load results for {0}'.format(objname)
        return
    
    # transform to preferred model variables
    res['chain'], parnames = transform_chain(res['chain'],res['model'])

    # mimic dynesty outputs
    res['logwt'] = np.log(res['weights'])+res['logz'][-1]
    res['logl'] = res['lnlikelihood']
    res['samples'] = res['chain']
    font_kwargs = {'fontsize': fs}

    # Plot a summary of the run.
    if plt_summary:
        print 'making SUMMARY plot'
        rfig, raxes = dyplot.runplot(res, mark_final_live=False, label_kwargs=font_kwargs)
        for ax in raxes:
            ax.xaxis.set_tick_params(labelsize=tick_fs)
            ax.yaxis.set_tick_params(labelsize=tick_fs)
            ax.yaxis.get_offset_text().set_size(fs)
        rfig.tight_layout()
        rfig.savefig(outfolder+objname+'_dynesty_summary.png',dpi=150)

    # Plot traces and 1-D marginalized posteriors.
    if plt_trace:
        print 'making TRACE plot'
        tfig, taxes = dyplot.traceplot(res, labels=parnames,label_kwargs=font_kwargs)
        for ax in taxes.ravel():
            ax.xaxis.set_tick_params(labelsize=tick_fs)
            ax.yaxis.set_tick_params(labelsize=tick_fs)
        tfig.tight_layout()
        tfig.savefig(outfolder+objname+'_dynesty_trace.png',dpi=130)

    # corner plot
    if plt_corner: 
        print 'making CORNER plot'
        subcorner(res, eout, parnames,outname=outfolder+objname)

    # sed plot
    if plt_sed:
        print 'making SED plot'
        pfig = sed_figure(sresults = [res], eout=[eout],
                          outname=outfolder+objname+'.sed.png')
        
def do_all(runname=None,**extras):
    """for a list of galaxies, make all plots
    the runname has to be accepted by generate_basenames
    extra arguments go to make_all_plots
    """

    filebase, param_basename, ancilname = prosp_dutils.generate_basenames(runname)
    for jj in xrange(len(filebase)):
        print 'iteration '+str(jj) 

        make_all_plots(filebase=filebase[jj],\
                       outfolder=os.getenv('APPS')+'/prospector_alpha/plots/'+runname+'/',
                       **extras)
    
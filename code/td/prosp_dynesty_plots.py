import numpy as np
import matplotlib.pyplot as plt
import os
from prosp_dutils import generate_basenames, smooth_spectrum, transform_zfraction_to_sfrfraction
from dynesty import plotting as dyplot
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, FuncFormatter
from prospector_io import load_prospector_data, find_all_prospector_results
from scipy.ndimage import gaussian_filter as norm_kde
from matplotlib import gridspec

plt.ioff() # don't pop up a window for each plot

# plotting variables
fs, tick_fs = 28, 22
obs_color = '#545454'

def subcorner(res, eout, parnames, outname=None, maxprob=False, boost=None):
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

    # change stellar mass, SFR, total mass, SFH by some boosting factor
    if boost is not None:
        # total mass
        for q in ['q50','q84','q16']: eout['thetas']['massmet_1'][q] = np.log10(10**eout['thetas']['massmet_1'][q]*boost)
        res['samples'][:,0] = np.log10(10**res['samples'][:,0] * boost)

        # stellar mass, SFR
        for q in ['q50','q84','q16','chain']:
            eout['extras']['sfr_100'][q] *= boost 
            eout['extras']['stellar_mass'][q] *= boost 
        eout['sfh']['sfh'] *= boost # SFH


    # create dynesty plot
    # maximum probability solution in red
    fig, axes = dyplot.cornerplot(res, show_titles=True, labels=parnames, truths=truths,
                                  truth_color='purple',
                                  label_kwargs=label_kwargs, title_kwargs=title_kwargs)
    for ax in axes.ravel():
        ax.xaxis.set_tick_params(labelsize=tick_fs*.7)
        ax.yaxis.set_tick_params(labelsize=tick_fs*.7)

    # extra parameters
    eout_toplot = ['stellar_mass','sfr_100', 'ssfr_100', 'avg_age', 'H alpha 6563', 'H alpha/H beta']
    not_log = ['half_time','H alpha/H beta']
    ptitle = [r'log(M$_*$)',r'log(SFR$_{\mathrm{100 Myr}}$)',
              r'log(sSFR$_{\mathrm{100 Myr}}$)',r'log(t$_{\mathrm{avg}}$) [Gyr]',
              r'log(EW$_{\mathrm{H \alpha}}$)',r'Balmer decrement']

    # either we create a new figure for extra parameters
    # or add to old figure
    # depending on dimensionality of model (and thus of the plot)
    if (axes.shape[0] < 6):

        # we usually don't have emission lines
        eout_toplot, ptitle = eout_toplot[:4], ptitle[:4]

        # generate fake results file for dynesty
        nsamp, nvar = eout['weights'].shape[0], len(eout_toplot)
        fres = {'samples': np.empty(shape=(nsamp,nvar)), 'weights': eout['weights']}
        for i in range(nvar): 
            the_chain = eout['extras'][eout_toplot[i]]['chain']
            if eout_toplot[i] in not_log:
                fres['samples'][:,i] = the_chain
            else:
                fres['samples'][:,i] = np.log10(the_chain)

        fig2, axes2 = dyplot.cornerplot(fres, show_titles=True, labels=ptitle, label_kwargs=label_kwargs, title_kwargs=title_kwargs)
       
        # add SFH plot
        sfh_ax = fig2.add_axes([0.7,0.7,0.25,0.25],zorder=32)
        add_sfh_plot([eout], fig2,
                     main_color = ['black'],
                     ax_inset=sfh_ax,
                     text_size=1.5,lw=2)
        fig2.savefig('{0}.corner.extra.png'.format(outname))
        plt.close(fig2)

    else:

        # add SFH plot
        sfh_ax = fig.add_axes([0.75,0.435,0.22,0.22],zorder=32)
        add_sfh_plot([eout], fig, main_color = ['black'], ax_inset=sfh_ax, text_size=2,lw=4)

        # create extra parameters
        axis_size = fig.get_axes()[0].get_position().size
        xs, ys = 0.44, 0.89
        xdelta, ydelta = axis_size[0]*1.6, axis_size[1]*2
        plotloc = 0
        for jj, ename in enumerate(eout_toplot):

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

            # logify. 
            if ename not in not_log:
                pchain = np.log10(pchain)
                qvalues = np.log10(qvalues)

            # make sure we're not producing infinities.
            # if we are, replace them with minimum.
            # if everything is infinity, skip and don't add the axis!
            # one failure mode here: if qvalues include an infinity!
            infty = ~np.isfinite(pchain)
            if infty.sum() == pchain.shape[0]:
                continue
            if infty.sum():
                pchain[infty] = pchain[~infty].min()

            # total obfuscated way to add in axis
            ax = fig.add_axes([xs+(jj%3)*xdelta, ys-(jj%2)*ydelta, axis_size[0], axis_size[1]])

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
        flatchain[:,zidx] = transform_zfraction_to_sfrfraction(flatchain[:,zidx]) 
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
        
        # create master time bin
        min_time = eout['sfh']['t'].min()
        max_time = eout['sfh']['t'].max()
        tvec = 10**np.linspace(np.log10(min_time),np.log10(max_time),num=50)

        # create median SFH
        perc = np.zeros(shape=(len(tvec),3))
        for jj in range(len(tvec)): 
            # nearest-neighbor 'interpolation'
            # exact answer for binned SFHs
            idx = np.abs(eout['sfh']['t'] - tvec[jj]).argmin(axis=-1)
            perc[jj,:] = dyplot._quantile(eout['sfh']['sfh'][np.arange(idx.shape[0]),idx],[0.16,0.50,0.84],weights=eout['weights'])

        #### plot SFH
        ax_inset.plot(tvec, perc[:,1],'-',color=main_color[i],lw=lw)
        ax_inset.fill_between(tvec, perc[:,0], perc[:,2], color=main_color[i], alpha=0.3)
        ax_inset.plot(tvec, perc[:,0],'-',color=main_color[i],alpha=0.3,lw=lw)
        ax_inset.plot(tvec, perc[:,2],'-',color=main_color[i],alpha=0.3,lw=lw)

        #### update plot ranges
        if 'tage' in eout['thetas'].keys():
            xmin = np.min([xmin,tvec.min()])
            xmax = np.max([xmax,tvec.max()])
            ymax = np.max([ymax,perc.max()])
            ymin = ymax*1e-4
        else:
            xmin = np.min([xmin,tvec.min()])
            xmax = np.max([xmax,tvec.max()])
            ymin = np.min([ymin,perc[perc>0].min()])
            ymax = np.max([ymax,perc.max()])

    #### labels, format, scales !
    xmin = np.min(tvec[tvec>0.01])
    ymin = np.clip(ymin,ymax*1e-5,np.inf)

    axlim_sfh=[xmax*1.01, xmin*1.0001, ymin*.7, ymax*1.4]
    ax_inset.axis(axlim_sfh)
    ax_inset.set_ylabel(r'SFR [M$_{\odot}$/yr]',fontsize=axfontsize*3,labelpad=1.5*text_size)
    ax_inset.set_xlabel(r't$_{\mathrm{lookback}}$ [Gyr]',fontsize=axfontsize*3,labelpad=1.5*text_size)
    
    ax_inset.xaxis.set_minor_formatter(FormatStrFormatter('%2.5g'))
    ax_inset.xaxis.set_major_formatter(FormatStrFormatter('%2.5g'))
    ax_inset.set_xscale('log',subsx=([3]))
    ax_inset.set_yscale('log',subsy=([3]))
    ax_inset.tick_params('both', length=lw*3, width=lw*.6, which='both',labelsize=axfontsize*3)
    for axis in ['top','bottom','left','right']: ax_inset.spines[axis].set_linewidth(lw*.6)

    ax_inset.xaxis.set_minor_formatter(FormatStrFormatter('%2.5g'))
    ax_inset.xaxis.set_major_formatter(FormatStrFormatter('%2.5g'))
    ax_inset.yaxis.set_major_formatter(FormatStrFormatter('%2.5g'))

def sed_figure(outname = None,
               colors = ['#1974D2'], sresults = None, eout = None,
               labels = ['spectrum (50th percentile)'],
               model_photometry = True, main_color=['black'],
               transcurves=False,
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
    textx, texty, deltay = 0.02, .95, .05
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
        modmags_bfit = eout[i]['obs']['mags'][0,mask]
        modspec_lam = eout[i]['obs']['lam_obs']
        nspec = modspec_lam.shape[0]
        try:
            spec_pdf = np.zeros(shape=(nspec,3))
            for jj in range(spec_pdf.shape[0]): spec_pdf[jj,:] = np.percentile(eout[i]['obs']['spec'][:,jj],[16.0,50.0,84.0])
        except:
            spec_pdf = np.stack((eout[i]['obs']['spec']['q16'],eout[i]['obs']['spec']['q50'],eout[i]['obs']['spec']['q84']),axis=1)

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
        spec_pdf *= (factor/modspec_lam/(1+zred)).reshape(nspec,1)
        modspec_lam = modspec_lam*(1+zred)/1e4
        
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
        pspec = smooth_spectrum(modspec_lam*1e4,yplt,200,minlam=1e3,maxlam=1e5)
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
        y0 = 10**((np.log10(ymax) - np.log10(ymin))/20.)*ymin
        for x0 in phot_wave_eff[~pflux]: phot.plot(x0, y0, linestyle='none',marker=u'$\u2193$',markersize=16,alpha=alpha,mew=0.5,mec='k',color=obs_color)
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
    phot.set_yscale('log',nonposy='clip')
    phot.set_xscale('log',nonposx='clip')
    resid.set_xscale('log',nonposx='clip',subsx=(2,5))
    resid.xaxis.set_minor_formatter(FormatStrFormatter('%2.4g'))
    resid.xaxis.set_major_formatter(FormatStrFormatter('%2.4g'))
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
    ax2.xaxis.set_minor_formatter(FormatStrFormatter('%2.4g'))
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%2.4g'))
    ax2.tick_params('both', pad=2.5, size=3.5, width=1.0, which='both',labelsize=ticksize)

    # remove ticks
    phot.set_xticklabels([])
    
    # add SFH 
    if add_sfh:
        sfh_ax = fig.add_axes([0.425,0.4,0.15,0.2],zorder=32)
        add_sfh_plot(eout, fig,
                     main_color = ['black'],
                     ax_inset=sfh_ax,
                     text_size=0.45,lw=1.13)

    if outname is not None:
        fig.savefig(outname, bbox_inches='tight', dpi=180)
        plt.close()

def make_all_plots(filebase=None,
                   outfolder=os.getenv('APPS')+'/prospector_alpha/plots/',
                   plt_summary=False,
                   plt_trace=False,
                   plt_corner=True,
                   plt_sed=True,
                   **opts):
    """Makes basic dynesty diagnostic plots for a single galaxy.
    """

    # make sure the output folder exists
    if not os.path.isdir(outfolder):
        os.makedirs(outfolder)

    # load galaxy output.
    objname = filebase.split('/')[-1]
    try:
        res, powell_results, model, eout = load_prospector_data(filebase, hdf5=True)
    except IOError,TypeError:
        print 'failed to load results for {0}'.format(objname)
        return

    if (res is None) or (eout is None):
        return

    # restore model
    if res['model'] is None:
        from prospect.models import model_setup
        # make filenames local
        for key in res['run_params']:
            if type(res['run_params'][key]) == unicode:
                if 'prospector_alpha' in res['run_params'][key]:
                    res['run_params'][key] = os.getenv('APPS')+'/prospector_alpha'+res['run_params'][key].split('prospector_alpha')[-1]
        pfile = model_setup.import_module_from_file(res['run_params']['param_file'])
        res['model'] = pfile.load_model(**res['run_params'])
        pfile = None
    
    # transform to preferred model variables
    res['chain'], parnames = transform_chain(res['chain'],res['model'])

    # mimic dynesty outputs
    res['logwt'] = np.log(res['weights'])+res['logz'][-1]
    res['logl'] = res['lnlikelihood']
    res['samples'] = res['chain']
    res['nlive'] = res['run_params']['nested_nlive_init']
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
        subcorner(res, eout, parnames,outname=outfolder+objname, **opts)

    # sed plot
    if plt_sed:
        print 'making SED plot'
        pfig = sed_figure(sresults = [res], eout=[eout],
                          outname=outfolder+objname+'.sed.png')
        
def do_all(runname=None,nobase=True,**extras):
    """for a list of galaxies, make all plots
    the runname has to be accepted by generate_basenames
    extra arguments go to make_all_plots
    """
    if nobase:
        filebase = find_all_prospector_results(runname)
    else:
        filebase, _, _ = prosp_dutils.generate_basenames(runname)
    for jj in range(len(filebase)):
        print 'iteration '+str(jj) 

        make_all_plots(filebase=filebase[jj],\
                       outfolder=os.getenv('APPS')+'/prospector_alpha/plots/'+runname+'/',
                       **extras)
    
import numpy as np
import matplotlib.pyplot as plt
import corner, os, copy, prosp_dutils
from prospect.io import read_results
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from prospect.models import model_setup
from magphys_plot_pref import jLogFormatter
from prospector_io import load_prospector_data
from dynesty.plotting import _quantile as wquant

plt.ioff() # don't pop up a window for each plot

obs_color = '#545454'

minorFormatter = jLogFormatter(base=10, labelOnlyBase=False)
majorFormatter = jLogFormatter(base=10, labelOnlyBase=True)

def subcorner(sample_results,  sps, model, extra_output, flatchain,
              outname=None, showpars=None,
              **kwargs):
    """
    Make a corner plot of the (thinned, latter) samples of the posterior
    parameter space.  Optionally make the plot only for a supplied subset
    of the parameters.
    """

    ### make title font slightly smaller
    title_kwargs = {'fontsize': 'medium'}

    ### transform to the "fun" variables!
    flatchain, parnames = transform_chain(flatchain,model)
    extents = return_extent(flatchain)

    fig = corner.corner(flatchain, labels = parnames, levels=[0.68,0.95,0.997], fill_contours=True, color='#0038A8',
                        quantiles=[0.16, 0.5, 0.84], verbose = False, hist_kwargs=dict(color='k'),
                        show_titles = True, plot_datapoints=False, title_kwargs=title_kwargs,
                        **kwargs)

    fig = add_to_corner(fig, sample_results, extra_output, sps, model, outname=outname,
                        maxprob=True, title_kwargs=title_kwargs)

    if outname is not None:
        fig.savefig('{0}.corner.png'.format(outname))
        plt.close(fig)
    else:
        return fig

def transform_chain(flatchain, model):

    parnames = np.array(model.theta_labels())
    if flatchain.ndim == 1:
        flatchain = np.atleast_2d(flatchain)

    ### turn fractional_dust1 into dust1
    if 'dust1_fraction' in model.free_params:
        d1idx, d2idx = model.theta_index['dust1_fraction'], model.theta_index['dust2']
        flatchain[:,d1idx] *= flatchain[:,d2idx]
        parnames[d1idx] = 'dust1'

    ### turn z_fraction into sfr_fraction
    if 'z_fraction' in model.free_params:
        zidx = model.theta_index['z_fraction']
        flatchain[:,zidx] = prosp_dutils.transform_zfraction_to_sfrfraction(flatchain[:,zidx]) 
        parnames = np.core.defchararray.replace(parnames,'z_fraction','sfr_fraction')

    return flatchain.squeeze(), parnames

def add_to_corner(fig, sample_results, extra_output, sps, model,outname=None,
                  maxprob=True,title_kwargs=None,twofigure=False):

    """
    adds in posterior distributions for specific parameters
    if we have truths, list them as text
    """
    
    plotquant = extra_output['extras'].get('flatchain',None)
    plotname  = extra_output['extras'].get('parnames',None)

    to_show = ['stellar_mass','sfr_100','ssfr_100','half_time']
    ptitle = [r'log(M$_*$)',r'log(SFR$_{\mathrm{100 Myr}}$)',
              r'log(sSFR$_{\mathrm{100 Myr}}$)',r't$_{\mathrm{half}}$ [Gyr]']

    showing = np.array([x in to_show for x in plotname])

    # extra text
    scale    = len(extra_output['quantiles']['parnames'])
    ttop     = 0.95-0.02*(12-scale)
    fs       = 24-(12-scale)
    
    # show maximum probability
    if maxprob:
        mprob,_ = transform_chain(extra_output['bfit']['maxprob_params'],model)
        maxprob_parnames = np.append(extra_output['quantiles']['parnames'],'lnprob')
        plt.figtext(0.75, ttop, 'best-fit',weight='bold',
                       horizontalalignment='left',fontsize=fs)
        for kk in xrange(len(maxprob_parnames)):
            if maxprob_parnames[kk] == 'mass':
               yplot = np.log10(mprob[kk])
            elif maxprob_parnames[kk] == 'lnprob':
                yplot = float(extra_output['bfit']['maxprob'])
            else:
                yplot = mprob[kk]

            # add parameter names if not covered by truths
            plt.figtext(0.84, ttop-0.02*(kk+1), maxprob_parnames[kk]+'='+"{:.2f}".format(yplot),
                   horizontalalignment='right',fontsize=fs)

    #### two options: add new figure, or combine with old
    if (scale < 6) or twofigure:

        # need flatchain + extents, ordered by to_show
        nshow = showing.sum()
        nchain = extra_output['extras']['flatchain'].shape[0]
        flatchain = np.empty(shape=(nchain,nshow))
        for i in xrange(nshow): 
            the_chain = extra_output['extras']['flatchain'][:,plotname == to_show[i]].squeeze()
            if to_show[i] == 'half_time':
                flatchain[:,i] = the_chain
            else:
                flatchain[:,i] = np.log10(the_chain)
        # extents
        extents = []
        for i in xrange(nshow):
            extents.append((np.percentile(flatchain[:,i],0.1),
                            np.percentile(flatchain[:,i],99.9)))

        fig2 = corner.corner(flatchain, labels=ptitle, levels=[0.68,0.95,0.997], fill_contours=True, color='#0038A8',
                             quantiles=[0.16, 0.5, 0.84], verbose = False, range = extents, hist_kwargs=dict(color='k'),
                             show_titles = True, plot_datapoints=False, title_kwargs=title_kwargs)
       
        #### add SFH plot
        sfh_ax = fig2.add_axes([0.7,0.7,0.25,0.25],zorder=32)
        add_sfh_plot([extra_output], fig2,
                     main_color = ['black'],
                     ax_inset=sfh_ax,
                     text_size=1.5,lw=2)
        fig2.savefig('{0}.corner.extra.png'.format(outname))

        plt.close(fig2)

    else:
        
        #### add SFH plot
        sfh_ax = fig.add_axes([0.72,0.425,0.25,0.25],zorder=32)
        add_sfh_plot([extra_output], fig,
                     main_color = ['black'],
                     ax_inset=sfh_ax,
                     text_size=3,lw=5)

        #### create my own axes here
        # size them using size of other windows
        axis_size = fig.get_axes()[0].get_position().size
        xs, ys = 0.44, 0.85
        xdelta, ydelta = axis_size[0]*1.6, axis_size[1]*1.7
        plotloc = 0

        for jj in xrange(len(plotname)):

            if showing[jj] == 0:
                continue

            ax = fig.add_axes([xs+(plotloc % 2)*xdelta, ys-(plotloc>1)*ydelta, axis_size[0], axis_size[1]])
            plotloc+=1

            if plotname[jj] == 'half_time':
                plot = plotquant[:,jj]
            else:
                plot = np.log10(plotquant[:,jj])
            plot = plot[np.isfinite(plot)]

            # Plot the histograms.
            n, b, p = ax.hist(plot, bins=50,
                              histtype="step",color='k',
                              range=[np.min(plot),np.max(plot)])

            # plot quantiles
            qvalues = np.log10([extra_output['extras']['q16'][jj],
                                extra_output['extras']['q50'][jj],
                                extra_output['extras']['q84'][jj]])

            if plotname[jj] == 'half_time':
                qvalues = 10**qvalues

            for q in qvalues:
                ax.axvline(q, ls="dashed", color='k')

            # display quantiles
            q_m = qvalues[1]-qvalues[0]
            q_p = qvalues[2]-qvalues[1]

            # format quantile display
            fmt = "{{0:{0}}}".format(".2f").format
            title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
            title = title.format(fmt(qvalues[1]), fmt(q_m), fmt(q_p))
            ax.set_title(title, va='bottom')
            ax.set_xlabel(ptitle[to_show.index(plotname[jj])],**title_kwargs)

            # axes
            # set min/max
            ax.set_xlim(np.percentile(plot,0.5),
                        np.percentile(plot,99.5))
            ax.set_ylim(0, 1.1 * np.max(n))
            ax.set_yticklabels([])
            ax.xaxis.set_major_locator(MaxNLocator(5))
            [l.set_rotation(45) for l in ax.get_xticklabels()]

    return fig

def add_sfh_plot(exout,fig,ax_loc=None,
                 main_color=None,tmin=0.01,
                 text_size=1,ax_inset=None,lw=1):
    
    '''
    add a small SFH plot at ax_loc
    text_size: multiply font size by this, to accomodate larger/smaller figures
    '''

    # set up plotting
    if ax_inset is None:
        if fig is None:
            ax_inset = ax_loc
        else:
            ax_inset = fig.add_axes(ax_loc,zorder=32)
    axfontsize=4*text_size

    xmin, ymin = np.inf, np.inf
    xmax, ymax = -np.inf, -np.inf
    for i, extra_output in enumerate(exout):
        
        #### load SFH
        t = extra_output['extras']['t_sfh']
        perc = np.zeros(shape=(len(t),3))
        for jj in xrange(len(t)): perc[jj,:] = wquant(extra_output['extras']['sfh'][jj,:],[0.16,0.5,0.84],weights=extra_output['weights'])

        #### plot SFH
        ax_inset.plot(t, perc[:,1],'-',color=main_color[i],lw=lw)
        ax_inset.fill_between(t, perc[:,0], perc[:,2], color=main_color[i], alpha=0.3)
        ax_inset.plot(t, perc[:,0],'-',color=main_color[i],alpha=0.3,lw=lw)
        ax_inset.plot(t, perc[:,2],'-',color=main_color[i],alpha=0.3,lw=lw)

        #### update plot ranges
        if 'tage' in extra_output['quantiles']['parnames']:
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
    ax_inset.set_ylabel(r'SFR [M$_{\odot}$/yr]',fontsize=axfontsize*3,labelpad=2*text_size)
    ax_inset.set_xlabel(r't$_{\mathrm{lookback}}$ [Gyr]',fontsize=axfontsize*3,labelpad=2*text_size)
    
    ax_inset.set_xscale('log')
    ax_inset.xaxis.set_tick_params(labelsize=axfontsize*3)

    ax_inset.set_yscale('log')
    ax_inset.yaxis.set_tick_params(labelsize=axfontsize*3)

    ax_inset.tick_params('both', length=lw*3, width=lw*.6, which='major')
    for axis in ['top','bottom','left','right']: ax_inset.spines[axis].set_linewidth(lw*.6)

    ax_inset.xaxis.set_major_formatter(FormatStrFormatter('%2.2g'))
    ax_inset.yaxis.set_major_formatter(FormatStrFormatter('%2.2g'))

def plot_sfh_fast(tau,tage,mass,tuniv=None):

    '''
    version of plot_sfh, but only for FAST outputs
    this means no chain sampling, and simple tau rather than delayed tau models
    if we specify tuniv, return instead (tuniv-t)
    '''
    
    t=np.linspace(0,tage,num=50)
    sfr = np.exp(-t/tau)
    sfr_int = mass/np.sum(sfr * (t[1]-t[0])*1e9)  # poor man's integral, integrate f*yrs
    sfr = sfr * sfr_int

    if tuniv:
        t = tuniv-t
        t = t[::-1]

    return t,sfr

def return_extent(chain):    
    
    # set range
    extents = []
    for ii in xrange(chain.shape[1]): extents.append(tuple(np.percentile(chain[:,ii],[0.05,99.95]).tolist()))
    
    return extents

def get_cmap(N,cmap='nipy_spectral'):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.'''

    import matplotlib.cm as cmx
    import matplotlib.colors as colors

    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap=cmap) 
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color


def show_chain(sample_results,legend=True, outname=None):
    
    '''
    plot the MCMC chain for all parameters
    '''
    
    ### names and dimensions
    parnames = np.array(sample_results['model'].theta_labels())
    chain = sample_results['chain']
    lnprob = sample_results['lnprobability']
    nwalkers, nsteps, npars = chain.shape
    nplot = npars+2 # also plot lnprob

    ### plot preferences
    nboldchain = 3
    alpha_bold = 0.9
    alpha = 0.05
    lw_bold = 2
    lw = 1
    cmap = get_cmap(nboldchain,cmap='brg')

    ### plot geometry
    ndim = len(parnames)
    ny = 4
    nx = int(np.ceil(nplot/float(ny)))
    nx = 5
    sz = np.array([nx,ny])
    factor = 3.2           # size of one side of one panel
    lbdim = 0.0 * factor   # size of margins
    whspace = 0.0 * factor         # w/hspace size
    plotdim = factor * sz + factor *(sz-1)* whspace
    dim = 2*lbdim + plotdim

    ### create plots
    fig, axarr = plt.subplots(ny, nx, figsize = (dim[0]*1.2, dim[1]))
    fig.subplots_adjust(wspace=0.5,hspace=0.2)
    axarr = np.ravel(axarr)
    
    ### remove some plots to make room for KL divergence
    off = [13,14,18,19]
    #off = [7,8]
    [axarr[o].axis('off') for o in off] # turn off 
    axarr = np.delete(axarr,off) # remove from array

    ### check for stuck walkers
    # must visit at least 10 unique points in lnprobability
    outliers = np.full(nwalkers,False,dtype=bool)
    for k in xrange(nwalkers):
        ncall = np.unique(sample_results['lnprobability'][k,:]).shape[0]
        if ncall <= 10:
            outliers[k] = True

    # remove stuck walkers
    nstuck = outliers.sum()
    print str(nstuck)+' stuck walkers found for '+sample_results['run_params'].get('objname','galaxy')
    if nstuck:
        chain = chain[~outliers,:,:]
        lnprob = lnprob[~outliers,:]
        nwalkers = chain.shape[0]

    ### plot chain in each parameter
    for i, ax in enumerate(axarr):
        if i < npars: # we're plotting variables
            for k in xrange(nboldchain,nwalkers): ax.plot(chain[k,:,i],'-', alpha=alpha, lw=lw,zorder=-1)
            for k in xrange(nboldchain): ax.plot(chain[k,:,i],'-', alpha=alpha_bold, lw=lw_bold, color=cmap(k),zorder=1)

            ax.set_ylabel(parnames[i])
            ax.set_ylim(chain[:,:,i].min()*0.95,chain[:,:,i].max()*1.05)

        elif i == npars: # we're plotting lnprob
            for k in xrange(nboldchain,nwalkers): ax.plot(lnprob[k,:],'-', alpha=alpha, lw=lw,zorder=-1)
            for k in xrange(nboldchain): ax.plot(lnprob[k,:],'-', alpha=alpha_bold, lw=lw_bold, color=cmap(k),zorder=1)

            ax.set_ylabel('ln(probability)')
            ax.set_ylim(lnprob.min()*0.95,lnprob.max()*1.05)

        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.set_xlabel('iteration')

    ### add KL divergence
    kl_ax = fig.add_axes([0.68, 0.1, 0.24, 0.35])
    cmap = get_cmap(npars)
    for i in xrange(npars): 
        kl_ax.plot(sample_results['kl_iteration'],np.log10(sample_results['kl_divergence'][:,i]),
                   'o',label=parnames[i],color=cmap(i),lw=1.5,linestyle='-',alpha=0.6)

    kl_ax.set_ylabel('log(KL divergence)')
    kl_ax.set_xlabel('iteration')
    kl_ax.set_xlim(0,nsteps*1.1)

    kl_div_lim = sample_results['run_params'].get('convergence_kl_threshold',0.018)
    kl_ax.axhline(np.log10(kl_div_lim), linestyle='--', color='red',lw=2,zorder=1)

    if legend:
        kl_ax.legend(prop={'size':5},ncol=2,numpoints=1,markerscale=0.7)

    if outname is not None:
        plt.savefig(outname+'.chain.png', bbox_inches='tight',dpi=110)
        plt.close()

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
    spec = copy.copy(extra_output['bfit']['spec'])
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

def sed_figure(outname = None,
               colors = ['#1974D2'], sresults = None, extra_output = None,
               labels = ['model spectrum'],
               model_photometry = True, main_color=['black'],
               fir_extra = False, ml_spec=False,transcurves=False,
               ergs_s_cm=True, add_sfh=False,
               **kwargs):
    """
    Plot the photometry for the model and data (with error bars), and
    plot residuals
    good complimentary color for the default one is '#FF420E', a light red
    """

    ms = 5
    alpha = 0.8
    
    #### set up plot
    fig = plt.figure()
    gs = mpl.gridspec.GridSpec(2,1, height_ratios=[3,1])
    gs.update(hspace=0)
    phot, res = plt.Subplot(fig, gs[0]), plt.Subplot(fig, gs[1])

    ### diagnostic text
    textx = 0.02
    texty = 0.95
    deltay = 0.05

    ### if we have multiple parts, color ancillary data appropriately
    if len(colors) > 1:
        main_color = colors

    #### iterate over things to plot
    for i,sample_results in enumerate(sresults):

        #### grab data for maximum probability model
        wave_eff, obsmags, obsmags_unc, modmags, chi, modspec, modlam = return_sedplot_vars(sample_results,extra_output[i],ergs_s_cm=ergs_s_cm)

        ### observations!
        if i == 0:
            xplot = wave_eff
            yplot = obsmags
            yerr = obsmags_unc

            positive_flux = obsmags > 0

            # PLOT OBSERVATIONS + ERRORS 
            phot.errorbar(xplot[positive_flux], yplot[positive_flux], yerr=yerr[positive_flux],
                          color=obs_color, marker='o', label='observed', alpha=alpha, linestyle=' ',ms=ms,
                          zorder=0,markeredgecolor='k')

        #### plot maximum probability model
        if model_photometry:
            phot.plot(wave_eff[positive_flux], modmags[positive_flux], color=colors[i], 
                      marker='o', ms=ms, linestyle=' ', label = 'model photometry', alpha=alpha, 
                      markeredgecolor='k',**kwargs)
        
        res.plot(wave_eff, chi, color=colors[i],
                 marker='o', linestyle=' ', label=labels[i], markeredgecolor='k',
                 ms=ms,alpha=alpha,markeredgewidth=0.7,**kwargs)        

        ###### spectra for q50 + 5th, 95th percentile
        w = extra_output[i]['observables']['lam_obs']
        spec_pdf = np.zeros(shape=(len(w),3))

        for jj in xrange(len(w)): spec_pdf[jj,:] = np.percentile(extra_output[i]['observables']['spec'][jj,:],[5.0,50.0,95.0])
        for pp in range(spec_pdf.shape[1]):
            spec_pdf[:,pp] = prosp_dutils.smooth_spectrum(modlam*1e4,spec_pdf[:,pp],200,minlam=1e3,maxlam=1e5)

        sfactor = 3e18/w
        if ergs_s_cm:
            sfactor *= 3631*1e-23
        nz = modspec > 0
        if ml_spec:
            phot.plot(modlam[nz], modspec[nz], linestyle='-',
                      color=colors[i], alpha=0.9,zorder=-1,label = 'model spectrum',**kwargs)
        else:
            phot.plot(modlam[nz], spec_pdf[nz,1]*sfactor[nz]/(sample_results['model'].params['zred'][0]+1), linestyle='-',
                      color=colors[i], alpha=0.9,zorder=-1,label = labels[i],**kwargs)  

        nz = spec_pdf[:,1] > 0
        phot.fill_between(w*(sample_results['model'].params['zred'][0]+1)/1e4, 
                          spec_pdf[:,0]*sfactor/(sample_results['model'].params['zred'][0]+1), 
                          spec_pdf[:,2]*sfactor/(sample_results['model'].params['zred'][0]+1),
                          color=colors[i],
                          alpha=0.3,zorder=-1)

        #### calculate and show reduced chi-squared
        chisq = np.sum(chi**2)
        ndof = np.sum(sample_results['obs']['phot_mask'])
        reduced_chisq = chisq/(ndof)

        phot.text(textx, texty-deltay*(i+1), r'best-fit $\chi^2$/N$_{\mathrm{phot}}$='+"{:.2f}".format(reduced_chisq),
              fontsize=10, ha='left',transform = phot.transAxes,color=main_color[i])

    xlim = (min(xplot)*0.5,25)

    ### apply plot limits
    phot.set_xlim(xlim)
    res.set_xlim(xlim)
    ymin, ymax = yplot[positive_flux].min()*0.5, yplot[positive_flux].max()*8

    #### add transmission curves
    if transcurves:
        dyn = 10**(np.log10(ymin)+(np.log10(ymax)-np.log10(ymin))*0.2)
        for f in sample_results['obs']['filters']: phot.plot(f.wavelength/1e4, f.transmission/f.transmission.max()*dyn+ymin,lw=1.5,color='0.3',alpha=0.7)
    phot.set_ylim(ymin, ymax)

    # if we have negatives:
    if positive_flux.sum() != len(obsmags):
        downarrow = [u'\u2193']
        y0 = 10**((np.log10(phot.get_ylim()[1]) - np.log10(phot.get_ylim()[0]))/20.)*phot.get_ylim()[0]
        for x0 in xplot[~positive_flux]: phot.plot(x0, y0, linestyle='none',marker=u'$\u2193$',markersize=16,alpha=alpha,mew=0.5,mec='k',color=obs_color)

    #### add RGB image
    '''
    try:
        imgname = os.getenv('APPS')+'/prospector_alpha/data/brownseds_data/rgb/'+sresults[0]['run_params']['objname'].replace(' ','_')+'.png'
        import matplotlib.image as mpimg
        img = mpimg.imread(imgname)
        ax_inset2 = fig.add_axes([0.46,0.34,0.15,0.15],zorder=32)
        ax_inset2.imshow(img)
        ax_inset2.set_axis_off()
    except IOError:
        print 'no RGB image'
    '''
    ### plot truths

    #### TEXT, FORMATTING, LABELS
    z_txt = sresults[0]['model'].params['zred'][0]
    phot.text(textx, texty, 'z='+"{:.2f}".format(z_txt),
              fontsize=10, ha='left',transform = phot.transAxes)

    # extra line
    res.axhline(0, linestyle=':', color='grey')
    res.yaxis.set_major_locator(MaxNLocator(5))

    # legend
    # make sure not to repeat labels
    from collections import OrderedDict
    handles, labels = phot.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    phot.legend(by_label.values(), by_label.keys(), 
                loc=1, prop={'size':8},
                scatterpoints=1,fancybox=True)
                
    # set labels
    fs, ls = 14, 12
    res.set_ylabel( r'$\chi$',fontsize=fs)
    if ergs_s_cm:
        phot.set_ylabel(r'$\nu f_{\nu}$ [erg/s/cm$^2$]',fontsize=fs)
    else:
        phot.set_ylabel(r'$\nu f_{\nu}$ [maggie Hz]',fontsize=fs)
    res.set_xlabel(r'$\lambda_{\mathrm{obs}}$ [$\mu$m]',fontsize=fs)
    phot.set_yscale('log',nonposx='clip')
    phot.set_xscale('log',nonposx='clip')
    res.set_xscale('log',nonposx='clip',subsx=(2,5))
    res.xaxis.set_minor_formatter(FormatStrFormatter('%2.2g'))
    res.xaxis.set_major_formatter(FormatStrFormatter('%2.2g'))
    res.tick_params('both', pad=3.5, size=3.5, width=1.0, which='both',labelsize=ls)
    phot.tick_params('y', which='major', labelsize=ls)


    # clean up and output
    fig.add_subplot(phot)
    fig.add_subplot(res)
    
    # set second x-axis
    y1, y2=phot.get_ylim()
    x1, x2=phot.get_xlim()
    ax2=phot.twiny()
    ax2.set_xticks(np.arange(0,10,0.2))
    ax2.set_xlim(x1/(1+z_txt), x2/(1+z_txt))
    ax2.set_xlabel(r'$\lambda_{\mathrm{rest}}$ [$\mu$m]',fontsize=fs)
    ax2.set_ylim(y1, y2)
    res.set_xscale('log',nonposx='clip',subsx=(2,5))
    res.xaxis.set_minor_formatter(FormatStrFormatter('%2.2g'))
    res.xaxis.set_major_formatter(FormatStrFormatter('%2.2g'))
    res.tick_params('both', pad=3.5, size=3.5, width=1.0, which='both',labelsize=ls)
    
    # remove ticks
    phot.set_xticklabels([])
    
    #### add SFH 
    if add_sfh:
        sfh_ax = fig.add_axes([0.381,0.39,0.18,0.24],zorder=32)
        add_sfh_plot(extra_output, fig,
                     main_color = ['black'],
                     ax_inset=sfh_ax,
                     text_size=0.7,lw=1.5)

    #### add insets
    if False:
        inset_ax = [fig.add_axes([0.63,0.71,0.11,0.14],zorder=32), fig.add_axes([0.76,0.71,0.11,0.14],zorder=32)]
        add_inset_pdf(extra_output[0],inset_ax,['stellar_mass','logzsol'],
                      [r'log(M/M$_{\odot}$)',r'log(Z/Z$_{\odot}$)'],text_size=0.5)

    if outname is not None:
        fig.savefig(outname, bbox_inches='tight', dpi=200)
        plt.close()

    #os.system('open '+outname)

def add_inset_pdf(extra_output,ax,pars,pnames,text_size=1,lw=1):
    
    axfontsize=4*text_size

    for i, p in enumerate(pars):
        
        if p in extra_output['quantiles']['parnames']:
            idx = extra_output['quantiles']['parnames'] == p
            plot = extra_output['quantiles']['sample_chain'][:,idx]
            qvalues = [extra_output['quantiles']['q16'][idx],
                       extra_output['quantiles']['q50'][idx],
                       extra_output['quantiles']['q84'][idx]]
        else:
            idx = extra_output['extras']['parnames'] == p
            plot = np.log10(extra_output['extras']['flatchain'][:,idx])
            qvalues = [np.log10(extra_output['extras']['q16'][idx]),
                       np.log10(extra_output['extras']['q50'][idx]),
                       np.log10(extra_output['extras']['q84'][idx])]

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
                   extra_output=None,
                   outfolder=os.getenv('APPS')+'/prospector_alpha/plots/',
                   sample_results=None,
                   param_name=None,
                   plt_chain=True,
                   plt_corner=True,
                   plt_sed=True):

    '''
    Driver. Loads output, makes all plots for a given galaxy.
    '''

    # make sure the output folder exists
    if not os.path.isdir(outfolder):
        os.makedirs(outfolder)

    if sample_results is None:
        try:
            sample_results, powell_results, model, extra_output = load_prospector_data(filebase, hdf5=True, load_extra_output=True)
        except TypeError:
            return
    else: # if we already have sample results, but want powell results
        try:
            _, powell_results, model, extra_output = load_prospector_data(filebase,no_sample_results=True, hdf5=True)
        except TypeError:
            return  

    run_params = model_setup.get_run_params(param_file=param_name)
    sps = model_setup.load_sps(**run_params)

    # BEGIN PLOT ROUTINE
    print 'MAKING PLOTS FOR ' + filebase.split('/')[-1] + ' in ' + outfolder
    objname = sample_results['run_params'].get('objname','galaxy')

    # chain plot
    flatchain = prosp_dutils.chop_chain(sample_results['chain'],**sample_results['run_params'])
    if plt_chain: 
        print 'MAKING CHAIN PLOT'

        show_chain(sample_results,outname=outfolder+objname)

    # corner plot
    if plt_corner: 
        print 'MAKING CORNER PLOT'
        subcorner(sample_results, sps, copy.deepcopy(sample_results['model']),
                  extra_output,flatchain,outname=outfolder+objname)

    # sed plot
    if plt_sed:
        print 'MAKING SED PLOT'
        
        # FAST fit?
        try:
            sample_results['run_params']['fastname']
            fast=1
        except:
            fast=0

        # plot
        pfig = sed_figure(sresults = [sample_results], extra_output=[extra_output],
                          outname=outfolder+objname+'.sed.png')
        
def plot_all_driver(runname=None,**extras):

    '''
    for a list of galaxies, make all plots
    '''

    filebase, parm_basename, ancilname=prosp_dutils.generate_basenames(runname)
    for jj in xrange(len(filebase)):
        print 'iteration '+str(jj) 

        make_all_plots(filebase=filebase[jj],\
                       outfolder=os.getenv('APPS')+'/prospector_alpha/plots/'+runname+'/',
                       param_name=parm_basename[jj],
                       **extras)
    
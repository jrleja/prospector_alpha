import numpy as np
import matplotlib.pyplot as plt
import os, hickle, td_io, prosp_dutils
from prospector_io import load_prospector_data
from scipy.ndimage import gaussian_filter as norm_kde
from matplotlib.ticker import FormatStrFormatter
import vis_params, vis_expsfh_params
from sedpy.observate import load_filters
from dynesty.plotting import _quantile as quantile
from matplotlib.ticker import MaxNLocator
from scipy.stats import uniform
import matplotlib as mpl
from copy import copy

trans = {
         'logmass': 'log(M/M$_{\odot}$)', 
         'ssfr_100': r'log(sSFR/yr$^{-1}$)',
         'dust2': r'diffuse dust', 
         'logzsol': r'log(Z/Z$_{\odot}$)', 
         'mean_age': r'stellar age [Gyr]',
         'dust_index': 'attenuation\ncurve',
         'dust1_fraction': 'birth-cloud\ndust', 
         'duste_qpah': 'PAH strength',
         'duste_gamma': 'hot dust\nfraction',
         'duste_umin': 'typical radiation\nintensity',
         'fagn': r'log(AGN fraction)',
         'agn_tau': r'AGN optical depth'
        }
fs_global = 10
lw = 2
alpha_posterior = 0.4
colors = {'samples':'k', 
          'prior': '#FF420E',
          'data': '#9400D3',
          'truth': '#1974D2'}

limits = {
         'ssfr_100':(-13,-8), 
         'fagn': (-3,-0.5)
         }

def collate_data(runname, filename=None, regenerate=False, **opts):
    """we need chains for all parameters (not SFH) + [sSFR 100, mass-weighted age]
    and observations, and spectra
    """

    ### if it's already made, load it and give it back
    # else, start with the making!
    if os.path.isfile(filename) and (regenerate == False):
        with open(filename, "r") as f:
            outdict=hickle.load(f)
        return outdict

    ### define output containers
    out = {'pars':{'mean_age':[],'ssfr_100':[]}, 'weights': [], 'mod_obs':[], 'data': [], 'filters':[]}

    ### fill them up
    out['names'] =  np.arange(1,11).astype(str)
    for i, name in enumerate(out['names']):
        
        try:
            res, _, mod, eout = load_prospector_data(None,runname=runname,objname=name)
            out['weights'] += [eout['weights']]
        except TypeError:
            print name +' is not available'
            continue

        # grab all theta chains
        for j,key in enumerate(mod.theta_labels()):
            # no SFHs
            if 'zfraction' in key:
                continue
            if key not in out['pars'].keys():
                out['pars'][key] = []
            if key == 'fagn':
                samp = np.log10(res['chain'][eout['sample_idx'],j])
            else:
                samp = res['chain'][eout['sample_idx'],j]
            out['pars'][key] += [samp]
        out['pars']['dust1_fraction'][-1] = out['pars']['dust1_fraction'][-1] * out['pars']['dust2'][-1]

        # mean age
        out['pars']['mean_age'] += [eout['extras']['avg_age']['chain'].tolist()]
        # use (SFR / total mass formed) as sSFR, easy to calculate analytically
        ssfr_100 = eout['extras']['ssfr_100']['chain'] 
        smass = eout['extras']['stellar_mass']['chain'] / (10**res['chain'][eout['sample_idx'],mod.theta_index['logmass']]).squeeze()
        out['pars']['ssfr_100'] += [np.log10(ssfr_100*smass).tolist()]

        # grab data
        if i == 0:
            sps = vis_params.load_sps(**vis_params.run_params)
        mod = vis_params.load_model(**vis_params.run_params)
        mod.params.update(res['obs']['true_params'])
        mod.params['nebemlineinspec'] = np.atleast_1d(True)
        res['obs']['true_spec'], _, _ = mod.mean_model(mod.theta, res['obs'], sps=sps)
        tobs = {
               'maggies': res['obs']['maggies'], 
               'maggies_unc': res['obs']['maggies_unc'],
               'wave_effective': res['obs']['wave_effective'],
               'spectrum': res['obs']['true_spec']
               }
        out['data'] += [tobs]

        # grab truths
        # rename dust
        if i == 0:
            out['truths'] = res['obs']['true_params']
            out['truths']['dust1_fraction'] *= out['truths']['dust2']
            out['truths']['fagn'] = np.log10(out['truths']['fagn'])

            # calculate sSFR, mean age for true model
            masses = vis_params.zfrac_to_masses(logmass=res['obs']['true_params']['logmass'], 
                                                z_fraction=res['obs']['true_params']['z_fraction'], 
                                                agebins=mod.params['agebins'])
            time_per_bin = np.diff(10**mod.params['agebins'], axis=-1)[:,0]
            age_in_bin = np.sum(10**mod.params['agebins'],axis=-1)/2.

            # fudge total mass into stellar mass by using Leja+15 mass-loss formula
            # this avoids a second model call
            sfr = masses[0] / time_per_bin[0]
            out['truths']['ssfr_100'] = np.log10(sfr / masses.sum())
            out['truths']['mean_age'] = ((age_in_bin[:,None] * masses).mean() / masses.mean())/1e9

        # grab observables
        out['mod_obs'] += [eout['obs']]
        out['filters'] += [vis_params.find_filters(i+1)]

    ### dump files and return
    hickle.dump(out,open(filename, "w"))

    return out

def do_all(runname='vis', outfolder=None,**opts):

    if outfolder is None:
        outfolder = os.getenv('APPS') + '/prospector_alpha/plots/'+runname+'/vis_plots/'
        if not os.path.isdir(outfolder):
            os.makedirs(outfolder)
            os.makedirs(outfolder+'data/')

    data = collate_data(runname,filename=outfolder+'data/dat.h5',**opts)
    for i in range(len(data['filters'])): data['filters'][i] = load_filters(data['filters'][i])

    if runname == 'vis':
        data['mod'] = vis_params.load_model(**vis_params.run_params)
    else:
        data['mod'] = vis_expsfh_params.load_model(**vis_expsfh_params.run_params)
    plot_addfilts(data,outfolder,runname)

def plot_addfilts(data,outfolder,runname):

    # loop over fits
    names_to_loop = ['1']+data['names'][:-1].tolist()
    xlim, ylim = [.09,30], [1e-8,7e-6]
    ssfr_prior = None
    for i, name in enumerate(names_to_loop):

        # show posterior?
        posterior = True
        if i == 0:
            posterior = False
            idx = 0
        else:
            idx = i-1

        # move wavelength
        if i == 6:
            xlim[1] = 550

        fig, sedax, stellax, dustax, dustemax, agnax = make_fig(phot=data['filters'][idx],posterior=posterior)

        # plot SED
        plot_sed(sedax,data['mod_obs'][idx],data['data'][idx],data['filters'][idx],xlim,ylim,posterior=posterior)

        # plot PDFs
        labels = [['logmass','ssfr_100','mean_age','logzsol'], \
                  ['dust2','dust1_fraction','dust_index'], \
                  ['duste_qpah','duste_gamma','duste_umin'], \
                  ['fagn','agn_tau']]
        axes = [stellax,dustax,dustemax,agnax]
        for par, ax in zip(labels,axes):
            for k, (p, a) in enumerate(zip(par,ax)):
                samp = np.array(data['pars'][p][idx])
                max = None
                if posterior:
                    max = plot_pdf(a,samp,data['weights'][idx])
                if (p == 'ssfr_100') & (i == 0):
                    ssfr_prior = samp
                plot_prior(a,data['mod'],p,max,runname,ssfr_prior=ssfr_prior)
                a.axvline(data['truths'][p], linestyle='-', color=colors['truth'],lw=lw,zorder=1)
                a.set_xlabel(trans[p],fontsize=fs_global-1)

        fig.subplots_adjust(hspace=0.00,wspace=0.0)
        if i == 0:
            name = '0'
        plt.savefig(outfolder+name+'.png',dpi=150)
        plt.close()

def make_fig(posterior=True,phot=None):

    # plot geometry
    fig = plt.figure(figsize=(12, 7))
    sedax = fig.add_axes([0.07,0.1,0.39,0.6])

    delx, dely = 0.04, 0.105
    sizex, sizey = 0.08, 0.14
    ystart = 0.805
    stellax = [fig.add_axes([0.665+(sizex+delx)*(j-1),ystart,sizex,sizey]) for j in range(4)]
    dustax = [fig.add_axes([0.74+(sizex+delx)*(j-1),ystart-dely-sizey,sizex,sizey]) for j in range(3)]
    dustemax = [fig.add_axes([0.74+(sizex+delx)*(j-1),ystart-2*(dely+sizey),sizex,sizey]) for j in range(3)]
    agnax = [fig.add_axes([0.82+(sizex+delx)*(j-1),ystart-3*(dely+sizey),sizex,sizey]) for j in range(2)]

    # add labels
    xt,yt = 0.535,0.84
    fig.text(xt-0.065,yt, 'Stellar\nparameters',color='black',fontsize=fs_global+2,weight='semibold',ha='center')
    fig.text(xt,yt-dely-sizey-0.015, 'Dust\nattenuation\nparameters',color='black',fontsize=fs_global+2,weight='semibold',ha='center')
    fig.text(xt,yt-2*(dely+sizey)-0.015, 'Dust\nemission\nparameters',color='black',fontsize=fs_global+2,weight='semibold',ha='center')
    fig.text(xt+0.07,yt-3*(dely+sizey)-0.015, 'AGN\nparameters',color='black',fontsize=fs_global+2,weight='semibold',ha='center')

    # add headers
    xt, yt = 0.79, 0.975
    delx = 0.06
    fig.text(xt-2*delx+0.01,yt, 'key:',fontsize=fs_global+2,weight='semibold',color='k')
    fig.text(xt-delx,yt, 'prior',fontsize=fs_global+2,weight='semibold',color=colors['prior'])
    fig.text(xt,yt, 'truth',fontsize=fs_global+2,weight='semibold',color=colors['truth'])
    if posterior:
        fig.text(xt+delx,yt, 'posterior',fontsize=fs_global+2,weight='semibold',color=colors['samples'],alpha=alpha_posterior)

    # add photometry
    xt, yt = 0.03, 0.95
    dely = 0.037
    if phot is not None:
        fnames = [f.name for f in phot]
        sdss = ['sdss_u','sdss_g','sdss_r','sdss_i','sdss_z','sdss_u0','sdss_g0','sdss_r0','sdss_i0','sdss_z0']
        match = ''.join([f.split('_')[1][0] if f in fnames else '' for f in sdss])
        if len(match) > 0:
            fig.text(xt,yt, 'optical: SDSS '+r'${0}$'.format("".join(match)),
                     fontsize=fs_global+3,color='green')

        twomass = ['twomass_J','twomass_H','twomass_Ks']
        match = ''.join([f.split('_')[-1] if f in fnames else '' for f in twomass])
        if len(match) > 0:
            fig.text(xt,yt-dely, 'NIR: 2MASS '+r'${0}$'.format("".join(match)),
                     fontsize=fs_global+3,color='#ee7600')

        galex = ['galex_FUV','galex_NUV']
        match = ','.join([f[-3:] if f in fnames else '' for f in galex])
        if len(match) > 1:
            fig.text(xt,yt-2*dely, 'UV: GALEX '+r'${0}$'.format("".join(match)),
                     fontsize=fs_global+3,color='#3232ff')

        irac = ['spitzer_irac_ch1','spitzer_irac_ch2','spitzer_irac_ch3','spitzer_irac_ch4']
        match = ''.join([f[-1] if f in fnames else '' for f in irac])
        if len(match) > 0:
            fig.text(xt,yt-3*dely, 'MIR: IRAC '+r'${0}$'.format(",".join(match)),
                     fontsize=fs_global+3,color='red')

        wise = ['wise_w1','wise_w2','wise_w3','wise_w4']
        match = ''.join([f[-1] if f in fnames else '' for f in wise])
        if len(match) > 0:
            fig.text(xt,yt-4*dely, '        WISE '+r'${0}$'.format(",".join(match)),
                     fontsize=fs_global+3,color='red')

        hpacs = ['herschel_pacs_70','herschel_pacs_100','herschel_pacs_160']
        match = ','.join([f.split('_')[-1] if f in fnames else '' for f in hpacs])
        if len(match) > 2:
            fig.text(xt,yt-5*dely, 'FIR: Herschel PACS/'+r'${0}$'.format("".join(match)),
                     fontsize=fs_global+3,color='#9b0000')

        hspire = ['herschel_spire_250','herschel_spire_350','herschel_spire_500']
        match = ','.join([f[-3:] if f in fnames else '' for f in hspire])
        if len(match) > 2:
            fig.text(xt,yt-6*dely, '       Herschel SPIRE/'+r'${0}$'.format("".join(match)),
                     fontsize=fs_global+3,color='#9b0000')

    return fig, sedax, stellax, dustax, dustemax, agnax

def plot_sed(ax,obs,truths,filters,xlim,ylim,truth=True,posterior=True):

    # plot info
    main_color = 'black'
    ms, alpha, fs, ticksize = 8, 1.0, 16, 12
    textx, texty, deltay = 0.02, .95, .05

    # pull out data
    phot_wave_eff = copy(truths['wave_effective'])
    obsmags = copy(truths['maggies'])
    obsmags_unc = copy(truths['maggies_unc'])
    modspec_lam = copy(obs['lam_obs'])
    spec_pdf = np.stack((obs['spec']['q16'],obs['spec']['q50'],obs['spec']['q84']),axis=1)
    spec_truth = copy(truths['spectrum'])

    # do units for photometry, spectra
    zred, factor = 0.0001, 3e18 * 3631*1e-23
    obsmags *= factor/phot_wave_eff
    obsmags_unc *= factor/phot_wave_eff
    phot_wave_eff /= 1e4
    spec_pdf *= (factor/modspec_lam/(1+zred))[:,None]
    spec_truth *= (factor/modspec_lam/(1+zred))
    modspec_lam = modspec_lam*(1+zred)/1e4
    
    # plot model spectra
    pspec = prosp_dutils.smooth_spectrum(modspec_lam*1e4,spec_pdf[:,1],4000,minlam=1e3,maxlam=2e4)
    tspec = prosp_dutils.smooth_spectrum(modspec_lam*1e4,spec_truth,4000,minlam=1e3,maxlam=2e4)

    nz = pspec > 0
    if posterior:
        ax.plot(modspec_lam[nz], pspec[nz], linestyle='-',
                  color=colors['samples'], alpha=0.9,zorder=-1,lw=lw)  
        ax.fill_between(modspec_lam[nz], spec_pdf[nz,0], spec_pdf[nz,2],
                          color=colors['samples'], alpha=alpha_posterior,zorder=-1,label='posterior')
    if truth:
        ax.plot(modspec_lam[nz], tspec[nz], linestyle='-',
                  color=colors['truth'], alpha=0.9,zorder=-1,label = 'truth',lw=lw)  

    # plot data
    pflux = obsmags > 0
    ax.errorbar(phot_wave_eff[pflux], obsmags[pflux], yerr=obsmags_unc[pflux],
                  color=colors['data'], marker='o', label='observed\nphotometry', alpha=alpha, 
                  linestyle=' ',ms=ms, zorder=0,markeredgecolor='k')

    # limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # add transmission curves
    # currently don't plot Herschel 500
    dyn = 10**(np.log10(ylim[0])+(np.log10(ylim[1])-np.log10(ylim[0]))*0.04)
    for f in filters: 
        if f.wave_effective/1e4 < 400:
            ax.plot(f.wavelength/1e4, f.transmission/f.transmission.max()*dyn+ylim[0],lw=lw,color=colors['data'],alpha=0.7)

    # legend
    ax.legend(loc=2, prop={'size':fs_global-2},
              scatterpoints=1,fancybox=True)
                
    # set labels
    ax.set_ylabel(r'$\nu f_{\nu}$',fontsize=fs)
    ax.set_xlabel(r'$\lambda_{\mathrm{obs}}$ [$\mu$m]',fontsize=fs)
    ax.set_yscale('log',nonposx='clip')
    ax.set_xscale('log',nonposx='clip',subsx=(2,5))
    ax.xaxis.set_minor_formatter(FormatStrFormatter('%3.3g'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%3.3g'))
    ax.tick_params(pad=3.5, size=3.5, width=1.0, which='both',labelsize=ticksize)

def plot_pdf(ax,samples,weights):

    # smoothing routine from dynesty
    bins = int(round(10. / 0.02))
    n, b = np.histogram(samples, bins=bins, weights=weights, range=[samples.min(),samples.max()],density=True)
    n = norm_kde(n, 10.)
    x0 = 0.5 * (b[1:] + b[:-1])
    y0 = n
    ax.fill_between(x0, y0/y0.max(), color=colors['samples'], alpha=alpha_posterior)

    # error bars
    qvalues = quantile(samples,np.array([0.16, 0.50, 0.84]),weights=weights)
    q_m = qvalues[1]-qvalues[0]
    q_p = qvalues[2]-qvalues[1]
    fmt = "{{0:{0}}}".format(".1f").format
    title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
    title = title.format(fmt(float(qvalues[1])), fmt(float(q_m)), fmt(float(q_p)))
    ax.set_title(title, va='center', fontsize=fs_global-2)

    return y0.max()

def plot_prior(ax,mod,par,max,runname,nsamp=100000,ssfr_prior=None):
    """plot prior by using information from Prospector prior object
    """
    
    # for some of these we custom sample a prior
    # for most, we use the built-in Prospector prior object
    parsamp = None
    if (par == 'ssfr_100'):
        prior = ssfr_prior
    elif (par == 'mean_age'):

        if runname == 'vis':

            # grab zfraction prior, sample
            logmass = 1 # doesn't matter, but needs definition
            zprior = mod._config_dict['z_fraction']['prior']
            agebins = mod.params['agebins']
            mass = np.zeros(shape=(agebins.shape[0],nsamp))

            # convert to mass in bins
            for n in range(nsamp): mass[:,n] = vis_params.zfrac_to_masses(logmass=logmass, z_fraction=zprior.sample(), agebins=agebins)

            # final conversion
            # convert to sSFR or mean age
            time_per_bin = 10**agebins[0,1] - 10**agebins[0,0]
            age_in_bin = np.sum(10**agebins,axis=-1)/2.
            prior = ((age_in_bin[:,None] * mass).mean(axis=0) / mass.mean(axis=0))/1e9

        elif runname == 'vis_expsfh':

            tage_prior = mod._config_dict['tage']['prior']
            logtau_prior = mod._config_dict['logtau']['prior']

            tage_samp = tage_prior.distribution.rvs(size=nsamp,*tage_prior.args,loc=tage_prior.loc,scale=tage_prior.scale)
            tau_samp = 10**logtau_prior.distribution.rvs(size=nsamp,*logtau_prior.args,loc=logtau_prior.loc,scale=logtau_prior.scale)
            prior = np.array([prosp_dutils.linexp_decl_sfh_avg_age(ta,tau) for (ta,tau) in zip(tage_samp,tau_samp)])

    elif par == 'dust1_fraction':

        # sample, then multiply dust2 by dust1_fraction prior 
        d2_prior = mod._config_dict['dust2']['prior']
        d1_prior = mod._config_dict['dust1_fraction']['prior']

        prior = d2_prior.distribution.rvs(size=nsamp,*d2_prior.args,loc=d2_prior.loc,scale=d2_prior.scale) * \
                d1_prior.distribution.rvs(size=nsamp,*d1_prior.args,loc=d1_prior.loc,scale=d1_prior.scale)

    elif par == 'fagn':

        parsamp = np.array([limits[par][0], limits[par][1]])
        fagn_lim = np.log10(mod._config_dict['fagn']['prior'].range)
        priorsamp = np.repeat(1./(fagn_lim[1]-fagn_lim[0]),2)

    else:
        prior = mod._config_dict[par]['prior']

        # sample PDF at regular intervals
        parsamp = np.linspace(prior.range[0],prior.range[1],nsamp)
        priorsamp = prior.distribution.pdf(parsamp,*prior.args,loc=prior.loc, scale=prior.scale)


    # build our own prior histogram if we don't have fancy built-in Prospector objects
    if parsamp is None:
        if par == 'ssfr_100':
            bins = int(round(10. / 0.02))
            n, b = np.histogram(prior, bins=bins, range=[prior.min(),prior.max()],density=True)
            priorsamp = norm_kde(n, 10.)
            parsamp = 0.5 * (b[1:] + b[:-1])
        else:
            priorsamp, bins = np.histogram(prior,bins=20,density=True,range=limits.get(par,None))
            parsamp = (bins[1:]+bins[:-1])/2.

    # plot prior
    if max is None:
        max = priorsamp.max()

    ax.plot(parsamp,priorsamp/max,color=colors['prior'],lw=lw,linestyle='--',label='prior')

    if par in limits.keys():
        ax.set_xlim(limits[par])

    # simple y-axis labels
    ax.yaxis.set_major_formatter(FormatStrFormatter('%i'))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))

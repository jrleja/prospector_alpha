import numpy as np
import matplotlib.pyplot as plt
import uvj_old_params as pfile
import os, pickle, hickle, td_io, prosp_dutils
from prospector_io import load_prospector_data
from scipy.ndimage import gaussian_filter as norm_kde
from matplotlib.ticker import FormatStrFormatter
import matplotlib.colors as colors
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import entropy
from dynesty.plotting import _quantile as wquantile
from scipy.interpolate import interp1d
import copy
from simulate_sfh_prior import draw_from_prior

minlogssfr, maxlogssfr = -13, -8
ssfr_lim = -10.5
bin_min = 10 # minimum bin count to show up on plot
nbins_3dhst = 20
kl_vmin, kl_vmax = 0, 2
plt.ioff()

def collate_data(runname, prior, filename=None, regenerate=False, **opts):
    
    ### if it's already made, load it and give it back
    # else, start with the making!
    if os.path.isfile(filename) and (regenerate == False):
        with open(filename, "r") as f:
            outdict=pickle.load(f)
            return outdict

    ### define output containers
    out = {'names':[],'weights':[],'log':{'ssfr_100':True,'avg_age':True},'pars':{},'chains':{},'kld':{},'lnlike':[]}
    entries = ['dust_index','ssfr_100','dust2','avg_age','logzsol']
    for e in entries: 
        out['pars'][e] = []
        out['chains'][e] = []
        out['kld'][e] = []
    out['uv'], out['vj'] = [], []

    ### fill them up
    out['names'] = [str(i+1) for i in range(625)]
    for i, name in enumerate(out['names']):
        res, _, _, eout = load_prospector_data(None,runname='uvj',objname='uvj_'+name)
        uv, vj = pfile.return_uvj(int(name))
        out['uv'] += [uv]
        out['vj'] += [vj]

        # if we haven't fit this yet (rerun all broken ones at some point)
        if eout is None:
            out['weights'] += [None]
            out['lnlike'] += [None]
            for key in out['pars'].keys(): 
                out['pars'][key] += [np.nan]
                out['chains'][key] += [None]
                out['kld'][key] += [None]
            print 'failed to load '+str(i)
            continue

        print 'loaded '+str(i)
        out['weights'] += [eout['weights']]
        out['lnlike'] += [res['lnlikelihood'].max()]
        for key in out['pars'].keys():
            # parameters and chains
            if key in eout['thetas'].keys():
                out['pars'][key] += [eout['thetas'][key]['q50']]
                idx = res['theta_labels'].index(key)
                out['chains'][key] += [res['chain'][eout['sample_idx'],idx].flatten()]
                out['kld'][key] += [kld(res['chain'][:,idx],res['weights'],prior[key])]
            else:
                out['pars'][key] += [eout['extras'][key]['q50']]
                out['chains'][key] += [eout['extras'][key]['chain'].flatten()]
                out['kld'][key] += [kld(np.log10(eout['extras'][key]['chain']),eout['weights'],np.log10(prior[key]))]

    ### dump files and return
    pickle.dump(out,open(filename, "w"))

    return out

def load_priors(pars,outloc,ndraw=100000,regenerate_prior=False):
    """ return ``ndraw'' samples from the prior 
    works for any model theta and many SFH parameters
    """

    mod = pfile.load_model(**pfile.run_params)

    # if we have an analytical function, sample at regular intervals across the function
    # otherwise, we sample numerically
    out = {}
    sfh_prior = draw_from_prior(pfile,outloc,ndraw=ndraw, sm=1.0, regenerate=regenerate_prior)
    for par in pars:
        out[par] = {}
        if par in mod.free_params:
            out[par] = mod._config_dict[par]['prior']
        elif (par == 'ssfr_100'):
            out[par] = sfh_prior['ssfr']
        elif (par == 'avg_age'):
            out[par] = sfh_prior['mwa']
        else:
            print 1/0

    return out

def kld(chain,weight,prior):
    nbins = 50
    if type(prior) is type(np.array([])):
        pdf_prior, bins = make_kl_bins(np.array(prior), nbins=nbins)
    else:
        bins = np.linspace(prior.range[0],prior.range[1], nbins+1)
        pdf_prior = prior.distribution.pdf(bins[1:]-(bins[1]-bins[0])/2.,*prior.args,loc=prior.loc, scale=prior.scale)
    pdf, _ = np.histogram(chain,bins=bins,weights=weight)
    kld_out = entropy(pdf,qk=pdf_prior) 
    return kld_out

def make_kl_bins(chain, nbins=10, weights=None):
    """Create bins with an ~equal number of data points in each 
    when there are empty bins, the KL divergence is undefined 
    this adaptive binning scheme avoids that problem
    """
    sorted = np.sort(chain)
    nskip = np.floor(chain.shape[0]/float(nbins)).astype(int)-1
    bins = sorted[::nskip]
    bins[-1] = sorted[-1]  # ensure the maximum bin is the maximum of the chain
    assert bins.shape[0] == nbins+1
    pdf, bins = np.histogram(chain, bins=bins,weights=weights)

    return pdf, bins

def do_all(runname='uvj', outfolder=None, regenerate_prior=False, **opts):

    if outfolder is None:
        outfolder = os.getenv('APPS') + '/prospector_alpha/plots/'+runname+'/uvj_plots/'
        if not os.path.isdir(outfolder):
            os.makedirs(outfolder)
            os.makedirs(outfolder+'data/')

    # global plotting parameters
    pars = ['ssfr_100','dust2','massmet_2','avg_age']# ,'dust_index'
    mock_pars = copy.copy(pars)
    mock_pars[mock_pars.index('massmet_2')] = 'logzsol'
    plabels = ['log(sSFR)', r'$\tau_{\mathrm{diffuse}}$', r'log(Z/Z$_{\odot}$)', r'log(stellar age/Gyr)'] # 'dust index'
    plabels_avg = ['log(<sSFR>)', r'<$\tau_{\mathrm{diffuse}}$>', r'log(<Z/Z$_{\odot}$>)', r'log(<stellar age>/Gyr)'] # 'dust index'
    lims = [(-14,-8.2),(0,2.4),(-1.99,0.2),(0.0,10)] # (-2.2,0.4)

    # mocks
    priors = load_priors(mock_pars+['dust_index'], outfolder+'data/prior.h5',regenerate_prior=regenerate_prior)
    uvj_mock = collate_data(runname,priors,filename=outfolder+'data/dat.h5',**opts)
    plot_mock_maps(uvj_mock, mock_pars, plabels, lims, outfolder+'mock_maps.png')
    plot_mock_maps(uvj_mock, mock_pars, plabels, lims, outfolder+'mock_kld.png',priors=priors,cmap='Reds')

    # plots from 3dhst uvj_mock
    # load uvj_mock, hard-coded
    with open('/Users/joel/code/python/prospector_alpha/plots/td_delta/fast_plots/data/fastcomp.h5', "r") as f:
        threedhst_data = hickle.load(f)
    threedhst_data['kld']['avg_age'] = threedhst_data['kld']['mwa']
    threedhst_data['fast']['uvj_dust_prosp'] = threedhst_data['fast']['uvj_dust_prosp'].astype(bool)

    plot_3d_maps(threedhst_data,pars,plabels_avg,lims,outfolder+'3dhst_kld_maps.png',kld=True,cmap='Reds')
    plot_3d_maps(threedhst_data,pars,plabels_avg,lims,outfolder+'3dhst_maps.png')
    plot_3d_hists(threedhst_data,pars,plabels,lims,outfolder+'3dhst_histograms.png')

    # quiescent fractions
    # both mocks and 3D-HST uvj_mock
    fig, ax = plt.subplots(1,2, figsize=(8,4))
    plot_mock_qfrac(uvj_mock,ax[0])
    plot_3dhst_qfrac(threedhst_data,ax[1])
    plt.tight_layout()
    plt.savefig(outfolder+'quiescent_map.png', dpi=150)
    plt.close()

    # for real uvj_mock: what are UVJ-quiescent but actually-starforming galaxies doing?
    plot_interloper_properties(threedhst_data, outfolder+'3dhst_interlopers.png',**opts)

def plot_interloper_properties(dat,outname,sederrs=False,density=False,**opts):
    """ distribution in mass, sSFR, metallicity, dust, age
    and average SEDs
    """

    # setting up plot geometry
    fig, axes = plt.subplots(1,5, figsize=(10,6))
    fig.subplots_adjust(bottom=0.65,top=0.9)
    sedax = fig.add_axes([0.4,0.1,0.5,0.4])

    # setting up plot options
    minmass = 10
    lims = [(minmass,12),(-14,-8),(0,2.4),(-2,0.2),(0,10)]
    nbins = 20
    histopts = {'drawstyle':'steps-mid','alpha':0.9, 'lw':2, 'linestyle': '-'}
    sedopts = {'fmt': 'o-', 'alpha':0.8,'lw':1, 'elinewidth':2.0,'capsize':2.0,'capthick':2}
    colors = ['#a32000', '#ff613a']
    fs = 12

    # defining plot parameters
    pars = ['stellar_mass','ssfr_100','dust2','massmet_2','avg_age']
    plabels = [r'log(M/M$_{\odot}$)','sSFR [100 Myr]', r'$\tau_{\mathrm{diffuse}}$', r'log(Z/Z$_{\odot}$)', r'<age>/Gyr'] # 'dust index'

    # defining indexes of 'true' quiescent and 'star-forming' quiescent
    sf_idx = (dat['fast']['uvj_prosp'] == 3) & (dat['prosp']['ssfr_100']['q50'] > ssfr_lim) & (dat['prosp']['stellar_mass']['q50'] > minmass)
    qu_idx = (dat['fast']['uvj_prosp'] == 3) & (dat['prosp']['ssfr_100']['q50'] < ssfr_lim) & (dat['prosp']['stellar_mass']['q50'] > minmass)
    ntot = float(sf_idx.sum()+qu_idx.sum())

    # figure text
    xs, dx, ys, dy = 0.03, 0.03, 0.45, 0.035
    weight = 'semibold'
    fig.text(xs,ys,'selected as UVJ-quiescent',fontsize=fs,weight=weight)
    fig.text(xs,ys-dy,r'and log(M/M$_{\odot}$)>'+"{:.1f}".format(minmass),fontsize=fs,weight=weight)
    fig.text(xs+dx,ys-2*dy,'log(sSFR)<{0:.1f} ({1:.1f}%)'.format(ssfr_lim,qu_idx.sum()/ntot*100),fontsize=fs,color=colors[0])
    fig.text(xs+dx,ys-3*dy,'log(sSFR)>{0:.1f} ({1:.1f}%)'.format(ssfr_lim,sf_idx.sum()/ntot*100),fontsize=fs,color=colors[1])

    # plot histograms
    for i, ax in enumerate(axes):
        data = dat['prosp'][pars[i]]['q50']

        x, hist1, hist2 = hist(data,qu_idx, sf_idx, nbins, lims[i],density=density)
        ax.plot(x,hist1,color=colors[0], **histopts)
        ax.plot(x,hist2,color=colors[1], **histopts)
        ax.set_ylim(0.0,ax.get_ylim()[1]*1.2)

        # titles and labels
        ax.set_xlim(lims[i])
        ax.set_xlabel(plabels[i],fontsize=fs*1.3)
        ax.set_yticklabels([])


    # pull out SED info
    restlam, mag, sn, ids = dat['phot_restlam'], dat['phot_mag'], dat['phot_sn'], dat['phot_ids']-1
    sf_idx, qu_idx = np.where(sf_idx)[0], np.where(qu_idx)[0]
    sf_match, qu_match = np.in1d(ids,sf_idx), np.in1d(ids,qu_idx)

    # pull out fluxes & errors, normalize by stellar mass
    fill_val = -1
    flux_norm = 10**(-mag/2.5) / 10**dat['prosp']['stellar_mass']['q50'][ids]
    flux_norm[np.isnan(flux_norm)] = fill_val
    errs = flux_norm / sn

    # generate edges of wavelength bins
    # then calculate median & dispersion in flux
    bins = np.array(discretize(restlam[sf_match | qu_match],40))
    midbins = (bins[1:] + bins[:-1])/2.
    sf, sf_eup, sf_edo, qu, qu_eup, qu_edo = [[] for i in range(6)]
    for i in range(bins.shape[0]-1):
        sf_idx = (restlam[sf_match] > bins[i]) & (restlam[sf_match] < bins[i+1])
        if sf_idx.sum() == 0:
            sf += [fill_val]
            sf_eup += [fill_val]
            sf_edo += [fill_val]
        else:
            med, eup, edo = np.nanpercentile(flux_norm[sf_match][sf_idx],[50,84,16])
            sf += [med]
            sf_eup += [eup]
            sf_edo += [edo]

        qu_idx = (restlam[qu_match] > bins[i]) & (restlam[qu_match] < bins[i+1])
        if qu_idx.sum() == 0:
            qu += [fill_val]
            qu_eup += [fill_val]
            qu_edo += [fill_val]
        else:
            med, eup, edo = np.nanpercentile(flux_norm[qu_match][qu_idx],[50,84,16])
            qu += [med]
            qu_eup += [eup]
            qu_edo += [edo]

    # normalize at 1um
    one_idx = np.abs(midbins-1).argmin()
    qu, qu_eup, qu_edo = np.array(qu)/qu[one_idx],np.array(qu_eup)/qu[one_idx],np.array(qu_edo)/qu[one_idx]
    sf, sf_eup, sf_edo = np.array(sf)/sf[one_idx],np.array(sf_eup)/sf[one_idx],np.array(sf_edo)/sf[one_idx]

    # define limits based off of errors
    if sederrs:
        qu_errs = prosp_dutils.asym_errors(qu, qu_eup, qu_edo)
        sf_errs = prosp_dutils.asym_errors(sf, sf_eup, sf_edo)
        ymin, ymax = np.min([qu_edo[qu_edo>0].min(),sf_edo[sf_edo>0].min()]), np.max([qu_eup[qu_eup>0].max(),sf_eup[sf_eup>0].max()])
    else:
        qu_errs, sf_errs = None, None
        ymin, ymax = np.min([qu[qu>0].min(),sf[sf>0].min()])*0.5, np.max([qu[qu>0].max(),sf[sf>0].max()])*2

    sedax.errorbar(midbins, qu, yerr=qu_errs,color=colors[0], **sedopts)
    sedax.errorbar(midbins, sf, yerr=sf_errs,color=colors[1], **sedopts)

    # limits
    sedax.set_ylim(ymin,ymax)

    # labels 
    sedax.set_xlabel(r'rest-frame wavelength [$\mu$m]',fontsize=fs*1.3)
    sedax.set_ylabel(r'median f$_{\nu}$/M$_{*}$',fontsize=fs*1.3)

    # scales
    sedax.set_xscale('log',subsx=(1,3),nonposx='clip')
    sedax.xaxis.set_major_formatter(FormatStrFormatter('%2.5g'))
    sedax.xaxis.set_minor_formatter(FormatStrFormatter('%2.5g'))
    sedax.set_yscale('log',subsy=(1,3),nonposy='clip')
    sedax.yaxis.set_major_formatter(FormatStrFormatter('%2.5g'))
    sedax.yaxis.set_minor_formatter(FormatStrFormatter('%2.5g'))
    sedax.tick_params(pad=3.5, size=3.5, width=1.0, which='both',labelsize=fs,right=True,labelright=True)

    plt.savefig(outname,dpi=200)
    plt.close()

def discretize(data, nbins):
    split = np.array_split(np.sort(data), nbins)
    cutoffs = [x[-1] for x in split]
    return cutoffs

def plot_3dhst_qfrac(dat, ax):
    """ this needs to be rewritten to do probabilistic assignments
    based on the full PDF, i.e. properly incorporating sSFR error bars
    in the quiescent fraction values
    """

    # pull out data, define ranges
    uv = dat['fast']['uv'] 
    vj = dat['fast']['vj'] 
    uv_lim, vj_lim = (0,2.5), (0,2.5)
    uv_range = np.linspace(uv_lim[0],uv_lim[1],nbins_3dhst)
    vj_range = np.linspace(vj_lim[0],vj_lim[1],nbins_3dhst)

    # generate map
    qvals = dat['prosp']['ssfr_100']['q50']
    qfrac_map = np.empty(shape=(nbins_3dhst,nbins_3dhst))
    uv_idx = np.digitize(uv,uv_range)
    vj_idx = np.digitize(vj,vj_range)
    for j in range(nbins_3dhst):
        for k in range(nbins_3dhst):
            idx = (uv_idx == k) & (vj_idx == j)
            if idx.sum() < bin_min:
                qfrac_map[k,j] = np.nan
            else:
                qfrac_map[k,j] = (qvals[idx] < ssfr_lim).sum() / float(idx.sum())

    # smooth and plot map
    qfrac_map = nansmooth(qfrac_map)
    img = ax.imshow(qfrac_map, origin='lower', cmap='coolwarm', extent=(0,2.5,0,2.5),vmin=0,vmax=1)
    ax.set_title('quiescent fraction\n (real data)')
    cbar = plt.colorbar(img, ax=ax,aspect=10,fraction=0.1)
    cbar.solids.set_rasterized(True)
    cbar.solids.set_edgecolor("face")

    ax.set_xlabel('V-J')
    ax.set_ylabel('U-V')
    plot_uvj_box(ax)

def plot_mock_qfrac(dat, ax):
    
    # pull out UV, VJ data
    uv = np.array(dat['uv'])
    vj = np.array(dat['vj'])
    nbin = int(np.sqrt(uv.shape))

    # interpolate over quantiles to get qfrac
    percentiles = np.linspace(0.01,0.99,99)

    # set up map
    qfrac_map = np.empty(shape=(nbin,nbin))

    idx = lnlike_cut(np.array(dat['lnlike']).astype(float))
    ssfr_chains = np.array(dat['chains']['ssfr_100'])
    ssfr_chains[idx] = None
    ssfr_chains = ssfr_chains.reshape(nbin,nbin).swapaxes(0,1)
    weights = np.array(dat['weights']).reshape(nbin,nbin).swapaxes(0,1)

    # generate map
    for j in range(nbin):
        for k in range(nbin):
            if ssfr_chains[k,j] is not None:
                # rewrite this such that we incorporate weights!!!!!!!!!
                val_percentiles = np.array(wquantile(np.log10(ssfr_chains[k,j]), percentiles, weights=weights[k,j]))
                qfrac = interp1d(val_percentiles,percentiles,  bounds_error = False, fill_value = 0)
                qfrac_map[k,j] = float(qfrac(ssfr_lim))
            else:
                qfrac_map[k,j] = np.nan
    qfrac_map = nansmooth(qfrac_map)

    # plot map
    img = ax.imshow(qfrac_map, origin='lower', cmap='coolwarm',
                         extent=(0,2.5,0,2.5),vmin=0,vmax=1)
    ax.set_title('quiescent fraction\n(mock data)')
    cbar = plt.colorbar(img, ax=ax,aspect=10,fraction=0.1)
    cbar.solids.set_rasterized(True)
    cbar.solids.set_edgecolor("face")

    ax.set_xlabel('V-J')
    ax.set_ylabel('U-V')
    plot_uvj_box(ax)

def lnlike_cut(lnlike):

    return (lnlike < -45.5) | (~np.isfinite(lnlike))

def plot_mock_maps(dat, pars, plabels, lims, outname, smooth=True, priors=None, cmap='plasma'):

    # pull out data, define ranges
    logpars = ['massmet_2','ssfr_100','avg_age']
    uv = np.array(dat['uv'])
    vj = np.array(dat['vj'])
    nbin = int(np.sqrt(uv.shape))

    idx_poor = lnlike_cut(np.array(dat['lnlike']).astype(float))

    # plot options
    if priors is not None:
        fig, axes = plt.subplots(2,2, figsize=(7,6))
        vmin, vmax, cax = kl_vmin, kl_vmax, fig.add_axes([0.85, 0.15, 0.05, 0.7])
    else:
        fig, axes = plt.subplots(2,2, figsize=(6,6))
        vmin, vmax, cax = None, None, None
    axes = axes.ravel()

    # construct maps
    for i, par in enumerate(pars):

        # either KLD map w/ prior
        # or median of PDF
        if priors is not None:
            vmap = np.array(dat['kld'][par])
            ax = None
        else:
            vmap = np.array(dat['pars'][par])
            if par in ['avg_age','ssfr_100']:
                vmap = np.log10(vmap)
            ax = axes[i]

        vmap[idx_poor] = np.nan
        valmap = vmap.reshape(nbin,nbin).swapaxes(0,1)
        if smooth:
            valmap = nansmooth(valmap)

        # show map
        img = axes[i].imshow(valmap, origin='lower', cmap=cmap,
                             extent=(0,2.5,0,2.5),vmin=vmin,vmax=vmax)
        axes[i].set_title(plabels[i])
        cbar = fig.colorbar(img, ax=ax, cax=cax, aspect=10,fraction=0.1)
        cbar.solids.set_rasterized(True)
        cbar.solids.set_edgecolor("face")

    # labels
    for a in axes.tolist():
        a.set_xlabel('V-J')
        a.set_ylabel('U-V')
        plot_uvj_box(a)

    plt.tight_layout(h_pad=0.05,w_pad=0.05)
    if priors is not None:
        fig.subplots_adjust(right=0.85)
    plt.savefig(outname, dpi=150)
    plt.close()

def plot(data,outfolder):
    """this is deprecated
    plots PDFs for a few fits
    """
    # loop over parameters
    nobj = len(data['names'])
    for key in data['pars'].keys():

        # plot geometry
        fig, ax = plt.subplots(nobj/2,2, figsize=(4.5, 2*nobj/2))
        ax = ax[:,0].tolist() + ax[:,1].tolist()

        # loop over fits
        min, max = np.nan, np.nan
        for i, name in enumerate(data['names']):

            samp = data['pars'][key][i]
            label = key
            if data['log'].get(key,False):
                samp = np.log10(data['pars'][key][i])
                label = 'log('+key+')'

            plot_pdf(ax[i],samp,data['weights'][i])

            # axis labels
            if (i == 3) or (i == 7):
                ax[i].set_xlabel(label)
            else:
                ax[i].set_xticklabels([])

            if i >= 4:
                ax[i].set_yticklabels([])
            else:
                ax[i].set_ylabel('PDF')

            # minima and maxima
            min = np.nanmin([min,np.min(ax[i].get_xlim())])
            max = np.nanmax([max,np.max(ax[i].get_xlim())])

            # objname 
            ax[i].text(0.05,0.9,str(name),transform=ax[i].transAxes)

        if key == 'ssfr_100':
            min, max = minlogssfr, maxlogssfr

        for a in ax: 
            a.set_xlim(min,max)
            a.set_ylim(0,1.1)

        plt.tight_layout(h_pad=0.0,w_pad=0.0)
        fig.subplots_adjust(hspace=0.00,wspace=0.0)
        plt.savefig(outfolder+key+'.png',dpi=150)
        plt.close()

def plot_3d_maps(dat,pars,plabels,lims,outname,kld=False,smooth=True,density=False,cmap='plasma'):

    # make plot geometry
    if kld:
        fig, axes = plt.subplots(2,2, figsize=(7,6))
        vmin, vmax, cax = kl_vmin, kl_vmax, fig.add_axes([0.85, 0.15, 0.05, 0.7])
    else:
        fig, axes = plt.subplots(2,2, figsize=(7,7))
        vmin, vmax, cax = None, None, None
    axes = axes.ravel()

    # pull out data, define ranges
    logpars = ['massmet_2','ssfr_100','avg_age']
    uv = dat['fast']['uv'] 
    vj = dat['fast']['vj'] 
    uv_lim, vj_lim = (0,2.5), (0,2.5)
    ncont = 20  # number of contours
    uv_range = np.linspace(uv_lim[0],uv_lim[1],nbins_3dhst)
    vj_range = np.linspace(vj_lim[0],vj_lim[1],nbins_3dhst)

    # density
    if density:
        fig.subplots_adjust(left=0.54,wspace=0.3,hspace=0.5,right=0.98)
        dax = fig.add_axes([0.08, 0.1, 0.38, 0.8])

        he, xedges, yedges = np.histogram2d(vj, uv, bins=(vj_range,uv_range))
        he[he<bin_min] = 0.0
        xmid = (xedges[1:] + xedges[:-1])/2.
        ymid = (yedges[1:] + yedges[:-1])/2.
        logn = np.log10(he.T)
        logn[(~np.isfinite(logn))] = np.nan
        logn_min, logn_max = np.nanmin(logn), np.nanmax(logn)
        contours = 10**np.linspace(logn_min,logn_max,ncont+1)

        cs = dax.contourf(xmid,ymid, he.T, contours, zorder=2, alpha=1.0, cmap='RdGy_r', 
                         norm=colors.LogNorm(vmin=10**logn_min, vmax=10**logn_max))
        dax.contour(xmid,ymid, he.T, contours[3:], linewidths=1, zorder=2, colors='k',alpha=0.5)
        dax.set_title(r'N$_{\mathrm{galaxies}}$',fontsize=14)
        dax.set_xlabel('V-J')
        dax.set_ylabel('U-V')
        plot_uvj_box(a)

        cb = fig.colorbar(cs, ax=dax, aspect=10, ticks=[1, 10, 30, 100, 300, 1000])
        cb.ax.set_yticklabels(['1', '10', '30', '100', '300', '1000'])
        cb.solids.set_edgecolor("face")

    # averages
    for i, par in enumerate(pars):

        qvals = dat['prosp'][par]['q50']
        if par == 'avg_age':
            qvals = np.log10(qvals)

        # we take the averages by double-looping over digitize
        # I hope it's not as slow as it seems
        avgmap = np.empty(shape=(nbins_3dhst,nbins_3dhst))
        uv_idx = np.digitize(uv,uv_range)
        vj_idx = np.digitize(vj,vj_range)
        for j in range(nbins_3dhst):
            for k in range(nbins_3dhst):
                idx = (uv_idx == k) & (vj_idx == j)
                if idx.sum() < bin_min:
                    avgmap[k,j] = np.nan
                else:
                    if kld:
                        avgmap[k,j] = np.nanmedian(dat['kld'][par][idx])
                    elif par in logpars:
                        avgmap[k,j] = np.log10(np.mean(10**qvals[idx]))
                    else:
                        avgmap[k,j] = np.mean(qvals[idx])

        if smooth:
            avgmap = nansmooth(avgmap)

        if kld:
            ax = None
        else:
            ax = axes[i]

        # show map
        img = axes[i].imshow(avgmap, origin='lower', cmap=cmap,
                             extent=(vj_lim[0],vj_lim[1],uv_lim[0],uv_lim[1]),
                             vmin=vmin,vmax=vmax)
        axes[i].set_title(plabels[i])
        cbar = fig.colorbar(img, ax=ax,cax=cax,aspect=8)
        cbar.solids.set_rasterized(True)
        cbar.solids.set_edgecolor("face")

    # labels
    for a in axes:
        a.set_xlabel('V-J')
        a.set_ylabel('U-V')
        plot_uvj_box(a)

    plt.tight_layout(h_pad=0.05,w_pad=0.05)
    if kld:
        fig.subplots_adjust(right=0.85)
    plt.savefig(outname, dpi=150)
    plt.close()

def plot_pdf(ax,samples,weights):

    # smoothing routine from dynesty
    bins = int(round(10. / 0.02))
    n, b = np.histogram(samples, bins=bins, weights=weights, range=[samples.min(),samples.max()])
    n = norm_kde(n, 10.)
    x0 = 0.5 * (b[1:] + b[:-1])
    y0 = n
    ax.fill_between(x0, y0/y0.max(), color='k', alpha = 0.6)

    # plot mean
    ax.axvline(np.average(samples, weights=weights), color='red',lw=2)

def plot_3d_hists(dat,pars,plabels,lims,outname):

    # parameters
    pars = ['ssfr_100','dust2','massmet_2','avg_age']# ,'dust_index'
    plabels = ['sSFR [100 Myr]', r'$\tau_{\mathrm{diffuse}}$', r'log(Z/Z$_{\odot}$)', r'mean age'] # 'dust index'
    lims = [(-14,-8.2),(0,2.4),(-1.99,0.2),(0.0,10)] # (-2.2,0.4)

    # axes and options
    fig, axes = plt.subplots(3,4, figsize=(7,5))
    nbins = 20
    histopts = {'drawstyle':'steps-mid','alpha':0.9, 'lw':2, 'linestyle': '-'}
    colors = [('#FF3D0D', '#1C86EE'),('#004c9e', '#4fa0f7'), ('#a32000', '#ff613a')]

    # text
    ylabels = ['all\ngalaxies', 'star-forming\ngalaxies', 'quiescent\ngalaxies']
    ytxt = [('quiescent', 'star-forming'), ('dusty','not dusty'), ('dusty','not dusty')]
    xt, yt, dy = 0.03, 0.91, 0.08
    fs = 9

    # indices + index math
    sf_idx = (dat['fast']['uvj_prosp'] == 1) | (dat['fast']['uvj_prosp'] == 2)
    qu_idx = (dat['fast']['uvj_prosp'] == 3)
    qu_dust_idx = (qu_idx) & (dat['fast']['uvj_dust_prosp'])
    qu_nodust_idx = (qu_idx) & (~dat['fast']['uvj_dust_prosp'])
    sf_dust_idx = (sf_idx) & (dat['fast']['uvj_dust_prosp'])
    sf_nodust_idx = (sf_idx) & (~dat['fast']['uvj_dust_prosp'])
    idx1 = [qu_idx, sf_nodust_idx, qu_nodust_idx]
    idx2 = [sf_idx, sf_dust_idx, qu_dust_idx]
    percs = np.round([
                      100*qu_idx.sum()/float(sf_idx.sum()+qu_idx.sum()), 
                      100*qu_dust_idx.sum()/float(qu_idx.sum()),
                      100*sf_dust_idx.sum()/float(sf_idx.sum())
                     ]).astype(int)

    # ssfr contamination
    qu_contam = (dat['prosp']['ssfr_100']['q50'][qu_idx] > ssfr_lim).sum() / float(qu_idx.sum())
    sf_contam = (dat['prosp']['ssfr_100']['q50'][sf_idx] < ssfr_lim).sum() / float(sf_idx.sum())
    print 'quiescent contamination: {0:.3f}'.format(qu_contam)
    print 'star-forming contamination: {0:.3f}'.format(sf_contam)

    # main loop
    for i, (par, lab) in enumerate(zip(pars,plabels)):
        
        # plot histograms
        data = dat['prosp'][par]['q50']
        for j, a in enumerate(axes[:,i]): 

            x, hist1, hist2 = hist(data,idx1[j], idx2[j], nbins, lims[i])
            a.plot(x,hist1,color=colors[j][0], **histopts)
            a.plot(x,hist2,color=colors[j][1], **histopts)
            a.set_ylim(0.0,a.get_ylim()[1]*1.2)

            # titles and labels
            a.set_xlim(lims[i])
            a.text(xt,yt,ytxt[j][0]+'({0}%)'.format(percs[j]),transform=a.transAxes,color=colors[j][0],fontsize=fs)
            a.text(xt,yt-dy,ytxt[j][1]+'({0}%)'.format(100-percs[j]),transform=a.transAxes,color=colors[j][1],fontsize=fs)
        axes[2,i].set_xlabel(lab)

    # tickmark labels
    for a in axes[:1,:].flatten(): a.set_xticklabels([])
    for a in axes[:,1:].flatten(): a.set_yticklabels([])
    for i,a in enumerate(axes[:,0].flatten()): a.set_ylabel(ylabels[i])

    plt.tight_layout(h_pad=0.0,w_pad=0.2)
    plt.subplots_adjust(hspace=0.00,wspace=0.0)
    plt.savefig(outname, dpi=150)
    plt.close()

def hist(par,idx1,idx2,nbins,alims,density=True):

    # bin on same scale, pad histogram
    _, bins = np.histogram(par[idx1 | idx2],bins=nbins,range=alims)
    hist1, _ = np.histogram(par[idx1],bins=bins,density=density,range=alims)
    hist2, _ = np.histogram(par[idx2],bins=bins,density=density,range=alims)
    hist1 = np.pad(hist1,1,'constant',constant_values=0.0)
    hist2 = np.pad(hist2,1,'constant',constant_values=0.0)

    # midpoint of each bin
    db = (bins[1]-bins[0])/2.
    x = np.concatenate((bins-db,np.atleast_1d(bins[-1]+db)))
    return x, hist1, hist2

def plot_uvj_box(ax,lw=2):

    # line is UV = 0.8*VJ+0.7
    # constant UV=1.3, VJ=1.5
    xlim = ax.get_xlim()
    ax.plot([xlim[0],0.75],[1.3,1.3],linestyle='-',color='k',lw=lw)
    ax.plot([0.75,1.5],[1.3,1.9],linestyle='-',color='k',lw=lw)
    ax.plot([1.5,1.5],[1.9,xlim[1]],linestyle='-',color='k',lw=lw)
    ax.set_xlim()

def nansmooth(valmap,sigma=1.0):

    V=valmap.copy()
    V[valmap!=valmap]=0
    VV=gaussian_filter(V.astype(float),sigma=sigma)

    W=0*valmap.copy()+1
    W[valmap!=valmap]=0
    WW=gaussian_filter(W.astype(float),sigma=sigma)

    Z=VV/WW
    Z[~np.isfinite(valmap.astype(float))] = np.nan
    return Z



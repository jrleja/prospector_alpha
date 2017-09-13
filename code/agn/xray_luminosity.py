import numpy as np
import prospector_io
import matplotlib.pyplot as plt
import prosp_dutils
import os
from astropy.cosmology import WMAP9
import magphys_plot_pref
from corner import quantile
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy import interpolate

dpi = 150

minorFormatter = magphys_plot_pref.jLogFormatter(base=10, labelOnlyBase=False)
majorFormatter = magphys_plot_pref.jLogFormatter(base=10, labelOnlyBase=True)

red = '#FF3D0D'
blue = '#1C86EE'

plotopts = {
         'fmt':'o',
         'ecolor':'k',
         'capthick':0.35,
         'elinewidth':0.35,
         'alpha':0.4
        } 

def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.'''

    import matplotlib.cm as cmx
    import matplotlib.colors as colors

    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='plasma') 
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color

def collate_data(alldata, **extras):

    ### preliminary stuff
    parnames = alldata[0]['pquantiles']['parnames']
    eparnames = alldata[0]['pextras']['parnames']
    xr_idx = eparnames == 'xray_lum'
    xray = prospector_io.load_xray_cat(xmatch = True, **extras)
    nsamp = 100000 # for newly defined variables

    #### for each object
    fagn, fagn_up, fagn_down, agn_tau, agn_tau_up, agn_tau_down, mass, mass_up, mass_down, lir_luv, lir_luv_up, lir_luv_down, xray_lum, xray_lum_err, database, observatory = [[] for i in range(16)]
    fagn_obs, fagn_obs_up, fagn_obs_down, lmir_lbol, lmir_lbol_up, lmir_lbol_down, xray_hardness, xray_hardness_err = [[] for i in range(8)]
    lagn, lagn_up, lagn_down, lsfr, lsfr_up, lsfr_down, lir, lir_up, lir_down = [], [], [], [], [], [], [], [], []
    sfr, sfr_up, sfr_down, ssfr, ssfr_up, ssfr_down, d2, d2_up, d2_down = [[] for i in range(9)]
    fmir, fmir_up, fmir_down, objname = [], [], [], []
    fmir_chain, agn_tau_chain = [], []
    for ii, dat in enumerate(alldata):
        objname.append(dat['objname'])

        #### mass, SFR, sSFR, dust2
        mass.append(dat['pextras']['q50'][eparnames=='stellar_mass'][0])
        mass_up.append(dat['pextras']['q84'][eparnames=='stellar_mass'][0])
        mass_down.append(dat['pextras']['q16'][eparnames=='stellar_mass'][0])
        sfr.append(dat['pextras']['q50'][eparnames=='sfr_100'][0])
        sfr_up.append(dat['pextras']['q84'][eparnames=='sfr_100'][0])
        sfr_down.append(dat['pextras']['q16'][eparnames=='sfr_100'][0])
        ssfr.append(dat['pextras']['q50'][eparnames=='ssfr_100'][0])
        ssfr_up.append(dat['pextras']['q84'][eparnames=='ssfr_100'][0])
        ssfr_down.append(dat['pextras']['q16'][eparnames=='ssfr_100'][0])
        d2.append(dat['pquantiles']['q50'][parnames=='dust2'][0])
        d2_up.append(dat['pquantiles']['q84'][parnames=='dust2'][0])
        d2_down.append(dat['pquantiles']['q16'][parnames=='dust2'][0])

        #### model f_agn, l_agn, fmir
        fagn.append(dat['pquantiles']['q50'][parnames=='fagn'][0])
        fagn_up.append(dat['pquantiles']['q84'][parnames=='fagn'][0])
        fagn_down.append(dat['pquantiles']['q16'][parnames=='fagn'][0])
        agn_tau.append(dat['pquantiles']['q50'][parnames=='agn_tau'][0])
        agn_tau_up.append(dat['pquantiles']['q84'][parnames=='agn_tau'][0])
        agn_tau_down.append(dat['pquantiles']['q16'][parnames=='agn_tau'][0])
        agn_tau_chain.append(dat['pquantiles']['sample_chain'][:,parnames=='agn_tau'].squeeze())
        lagn.append(dat['pextras']['q50'][eparnames=='l_agn'][0])
        lagn_up.append(dat['pextras']['q84'][eparnames=='l_agn'][0])
        lagn_down.append(dat['pextras']['q16'][eparnames=='l_agn'][0])
        fmir.append(dat['pextras']['q50'][eparnames=='fmir'][0])
        fmir_up.append(dat['pextras']['q84'][eparnames=='fmir'][0])
        fmir_down.append(dat['pextras']['q16'][eparnames=='fmir'][0])
        fmir_chain.append(dat['pextras']['flatchain'][:,eparnames=='fmir'].squeeze())

        #### L_UV / L_IR, LMIR/LBOL
        cent, eup, edo = quantile(np.random.choice(dat['lir'],nsamp) / np.random.choice(dat['luv'],nsamp), [0.5, 0.84, 0.16])

        lir_luv.append(cent)
        lir_luv_up.append(eup)
        lir_luv_down.append(edo)

        cent, eup, edo = quantile(np.random.choice(dat['lir'],nsamp), [0.5, 0.84, 0.16])

        lir.append(cent)
        lir_up.append(eup)
        lir_down.append(edo)

        cent, eup, edo = quantile(np.random.choice(dat['lmir'],nsamp) / np.random.choice(dat['pextras']['flatchain'][:,dat['pextras']['parnames'] == 'lbol'].squeeze(),nsamp), [0.5, 0.84, 0.16])

        lmir_lbol.append(cent)
        lmir_lbol_up.append(eup)
        lmir_lbol_down.append(edo)

        #### x-ray fluxes
        # match
        idx = xray['objname'] == dat['objname']
        if idx.sum() != 1:
            print 1/0
        xflux = xray['flux'][idx][0]
        xflux_err = xray['flux_err'][idx][0]

        # flux is in ergs / cm^2 / s, convert to erg /s 
        pc2cm =  3.08568E18
        dfactor = 4*np.pi*(dat['residuals']['phot']['lumdist']*1e6*pc2cm)**2
        xray_lum.append(xflux * dfactor)
        xray_lum_err.append(xflux_err * dfactor)
        xray_hardness.append(xray['hardness'][idx][0])
        xray_hardness_err.append(xray['hardness_err'][idx][0])

        #### CALCULATE F_AGN_OBS
        # take advantage of the already-computed conversion between FAGN (model) and LAGN (model)
        fagn_chain = dat['pquantiles']['sample_chain'][:,parnames=='fagn']
        lagn_chain = dat['pextras']['flatchain'][:,eparnames == 'l_agn']
        conversion = (lagn_chain / fagn_chain).squeeze()

        ### calculate L_AGN chain
        scale = xray_lum_err[-1]
        if scale <= 0:
            lagn_chain = np.repeat(xray_lum[-1], conversion.shape[0])
        else: 
            lagn_chain = np.random.normal(loc=xray_lum[-1], scale=scale, size=conversion.shape[0])
        obs_fagn_chain = lagn_chain / conversion
        cent, eup, edo = quantile(obs_fagn_chain, [0.5, 0.84, 0.16])

        fagn_obs.append(cent)
        fagn_obs_up.append(eup)
        fagn_obs_down.append(edo)

        ##### L_OBS / L_SFR(MODEL)
        # sample from the chain, assume gaussian errors for x-ray fluxes
        chain = dat['pextras']['flatchain'][:,xr_idx].squeeze()

        if scale <= 0:
            subchain =  np.repeat(xray_lum[-1], nsamp) / \
                        np.random.choice(chain,nsamp)
        else:
            subchain =  np.random.normal(loc=xray_lum[-1], scale=scale, size=nsamp) / \
                        np.random.choice(chain,nsamp)

        cent, eup, edo = quantile(subchain, [0.5, 0.84, 0.16])

        lsfr.append(cent)
        lsfr_up.append(eup)
        lsfr_down.append(edo)

        #### database and observatory
        database.append(str(xray['database'][idx][0]))
        try:
            observatory.append(str(xray['observatory'][idx][0]))
        except KeyError:
            observatory.append(' ')

    out = {}
    out['objname'] = objname
    out['database'] = database
    out['observatory'] = observatory
    out['mass'] = mass
    out['mass_up'] = mass_up
    out['mass_down'] = mass_down
    out['sfr'] = sfr
    out['sfr_up'] = sfr_up
    out['sfr_down'] = sfr_down
    out['ssfr'] = ssfr
    out['ssfr_up'] = ssfr_up
    out['ssfr_down'] = ssfr_down
    out['d2'] = d2
    out['d2_up'] = d2_up
    out['d2_down'] = d2_down
    out['lir_luv'] = lir_luv
    out['lir_luv_up'] = lir_luv_up
    out['lir_luv_down'] = lir_luv_down
    out['lir'] = lir
    out['lir_up'] = lir_up
    out['lir_down'] = lir_down
    out['lmir_lbol'] = lmir_lbol
    out['lmir_lbol_up'] = lmir_lbol_up
    out['lmir_lbol_down'] = lmir_lbol_down
    out['fagn'] = fagn
    out['fagn_up'] = fagn_up
    out['fagn_down'] = fagn_down
    out['agn_tau'] = agn_tau
    out['agn_tau_up'] = agn_tau_up
    out['agn_tau_down'] = agn_tau_down
    out['agn_tau_chain'] = agn_tau_chain
    out['fmir'] = fmir
    out['fmir_up'] = fmir_up
    out['fmir_down'] = fmir_down
    out['fmir_chain'] = fmir_chain
    out['fagn_obs'] = fagn_obs
    out['fagn_obs_up'] = fagn_obs_up
    out['fagn_obs_down'] = fagn_obs_down
    out['lagn'] = lagn 
    out['lagn_up'] = lagn_up
    out['lagn_down'] = lagn_down
    out['lsfr'] = lsfr
    out['lsfr_up'] = lsfr_up
    out['lsfr_down'] = lsfr_down
    out['xray_luminosity'] = xray_lum
    out['xray_luminosity_err'] = xray_lum_err
    out['xray_hardness'] = xray_hardness
    out['xray_hardness_err'] = xray_hardness_err

    for key in out.keys(): out[key] = np.array(out[key])

    #### ADD WISE PHOTOMETRY
    from wise_colors import collate_data as wise_phot
    from wise_colors import vega_conversions
    wise = wise_phot(alldata)

    #### generate x, y values
    w1w2 = -2.5*np.log10(wise['obs_phot']['wise_w1'])+2.5*np.log10(wise['obs_phot']['wise_w2'])
    w1w2 += vega_conversions('wise_w1') - vega_conversions('wise_w2')
    out['w1w2'] = w1w2

    return out

def make_plot(agn_evidence,runname='brownseds_agn',alldata=None,outfolder=None,maxradius=30,idx=Ellipsis,**popts):

    #### load alldata
    if alldata is None:
        alldata = prospector_io.load_alldata(runname=runname)

    #### make output folder if necessary
    if outfolder is None:
        outfolder = os.getenv('APPS')+'/prospector_alpha/plots/'+runname+'/agn_plots/'
        if not os.path.isdir(outfolder):
            os.makedirs(outfolder)

    #### collate data
    pdata = collate_data(alldata, maxradius=maxradius)

    ### plot PDF scatterplots
    plot_pdf_scatter(pdata, alldata, idx=idx, outname=outfolder+'xray_pdf.png', **popts)

    ### PLOT VERSUS OBSERVED X-RAY FLUX
    outname = 'xray_lum_fagn_model.png'
    fig,ax = plot(pdata,
                  ypar='fagn',ylabel = r'log(f$_{\mathrm{AGN,MIR}}$)',xrb_color_flag=False, idx=idx, **popts)
    plt.tight_layout()
    plt.savefig(outfolder+outname,dpi=dpi)
    plt.close()

    ### PLOT VERSUS 'TRUE' X-RAY FLUX
    outname = 'xray_lum_sfrcorr_fagn_model.png'
    fig,ax = plot(pdata,
                  ypar='fagn',ylabel = r'f$_{\mathrm{AGN,MIR}}$',
                  xpar='lsfr',xlabel = r'L$_{\mathrm{X}}$(observed)/L$_{\mathrm{XRB}}$(model)',
                  idx = idx, **popts)
    plt.tight_layout()
    plt.savefig(outfolder+outname,dpi=dpi)
    plt.close()

    ### SFR, sSFR, dust2, LUV/LIR versus FAGN
    fig,ax = plot_model_corrs(pdata,idx=idx,**popts)
    plt.savefig(outfolder+'fagn_versus_galaxy_properties.png',dpi=dpi)
    plt.close()


    agn_evidence['xray_luminosity'] = pdata['xray_luminosity']
    agn_evidence['xray_luminosity_err'] = pdata['xray_luminosity_err']
    agn_evidence['xrb_flag'] = xray_cuts(pdata)
    
    return agn_evidence

def xray_cuts(pdata):
    lower_sigma = pdata['lsfr_down']
    return lower_sigma > 1

def plot_pdf_scatter(pdata, alldata, idx=Ellipsis, outname=None,
                     log=True, **popts):

    '''
    get xdata and create shape
    # need PDF
    # create histogram
    # smooth it? or more likely load the chains for the full PDF
    # make a PDF polygon! fixed width (ignore x-errors, miniscule)
    # fill in with light color, do line down the center, mark q50 with red point
    '''

    ### set up parameters
    fig, ax = plt.subplots(1,2,figsize=(12,6))

    ### deal with x-axis
    xpar = 'xray_luminosity'
    pidx = (pdata[xpar][idx] > 0)
    #pdata = load_full_pdf(pdata,alldata,pidx)
    xplot = np.log10(pdata[xpar][idx][pidx])
    xerr_1d = pdata['xray_luminosity_err'][idx][pidx]
    xlabel = r'log(L$_{\mathrm{X}}$(observed)) [erg/s]'

    ### deal with y-axis
    ypars = ['fmir','agn_tau']
    ylabel = [r'f$_{\mathrm{AGN,MIR}}$',r'$\tau_{\mathrm{AGN}}$']
    nbins = 20
    clip = [5,95]

    lims = [(0.03,1),(2,100)]
    sig = xray_cuts(pdata)[idx][pidx]
    ### loop over two y-parameters (AGN tau and fmir)
    for ii, yp in enumerate(ypars):

        q50 = pdata[yp][idx][pidx]
        q84 = pdata[yp+'_up'][idx][pidx]
        q16 = pdata[yp+'_down'][idx][pidx]

        for color,id in zip([red,blue],[sig,~sig]):
            yerr = prosp_dutils.asym_errors(q50[id],q84[id],q16[id])
            ax[ii].scatter(xplot[id], q50[id], marker='o', color=color,s=52,zorder=11,alpha=0.9,edgecolors='k')
            ax[ii].errorbar(xplot[id], q50[id], yerr=yerr, zorder=-5,ms=0.0, ecolor='k', linestyle=' ',
                            capthick=0.8, elinewidth=0.4, alpha=0.7)

        ### labels
        ax[ii].set_ylabel(ylabel[ii])
        ax[ii].set_xlabel(xlabel)

        ### limits
        xlim = (xplot.min()-0.5,xplot.max()+0.5)
        ax[ii].set_xlim(xlim)
        ax[ii].xaxis.set_major_locator(MaxNLocator(5))
        ax[ii].set_ylim(lims[ii])

        ax[ii].set_yscale('log',nonposx='clip',subsy=(1,2,4))
        ax[ii].yaxis.set_minor_formatter(minorFormatter)
        ax[ii].yaxis.set_major_formatter(majorFormatter)
        for tl in ax[ii].get_yticklabels():tl.set_visible(False)


            #ylim = (10**(agn_chain.min()-0.2),10**(agn_chain.max()+0.2)) # have to put in log if-statement


        ### grab chain, create histogram
        '''
        agn_chain = pdata[yp+'_chain'][idx][pidx]
        if log:
            agn_chain = np.log10(agn_chain)
    
        ### loop over each galaxy PDF
        polygons = []
        for k, chain in enumerate(agn_chain):

            ### clip chain
            ends = np.percentile(chain,clip)
            clipped_chain = chain[(chain > ends[0]) & (chain < ends[1])]

            ### create polygon
            hist, bins = np.histogram(clipped_chain,bins=nbins,density=True)

            xpoints = (hist/hist.max()) * xwidth + xplot[k]
            xpoints = np.array([xplot[k]] + xpoints.tolist() + [xplot[k]]) # add padding

            delta = bins[1]-bins[0]
            ypoints = (bins[:-1]+bins[1:])/2.
            ypoints = np.array([ypoints.min()-delta] + ypoints.tolist() + [ypoints.max()+delta]) # add padding

            if log:
                ypoints = 10**ypoints

            coords = np.vstack((xpoints,ypoints)).transpose()

            polygons.append(Polygon(coords,closed=True))

        p = PatchCollection(polygons, alpha=0.4, facecolor=blue, edgecolor='blue', linewidth=1)
        ax[ii].add_collection(p)
        '''

    plt.tight_layout()
    plt.savefig(outname,dpi=150)
    plt.close()

def plot(pdata,
         ypar=None, ylabel=None, 
         xpar='xray_luminosity', xlabel=r'L$_{\mathrm{X}}$(observed) [erg/s]',idx=Ellipsis,
         xrb_color_flag=False, **popts):
    '''
    plots a color-color BPT scatterplot
    '''

    xplot = pdata[xpar]
    yplot = np.log10(pdata[ypar])

    if xpar == 'xray_luminosity':
        xmin, xmax = 5e35,2e44
        xerr_1d = pdata['xray_luminosity_err']
    elif xpar == 'lsfr':
        xmin, xmax = 5e35,2e45
        xmin, xmax = 2.5e-3,2e3

    #### plot photometry
    fig, ax = plt.subplots(1,1, figsize=(6, 6))

    ### figure out errors
    yerr =  prosp_dutils.asym_errors(pdata[ypar], pdata[ypar+'_up'], pdata[ypar+'_down'],log=True)
    if xpar == 'xray_luminosity':
        xerr = xerr_1d
    else:
        xerr =  prosp_dutils.asym_errors(pdata[xpar], pdata[xpar+'_up'], pdata[xpar+'_down'])

    cidx = np.ones_like(pdata[xpar],dtype=bool)
    cidx[idx] = False
    s = 120

    if xrb_color_flag:

        for ind, shape, alph in zip([cidx,~cidx],[popts['nofmir_shape'],popts['fmir_shape']],[popts['nofmir_alpha'],popts['fmir_alpha']]):
            lower_sigma = xray_cuts(pdata)[ind]
            significant = lower_sigma > 1

            ax.errorbar(xplot[ind][significant], yplot[ind][significant], 
                        yerr=[yerr[0][ind][significant],yerr[1][ind][significant]], xerr=xerr[ind][significant],
                        zorder=-5,ms=0.0,
                        **plotopts)
            ax.scatter(xplot[ind][significant], yplot[ind][significant], marker=shape, color=red,s=s,zorder=11,alpha=alph,edgecolors='k')
            ax.errorbar(xplot[ind][~significant], yplot[ind][~significant], 
                        yerr=[yerr[0][ind][~significant],yerr[1][ind][~significant]], xerr=xerr[ind][~significant],
                        zorder=-5,ms=0.0,
                        **plotopts)
            ax.scatter(xplot[ind][~significant], yplot[ind][~significant], marker=shape, color=blue,s=s,zorder=10,alpha=alph,edgecolors='k')

            #ax.text(0.98,0.18,r'L$_{\mathrm{X}}$ consistent',transform=ax.transAxes,color=blue,ha='right')
            #ax.text(0.98,0.14,'with XRBs',transform=ax.transAxes,color=blue,ha='right')
            #ax.text(0.98,0.10,r'L$_{\mathrm{X}}$ inconsistent',transform=ax.transAxes,color=red,ha='right')
            #ax.text(0.98,0.06,'with XRBs',transform=ax.transAxes,color=red,ha='right')
    else:
        ax.errorbar(xplot, yplot, yerr=yerr, xerr=xerr,ms=0.0,zorder=-2,
                    **plotopts)
        ax.scatter(xplot[cidx], yplot[cidx], marker=popts['nofmir_shape'], alpha=popts['nofmir_alpha'], color=popts['nofmir_color'],s=s*0.7,zorder=10,edgecolors='k')
        ax.scatter(xplot[~cidx], yplot[~cidx], marker=popts['fmir_shape'], alpha=popts['fmir_alpha'], color=popts['fmir_color'],s=s,zorder=10,edgecolors='k')

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(-4,0)
    ax.axvline(1e42, linestyle='--', color='k',lw=1,zorder=-1)
    ax.axhline(-1.0, linestyle='--', color=popts['fmir_color'],lw=1,zorder=-1)
    ax.text(1.4e36,-0.95,'photometric AGN', color=popts['fmir_color'],weight='semibold',fontsize=12)
    ax.text(1.3e42,-2.0,'X-ray\nAGN', color='k',weight='semibold',fontsize=12)

    ax.set_xscale('log',nonposx='clip',subsx=([1]))
    for tl in ax.get_xticklabels():tl.set_visible(False)

    #ax.set_yscale('log',nonposy='clip',subsy=([1]))

    #ax.text(0.05,0.05, r'N$_{\mathrm{match}}$='+str((pdata['xray_luminosity'] > xmin).sum()), transform=ax.transAxes)

    return fig, ax

def plot_model_corrs(pdata,color_by=None,idx=None,**popts):

    fig, ax = plt.subplots(2,2, figsize=(11, 10))
    cb_ax = fig.add_axes([0.83, 0.15, 0.05, 0.7])
    fig.subplots_adjust(right=0.8,wspace=0.3,hspace=0.3,left=0.12)
    ax = np.ravel(ax)

    #### fagn labeling
    xlabel = r'log(f$_{\mathrm{AGN,MIR}}$)'
    x = np.log10(pdata['fagn'])
    xerr =  prosp_dutils.asym_errors(pdata['fagn'], 
                                     pdata['fagn_up'],
                                     pdata['fagn_down'],log=True)

    #### y-axis
    ypar = ['mass','sfr','ssfr','lir']
    ylabels = [r'log(M$_{*}$) [M$_{\odot}$]', r'log(SFR) [M$_{\odot}$/yr]',
               r'log(sSFR) [yr$^{-1}$]',r'log(L$_{\mathrm{IR}}$)']
    cb = pdata['d2']
    for ii, yp in enumerate(ypar):

        log = True
        y = np.log10(pdata[yp])

        # find the complement
        cidx = np.ones_like(y,dtype=bool)
        cidx[idx] = False

        yerr =  prosp_dutils.asym_errors(pdata[yp], 
                                          pdata[yp+'_up'],
                                          pdata[yp+'_down'],log=log)
        ax[ii].errorbar(y,x,xerr=yerr, yerr=xerr, ms=0.0,zorder=-2,**plotopts)

        vmin, vmax = cb.min(), cb.max()
        ax[ii].scatter(y[cidx],x[cidx], marker=popts['nofmir_shape'],c=cb[cidx], cmap=popts['cmap'], \
                       s=90,zorder=10, vmin=vmin, vmax=vmax,edgecolors='k')
        pts = ax[ii].scatter(y[idx],x[idx], marker=popts['fmir_shape'],c=cb[idx], cmap=popts['cmap'], \
                             s=90,zorder=10, vmin=vmin, vmax=vmax,edgecolors='k')

        ax[ii].set_xlabel(ylabels[ii])
        ax[ii].set_ylabel(xlabel)
        ax[ii].xaxis.set_major_locator(MaxNLocator(4))

    cb = fig.colorbar(pts, cax=cb_ax)
    #cb.ax.set_title(r'$\tau_{V}$', fontdict={'fontweight':'bold','verticalalignment':'bottom'})
    cb.set_label(r'$\tau_{V}$', fontdict={'fontweight':'bold','fontsize':26})
    cb.solids.set_rasterized(True)
    cb.solids.set_edgecolor("face")

    return fig, ax













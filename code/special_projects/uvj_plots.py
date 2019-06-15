import numpy as np
import matplotlib.pyplot as plt
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
from stack_td_sfh import sfr_ms
from matplotlib.ticker import MaxNLocator
from sedpy import observate
from scipy.optimize import minimize

# plotting conventions
minlogssfr, maxlogssfr = -13, -8
ssfr_lim = -10. # defining quiescent
uv_lim, vj_lim = (0,2.5), (0,2.5)
ms_threesig = 0.9
minmass, maxmass = 10, np.inf
bin_min = 10 # minimum bin count to show up on plot
kl_vmin, kl_vmax = 0, 2
plt.ioff()

# constants
absolute_mags = np.array([5.11,4.65,4.53])  # http://mips.as.arizona.edu/~cnaw/sun.html
ml_sun = 1./10**(absolute_mags/-2.5)

# global colors
interloper_colors = ['#cc0000', '#3a76ff']

def do_all(runname='uvj', outfolder=None, make_legacy_plots=False, use_muzzin=False, use_masscut=True, smooth=1, ssfr_cut=True, **opts):

    if outfolder is None:
        outfolder = os.getenv('APPS') + '/prospector_alpha/plots/'+runname+'/uvj_plots/'
        if not os.path.isdir(outfolder):
            os.makedirs(outfolder)
            os.makedirs(outfolder+'data/')

    # swap parameter file based on runname
    # then broadcast relevant variables to global
    if runname == 'uvj':
        import uvj_dirichlet_params as pfile
    else:
        import uvj_params as pfile
    mod = pfile.load_model(**pfile.run_params)
    agemid = (10**mod.params['agebins']).mean(axis=1)/1e9
    agediff = np.diff(10**mod.params['agebins'],axis=1).flatten()
    global mod, pfile, agemid, agediff

    # global plotting parameters
    pars = ['ml_g0','dust2','ssfr_100','avg_age','massmet_2']# ,'dust_index','ml_r0','ml_i0'
    plabels = [r'log(M/L$_{\mathrm{g}}$)', r'$\tau_{\mathrm{dust}}$',r'log(sSFR/yr$^{-1}$)', r'log(stellar age/Gyr)',r'log(Z/Z$_{\odot}$)'] # 'dust index'
    lims = [(-1,0.8),(0,1.8),(-11.5,-8.6),(0,0.6),(-0.8,0.1)] # (-2.2,0.4)

    # grab data from 3dhst
    with open('/Users/joel/code/python/prospector_alpha/plots/td_delta/fast_plots/data/fastcomp.h5', "r") as f:
        threedhst_data = hickle.load(f)

    if use_muzzin: threedhst_data['uvj_prosp'] = threedhst_data['fast']['uvj_prosp_muzz']

    # define some useful quantities
    threedhst_data = add_extra_quantities(threedhst_data,use_muzzin=use_muzzin)
    threedhst_data = calculate_new_colors(threedhst_data,outfolder+'data/fuv.h5',use_masscut=use_masscut,**opts)

    # load prior+mock data
    mock_pars = copy.copy(pars)
    mock_pars[mock_pars.index('massmet_2')] = 'logzsol'
    prior_file = os.getenv('APPS')+'/sfh_prior/data/3dhst_1.0.pickle'
    priors = load_priors(mock_pars+['ml_r0','ml_i0'], prior_file)
    uvj_mock = collate_data(runname,priors,filename=outfolder+'data/dat.h5',**opts)

    # code to compare sSFR posteriors for UVJ-selected galaxies
    '''
    idx_sf, idx_samp = 181, 189
    ssfr_prior = np.clip(np.log10(priors['ssfr_100']),-14,-8)
    ssfr_obj = np.clip(np.log10(uvj_mock['chains']['ssfr_100'][idx_samp]), -14, -8)
    ssfr_sf = np.clip(np.log10(uvj_mock['chains']['ssfr_100'][idx_sf]), -14, -8)  
    plt.hist(ssfr_prior, color='k', lw=2, density=True, histtype='step',label='prior',bins=15)
    plt.hist(ssfr_obj, weights=uvj_mock['weights'][idx_samp], color='red', lw=2, density=True, histtype='step',label='UV,VJ=(0.75,1.45)',bins=15) 
    plt.hist(ssfr_sf, weights=uvj_mock['weights'][idx_sf], color='blue', lw=2, density=True, histtype='step',label='UV,VJ=(0.75,0.65)',bins=8) 

    plt.xlabel(r'log(sSFR/yr$^{-1}$)',fontsize=16)
    plt.ylabel('density',fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.show()
    '''

    # fancy SFH plot
    #fancy_sfh_plot(uvj_mock,runname,outfolder+'UVJ_sfh_dust.pdf',use_muzzin=use_muzzin)

    # alternate UVJ plots
    alternate_uvj(threedhst_data,outfolder+'alternate_UVJ.pdf',use_muzzin=use_muzzin,use_masscut=use_masscut,ssfr_cut=ssfr_cut,**opts)
    print 1/0
    # additional color cuts?
    additional_color_cuts(threedhst_data,outfolder+'additional_color_cuts.pdf',use_masscut=use_masscut,ssfr_cut=ssfr_cut)

    #os.system('open '+outfolder+'additional_color_cuts.png')

    # parameter maps
    #fig, axes = plt.subplots(2,4, sharex=True, sharey=True, figsize=(10,5))
    fig, axes = plt.subplots(2,5, sharex=True, sharey=True, figsize=(12.5,5))
    fig.subplots_adjust(wspace=0.01,hspace=0.01,top=0.865)
    pidx = plot_3d_maps(threedhst_data,pars,plabels,lims=lims,fig=fig,axes=axes[1,:],uv_range=uvj_mock['uv'],vj_range=uvj_mock['vj'],use_muzzin=use_muzzin,smooth=smooth)
    plot_mock_maps(uvj_mock, mock_pars, plabels, lims=lims, fig=fig, axes=axes[0,:],smooth=smooth,pidx=pidx,use_muzzin=use_muzzin)
    apos1,apos2 = axes[0,0].get_position(), axes[1,0].get_position()
    fig.text(apos1.x0-0.07,apos1.y0+apos1.height/2.,'UVJ photometry\nalone',rotation=90,fontsize=14,weight='bold',ha='center',va='center',ma='center')
    fig.text(apos1.x0-0.055,apos2.y0+apos1.height/2.,'19-45 bands\nof photometry',rotation=90,fontsize=14,weight='bold',ha='right',va='center',ma='center')
    plt.savefig(outfolder+'parameter_maps.pdf', dpi=250)
    plt.close()

    # KLD maps
    #fig, axes = plt.subplots(2,4, sharex=True, sharey=True, figsize=(10,5))
    fig, axes = plt.subplots(2,5, sharex=True, sharey=True, figsize=(12.5,5))
    fig.subplots_adjust(wspace=0.01,hspace=0.01,top=0.9,left=0.1)
    pidx = plot_3d_maps(threedhst_data,pars,plabels,fig=fig,axes=axes[1,:],kld=True,cmap='Reds',uv_range=uvj_mock['uv'],vj_range=uvj_mock['vj'],use_muzzin=use_muzzin,smooth=smooth)
    plot_mock_maps(uvj_mock, mock_pars, plabels, priors=priors, cmap='Reds', fig=fig,axes=axes[0,:],pidx=pidx,use_muzzin=use_muzzin,smooth=smooth)
    apos1,apos2 = axes[0,0].get_position(), axes[1,0].get_position()
    fig.text(apos1.x0-0.07,apos1.y0+apos1.height/2.,'UVJ photometry\nalone',rotation=90,fontsize=14,weight='bold',ha='center',va='center')
    fig.text(apos1.x0-0.055,apos2.y0+apos1.height/2.,'19-45 bands\nof photometry',rotation=90,fontsize=14,weight='bold',ha='right',va='center',ma='center')
    plt.savefig(outfolder+'kld_maps.pdf', dpi=250)
    plt.close()

    ### EVERYTHING BELOW THIS COMMENT IS DEPRECATED BUT STILL RUNS
    if make_legacy_plots:
        # quiescent fractions
        # both mocks and 3D-HST uvj_mock
        fig, ax = plt.subplots(1,2, figsize=(8,4))
        plot_mock_qfrac(uvj_mock,ax[0],use_muzzin=use_muzzin)
        plot_3dhst_qfrac(threedhst_data,ax[1],use_muzzin=use_muzzin)
        plt.tight_layout()
        plt.savefig(outfolder+'quiescent_map.png', dpi=150)
        plt.close()

        # for real uvj_mock: what are UVJ-quiescent but actually-starforming galaxies doing?
        plot_3d_hists(threedhst_data,pars,plabels,lims,outfolder+'3dhst_histograms.png')
        plot_interloper_properties(threedhst_data, outfolder+'3dhst_interlopers.png',use_masscut=use_masscut,ssfr_cut=ssfr_cut,**opts)
        rejuvenation(threedhst_data,outfolder+'rejuvenation.png',use_masscut=use_masscut,ssfr_cut=ssfr_cut,**opts)
        sfseq_location(threedhst_data,outfolder+'3dhst_sfseq.png',use_masscut=use_masscut,**opts)
        age_color(threedhst_data,outfolder+'3dhst_age_color.png',**opts)
        plot_quiescent_sfhs(outfolder+'3dhst_interloper_sfhs.png',ax,use_masscut=use_masscut,**opts)

def additional_color_cuts(dat,outname,add_uv=True,use_masscut=False,ssfr_cut=False,plot_ssfr_med=True,**opts):

    # defining indexes of 'true' quiescent and 'star-forming' quiescent
    # then project into color-space
    sf_idx, qu_idx = define_interlopers(dat,use_masscut=use_masscut,mcut_first=True,ssfr_cut=ssfr_cut)
    sf_where, qu_where = np.where(sf_idx)[0], np.where(qu_idx)[0]
    all_where = np.sort(sf_where.tolist()+qu_where.tolist())
    interloper_idx = np.in1d(all_where,sf_where)
    all_idx = np.where(sf_idx | qu_idx)[0]
    mcut_idx = np.where((dat['prosp']['stellar_mass']['q50'] > minmass) & (dat['prosp']['stellar_mass']['q50'] < maxmass))[0]

    # plot options
    popts = {'ms':0.5,'color':'0.5','alpha':0.4,'zorder':-1,'rasterized':True}
    quopts = {'ms':0.5,'color':'red','alpha':0.4,'zorder':-1,'rasterized':True}

    if ssfr_lim:
        lim = ssfr_lim
        ylim = (-14,-7.5)
        yplot = dat['prosp']['ssfr_100']['q50']
        dytxt = 0.1
        ylabel = r'log(sSFR/yr$^{-1}$)'
    else:
        lim = -ms_threesig
        ylim = (-4.5,1.5)
        yplot = (np.log10(dat['prosp']['sfr_100']['q50'])-dat['logsfr_ms'])
        dytxt = 0.05
        ylabel = r'$\Delta$logSFR$_{\mathrm{MS}}$'

    if add_uv:
        #fig, ax = plt.subplots(1,3,figsize=(11,6))
        #fig.subplots_adjust(top=0.48,bottom=0.08,left=0.1,right=0.9)
        #uv_ax = fig.add_axes([0.2, 0.56, 0.26, 0.39])
        #cq_ax = fig.add_axes([0.55, 0.56, 0.26, 0.39])

        fig, axes = plt.subplots(2,2,figsize=(7,7))
        uv_ax = axes[0,0]
        cq_ax = axes[0,1]
        ax = axes[1,:]
        cq_ax.set_title('Optical & NIR colors',fontsize=12,weight='bold')
        uv_ax.set_title('Optical color',fontsize=12,weight='bold')
        ax[0].set_title('FUV-Optical color',fontsize=12,weight='bold')
        ax[1].set_title('Optical-MIR color',fontsize=12,weight='bold')
        plot_dlogsfr_uv(dat,yplot,uv_ax,popts,quopts,use_masscut=use_masscut,plot_ssfr_med=plot_ssfr_med,**opts)
    else:
        fig, ax = plt.subplots(1,2,figsize=(9,6))
        fig.subplots_adjust(top=0.48,bottom=0.08,left=0.2,right=0.8)
        cq_ax = fig.add_axes([0.37, 0.6, 0.26, 0.39])
    plot_dlogsfr_cq(dat,yplot,cq_ax,popts,quopts,use_masscut=use_masscut,plot_ssfr_med=plot_ssfr_med,**opts)
    fs = 9

    # measure deltaSFR, grab indices
    sf_idx, qu_idx = define_interlopers(dat,use_masscut=use_masscut,ssfr_cut=ssfr_cut)
    qu_all_idx = (sf_idx | qu_idx)
    sf_mcut_idx = define_starformers(dat,use_masscut=use_masscut,mcut_first=True,ssfr_cut=ssfr_cut)
    sf_all_idx = define_starformers(dat,use_masscut=use_masscut,ssfr_cut=ssfr_cut)

    # things to plot
    colors = [dat['new_colors']['FUV-V'], dat['new_colors']['V-W3']] #dat['prosp']['ha_ew']['q50']]

    for i, a in enumerate(ax):

        # plot color vs sSFR for quiescent galaxies
        color = np.array(colors[i])
        a.plot(color[all_idx],yplot[qu_all_idx],'o',**quopts)#linestyle=' ',color='k',alpha=0.4,ms=1.5)
        a.plot(color[sf_mcut_idx],yplot[sf_all_idx],'o',**popts)#linestyle=' ',color='k',alpha=0.4,ms=1.5)

        # color-code by F_MIR
        #a.scatter(color,yplot[mcut_idx],marker='o',c=dat['prosp']['fmir']['q50'][mcut_idx],cmap=plt.cm.plasma,s=2,alpha=0.6)

        # plot <sSFR>(x)
        if plot_ssfr_med:
            y,x,bcount = prosp_dutils.running_median(yplot[mcut_idx],color,nbins=25,avg=False,return_bincount=True)
            ok = (bcount > 20)
            a.plot(x[ok],y[ok],'-',lw=2,color='k')

        # pearson coefficient
        coeff = np.corrcoef(color[all_idx],yplot[qu_all_idx])[0,1]
        coeff = np.corrcoef(color,yplot[mcut_idx])[0,1]
        if (i == 0):
            ha, xt = 'left', 0.03
        else:
            ha, xt = 'right', 0.97
        a.text(xt,0.04,r'|R$_{x,y}$|='+"{:.2f}".format(np.abs(coeff)),fontsize=fs*1.1,transform=a.transAxes,ha=ha,color='k',weight='bold')


    # labels and ticklabels
    ax[0].set_xlabel(r'FUV$-$V',fontsize=fs+4)
    ax[1].set_xlabel(r'V$-$W3',fontsize=fs+4)
    #ax[2].set_xlabel(r'H$_{\alpha}$ EW [$\AA]',fontsize=fs+4)
    for a in axes.ravel(): 
        a.set_ylabel(ylabel,fontsize=fs+4)
        #a.axhline(lim,linestyle='--',color='blue',lw=1.5,zorder=2)
        a.set_ylim(ylim)
    #axes[0,0].text(0.0,lim+dytxt,'"star\nforming"',fontsize=fs*1.1,ha='left',va='bottom',ma='center',color='blue')
    ax[0].set_xlim(0.,10)
    ax[0].errorbar(7.,-13.8,xerr=0.5,lw=1.7,color='k',ms=0,capsize=4,capthick=1.7)
    ax[0].text(6.7,-13.7,'colors of local\nellipticals',ma='center',va='center',ha='right',weight='bold',fontsize=6.8,color='black')

    #for a in ax[1:]:         
    #    for tl in a.get_yticklabels():tl.set_visible(False)

    plt.tight_layout()
    #fig.subplots_adjust(wspace=0.0,hspace=0.0)
    plt.savefig(outname,dpi=300)
    plt.close()

def ha_ew_distr(data_3dhst,ax,use_masscut=False,ssfr_cut=False,fs=12):
    """ code from Sandro
    """

    from astropy.table import Table

    ew_min, ew_max = 0, 100

    path_catalogs_3DHST = '/Users/joel/code/python/prospector_alpha/plots/td_delta/fast_plots/data/'
    summary_3DHST_cat = Table.read(path_catalogs_3DHST + '3DHST_combined_catalog.dat', format='ascii')

    obj_name_cat = []

    for ii in range(len(summary_3DHST_cat)):
        if (summary_3DHST_cat[ii]['field'] == 'GOODS-N'):
            obj_name_cat.append('GOODSN_' + str(int(summary_3DHST_cat[ii]['id'])))
        elif (summary_3DHST_cat[ii]['field'] == 'GOODS-S'):
            obj_name_cat.append('GOODSS_' + str(int(summary_3DHST_cat[ii]['id'])))
        else:
            obj_name_cat.append(summary_3DHST_cat[ii]['field'] + '_' + str(int(summary_3DHST_cat[ii]['id'])))

    obj_name_cat = np.array(obj_name_cat)

    def get_prop(idx, prop_id):
        id_list = np.array(data_3dhst['objname'])[idx]
        prop_list = []
        for ii_id in id_list:
            idx_match = (obj_name_cat == ii_id)
            try:
                prop_list.append(summary_3DHST_cat[prop_id][idx_match].data[0])
            except IndexError:
                prop_list.append(np.nan)
        prop_list = np.array(prop_list)
        return(prop_list)

    # check Halpha EW
    sf_idx, qu_idx = define_interlopers(data_3dhst,use_masscut=use_masscut,ssfr_cut=ssfr_cut)
    Ha_list_sf = get_prop(sf_idx, 'HaEQW_flux')
    Ha_list_qu = get_prop(qu_idx, 'HaEQW_flux')

    print 'frac nan = ', 1.0*np.sum(np.isnan(Ha_list_sf))/len(Ha_list_sf)
    print 'frac nan = ', 1.0*np.sum(np.isnan(Ha_list_qu))/len(Ha_list_qu)
    print 'frac -99 = ', 1.0*np.sum(Ha_list_sf < 0.0)/len(Ha_list_sf)
    print 'frac -99 = ', 1.0*np.sum(Ha_list_qu < 0.0)/len(Ha_list_qu)

    # enforce limits
    Ha_list_sf[Ha_list_sf == -99] = np.nan
    Ha_list_qu[Ha_list_qu == -99] = np.nan
    Ha_list_qu = np.clip(Ha_list_qu, ew_min, ew_max)
    Ha_list_sf = np.clip(Ha_list_sf, ew_min, ew_max)

    bins = np.linspace(ew_min, ew_max, 11)
    histopts = {'alpha':0.9, 'lw':2, 'normed':True}
    ax.hist(Ha_list_sf[np.isfinite(Ha_list_sf)], histtype='step', bins=bins,color=interloper_colors[1], **histopts)
    ax.hist(Ha_list_qu[np.isfinite(Ha_list_qu)], histtype='step', bins=bins,color=interloper_colors[0], **histopts)

    ax.set_xlabel(r'$\mathrm{EW}_{\mathrm{H}\alpha} [\AA]$',fontsize=fs*1.3)
    ax.set_yticklabels([])
    ax.set_title('observed\nquantity',weight='semibold')

    # modify tick labels to highlight limits
    ax.set_xticklabels(["don't know what this one does",'<0','50','100+'])
    ax.legend(fontsize=14, frameon=False)

def fancy_sfh_plot(dat,runname,outname,navg=1,add_dust=True,use_muzzin=False):
    """ navg lets you average over N of the first time bins
    """

    # plot geometry
    if add_dust:
        fig, axes = plt.subplots(2,4,figsize=(6.5,6))
        ax,dax = axes[0,:], axes[1,:]
        fig.subplots_adjust(top=0.51,wspace=0.0,hspace=0.38,bottom=0.07)
        uvj_ax = fig.add_axes([0.32, 0.61, 0.36, 0.37])
    else:
        fig, ax = plt.subplots(1,4,figsize=(8,5))
        fig.subplots_adjust(top=0.38,hspace=0.0,wspace=0.0,bottom=0.095)
        uvj_ax = fig.add_axes([0.34, 0.47, 0.32, 0.52])

    # plot options
    median_color = 'k'
    err_color = '0.3'
    draw_color = 'red'
    lw = 2
    ylim_sfh = [-12.5,-8.]
    fs = 12
    ts = fs-4
    arrowalpha = 0.65

    # pick out four colors
    # quiescent (lower left), quiescent (upper right), star-forming (lower left), star-forming (upper-right)
    uv = [1.5,1.9,0.9,1.8]
    vj = [0.7,1.2,0.8,2]

    ### UVJ PLOT
    uvj_ax.set_xlim(vj_lim)
    uvj_ax.set_ylim(uv_lim)
    plot_uvj_box(uvj_ax,lw=2,use_muzzin=use_muzzin)

    # label color points
    for i in range(len(uv)):
        uvj_ax.plot(vj[i],uv[i],'s',color='k',ms=4)
        uvj_ax.text(vj[i],uv[i]+0.1,str(i+1),fontsize=fs+2,ha='center')

    # dust, sSFR arrows
    uvj_ax.annotate('sSFR',
                    (2.2, 0.5),(0.8, 2.),
                    ha="right", va="center",
                    size=fs, color="#3a5cbf",
                    arrowprops=dict(arrowstyle='fancy',
                                shrinkA=5,
                                shrinkB=5,
                                mutation_scale=30,
                                fc="#3a5cbf", ec="#3a5cbf",
                                connectionstyle="arc3,rad=-0.05",
                                alpha=arrowalpha
                                ),
                    bbox=dict(boxstyle="square", fc="w"),zorder=2)

    uvj_ax.annotate('dust',
                    (2.0, 1.8),(0.7, 0.3),
                    ha="right", va="center",
                    size=fs, color='#ec0000',
                    arrowprops=dict(arrowstyle='fancy',
                                shrinkA=5,
                                shrinkB=5,
                                mutation_scale=30,
                                fc="#ec0000", ec="#ec0000",
                                connectionstyle="arc3,rad=-0.05",
                                alpha=arrowalpha
                                ),
                    bbox=dict(boxstyle="square", fc="w"),zorder=2)

    uvj_ax.text(0.05,1.35,"'quiescent'",fontsize=fs-5)
    uvj_ax.text(0.05,1.18,"'star-forming'",fontsize=fs-5)

    # labels, limits
    uvj_ax.set_xlim(vj_lim)
    uvj_ax.set_ylim(uv_lim)
    uvj_ax.set_xlabel('V$-$J',fontsize=fs,labelpad=2)
    uvj_ax.set_ylabel('U$-$V',fontsize=fs)
    uvj_ax.tick_params(labelsize=ts)

    ### SFH PLOT
    for i,a in enumerate(ax):

        # find mock closest to point
        # then load results
        has_results = np.where(np.array(dat['lnlike']) != None)[0]
        idx = has_results[np.sqrt((np.array(dat['uv'])[has_results]-uv[i])**2+(np.array(dat['vj'])[has_results]-vj[i])**2).argmin()]
        res, _, _, eout = load_prospector_data(None,runname=runname,objname=runname+'_'+dat['names'][idx])
        eout['sfh']['sfh'] = eout['sfh']['sfh'][:,::2]/(10**res['chain'][eout['sample_idx'],0])[:,None]

        # plot median, +/- 1sigma
        nt = 7-(navg-1)
        perc = np.zeros(shape=(nt,3))
        for j in range(nt):
            if j == 0:
                perc[j,:] = np.percentile((eout['sfh']['sfh'][:,:navg] * agediff[:navg]).sum(axis=1) / agediff[:navg].sum(),[16,50,84])
            else: 
                perc[j,:] = np.percentile(eout['sfh']['sfh'][:,j+navg-1],[16,50,84])

        perc = np.log10(perc)
        aplot = np.concatenate(([np.mean(agemid[:navg])],agemid[navg:]))
        a.plot(aplot, perc[:,1],'-',color=median_color,lw=lw)
        a.fill_between(aplot, perc[:,0], perc[:,2], color=err_color, alpha=0.3)
        #a.plot(aplot, perc[:,0],'-',color=err_color,lw=lw)
        #a.plot(aplot, perc[:,2],'-',color=err_color,lw=lw)

        # now plot N draws from SFH
        ndraws = 10
        for n in range(ndraws): 
            adraw = np.concatenate(([(eout['sfh']['sfh'][n,:navg] * agediff[:navg]).sum() / agediff[:navg].sum()],eout['sfh']['sfh'][n,navg:]))
            a.plot(aplot,np.log10(adraw),'-',color=draw_color,alpha=0.35,lw=lw-1.25,zorder=-1)

        # show averaging timescale
        if (navg > 1):
            tscale = int(np.around(10**mod.params['agebins'][navg-1,1]/1e6/100.)*100)
            a.errorbar([a.get_xlim()[0],tscale/1e3],
                       [ylim_sfh[0]+0.1,ylim_sfh[0]+0.1],
                       lw=1,elinewidth=1, color='k')
            xt = 10**((np.log10(a.get_xlim()[0]+0.1) + np.log10(tscale/1e3))/1.8)
            a.text(xt,ylim_sfh[0]+0.4,'{0} Myr\naverage'.format(tscale),fontsize=ts-3,ha='center',va='center',ma='center')
        a.set_xlim(aplot.min(),aplot.max())

        # limits, labels, text
        a.set_ylim(ylim_sfh)
        a.set_xlabel(r't$_{\mathrm{lookback}}$ [Gyr]',fontsize=fs,labelpad=0)
        a.text(0.96,0.98,str(i+1),fontsize=fs+2,transform=a.transAxes,va='top',ha='right')
        a.set_xscale('log',subsx=(1,3),nonposx='clip')
        a.xaxis.set_major_formatter(FormatStrFormatter('%2.5g'))
        a.xaxis.set_minor_formatter(FormatStrFormatter('%2.5g'))
        a.tick_params('both', pad=3.5, size=3.5, width=1.0, which='both',labelsize=ts)
        if (i > 0):
            for tl in a.get_yticklabels():tl.set_visible(False)

        # if we're adding dust PDFs
        if add_dust:
            dhist = res['chain'][eout['sample_idx'],res['theta_labels'].index('dust2')]
            n, b, p = dax[i].hist(dhist, bins=np.linspace(0,2.5,26),
                                  histtype="step",color='k',
                                  range=[dhist.min(),dhist.max()],
                                  weights=eout['weights'])
            dax[i].set_xlim(0,2.5)
            dax[i].set_xlabel(r'$\tau_{\mathrm{dust}}$',fontsize=fs,labelpad=0)
            for tl in dax[i].get_yticklabels():tl.set_visible(False)
            if (i <= 2):
                dax[i].text(0.96,0.98,str(i+1),fontsize=fs+2,transform=dax[i].transAxes,va='top',ha='right')
            else:
                dax[i].text(0.04,0.98,str(i+1),fontsize=fs+2,transform=dax[i].transAxes,va='top',ha='left')


    ax[0].set_ylabel(r'log(sSFR/yr$^{-1}$)',fontsize=fs)
    if add_dust:
        dax[0].set_ylabel('dust posterior\nprobability density')

    plt.savefig(outname,dpi=300)
    plt.close()

def calculate_new_colors(dat, filename=None, regenerate_fuv=False, use_masscut=False, **opts):
    
    ### if it's already made, load it and give it back
    # else, start with the making!
    if os.path.isfile(filename) and (regenerate_fuv == False):
        with open(filename, "r") as f:
            indict = hickle.load(f)
        dat['new_colors'] = indict
        return dat

    # pick out only UVJ-quiescent galaxies
    idxs = np.where(dat['prosp']['stellar_mass']['q50'] > minmass)[0]

    # load parameter files, filters
    import td_delta_params as pfile 
    filters = ['galex_FUV', 'galex_NUV','wise_w3']
    fobs = {'filters': observate.load_filters(filters), 'wavelength': None}
    nsamp = 100

    colors = ['FUV-U', 'NUV-U', 'FUV-V', 'NUV-V','V-W3','J-W3']
    colors = ['FUV-V','V-W3','U-V','V-J'] # narrow down range
    out = {'median': {color:[] for color in colors},
           'chain': {color:np.zeros(shape=(nsamp,idxs.shape[0])) for color in colors}}
    for j, idx in enumerate(idxs):

        # load output
        objname = dat['objname'][idx]
        name = os.getenv('APPS')+'/prospector_alpha/results/td_delta/'+objname.split('_')[0]+'/'+objname
        res, _, _, eout = load_prospector_data(name)
        if res is None:
            print objname+'not postprocessed, making it now'
            import postprocessing
            param_name = os.getenv('APPS')+'/prospector_alpha/parameter_files/td_delta_params.py'
            postprocessing.post_processing(param_name, objname=objname,sps=sps,plot=False)
            res, _, _, eout = load_prospector_data(name)

        # load model, set redshift to zero
        mod = pfile.load_model(zred=eout['zred'],**pfile.run_params)
        mod.params['zred'] = 0.0
        if j == 0:
            sps = pfile.load_sps(**pfile.run_params)

        # sample from posterior
        fout = np.zeros(shape=(nsamp,len(filters)))
        for i in range(nsamp): _,fout[i,:],_ = mod.mean_model(res['chain'][eout['sample_idx'][i],:], fobs, sps=sps)

        # calculate magnitudes
        fuv, nuv, w3 = (25 - 2.5*np.log10(fout)).T
        uband = 25 - 2.5*np.log10(eout['obs']['uvj'][:nsamp,0])
        vband = 25 - 2.5*np.log10(eout['obs']['uvj'][:nsamp,1])
        jband = 25 - 2.5*np.log10(eout['obs']['uvj'][:nsamp,2])

        # calculate colors
        weights = res['weights'][eout['sample_idx']][:nsamp]
        #out['FUV-U'] += wquantile(fuv-uband,[0.5],weights=weights)
        #out['NUV-U'] += wquantile(nuv-uband,[0.5],weights=weights)
        out['median']['FUV-V'] += wquantile(fuv-vband,[0.5],weights=weights)
        #out['NUV-V'] += wquantile(nuv-vband,[0.5],weights=weights)
        out['median']['V-W3'] += wquantile(vband-w3,[0.5],weights=weights)
        #out['J-W3'] += wquantile(jband-w3,[0.5],weights=weights)
        out['median']['V-J'] += wquantile(vband-jband,[0.5],weights=weights)
        out['median']['U-V'] += wquantile(uband-vband,[0.5],weights=weights)

        # include distributions
        out['chain']['FUV-V'][:,j] = fuv-vband
        out['chain']['V-W3'][:,j] = vband-w3
        out['chain']['V-J'][:,j] = vband-jband
        out['chain']['U-V'][:,j] = uband-vband

        print 'done '+str(j)

    hickle.dump(out,open(filename, "w"))
    dat['new_colors'] = out
    return dat

def add_extra_quantities(threedhst_data,use_muzzin=False):
    """ a little helper function to add useful definitions
    """

    threedhst_data['kld']['avg_age'] = threedhst_data['kld']['mwa']
    threedhst_data['fast']['uvj_dust_prosp'] = threedhst_data['fast']['uvj_dust_prosp'].astype(bool)
    threedhst_data['logsfr_ms'] = sfr_ms(threedhst_data['fast']['z'],threedhst_data['prosp']['stellar_mass']['q50'],adjust_sfr=-0.35)
    if use_muzzin:
        a,b = 0.66,0.75
    else:
        a,b = 0.625,0.781
    threedhst_data['sq_color'] = a*threedhst_data['fast']['vj']+b*threedhst_data['fast']['uv']
    threedhst_data['cq_color'] = -a*threedhst_data['fast']['vj']+b*threedhst_data['fast']['uv']

    return threedhst_data

def collate_data(runname, prior, filename=None, regenerate=False, **opts):
    
    ### if it's already made, load it and give it back
    # else, start with the making!
    if os.path.isfile(filename) and (regenerate == False):
        with open(filename, "r") as f:
            outdict = pickle.load(f)
        return outdict

    ### define output containers
    out = {'names':[],'weights':[],'log':{'ssfr_100':True,'avg_age':True},'pars':{},'chains':{},'kld':{},'lnlike':[]}
    entries = ['ssfr_100','dust2','avg_age','logzsol']
    mls = ['ml_g0','ml_r0','ml_i0']
    entries += mls
    for e in entries: 
        out['pars'][e] = []
        out['chains'][e] = []
        out['kld'][e] = []
    out['uv'], out['vj'] = [], []

    ### fill them up
    out['names'] = [str(i+1) for i in range(625)]
    for i, name in enumerate(out['names']):
        res, _, _, eout = load_prospector_data(None,runname=runname,objname=runname+'_'+name)
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
            if key in eout['thetas'].keys(): # normal parameter
                out['pars'][key] += [eout['thetas'][key]['q50']]
                idx = res['theta_labels'].index(key)
                #out['chains'][key] += [res['chain'][eout['sample_idx'],idx].flatten()]
                out['kld'][key] += [kld(res['chain'][:,idx],res['weights'],prior[key])]
            elif key in eout['extras'].keys(): # calculated in postprocessing
                out['pars'][key] += [eout['extras'][key]['q50']]
                out['chains'][key] += [eout['extras'][key]['chain'].flatten()]
                out['kld'][key] += [kld(np.log10(eout['extras'][key]['chain']),eout['weights'],np.log10(prior[key]))]
            else: # we're doing it live (M/Ls)
                idx = mls.index(key)
                chain = (eout['extras']['stellar_mass']['chain']/np.array(eout['obs']['rf'][:,idx]))/ml_sun[idx]
                out['pars'][key] += [wquantile(chain,[0.5,0.84,0.16],weights=eout['weights'])[0]]
                #out['chains'][key] += [chain.flatten()]
                out['kld'][key] += [kld(chain,eout['weights'],np.log10(prior[key]))]


    ### dump files and return
    pickle.dump(out,open(filename, "w"))

    return out

def load_priors(pars,outloc,ndraw=100000,regenerate_prior=False):
    """ return ``ndraw'' samples from the prior 
    works for any model theta and many SFH parameters
    """

    # if we have an analytical function, sample at regular intervals across the function
    # otherwise, we sample numerically
    out = {}
    with open(outloc, "rb") as f:
        sfh_prior = pickle.load(f)
    for par in pars:
        out[par] = {}
        if par in mod.free_params:
            out[par] = mod._config_dict[par]['prior']
        elif (par == 'ssfr_100'):
            out[par] = sfh_prior['ssfr']
        elif (par == 'avg_age'):
            out[par] = sfh_prior['mwa']
        else:
            out[par] = sfh_prior[par]

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
    bidx = ~np.isfinite(chain)
    if (bidx.sum() > 0):
        chain[bidx] = chain[~bidx].min()

    sorted = np.sort(chain)
    nskip = np.floor(chain.shape[0]/float(nbins)).astype(int)-1
    bins = sorted[::nskip]
    bins[-1] = sorted[-1]  # ensure the maximum bin is the maximum of the chain
    assert bins.shape[0] == nbins+1
    pdf, bins = np.histogram(chain, bins=bins,weights=weights)

    return pdf, bins

def define_interlopers(dat,use_masscut=True,mcut_first=False,ssfr_cut=False):

    mcut = (dat['prosp']['stellar_mass']['q50'] > minmass) & (dat['prosp']['stellar_mass']['q50'] < maxmass)# & (dat['fast']['z'] < 1.0) & (dat['fast']['z'] > 0.5)
    if (use_masscut) & (mcut_first):
        idx = mcut
    else:
        idx = np.ones_like(dat['logsfr_ms'],dtype=bool)
    uvj_flag = dat['fast']['uvj_prosp'][idx]
    sfr = np.log10(dat['prosp']['sfr_100']['q50'])[idx]
    logsfr_ms = dat['logsfr_ms'][idx]

    uvj_quiescent = (uvj_flag == 3)
    if ssfr_cut:
        sfing = (dat['prosp']['ssfr_100']['q50'][idx] > ssfr_lim)
    else:
        sfing = (sfr > logsfr_ms-ms_threesig)

    sf_idx = uvj_quiescent & (sfing)
    qu_idx = (uvj_flag == 3) & (~sfing)
    if (use_masscut) & (not mcut_first):
        sf_idx = sf_idx & mcut
        qu_idx = qu_idx & mcut
    return sf_idx, qu_idx

def define_starformers(dat,use_masscut=True,mcut_first=False,ssfr_cut=False):

    mcut = (dat['prosp']['stellar_mass']['q50'] > minmass) & (dat['prosp']['stellar_mass']['q50'] < maxmass)# & (dat['fast']['z'] < 1.0) & (dat['fast']['z'] > 0.5)
    if (use_masscut) & (mcut_first):
        idx = mcut
    else:
        idx = np.ones_like(dat['logsfr_ms'],dtype=bool)
    logsfr_ms = dat['logsfr_ms'][idx]

    if ssfr_cut:
        sfing = (dat['prosp']['ssfr_100']['q50'][idx] > ssfr_lim)
    else:
        sfing = (np.log10(dat['prosp']['sfr_100']['q50'])[idx] > logsfr_ms-ms_threesig)

    sf_idx = ((dat['fast']['uvj_prosp'][idx] == 1) | (dat['fast']['uvj_prosp'][idx] == 2)) & sfing
    if (use_masscut) & (not mcut_first):
        sf_idx = sf_idx & mcut
    return sf_idx

def plot_quiescent_sfhs(ax,fs=12,regenerate_sfh=False,use_masscut=False,ssfr_cut=False,**opts):

    # data
    dat = stack_quiescent_sfhs(regenerate_sfh=regenerate_sfh,use_masscut=use_masscut,ssfr_cut=ssfr_cut)
    time = np.logspace(7.3,9.95,200)/1e9  # PULL FROM DICT AFTER RERUN

    # plot options
    #fig, ax = plt.subplots(1,2,figsize=(7.5,3.5))
    lopts = {'alpha': 0.9, 'linestyle': '-', 'lw': 3}
    fillopts = {'alpha': 0.15, 'zorder': -1}

    for i, key in enumerate(['qu','sf']):

        ax.fill_between(time, dat[key]['errdown'], dat[key]['errup'], 
                           color=interloper_colors[i], **fillopts)
        ax.plot(time, dat[key]['median'],
                   color=interloper_colors[i], **lopts)

        #ax[i].set_xlim(opts['xlim_t'])
        ax.set_ylim(3e-13,1e-9)
        ax.set_xlabel(r'time [Gyr]',fontsize=fs*1.3)
        ax.set_ylabel(r'SFR/M$_{\mathrm{formed}}$ [yr$^{-1}$]',fontsize=fs*1.3)

        ax.set_xscale('log',subsx=(1,3),nonposx='clip')
        ax.xaxis.set_major_formatter(FormatStrFormatter('%2.5g'))
        ax.xaxis.set_minor_formatter(FormatStrFormatter('%2.5g'))
        ax.set_yscale('log',nonposy='clip')
        ax.tick_params('both', pad=3.5, size=3.5, width=1.0, which='both',labelsize=fs)

def stack_quiescent_sfhs(sfhdat=None, tdat=None, regenerate_sfh=True,use_masscut=False,ssfr_cut=False,**opts):
    """ taken from stack_td_sfh on 12/20/18
    """

    dloc = os.getenv('APPS') + '/prospector_alpha/plots/td_delta/fast_plots/data/uvj_sfh.h5'
    if not regenerate_sfh:
        with open(dloc, "r") as f:
            out = hickle.load(f)
        return out

    # load SFH chains
    if sfhdat is None:
        dloc = os.getenv('APPS') + '/prospector_alpha/plots/td_delta/fast_plots/data/stacksfh.h5'
        with open(dloc, "r") as f:
            sfhdat = hickle.load(f)
        #return sfhdat

    # load 3D-HST data
    if tdat is None:
        with open('/Users/joel/code/python/prospector_alpha/plots/td_delta/fast_plots/data/fastcomp.h5', "r") as f:
            tdat = hickle.load(f)
        #return tdat

    tdat = add_extra_quantities(tdat)

    # define interlopers, translate to SFH indexes
    sfidx, quidx = define_interlopers(tdat,use_masscut=use_masscut,ssfr_cut=ssfr_cut)
    sf_idx, qu_idx = [np.where(np.in1d(sfhdat['objname'],np.array(tdat['objname'])[idx]))[0] for idx in [sfidx,quidx]]

    # intermediate array for interpolation
    ssfrmin, ssfrmax = -13, -8 # intermediate interpolation onto regular sSFR grid
    ssfr_arr = 10**np.linspace(ssfrmin,ssfrmax,1001)

    # define stack properties and output dictionary
    out = {'qu':{'idx':qu_idx,'median_sfh':np.zeros(shape=(7,quidx.sum())), 'time_sfh': np.zeros(shape=(7,quidx.sum()))},\
           'sf':{'idx':sf_idx,'median_sfh':np.zeros(shape=(7,sfidx.sum())), 'time_sfh': np.zeros(shape=(7,sfidx.sum()))}}
    nt = 200
    ntime_array = 14 # = N_SFH_BINS * 2
    nsamps = 3000 # number of posterior samples
    tbins = np.logspace(7.3,9.95,nt)/1e9  # upper limit is t_universe at z=0.5

    # stack for both definitions
    for key in out.keys():

        # pull out sSFH and weights for these objects without creating a numpy array of the whole survey (which takes forever)
        idxs = out[key]['idx']
        ngal = idxs.shape[0]
        ssfr = np.empty(shape=(nsamps,nt,ngal))
        ssfh, weights, times = np.empty(shape=(nsamps,ntime_array,ngal)), np.empty(shape=(nsamps,ngal)), np.empty(shape=(ntime_array,ngal))
        for m, idx in enumerate(idxs): 
            weights[:,m] = sfhdat['weights'][idx]
            ssfh[:,:,m] = sfhdat['ssfh'][idx]
            times[:,m] = sfhdat['sfh_t'][idx]

            # save medians and times
            for nn in range(7): out[key]['median_sfh'][nn,m] = wquantile((sfhdat['ssfh'][idx][:,::2])[:,nn], [0.5],weights=weights[:,m])[0]
            out[key]['time_sfh'][:,m] = sfhdat['sfh_t'][idx][0]

        # now loop over each output time to calculate distributions
        # must interpolate (SFR / M) chains onto regular time grid
        # we do this with nearest-neighbor interpolation which is precisely correct for step-function SFH
        for key2 in ['median','err','errup','errdown']: out[key][key2] = []
        for i in range(nt):

            # find where each galaxy contributes
            # only include a galaxy in a bin if it overlaps with the bin!
            b_idx = tbins[i] < times.max(axis=0)
            ssfrs = ssfh[:,np.abs(times - tbins[i]).argmin(axis=0)[b_idx],b_idx]
            weights_for_hist = weights[:,b_idx].flatten()
            ssfr_for_hist = np.clip(ssfrs,10**ssfrmin,10**ssfrmax).flatten()

            # now histogram and calculate quantiles
            hist,_ = np.histogram(ssfr_for_hist, density=False, weights=weights_for_hist, bins=ssfr_arr)
            median, errup, errdown = wquantile((ssfr_arr[1:]+ssfr_arr[:-1])/2., [0.5,.84,.16],weights=hist)

            out[key]['median'] += [median]
            out[key]['errup'] += [errup]
            out[key]['errdown'] += [errdown]

        # save and dump
        out[key]['err'] = prosp_dutils.asym_errors(np.array(out[key]['median']),
                                                   np.array(out[key]['errup']),
                                                   np.array(out[key]['errdown']))

    out['t'] = tbins
    hickle.dump(out,open(dloc, "w"))

def rejuvenation(tdat, outname, regenerate_sfh=False,use_masscut=False,ssfr_cut=False, **opts):
    stack = stack_quiescent_sfhs(sfhdat=None, tdat=tdat, regenerate_sfh=regenerate_sfh,use_masscut=use_masscut,ssfr_cut=ssfr_cut, **opts)


    # here we play with different definitions of 'rejuvenating'
    # from the photometric SFHs
    sfh_100myr = stack['sf']['median_sfh'][0,:]*0.3 + stack['sf']['median_sfh'][1,:]*0.7


    idx_rejuv = (stack['sf']['median_sfh'][0,:] > stack['sf']['median_sfh'][1,:])  # 0-30 > 30-100 : 31%
    idx_rejuv = sfh_100myr > stack['sf']['median_sfh'][2,:] # 0-100 Myr > (next bin): 27%
    idx_rejuv = (stack['sf']['median_sfh'][0,:] / stack['sf']['median_sfh'][1,:]) > 2 # 0-30 Myr / 30-100 Myr > 2: 3%
    idx_rejuv = (sfh_100myr/stack['sf']['median_sfh'][2,:]) > 2 # 0-100 Myr / (next bin) > 2: 3%
    print 'fraction of rejuvenating galaxies: {0:1f}'.format(idx_rejuv.sum() / float(idx_rejuv.shape[0]))

def age_color(dat,outname,redshift_cut=True,**opts):

    fig, ax = plt.subplots(1,1, figsize=(3.5,3.5))

    idx = (dat['fast']['uvj_prosp'] == 3) & (dat['prosp']['stellar_mass']['q50'] > minmass) & (dat['prosp']['stellar_mass']['q50'] < maxmass)
    if redshift_cut:
        idx = idx & (dat['fast']['z'] > 1.5) & (dat['fast']['z'] < 2.5)
    age = dat['prosp']['half_time']['q50'][idx]
    color = dat['sq_color'][idx]

    # plot measured relationship
    ax.plot(color,np.log10(age*1e9),'o',ms=2.4,alpha=0.8,color='k',mew=0.0)

    # running median
    x1, y1, bincount = prosp_dutils.running_median(color,np.log10(age*1e9),avg=False,return_bincount=True,nbins=15)
    ok = bincount > 10
    ax.plot(x1[ok],y1[ok],'-',color='#4d4dff',lw=3)

    # plot sirio's relationship
    x = np.linspace(color.min(),color.max(),400)
    y = 7.03 + 1.12 * x
    ax.plot(x,y,'-',color='red',lw=3)
    ax.plot(x,y-0.3,'--',color='red',lw=1.1)
    ax.plot(x,y+0.3,'-',color='red',lw=1.1)
    ax.set_xlabel('S$_Q$')
    ax.set_ylabel('median age [yr]')
    if redshift_cut:
        ax.text(0.05,0.95,'1.5 < z < 2.5',transform=ax.transAxes)

    ax.set_ylim(8.6,9.8)

    plt.tight_layout()
    plt.savefig(outname,dpi=150)
    plt.close()

def plot_dlogsfr_uv(dat,yplot,ax,popts,quopts,fs=9,lw=2,use_muzzin=False,use_masscut=False,plot_ssfr_med=False,**opts):
    """ C_Q versus deltalogSFR
    indicate "UVJ-quiescent" with vertical line
    and "MS-quiescent" with horizontal line
    in corner, indicate % of star-forming which are quiescent, etc.
    """

    # plot
    uvj_qu_idx = (dat['fast']['uvj_prosp'] == 3)
    uvj_sf_idx = (dat['fast']['uvj_prosp'] == 1) | (dat['fast']['uvj_prosp'] == 2)
    if use_masscut:
        mcut =  (dat['prosp']['stellar_mass']['q50'] > minmass) &  (dat['prosp']['stellar_mass']['q50'] < maxmass)
        uvj_qu_idx = uvj_qu_idx & mcut
        uvj_sf_idx = uvj_sf_idx & mcut

    ax.plot(dat['fast']['uv'][uvj_qu_idx],yplot[uvj_qu_idx],'o',label='UVJ\nquiescent',**quopts)
    ax.plot(dat['fast']['uv'][uvj_sf_idx],yplot[uvj_sf_idx],'o',label='UVJ\nstarforming',**popts)

    if plot_ssfr_med:
        y,x,bcount = prosp_dutils.running_median(yplot[mcut],dat['fast']['uv'][mcut],nbins=25,avg=False,return_bincount=True)
        ok = (bcount > 20)
        ax.plot(x[ok],y[ok],'-',lw=2,color='k',label='median sSFR(x)')

    ax.set_xlabel(r'U$-$V',fontsize=fs*1.3)

    # corr coef
    coeff = np.corrcoef(dat['fast']['uv'][uvj_qu_idx],yplot[uvj_qu_idx])[0,1]
    ax.text(0.97,0.04,r'|R$_{x,y}$|='+"{:.2f}".format(np.abs(coeff)),fontsize=fs*1.1,transform=ax.transAxes,ha='right',weight='bold')

    # legend hax
    legend = ax.legend(loc=3,prop={'size':7},frameon=True)
    for leg in legend.legendHandles: 
        leg._legmarker.set_markersize(4)
        leg._legmarker.set_alpha(0.9)

    # text
    #ax.text(0.03,0.16,'UVJ\nstar-forming',transform=ax.transAxes,color=popts['color'],fontsize=fs+1,ha='left',ma='left',weight='bold')
    #ax.text(0.03,0.04,'UVJ\nquiescent',transform=ax.transAxes,color=quopts['color'],fontsize=fs+1,ha='left',ma='left',weight='bold')

def plot_dlogsfr_cq(dat,yplot,ax,popts,quopts, fs=9,lw=2,use_muzzin=False,use_masscut=False,plot_ssfr_med=False,**opts):
    """ C_Q versus deltalogSFR
    indicate "UVJ-quiescent" with vertical line
    and "MS-quiescent" with horizontal line
    in corner, indicate % of star-forming which are quiescent, etc.
    """

    # plot
    uvj_qu_idx = (dat['fast']['uvj_prosp'] == 3)
    uvj_sf_idx = (dat['fast']['uvj_prosp'] == 1) | (dat['fast']['uvj_prosp'] == 2)
    if use_masscut:
        mcut =  (dat['prosp']['stellar_mass']['q50'] > minmass) &  (dat['prosp']['stellar_mass']['q50'] < maxmass)
        uvj_qu_idx = uvj_qu_idx & mcut
        uvj_sf_idx = uvj_sf_idx & mcut
    ax.plot(dat['cq_color'][uvj_qu_idx],yplot[uvj_qu_idx],'o',**quopts)
    ax.plot(dat['cq_color'][uvj_sf_idx],yplot[uvj_sf_idx],'o',**popts)

    if plot_ssfr_med:
        y,x,bcount = prosp_dutils.running_median(yplot[mcut],dat['cq_color'][mcut],nbins=25,avg=False,return_bincount=True)
        ok = (bcount > 20)
        ax.plot(x[ok],y[ok],'-',lw=2,color='k')

    ax.set_xlabel(r'sSFR direction in UVJ-space',fontsize=fs*1.3)

    # add UVJ-quiescent line ("C_Q")
    if use_muzzin:
        uvj_qu = 0.44238
    else:
        uvj_qu = 0.54655

    # corr coef
    coeff = np.corrcoef(dat['cq_color'][uvj_qu_idx],yplot[uvj_qu_idx])[0,1]
    ax.text(0.03,0.04,r'|R$_{x,y}$|='+"{:.2f}".format(np.abs(coeff)),fontsize=fs*1.1,transform=ax.transAxes,ha='left',weight='bold')

def sfseq_location(dat,outname,ax=None,fig=None,fs=16,sederrs=False,use_masscut=False,**opts):
    """ distribution in mass, sSFR, metallicity, dust, age
    and average SEDs
    """

    # setting up plot geometry + options
    if ax is None:
        fig, ax = plt.subplots(1,2, figsize=(9,4.5))
        fig.subplots_adjust(hspace=0.0,bottom=0.15,top=0.8)

    # defining indexes of 'true' quiescent and 'star-forming' quiescent
    sf_idx, qu_idx = define_interlopers(dat,use_masscut=use_masscut)
    background_idx = (~sf_idx) & (~qu_idx)
    ntot = float(sf_idx.sum()+qu_idx.sum())

    # pull out physics
    # we ignore asymmetric errors for now
    sfr = np.log10(dat['prosp']['sfr_100']['q50'])
    sfr_err = (np.log10(dat['prosp']['sfr_100']['q84']/dat['prosp']['sfr_100']['q50']) + np.log10(dat['prosp']['sfr_100']['q50']/dat['prosp']['sfr_100']['q16']))/2.
    mass = dat['prosp']['stellar_mass']['q50']
    zred = dat['fast']['z']
    delta_sfr = sfr - dat['logsfr_ms']

    # plot!
    popts = {'marker':'o','cmap':'plasma','s':4,'alpha':0.65}
    for idx in [qu_idx,sf_idx]: 
        if (len(ax) > 1):
            pts_cq = ax[0].scatter(mass[idx],delta_sfr[idx], c=dat['cq_color'][idx], vmin=0.45,vmax=0.8,**popts)
            pts_sq = ax[1].scatter(mass[idx],delta_sfr[idx], c=dat['sq_color'][idx], vmin=1.3,vmax=2.7,**popts)
        else:
            pts_cq = ax[0].scatter(mass[idx],delta_sfr[idx], c=dat['cq_color'][idx], vmin=0.45,vmax=0.8,**popts)
            pts_sq = None
    pts = [pts_cq,pts_sq]
    labels = [r'C$_Q$ (perpendicular)',r'S$_Q$ (parallel)']
    for i, a in enumerate(ax):
        a.plot(mass[background_idx],delta_sfr[background_idx],'o',ms=0.5,color='0.5',alpha=0.4,zorder=-1)

        pos = a.get_position()
        xpad, x_width = 0.005, 0.03
        cax = fig.add_axes([pos.x1+xpad, pos.y0, x_width, pos.height]) 

        cbar = fig.colorbar(pts[i], cax=cax)
        cbar.solids.set_rasterized(True)
        cbar.solids.set_edgecolor("face")
        cax.set_ylabel(labels[i],labelpad=5,fontsize=fs*1.3)
        cax.yaxis.set_label_position('right')
        cbar.ax.tick_params(labelsize=fs-4)

        a.set_xlabel(r'log(M/M$_{\odot}$)',fontsize=fs*1.3)
        a.axhline(-0.9,linestyle='--',color='k',lw=2,zorder=1)
        a.text(11.99,-0.85,'star-forming\nselection',ha='right',fontsize=fs*0.9,va='bottom',ma='center')

        a.set_xlim(minmass,12)
        a.set_ylim(-4,2)

    ax[0].set_ylabel(r'$\Delta$log(SFR)',fontsize=fs*1.3)
    if (len(ax) > 1):
        for tl in ax[1].get_yticklabels():tl.set_visible(False)
        plt.savefig(outname,dpi=200)
        plt.close()

def plot_interloper_properties(dat,outname,sederrs=False,density=True,use_masscut=False,ssfr_cut=False,**opts):
    """ distribution in mass, sSFR, metallicity, dust, age
    and average SEDs
    """

    # setting up plot geometry
    """
    fig, axes = plt.subplots(1,6, figsize=(12,6))
    fig.subplots_adjust(bottom=0.65,top=0.9,left=0.24,right=0.98)
    sfhax = fig.add_axes([0.08,0.1,0.39,0.4])
    sedax = fig.add_axes([0.56,0.1,0.39,0.4])
    """
    fig, axes = plt.subplots(1,6, figsize=(8,12))
    #fig.subplots_adjust(bottom=0.86,top=0.96,left=0.05,right=0.98)
    xstart, xwid, ywid = 0.22,0.7, 0.2
    msax = fig.add_axes([xstart-0.05,0.78,xwid,ywid])
    fig.subplots_adjust(bottom=0.59,top=0.69,left=0.15,right=0.98)
    sedax = fig.add_axes([xstart,0.32,xwid,ywid])
    sfhax = fig.add_axes([xstart,0.05,xwid,ywid])

    # setting up plot options
    lims = [(minmass,12),(-14,-8),(0,2.4),(-2,0.2),(0,10)]
    nbins = 20
    histopts = {'drawstyle':'steps-mid','alpha':0.9, 'lw':2, 'linestyle': '-'}
    sedopts = {'fmt': 'o-', 'alpha':0.8,'lw':1, 'elinewidth':2.0,'capsize':2.0,'capthick':2}
    fs = 14

    # make SFH, MS plot
    plot_quiescent_sfhs(sfhax,fs=fs,use_masscut=use_masscut,ssfr_cut=ssfr_cut)
    sfseq_location(dat,outname,ax=[msax],fig=fig,sederrs=False,fs=fs,use_masscut=use_masscut)

    # defining plot parameters
    pars = ['stellar_mass','ssfr_100','dust2','massmet_2','avg_age']
    plabels = [r'log(M/M$_{\odot}$)',r'sSFR$_{\mathrm{100\;Myr}}$', r'$\tau_{\mathrm{dust}}$', r'log(Z/Z$_{\odot}$)', r'<age>/Gyr'] # 'dust index'

    # defining indexes of 'true' quiescent and 'star-forming' quiescent
    sf_idx, qu_idx = define_interlopers(dat,use_masscut=use_masscut)
    ntot = float(sf_idx.sum()+qu_idx.sum())

    """
    # interlopers are a little dustier
    fig, ax = plt.subplots(1,1)
    #ax.plot(dat['prosp']['dust2']['q50'][qu_idx],dat['prosp']['ssfr_100']['q50'][qu_idx],'o',linestyle=' ',color='red',alpha=0.7,ms=3.5)
    #ax.plot(dat['prosp']['dust2']['q50'][sf_idx],dat['prosp']['ssfr_100']['q50'][sf_idx],'o',linestyle=' ',color='blue',alpha=0.7,ms=3.5)
    plt.show()
    print 1/0
    """

    # figure text
    """
    xs, dx, ys, dy = 0.02, 0.00, 0.88, 0.035
    weight = 'semibold'
    fig.text(xs,ys,'selected as',fontsize=fs,weight=weight)
    fig.text(xs,ys-dy,'UVJ-quiescent',fontsize=fs,weight=weight)
    fig.text(xs,ys-2*dy,r'and log(M/M$_{\odot}$)>'+"{:.1f}".format(minmass),fontsize=fs,weight=weight)
    fig.text(xs+dx,ys-3.5*dy,r'SFR<(SFR$_{\mathrm{MS}}-3\sigma_{\mathrm{MS}}$)'+' (' + str(int(np.round(qu_idx.sum()/ntot*100)))+'%)',fontsize=fs,color=interloper_colors[0])
    fig.text(xs+dx,ys-4.5*dy,r'SFR>(SFR$_{\mathrm{MS}}-3\sigma_{\mathrm{MS}}$)'+' (' + str(int(np.round(sf_idx.sum()/ntot*100)))+'%)',fontsize=fs,color=interloper_colors[1])
    """
    xs,ys = 0.065,0.71
    fig.text(xs,ys,'legend:',fontsize=fs,weight='bold')
    fig.text(xs+0.12,ys,r'SFR<(SFR$_{\mathrm{MS}}-3\sigma_{\mathrm{MS}}$)'+' (' + str(int(np.round(qu_idx.sum()/ntot*100)))+'%),',fontsize=fs,color=interloper_colors[0])
    fig.text(xs+0.46,ys,r'SFR>(SFR$_{\mathrm{MS}}-3\sigma_{\mathrm{MS}}$)'+' (' + str(int(np.round(sf_idx.sum()/ntot*100)))+'%)',fontsize=fs,color=interloper_colors[1])
    
    # labels
    xt = 0.01
    fig.text(xt,0.88,'star-forming\nsequence',rotation=90,fontsize=fs*1.4,weight='bold',va='center',ma='center')
    fig.text(xt,0.64,'derived\nproperties',rotation=90,fontsize=fs*1.4,weight='bold',va='center',ma='center')
    fig.text(xt,0.42,'median SED',rotation=90,fontsize=fs*1.4,weight='bold',va='center',ma='center')
    fig.text(xt,0.15,'derived\nSFHs',rotation=90,fontsize=fs*1.4,weight='bold',va='center',ma='center')

    # plot histograms
    for i, ax in enumerate(axes[:-1]):
        data = dat['prosp'][pars[i]]['q50']

        x, hist1, hist2 = hist(data,qu_idx, sf_idx, nbins, lims[i],density=density)
        ax.plot(x,hist1,color=interloper_colors[0], **histopts)
        ax.plot(x,hist2,color=interloper_colors[1], **histopts)
        ax.set_ylim(0.0,ax.get_ylim()[1]*1.2)

        # titles and labels
        ax.set_xlim(lims[i])
        ax.set_xlabel(plabels[i],fontsize=fs)
        ax.set_yticklabels([])

    # finish with H-alpha histogram
    ha_ew_distr(dat,axes[-1],use_masscut=use_masscut,ssfr_cut=ssfr_cut)

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

    sedax.errorbar(midbins, qu, yerr=qu_errs,color=interloper_colors[0], **sedopts)
    sedax.errorbar(midbins, sf, yerr=sf_errs,color=interloper_colors[1], **sedopts)

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

    # add filters
    fnames = ['bessell_U','bessell_V','twomass_J'] # ['galex_FUV', 'galex_NUV'
    filters = observate.load_filters(fnames)

    dyn = 0.003
    for j, f in enumerate(filters): 
        smoothed_trans = f.transmission/f.transmission.max()
        if fnames[j] == 'twomass_J':
            smoothed_trans = norm_kde(f.transmission/f.transmission.max(),4)
        sedax.plot(f.wavelength/1e4, smoothed_trans*dyn+ymin,lw=1.5,color='0.3')
        sedax.text(f.wave_effective/1e4, dyn*1.5, fnames[j][-1], color='0.3',fontsize=16,ha='center',weight='bold')

    plt.savefig(outname,dpi=200)
    plt.close()

def discretize(data, nbins):
    split = np.array_split(np.sort(data), nbins)
    cutoffs = [x[-1] for x in split]
    return cutoffs

def alternate_uvj(dat,outloc,use_muzzin=False,use_masscut=False,selection='pure',ssfr_cut=False,**opts):

    # use mass-cut (or not) to define sample
    idx = np.ones_like(dat['prosp']['stellar_mass']['q50'])
    if use_masscut:
        idx = dat['prosp']['stellar_mass']['q50'] > minmass

    # define color-coding
    popts = {'marker':'o','cmap':'RdYlBu','s':0.5,'alpha':0.35,'rasterized':True}

    # define plot
    xleft = 0.88
    fig, ax = plt.subplots(1,3,figsize=(11,3.5))
    fig.subplots_adjust(right=xleft)
    fs, lw = 12, 2.

    # define plot variables
    if ssfr_cut:
        cpar = dat['prosp']['ssfr_100']['q50'][idx]
        sfing = cpar > ssfr_lim
        cpar_label = r'log(sSFR/yr$^{-1}$)'
        popts['vmin'], popts['vmax'] = -13, -8
    else:
        cpar = (np.log10(dat['prosp']['sfr_100']['q50'])-dat['logsfr_ms'])[idx]
        sfing = cpar > (-ms_threesig)
        cpar_label = r'$\Delta$logSFR$_{\mathrm{MS}}$'
        popts['vmin'], popts['vmax'] = -3, 1

    n_qu_tot, n_sf_tot = float((~sfing).sum()), float((sfing).sum())

    pvars = {'xlabel': ['V$-$J','V$-$J','V$-$W3'],
             'ylabel': ['U$-$V','FUV$-$V','FUV$-$V'],
             'xvar': [np.array(dat['new_colors']['median']['V-J']),np.array(dat['new_colors']['median']['V-J']),np.array(dat['new_colors']['median']['V-W3'])],
             'yvar': [np.array(dat['new_colors']['median']['U-V']),np.array(dat['new_colors']['median']['FUV-V']),np.array(dat['new_colors']['median']['FUV-V'])],
             'xvar_samps': [np.array(dat['new_colors']['chain']['V-J']),np.array(dat['new_colors']['chain']['V-J']),np.array(dat['new_colors']['chain']['V-W3'])],
             'yvar_samps': [np.array(dat['new_colors']['chain']['U-V']),np.array(dat['new_colors']['chain']['FUV-V']),np.array(dat['new_colors']['chain']['FUV-V'])]
             }

    if selection == 'pure':
        tstr = 'pure'
    elif selection == 'complete':
        tstr = 'complete'
    else:
        tsfr = 'both'

    # our minimization function + starting points
    pstart = [[0.92,0.75,1.49,1.46], [3.89, 0.8, 4.6, 1.38],[2.91,2.77,3.01,1.32]]
    def min_fnct(x,args):
        """fit a line with a special objective function
           x is an array of (SLOPE,INTERCEPT,YSTOP,XSTOP)
           args[0]: x-color
           args[1]: y-color
           args[2]: delta logSFR_MS
           return: (N_STARFORMING_ID + N_QUIESCENT_ID) / N_TOTAL  (i.e., percentage of total correct classifications)
        """
        slope, intercept, ystop, xstop = x
        xcolor, ycolor, deltalsfr = args

        # separate star-forming / quiescent based on colors
        y_expected = slope*xcolor + intercept
        sfing_color = (y_expected > ycolor) | (ycolor < ystop) | (xcolor > xstop)

        if selection == 'pure':
            n_sf = float(sfing_color.sum())
            n_qu = float((~sfing_color).sum())
            eff = ((~sfing_color) & (~sfing)).sum()/n_qu + (sfing_color & sfing).sum()/n_sf
        elif selection == 'complete':
            n_sf = n_sf_tot
            n_qu = n_qu_tot
            eff = ((~sfing_color) & (~sfing)).sum()/n_qu + (sfing_color & sfing).sum()/n_sf 
        else:
            n_sf = n_sf_tot
            n_qu = n_qu_tot
            eff =  ((~sfing_color) & (~sfing)).sum()/n_qu + (sfing_color & sfing).sum()/n_sf
            n_sf = float(sfing_color.sum())
            n_qu = float((~sfing_color).sum())
            eff += ((~sfing_color) & (~sfing)).sum()/n_qu + (sfing_color & sfing).sum()/n_sf

        # hack to avoid n_qu = 0!
        if np.isnan(eff):
            return np.inf 

        return -eff

    # plot
    for i, a in enumerate(ax):
        pts = a.scatter(pvars['xvar'][i],pvars['yvar'][i],c=cpar,**popts)
        a.set_xlabel(pvars['xlabel'][i],fontsize=fs)
        a.set_ylabel(pvars['ylabel'][i],fontsize=fs)

        # add divisions
        if False: # UVJ, but now we optimize so never do this
            plot_uvj_box(a,use_muzzin=use_muzzin,lw=lw)

            # calculate success rates
            sf_uvj = (dat['fast']['uvj_prosp'][idx] == 1) | (dat['fast']['uvj_prosp'][idx] == 2)
            qu_uvj = (dat['fast']['uvj_prosp'][idx] == 3)

            sf_pure = (sf_uvj & sfing).sum() / float(sf_uvj.sum()) * 100
            qu_pure = (qu_uvj & (~sfing)).sum() / float(qu_uvj.sum()) * 100

            sf_complete = (sf_uvj & sfing).sum() / n_sf_tot * 100
            qu_complete = ((qu_uvj) & (~sfing)).sum() / n_qu_tot * 100

        else: # we make our own
            res = minimize(min_fnct, pstart[i], method='Nelder-Mead', options={'xtol': 1e-8, 'disp': True, 'maxiter':1000}, args=[pvars['xvar'][i],pvars['yvar'][i],cpar])

            # print best-fit
            print res.x

            # calculate success rates, drawing from posterior to simulate 'noise'
            sf_pure, qu_pure, sf_complete, qu_complete = [[] for k in range(4)]
            for j in range(100):
                xsamp, ysamp = pvars['xvar_samps'][i][j,:], pvars['yvar_samps'][i][j,:]
                y_expected = res.x[0]*xsamp + res.x[1]
                sfing_color = (y_expected > ysamp) | (ysamp < res.x[2]) | (xsamp > res.x[3])

                sf_pure += [(sfing_color & sfing).sum() / float(sfing_color.sum()) * 100]
                qu_pure += [((~sfing_color) & (~sfing)).sum() / float((~sfing_color).sum()) * 100]

                sf_complete += [(sfing_color & sfing).sum() / n_sf_tot * 100]
                qu_complete += [((~sfing_color) & (~sfing)).sum() / n_qu_tot * 100]

            sf_pure = np.median(sf_pure)
            sf_complete = np.median(sf_complete)
            qu_pure = np.median(qu_pure)
            qu_complete = np.median(qu_complete)

            # plot line
            xlim, ylim = np.array(a.get_xlim()), np.array(a.get_ylim())
            x0 = (res.x[2]-res.x[1])/res.x[0]
            y0 = res.x[0]*res.x[3]+res.x[1]
            a.plot([x0,res.x[3]],[res.x[2],y0],linestyle='-',color='k',lw=lw,zorder=3)
            a.plot([xlim[0],x0],[res.x[2],res.x[2]],linestyle='-',color='k',lw=lw,zorder=3)
            a.plot([res.x[3],res.x[3]],[y0,ylim[1]],linestyle='-',color='k',lw=lw,zorder=3)
            a.set_xlim(xlim)
            a.set_ylim(ylim)

        # note efficiency
        if selection == 'pure':
            a.text(0.97,0.05,'SF: {0}% pure'.format(int(np.round(sf_pure))),fontsize=fs-2,transform=a.transAxes,ha='right')
            a.text(0.97,0.1,'QU: {0}% pure'.format(int(np.round(qu_pure))),fontsize=fs-2,transform=a.transAxes,ha='right')
        elif selection == 'complete':
            a.text(0.97,0.05,'SF: {0}% complete'.format(int(np.round(sf_complete))),fontsize=fs-2,transform=a.transAxes,ha='right')
            a.text(0.97,0.1,'QU: {0}% complete'.format(int(np.round(qu_complete))),fontsize=fs-2,transform=a.transAxes,ha='right')
        else:
            a.text(0.97,0.05,'SF: {0}% pure/{1}% complete'.format(int(np.round(sf_pure)), int(np.round(sf_complete))),fontsize=fs-2,transform=a.transAxes,ha='right')
            a.text(0.97,0.1,'QU: {0}% pure/{1}% complete'.format(int(np.round(qu_pure)), int(np.round(qu_complete))),fontsize=fs-2,transform=a.transAxes,ha='right')

    # colorbar
    fig.tight_layout(rect=(0,0,xleft,1)) # adjust first for better results
    xpad, x_width = 0.03, 0.03
    pos = ax[2].get_position()
    cax = fig.add_axes([xleft, pos.y0, x_width, pos.height]) 

    cbar = fig.colorbar(pts, cax=cax)
    cbar.solids.set_rasterized(True)
    cbar.solids.set_edgecolor("face")
    cax.set_ylabel(cpar_label,labelpad=5,fontsize=fs*1.3)
    cax.yaxis.set_label_position('right')
    cbar.ax.tick_params(labelsize=fs-4)

    # custom limits
    ax[0].set_xlim(-0.2,2.8)
    ax[0].set_ylim(0,2.7)
    ax[1].set_xlim(-0.2,2.8)
    ax[1].set_ylim(-0.1,9)
    ax[2].set_ylim(-0.1,9)

    plt.savefig(outloc,dpi=300)
    plt.close()

def plot_3dhst_qfrac(dat, ax,use_muzzin=False):  
    """ this needs to be rewritten to do probabilistic assignments
    based on the full PDF, i.e. properly incorporating sSFR error bars
    in the quiescent fraction values
    """

    # pull out data, define ranges
    idx = dat['prosp']['stellar_mass']['q50'] > minmass
    uv = dat['fast']['uv'][idx]
    vj = dat['fast']['vj'][idx]
    nbins_3dhst = 20
    uv_range = np.linspace(uv_lim[0],uv_lim[1],nbins_3dhst)
    vj_range = np.linspace(vj_lim[0],vj_lim[1],nbins_3dhst)

    # generate map
    qvals = dat['prosp']['ssfr_100']['q50'][idx]
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
    plot_uvj_box(ax,use_muzzin=use_muzzin)

def plot_mock_qfrac(dat, ax,use_muzzin=False):
    
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
    plot_uvj_box(ax,use_muzzin=use_muzzin)

def lnlike_cut(lnlike):

    #return (lnlike < -45.5) | (~np.isfinite(lnlike))
    return (lnlike < -45.5) | (~np.isfinite(lnlike))

def plot_mock_maps(dat, pars, plabels, lims=None, smooth=True, priors=None, cmap='plasma', fig=None, axes=None, pidx=None,use_muzzin=False):

    # pull out data, define ranges
    logpars = ['massmet_2','ssfr_100','avg_age']
    uv = np.array(dat['uv'])
    vj = np.array(dat['vj'])
    nbin = int(np.sqrt(uv.shape))
    fs = 13.5
    ts = fs - 4

    #idx_poor = lnlike_cut(np.array(dat['lnlike']).astype(float))

    # construct maps
    for i, par in enumerate(pars):

        idx_poor = (~pidx[i].T.flatten()) | (np.array(dat['lnlike']) == None)

        # either KLD map w/ prior
        # or median of PDF
        if priors is not None:
            vmap = np.array(dat['kld'][par])
            ax = None
        else:
            vmap = np.array(dat['pars'][par])
            if par in ['avg_age','ssfr_100','ml_g0']:
                vmap = np.log10(vmap)
            if par is 'ssfr_100':
                vmap = np.clip(vmap,-13,np.inf)
            ax = axes[i]

        vmap[idx_poor] = np.nan
        valmap = vmap.reshape(nbin,nbin).swapaxes(0,1)
        if smooth:
            valmap = nansmooth(valmap,sigma=smooth)

        # plot options
        if lims is None:
            vmin, vmax = kl_vmin, kl_vmax
        else:
            vmin, vmax = lims[i]
            print vmin, vmax
            print np.nanmin(valmap), np.nanmax(valmap)

        # show map
        img = axes[i].imshow(valmap, origin='lower', cmap=cmap,
                             extent=(0,2.5,0,2.5),vmin=vmin,vmax=vmax)
        axes[i].tick_params('both', labelsize=ts)

        #if i == (len(pars)-1):
        #    print 1/0

        # colorbar
        if priors is None:
            pos = axes[i].get_position()
            ypad, y_width = 0.005, 0.03
            xpad = 0.01
            cax = fig.add_axes([pos.x0+xpad, pos.y1+ypad, pos.width-xpad*2, y_width]) 

            ticks = np.linspace(vmin,vmax,5)
            cbar = fig.colorbar(img, cax=cax, aspect=10,orientation='horizontal',ticks=ticks)
            cbar.solids.set_rasterized(True)
            cbar.solids.set_edgecolor("face")
            cax.xaxis.set_ticks_position('top')
            cax.set_title(plabels[i],pad=20,fontsize=fs)
            cbar.ax.tick_params(labelsize=ts-2)
            cbar.ax.set_xticklabels(["{:.1f}".format(x) for x in ticks])
            #if i != (len(pars)-1): plt.setp(cax.get_xticklabels()[-1], visible=False)

        else:
            axes[i].set_title(plabels[i],fontsize=fs)

        """
        we do a single colorbar in the 3dhst function now
        else:
            if (i == len(pars) - 1):
                pos = axes[i].get_position()
                xpad, x_width = 0.005, 0.03
                cax = fig.add_axes([pos.x1+xpad, pos.y0, x_width, pos.height]) 

                cbar = fig.colorbar(img, cax=cax, aspect=10)
                cbar.solids.set_rasterized(True)
                cbar.solids.set_edgecolor("face")
                cbar.set_label('KL divergence',size=fs)
                cbar.ax.tick_params(labelsize=ts)
        """
    # axis labels
    axes[0].set_ylabel('U-V',fontsize=fs)
    if priors is not None:
        for a in axes.tolist(): a.set_xlabel('V-J',fontsize=fs)
        plt.setp(axes[0].get_yticklabels()[-1], visible=False)
        #for a in axes[:-1].tolist(): plt.setp(a.get_xticklabels()[-1], visible=False)

    # UVJ box
    for a in axes.tolist(): plot_uvj_box(a,use_muzzin=use_muzzin)

def plot_3d_maps(dat,pars,plabels,lims=None,kld=False,smooth=True,density=False,cmap='plasma',fig=None,axes=None,uv_range=None,vj_range=None,use_muzzin=False):

    # make plot geometry
    fs = 13.5
    ts = fs - 4

    # pull out data, define ranges
    logpars = ['massmet_2','ssfr_100','avg_age']
    tolog = ['fmir']

    sidx = (dat['prosp']['stellar_mass']['q50'] > minmass) #& (dat['fast']['z'] > 0.75) & (dat['fast']['z'] < 1.25)
    uv = dat['fast']['uv'][sidx] 
    vj = dat['fast']['vj'][sidx]

    # convert mock list of UVJ colors
    # into N_colors and list of edges for bins
    uv_lim, vj_lim = (0,2.5), (0,2.5)
    uv_range, vj_range = np.unique(uv_range), np.unique(vj_range)
    nbins_3dhst = uv_range.shape[0]
    pidx = []

    # averages
    for i, par in enumerate(pars):

        qvals = dat['prosp'][par]['q50'][sidx]
        if (par == 'avg_age') | (par == 'ml_g0'):
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
                        avgmap[k,j] = np.nanmedian(dat['kld'][par][sidx][idx])
                        #avgmap[k,j] = np.nanpercentile(dat['kld'][par][sidx][idx],[95])
                    elif par in logpars:
                        avgmap[k,j] = np.log10(np.median(10**qvals[idx]))
                    elif par in tolog:
                        avgmap[k,j] = np.log10(np.median(qvals[idx]))
                    else:
                        avgmap[k,j] = np.median(qvals[idx])

        if smooth:
            avgmap = nansmooth(avgmap)

        if kld:
            ax = None
        else:
            ax = axes[i]

        # plot options
        if lims is None:
            vmin, vmax = kl_vmin, kl_vmax
        else:
            vmin, vmax = lims[i]

        # show map
        img = axes[i].imshow(avgmap, origin='lower', cmap=cmap,
                             extent=(vj_lim[0],vj_lim[1],uv_lim[0],uv_lim[1]),
                             vmin=vmin,vmax=vmax)
        axes[i].tick_params('both', labelsize=ts)
        pidx += [np.isfinite(avgmap)]

        # colorbar
        """
        this is done in mock_maps now!
        if not kld:
            pos = axes[i].get_position()
            ypad, y_width = 0.005, 0.03
            cax = fig.add_axes([pos.x0, pos.y1+ypad, pos.width, y_width]) 

            cbar = fig.colorbar(img, cax=cax, aspect=10,orientation='horizontal')
            cbar.solids.set_rasterized(True)
            cbar.solids.set_edgecolor("face")
            cax.xaxis.set_ticks_position('top')
            cax.set_title(plabels[i],pad=20,fontsize=fs)
            cax.xaxis.set_major_locator(MaxNLocator(6))
            cbar.ax.tick_params(labelsize=ts)
        """
        if kld:
            if (i == len(pars) - 1):
                pos = axes[i].get_position()
                xpad, x_width = 0.005, 0.03
                cax = fig.add_axes([pos.x1+xpad, pos.y0, x_width, 2*pos.height+0.01]) 

                cbar = fig.colorbar(img, cax=cax, aspect=10)
                cbar.solids.set_rasterized(True)
                cbar.solids.set_edgecolor("face")
                cbar.set_label(r'information gained (D$_{\mathrm{KL}}$)',size=fs)
                cbar.ax.tick_params(labelsize=ts)

                # add limits
                ticks = cbar.get_ticks()
                tick_labels = ['{0:.2f}'.format(t) for t in ticks]
                tick_labels[-1] = tick_labels[-1]+'+'
                cbar.set_ticks(ticks)
                cbar.set_ticklabels(tick_labels)

                #plt.setp(cax.get_xticklabels()[-1], visible=False)


    # axis labels
    axes[0].set_ylabel('U-V',fontsize=fs)
    for a in axes.tolist(): a.set_xlabel('V-J',fontsize=fs)
    plt.setp(axes[0].get_yticklabels()[-1], visible=False)
    for a in axes[:-1].tolist(): plt.setp(a.get_xticklabels()[-1], visible=False)

    # UVJ box
    for a in axes.tolist(): plot_uvj_box(a,use_muzzin=use_muzzin)
    return pidx

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
    plabels = ['sSFR [100 Myr]', r'$\tau_{\mathrm{dust}}$', r'log(Z/Z$_{\odot}$)', r'mean age'] # 'dust index'
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
    sf_idx = ((dat['fast']['uvj_prosp'] == 1) | (dat['fast']['uvj_prosp'] == 2)) & (dat['prosp']['stellar_mass']['q50'] > minmass)
    qu_idx = (dat['fast']['uvj_prosp'] == 3) & (dat['prosp']['stellar_mass']['q50'] > minmass)
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
    qu_contam = (np.log10(dat['prosp']['sfr_100']['q50'][qu_idx]) > dat['logsfr_ms'][qu_idx]-ms_threesig).sum() / float(qu_idx.sum())
    sf_contam = (np.log10(dat['prosp']['sfr_100']['q50'][sf_idx]) < dat['logsfr_ms'][sf_idx]-ms_threesig).sum() / float(sf_idx.sum())
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

def plot_uvj_box(ax,lw=2,use_muzzin=False):

    # (Muzzin+13 use: U - V = (V - J)*0.88 + 0.59, [1.0 < z < 4.0])
    # this would give (0.806818182,1.3)
    # line is UV = 0.8*VJ+0.7  (Whitaker+12)
    # constant UV = 1.3, VJ = 1.5
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    if use_muzzin:
        ax.plot([xlim[0],0.807],[1.3,1.3],linestyle='-',color='k',lw=lw)
        ax.plot([0.807,1.5],[1.3,1.91],linestyle='-',color='k',lw=lw)
        ax.plot([1.5,1.5],[1.91,ylim[1]],linestyle='-',color='k',lw=lw)
    else:
        ax.plot([xlim[0],0.75],[1.3,1.3],linestyle='-',color='k',lw=lw)
        ax.plot([0.75,1.5],[1.3,1.9],linestyle='-',color='k',lw=lw)
        ax.plot([1.5,1.5],[1.9,ylim[1]],linestyle='-',color='k',lw=lw)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

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



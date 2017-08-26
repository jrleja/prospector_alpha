import numpy as np
from prosp_diagnostic_plots import add_sfh_plot
import os
import matplotlib.pyplot as plt
import magphys_plot_pref
import matplotlib as mpl
from prosp_dutils import running_median, smooth_spectrum
from composite_images import collate_spectra as stolen_collate_spectra
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
from matplotlib.ticker import MaxNLocator

mask_si_abs = (8.5,12.5)

minorFormatter = magphys_plot_pref.jLogFormatter(base=10, labelOnlyBase=False)
majorFormatter = magphys_plot_pref.jLogFormatter(base=10, labelOnlyBase=True)

#### choose galaxies
to_plot = ['IRAS 08572+3915','NGC 3690', 'NGC 7674','UGC 08696', 'CGCG 049-057', 'NGC 4552'] # in a specific order

def median_by_band(x,y,avg=False,log=False):

    ##### get filter effective wavelengths for sorting
    delz = 0.15
    from brownseds_np_params import translate_filters
    from sedpy import observate

    filtnames = np.array(translate_filters(0,full_list=True))
    filts = observate.load_filters(filtnames[filtnames != 'nan'])
    wave_effective = np.array([filt.wave_effective for filt in filts])/1e4
    wave_effective.sort()

    ### remove overlapping filters
    ### and ones outside the plot
    delta_lam = 0.06
    for i in range(wave_effective.shape[0]):
        if i >= wave_effective.shape[0]:
            continue
        if ((np.abs(1-wave_effective/wave_effective[i]) < delta_lam).sum() > 1):
            wave_effective = np.delete(wave_effective,i)

    wave_effective = wave_effective[wave_effective < 25]


    avglam, outval, outval_up, outval_down, outval_upup, outval_downdown = [np.array([]) for i in range(6)]
    for lam in wave_effective:
        #in_bounds = (x <= lam) & (x > lam/(1+delz))
        in_bounds = (np.abs(1-lam/x) < delta_lam)
        avglam = np.append(avglam, np.mean(x[in_bounds]))
        
        mid, top, bot, toptop, botbot = np.percentile(y[in_bounds], [50, 84, 16, 95, 5])
        outval_up = np.append(outval_up, top)
        outval_down = np.append(outval_down, bot)
        outval_upup = np.append(outval_upup, toptop)
        outval_downdown = np.append(outval_downdown, botbot)
        
        if avg:
            mean, sig = y[in_bounds].mean(), np.std(y[in_bounds])
            outval = np.append(outval, mean)
            #outval_up = np.append(outval_up, mean+sig)
            #outval_down = np.append(outval_down, mean-sig)
            #outval_upup = np.append(outval_upup, mean+2*sig)
            #outval_downdown = np.append(outval_downdown, mean-2*sig)
        else:
            outval = np.append(outval, mid)

    if log:
        return avglam, -np.log10(1-outval), -np.log10(1-outval_up), -np.log10(1-outval_down), -np.log10(1-outval_upup), -np.log10(1-outval_downdown)
    else:
        return avglam, outval, outval_up, outval_down, outval_upup, outval_downdown


def collate_data(alldata, alldata_noagn):

    ### package up information
    the_data = [alldata, alldata_noagn]
    data_label = ['agn','no_agn']
    output = {}
    for ii in xrange(2):

        #### generate containers
        # SFH
        sfh = {
               'perc':[],
               't_sfh':[],
              }

        # residuals
        residuals = {}
        labels = ['Optical','Akari','Spitzer IRS']
        for l in labels: residuals[l] = {'lam':[],'resid':[]}

        # phot residuals
        phot_residuals = {'resid':[],'lam_obs':[]}

        # model parameters
        objname = []
        model_pars = {}
        pnames = ['fagn', 'agn_tau','duste_qpah']
        for p in pnames: 
            model_pars[p] = {'q50':[],'q84':[],'q16':[]}
        parnames = alldata[0]['pquantiles']['parnames']

        #### load model information
        for dat in the_data[ii]:

            #### model parameters [NEW MODEL ONLY]
            objname.append(dat['objname'])
            if data_label[ii] == 'agn':
                for key in model_pars.keys():
                    model_pars[key]['q50'].append(dat['pquantiles']['q50'][parnames==key][0])
                    model_pars[key]['q84'].append(dat['pquantiles']['q84'][parnames==key][0])
                    model_pars[key]['q16'].append(dat['pquantiles']['q16'][parnames==key][0])

            #### SFH
            sfh['t_sfh'].append(dat['pextras']['t_sfh'])
            perc = np.zeros(shape=(len(sfh['t_sfh'][-1]),3))
            for jj in xrange(perc.shape[0]): perc[jj,:] = np.percentile(dat['pextras']['sfh'][jj,:],[16.0,50.0,84.0])
            sfh['perc'].append(perc)

            #### photometric residuals
            phot_residuals['resid'].append(dat['residuals']['phot']['chi_prosp'])
            phot_residuals['lam_obs'].append(dat['residuals']['phot']['lam_obs']/(1+dat['residuals']['phot']['z']))

            #### spectral residuals
            for key in residuals.keys():
                if key in dat['residuals'].keys():
                    residuals[key]['lam'].append(dat['residuals'][key]['obs_restlam'])
                    residuals[key]['resid'].append(dat['residuals'][key]['prospector_resid'])
                else:
                    residuals[key]['lam'].append(None)
                    residuals[key]['resid'].append(None)

        #### numpy arrays
        for key in residuals.keys(): 
            for key2 in residuals[key].keys():
                residuals[key][key2] = np.array(residuals[key][key2])
        for key in sfh.keys(): sfh[key] = np.array(sfh[key])
        for key in phot_residuals.keys(): phot_residuals[key] = np.array(phot_residuals[key])
        for key in model_pars.keys(): 
            for key2 in model_pars[key].keys():
                model_pars[key][key2] = np.array(model_pars[key][key2])

        output[data_label[ii]] = {}
        output[data_label[ii]]['objname'] = objname
        output[data_label[ii]]['model_pars'] = model_pars
        output[data_label[ii]]['sfh'] = sfh
        output[data_label[ii]]['residuals'] = residuals
        output[data_label[ii]]['phot_residuals'] = phot_residuals

    return output

def plot_comparison(idx_plot=None,outfolder=None,
                    runname='brownseds_agn',runname_noagn='brownseds_np',alldata=None,alldata_noagn=None, **popts):

    #### load alldata
    if alldata is None:
        import brown_io

        alldata = brown_io.load_alldata(runname=runname)
        alldata_noagn = brown_io.load_alldata(runname=runname_noagn)

    #### make output folder if necessary
    if outfolder is None:
        outfolder = os.getenv('APPS')+'/prospector_alpha/plots/'+runname+'/agn_plots/'
        if not os.path.isdir(outfolder):
            os.makedirs(outfolder)
    
    ### collate data
    ### choose galaxies with largest 10 F_AGN
    pdata = collate_data(alldata,alldata_noagn)
    if idx_plot is None:
        idx_plot = pdata['agn']['model_pars']['fagn']['q50'].argsort()[-10:]
    
    ### take collate data from composite images
    fig, ax = plt.subplots(2,3, figsize=(15, 10))
    ax = ax.ravel()
    fig = plot_residuals(pdata,idx_plot,outfolder,r'High f$_{\mathrm{AGN,MIR}}$',ax=ax[:3],fig=fig,**popts)
    fig = plot_residuals(pdata,np.arange(len(pdata['agn']['model_pars']['fagn']['q50'])),outfolder,'Full sample',ax=ax[3:],fig=fig,**popts)
    fig.tight_layout()
    fig.savefig(outfolder+'residuals.png',dpi=150)
    plt.close()

    plot_sfh(pdata,idx_plot,outfolder,**popts)

    pdata_stolen = {'fmir':[], 'fmir_up':[], 'fmir_do':[]}
    for name,idx in zip(to_plot,idx_plot):
        plt_idx = pdata['agn']['objname'].index(name)
        if name not in np.array(pdata['agn']['objname'])[idx_plot]:
            idx_plot = np.concatenate((idx_plot,np.atleast_1d(plt_idx)))

        pdata_stolen['fmir'].append(pdata['agn']['model_pars']['fagn']['q50'][plt_idx])
        pdata_stolen['fmir_up'].append(pdata['agn']['model_pars']['fagn']['q84'][plt_idx] - pdata_stolen['fmir'][-1])
        pdata_stolen['fmir_do'].append(pdata_stolen['fmir'][-1] - pdata['agn']['model_pars']['fagn']['q16'][plt_idx])

    pdata_stolen = stolen_collate_spectra(alldata,alldata_noagn,idx_plot,pdata_stolen,runname,['WISE W1','WISE W2'])
    fig = sedfig(pdata_stolen,**popts)
    fig.savefig(outfolder+'sed_residuals.png',dpi=150)
    plt.close()


def add_txt(ax,pdata,fs=12,x=0.05,y=0.88,dy=0.075,ha='left',color=None,**extras):

    ax.text(x,y-dy,'NO AGN',fontsize=fs,transform=ax.transAxes,ha=ha,color=color[1],**extras)
    ax.text(x,y,'AGN',fontsize=fs,transform=ax.transAxes,ha=ha,color=color[0],**extras)

def add_identifier(ax,idx,pdata,fs=12,x=0.94,y=0.88,dy=0.08,weight='bold'):

    ax.text(x,y,pdata['agn']['objname'][idx],fontsize=fs,transform=ax.transAxes,ha='right',weight=weight)

    mid = pdata['agn']['model_pars']['fagn']['q50'][idx]
    fmt = "{{0:{0}}}".format(".2f").format
    text = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
    text = text.format(fmt(mid), \
                       fmt(mid-pdata['agn']['model_pars']['fagn']['q16'][idx]), \
                       fmt(pdata['agn']['model_pars']['fagn']['q84'][idx]-mid))
    text = "{0} = {1}".format(r'f$_{\mathrm{AGN,MIR}}$', text)

    ax.text(x,y-dy,text,fontsize=fs,transform=ax.transAxes,ha='right')

def plot_sfh(pdata,idx_plot,outfolder,**popts):

    ### open figure
    fig, ax = plt.subplots(3,4, figsize=(14, 10))

    ax = np.ravel(ax)
    fs = 11.5
    idx_plot = idx_plot[::-1]

    ### begin loop
    for ii,idx in enumerate(idx_plot):

        if ii >= 12:
            continue

        pmin,pmax = np.inf, -np.inf
        for key in pdata.keys():

            #### load SFH properties
            t = pdata[key]['sfh']['t_sfh'][idx]
            perc = pdata[key]['sfh']['perc'][idx]
            color = popts[key.replace('_','')+'_color']

            ### plot SFH
            ax[ii].plot(t, perc[:,1],'-',color=color,lw=2.5,alpha=0.9)
            ax[ii].fill_between(t, perc[:,0], perc[:,2], color=color, alpha=0.15)

            ### labels and ranges
            pmin,pmax = np.min([pmin,perc.min()]), np.max([pmax,perc.max()])
            if (ii == 0) | (ii == 4) | (ii == 8):
                ax[ii].set_ylabel(r'SFR [M$_{\odot}$/yr]',fontsize=fs*1.5)
            if (ii >= 8):
                ax[ii].set_xlabel('lookback time [Gyr]',fontsize=fs*1.5)

            ax[ii].set_xscale('log',nonposx='clip',subsx=([1]))
            ax[ii].xaxis.set_minor_formatter(minorFormatter)
            ax[ii].xaxis.set_major_formatter(majorFormatter)
            for tl in ax[ii].get_xticklabels():tl.set_visible(False)

            ax[ii].set_yscale('log',nonposy='clip',subsy=([1]))
            ax[ii].yaxis.set_minor_formatter(minorFormatter)
            ax[ii].yaxis.set_major_formatter(majorFormatter)
            for tl in ax[ii].get_yticklabels():tl.set_visible(False)

        add_identifier(ax[ii],idx,pdata, fs=fs,weight='bold',y=0.92,x=0.975)
        add_txt(ax[ii],pdata,fs=fs,color=[popts['agn_color'],popts['noagn_color']],weight='bold',y=0.92,x=0.025)

        ax[ii].set_ylim(pmin*0.2,pmax*8)
        ax[ii].set_xlim(t.min()*30,t.max())

    # turn off unused plots
    for i in xrange(ii+1,ax.shape[0]): ax[i].axis('off')

    plt.tight_layout(h_pad=0.1,w_pad=0.06)
    plt.savefig(outfolder+'sfh_comparison.png',dpi=150)
    plt.close()

def plot_residuals(pdata,idx_plot,outfolder,instring,avg=True,log=False,ax=None,fig=None,**popts):

    #### plot geometry
    if ax is None:
        fig, ax = plt.subplots(2,2, figsize=(11, 10))
        ax = ax.ravel()
    fs = 14

    #### plot options
    median_opts = {'alpha':0.8,'lw':3}

    #### data holders
    photx, photy_noagn, photy_agn = [],[],[]
    specx, specy_noagn, specy_agn = [[],[],[]], [[],[],[]], [[],[],[]]

    xlim_phot = [0.2,30]
    #ylim_phot = [-0.4,0.4]
    ylim_phot = [-2.7,2.7]
    ylim_phot = [-0.2,25]
    if log:
        ylim_phot = [-0.15,0.15]
    ylim_spec = [[-.38,.38], #[[-.35,.35],
                 [-.38,.38]]
    xlim_spec = [[2.4,4.84],[5,34]]#[[0.35,0.7],

    ### begin loop over galaxies
    for idx in idx_plot:

        #### include galaxy photometry
        photx += (pdata['agn']['phot_residuals']['lam_obs'][idx]/1e4).tolist()
        photy_noagn += (pdata['no_agn']['phot_residuals']['resid'][idx]**2).tolist()
        photy_agn += (pdata['agn']['phot_residuals']['resid'][idx]**2).tolist()

        #### include galaxy spectroscopy
        for i,key in enumerate(['Akari','Spitzer IRS']):
            if pdata['agn']['residuals'][key]['resid'][idx] is not None:
                specx[i] += pdata['agn']['residuals'][key]['lam'][idx].tolist()
                specy_noagn[i] += pdata['no_agn']['residuals'][key]['resid'][idx].tolist()
                specy_agn[i] += pdata['agn']['residuals'][key]['resid'][idx].tolist()

    #### plot medians
    x, y_noagn,y_noagn_up,y_noagn_down, y_noagn_upup, y_noagn_downdown = median_by_band(np.array(photx),np.array(photy_noagn),avg=avg)
    x, y_agn,y_agn_up,y_agn_down, y_agn_upup, y_agn_downdown = median_by_band(np.array(photx),np.array(photy_agn),avg=avg)
    ax[0].plot(x, y_agn,'o', ms=10, color=popts['agn_color'],linestyle='-' , lw=2,alpha=0.8)
    ax[0].plot(x, y_noagn,'o', ms=10, color=popts['noagn_color'],linestyle='-' , lw=2, alpha=0.8)
    #ax[0].plot(np.array(photx),np.array(photy_noagn), 'o', color=popts['noagn_color'], alpha=0.1, ms=3.5, mew=0.0,zorder=-1)
    #ax[0].plot(np.array(photx),np.array(photy_agn), 'o', color=popts['agn_color'], alpha=0.1, ms=3.5, mew=0.0,zorder=-1)
    ax[0].fill_between(x, y_agn_down, y_agn_up, color=popts['agn_color'], alpha=0.2)
    ax[0].fill_between(x, y_noagn_down, y_noagn_up, color=popts['noagn_color'], alpha=0.2)
    #ax[0].fill_between(x, y_agn_downdown, y_agn_upup, color=popts['agn_color'], alpha=0.2)
    #ax[0].fill_between(x, y_noagn_downdown, y_noagn_upup, color=popts['noagn_color'], alpha=0.2)

    for i, key in enumerate(['Akari','Spitzer IRS']):
        # horrible hack to pick out wavelengths. this preserves ~native spacing, downsampled by 3
        bins = pdata['agn']['residuals'][key]['lam'][idx][::4]
        bins[0] *= 0.96
        bins[-1] *= 1.04

        x, y_agn = running_median(np.array(specx[i]),np.array(specy_agn[i]),bins=bins,avg=avg)
        x, y_noagn = running_median(np.array(specx[i]),np.array(specy_noagn[i]),bins=bins,avg=avg)

        nbins = len(bins)-1
        yagn_up, yagn_down, ynoagn_up, ynoagn_down = [np.empty(nbins) for k in xrange(4)]
        for k in xrange(nbins):
            bidx = (np.array(specx[i]) > bins[k]) & (np.array(specx[i]) <= bins[k+1])
            yagn_up[k], yagn_down[k] = np.nanpercentile(np.array(specy_agn[i])[bidx],[84,16])
            ynoagn_up[k], ynoagn_down[k] = np.nanpercentile(np.array(specy_noagn[i])[bidx],[84,16])

        if key == 'Optical':
            sigsmooth = 800
            y_agn = smooth_spectrum(x*1e4, y_agn, sigsmooth)
            yagn_up = smooth_spectrum(x*1e4, yagn_up, sigsmooth)
            yagn_down = smooth_spectrum(x*1e4, yagn_down, sigsmooth)
            y_noagn = smooth_spectrum(x*1e4, y_noagn, sigsmooth)
            ynoagn_up = smooth_spectrum(x*1e4, ynoagn_up, sigsmooth)
            ynoagn_down = smooth_spectrum(x*1e4, ynoagn_down, sigsmooth)

        if key == 'Spitzer IRS':
            mask = (x < mask_si_abs[0])

            ax[i+1].plot(x[mask], y_agn[mask], color=popts['agn_color'],**median_opts)
            ax[i+1].fill_between(x[mask], yagn_down[mask], yagn_up[mask], color=popts['agn_color'], alpha=0.2)
            ax[i+1].plot(x[mask], y_noagn[mask], color=popts['noagn_color'],**median_opts)
            ax[i+1].fill_between(x[mask], ynoagn_down[mask], ynoagn_up[mask], color=popts['noagn_color'], alpha=0.2)

            mask = (x > mask_si_abs[1])

            ax[i+1].plot(x[mask], y_agn[mask], color=popts['agn_color'],**median_opts)
            ax[i+1].fill_between(x[mask], yagn_down[mask], yagn_up[mask], color=popts['agn_color'], alpha=0.2)
            ax[i+1].plot(x[mask], y_noagn[mask], color=popts['noagn_color'],**median_opts)
            ax[i+1].fill_between(x[mask], ynoagn_down[mask], ynoagn_up[mask], color=popts['noagn_color'], alpha=0.2)

        else:
            ax[i+1].plot(x, y_agn, color=popts['agn_color'],**median_opts)
            ax[i+1].fill_between(x, yagn_down, yagn_up, color=popts['agn_color'], alpha=0.2)
            ax[i+1].plot(x, y_noagn, color=popts['noagn_color'],**median_opts)
            ax[i+1].fill_between(x, ynoagn_down, ynoagn_up, color=popts['noagn_color'], alpha=0.2)

    #### labels and scales
    ymin, ymax = np.repeat(-0.3,4), np.repeat(0.3,4)
    labelpad=-1.5
    ax[0].set_xlabel(r'rest-frame wavelength [$\mu$m]',labelpad=labelpad)
    #ax[0].set_ylabel(r'(f$_{\mathrm{obs}}$-f$_{\mathrm{model}}$)/f$_{\mathrm{obs}}$')
    #ax[0].set_ylabel(r'(f$_{\mathrm{obs}}$-f$_{\mathrm{model}}$)/$\sigma$$_{\mathrm{obs}}$')
    ax[0].set_ylabel(r'[(f$_{\mathrm{obs}}$-f$_{\mathrm{model}}$)/$\sigma$$_{\mathrm{obs}}]^2$')

    if log:
        ax[0].set_ylabel(r'log(f$_{\mathrm{obs}}$/f$_{\mathrm{model}}$)')

    ax[0].text(0.05,0.86,'photometry',fontsize=fs,transform=ax[0].transAxes)
    ax[0].text(0.05,0.92,instring,fontsize=fs,transform=ax[0].transAxes,weight='semibold')
    ax[0].text(0.97,0.92,'Model without AGN',transform=ax[0].transAxes,color=popts['noagn_color'],fontsize=fs, ha='right')
    ax[0].text(0.97,0.86,'Model with AGN',transform=ax[0].transAxes,color=popts['agn_color'],fontsize=fs, ha='right')

    ax[0].set_xscale('log',nonposx='clip',subsx=(1,2,5))
    ax[0].xaxis.set_minor_formatter(minorFormatter)
    ax[0].xaxis.set_major_formatter(majorFormatter)
    for tl in ax[0].get_xticklabels():tl.set_visible(False)
    ax[0].set_xlim(xlim_phot)
    ax[0].set_ylim(ylim_phot)
    #ax[0].axhline(0, linestyle='--', color='k',lw=2,zorder=-1)
    #ax[0].yaxis.set_major_locator(MaxNLocator(5))

    for i, key in enumerate(['Akari','Spitzer IRS']):
        sub = (1,2,3,4,5)
        if i+1 == 1:
            sub = (1,2,3,4,5,6)
        ax[i+1].set_ylabel(r'log(f$_{\mathrm{obs}}$/f$_{\mathrm{model}}$)')
        ax[i+1].set_xlabel(r'rest-frame wavelength [$\mu$m]',labelpad=labelpad)
        ax[i+1].set_ylim(ylim_spec[i])
        ax[i+1].set_xlim(xlim_spec[i])
        ax[i+1].text(0.05,0.86,key,fontsize=fs,transform=ax[i+1].transAxes)

        ax[i+1].set_xscale('log',nonposx='clip',subsx=sub)
        ax[i+1].xaxis.set_minor_formatter(minorFormatter)
        ax[i+1].xaxis.set_major_formatter(majorFormatter)
        for tl in ax[i+1].get_xticklabels():tl.set_visible(False)
        ax[i+1].axhline(0, linestyle='--', color='k',lw=2,zorder=-1)
        ax[i+1].text(0.05,0.92,instring,fontsize=fs,transform=ax[i+1].transAxes,weight='semibold')

    return fig

def sedfig(pdata,**popts):

    fig = plt.figure(figsize=(15, 10))

    left, top = 0.1,0.97 # margins

    sedax, resax = [], []
    axwid, axheight, delx, dely = 0.251, 0.42, 0.067, 0.06

    left = [left, left+axwid+delx, left+2*(axwid+delx), left, left+axwid+delx, left+2*(axwid+delx)]
    right = [left[0]+axwid, left[1]+axwid, left[2]+axwid, left[0]+axwid, left[1]+axwid, left[2]+axwid]
    top = [top, top,top, top-axheight-dely, top-axheight-dely, top-axheight-dely]
    bot = [top[0]-axheight, top[1]-axheight, top[2]-axheight, top[3]-axheight, top[4]-axheight, top[5]-axheight]
    for i in xrange(6):
        gs = gridspec.GridSpec(2,1, height_ratios=[2,1])
        gs.update(left=left[i],right=right[i],bottom=bot[i],top=top[i],hspace=0)
        ax_sed, ax_res = plt.Subplot(fig, gs[0]), plt.Subplot(fig, gs[1])
        fig.add_subplot(ax_sed)
        fig.add_subplot(ax_res)

        sedax.append(ax_sed)
        resax.append(ax_res)

    sedax, resax = np.array(sedax), np.array(resax)

    wavlims = (1,30)
    alpha = 0.75
    lw = 1.5
    lw_obs = 3
    factor = 3e14*3631*1e-23 # conversion factor, divide by wave in microns

    for ii in range(len(sedax)):

        idx = pdata['observables']['objname'].index(to_plot[ii])

        #### now plot the SED
        mod_wave = pdata['observables']['wave'][idx]/1e4
        wav_idx = (mod_wave > wavlims[0]) & \
                  (mod_wave < wavlims[1]) & \
                  (mod_wave < mask_si_abs[0])
        yon, yoff = pdata['observables']['agn_on_spec'][idx][wav_idx]*factor/mod_wave[wav_idx], pdata['observables']['agn_off_spec'][idx][wav_idx]*factor/mod_wave[wav_idx]
        sedax[ii].plot(mod_wave[wav_idx],yon, lw=lw, alpha=alpha, color=popts['agn_color'],zorder=1)
        sedax[ii].plot(mod_wave[wav_idx],yoff, lw=lw, alpha=alpha, color=popts['noagn_color'])
        min, max = np.min([yon.min(),yoff.min()]), np.max([yoff.max(),yon.max()])

        wav_idx = (mod_wave > wavlims[0]) & \
                  (mod_wave < wavlims[1]) & \
                  (mod_wave > mask_si_abs[1])
        yon, yoff = pdata['observables']['agn_on_spec'][idx][wav_idx]*factor/mod_wave[wav_idx], pdata['observables']['agn_off_spec'][idx][wav_idx]*factor/mod_wave[wav_idx]
        sedax[ii].plot(mod_wave[wav_idx],yon, lw=lw, alpha=alpha, color=popts['agn_color'],zorder=1, label='Model with AGN')
        sedax[ii].plot(mod_wave[wav_idx],yoff, lw=lw, alpha=alpha, color=popts['noagn_color'], label='Model without AGN')
        min, max = np.min([min,yon.min(),yoff.min()]), np.max([max,yon.max(),yoff.max()])

        on_interp = interp1d(mod_wave, pdata['observables']['agn_on_spec'][idx]*factor/mod_wave, bounds_error = False, fill_value = 0)
        off_interp = interp1d(mod_wave, pdata['observables']['agn_off_spec'][idx]*factor/mod_wave, bounds_error = False, fill_value = 0)

        if type(pdata['observables']['spit_lam'][idx]) is np.ndarray:
            wave = pdata['observables']['spit_lam'][idx]/1e4
            flux = smooth_spectrum(wave*1e4, pdata['observables']['spit_flux'][idx]*factor/wave, 2000)

            wav_idx = (wave > wavlims[0]) & \
                      (wave < wavlims[1]) & \
                      (wave < mask_si_abs[0])
            sedax[ii].plot(wave[wav_idx],flux[wav_idx],
                           lw=lw_obs, color='black',zorder=-1)
            resax[ii].plot(wave[wav_idx], np.log10(flux[wav_idx]/on_interp(wave[wav_idx])),
                           lw=lw, alpha=alpha, color=popts['agn_color'])
            resax[ii].plot(wave[wav_idx], np.log10(flux[wav_idx]/off_interp(wave[wav_idx])),
                           lw=lw, alpha=alpha, color=popts['noagn_color'])

            wav_idx = (wave > wavlims[0]) & \
                      (wave < wavlims[1]) & \
                      (wave > mask_si_abs[1])
            sedax[ii].plot(wave[wav_idx],flux[wav_idx],
                           lw=lw_obs, color='black',zorder=-1, label='observed')
            resax[ii].plot(wave[wav_idx], np.log10(flux[wav_idx]/on_interp(wave[wav_idx])),
                           lw=lw, alpha=alpha, color=popts['agn_color'],zorder=1)
            resax[ii].plot(wave[wav_idx], np.log10(flux[wav_idx]/off_interp(wave[wav_idx])),
                           lw=lw, alpha=alpha, color=popts['noagn_color'])

            min, max = np.min([min,(pdata['observables']['spit_flux'][idx][wav_idx]*factor/wave[wav_idx]).min()]), np.max([max,(pdata['observables']['spit_flux'][idx][wav_idx]*factor/wave[wav_idx]).max()])
        if type(pdata['observables']['ak_lam'][idx]) is np.ndarray:
            wave = pdata['observables']['ak_lam'][idx]/1e4
            flux = smooth_spectrum(wave*1e4, pdata['observables']['ak_flux'][idx]*factor/wave, 1000.)

            wav_idx = (wave > wavlims[0]) & (wave < wavlims[1])
            sedax[ii].plot(wave[wav_idx],flux[wav_idx], 
                          lw=lw_obs, color='black',zorder=-1)

            resax[ii].plot(wave[wav_idx], np.log10(flux[wav_idx]/on_interp(wave[wav_idx])),
                           lw=lw, alpha=alpha, color=popts['agn_color'],zorder=1)
            resax[ii].plot(wave[wav_idx], np.log10(flux[wav_idx]/off_interp(wave[wav_idx])),
                           lw=lw, alpha=alpha, color=popts['noagn_color'])

            min, max = np.min([min,(pdata['observables']['ak_flux'][idx][wav_idx]*factor/wave[wav_idx]).min()]), np.max([max,(pdata['observables']['ak_flux'][idx][wav_idx]*factor/wave[wav_idx]).max()])

        ### labels and name
        sedax[ii].set_xlim(wavlims)
        resax[ii].set_xlim(wavlims)
        sedax[ii].set_ylim(min,max*2.5)
        if ii == 2:
            sedax[ii].set_ylim(min,1e-9)
        fs = 14
        xs, ys, dely, ha = 0.025, 0.89, 0.095, 'left'
        if ii == 5:
            xs, ha = 0.97, 'right'
        sedax[ii].text(xs,ys,pdata['observables']['objname'][idx],transform=sedax[ii].transAxes,color='black',ha=ha,fontsize=fs+2)
        
        fmt = "{{0:{0}}}".format(".2f").format
        text = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
        text = text.format(fmt(pdata['fmir'][ii]), fmt(pdata['fmir_do'][ii]), fmt(pdata['fmir_up'][ii]))
        text = "{0} = {1}".format(r'f$_{\mathrm{AGN,MIR}}$', text)
        sedax[ii].text(xs,ys-dely,text,transform=sedax[ii].transAxes,color='black',ha=ha,fontsize=fs+2)

        if ii == 0:
            sedax[ii].legend(loc=4, prop={'size':9.5},
                             scatterpoints=1,fancybox=True)

        ### axis scaling
        sedax[ii].set_yscale('log',nonposx='clip',subsx=(1,2,4))
        sedax[ii].set_xscale('log',nonposx='clip',subsx=(1,2,4))
        sedax[ii].xaxis.set_minor_formatter(minorFormatter)
        sedax[ii].xaxis.set_major_formatter(majorFormatter)
        for tl in sedax[ii].get_xticklabels():tl.set_visible(False)

        resax[ii].set_xscale('log',nonposx='clip',subsx=(1,2,4))
        resax[ii].xaxis.set_minor_formatter(minorFormatter)
        resax[ii].xaxis.set_major_formatter(majorFormatter)
        resax[ii].axhline(0, linestyle='--', color='0.2',lw=1.5,zorder=-1)
        for tl in resax[ii].get_xticklabels():tl.set_visible(False)

        ### axis labels
        if ii >= 3:
            resax[ii].set_xlabel(r'wavelength $\mu$m')
        if (ii == 0) or (ii == 3):
            sedax[ii].set_ylabel(r'$\nu$f$_{\nu}$ [ergs/s/cm$^{-2}$]')
            resax[ii].set_ylabel(r'log(f$_{\mathrm{obs}}$/f$_{\mathrm{mod}}$)')
        reslim = np.max(resax[ii].get_ylim())
        resax[ii].set_ylim(-reslim,reslim)
        resax[ii].set_ylim(-0.6,0.6)
        if ii == 0:
            resax[ii].set_ylim(-1,1)

        resax[ii].yaxis.set_major_locator(MaxNLocator(5))

        ### remove x-tick labels
        labels = [item.get_text() for item in sedax[ii].get_xticklabels()]
        empty_string_labels = ['']*len(labels)
        sedax[ii].set_xticklabels(empty_string_labels)

    return fig






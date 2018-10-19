import numpy as np
import matplotlib.pyplot as plt
import os, hickle, pickle, prosp_dutils
from prospector_io import load_prospector_data, find_all_prospector_results
from astropy.cosmology import WMAP9
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from dynesty.plotting import _quantile as weighted_quantile
from collections import OrderedDict
from scipy.interpolate import interp2d
import td_delta_params as pfile
from plot_sample_selection import mass_completeness
from integrate_sfrd import mf_phi, mf_parameters, load_zfourge_mf
from td_io import load_fast
from scipy.ndimage import gaussian_filter as norm_kde
from astropy.convolution import convolve

plt.ioff()

popts = {'fmt':'o', 'capthick':1.5,'elinewidth':1.5,'ms':9,'alpha':0.8,'color':'0.3'}
red = '#FF3D0D'
dpi = 160
cmap = 'cool'
minsfr = 0.01
colors = ['#66c2a5','#fc8d62','#abc5fc','#e78ac3','#a6d854','k']

def do_all_mass(runname='td_delta', outfolder=None, regenerate=False, regenerate_stack=False, **opts):

    if outfolder is None:
        outfolder = os.getenv('APPS') + '/prospector_alpha/plots/'+runname+'/fast_plots/'
        if not os.path.isdir(outfolder):
            os.makedirs(outfolder)
            os.makedirs(outfolder+'data/')

    stack_opts = {
              'zstart': [0.6, 1.1, 1.6],
              'dz': 0.1,
              'dm': 0.2, 
              'colors': ['#45ADA8','#FC913A','#FF4E50'],
              'high_mass_cutoff': 11.6
             }
    stack_opts.update(opts)

    filename = outfolder+'data/dm_stack.pickle'
    filename_fast = outfolder+'data/dm_stack_fast.pickle'
    if os.path.isfile(filename) and regenerate_stack == False:
        with open(filename, "r") as f:
            stack = pickle.load(f)
        with open(filename_fast, "r") as f:
            faststack = pickle.load(f)
    else:
        data = collate_data(runname,filename=outfolder+'data/stacksfh.h5',regenerate=regenerate,**opts)
        faststack, fdict = stack_sfh_fast(data,regenerate_stack=regenerate_stack,**stack_opts)
        pickle.dump(faststack,open(filename_fast, "w"))
        stack = stack_sfh_dm(data,fdict,regenerate_stack=regenerate_stack, **stack_opts)
        pickle.dump(stack,open(filename, "w"))

    #plot_avg_sfh(stack, faststack, outfolder, **stack_opts)
    evolve_mf_backwards(stack, faststack, outfolder, **stack_opts)

def stack_sfh_fast(dat,**opts):
    """ do the same for FAST
    concerns: we want to use Prospector mass to make sure we're stacking evenly?
    or should we switch the other to FAST mass to make sure we're consistent?  <-----
        inside loop: must generate SFR(t)

    make sure to output FAST mass for each galaxy, so we can use for Prospector stacking
    """

    # load FAST data
    allfields, fastlist = ['AEGIS','COSMOS','GOODSN','GOODSS','UDS'], []
    for f in allfields: fastlist.append(load_fast('td_new',f))

    # generate output container
    stack = {str(z):{} for z in opts['zstart']}
    ssfrmin, ssfrmax = -14, -7.5 # intermediate interpolation onto regular sSFR grid
    nt = 201
    ssfr_arr = 10**np.linspace(ssfrmin,ssfrmax,nt)
    stack['ssfr_arr'] = ssfr_arr

    fdict_full = {}
    for zfloat in opts['zstart']:

        # generate mass vector
        # calculate mass completeness; perform at highest redshift in stack to be conservative
        # note this is in FAST MASS, so it's systematically offset by ~0.2 dex...
        z = str(zfloat)
        mcomplete = mass_completeness(zfloat+opts['dz'])
        stack[z]['mvec'] = np.arange(mcomplete,opts['high_mass_cutoff'],opts['dm'])
        tbins = np.linspace(0,WMAP9.age(zfloat).value*1e9, nt)
        stack[z]['tbins'] = tbins
        for name in ['sfr_med', 'sfr_eup', 'sfr_edo']: stack[z][name] = np.zeros(shape=(stack[z]['mvec'].shape[0],nt))
        stack[z]['full_pdf'] = []

        # for each main sequence bin, in mass, stack SFH
        # save FAST masses so they can be used for Prospector stacking
        fdict_full[z] = {f:[] for f in ['mass','age','tau','zred','id']}
        for i, mass in enumerate(stack[z]['mvec']):

            # bookkeeping
            print 'z={0}+/-{1}, m={2}-{3}'.format(z,opts['dz'],mass,mass+opts['dm'])

            # for EACH FIELD, extract every galaxy meeting mass + redshift requirements
            # must pull out tau, tage, mass
            fdict = {f:[] for f in ['mass','age','tau','zred','id']}
            for j, fast in enumerate(fastlist):
                indexes = np.where((np.abs(fast['z']-zfloat) < opts['dz']) & \
                                   (np.abs(fast['lmass']-mass+opts['dm']/2.) < opts['dm']/2.))[0]
                fdict['mass'] += fast['lmass'][indexes].tolist()
                fdict['age'] += (10**fast['lage'][indexes]).tolist()
                fdict['tau'] += (10**fast['ltau'][indexes]).tolist()
                fdict['zred'] += fast['z'][indexes].tolist()
                fdict['id'] += [allfields[j]+'_'+str(f) for f in fast['id'][indexes]]
            for key in fdict.keys(): fdict[key] = np.array(fdict[key])

            # calculate (SFR / M) on regular time grid
            ngal = fdict['mass'].shape[0]
            print '\t{0} galaxies in bin'.format(ngal)
            ssfr = np.zeros(shape=(nt,ngal))
            for j in range(ngal):
                tidx = tbins < fdict['age'][j]
                ssfr[tidx,j] = (np.exp(-tbins[tidx]/fdict['tau'][j]) * (fdict['tau'][j]*(1-np.exp(-fdict['age'][j]/fdict['tau'][j])))**-1)[::-1]

            # weight by (t_univ(z=zgal) / t_univ(z=z_min)) to account for variation in t_univ
            # i.e. all galaxies in a given stack should have the same average sSFR
            ssfr *= WMAP9.age(fdict['zred']).value*1e9/tbins.max()

            # now create stacked sSFR
            # TO IMPROVE: draw individual SFHs...
            # calculate weighted quantiles from sSFR PDF at fixed T
            median, errup, errdown = np.percentile(ssfr, [50,84,16],axis=1)
            smooth_median = norm_kde(median,3)
            norm = np.trapz(smooth_median,x=tbins)
            stack[z]['sfr_med'][i,:] = smooth_median/norm
            stack[z]['sfr_eup'][i,:] = errup/norm
            stack[z]['sfr_edo'][i,:] = errdown/norm
            stack[z]['full_pdf'] += [ssfr]

            # transfer FAST properties to use in Prospector stacking
            for key in fdict.keys(): fdict_full[z][key] += fdict[key].tolist()

    return stack,fdict_full

def stack_sfh_dm(data,fdict,nt=None, **opts):
    """stack in dM in narrow redshift bins
    """

    # generate output container
    stack = {str(z):{} for z in opts['zstart']}
    data['zred'] = np.array(data['zred'])
    nsamps = 3000 # number of samples in posterior files

    ssfrmin, ssfrmax, nssfr = -14, -7.5, 201 # intermediate interpolation onto regular sSFR grid
    ssfr_arr = 10**np.linspace(ssfrmin,ssfrmax,nssfr)
    stack['ssfr_arr'] = ssfr_arr

    for zfloat in opts['zstart']:

        # load time bins at central redshift
        z = str(zfloat)
        pfile.run_params['zred'] = zfloat
        mod = pfile.load_model(**pfile.run_params)
        stack[z]['agebins'] = mod.params['agebins']
        nt = stack[z]['agebins'].shape[0]
        tbins = np.mean(10**stack[z]['agebins'], axis=1)/1e9
        agediff = np.diff(10**stack[z]['agebins'],axis=1).flatten()

        # generate mass vector
        # calculate mass completeness; perform at highest redshift in stack to be conservative
        mcomplete = mass_completeness(zfloat+opts['dz'])
        stack[z]['mvec'] = np.arange(mcomplete,opts['high_mass_cutoff'],opts['dm'])
        nmass = stack[z]['mvec'].shape[0]
        for name in ['sfr_med', 'sfr_eup', 'sfr_edo']: stack[z][name] = np.zeros(shape=(nmass,nt))
        stack[z]['full_pdf'] = np.zeros(shape=(nmass,nt,nssfr-1))

        # for each main sequence bin, in mass, stack SFH
        for i, mass in enumerate(stack[z]['mvec']):

            # bookkeeping
            print 'z={0}+/-{1}, m={2}-{3}'.format(z,opts['dz'],mass,mass+opts['dm'])

            # define galaxies in redshift/mass bin
            # use FAST mass for this selection
            fmatches = np.where((np.abs(np.array(fdict[z]['zred'])-zfloat) < opts['dz']) & (np.abs(fdict[z]['mass']-mass+opts['dm']/2.) < opts['dm']/2.))[0]
            ids = np.array(fdict[z]['id'])[fmatches]
            indexes = np.where(np.in1d(data['objname'],ids))[0]

            # calculate (SFR / M) chains on regular time grid
            # each draw has its own (SFR,t) vector with an associated weight
            # here we transform into (SFR/M)(t) and interpolate onto regular time grid
            # we do this with nearest-neighbor interpolation which is precisely correct for step-function SFH
            ngal = indexes.shape[0]
            print '\t{0} galaxies in bin'.format(ngal)
            ssfr,weights = np.empty(shape=(nsamps,nt,ngal)), np.empty(shape=(nsamps,ngal))

            for m, idx in enumerate(indexes):
                # weight by (t_univ(z=zgal) / t_univ(z=z_min)) to account for variation in t_univ
                # i.e. all galaxies in a given stack should have the same average sSFR
                weight = data['weights'][idx]
                sfh = data['ssfh'][idx]
                t_sfh = data['sfh_t'][idx]
                ssfh = sfh * t_sfh.max()/tbins.max()
                for j in range(nt):
                    if tbins[j] > t_sfh.max():
                        ssfr[:,j,m] = np.nan
                    else:
                        ssfr[:,j,m] = ssfh[:,np.abs(t_sfh - tbins[j]).argmin()]
                weights[:,m] = weight

            # now create stacked sSFR
            # this returns weighted sSFR median + quantiles
            # TO IMPROVE: draw individual SFHs...
            for j in range(nt):
                # construct empty sSFR PDF and determine which galaxies contribute
                in_samp = np.where(np.isfinite(ssfr[0,j,:]))[0]
                hist = np.zeros(shape=ssfr_arr.shape[0]-1)

                # for each object in sample, add to sSFR PDF
                for idx in in_samp:
                    ssfr_in_hist = np.clip(ssfr[:,j,idx],10**ssfrmin,10**ssfrmax)
                    g1,_ = np.histogram(ssfr_in_hist, density=False,weights=weights[:,idx], bins=ssfr_arr)
                    hist += g1

                # calculate weighted quantiles from sSFR PDF at fixed time
                median, errup, errdown = weighted_quantile((ssfr_arr[1:]+ssfr_arr[:-1])/2., [0.5,.84,.16],weights=hist)

                stack[z]['sfr_med'][i,j] = median
                stack[z]['sfr_eup'][i,j] = errup
                stack[z]['sfr_edo'][i,j] = errdown
                stack[z]['full_pdf'][i,j,:] = hist

            # normalize that shit
            norm = (stack[z]['sfr_med'][i,:]*agediff).sum()
            stack[z]['sfr_med'][i,:] /= norm
            stack[z]['sfr_eup'][i,:] /= norm
            stack[z]['sfr_edo'][i,:] /= norm

    return stack

def plot_avg_sfh(dat,fdat,outfolder,**opts):

    fig, ax = plt.subplots(1, 3, figsize = (12,4))
    ylim = (-13,-8)
    fs = 10
    for i, zfloat in enumerate(opts['zstart']):
        z = str(zfloat)

        central_mass = dat[z]['mvec'] + opts['dm']/2.
        nmass = central_mass.shape[0]
        cmap = prosp_dutils.get_cmap(nmass,cmap='nipy_spectral')
        median_ssfr_fast = fdat[z]['sfr_med']
        tfast = fdat[z]['tbins']

        for j in range(nmass): ax[i].plot(np.log10(tfast),np.log10(median_ssfr_fast[j,:]),
                                          color=cmap(j),label="{0:.1f}".format(central_mass[j]))
        ax[i].legend(ncol=3,prop={'size':fs*0.7})

        ax[i].set_ylim(ylim)

    plt.tight_layout()
    plt.show()
    print 1/0

def evolve_mf_backwards(dat,fdat,outfolder,use_median=False,plot_kernels=False,**opts):
    """for now, we assume each element dm has the same SFH
    more complex assumptions wait for hierarchical bayes...

    For each redshift bin:
        (1) Generate phi(M) at this redshift
        (2) propagate backwards into N predictions (how to decide? by availability of data?)
            -- subtract SFR * MLOSS
        (3) Plot predictions versus truth
    """

    # load data
    mf, mf_z = load_zfourge_mf()
    for key in mf_z.keys(): mf_z[key] = np.array(mf_z[key])
    mf_z_mid = (np.array(mf_z['z_low']) + np.array(mf_z['z_up']))/2.

    # plot options
    nz_start = len(opts['zstart'])
    zlabel_idx = mf_z_mid > opts['zstart'][0]
    nz_window = zlabel_idx.sum()
    xlim = (8,11.9)
    ylim = (-6,-1.1)

    fig, axes = plt.subplots(nz_start, nz_window, figsize = (12,6))
    fs, lw = 10, 1.5
    fcolor = 'crimson'
    pcolor = 'blue'

    for j,zfloat in enumerate(opts['zstart']):

        # pull out mass, redshift
        z = str(zfloat)
        central_mass = dat[z]['mvec']+opts['dm']/2.
        nmass = len(central_mass)

        # create time vectors for FAST and Prospector
        agebins = 10**dat[z]['agebins']
        agediff = np.diff(agebins,axis=1).flatten()
        tfast = fdat[z]['tbins']

        # generate starting number densities / masses
        # units of SFR / Mpc^-3 / dex
        ndens_start = np.log10(mf_phi(mf_parameters(zfloat),central_mass))

        # extract mean and PDF for sSFR
        median_ssfr = dat[z]['sfr_med']
        ssfr_pdf = dat[z]['full_pdf']
        median_ssfr_fast = fdat[z]['sfr_med']
        ssfr_pdf_fast = fdat[z]['full_pdf']

        # we predict mass function for z > zobs
        # loop over each prediction window to predict mass function evolution
        predict_idx = np.where(mf_z_mid > zfloat)[0]
        for i, idx in enumerate(predict_idx):

            # determine plotting axis
            ax = axes[j,i+(nz_window-predict_idx.shape[0])]

            # calculate and display delta(t)
            dt = (WMAP9.age(zfloat).value - WMAP9.age(mf_z_mid[idx]).value)*1e9
            if np.log10(dt) > 9:
                dt_display = "{0:.1f}".format(dt/1e9)+' Gyr'
            else:
                round_to_nearest = 50
                dt_display = str(np.round(dt/1e6/round_to_nearest).astype(int)*round_to_nearest)+' Myr'
            ax.text(0.96,0.9,r'$\Delta t$='+dt_display,fontsize=fs,transform=ax.transAxes,ha='right')

            # calculate MF predictions for prosopector
            idx_sfr = (agebins < dt)
            full_idx, partial_idx = idx_sfr.sum(axis=1) == 2, idx_sfr.sum(axis=1) == 1
            partial_tvec = dt - agebins[partial_idx,0]
            if use_median: # the easy way; we shift the mass bins by average mass growth
                mdiff = ((agediff[full_idx]*median_ssfr[:,full_idx]).sum(axis=1) + (partial_tvec[0]*median_ssfr[:,partial_idx].flatten()))
                mplot = np.log10(10**central_mass*(1-mdiff))
                ndens_plot = ndens_start
            else: 
                """
                 the hard way. we convolve the MF with the growth kernels.
                 FOR EACH LOOKBACK BIN
                   -- create mdiff kernel
                   -- convolve starting mass function with kernel(M)
                   -- pass convolved mass function to next loop
                """

                # for each SFH age bin between zstart and ztarget
                nbins = (agebins[:,1] < dt).sum()
                ndens = 10**ndens_start.copy()
                for m in range(nbins):

                    # transform x-axis of SFR kernel into mass by multiplying
                    # if it's not the last time bin in the series, we use full bin width
                    if (m < nbins-1):
                        dt_bin = agediff[m]
                    else:
                        dt_bin = partial_tvec
                    mdiff_grid = (dat['ssfr_arr'][1:]+dat['ssfr_arr'][:-1])/2.*dt

                    # create ndens output vector and convolve each dM with kernel
                    if (plot_kernels) & (m != 0):
                        fakefig, fakeax = plt.subplots(1,1)
                        cmap = prosp_dutils.get_cmap(18,cmap='nipy_spectral')
                    ndens_out = np.zeros_like(ndens)
                    for k in range(nmass): 

                        # generate input number density for convolution
                        ndens_convolve = np.zeros(nmass)
                        ndens_convolve[k] = ndens[k]

                        # construct low-res mass kernel by integrating over high-res mass kernel for accuracy
                        # (interpolation would be a bad approximation...)
                        nkernel, bins_below_minmass = 31, 8
                        kernel = np.zeros(nkernel)
                        mdiff_kernel_highres = dat[z]['full_pdf'][k,m,:]
                        for l in range(k+1+bins_below_minmass): kernel[nkernel/2-l] = prosp_dutils.integral_average(-mdiff_grid[::-1],mdiff_kernel_highres[::-1],opts['dm']*(-l-0.5),opts['dm']*(0.5-l))*opts['dm']
                        if (plot_kernels) & (m != 0):
                            fakeax.plot(np.linspace(-3,3,num=31),kernel/kernel.sum(),color=cmap(k),label='{:.2f}'.format(central_mass[k]),lw=2,alpha=0.6)
                        ndens_out += convolve(ndens_convolve,kernel/kernel.sum(),normalize_kernel=True)
                    
                    if (plot_kernels) & (m != 0):
                        fakeax.legend()
                        plt.show()
                        print 1/0
                    # propagate output to next input
                    ndens = ndens_out
                
                # set plotting quantities
                ndens_plot = np.log10(ndens)
                mplot = central_mass

            # calculate predictions for FAST
            if True:
                mdiff_fast = norm_kde([prosp_dutils.integral_average(tfast,median_ssfr_fast[k,:],0,dt)*dt for k in range(nmass)],2).flatten()
                mplot_fast = np.log10(10**central_mass*(1-mdiff_fast))
                ndens_plot_fast = ndens_start
            else:
                pass

            # plot predictions for FAST, Prospector
            pline = ax.plot(mplot,ndens_plot, color=pcolor, linestyle='--',label='Prospector',lw=lw*1.5)
            fline = ax.plot(mplot_fast,ndens_plot_fast, color=fcolor, linestyle=':',label='FAST',lw=lw*1.5)

            # if we have no FAST data in the plot range,
            # draw an arrow letting the viewer know what happened
            if (np.nanmax(mdiff_fast) < xlim[0]) | (np.isfinite(mdiff_fast).sum() == 0):
                ax.arrow(0.19,0.06,-0.16,0.0,color=fcolor,transform=ax.transAxes,head_width=0.055,width=0.01,head_length=0.024,length_includes_head=True)
                ax.text(0.02,0.1,'FAST',color=fcolor,transform=ax.transAxes,fontsize=fs*0.8,weight='roman')

            # plot data
            idx_use = mf['logphi'+str(idx)] > -10
            mf_med, mf_errup, mf_errdown = mf['logphi'+str(idx)][idx_use], mf['eup'+str(idx)][idx_use], mf['elo'+str(idx)][idx_use]
            err = prosp_dutils.asym_errors(mf_med,mf_med+mf_errup,mf_med-mf_errdown)
            oline = ax.errorbar(mf['logm'][idx_use],mf_med,yerr=err, color='k', linestyle='-',lw=lw*0.8,fmt='o',ms=2.5,elinewidth=lw*0.8)

    # legend
    legend = fig.legend((oline, pline[0], fline[0]), ('Observed', 'Prospector SFHs', 'FAST SFHs'), loc=(0.07,0.1),
                         fancybox=True,scatterpoints=1,prop={'size':fs*1.3},title='Legend')
    legend.set_title('Legend',prop={'size':fs*1.8,'weight':'roman'})

    # labels, axes, and all the pretty things
    xsuper = [0.02,0.285,0.545] # this must be customized due to ragged left edge of axes
    for j,zfloat in enumerate(opts['zstart']):

        # left super-labels
        fig.text(xsuper[j],0.81-0.29*j,r'z$_{\mathrm{start}}$='+str(zfloat),ha='center',va='center',rotation=90,fontsize=fs*1.5,weight='bold')
        for i in range(nz_window):

            # top super-labels
            if j == (nz_start-1):
                txt = str(mf_z['z_low'][zlabel_idx][i])+' < '+str(mf_z['z_up'][zlabel_idx][i])
                fig.text(0.145+0.13*i,0.97,txt,ha='center',va='center',fontsize=fs,weight='bold')

            # turn axis off if there's no data
            # else set limits
            if np.max(axes[j,i].get_xlim()) == 1:
                axes[j,i].axis('off')
                continue
            else:
                axes[j,i].set_xlim(xlim)
                axes[j,i].set_ylim(ylim)

            # y-edges
            # if it's the first edge or if the previous edge is off, add labels
            if (i == 0) | (axes[j,i-1].axis()[1] == 1.0):
                axes[j,i].set_ylabel(r'log($\phi$/Mpc$^{3}$/dex)',fontsize=fs,labelpad=0.7)
            else:
                for tl in axes[j,i].get_yticklabels():tl.set_visible(False)

            # x-edges
            # if it's the bottom OR IF the axis below this is going to be turned OFF
            if (j == (nz_start-1)) | (axes[np.clip(j+1,0,nz_start-1),i].axis()[1] == 1.0):
                axes[j,i].set_xlabel(r'log(M$_*$/M$_{\odot}$)',fontsize=fs,labelpad=0.7)
            else:
                for tl in axes[j,i].get_xticklabels():tl.set_visible(False)

    plt.tight_layout(w_pad=0.03,h_pad=0.03)
    fig.subplots_adjust(top=0.95,left=0.08,hspace=0.1,wspace=0.1)
    plt.show()
    plt.savefig(outfolder+'evolve_mf_backwards.png',dpi=200)
    print 1/0
    plt.close()

def do_all(runname='td_delta', outfolder=None, regenerate=False, regenerate_stack=False, **opts):

    if outfolder is None:
        outfolder = os.getenv('APPS') + '/prospector_alpha/plots/'+runname+'/fast_plots/'
        if not os.path.isdir(outfolder):
            os.makedirs(outfolder)
            os.makedirs(outfolder+'data/')

    stack_opts = {
              'sigma_sf':0.3,                  # scatter in the star-forming sequence, in dex
              'nbins_horizontal':3,            # number of bins in horizontal stack
              'nbins_vertical':4,              # number of bins in vertical stack
              'horizontal_bin_colors': ['#45ADA8','#FC913A','#FF4E50'],
              'vertical_bin_colors': ['#ca0020','#f4a582','#008080','#045a8c'],
              'low_mass_cutoff':9.,          # log(M) where we stop stacking and plotting
              'high_mass_cutoff': 11.5,
              'ylim_horizontal_sfr': (-0.8,3),
              'ylim_horizontal_ssfr': (1e-11,1.5e-9),
              'ylim_vertical_sfr': (-2,2.9),
              'ylim_vertical_ssfr': (1e-13,4e-9),
              'xlim_t': (0.05,9),
              'show_disp':[0.16,0.84],         # percentile of population distribution to show on plot
              'adjust_sfr': -0.3,             # adjust whitaker SFRs by how much?
              'zbins': [(0.5,1.),(1.,1.5),(1.5,2.),(2.,2.5)]
             }
    stack_opts['zbin_labels'] = ["{0:.1f}".format(z1)+'<z<'+"{0:.1f}".format(z2) for (z1, z2) in stack_opts['zbins']]

    filename = outfolder+'data/single_sfh_stack.h5'
    if os.path.isfile(filename) and regenerate_stack == False:
        with open(filename, "r") as f:
            stack = hickle.load(f)
    else:
        data = collate_data(runname,filename=outfolder+'data/stacksfh.h5',regenerate=regenerate,**opts)
        stack = stack_sfh(data,regenerate_stack=regenerate_stack, **stack_opts)
        hickle.dump(stack,open(filename, "w"))

    plot_stacked_sfh(stack,outfolder, **stack_opts)


def collate_data(runname, filename=None, regenerate=False, **opts):
    """ pull out all of the necessary information from the individual data files
    this takes awhile, so this data is saved to disk.
    currently this saves point estimates for objname, stellar mass, SFR (30,100), avg_age, zred
    and chains for SFR(t), t, weight
    """

    # if it's already made, load it and give it back
    # else, start with the making!
    if os.path.isfile(filename) and regenerate == False:
        print 'loading all data'
        with open(filename, "r") as f:
            outdict=hickle.load(f)

        return outdict

    # define output containers
    outvar = ['stellar_mass', 'sfr_100','avg_age']
    outdict = {q: {f: [] for f in ['q50','q84','q16']} for q in outvar}
    for f in ['objname','sfh_t', 'weights', 'ssfh', 'zred']: outdict[f] = [] 

    basenames,_,_ = prosp_dutils.generate_basenames(runname)
    for i, name in enumerate(basenames):

        # load output from fit
        try:
            res, _, model, eout = load_prospector_data(name)
        except:
            print name.split('/')[-1]+' failed to load. skipping.'
            continue
        if (res is None) or (eout is None):
            continue

        outdict['objname'] += [name.split('/')[-1]]
        print 'loaded ' + outdict['objname'][-1]

        # agebins (and generate model)
        outdict['sfh_t'] += [eout['sfh']['t'][0]]
        tm_chain = res['chain'][eout['sample_idx'],res['theta_labels'].index('massmet_1')]
        outdict['ssfh'] += [eout['sfh']['sfh']/10**tm_chain[:,None]]
        outdict['weights'] += [eout['weights']]

        # extra variables
        outdict['zred'] += [eout['zred']]
        for v in outvar:
            if v in eout['thetas'].keys():
                for f in ['q50','q84','q16']: outdict[v][f] += [eout['thetas'][v][f]]
            else:
                for f in ['q50','q84','q16']: outdict[v][f] += [eout['extras'][v][f]]

    # dump files and return
    hickle.dump(outdict,open(filename, "w"))
    return outdict

def stack_sfh(data,nt=None, **opts):
    """stack in VERTICAL and HORIZONTAL slices on the main sequence
    we measure the average sSFR for each galaxy
    then report the median of the average for each bin
    the error bars are the 1sigma percentiles of the averages
    """

    # stack horizontal
    # first iterate over redshift
    stack = {'hor':{},'vert': {}}
    data['zred'] = np.array(data['zred'])
    nsamps = 3000 # number of samples in posterior files

    ssfrmin, ssfrmax = -13, -8 # intermediate interpolation onto regular sSFR grid
    ssfr_arr = 10**np.linspace(ssfrmin,ssfrmax,1001)

    for (z1, z2) in opts['zbins']:

        # generate zavg, add containers
        zavg = (z1+z2)/2.
        zstr = "{:.2f}".format(zavg)

        # setup time bins
        # here we fuse the two youngest bins into one for presentation purposes
        mod = pfile.load_model(zred=zavg,**pfile.run_params)
        agebins = mod.params['agebins'][1:,:]
        agebins[0,0] = 7
        #tbins = 10**np.mean(agebins,axis=1)/1e9
        tbins = np.logspace(7.3,agebins.max(),200)/1e9
        nt = len(tbins)
        stack['hor'][zstr], stack['vert'][zstr] = {'t':tbins}, {'t':tbins}

        # define galaxies in redshift bin
        zidx = np.where((data['zred'] > z1) & (data['zred'] <= z2))[0]

        # calculate SFR(MS) for each galaxy
        # perhaps should calculate at z_gal for accuracy?
        stellar_mass = np.log10(data['stellar_mass']['q50'])[zidx]
        logsfr = np.log10(data['sfr_100']['q50'])[zidx]
        logsfr_ms = sfr_ms(np.full(stellar_mass.shape[0],zavg),stellar_mass,**opts)
        on_ms = np.where((stellar_mass > opts['low_mass_cutoff']) & \
                         (stellar_mass < opts['high_mass_cutoff']) & \
                         (np.abs(logsfr - logsfr_ms) < opts['sigma_sf']))[0]

        # save mass ranges and which galaxies are on MS
        stack['hor'][zstr]['mass_range'] = (stellar_mass[on_ms].min(),stellar_mass[on_ms].max())
        stack['hor'][zstr]['mass_bins'] = np.linspace(stellar_mass[on_ms].min(),stellar_mass[on_ms].max(),opts['nbins_horizontal']+1)
        stack['hor'][zstr]['on_ms'] = on_ms
        percentiles = [0.5,opts['show_disp'][1],opts['show_disp'][0]]

        # for each main sequence bin, in mass, stack SFH
        for j in range(opts['nbins_horizontal']):
            print 'horizontal bin {0}'.format(j)
            # what galaxies are in this mass bin?
            # save individual mass and SFR
            in_bin = np.where((stellar_mass[on_ms] >= stack['hor'][zstr]['mass_bins'][j]) & \
                              (stellar_mass[on_ms] <= stack['hor'][zstr]['mass_bins'][j+1]))[0]

            tdict = {key:[] for key in ['median','err','errup','errdown']}
            tdict['logm'],tdict['logsfr'] = stellar_mass[on_ms][in_bin],logsfr[on_ms][in_bin]

            # calculate (SFR / M) chains on regular time grid
            # each draw has its own (SFR,t) vector with an associated weight
            # here we transform into (SFR/M)(t) and interpolate onto regular time grid
            # we do this with nearest-neighbor interpolation which is precisely correct for step-function SFH
            ngal = in_bin.shape[0]
            print '{0} galaxies in horizontal bin {1}'.format(ngal,j)
            ssfr,weights = np.empty(shape=(nsamps,nt,ngal)), np.empty(shape=(nsamps,ngal))

            indexes = zidx[on_ms][in_bin]
            for m, idx in enumerate(indexes):
                # weight by (t_univ(z=zgal) / t_univ(z=z_min)) to account for variation in t_univ
                # i.e. all galaxies in a given stack should have the same average sSFR
                weight = data['weights'][idx]
                sfh = data['ssfh'][idx]
                t_sfh = data['sfh_t'][idx]
                ssfh = sfh * t_sfh.max()/tbins.max()
                ssfh[:,:4] = (ssfh[:,0]*0.3 + ssfh[:,2]*0.7)[:,None] # weighted average to get ssfr_100 chain
                for i in range(nt):
                    if tbins[i] > t_sfh.max():
                        ssfr[:,i,m] = np.nan
                    else:
                        ssfr[:,i,m] = ssfh[:,np.abs(t_sfh - tbins[i]).argmin()]
                weights[:,m] = weight

            # now create stacked sSFR
            # this returns weighted sSFR median + quantiles
            for i in range(nt):
                # construct empty sSFR PDF and determine which galaxies contribute
                in_samp = np.where(np.isfinite(ssfr[0,i,:]))[0]
                hist = np.zeros(shape=ssfr_arr.shape[0]-1)

                # for each object in sample, add to sSFR PDF
                for idx in in_samp:
                    ssfr_in_hist = np.clip(ssfr[:,i,idx],10**ssfrmin,10**ssfrmax)
                    g1,_ = np.histogram(ssfr_in_hist, density=False,weights=weights[:,idx], bins=ssfr_arr)
                    hist += g1

                # calculate weighted quantiles from sSFR PDF at fixed T
                median, errup, errdown = weighted_quantile((ssfr_arr[1:]+ssfr_arr[:-1])/2., [0.5,.84,.16],weights=hist)

                tdict['median'] += [median]
                tdict['errup'] += [errup]
                tdict['errdown'] += [errdown]

            # save and dump
            tdict['err'] = prosp_dutils.asym_errors(np.array(tdict['median']),
                                                    np.array(tdict['errup']),
                                                    np.array(tdict['errdown']))
            stack['hor'][zstr]['bin'+str(j)] = tdict

        # stack vertical
        for j in range(opts['nbins_vertical']):
            print 'vertical bin {0}'.format(j)
            # define bin edges, then define members of the bin
            # special definition for first bin so we get quiescent galaxies too (?)
            tdict = {key:[] for key in ['median','err','errup','errdown']}
            sigup, sigdown = opts['sigma_sf']*2*(j-1.5), opts['sigma_sf']*2*(j-2.5)
            if j != 0 :
                in_bin = np.where((stellar_mass > opts['low_mass_cutoff']) & \
                         (stellar_mass < opts['high_mass_cutoff']) & \
                         ((logsfr - logsfr_ms) >= sigdown) & \
                         ((logsfr - logsfr_ms) < sigup))[0]
            else:
                in_bin = np.where((stellar_mass > opts['low_mass_cutoff']) & \
                         (stellar_mass < opts['high_mass_cutoff']) & \
                         ((logsfr - logsfr_ms) < sigup))[0]
            tdict['logm'],tdict['logsfr'] = stellar_mass[in_bin],logsfr[in_bin]

            # calculate (SFR / M) chains on regular time grid
            # each draw has its own (SFR,t) vector with an associated weight
            # here we transform into (SFR/M)(t) and interpolate onto regular time grid
            # we do this with nearest-neighbor interpolation which is precisely correct for step-function SFH
            ngal = in_bin.shape[0]
            print '{0} galaxies in vertical bin {1}'.format(ngal,j)
            ssfr,weights = np.empty(shape=(nsamps,nt,ngal)), np.empty(shape=(nsamps,ngal))
            indexes = zidx[in_bin]
            for m, idx in enumerate(indexes):
                # weight by (t_univ(z=zgal) / t_univ(z=z_min)) to account for variation in t_univ
                # i.e. all galaxies in a given stack should have the same average sSFR
                weight = data['weights'][idx]
                sfh = data['ssfh'][idx]
                t_sfh = data['sfh_t'][idx]
                ssfh = sfh * t_sfh.max()/tbins.max()
                ssfh[:,:4] = (ssfh[:,0]*0.3 + ssfh[:,2]*0.7)[:,None] # weighted average to get ssfr_100 chain
                for i in range(nt):
                    if tbins[i] > t_sfh.max():
                        ssfr[:,i,m] = np.nan
                    else:
                        ssfr[:,i,m] = ssfh[:,np.abs(t_sfh - tbins[i]).argmin()]
                weights[:,m] = weight

            # now create stacked sSFR
            # this returns weighted sSFR median + quantiles
            for i in range(nt):
                # construct empty sSFR PDF and determine which galaxies contribute
                in_samp = np.where(np.isfinite(ssfr[0,i,:]))[0]
                hist = np.zeros(shape=ssfr_arr.shape[0]-1)

                # for each object in sample, add to sSFR PDF
                for idx in in_samp:
                    ssfr_in_hist = np.clip(ssfr[:,i,idx],10**ssfrmin,10**ssfrmax)
                    g1,_ = np.histogram(ssfr_in_hist, normed=True,weights=weights[:,idx], bins=ssfr_arr)
                    hist += g1

                # calculate weighted quantiles from sSFR PDF at fixed T
                median, errup, errdown = weighted_quantile((ssfr_arr[1:]+ssfr_arr[:-1])/2., [0.5,.84,.16],weights=hist)

                tdict['median'] += [median]
                tdict['errup'] += [errup]
                tdict['errdown'] += [errdown]

            tdict['err'] = prosp_dutils.asym_errors(np.array(tdict['median']),
                                                    np.array(tdict['errup']),
                                                    np.array(tdict['errdown']))

            ### name your bin something creative!
            stack['vert'][zstr]['bin'+str(j)] = tdict

    return stack

def plot_stacked_sfh(dat,outfolder,**opts):

    # I/O
    outname_ms_horizontal = outfolder+'ms_horizontal_stack.png'
    outname_ms_vertical = outfolder+'ms_vertical_stack.png'

    # options for points on MS (left-hand side)
    fontsize = 14
    figsize = (13,8)
    ms_plot_opts = {
                    'alpha':0.5,
                    'mew':0.2,
                    'mec': 'k',
                    'linestyle':' ',
                    'marker':'o',
                    'ms':1.5
                   }
    # options for stripes delineating MS bins (top panels)
    size = 1.25
    ms_line_plot_opts = {
                         'lw':2.*size,
                         'linestyle':'-',
                         'alpha':1,
                         'zorder':-32
                        }
    # options for the stack plots (bottom panels)
    x_stack_offset = 0.02
    stack_plot_opts = {
                      'alpha':0.9,
                      'fmt':'o',
                      'mew':0.8,
                      'linestyle':'-',
                      'ms':10,
                      'lw':1,
                      'markeredgecolor': 'k',
                      'elinewidth':1.75,
                      'capsize':3,
                      'capthick':3
                      }
    lopts = {
             'alpha': 0.9,
             'ms': 0.0,
             'linestyle': '-',
             'zorder': 1,
             'lw': 3
            }
    fillopts = {
                'alpha': 0.25,
                'zorder': -1
               }


    # horizontal stack figure
    fig, ax = plt.subplots(2,4, figsize=figsize)
    for j, (z1,z2) in enumerate(opts['zbins']):
        
        # calculate zavg for bin placement
        zavg = (z1+z2)/2.
        zstr = "{:.2f}".format(zavg)
        plot_main_sequence(ax[0,j],zavg,**opts)

        for i in range(opts['nbins_horizontal']):
            
            # grab bin dictionary
            bdict = dat['hor'][zstr]['bin'+str(i)]

            # plot star-forming sequence
            ax[0,j].plot(bdict['logm'],bdict['logsfr'],
                         color=opts['horizontal_bin_colors'][i],
                         **ms_plot_opts)
            
            ax[1,j].fill_between(dat['hor'][zstr]['t'], bdict['errdown'], bdict['errup'], 
                               color=opts['horizontal_bin_colors'][i], **fillopts)
            ax[1,j].plot(dat['hor'][zstr]['t'], bdict['median'],
                       color=opts['horizontal_bin_colors'][i], **lopts)

            """
            # plot SFH stacks

            log_mean_t = np.log10(dat['hor'][zstr]['t']*1e9)
            ax[1,j].errorbar(10**(log_mean_t-x_stack_offset*(i-1)),bdict['median'],
                             yerr=bdict['err'],
                             color=opts['horizontal_bin_colors'][i],
                             **stack_plot_opts)
            """
        # labels and ranges
        ax[0,j].set_xlabel(r'log(M$_{*}$/M$_{\odot}$)',fontsize=fontsize)
        ax[0,j].set_xlim(dat['hor'][zstr]['mass_bins'].min()-0.5,dat['hor'][zstr]['mass_bins'].max()+0.5)
        ax[0,j].set_ylim(opts['ylim_horizontal_sfr'])
        ax[0,j].tick_params('both', pad=3.5, size=3.5, width=1.0, which='both',labelsize=fontsize)

        if j == 0:
            ax[0,j].set_ylabel(r'log(SFR)',fontsize=fontsize)
            ax[1,j].set_ylabel(r'SFR$_{\mathrm{bin}}$ / M$_{\mathrm{formed}}$ [yr$^{-1}$]',fontsize=fontsize)
            ax[0,j].text(0.98, 0.92, opts['zbin_labels'][j],ha='right',transform=ax[0,j].transAxes,fontsize=fontsize)
        else:
            ax[0,j].text(0.02, 0.92, opts['zbin_labels'][j],ha='left',transform=ax[0,j].transAxes,fontsize=fontsize)
            for a in ax[:,j]:
                for tl in a.get_yticklabels():tl.set_visible(False)
                plt.setp(a.get_yminorticklabels(), visible=False)

        ax[1,j].set_xlim(opts['xlim_t'])
        ax[1,j].set_ylim(opts['ylim_horizontal_ssfr'])
        ax[1,j].set_xlabel(r'time [Gyr]',fontsize=fontsize)

        ax[1,j].set_xscale('log',subsx=(1,3),nonposx='clip')
        ax[1,j].set_yscale('log',nonposy='clip')
        ax[1,j].xaxis.set_major_formatter(FormatStrFormatter('%2.5g'))
        ax[1,j].xaxis.set_minor_formatter(FormatStrFormatter('%2.5g'))
        ax[1,j].tick_params('both', pad=3.5, size=3.5, width=1.0, which='both',labelsize=fontsize)

        # plot mass ranges 
        # must happen after ylim is determined
        ylim = ax[0,j].get_ylim()
        for i in xrange(opts['nbins_horizontal']):
            idx = 'massbin'+str(i)
            if i == 0:
                dlowbin = -0.01*size
            else:
                dlowbin = 0.017*size
            if i == opts['nbins_horizontal']-1:
                dhighbin = 0.01*size
            else:
                dhighbin = -0.017*size

            ax[0,j].plot(np.repeat(dat['hor'][zstr]['mass_bins'][i]+dlowbin,2),ylim,
                       color=opts['horizontal_bin_colors'][i],
                       **ms_line_plot_opts)
            ax[0,j].plot(np.repeat(dat['hor'][zstr]['mass_bins'][i+1]+dhighbin,2),ylim,
                       color=opts['horizontal_bin_colors'][i],
                       **ms_line_plot_opts)
            ax[0,j].set_ylim(ylim)

    plt.tight_layout(w_pad=0.0)
    fig.subplots_adjust(wspace=0.0)
    plt.savefig(outname_ms_horizontal,dpi=150)
    plt.close()


    # vertical stack figure
    fig, ax = plt.subplots(2,4, figsize=figsize)

    # plot main sequence position and stacks
    for j, (z1,z2) in enumerate(opts['zbins']):

        # calculate zavg for bin placement
        labels = ['quiescent','below MS','on MS','above MS']
        zavg = (z1+z2)/2.
        zstr = "{:.2f}".format(zavg)

        for i in range(opts['nbins_vertical']):

            # plot star-forming sequence
            bdict = dat['vert'][zstr]['bin'+str(i)]
            ax[0,j].plot(bdict['logm'],bdict['logsfr'],
                       color=opts['vertical_bin_colors'][i],
                       **ms_plot_opts)

            
            ax[1,j].fill_between(dat['vert'][zstr]['t'], bdict['errdown'], bdict['errup'], 
                               color=opts['vertical_bin_colors'][i], **fillopts)
            ax[1,j].plot(dat['vert'][zstr]['t'], bdict['median'],
                       color=opts['vertical_bin_colors'][i], **lopts)

            """
            # plot SFH stacks
            log_mean_t = np.log10(dat['vert'][zstr]['t']*1e9)
            ax[1,j].errorbar(10**(log_mean_t-x_stack_offset*(i-1)),bdict['median'],
                             yerr=bdict['err'],
                             color=opts['vertical_bin_colors'][i],
                             **stack_plot_opts)
            """
            if i == 0:
                minmass,maxmass = bdict['logm'].min(),bdict['logm'].max()
                minsfr,maxsfr = bdict['logsfr'].min(),bdict['logsfr'].max()
            else:
                minmass = np.min([minmass,bdict['logm'].min()])
                maxmass = np.max([maxmass,bdict['logm'].max()])
                minsfr = np.min([minsfr,bdict['logsfr'].min()])
                maxsfr = np.max([maxsfr,bdict['logsfr'].max()])

        # labels and ranges
        xlim = (minmass-0.6,maxmass+0.3)
        ax[0,j].set_xlim(xlim)
        ax[0,j].set_ylim(opts['ylim_vertical_sfr'])
        ax[0,j].tick_params('both', pad=3.5, size=3.5, width=1.0, which='both',labelsize=fontsize)
        ax[0,j].set_xlabel(r'log(M$_{*}$/M$_{\odot}$)',fontsize=fontsize)

        ax[1,j].set_xlim(opts['xlim_t'])
        ax[1,j].set_ylim(opts['ylim_vertical_ssfr'])
        ax[1,j].tick_params('both', pad=3.5, size=3.5, width=1.0, which='both',labelsize=fontsize)
        ax[1,j].set_xscale('log',subsx=(1,3),nonposx='clip')
        ax[1,j].set_yscale('log',nonposy='clip')
        ax[1,j].xaxis.set_major_formatter(FormatStrFormatter('%2.5g'))
        ax[1,j].xaxis.set_minor_formatter(FormatStrFormatter('%2.5g'))
        ax[1,j].set_xlabel(r'time [Gyr]',fontsize=fontsize)

        # plot mass ranges
        xr = np.linspace(xlim[0],xlim[1],50)
        yr = sfr_ms(np.full(xr.shape[0],zavg),xr,**opts)
        idx = 1
        for i in range(opts['nbins_vertical']):

            # complicated logic for nudges to bottom, top lines
            dlowbin = 0.017*size
            if i == 0:
                dlowbin = -0.01*size
            
            dhighbin = -0.017*size
            if i == opts['nbins_vertical']-1:
                dhighbin = 0.01*size

            # where are we drawing the lines?
            sigup, sigdown = opts['sigma_sf']*2*(i-1.5), opts['sigma_sf']*2*(i-2.5)

            # no bottom line for the bottom one
            # adjust label for bottom one
            if i != 0:
                ax[0,j].plot(xr,yr+sigdown+dlowbin,
                             color=opts['vertical_bin_colors'][i],
                             label = labels[i],
                             **ms_line_plot_opts)

            # no top line for the top one
            if i != opts['nbins_vertical']-1:
                ax[0,j].plot(xr,yr+sigup+dhighbin,
                           color=opts['vertical_bin_colors'][i],
                           label = labels[i],
                           **ms_line_plot_opts)

        
        if j == 0:
            ax[0,j].set_ylabel(r'log(SFR M$_{\odot}$ yr$^{-1}$)',fontsize=fontsize)
            ax[1,j].set_ylabel(r'SFR(t)/M$_{\mathrm{tot}}$ [yr$^{-1}$]',fontsize=fontsize)
            ax[0,j].text(0.98, 0.92, opts['zbin_labels'][j],ha='right',transform=ax[0,j].transAxes,fontsize=fontsize)

            handles, labels = ax[0,j].get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            ax[0,j].legend(by_label.values(), by_label.keys(),
                           loc=2, prop={'size':9},
                           scatterpoints=1,fancybox=True,ncol=1)
        else:
            ax[0,j].text(0.02, 0.92, opts['zbin_labels'][j],ha='left',transform=ax[0,j].transAxes,fontsize=fontsize)
            for a in ax[:,j]:
                for tl in a.get_yticklabels():tl.set_visible(False)
                plt.setp(a.get_yminorticklabels(), visible=False)



    plt.tight_layout(w_pad=0.0)
    fig.subplots_adjust(wspace=0.0)
    plt.savefig(outname_ms_vertical,dpi=150)
    plt.close()

def plot_main_sequence(ax,z,sigma_sf=None,low_mass_cutoff=7, **junk):

    mass = np.linspace(low_mass_cutoff,12,40)
    sfr = sfr_ms(np.full(mass.shape[0],z),mass,**junk)

    ax.plot(mass, sfr,
              color='green',
              alpha=0.8,
              lw=2.5,
              label='Whitaker+14',
              zorder=-1)
    ax.fill_between(mass, sfr-sigma_sf, sfr+sigma_sf, 
                      color='green',
                      alpha=0.3)

def sfr_ms(z,logm,adjust_sfr=0.0,**opts):
    """ returns the SFR of the star-forming sequence from Whitaker+14
    as a function of mass and redshift
    note that this is strictly only valid over 0.5 < z < 2.5
    we use the broken power law form (as opposed to the quadratic form)
    """
    z, logm = np.atleast_1d(z), np.atleast_1d(logm)

    # parameters from whitaker+14
    zwhit = np.array([0.75, 1.25, 1.75, 2.25])
    alow = np.array([0.94,0.99,1.04,0.91])
    ahigh = np.array([0.14,0.51,0.62,0.67])
    b = np.array([1.11, 1.31, 1.49, 1.62])

    def sfr_bpl(alow,ahigh,b,logm):
        if logm < 10.2:
            return alow*(logm - 10.2) + b
        else:
            return ahigh*(logm - 10.2) + b

    # build grids
    outsfr = []
    for (m,zred) in zip(logm,z):
        logsfr = sfr_bpl(alow,ahigh,b,m)
        outsfr += [np.interp(zred,zwhit,logsfr)]
    outsfr = np.array(outsfr)
    if adjust_sfr:
        outsfr += adjust_sfr

    return outsfr



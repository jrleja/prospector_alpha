import numpy as np
import matplotlib.pyplot as plt
import os, hickle, td_io, prosp_dutils, copy
from prospector_io import load_prospector_data, find_all_prospector_results
from astropy.cosmology import WMAP9
import brownseds_highz_params as pfile
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from dynesty.plotting import _quantile as weighted_quantile
from collections import OrderedDict

plt.ioff()

popts = {'fmt':'o', 'capthick':1.5,'elinewidth':1.5,'ms':9,'alpha':0.8,'color':'0.3'}
red = '#FF3D0D'
dpi = 160
cmap = 'cool'
minsfr = 0.01

def salim_mainsequence(mass,ssfr=False,**junk):
    '''
    mass in log, returns SFR in log
    '''
    ssfr_salim = -0.35*(mass-10)-9.83
    if ssfr:
        return ssfr_salim
    salim_sfr = np.log10(10**ssfr_salim*10**mass)
    return salim_sfr

def transform_zfraction_to_mfraction(zfraction,time_per_bin):
    """vectorized, partially ripped from prosp_dutils
    """
    
    sfr_fraction = np.zeros_like(zfraction)
    sfr_fraction[...,0] = 1-zfraction[...,0]
    for i in range(1,sfr_fraction.shape[-1]): sfr_fraction[...,i] = np.prod(zfraction[...,:i],axis=-1)*(1-zfraction[...,i])
    sfr_fraction_full = np.concatenate((sfr_fraction,(1-sfr_fraction.sum(axis=-1))[...,None]),axis=-1)

    mass_fraction = sfr_fraction_full*time_per_bin[:,None,:]
    mass_fraction /= mass_fraction.sum(axis=-1)[...,None]

    return mass_fraction

def integrate_sfh_vectorized(t1,t2,agebins,zfraction):
    """ integrate star formation history from t1 to t2
    returns fraction of *total* (not stellar) mass formed in time inteval
    AGEBINS is shape (NGAL, NBIN, 2)
    ZFRACTION is shape (NGAL, NSAMP, NBIN-1)
    """

    # translate agebins to time bins in correct units
    linear_bins = 10**agebins/1e9
    time_per_bin = linear_bins[...,1]-linear_bins[...,0]
    maxtimes, mintimes = linear_bins[...,1].max(axis=1), linear_bins[...,0].min(axis=1)
    time_bins = maxtimes[:,None,None] - linear_bins

    # mass fractions
    mfraction = transform_zfraction_to_mfraction(zfraction,time_per_bin)

    # sanitize time inputs so they are in the bin range.
    t1 = np.clip(mintimes, t1, maxtimes)
    t2 = np.clip(maxtimes, mintimes, t2)

    # calculate time-weights for each bin
    time_above_min = -np.diff(np.clip(time_bins - t1[:,None,None],0,np.inf)).squeeze()
    time_below_max = np.diff(np.clip(t2[:,None,None]-time_bins,0,np.inf)).squeeze()
    tweights = np.minimum(time_below_max,time_above_min) / time_per_bin

    # convert to total mass formed
    mformed = (mfraction * tweights[:,None,:]).sum(axis=-1)

    return mformed

def calculate_sfr(agebins, zfraction, timescale, dt):
    """ timescale = size of time chunk
        dt = time before observation
        returns (FRACTIONAL MASS FORMED IN TIME INTERVAL) / (TIME INTERVAL)
    """

    linear_bins = 10**agebins/1e9
    maxtimes = linear_bins[...,1].max(axis=1)
    sfr = integrate_sfh_vectorized(maxtimes-timescale-dt, maxtimes-dt, agebins,zfraction) / timescale

    return sfr

def do_all(runname='brownseds_highz', outfolder=None, regenerate=False, regenerate_stack=False, **opts):

    if outfolder is None:
        outfolder = os.getenv('APPS') + '/prospector_alpha/plots/'+runname+'/pcomp/'
        if not os.path.isdir(outfolder):
            os.makedirs(outfolder)
            os.makedirs(outfolder+'data/')

    stack_opts = {
              'sigma_sf':0.3,                  # scatter in the star-forming sequence, in dex
              'nbins_horizontal':3,            # number of bins in horizontal stack
              'nbins_vertical':4,              # number of bins in vertical stack
              'horizontal_bin_colors': ['#45ADA8','#FC913A','#FF4E50'],
              'vertical_bin_colors': ['red','#FC913A','#45ADA8','#323299'],
              'low_mass_cutoff':9.5,          # log(M) where we stop stacking and plotting
              'high_mass_cutoff': 11.5,
              'ylim_horizontal_sfr': (-0.8,3),
              'ylim_horizontal_ssfr': (1e-13,1e-9),
              'ylim_vertical_sfr': (-3,3),
              'ylim_vertical_ssfr': (1e-13,1e-9),
              'xlim_t': (1e7,1.4e10),
              'show_disp':[0.16,0.84]         # percentile of population distribution to show on plot
             }

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
    """

    # if it's already made, load it and give it back
    # else, start with the making!
    if os.path.isfile(filename) and regenerate == False:
        print 'loading all data'
        with open(filename, "r") as f:
            outdict=hickle.load(f)

        return outdict

    # define output containers
    outvar = ['stellar_mass','sfr_30', 'sfr_100','half_time']
    outdict = {q: {f: [] for f in ['q50','q84','q16']} for q in outvar}
    for f in ['objname','agebins', 'weights', 'z_fraction']: outdict[f] = [] 

    # we want MASS, SFR_100, Z_FRACTION CHAIN, and AGEBINS for each galaxy
    pfile.run_params['zred'] = None # make sure this is reset
    basenames = find_all_prospector_results(runname)
    for i, name in enumerate(basenames):

        # load output from fit
        try:
            res, _, model, prosp = load_prospector_data(name)
        except:
            print name.split('/')[-1]+' failed to load. skipping.'
            continue
        if (res is None) or (prosp is None):
            continue

        outdict['objname'] += [name.split('/')[-1]]
        print 'loaded ' + outdict['objname'][-1]

        # agebins (and generate model)
        pfile.run_params['objname'] = outdict['objname'][-1]
        model = pfile.load_model(**pfile.run_params)
        outdict['agebins'] += [model.params['agebins']]

        # zfraction
        zidx = model.theta_index['z_fraction']
        outdict['z_fraction'] += [res['chain'][prosp['sample_idx'], zidx]]
        outdict['weights'] += [prosp['weights']]

        # extra variables
        for v in outvar:
            for f in ['q50','q84','q16']: outdict[v][f] += [prosp['extras'][v][f]]

    # dump files and return
    hickle.dump(outdict,open(filename, "w"))
    return outdict

def stack_sfh(data, **opts):
    """stack in VERTICAL and HORIZONTAL slices on the main sequence
    we measure the average sSFR for each galaxy
    then report the median of the average for each bin
    the error bars are the 1sigma percentiles of the averages
    """

    # stack horizontal
    # first iterate over redshift
    stack = {'hor':{},'vert': {}}

    # calculate the time bins at average redshift
    model = pfile.load_model(**pfile.run_params)
    stack['hor'], stack['vert'] = {}, {}
    stack['hor']['agebins'] = 10**model.params['agebins']
    stack['vert']['agebins'] = 10**model.params['agebins']

    # calculate SFR(MS) for each galaxy
    # perhaps should calculate at z_gal for accuracy?
    stellar_mass = np.log10(data['stellar_mass']['q50'])
    logsfr = np.log10(data['sfr_30']['q50'])
    logsfr_ms = salim_mainsequence(stellar_mass,**opts)
    on_ms = (stellar_mass > opts['low_mass_cutoff']) & \
            (stellar_mass < opts['high_mass_cutoff']) & \
            (np.abs(logsfr - logsfr_ms) < opts['sigma_sf'])

    # save mass ranges and which galaxies are on MS
    stack['hor']['mass_range'] = (stellar_mass[on_ms].min(),stellar_mass[on_ms].max())
    stack['hor']['mass_bins'] = np.linspace(stellar_mass[on_ms].min(),stellar_mass[on_ms].max(),opts['nbins_horizontal']+1)
    stack['hor']['on_ms'] = on_ms
    percentiles = [0.5,opts['show_disp'][1],opts['show_disp'][0]]

    # for each main sequence bin, in mass, stack SFH
    for j in range(opts['nbins_horizontal']):
        
        # what galaxies are in this mass bin?
        # save individual mass and SFR
        in_bin = (stellar_mass[on_ms] >= stack['hor']['mass_bins'][j]) & \
                 (stellar_mass[on_ms] <= stack['hor']['mass_bins'][j+1])

        tdict = {key:[] for key in ['median','err','errup','errdown']}
        tdict['logm'],tdict['logsfr'] = stellar_mass[on_ms][in_bin],logsfr[on_ms][in_bin]

        # calculate sSFR chains (fn / sum(tn*fn))
        # first transform to SFR fraction
        # then add the nth bin
        # finally, ensure it's normalized (no floating point errors)
        # this is where we'd use the individual agebins if we so chose!
        outfrac, outweight = [], []
        for zf,weight,agebins in zip(np.array(data['z_fraction'])[on_ms][in_bin], 
                                     np.array(data['weights'])[on_ms][in_bin],
                                     np.array(data['agebins'])[on_ms][in_bin]):
            frac = prosp_dutils.transform_zfraction_to_sfrfraction(zf)
            frac = np.concatenate((frac, (1-frac.sum(axis=1))[:,None]),axis=1)
            time_per_bin = np.diff(10**agebins, axis=-1)[:,0]
            norm = (frac * time_per_bin).sum(axis=1) # sum(fn*tn)
            outfrac += [frac/norm[:,None]]
            outweight += [weight]

        # calculate weighted mean for each SFR bin
        for i in range(time_per_bin.shape[0]):
            frac = [f[:,i] for f in outfrac]
            mean,errup,errdown = calculate_median(frac, outweight)
            tdict['median'] += [mean]
            tdict['errup'] += [errup]
            tdict['errdown'] += [errdown]

        # save and dump
        tdict['err'] = prosp_dutils.asym_errors(np.array(tdict['median']),
                                                np.array(tdict['errup']),
                                                np.array(tdict['errdown']))
        stack['hor']['bin'+str(j)] = tdict

    # stack vertical
    for j in range(opts['nbins_vertical']):

        # define bin edges, then define members of the bin
        # special definition for first bin so we get quiescent galaxies too (?)
        tdict = {key:[] for key in ['median','err','errup','errdown']}
        sigup, sigdown = opts['sigma_sf']*2*(j-1.5), opts['sigma_sf']*2*(j-2.5)
        if j != 0 :
            in_bin = (stellar_mass > opts['low_mass_cutoff']) & \
                     (stellar_mass < opts['high_mass_cutoff']) & \
                     ((logsfr - logsfr_ms) >= sigdown) & \
                     ((logsfr - logsfr_ms) < sigup)
        else:
            in_bin = (stellar_mass > opts['low_mass_cutoff']) & \
                     (stellar_mass < opts['high_mass_cutoff']) & \
                     ((logsfr - logsfr_ms) < sigup)
        tdict['logm'],tdict['logsfr'] = stellar_mass[in_bin],logsfr[in_bin]

        # calculate sSFR chains (fn / sum(tn*fn))
        # first transform to SFR fraction
        # then add the nth bin
        # finally, ensure it's normalized (no floating point errors)
        # this is where we'd use the individual agebins if we so chose!
        outfrac, outweight = [], []
        for zf,weight,agebins in zip(np.array(data['z_fraction'])[in_bin], 
                                     np.array(data['weights'])[in_bin],
                                     np.array(data['agebins'])[in_bin]):
            frac = prosp_dutils.transform_zfraction_to_sfrfraction(zf)
            frac = np.concatenate((frac, (1-frac.sum(axis=1))[:,None]),axis=1)
            time_per_bin = np.diff(10**agebins, axis=-1)[:,0]
            norm = (frac * time_per_bin).sum(axis=1) # sum(fn*tn)
            outfrac += [frac/norm[:,None]]
            outweight += [weight]

        # calculate median of sSFR for each SFR bin
        for i in range(time_per_bin.shape[0]):
            frac = [f[:,i] for f in outfrac]
            mean,errup,errdown = calculate_median(frac, outweight)
            tdict['median'] += [mean]
            tdict['errup'] += [errup]
            tdict['errdown'] += [errdown]

        tdict['err'] = prosp_dutils.asym_errors(np.array(tdict['median']),
                                                 np.array(tdict['errup']),
                                                 np.array(tdict['errdown']))

        ### name your bin something creative!
        stack['vert']['bin'+str(j)] = tdict

    return stack

def plot_stacked_sfh(dat,outfolder,**opts):

    # I/O
    outname_ms_horizontal = outfolder+'ms_new_horizontal_stack.png'
    outname_ms_vertical = outfolder+'ms_new_vertical_stack.png'

    # options for points on MS (left-hand side)
    fontsize = 14
    figsize = (8,4)
    ms_plot_opts = {
                    'alpha':0.9,
                    'mew':0.2,
                    'mec': 'k',
                    'linestyle':' ',
                    'marker':'o',
                    'ms':4
                   }
    # options for stripes delineating MS bins (left-hand side)
    size = 1.25
    ms_line_plot_opts = {
                         'lw':2.*size,
                         'linestyle':'-',
                         'alpha':0.8,
                         'zorder':-32
                        }
    # options for the stack plots (right-hand side)
    x_stack_offset = 0.03
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

    # horizontal stack figure
    fig, ax = plt.subplots(1,2, figsize=figsize)
    plot_main_sequence(ax[0],**opts)

    for i in range(opts['nbins_horizontal']):
        
        # grab bin dictionary
        bdict = dat['hor']['bin'+str(i)]

        # plot star-forming sequence
        ax[0].plot(bdict['logm'],bdict['logsfr'],
                     color=opts['horizontal_bin_colors'][i],
                     **ms_plot_opts)

        # plot SFH stacks
        log_mean_t = np.log10(np.mean(dat['hor']['agebins'],axis=1))
        ax[1].errorbar(10**(log_mean_t-x_stack_offset*(i-1)),bdict['median'],
                         yerr=bdict['err'],
                         color=opts['horizontal_bin_colors'][i],
                         **stack_plot_opts)

    # labels and ranges
    ax[0].set_xlabel(r'log(M$_{*}$/M$_{\odot}$)',fontsize=fontsize)
    ax[0].set_xlim(dat['hor']['mass_bins'].min()-0.5,dat['hor']['mass_bins'].max()+0.5)
    ax[0].set_ylim(opts['ylim_horizontal_sfr'])
    ax[0].tick_params('both', pad=3.5, size=3.5, width=1.0, which='both',labelsize=fontsize)

    ax[0].set_ylabel(r'log(SFR)',fontsize=fontsize)
    ax[1].set_ylabel(r'SFR$_{\mathrm{bin}}$ / M$_{\mathrm{formed}}$ [yr$^{-1}$]',fontsize=fontsize)

    ax[1].set_xlim(opts['xlim_t'])
    ax[1].set_ylim(opts['ylim_horizontal_ssfr'])
    ax[1].set_xlabel(r'time [yr]',fontsize=fontsize)

    ax[1].set_xscale('log',nonposx='clip')
    ax[1].set_yscale('log',nonposy='clip')
    ax[1].tick_params('both', pad=3.5, size=3.5, width=1.0, which='both',labelsize=fontsize)

    # plot mass ranges 
    # must happen after ylim is determined
    ylim = ax[0].get_ylim()
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

        ax[0].plot(np.repeat(dat['hor']['mass_bins'][i]+dlowbin,2),ylim,
                   color=opts['horizontal_bin_colors'][i],
                   **ms_line_plot_opts)
        ax[0].plot(np.repeat(dat['hor']['mass_bins'][i+1]+dhighbin,2),ylim,
                   color=opts['horizontal_bin_colors'][i],
                   **ms_line_plot_opts)
        ax[0].set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(outname_ms_horizontal,dpi=150)
    plt.close()


    # vertical stack figure
    fig, ax = plt.subplots(1,2, figsize=figsize)

    # calculate zavg for bin placement
    labels = ['quiescent','below MS','on MS','above MS']

    for i in range(opts['nbins_vertical']):

        # plot star-forming sequence
        bdict = dat['vert']['bin'+str(i)]
        ax[0].plot(bdict['logm'],bdict['logsfr'],
                   color=opts['vertical_bin_colors'][i],
                   **ms_plot_opts)

        # plot SFH stacks
        log_mean_t = np.log10(np.mean(dat['vert']['agebins'],axis=1))
        ax[1].errorbar(10**(log_mean_t-x_stack_offset*(i-1)),bdict['median'],
                         yerr=bdict['err'],
                         color=opts['vertical_bin_colors'][i],
                         **stack_plot_opts)
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
    ax[0].set_xlim(xlim)
    ax[0].set_ylim(opts['ylim_vertical_sfr'])
    ax[0].tick_params('both', pad=3.5, size=3.5, width=1.0, which='both',labelsize=fontsize)
    ax[0].set_xlabel(r'log(M$_{*}$/M$_{\odot}$)',fontsize=fontsize)

    ax[1].set_xlim(opts['xlim_t'])
    ax[1].set_ylim(opts['ylim_vertical_ssfr'])
    ax[1].tick_params('both', pad=3.5, size=3.5, width=1.0, which='both',labelsize=fontsize)
    ax[1].set_xscale('log',nonposx='clip')
    ax[1].set_yscale('log',nonposy='clip')
    ax[1].set_xlabel(r'time [yr]',fontsize=fontsize)

    # plot mass ranges
    xr = np.linspace(xlim[0],xlim[1],50)
    yr = salim_mainsequence(xr,**opts)
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
            ax[0].plot(xr,yr+sigdown+dlowbin,
                         color=opts['vertical_bin_colors'][i],
                         label = labels[i],
                         **ms_line_plot_opts)

        # no top line for the top one
        if i != opts['nbins_vertical']-1:
            ax[0].plot(xr,yr+sigup+dhighbin,
                       color=opts['vertical_bin_colors'][i],
                       label = labels[i],
                       **ms_line_plot_opts)

    
    ax[0].set_ylabel(r'log(SFR M$_{\odot}$ yr$^{-1}$)',fontsize=fontsize)
    ax[1].set_ylabel(r'median SFR(t)/M$_{\mathrm{tot}}$ [yr$^{-1}$]',fontsize=fontsize)

    handles, labels = ax[0].get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax[0].legend(by_label.values(), by_label.keys(),
                   loc=2, prop={'size':9},
                   scatterpoints=1,fancybox=True,ncol=1)


    plt.tight_layout()
    plt.savefig(outname_ms_vertical,dpi=150)
    plt.close()

def plot_main_sequence(ax,sigma_sf=None,low_mass_cutoff=7, **junk):

    mass = np.linspace(low_mass_cutoff,12,40)
    sfr = salim_mainsequence(mass,**junk)

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
    note that this is only valid over 0.5 < z < 2.5.
    we use the broken power law form (as opposed to the quadratic form)
    """
    z, logm = np.atleast_1d(z), np.atleast_1d(logm)

    # parameters from whitaker+14
    zwhit = np.array([0.75, 1.25, 1.75, 2.25])
    alow = np.array([0.94,0.99,1.04,0.91])
    ahigh = np.array([0.14,0.51,0.62,0.67])
    b = np.array([1.11, 1.31, 1.49, 1.62])

    # check with redshift
    idx = np.where(zwhit == z[0])[0][0]
    if idx == -1:
        print 'you know this is really poorly implemented, fix it'
        print 1/0

    # generate SFR(M) at all redshifts 
    log_sfr = alow[idx]*(logm - 10.2) + b[idx]
    high = (logm > 10.2).squeeze()
    log_sfr[high] = ahigh[idx]*(logm[high] - 10.2) + b[idx]
    
    if adjust_sfr:
        log_sfr += adjust_sfr

    return log_sfr

def calculate_median(frac, weight, ndraw=500000):
    '''
    given list of N PDFs
    calculate the mean by sampling from the PDFs
    '''
    nobj = len(frac)
    mean_array = np.zeros(shape=(ndraw,nobj))
    for i,f in enumerate(frac): mean_array[:,i] = np.random.choice(f,size=ndraw,p=weight[i]/weight[i].sum())
    mean_pdf = mean_array.mean(axis=1)
    mean, errup, errdown = np.percentile(mean_pdf, [0.5,0.84,0.16])

    return mean, errup, errdown

'''
def calculate_median(pdf, weight):
    """given list of N PDFs, sum them
    and return median + 16th, 84th percentile of sum
    """
    ssfrmin, ssfrmax = -13, -8
    sfrac_arr = 10**np.linspace(ssfrmin,ssfrmax,1001)
    hist = np.zeros(shape=sfrac_arr.shape[0]-1)

    for i, (f,w) in enumerate(zip(pdf,weight)):
        g1,_ = np.histogram(np.clip(f,10**ssfrmin,10**ssfrmax), normed=True,weights=w, bins=sfrac_arr)
        hist += g1
    median, errup, errdown = weighted_quantile((sfrac_arr[1:]+sfrac_arr[:-1])/2., [0.5,.84,.16],weights=hist)
    """
    nobj = len(pdf)
    median_array = []

    for i,(f,w) in enumerate(zip(pdf,weight)): median_array += [weighted_quantile(f, np.array([0.5]), weights=w)]
    median, errup, errdown = np.percentile(median_array, [50,84,16])
    """
    return median, errup, errdown
'''

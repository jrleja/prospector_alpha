import numpy as np
import matplotlib.pyplot as plt
import os, hickle, td_io, prosp_dutils, copy
from prospector_io import load_prospector_data
from astropy.cosmology import WMAP9
import td_huge_params as pfile
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from dynesty.plotting import _quantile as weighted_quantile

plt.ioff()

popts = {'fmt':'o', 'capthick':1.5,'elinewidth':1.5,'ms':9,'alpha':0.8,'color':'0.3'}
red = '#FF3D0D'
dpi = 160
cmap = 'cool'
minsfr = 0.01

def do_all(runname='td_huge', outfolder=None,**opts):

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
              'vertical_bin_colors': ['red','#FC913A','#45ADA8','#323299'],
              'low_mass_cutoff':9.0,          # log(M) where we stop stacking and plotting
              'show_disp':[0.16,0.84],         # percentile of population distribution to show on plot
              'adjust_sfr': -0.25,             # adjust whitaker SFRs by how much?
              'zbins': [(0.5,1.),(1.,1.5),(1.5,2.),(2.,2.5)]
             }

    data = collate_data(runname,filename=outfolder+'data/stacksfh.h5',**opts)
    stack = stack_sfh(data, **stack_opts)
    plot_stacked_sfh(stack,outfolder, **stack_opts)


def collate_data(runname, filename=None, regenerate=False, **opts):
    """ pull out all of the necessary information from the individual data files
    this takes awhile, so this data is saved to disk.
    """


    # if it's already made, load it and give it back
    # else, start with the making!
    if os.path.isfile(filename) and regenerate == False:
        with open(filename, "r") as f:
            outdict=hickle.load(f)

        return outdict

    # define output containers
    outvar = ['stellar_mass','sfr_100','half_time']
    outdict = {q: {f: [] for f in ['q50','q84','q16']} for q in outvar}
    for f in ['objname','agebins', 'weights', 'z_fraction', 'zred']: outdict[f] = [] 

    # we want MASS, SFR_100, Z_FRACTION CHAIN, and AGEBINS for each galaxy
    pfile.run_params['zred'] = None # make sure this is reset
    basenames, _, _ = prosp_dutils.generate_basenames(runname)
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
        outdict['zred'] += [float(model.params['zred'])]

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
    for (z1, z2) in opts['zbins']:

        # generate zavg, add containers
        zavg = (z1+z2)/2.
        zstr = "{:.2f}".format(zavg)

        # calculate the time bins at average redshift
        pfile.run_params['zred'] = zavg
        model = pfile.load_model(**pfile.run_params)
        stack['hor'][zstr], stack['vert'][zstr] = {}, {}
        stack['hor'][zstr]['agebins'] = 10**model.params['agebins']
        stack['vert'][zstr]['agebins'] = 10**model.params['agebins']

        # define galaxies in redshift bin
        data['zred'] = np.array([float(z) for z in data['zred']]) # this can be removed once regenerate is run again
        zidx = (data['zred'] > z1) & (data['zred'] <= z2)

        # calculate SFR(MS) for each galaxy
        # perhaps should calculate at z_gal for accuracy?
        stellar_mass = np.log10(data['stellar_mass']['q50'])[zidx]
        logsfr = np.log10(data['sfr_100']['q50'])[zidx]
        logsfr_ms = sfr_ms(zavg,stellar_mass,**opts)
        on_ms = (stellar_mass > opts['low_mass_cutoff']) & \
                (np.abs(logsfr - logsfr_ms) < opts['sigma_sf'])

        # save mass ranges and which galaxies are on MS
        stack['hor'][zstr]['mass_range'] = (stellar_mass[on_ms].min(),stellar_mass[on_ms].max())
        stack['hor'][zstr]['mass_bins'] = np.linspace(stellar_mass[on_ms].min(),stellar_mass[on_ms].max(),opts['nbins_horizontal']+1)
        stack['hor'][zstr]['on_ms'] = on_ms
        percentiles = [0.5,opts['show_disp'][1],opts['show_disp'][0]]

        # for each main sequence bin, in mass, stack SFH
        for j in range(opts['nbins_horizontal']):
            
            # what galaxies are in this mass bin?
            # save individual mass and SFR
            in_bin = (stellar_mass[on_ms] >= stack['hor'][zstr]['mass_bins'][j]) & \
                     (stellar_mass[on_ms] <= stack['hor'][zstr]['mass_bins'][j+1])
            tdict = {key:[] for key in ['mean','err','errup','errdown']}
            tdict['logm'],tdict['logsfr'] = stellar_mass[on_ms][in_bin],logsfr[on_ms][in_bin]

            # calculate sSFR chains (fn / sum(tn*fn))
            # first transform to SFR fraction
            # then add the nth bin
            # finally, ensure it's normalized (no floating point errors)
            # this is where we'd use the individual agebins if we so chose!
            outfrac, outweight = [], []
            for zf,weight,agebins in zip(np.array(data['z_fraction'])[zidx][on_ms][in_bin], 
                                         np.array(data['weights'])[zidx][on_ms][in_bin],
                                         np.array(data['agebins'])[zidx][on_ms][in_bin]):
                frac = prosp_dutils.transform_zfraction_to_sfrfraction(zf)
                frac = np.concatenate((frac, (1-frac.sum(axis=1))[:,None]),axis=1)
                time_per_bin = np.diff(10**agebins, axis=-1)[:,0]
                norm = (frac * time_per_bin).sum(axis=1) # sum(fn*tn)
                outfrac += [frac/norm[:,None]]
                outweight += [weight]

            # calculate weighted mean for each SFR bin
            for i in range(time_per_bin.shape[0]):
                frac = [f[:,i] for f in outfrac]
                mean,errup,errdown = calculate_mean(frac, outweight)
                tdict['mean'] += [mean]
                tdict['errup'] += [errup]
                tdict['errdown'] += [errdown]

            # save and dump
            tdict['err'] = prosp_dutils.asym_errors(np.array(tdict['mean']),
                                                    np.array(tdict['errup']),
                                                    np.array(tdict['errdown']))
            stack['hor'][zstr]['bin'+str(j)] = tdict

        # stack vertical
        for j in range(opts['nbins_vertical']):

            # define bin edges, then define members of the bin
            # special definition for first bin so we get quiescent galaxies too (?)
            tdict = {key:[] for key in ['mean','err','errup','errdown']}
            sigup, sigdown = opts['sigma_sf']*2*(j-1.5), opts['sigma_sf']*2*(j-2.5)
            if j != 0 :
                in_bin = (stellar_mass > opts['low_mass_cutoff']) & \
                         ((logsfr - logsfr_ms) >= sigdown) & \
                         ((logsfr - logsfr_ms) < sigup)
            else:
                in_bin = (stellar_mass > opts['low_mass_cutoff']) & \
                         ((logsfr - logsfr_ms) < sigup)
            tdict['logm'],tdict['logsfr'] = stellar_mass[in_bin],logsfr[in_bin]

            # calculate sSFR chains (fn / sum(tn*fn))
            # first transform to SFR fraction
            # then add the nth bin
            # finally, ensure it's normalized (no floating point errors)
            # this is where we'd use the individual agebins if we so chose!
            outfrac, outweight = [], []
            for zf,weight,agebins in zip(np.array(data['z_fraction'])[zidx][in_bin], 
                                         np.array(data['weights'])[zidx][in_bin],
                                         np.array(data['agebins'])[zidx][in_bin]):
                frac = prosp_dutils.transform_zfraction_to_sfrfraction(zf)
                frac = np.concatenate((frac, (1-frac.sum(axis=1))[:,None]),axis=1)
                time_per_bin = np.diff(10**agebins, axis=-1)[:,0]
                norm = (frac * time_per_bin).sum(axis=1) # sum(fn*tn)
                outfrac += [frac/norm[:,None]]
                outweight += [weight]

            # calculate weighted mean for each SFR bin
            for i in range(time_per_bin.shape[0]):
                frac = [f[:,i] for f in outfrac]
                mean,errup,errdown = calculate_mean(frac, outweight)
                tdict['mean'] += [mean]
                tdict['errup'] += [errup]
                tdict['errdown'] += [errdown]

            tdict['err'] = prosp_dutils.asym_errors(np.array(tdict['mean']),
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
    ms_plot_opts = {
                    'alpha':0.5,
                    'mew':0.2,
                    'mec': 'k',
                    'linestyle':' ',
                    'marker':'o',
                    'ms':1.5
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
                      'alpha':0.7,
                      'fmt':'o',
                      'mew':0.8,
                      'linestyle':'-',
                      'ms':8.5,
                      'lw':1,
                      'markeredgecolor': 'k',
                      'elinewidth':1.75,
                      'capsize':4,
                      'capthick':4
                      }

    # horizontal stack figure
    fig, ax = plt.subplots(4,2, figsize=(8,15))
    for j, (z1,z2) in enumerate(opts['zbins']):
        
        # calculate zavg for bin placement
        zavg = (z1+z2)/2.
        zstr = "{:.2f}".format(zavg)
        plot_main_sequence(ax[j,0],zavg,**opts)

        for i in range(opts['nbins_horizontal']):
            
            # grab bin dictionary
            bdict = dat['hor'][zstr]['bin'+str(i)]

            # plot star-forming sequence
            ax[j,0].plot(bdict['logm'],bdict['logsfr'],
                         color=opts['horizontal_bin_colors'][i],
                         **ms_plot_opts)

            # plot SFH stacks
            log_mean_t = np.mean(np.log10(dat['hor'][zstr]['agebins']),axis=1)
            log_mean_t = np.clip(log_mean_t,7.5,np.inf) # nothing lower than ~30 Myr
            ax[j,1].errorbar(10**(log_mean_t-x_stack_offset*(i-1)),bdict['mean'],
                             yerr=bdict['err'],
                             color=opts['horizontal_bin_colors'][i],
                             **stack_plot_opts)

        # labels and ranges
        ax[j,0].set_xlabel(r'log(M$_{*}$)',fontsize=fontsize)
        ax[j,0].set_ylabel(r'log(SFR)',fontsize=fontsize)
        ax[j,0].set_xlim(dat['hor'][zstr]['mass_bins'].min()-0.5,dat['hor'][zstr]['mass_bins'].max()+0.5)
        ax[j,0].tick_params('both', pad=3.5, size=3.5, width=1.0, which='both',labelsize=fontsize)

        ax[j,1].set_xlim(10**(log_mean_t.min()-0.3),10**(log_mean_t.max()+0.3))
        ax[j,1].set_xlabel(r'time [yr]',fontsize=fontsize)
        ax[j,1].set_ylabel(r'SFR$_{\mathrm{bin}}$ / M$_{\mathrm{total}}$ [yr$^{-1}$]',fontsize=fontsize)
        ax[j,1].text(0.98, 0.92, 'z='+zstr,ha='right',transform=ax[j,1].transAxes,fontsize=fontsize)

        ax[j,1].set_xscale('log',nonposx='clip')
        ax[j,1].set_yscale('log',nonposy='clip')
        ax[j,1].tick_params('both', pad=3.5, size=3.5, width=1.0, which='both',labelsize=fontsize)

        # plot mass ranges 
        # must happen after ylim is determined
        ylim = ax[j,0].get_ylim()
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

            ax[j,0].plot(np.repeat(dat['hor'][zstr]['mass_bins'][i]+dlowbin,2),ylim,
                       color=opts['horizontal_bin_colors'][i],
                       **ms_line_plot_opts)
            ax[j,0].plot(np.repeat(dat['hor'][zstr]['mass_bins'][i+1]+dhighbin,2),ylim,
                       color=opts['horizontal_bin_colors'][i],
                       **ms_line_plot_opts)
            ax[j,0].set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(outname_ms_horizontal,dpi=150)
    plt.close()


    # vertical stack figure
    fig, ax = plt.subplots(4,2, figsize=(8,15))

    # plot main sequence position and stacks
    for j, (z1,z2) in enumerate(opts['zbins']):

        # calculate zavg for bin placement
        zavg = (z1+z2)/2.
        zstr = "{:.2f}".format(zavg)

        for i in range(opts['nbins_vertical']):

            # plot star-forming sequence
            bdict = dat['vert'][zstr]['bin'+str(i)]
            ax[j,0].plot(bdict['logm'],bdict['logsfr'],
                       color=opts['vertical_bin_colors'][i],
                       **ms_plot_opts)

            # plot SFH stacks
            log_mean_t = np.mean(np.log10(dat['vert'][zstr]['agebins']),axis=1)
            log_mean_t = np.clip(log_mean_t,7.5,np.inf) # nothing lower than ~30 Myr
            ax[j,1].errorbar(10**(log_mean_t-x_stack_offset*(i-1)),bdict['mean'],
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
        ax[j,0].set_xlabel(r'log(M$_{*}$)',fontsize=fontsize)
        ax[j,0].set_ylabel(r'log(SFR)',fontsize=fontsize)
        xlim = (minmass-0.6,maxmass+0.3)
        #ylim = (minsfr-0.3,maxsfr+0.5)
        ax[j,0].set_xlim(xlim)
        ax[j,0].tick_params('both', pad=3.5, size=3.5, width=1.0, which='both',labelsize=fontsize)

        ax[j,1].set_xlim(10**(log_mean_t.min()-0.3),10**(log_mean_t.max()+0.3))
        ax[j,1].set_xlabel(r'time [yr]',fontsize=fontsize)
        ax[j,1].set_ylabel(r'SFR$_{\mathrm{bin}}$ / M$_{\mathrm{tot}}$ [yr$^{-1}$]',fontsize=fontsize)
        ax[j,1].set_xscale('log',nonposx='clip')
        ax[j,1].set_yscale('log',nonposy='clip')
        ax[j,1].tick_params('both', pad=3.5, size=3.5, width=1.0, which='both',labelsize=fontsize)
        ax[j,1].text(0.98, 0.92, 'z='+zstr,ha='right',transform=ax[j,1].transAxes,fontsize=fontsize)

        # plot mass ranges + labels
        labels = ['quiescent','<2$\sigma$ MS','on MS','>2$\sigma$ MS']
        xr = np.linspace(xlim[0],xlim[1],50)
        yr = sfr_ms(zavg,xr,**opts)
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
            ytext = yr[idx]+(sigup+sigdown)/2.+0.25
            if i == 0:
                ytext = yr[idx]+sigup+0.1
            else:
                ax[j,0].plot(xr,yr+sigdown+dlowbin,
                             color=opts['vertical_bin_colors'][i],
                             **ms_line_plot_opts)

            # no top line for the top one
            if i != opts['nbins_vertical']-1:
                ax[j,0].plot(xr,yr+sigup+dhighbin,
                           color=opts['vertical_bin_colors'][i],
                           **ms_line_plot_opts)

            ax[j,0].text(xr[idx],ytext,labels[i],color=opts['vertical_bin_colors'][i], rotation=24)

    plt.tight_layout()
    plt.savefig(outname_ms_vertical,dpi=150)
    plt.close()

def plot_main_sequence(ax,z,sigma_sf=None,low_mass_cutoff=7, **junk):

    mass = np.linspace(low_mass_cutoff,12,40)
    sfr = sfr_ms(z,mass,**junk)

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
    idx = np.where(zwhit == z[0])
    idx = idx[0][0]
    if idx == -1:
        print 'this is really poorly implemented, fix it'
        print 1/0

    # generate SFR(M) at all redshifts 
    log_sfr = alow[idx]*(logm - 10.2) + b[idx]
    high = (logm > 10.2).squeeze()
    log_sfr[high] = ahigh[idx]*(logm[high] - 10.2) + b[idx]
    
    if adjust_sfr:
        log_sfr += adjust_sfr

    return log_sfr

def calculate_mean(pdf, weight):
    """given list of N PDFs, calculate mean(PDF) for each item
    then return the MEDIAN of the mean, and the +/- 1 sigma errors
    """
    nobj = len(pdf)
    mean_array = []
    for i,(f,w) in enumerate(zip(pdf,weight)): mean_array += [np.average(f, weights=w)]
    mean, errup, errdown = np.percentile(mean_array, [50,84,16])
    return mean, errup, errdown





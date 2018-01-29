import numpy as np
import matplotlib.pyplot as plt
import os, hickle, td_io, copy
from prospector_io import load_prospector_data
from prosp_dutils import generate_basenames, asym_errors
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from dynesty.plotting import _quantile as weighted_quantile
from stack_td_sfh import sfr_ms
from astropy.io import ascii

plt.ioff()

popts = {'fmt':'o', 'capthick':1.5,'elinewidth':1.5,'ms':9,'alpha':0.8,'color':'0.3'}
red = '#FF3D0D'
dpi = 160
cmap = 'cool'
minsfr = 0.01
filename =  os.getenv('APPS') + '/prospector_alpha/plots/td_huge/fast_plots/data/agn.h5'

opts = {
          'sigma_ms':0.3,                  # scatter in the star-forming sequence, in dex
          'xlim': (9, 11.5),               # x-limit
          'ylim': (0.02,0.53),              # y-limit
          'fmir_grid': 10**np.linspace(np.log10(1e-4),np.log10(1),500),
          'nmassbins': 5,                  # number of mass bins
          'xshift': 0.05,                  # x-shift between SFR bins
          'sfr_colors': ['#45ADA8','#FC913A','#FF4E50'],
          'sfr_labels': ['above MS', 'on MS', 'below MS'],
          'adjust_sfr': -0.2,              # adjust whitaker SFRs by how much in dex?
          'zbins': [(0.5,1.),(1.,1.5),(1.5,2.),(2.,2.5)],
          'use_fagn': False,               # otherwise we use fmir
          'tenth_percentile': False,        # use f_X > 0.1 as the criteria
          'one_sigma': False               # use f_X-sigma > 0.1 as the criteria
         }
opts['massbins'] = np.linspace(opts['xlim'][0],opts['xlim'][1],opts['nmassbins']+1)
opts['zbin_labels'] = ["{0:.1f}".format(z1)+'<z<'+"{0:.1f}".format(z2) for (z1, z2) in opts['zbins']]

def do_all(runname='td_huge', outfolder=None, data=None, stack=None, regenerate=False, **opts):

    if outfolder is None:
        outfolder = os.getenv('APPS') + '/prospector_alpha/plots/'+runname+'/fast_plots/'
        if not os.path.isdir(outfolder):
            os.makedirs(outfolder)
            os.makedirs(outfolder+'data/')

    if data is None:
        data = collate_data(runname,filename=outfolder+'data/agn.h5',regenerate=regenerate,**opts)
    if stack is None:
        stack = stack_agn_bins(data, **opts)

    outname = 'agn_strength'
    if opts['tenth_percentile']:
        outname = 'agn_strength_10thpercentile'
        if opts['one_sigma']:
            outname += '_1sig'

    agn_plots(stack, outfolder+outname+'.png', opts)

def collate_data(runname, filename=filename, regenerate=False, **opts):
    """ pull out all of the necessary information from the individual data files
    this takes awhile, so this data is saved to disk.
    """

    '''
    lines to fix data
    with the current single-list issue
    fmirtemp = [data['fmir'][i*3000:(1+i)*3000] for i in range(54331)]
    data['fmir'] = fmirtemp
    '''

    # if it's already made, load it and give it back
    # else, start with the making!
    if os.path.isfile(filename) and regenerate == False:
        with open(filename, "r") as f:
            outdict=hickle.load(f)

        return outdict

    # define output containers
    outvar = ['stellar_mass','sfr_100', 'fagn', 'fmir']
    outdict = {q: {f: [] for f in ['q50','q84','q16']} for q in outvar}
    for f in ['objname', 'weights', 'fmir_chain', 'fagn_chain', 'zred', 'mips_sn']: outdict[f] = [] 

    # we want MASS, SFR_100, F_AGN, F_MIR CHAIN, for each galaxy
    basenames, _, _ = generate_basenames(runname)
    for i, name in enumerate(basenames):

        # load. do we keep it? check redshift
        objname = name.split('/')[-1]
        datdir = os.getenv('APPS')+'/prospector_alpha/data/3dhst/'
        datname = datdir + objname.split('_')[0] + '_' + runname + '.dat'
        dat = ascii.read(datname)
        idx = dat['phot_id'] == int(objname.split('_')[-1])
        zred = float(dat['z_best'][idx])
        if (zred < 0.5) or (zred > 2.5):
            print 'zred={0} for '.format(zred)+objname+', skipping'
            continue

        # load output from fit
        try:
            res, _, model, prosp = load_prospector_data(name)
        except:
            print name.split('/')[-1]+' failed to load. skipping.'
            continue
        if (res is None) or (prosp is None):
            continue

        outdict['zred'] += [zred]
        outdict['objname'] += [objname]
        print 'loaded ' + objname

        # load up chains
        fidx = model.theta_index['fagn']
        outdict['fagn_chain'] += [res['chain'][prosp['sample_idx'], fidx]]
        outdict['fmir_chain'] += [prosp['extras']['fmir']['chain']]
        outdict['weights'] += [prosp['weights'].tolist()]

        # extra variables
        for v in outvar:
            for f in ['q50','q84','q16']: 
                if v == 'fagn':
                    outdict[v][f] += [prosp['thetas'][v][f]]
                else:
                    outdict[v][f] += [prosp['extras'][v][f]]
        
        # mips
        midx = np.array(['mips' in f.name for f in res['obs']['filters']],dtype=bool)
        if (midx.sum() == 0) | (res['obs']['phot_mask'][midx] == False):
            print 'no MIPS data!'
            outdict['mips_sn'] += [0]
        else:
            outdict['mips_sn'] += (res['obs']['maggies'][midx] / res['obs']['maggies_unc'][midx]).tolist()

    # dump files and return
    hickle.dump(outdict,open(filename, "w"))
    return outdict

def stack_agn_bins(data,**opts):
    """this is where the stacking occurs
    we take a different tack than with SFHs: we want to include measurement error here
    this is because measurement error is HUGE (~10x not unusual)
    
    here we do two things:
        (1) measure the median fmir in mass bins.
            -- assume stellar mass errors are zero for this
        (2) simulate measurement errors from a perfect distribution: what is the width of this distribution? pair two random points from the PDF, 
            1000x per PDF. take all as independent measurements. overplot as measurement error.
    """

    # generate output containers
    fmt =      {label:[] for label in ['q50_stack','q84_stack','q16_stack','median_logwidth','median','scatter']}
    sfr_dict = {label:copy.deepcopy(fmt) for label in opts['sfr_labels']}
    out =      {label:copy.deepcopy(sfr_dict) for label in opts['zbin_labels']}

    parstring = 'fmir'
    if opts['use_fagn']:
        parstring = 'fagn'

    # loop over redshift bins
    for i, zlabel in enumerate(opts['zbin_labels']):

        # what galaxies are in our redshift bins?
        z1, z2 = opts['zbins'][i][0], opts['zbins'][i][1]
        idx = (np.array(data['zred']) > z1) & (np.array(data['zred']) <= z2)

        # pull out critical quantities
        sm = np.log10(data['stellar_mass']['q50'])[idx]
        sfr = np.log10(data['sfr_100']['q50'])[idx]
        fmir = np.array(data[parstring]['q50'])[idx]
        fmir_up = np.array(data[parstring]['q84'])[idx]
        fmir_down = np.array(data[parstring]['q16'])[idx]
        pdfs = np.array(data[parstring+'_chain'])[idx]
        weights = np.array(data['weights'])[idx]
        logwidths = (np.log10(data[parstring]['q84'])[idx] - np.log10(data[parstring]['q16'])[idx])/2.

        # loop over SFR bins
        # use Whitaker+14 SFR adjusted downwards for now (not that bad?)
        # assume sigma = 0.3, take within 1 sigma of MS
        sfr_expected = sfr_ms((z1 + z2)/2.,sm,adjust_sfr=opts['adjust_sfr'])
        for j, sfrlabel in enumerate(opts['sfr_labels']):

            # choose indexes
            if j == 0:
                idx_bin = ((sfr_expected+opts['sigma_ms']) < sfr)
            if j == 1:
                idx_bin = ((sfr_expected-opts['sigma_ms']) < sfr) & ((sfr_expected+opts['sigma_ms']) > sfr)
            if j == 2:
                idx_bin = ((sfr_expected-opts['sigma_ms']) > sfr)

            # pick out fmir chains + weights
            fmir_pdfs = pdfs[idx_bin].squeeze()
            fmir_weights = weights[idx_bin]/weights[idx_bin].sum(axis=1)[:,None]
            fmir_logwidths = logwidths[idx_bin]
            fmir_values = fmir[idx_bin]

            # finally, loop over massbins
            for k in range(len(opts['massbins'])-1):

                idx = (sm[idx_bin] > opts['massbins'][k]) & (sm[idx_bin] <= opts['massbins'][k+1])
                n_in_bin = idx.sum()

                # if we want to go with the tenth percentile criteria...
                if opts['tenth_percentile']:

                    if opts['one_sigma']:
                        out[zlabel][sfrlabel]['q50_stack'] += [((fmir_down[idx_bin])[idx] > 0.1).sum() / float(idx.sum())]
                    else:
                        fmir_perc = []
                        for n in range(100):
                            
                            # draw randomly
                            draw = np.random.uniform(size=n_in_bin)
                            cumsum = np.cumsum(fmir_weights[idx],axis=1)
                            rand_idx = np.abs(cumsum - draw[:,None]).argmin(axis=1)
                            chain = (fmir_pdfs[idx])[np.arange(n_in_bin),rand_idx]

                            fmir_perc += [(chain > 0.1).sum() / float(n_in_bin)]

                            print n
                        mid, up, down = np.percentile(fmir_perc,[50,84,16])
                        out[zlabel][sfrlabel]['q50_stack'] += [mid]
                        out[zlabel][sfrlabel]['q84_stack'] += [up]
                        out[zlabel][sfrlabel]['q16_stack'] += [down]
                else:

                    # estimate mean + scatter
                    out[zlabel][sfrlabel]['q50_stack'] += [np.median(fmir_values[idx])]
                    out[zlabel][sfrlabel]['q84_stack'] += [np.percentile(fmir_values[idx],[84])[0]]
                    out[zlabel][sfrlabel]['q16_stack'] += [np.percentile(fmir_values[idx],[16])[0]]

                # now estimate typical measurement error
                out[zlabel][sfrlabel]['median_logwidth'] += [np.median(fmir_logwidths[idx])]

    return out

def agn_plots(plot,outname,opts):
    """this is where the plotting occurs
    """

    # plot information
    fig, axes = plt.subplots(2, 2, figsize = (5,5))
    axes = np.ravel(axes)
    mbins = (opts['massbins'][1:] + opts['massbins'][:-1])/2.

    # labels (yeah sorry about this logic, whoever happens to read this)
    # (i assure you it does not have to be this complicated)
    if opts['tenth_percentile']:
        ylabel = 'fraction with f$_{\mathrm{AGN,MIR}}$'
        if opts['use_fagn']:
            ylabel = 'fraction with f$_{\mathrm{AGN}}$'
        ylabel += ' > 0.1'
        if opts['one_sigma']:
            ylabel += '\n at 1$\sigma$ confidence'
    else:
        ylabel = 'f$_{\mathrm{AGN,MIR}}$'
        if opts['use_fagn']:
            ylabel = 'f$_{\mathrm{AGN}}$'
        ylabel = 'median '+ylabel

    for i, zlabel in enumerate(opts['zbin_labels']):
        print zlabel
        for j, sfrlabel in enumerate(opts['sfr_labels']):
            print sfrlabel

            ymeasure = np.array(plot[zlabel][sfrlabel]['q50_stack'])
            if opts['one_sigma']:
                yerror = None
            else:
                yerror = asym_errors(ymeasure, np.array(plot[zlabel][sfrlabel]['q84_stack']), np.array(plot[zlabel][sfrlabel]['q16_stack']))
            axes[i].errorbar(mbins+(j-1)*opts['xshift'],ymeasure,color=opts['sfr_colors'][j],yerr=yerror,
                            label=opts['sfr_labels'][j],fmt='o',ms=3,linestyle='-')

            if yerror is not None:            
                distr_scatter = (np.log10(plot[zlabel][sfrlabel]['q84_stack']) - np.log10(plot[zlabel][sfrlabel]['q16_stack']))/2.
                typical_err = np.array(plot[zlabel][sfrlabel]['median_logwidth'])
                error_ratio = typical_err/distr_scatter
                for k, m in enumerate(mbins): print '\t M=' + "{0:.2f}".format(m) + '  ' + "{0:.2f}".format(error_ratio[k])

        # scale
        #axes[i].set_yscale('log',nonposy='clip',subsy=([3]))
        axes[i].yaxis.set_minor_formatter(FormatStrFormatter('%2.4g'))
        axes[i].yaxis.set_major_formatter(FormatStrFormatter('%2.4g'))

        # labels
        if i > 1:
            axes[i].set_xlabel('log(M/M$_{\odot}$)')
        else:
            for tl in axes[i].get_xticklabels():tl.set_visible(False)
            plt.setp(axes[i].get_xminorticklabels(), visible=False)
        if (i == 2) or (i == 0):
            axes[i].set_ylabel(ylabel)
        else:
            for tl in axes[i].get_yticklabels():tl.set_visible(False)
            plt.setp(axes[i].get_yminorticklabels(), visible=False)

        # text
        axes[i].text(0.96,0.92,zlabel,transform=axes[i].transAxes,ha='right')

    for a in axes:
        a.set_xlim(opts['xlim'])
        a.set_ylim(opts['ylim'])

    axes[0].legend(loc=2, prop={'size':8},
                   scatterpoints=1,fancybox=True)
    plt.tight_layout(h_pad=-0.1,w_pad=-0.1)
    fig.subplots_adjust(wspace=0.0,hspace=0.0)
    plt.savefig(outname,dpi=dpi)
    plt.close()


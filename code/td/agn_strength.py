import numpy as np
import matplotlib.pyplot as plt
import os, hickle, td_io, copy
from prospector_io import load_prospector_data
from prosp_dutils import generate_basenames, asym_errors, get_cmap
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from dynesty.plotting import _quantile as weighted_quantile
from astropy.io import ascii

plt.ioff()

popts = {'fmt':'o', 'capthick':1.5,'elinewidth':1.5,'ms':9,'alpha':0.8,'color':'0.3'}
red = '#FF3D0D'
dpi = 210
cmap = 'cool'
minsfr = 0.01
filename =  os.getenv('APPS') + '/prospector_alpha/plots/td_huge/fast_plots/data/agn.h5'

opts = {
          'xlim': (9, 11.5),               # x-limit
          'ylim': (0.00,0.53),              # y-limit
          'fmir_grid': 10**np.linspace(np.log10(1e-4),np.log10(1),500),
          'xshift': 0.02,                  # x-shift between mass bins
          'nmassbins': 8,                  # number of mass bins
          'zbins': [(0.5,1.),(1.,1.5),(1.5,2.),(2.,2.5)],
          'colors': ['#0202d6','#31A9B8','#FF9100','#FF420E'],
          'color': '0.2',
          'use_fagn': False,               # otherwise we use fmir
          'tenth_percentile': True,        # use f_X > 0.1 as the criteria
          'one_sigma': True               # use f_X-sigma > 0.1 as the criteria
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
    out =      {label:copy.deepcopy(fmt) for label in opts['zbin_labels']}

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
        fmir = np.array(data[parstring]['q50'])[idx]
        fmir_up = np.array(data[parstring]['q84'])[idx]
        fmir_down = np.array(data[parstring]['q16'])[idx]
        pdfs = np.array(data[parstring+'_chain'])[idx]
        weights = np.array(data['weights'])[idx]
        logwidths = (np.log10(data[parstring]['q84'])[idx] - np.log10(data[parstring]['q16'])[idx])/2.

        # loop over massbins
        for k in range(len(opts['massbins'])-1):

            idx = (sm > opts['massbins'][k]) & (sm <= opts['massbins'][k+1])
            n_in_bin = idx.sum()

            # if we want to go with the tenth percentile criteria...
            if opts['tenth_percentile']:

                if opts['one_sigma']:

                    # bootstrap errors
                    ngal, ndraw = idx.sum(), 1000
                    idx_draw = np.random.randint(ngal, size=(ngal,ndraw))
                    onesig = ((fmir_down[idx])[idx_draw] > 0.1).sum(axis=0) / float(ngal)
                    mid, up, down = np.percentile(onesig,[50,84,16])
                    out[zlabel]['q50_stack'] += [mid]
                    out[zlabel]['q84_stack'] += [up]
                    out[zlabel]['q16_stack'] += [down]
                else:
                    fmir_perc = []
                    for n in range(100):
                        
                        # draw randomly
                        draw = np.random.uniform(size=n_in_bin)
                        cumsum = np.cumsum(weights[idx],axis=1)
                        rand_idx = np.abs(cumsum - draw[:,None]).argmin(axis=1)
                        chain = (pdfs[idx])[np.arange(n_in_bin),rand_idx]

                        fmir_perc += [(chain > 0.1).sum() / float(n_in_bin)]

                        print n
                    mid, up, down = np.percentile(fmir_perc,[50,84,16])
                    out[zlabel]['q50_stack'] += [mid]
                    out[zlabel]['q84_stack'] += [up]
                    out[zlabel]['q16_stack'] += [down]
            else:

                # estimate mean + scatter
                out[zlabel]['q50_stack'] += [np.median(fmir[idx])]
                out[zlabel]['q84_stack'] += [np.percentile(fmir[idx],[84])[0]]
                out[zlabel]['q16_stack'] += [np.percentile(fmir[idx],[16])[0]]

            # now estimate typical measurement error
            out[zlabel]['median_logwidth'] += [np.median(logwidths[idx])]

    return out

def agn_plots(plot,outname,opts):
    """this is where the plotting occurs
    """

    # plot information
    fig, ax = plt.subplots(1, 1, figsize = (4,3.5))
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
        ymeasure = np.array(plot[zlabel]['q50_stack'])
        #if opts['one_sigma']:
        #    yerror = None
        #else:
        yerror = asym_errors(ymeasure, np.array(plot[zlabel]['q84_stack']), np.array(plot[zlabel]['q16_stack']))
        ax.errorbar(mbins+(i-1)*opts['xshift'],ymeasure,color=opts['colors'][i],yerr=yerror,
                        label=zlabel,fmt='o',ms=5,linestyle='-',lw=1.5,alpha=0.9,elinewidth=1.5)

        if yerror is not None:            
            distr_scatter = (np.log10(plot[zlabel]['q84_stack']) - np.log10(plot[zlabel]['q16_stack']))/2.
            typical_err = np.array(plot[zlabel]['median_logwidth'])
            error_ratio = typical_err/distr_scatter
            for k, m in enumerate(mbins): print '\t M=' + "{0:.2f}".format(m) + '  ' + "{0:.2f}".format(error_ratio[k])

    # scale
    #ax.set_yscale('log',nonposy='clip',subsy=([3]))
    ax.yaxis.set_minor_formatter(FormatStrFormatter('%2.4g'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%2.4g'))

    # labels
    ax.set_xlabel('log(M/M$_{\odot}$)')
    ax.set_ylabel(ylabel)

    ax.set_xlim(opts['xlim'])
    ax.set_ylim(opts['ylim'])

    ax.legend(loc=2, prop={'size':8},
                   scatterpoints=1,fancybox=True)
    plt.tight_layout()
    plt.savefig(outname,dpi=dpi)
    plt.close()


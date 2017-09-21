import numpy as np
import matplotlib.pyplot as plt
import prosp_dutils
import magphys_plot_pref
from matplotlib.ticker import MaxNLocator
import copy, os, pickle
import matplotlib as mpl
from prospector_io import load_prospector_extra

mpl.rcParams.update({'font.size': 18})
mpl.rcParams.update({'font.weight': 500})
mpl.rcParams.update({'axes.labelweight': 500})

def load_alldata(runname,filename,regenerate=False):

    if regenerate or not os.path.isfile(filename):
        basenames, _, _ = prosp_dutils.generate_basenames(runname)
        alldata = []
        for i, name in enumerate(basenames):

            # prospector first
            try:
                prosp = load_prospector_extra(name)
            except (IOError, TypeError):
                print name.split('/')[-1]+' failed to load. skipping.'
                continue
            dat = {'thetas':prosp['thetas'],'extras':prosp['extras']}
            alldata.append(dat)
        pickle.dump(alldata,open(filename, "w"))
    else:
        with open(filename, "r") as f:
            alldata=pickle.load(f)
    return alldata

def arrange_data(alldata):

    ### normal parameter labels
    parnames_all = alldata[0]['thetas'].keys()
    parnames = [par for par in parnames_all if 'z_fraction' not in par]

    ### extra parameters
    eparnames_all = alldata[0]['extras'].keys()
    eparnames = ['sfr_100', 'ssfr_100', 'half_time']

    parlabels = {
                 'massmet_1':r'log(M/M$_{\odot}$)', 
                 'dust2': r'diffuse dust', 
                 'massmet_2': r'log(Z/Z$_{\odot}$)', 
                 'dust_index': r'diffuse dust index',
                 'dust1_fraction': r'(diffuse/young) dust', 
                 'duste_qpah': r'Q$_{\mathrm{PAH}}$',
                 'fagn': r'log(f$_{\mathrm{AGN}}$)',
                 'agn_tau': r'$\tau_{\mathrm{AGN}}$',
                 'sfr_100': r'log(SFR) [100 Myr]',
                 'ssfr_100': r'log(sSFR) [100 Myr]',
                 'half_time': r"log(t$_{\mathrm{half-mass}})$ [Gyr]"
                 }

    ### setup dictionary
    outvals, outq, outerrs, outlabels = {},{},{},{}
    for ii,par in enumerate(parnames): 
        outvals[par] = []
        outq[par] = {}
        outq[par]['q50'],outq[par]['q84'],outq[par]['q16'] = [],[],[]
    for ii,par in enumerate(eparnames):
        outvals[par] = []
        outq[par] = {}
        outq[par]['q50'],outq[par]['q84'],outq[par]['q16'] = [],[],[]

    ### fill with data
    for dat in alldata:
        for par in parnames:
            if par == 'fagn':
                for q in outq[par].keys(): outq[par][q].append(np.log10(dat['thetas'][par][q]))
                outvals[par].append(np.log10(dat['thetas'][par]['q50']))
            else:
                for q in outq[par].keys(): outq[par][q].append(dat['thetas'][par][q])
                outvals[par].append(dat['thetas'][par]['q50'])
        for par in eparnames:
            for q in outq[par].keys(): outq[par][q].append(np.log10(dat['extras'][par][q]))
            outvals[par].append(np.log10(dat['extras'][par]['q50']))

    ### do the errors
    for par in parlabels.keys():
        outerrs[par] = prosp_dutils.asym_errors(np.array(outq[par]['q50']), 
                                                np.array(outq[par]['q84']),
                                                np.array(outq[par]['q16']),log=False)
        outvals[par] = np.array(outvals[par])

    ### fill output
    out = {}
    out['median'] = outvals
    out['errs'] = outerrs
    out['parlabels'] = parlabels
    out['ordered_labels'] = np.array(parnames + eparnames)

    return out
    
def allpar_plot(runname='td_massive',outfolder=None,lowmet=True,regenerate=False):

    ### I/O stuff
    if outfolder is None:
        outfolder = os.getenv('APPS') + '/prospector_alpha/plots/'+runname+'/fast_plots/'
    if not os.path.isdir(outfolder):
        os.makedirs(outfolder)
        os.makedirs(outfolder+'data/')
    filename=outfolder+'data/allpar_plot.h5'

    ### load data
    alldata = load_alldata(runname,filename,regenerate=regenerate)
    dat = arrange_data(alldata)
    npars = len(dat['parlabels'].keys())

    ### plot preferences
    fig, ax = plt.subplots(ncols=npars, nrows=npars, figsize=(npars*3,npars*3))
    fig.subplots_adjust(wspace=0.0,hspace=0.0,top=0.95,bottom=0.05,left=0.05,right=0.95)
    opts = {
            'color': '#1C86EE',
            'mew': 1.5,
            'alpha': 0.6,
            'fmt': 'o'
           }
    hopts = copy.deepcopy(opts)
    hopts['color'] = '#FF420E'

    ### color low-metallicity, high-mass galaxies
    met_q50 = np.array([data['thetas']['massmet_2']['q50'] for data in alldata])
    mass_q50 = np.array([data['thetas']['massmet_1']['q50'] for data in alldata])
    hflag = (met_q50 < -1.0) & (mass_q50 > 9.5)
    outname = outfolder+'all_parameter.png'

    # plot
    for yy, ypar in enumerate(dat['ordered_labels']):
        for xx, xpar in enumerate(dat['ordered_labels']):

            # turn off the dumb ones
            if xx >= yy:
                ax[yy,xx].axis('off')
                continue

            ax[yy,xx].errorbar(dat['median'][xpar][~hflag],dat['median'][ypar][~hflag],
                               xerr=[dat['errs'][xpar][0][~hflag],dat['errs'][xpar][1][~hflag]], 
                               yerr=[dat['errs'][ypar][0][~hflag],dat['errs'][ypar][1][~hflag]], 
                               **opts)

            ax[yy,xx].errorbar(dat['median'][xpar][hflag],dat['median'][ypar][hflag],
                               xerr=[dat['errs'][xpar][0][hflag],dat['errs'][xpar][1][hflag]], 
                               yerr=[dat['errs'][ypar][0][hflag],dat['errs'][ypar][1][hflag]], 
                               **hopts)

            #### RANGE
            minx,maxx = dat['median'][xpar].min(),dat['median'][xpar].max()
            dynx = (maxx-minx)*0.1
            ax[yy,xx].set_xlim(minx-dynx, maxx+dynx)

            miny,maxy = dat['median'][ypar].min(),dat['median'][ypar].max()
            dyny = (maxy-miny)*0.1
            ax[yy,xx].set_ylim(miny-dyny, maxy+dyny)

            #### LABELS
            if xx % npars == 0:
                ax[yy,xx].set_ylabel(dat['parlabels'][ypar])
            else:
                for tl in ax[yy,xx].get_yticklabels():tl.set_visible(False)

            if yy == npars-1:
                ax[yy,xx].set_xlabel(dat['parlabels'][xpar])
            else:
                for tl in ax[yy,xx].get_xticklabels():tl.set_visible(False)

            ax[yy,xx].xaxis.set_major_locator(MaxNLocator(5))
            ax[yy,xx].yaxis.set_major_locator(MaxNLocator(5))

            plt.setp(ax[yy,xx].xaxis.get_majorticklabels(), rotation=-55, horizontalalignment='center')

    plt.savefig(outname,dpi=100)
    plt.close()

    mpl.rcParams.update({'font.weight': 400})
    mpl.rcParams.update({'axes.labelweight': 400})






















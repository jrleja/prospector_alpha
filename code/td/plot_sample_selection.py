import numpy as np
import matplotlib.pyplot as plt
from select_td_sample import load_master_sample
import os, hickle

def load_master(filename=None,regenerate=False,sids=None):
    """ we try to load it first because it's big
    """
    if os.path.isfile(filename) and regenerate == False:
        with open(filename, "r") as f:
            outdict = hickle.load(f)
    else:
        outdict = load_master_sample()
        outdict['sidx'] = np.array([outdict['id'].index(name) for name in sids])
        hickle.dump(outdict,open(filename, "w"))

    return outdict

def do_all(runname='td_huge',outfolder=None,regenerate=False,**opts):
    """compare sample selection to parent 3D-HST sample
    this is defined as (parent = phot_flag == 1)
    """

    # folder maintenance
    if outfolder is None:
        outfolder = os.getenv('APPS') + '/prospector_alpha/plots/'+runname+'/fast_plots/'
        if not os.path.isdir(outfolder):
            os.makedirs(outfolder)
            os.makedirs(outfolder+'data/')

    # load master sample + sample IDs
    sids = np.genfromtxt('/Users/joel/code/python/prospector_alpha/data/3dhst/'+runname+'.ids',
                         dtype=[('objnames', '|S40')])['objnames'].tolist()
    data = load_master(filename=outfolder+'data/master.hickle',regenerate=regenerate,sids=sids)
    for key in data: data[key] = np.array(data[key])
    # plot
    plot(data, outfolder=outfolder)

def plot(data, outfolder=None):

    ### physics choices
    zbins = [(0.5,1.),(1.,1.5),(1.5,2.),(2.,2.5),(2.5,3.0)]
    mass_options = {
                    'data': data['fast_logmass'],
                    'min': 8,
                    'max': 12,
                    'ylabel': r'N*M$_{\mathrm{stellar}}$',
                    'xlabel': r'log(M$_*$/M$_{\odot}$)',
                    'norm': 1e13,
                    'name': 'rhomass_selection.png'
                   }

    sfr_options = {
                    'data': np.log10(np.clip(data['uvir_sfr'],0.001,np.inf)),
                    'min': -3,
                    'max': 5,
                    'ylabel': r'N*SFR',
                    'xlabel': r'log(SFR) [M$_{\odot}$/yr]',
                    'norm': 1e4,
                    'name': 'rhosfr_selection.png'
                   }

    ### plot choices
    nbins = 20
    histopts = {'drawstyle':'steps-mid','alpha':1.0, 'lw':1, 'linestyle': '-'}
    fontopts = {'fontsize':10}

    # plot mass + mass density distribution
    for opt in [mass_options,sfr_options]:
        fig, ax = plt.subplots(5,2,figsize=(5,9))
        for i,zbin in enumerate(zbins):

            # masses & indexes
            idx = (data['zbest'] > zbin[0]) & \
                  (data['zbest'] < zbin[1]) & \
                  (opt['data'] > opt['min']) & \
                  (opt['data'] < opt['max'])
            sample_idx = [data['sidx']]
            sidx = (data['zbest'][sample_idx] > zbin[0]) & \
                   (data['zbest'][sample_idx] < zbin[1]) & \
                   (opt['data'][sample_idx] > opt['min']) & \
                   (opt['data'][sample_idx] < opt['max'])
            master_dat = opt['data'][idx]
            sample_dat = opt['data'][sample_idx][sidx]

            # mass histograms. get bins from master histogram
            hist_master, bins = np.histogram(master_dat,bins=nbins,density=False)
            hist_sample, bins = np.histogram(sample_dat,bins=bins,density=False)
            bins_mid = (bins[1:]+bins[:-1])/2.

            # mass distribution
            ax[i,0].plot(bins_mid,hist_master,color='0.4', **histopts)
            ax[i,0].plot(bins_mid,hist_sample,color='red', **histopts)

            # mass density distribution
            ax[i,1].plot(bins_mid,hist_master*10**bins_mid/opt['norm'],color='0.4', **histopts)
            ax[i,1].plot(bins_mid,hist_sample*10**bins_mid/opt['norm'],color='red', **histopts)

            # axis labels
            ax[i,0].set_ylabel(r'N')
            ax[i,1].set_ylabel(opt['ylabel'])
            for a in ax[i,:]: a.set_ylim(a.get_ylim()[0],a.get_ylim()[1]*1.1)

            # text labels
            rhofrac = (10**sample_dat).sum() / (10**master_dat).sum()
            ax[i,0].text(0.98, 0.91,'{:1.1f} < z < {:1.1f}'.format(zbin[0],zbin[1]), transform=ax[i,0].transAxes,ha='right',**fontopts)
            ax[i,1].text(0.01, 0.91,r'$\rho_{\mathrm{sample}}/\rho_{\mathrm{total}}$='+'{:1.2f}'.format(rhofrac),
                         transform=ax[i,1].transAxes,ha='left',**fontopts)

        # only bottom axes
        for a in ax[-1,:]: a.set_xlabel(opt['xlabel'])

        fig.tight_layout()
        fig.subplots_adjust(wspace=0.4,hspace=0.0)
        fig.savefig(outfolder+opt['name'],dpi=150)

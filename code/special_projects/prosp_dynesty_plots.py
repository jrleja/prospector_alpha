# specific to comparison
from prospector_io import load_prospector_data
from prosp_dutils import generate_basenames, chop_chain, asym_errors
from prosp_diagnostic_plots import transform_chain
import corner
import numpy as np
import hickle
from matplotlib.ticker import MaxNLocator

# for plots
from dynesty import plotting as dyplot
from prospect.io import read_results
import os
import matplotlib.pyplot as plt

plt.ioff()
fs = 22
tick_fs = 16

def dcorner(sresults, flatchain, pnames, filename=None, show_titles=True, range=None):

    fig = corner.corner(flatchain,
                        labels=pnames, plot_datapoints=False,
                        quantiles=[0.16, 0.5, 0.84], title_kwargs = {'fontsize': 'medium'}, show_titles=True,
                        fill_contours=True, range=range,levels=[0.68,0.95],color='blue',hist_kwargs={'normed':True})
    
    fig.text(0.3,0.95, 'dynesty MAP='+"{:.2f}".format(sresults['lnprobability'].max()),color='blue',fontsize=40)
    if filename is not None:
        plt.savefig(filename,dpi=120)
    return fig

def ecorner(sresults, flatchain, fig, pnames, filename, d_pnames=None, range=None):

    ### resort by d_pnames (sigh)
    idx = np.zeros(flatchain.shape[1],dtype=int)
    for i, d_pname in enumerate(d_pnames): idx[i] = np.where(pnames == d_pname)[0]
    flatchain = flatchain[:,idx]
    labels = d_pnames
    corner.corner(flatchain, color='red',
                  labels=labels, plot_datapoints=False,quantiles=[0.16, 0.5, 0.84],
                  title_kwargs = {'fontsize': 'medium'}, fig=fig, fill_contours=True,
                  range=range,levels=[0.68,0.95],hist_kwargs={'normed':True})#contourf_kwargs={'alpha':0.5})
    fig.text(0.3,0.92, 'emcee MAP='+"{:.2f}".format(sresults['lnprobability'].max()),color='red',fontsize=40)
    plt.savefig(filename,dpi=120)
    plt.close()

def return_dat(runname, runname_comp, outfolder=None, pltcorner=False, dynesty_plots=False, regenerate=False):

    filebase, pfile, ancilname = generate_basenames(runname)
    filebase2, pfile2, ancilname2 = generate_basenames(runname_comp)
    size = 500000
    objname = []

    if not regenerate:
        with open(outfolder+'comparison.hickle', "r") as f:
            dat = hickle.load(f)
        return dat

    for i, file in enumerate(filebase):
        
        sresults, _, model, _ = load_prospector_data("".join(file.split('brownseds_agn_dynesty_')),load_extra_output=False)
        # this is how we detect bad dynesty runs
        if model is None:
            print 'failed to load '+ file.split('_')[-1]
            continue
        flatchain = sresults['chain'][np.random.choice(sresults['chain'].shape[0], size, p=sresults['weights'], replace=True),:]
        flatchain, pnames = transform_chain(flatchain, model)

        sresults_mcmc, _, model_mcmc, _ = load_prospector_data(filebase2[i],load_extra_output=False)
        flatchain_emcee = chop_chain(sresults_mcmc['chain'],**sresults_mcmc['run_params'])
        flatchain_emcee, pnames_emcee = transform_chain(flatchain_emcee, model_mcmc)

        ### add in priors to dynesty
        # use mcmc and resort by mcmc pars
        idx = np.zeros(sresults['chain'].shape[1],dtype=int)
        for n, name in enumerate(pnames_emcee): idx[n] = np.where(pnames==name)[0]
        for n in range(sresults['lnprobability'].shape[0]):
            theta = sresults['chain'][n,idx]
            lnprior = model_mcmc.prior_product(theta)
            sresults['lnprobability'][n] += lnprior

        ### define output containers
        if len(objname) == 0:
            parnames = model.theta_labels()
            mcmc, dynesty = {}, {}
            percs = ['q50','q84','q16','q2.5','q97.5',]
            for dic in [mcmc, dynesty]:
                for par in parnames: 
                    dic[par] = {}
                    for p in percs: dic[par][p] = []
                dic['maxlnprob'] = []

        ### corner plot?
        outname = file.split('results')[0]+'plots/'+runname+'/'
        if pltcorner:
            fig = dcorner(sresults, flatchain, pnames, range=None)
            ecorner(sresults_mcmc, flatchain_emcee, fig, pnames_emcee, outname+file.split('_')[-1]+'.corner.png',
                    d_pnames = pnames, range=None)

        if dynesty_plots:
            dynesty_plot(sresults,model,outfolder=outname, plot_summary=True,plot_trace=True,plot_corner=False)

        ### load all data
        for i,par in enumerate(parnames):
            p1 = flatchain[:,i]
            p2 = flatchain_emcee[:,model_mcmc.theta_labels().index(par)].squeeze()
            for dic, chain in zip([dynesty,mcmc],[p1,p2]): 
                for q in dic[par].keys(): dic[par][q].append(float(np.percentile(chain, [float(q[1:])])[0]))


        dynesty['maxlnprob'].append(sresults['lnprobability'].max())
        mcmc['maxlnprob'].append(sresults_mcmc['lnprobability'].max())
        objname.append(sresults['run_params'].get('objname','galaxy'))
        plt.close()

    dat = {'dynesty': dynesty, 'emcee': mcmc}
    dat['labels'] = pnames
    dat['objname'] = np.array(objname,dtype=str)

    hickle.dump(dat,open(outfolder+'comparison.hickle', "w"))

    return dat

def plot_all(runname='brownseds_agn_dynesty',runname_comp = 'brownseds_agn',pltcorner=False,
             dynesty_plots=False, dat=None, outfolder=None, regenerate=False):
    """ this compares the output parameters
    """

    outfolder = os.getenv('APPS')+'/prospector_alpha/plots/'+runname+'/dynesty_plots/'
    if not os.path.isdir(outfolder):
        os.makedirs(outfolder)

    if dat is None:
        dat = return_dat(runname, runname_comp, pltcorner=pltcorner, outfolder=outfolder,
                         dynesty_plots=dynesty_plots, regenerate=regenerate)

    #### plot photometry
    fig, ax = plt.subplots(4,4, figsize=(14,14))
    fig2, ax2 = plt.subplots(4,4, figsize=(14,14))
    fig3, ax3 = plt.subplots(4,4, figsize=(14,14))
    fignest, axnest = plt.subplots(2,3, figsize=(15, 10))

    ax = np.ravel(ax)
    ax2 = np.ravel(ax2)
    ax3 = np.ravel(ax3)
    axnest = np.ravel(axnest)

    red = '0.5'
    cdynesty = '#FF3D0D'
    cemcee = '#1C86EE'
    erropts = {
              'alpha': 0.5,
              'capthick':0.3,
              'elinewidth':0.3,
              'ecolor': '0.1',
              'linestyle': ' '
              }

    pointopts = {
                 'marker':'o',
                 'alpha':0.9,
                 's':40,
                 'edgecolor': 'k'
                }

    dat['labels'] = dat['emcee'].keys()
    dat['labels'].remove('maxlnprob')
    for ii, par in enumerate(dat['labels']):

        if par == 'fagn':
            log = True
        else:
            log = False

        xerr = asym_errors(np.array(dat['emcee'][par]['q50']), 
                           np.array(dat['emcee'][par]['q84']), 
                           np.array(dat['emcee'][par]['q16']),
                           log=log)
        yerr = asym_errors(np.array(dat['dynesty'][par]['q50']), 
                           np.array(dat['dynesty'][par]['q84']),
                           np.array(dat['dynesty'][par]['q16']),
                           log=log)
        xplot = dat['emcee'][par]['q50']
        yplot = dat['dynesty'][par]['q50']


        if par == 'fagn':
            xplot = np.log10(xplot)
            yplot = np.log10(yplot)
            par = 'log(fagn)'

        ax[ii].errorbar(xplot, yplot, 
                        yerr=yerr, xerr=xerr,
                        ms=0.0, zorder=-2,
                        **erropts)

        ax[ii].scatter(xplot, yplot, color=red, **pointopts)

        ax[ii].set_ylabel('dynesty')
        ax[ii].set_xlabel('emcee')
        ax[ii].set_title(par)

        limits = np.array([ax[ii].get_ylim(),ax[ii].get_xlim()])
        lo,hi = limits.min(),limits.max()
        ax[ii].set_xlim(lo,hi)
        ax[ii].set_ylim(lo,hi)

        ax[ii].plot([lo,hi],[lo,hi], linestyle='--',color='0.5',lw=1.5,zorder=-5)

        ax[ii].xaxis.set_major_locator(MaxNLocator(5))
        ax[ii].yaxis.set_major_locator(MaxNLocator(5))

    ii+=1
    max = 20
    diff = np.clip(np.array(dat['emcee']['maxlnprob']) - np.array(dat['dynesty']['maxlnprob']),-max,max)

    nbins = 20
    n, b, p = ax[ii].hist(diff,
                          nbins, histtype='bar',
                          alpha=0.0,lw=1,log=True)
    ax[ii].set_xlabel('max(lnprob) [emcee-dynesty]')
    ax[ii].set_ylabel('N')
    ax[ii].set_ylim(8e-1,1e2)

    for ii, par in enumerate(dat['labels']):

        xplot = np.array(dat['emcee'][par]['q84']) - np.array(dat['emcee'][par]['q16'])
        yplot = np.array(dat['dynesty'][par]['q84']) - np.array(dat['dynesty'][par]['q16'])

        if par == 'fagn':
            xplot = np.log10(dat['emcee'][par]['q84']) - np.log10(dat['emcee'][par]['q16'])
            yplot = np.log10(dat['dynesty'][par]['q84']) - np.log10(dat['dynesty'][par]['q16'])
            par = 'log(fagn)'

        ax2[ii].scatter(xplot, yplot, color=red, **pointopts)
        
        ax2[ii].set_ylabel(r'dynesty 1$\sigma$ error')
        ax2[ii].set_xlabel(r'emcee 1$\sigma$ error')
        ax2[ii].set_title(par)

        limits = np.array([ax2[ii].get_ylim(),ax2[ii].get_xlim()])
        lo,hi = 0, limits.max()
        ax2[ii].set_ylim(lo,hi)
        ax2[ii].set_xlim(lo,hi)

        ax2[ii].plot([lo,hi],[lo,hi], linestyle='--',color='0.5',lw=1.5,zorder=-5)

        ax2[ii].xaxis.set_major_locator(MaxNLocator(5))
        ax2[ii].yaxis.set_major_locator(MaxNLocator(5))

    for ii, par in enumerate(dat['labels']):

        xplot = np.array(dat['emcee'][par]['q97.5']) - np.array(dat['emcee'][par]['q2.5'])
        yplot = np.array(dat['dynesty'][par]['q97.5']) - np.array(dat['dynesty'][par]['q2.5'])

        if par == 'fagn':
            xplot = np.log10(dat['emcee'][par]['q97.5']) - np.log10(dat['emcee'][par]['q2.5'])
            yplot = np.log10(dat['dynesty'][par]['q97.5']) - np.log10(dat['dynesty'][par]['q2.5'])
            par = 'log(fagn)'

        ax3[ii].scatter(xplot, yplot, color=red, **pointopts)
        
        ax3[ii].set_ylabel(r'dynesty 2$\sigma$ error')
        ax3[ii].set_xlabel(r'emcee 2$\sigma$ error')
        ax3[ii].set_title(par)

        limits = np.array([ax3[ii].get_ylim(),ax3[ii].get_xlim()])
        lo,hi = 0, limits.max()
        ax3[ii].set_ylim(lo,hi)
        ax3[ii].set_xlim(lo,hi)

        ax3[ii].plot([lo,hi],[lo,hi], linestyle='--',color='0.5',lw=1.5,zorder=-5)

        ax3[ii].xaxis.set_major_locator(MaxNLocator(5))
        ax3[ii].yaxis.set_major_locator(MaxNLocator(5))


    ax2[ii+1].axis('off')

    fig.tight_layout()
    fig.savefig(outfolder+'parameters.png',dpi=150)
    fig2.tight_layout()
    fig2.savefig(outfolder+'onesig_errors.png',dpi=150)
    fig3.tight_layout()
    fig3.savefig(outfolder+'twosig_errors.png',dpi=150)
    plt.close()
    plt.close()
    plt.close()

    return dat


def dynesty_plot(sample_results,model,outfolder='', plot_summary=True,plot_trace=True,plot_corner=True):
    """ this creates all of the default dynesty plots
    """

    # I/O
    '''
    model_filename = '/Users/joel/code/python/prospector_alpha/results/brownseds_agn_dynesty/Arp 118_1504981910_model'
    mcmc_filename = '/Users/joel/code/python/prospector_alpha/results/brownseds_agn_dynesty/Arp 118_1504981910_mcmc.h5'
    sample_results, powell_results, model = read_results.results_from(mcmc_filename, model_file=model_filename,inmod=None)
    '''
    # recreate dynesty variables for dynesty plotting
    objname = sample_results['run_params']['objname']
    sample_results['logwt'] = np.log(sample_results['weights'])+sample_results['logz'][-1]
    sample_results['logl'] = sample_results['lnprobability']
    sample_results['samples'] = sample_results['chain']
    parnames = model.theta_labels()
    font_kwargs = {'fontsize': fs}

    # Plot a summary of the run.
    if plot_summary:
        rfig, raxes = dyplot.runplot(sample_results,mark_final_live=False, label_kwargs=font_kwargs)
        for ax in raxes:
            ax.xaxis.set_tick_params(labelsize=tick_fs)
            ax.yaxis.set_tick_params(labelsize=tick_fs)
            ax.yaxis.get_offset_text().set_size(fs)
        rfig.tight_layout()
        rfig.savefig(outfolder+objname+'_dynesty_summary.png',dpi=150)

    # Plot traces and 1-D marginalized posteriors.
    if plot_trace:
        tfig, taxes = dyplot.traceplot(sample_results,# connect=True, connect_highlight=range(4),
                                       labels=parnames,label_kwargs=font_kwargs)
        for ax in taxes.ravel():
            ax.xaxis.set_tick_params(labelsize=tick_fs)
            ax.yaxis.set_tick_params(labelsize=tick_fs)
        tfig.tight_layout()
        tfig.savefig(outfolder+objname+'_dynesty_trace.png',dpi=130)

    # Plot the 2-D marginalized posteriors.
    if plot_corner:
        cfig, caxes = dyplot.cornerplot(sample_results, show_titles=True, labels=parnames,
                                        label_kwargs={'fontsize':fs*.7}, title_kwargs={'fontsize':fs*.7})
        for ax in caxes.ravel():
            ax.xaxis.set_tick_params(labelsize=tick_fs*.7)
            ax.yaxis.set_tick_params(labelsize=tick_fs*.7)
        cfig.savefig('dynesty_corner.png',dpi=130)
    
    plt.close()

import matplotlib.pyplot as plt
import corner
import numpy as np
from prospect.io import read_results
from prospect.models import model_setup
from brown_io import load_prospector_data
from prosp_dutils import generate_basenames, chop_chain, asym_errors
from prosp_diagnostic_plots import transform_chain
from matplotlib.ticker import MaxNLocator
import os

plt.ioff()

def ncorner(sresults, model, filename=None, show_titles=True, range=None):

    chain = sresults['chain'][np.random.choice(sresults['chain'].shape[0], int(1e6), p=sresults['weights'], replace=True),:]
    chain, pnames = transform_chain(chain, model)

    fig = corner.corner(chain,
                        labels=pnames, plot_datapoints=False,
                        quantiles=[0.16, 0.5, 0.84], title_kwargs = {'fontsize': 'medium'}, show_titles=True,
                        fill_contours=True, range=range,levels=[0.68,0.95],color='blue',hist_kwargs={'normed':True})
                        #contourf_kwargs={'alpha':0.5})
    
    fig.text(0.3,0.95, 'nestle MAP='+"{:.2f}".format(sresults['lnprobability'].max()),color='blue',fontsize=40)
    if filename is not None:
        plt.savefig(filename,dpi=120)
    return fig, pnames

def ecorner(sresults,flatchain, fig, epnames, filename, n_pnames=None, range=None):

    ### resort by n_pnames (sigh)
    if n_pnames is not None:
        idx = np.zeros(flatchain.shape[1],dtype=int)
        for i, nn in enumerate(n_pnames): idx[i] = np.where(epnames==nn)[0]
        flatchain = flatchain[:,idx]
        labels = n_pnames
    else:
        labels = epnames

    corner.corner(flatchain, color='red',
                  labels=labels, plot_datapoints=False,quantiles=[0.16, 0.5, 0.84],
                  title_kwargs = {'fontsize': 'medium'}, fig=fig, fill_contours=True,
                  range=range,levels=[0.68,0.95],hist_kwargs={'normed':True})#contourf_kwargs={'alpha':0.5})
    fig.text(0.3,0.92, 'emcee MAP='+"{:.2f}".format(sresults['lnprobability'].max()),color='red',fontsize=40)
    plt.savefig(filename,dpi=120)
    plt.close()

def plot_chain(sresults, pnames, outfig_name):

    fig, ax = plt.subplots(4, 5, figsize = (15,11))
    ax = np.ravel(ax)

    percentile_lim = 90

    opts = {
            'lw': 1,
            'color': 'k'
    }

    ### plot thinned chain
    thin = 20
    nsamp = float(sresults['chain'].shape[0])
    xp = np.arange(np.ceil(nsamp/thin))*thin
    for i, par in enumerate(pnames):
        ax[i].plot(xp, sresults['chain'][::thin,i],**opts)
        ax[i].set_ylabel(par)
        ax[i].set_xlabel('iteration')
        ax[i].xaxis.set_major_locator(MaxNLocator(4))
        [l.set_rotation(45) for l in ax[i].get_xticklabels()]

    idx = sresults['lnprobability'] > np.percentile(sresults['lnprobability'],percentile_lim)
    ax[i+1].plot(np.arange(sresults['lnprobability'].shape[0])[idx],sresults['lnprobability'][idx],**opts)
    ax[i+1].set_ylabel('lnprobability')
    ax[i+1].set_xlabel('iteration')
    [l.set_rotation(45) for l in ax[i+1].get_xticklabels()]

    ax[i+1].xaxis.set_major_locator(MaxNLocator(4))

    nbins = 50
    weights = np.clip(sresults['weights'],np.percentile(sresults['weights'],percentile_lim),np.inf)
    n, b, p = ax[i+2].hist(np.log10(weights),bins=nbins,normed=False,color='k',log=True)
    ax[i+2].set_ylabel('N')
    ax[i+2].set_xlabel('log(weights) (clipped @ 90th percentile)')
    ax[i+2].set_ylim(0.1,n.max()*5)

    ax[i+2].xaxis.set_major_locator(MaxNLocator(4))
    ax[i+3].axis('off')
    ax[i+4].axis('off')
    ax[i+5].axis('off')

    ### text
    xstart, delx = 0.1, 0.3
    ystart, dely, yend = 0.96, 0.025, 0.9
    xt, yt = xstart, ystart
    pars = ['logvol', 'h_information', 'ncall', 'logz', 'logzerr', 'dlogz']
    str_fmt = [".2f", ".2f", "", ".2f", ".4f", ".3f"]
    for strfmt, par in zip(str_fmt,pars):
        if type(sresults[par]) is np.ndarray:
            txt = sresults[par][-1]
        else:
            txt = sresults[par]
        fmt = "{{0:{0}}}".format(strfmt).format
        text = r"$\mathbf{{{0}}}={1}$".format(par.replace('_',' '),fmt(txt))
        fig.text(xt, yt, text, fontsize=14)
        yt-= dely
        if yt < yend:
            yt = ystart
            xt = xt+delx

    plt.tight_layout(rect=(0,0,1,0.9))
    plt.savefig(outfig_name,dpi=120)
    plt.close()

def return_dat(runname, runname_comp, pltcorner=False, pltchain=False):

    #filebase, pfile, ancilname = generate_basenames(runname)
    #filebase2, pfile2, ancilname2 = generate_basenames(runname_comp)
    size = 500000

    filebase = ['/Users/joel/code/python/threedhst_bsfh/results/guillermo_nestle/guillermo_nestle']
    pfile = ['/Users/joel/code/python/threedhst_bsfh/parameter_files/guillermo_nestle/guillermo_nestle_params.py']
    filebase2 = ['/Users/joel/code/python/threedhst_bsfh/results/guillermo/guillermo']
    pfile2 = ['/Users/joel/code/python/threedhst_bsfh/parameter_files/guillermo/guillermo_params.py']


    #bad =  ['NGC 0584','UGCA 166','Mrk 1450','UM 461','UGC 06850','NGC 4125','NGC 4551','Mrk 0475']

    objname, hinformation, logz, logzerr, dlogz, ncall = [], [], [], [], [], []
    for i, file in enumerate(filebase):
        
        sresults, _, model, _ = load_prospector_data(file,load_extra_output=False)

        ### some of the nestle runs terminated due to time limit
        ### must regenerate the models
        if model is None:
            run_params = model_setup.get_run_params(param_file=pfile)
            run_params['objname'] = file.split('_')[-1]
            model = model_setup.load_model(**run_params)

        sresults_mcmc, _, model_mcmc, _ = load_prospector_data(filebase2[i],load_extra_output=False)
        flatchain = chop_chain(sresults_mcmc['chain'],**sresults_mcmc['run_params'])
        flatchain, pnames_emcee = transform_chain(flatchain, model_mcmc)

        ### save some stuff
        # logz_est.append(sresults['logvol'][sresults['lnprobability'].argmax()]+np.log10(np.exp(sresults['lnprobability'].max()))
        npoints = sresults['run_params']['nestle_npoints']
        logz_remain = np.max(sresults['lnprobability']) - (sresults['chain'].shape[0]/1000.)
        
        dlogz.append(np.logaddexp(sresults['logz'], logz_remain) - sresults['logz'])
        ncall.append(sresults['ncall'])
        hinformation.append(sresults['h_information'][0])
        logz.append(sresults['logz'][0])
        logzerr.append(sresults['logzerr'][0])
        sresults['dlogz'] = dlogz[-1]

        ### define output containers
        if len(objname) == 0:
            parnames = model.theta_labels()
            mcmc, nestle = {}, {}
            percs = ['q50','q84','q16','q2.5','q97.5',]
            for dic in [mcmc, nestle]:
                for par in parnames: 
                    dic[par] = {}
                    for p in percs: dic[par][p] = []
                dic['maxlnprob'] = []

        ### corner plot?
        outname = file.split('results')[0]+'plots/'+runname+'/'+file.split('_')[-1]
        if pltcorner:
            # outname = outname+'.corner.png'
            range = [tuple(np.percentile(flatchain[:,i],[1,99]).tolist()) for i in xrange(flatchain.shape[1])]
            range = None
            fig, nestle_pnames = ncorner(sresults, model, range=range)
            ecorner(sresults_mcmc, flatchain, fig, pnames_emcee, outname+'.corner.png',n_pnames = nestle_pnames, range=range)
        if pltchain:
            plot_chain(sresults, parnames, outname+'.chain.png')

        ### load all data
        for i,par in enumerate(parnames):
            p1 = np.random.choice(sresults['chain'][:,i],size=size, p=sresults['weights'])
            p2 = flatchain[:,i]
            for dic, chain in zip([nestle,mcmc],[p1,p2]): 
                for q in dic[par].keys(): dic[par][q].append(np.percentile(chain, [float(q[1:])])[0])

        nestle['maxlnprob'].append(sresults['lnprobability'].max())
        mcmc['maxlnprob'].append(sresults_mcmc['lnprobability'].max())
        objname.append(sresults['run_params'].get('objname','galaxy'))
        print 1/0
    dat = {'nestle': nestle, 'emcee': mcmc}
    dat['labels'] = model.theta_labels()
    dat['objname'] = objname

    nestle_pars = {
                   'hinformation': hinformation,
                   'logz': logz,
                   'logzerr': logzerr,
                   'dlogz': dlogz,
                   'ncall': ncall
                  }
    dat['nestle_pars'] = nestle_pars
    return dat

def plot_all(runname='brownseds_agn_nestle',runname_comp = 'brownseds_agn',pltcorner=False, pltchain=False,
             dat=None, outfolder=None):
    
    if dat is None:
        dat = return_dat(runname, runname_comp, pltcorner=pltcorner, pltchain=pltchain)
        return dat

    outfolder = os.getenv('APPS')+'/threedhst_bsfh/plots/'+runname+'/nestle_plots/'
    if not os.path.isdir(outfolder):
        os.makedirs(outfolder)

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
    cnestle = '#FF3D0D'
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

    dlogzthresh = '4'
    better_emcee = np.array(dat['nestle_pars']['dlogz']).squeeze() > int(dlogzthresh)
    print 'dlogz fail: '+",".join(np.array(dat['objname'])[better_emcee].tolist())

    higher_met = (np.array(dat['nestle']['logzsol']['q50']) - np.array(dat['emcee']['logzsol']['q50'])) > 0.8
    print 'higher metallicity for nestle: ' + ", ".join(np.array(dat['objname'])[higher_met].tolist())

    waybetteremcee = np.array(dat['emcee']['maxlnprob']) > np.array(dat['nestle']['maxlnprob'])+50
    print 'better fit for prospector: ' + ",".join(np.array(dat['objname'])[waybetteremcee].tolist())

    badmass = np.array(dat['nestle']['logmass']['q84']) -  np.array(dat['nestle']['logmass']['q16']) < 0.01
    print 'shit error bars for nestle: ' + ",".join(np.array(dat['objname'])[badmass].tolist())


    nbins = 50
    axnest[0].hist(np.array(dat['nestle_pars']['logz']).squeeze(),bins=nbins,normed=False,alpha=0.6,color='blue')
    axnest[0].set_xlabel('log(z)')
    axnest[0].set_ylabel('N')

    axnest[1].hist(np.log10(dat['nestle_pars']['logzerr']),bins=nbins,normed=False,alpha=0.6,color='blue')
    axnest[1].set_xlabel(r'log(z) err')
    axnest[1].set_ylabel('N')

    axnest[2].hist(np.log10(dat['nestle_pars']['ncall']),bins=nbins,normed=False,alpha=0.6,color='blue')
    axnest[2].set_xlabel(r'log(n$_{\mathrm{call}}$)')
    axnest[2].set_ylabel('N')

    axnest[3].hist(np.array(dat['nestle_pars']['dlogz']).squeeze(),bins=nbins,normed=False,alpha=0.6,color='blue')
    axnest[3].set_xlabel(r'dlog(z)')
    axnest[3].set_ylabel('N')

    axnest[4].hist(dat['nestle_pars']['hinformation'],bins=nbins,normed=False,alpha=0.6,color='blue')
    axnest[4].set_xlabel(r'H information')
    axnest[4].set_ylabel('N')

    fignest.tight_layout()
    fignest.savefig(outfolder+'nestle_parameters.png',dpi=150)

    for ii, par in enumerate(dat['labels']):

        if par == 'fagn':
            log = True
        else:
            log = False

        xerr = asym_errors(np.array(dat['emcee'][par]['q50']), 
                           np.array(dat['emcee'][par]['q84']), 
                           np.array(dat['emcee'][par]['q16']),
                           log=log)
        yerr = asym_errors(np.array(dat['nestle'][par]['q50']), 
                           np.array(dat['nestle'][par]['q84']),
                           np.array(dat['nestle'][par]['q16']),
                           log=log)
        xplot = dat['emcee'][par]['q50']
        yplot = dat['nestle'][par]['q50']


        if par == 'fagn':
            xplot = np.log10(xplot)
            yplot = np.log10(yplot)
            par = 'log(fagn)'

        ax[ii].errorbar(xplot, yplot, 
                        yerr=yerr, xerr=xerr,
                        ms=0.0, zorder=-2,
                        **erropts)

        ax[ii].scatter(xplot, yplot, color=red, **pointopts)
        ax[ii].scatter(np.array(xplot)[better_emcee], np.array(yplot)[better_emcee], color=cemcee, **pointopts)

        ax[ii].text(0.05,0.95, 'dlogz > '+dlogzthresh, fontsize=8, color=cemcee, transform=ax[ii].transAxes)

        ax[ii].set_ylabel('nestle')
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
    diff = np.clip(np.array(dat['emcee']['maxlnprob']) - np.array(dat['nestle']['maxlnprob']),-max,max)

    nbins = 20
    n, b, p = ax[ii].hist(diff,
                          nbins, histtype='bar',
                          alpha=0.0,lw=1,log=True)
    n, b, p = ax[ii].hist(diff[better_emcee],
                         bins=b, histtype='bar',
                         color=cemcee,log=True,
                         alpha=0.75,lw=1)
    n, b, p = ax[ii].hist(diff[~better_emcee],
                         bins=b, histtype='bar',
                         color=red,log=True,alpha=0.75,
                         lw=1)
    ax[ii].set_xlabel('max(lnprob) [emcee-nestle]')
    ax[ii].set_ylabel('N')
    ax[ii].set_ylim(8e-1,1e2)

    for ii, par in enumerate(dat['labels']):

        xplot = np.array(dat['emcee'][par]['q84']) - np.array(dat['emcee'][par]['q16'])
        yplot = np.array(dat['nestle'][par]['q84']) - np.array(dat['nestle'][par]['q16'])

        if par == 'fagn':
            xplot = np.log10(dat['emcee'][par]['q84']) - np.log10(dat['emcee'][par]['q16'])
            yplot = np.log10(dat['nestle'][par]['q84']) - np.log10(dat['nestle'][par]['q16'])
            par = 'log(fagn)'

        ax2[ii].scatter(xplot, yplot, color=red, **pointopts)
        ax2[ii].scatter(xplot[better_emcee], yplot[better_emcee], color=cemcee, **pointopts)
        
        ax2[ii].set_ylabel(r'nestle 1$\sigma$ error')
        ax2[ii].set_xlabel(r'emcee 1$\sigma$ error')
        ax2[ii].set_title(par)

        ax2[ii].text(0.05,0.95, 'better emcee', fontsize=8, color=cemcee, transform=ax2[ii].transAxes)

        limits = np.array([ax2[ii].get_ylim(),ax2[ii].get_xlim()])
        lo,hi = 0, limits.max()
        ax2[ii].set_ylim(lo,hi)
        ax2[ii].set_xlim(lo,hi)

        ax2[ii].plot([lo,hi],[lo,hi], linestyle='--',color='0.5',lw=1.5,zorder=-5)

        ax2[ii].xaxis.set_major_locator(MaxNLocator(5))
        ax2[ii].yaxis.set_major_locator(MaxNLocator(5))

    for ii, par in enumerate(dat['labels']):

        xplot = np.array(dat['emcee'][par]['q97.5']) - np.array(dat['emcee'][par]['q2.5'])
        yplot = np.array(dat['nestle'][par]['q97.5']) - np.array(dat['nestle'][par]['q2.5'])

        if par == 'fagn':
            xplot = np.log10(dat['emcee'][par]['q97.5']) - np.log10(dat['emcee'][par]['q2.5'])
            yplot = np.log10(dat['nestle'][par]['q97.5']) - np.log10(dat['nestle'][par]['q2.5'])
            par = 'log(fagn)'

        ax3[ii].scatter(xplot, yplot, color=red, **pointopts)
        ax3[ii].scatter(xplot[better_emcee], yplot[better_emcee], color=cemcee, **pointopts)
        
        ax3[ii].set_ylabel(r'nestle 2$\sigma$ error')
        ax3[ii].set_xlabel(r'emcee 2$\sigma$ error')
        ax3[ii].set_title(par)

        ax3[ii].text(0.05,0.95, 'better emcee', fontsize=8, color=cemcee, transform=ax3[ii].transAxes)

        limits = np.array([ax2[ii].get_ylim(),ax2[ii].get_xlim()])
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


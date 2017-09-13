import numpy as np
from prosp_diagnostic_plots import add_sfh_plot
import os
import matplotlib.pyplot as plt
import magphys_plot_pref
import matplotlib as mpl
from magphys_plots import median_by_band
import prosp_dutils

blue = '#1C86EE' 

def collate_data(alldata, alldata_noagn):

    ### package up information
    the_data = [alldata, alldata_noagn]
    data_label = ['agn','no_agn']
    output = {}
    for ii in xrange(2):

        # model parameters
        objname = []
        model_pars = {}
        pnames = ['fagn', 'logzsol', 'stellar_mass']
        for p in pnames: 
            model_pars[p] = {'q50':[],'q84':[],'q16':[],'chain':[]}

        #### load model information
        for dat in the_data[ii]:

            parnames = dat['pquantiles']['parnames']
            eparnames = dat['pextras']['parnames']

            for key in model_pars.keys():
                if key in parnames:
                    model_pars[key]['q50'].append(dat['pquantiles']['q50'][parnames==key][0])
                    model_pars[key]['q84'].append(dat['pquantiles']['q84'][parnames==key][0])
                    model_pars[key]['q16'].append(dat['pquantiles']['q16'][parnames==key][0])
                    model_pars[key]['chain'].append(dat['pquantiles']['sample_chain'][:,parnames==key].flatten())
                elif key in eparnames:
                    model_pars[key]['q50'].append(dat['pextras']['q50'][eparnames==key][0])
                    model_pars[key]['q84'].append(dat['pextras']['q84'][eparnames==key][0])
                    model_pars[key]['q16'].append(dat['pextras']['q16'][eparnames==key][0])

        for key in model_pars.keys(): 
            for key2 in model_pars[key].keys():
                model_pars[key][key2] = np.array(model_pars[key][key2])

        output[data_label[ii]] = model_pars

    return output

def plot_comparison(runname='brownseds_agn',runname_noagn='brownseds_np',alldata=None,alldata_noagn=None,outfolder=None,plt_idx=None,**popts):

    #### load alldata
    if alldata is None:
        import prospector_io

        alldata = prospector_io.load_alldata(runname=runname)
        alldata_noagn = prospector_io.load_alldata(runname=runname_noagn)

    #### make output folder if necessary
    if outfolder is None:
        outfolder = os.getenv('APPS')+'/prospector_alpha/plots/'+runname+'/agn_plots/'
        if not os.path.isdir(outfolder):
            os.makedirs(outfolder)
    
    if plt_idx is None:
        plt_idx = np.full(129,True,dtype=bool)

    ### collate data
    ### choose galaxies with largest 10 F_AGN
    pdata = collate_data(alldata,alldata_noagn)

    #### MASS-METALLICITY
    fig,ax = plot_massmet(pdata,plt_idx,**popts)
    fig.tight_layout()
    fig.savefig(outfolder+'delta_massmet.png',dpi=150)
    plt.close()

def drawArrow(A, B, ax):
    ax.arrow(A[0], A[1], B[0] - A[0], B[1] - A[1],
              head_width=0.05, width=0.01,length_includes_head=True,color='grey',alpha=0.6)

def plot_massmet(pdata,plt_idx,**popts):

    fig, ax = plt.subplots(1,1,figsize=(6.5,6))
    ylim = (-2.4,0.5)
    xlabel = r'log(M$_*$) [M$_{\odot}$]'
    ylabel = r'log(Z/Z$_{\odot}$)'

    ### pull out fagn
    nonagn_idx = np.array([0 if i in plt_idx else 1 for i in range(129)],dtype=bool)
    fagn = np.log10(pdata['agn']['fagn']['q50'][plt_idx])

    ### pick out errors
    err_mass_agn = prosp_dutils.asym_errors(pdata['agn']['stellar_mass']['q50'][plt_idx],pdata['agn']['stellar_mass']['q84'][plt_idx], pdata['agn']['stellar_mass']['q16'][plt_idx],log=True)
    err_mass_noagn = prosp_dutils.asym_errors(pdata['no_agn']['stellar_mass']['q50'][plt_idx],pdata['no_agn']['stellar_mass']['q84'][plt_idx], pdata['no_agn']['stellar_mass']['q16'][plt_idx],log=True)
    err_met_agn = prosp_dutils.asym_errors(pdata['agn']['logzsol']['q50'][plt_idx],pdata['agn']['logzsol']['q84'][plt_idx], pdata['agn']['logzsol']['q16'][plt_idx])
    err_met_noagn = prosp_dutils.asym_errors(pdata['no_agn']['logzsol']['q50'][plt_idx],pdata['no_agn']['logzsol']['q84'][plt_idx], pdata['no_agn']['logzsol']['q16'][plt_idx])
    mass_agn = np.log10(pdata['agn']['stellar_mass']['q50'])
    mass_noagn = np.log10(pdata['no_agn']['stellar_mass']['q50'])

    ### plots
    alpha = 0.7
    pts = ax.scatter(mass_noagn[plt_idx],pdata['no_agn']['logzsol']['q50'][plt_idx], marker='o', color=popts['noagn_color'],s=50,zorder=10,alpha=alpha,edgecolors='k')
    pts = ax.scatter(mass_agn[plt_idx],pdata['agn']['logzsol']['q50'][plt_idx], marker='o', color=popts['agn_color'],s=50,zorder=10,alpha=alpha,edgecolors='k')
    ax.scatter(mass_agn[nonagn_idx],pdata['agn']['logzsol']['q50'][nonagn_idx], marker='o', color='0.4',s=10,zorder=-10,alpha=0.4)
    '''
    ax.errorbar(mass_noagn[plt_idx],pdata['no_agn']['logzsol']['q50'][plt_idx], 
                     yerr=err_met_noagn, xerr=err_mass_noagn,
                     fmt='o', ecolor='0.5', capthick=0.5,elinewidth=0.5,
                     ms=0,alpha=0.4,zorder=-5)

    ax.errorbar(mass_agn[plt_idx],pdata['agn']['logzsol']['q50'][plt_idx], 
                     yerr=err_met_agn, xerr=err_mass_agn,
                     fmt='o', ecolor='0.5', capthick=0.5,elinewidth=0.5,
                     ms=0,alpha=0.4,zorder=-5)
    '''
    for ii in xrange(len(plt_idx)):
        old = (mass_noagn[plt_idx][ii],pdata['no_agn']['logzsol']['q50'][plt_idx[ii]])
        new = (mass_agn[plt_idx][ii],pdata['agn']['logzsol']['q50'][plt_idx[ii]])
        drawArrow(old,new,ax)

    massmet = np.loadtxt(os.getenv('APPS')+'/prospector_alpha/data/gallazzi_05_massmet.txt')
    lw = 2.5
    color = 'green'
    ax.plot(massmet[:,0], massmet[:,1],
           color=color,
           lw=lw,
           linestyle='--',
           zorder=-1,
           label='Gallazzi et al. 2005')
    ax.plot(massmet[:,0],massmet[:,2],
           color=color,
           lw=lw,
           zorder=-1)
    ax.plot(massmet[:,0],massmet[:,3],
           color=color,
           lw=lw,
           zorder=-1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)
    ax.set_xlim(9,11.5)
    ax.legend(loc=2,prop={'size':14},frameon=False)
    
    ### new plot
    median_z_noagn = np.interp(mass_noagn[plt_idx], massmet[:,0], massmet[:,1])
    lower_z_noagn = np.interp(mass_noagn[plt_idx], massmet[:,0], massmet[:,2])
    upper_z_noagn = np.interp(mass_noagn[plt_idx], massmet[:,0], massmet[:,3])
    median_z_agn = np.interp(mass_agn[plt_idx], massmet[:,0], massmet[:,1])
    lower_z_agn = np.interp(mass_agn[plt_idx], massmet[:,0], massmet[:,2])
    upper_z_agn = np.interp(mass_agn[plt_idx], massmet[:,0], massmet[:,3])
    
    nboot, nidx = 10000, len(plt_idx)
    deviation_noagn_mean, deviation_agn_mean = [], []
    for i in xrange(nboot):

        # choose from galaxies, with replacement (bootstrap!)
        new_plt_idx = np.random.choice(plt_idx,nidx,replace=True)

        # choose randomly from galaxy PDFs (also choose plt_idx randomly with replacement before this?)
        logzsol_agn, logzsol_noagn = [], []
        for idx in new_plt_idx: 
            logzsol_noagn.append(np.random.choice(pdata['no_agn']['logzsol']['chain'][idx],1)[0])
            logzsol_agn.append(np.random.choice(pdata['agn']['logzsol']['chain'][idx],1)[0])

        deviation_noagn = np.array(logzsol_noagn) - median_z_noagn 
        up = deviation_noagn > 0
        deviation_noagn[up] /= (upper_z_noagn[up]-median_z_noagn[up])
        deviation_noagn[~up] /= (median_z_noagn[~up]-lower_z_noagn[~up])

        deviation_agn = np.array(logzsol_agn) - median_z_agn 
        up = deviation_agn > 0
        deviation_agn[up] /= (upper_z_agn[up]-median_z_agn[up])
        deviation_agn[~up] /= (median_z_agn[~up]-lower_z_agn[~up])

        deviation_noagn_mean.append(deviation_noagn.mean())
        deviation_agn_mean.append(deviation_agn.mean())

    ondo, on, onup = np.percentile(deviation_agn_mean,[16,50,84])
    offdo, off, offup = np.percentile(deviation_noagn_mean,[16,50,84])

    # format display
    fmt = "{{0:{0}}}".format(".1f").format
    textoff = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
    textoff = textoff.format(fmt(off), fmt(off-offdo), fmt(offup-off))
    texton = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
    texton = texton.format(fmt(on), fmt(on-ondo), fmt(onup-on))
    
    textoff = "mean offset (AGN off): {0} {1}".format(textoff,r"$\sigma_{\mathrm{SDSS}}$")
    texton = "mean offset (AGN on): {0} {1}".format(texton,r"$\sigma_{\mathrm{SDSS}}$")

    ax.text(0.04,0.11, textoff,
            transform=ax.transAxes,fontsize=15,color=popts['noagn_color'])
    ax.text(0.04,0.05, texton,
            transform=ax.transAxes,fontsize=15,color=popts['agn_color'])

    return fig,ax
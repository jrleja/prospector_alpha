import numpy as np
import brown_io
import matplotlib.pyplot as plt
import agn_plot_pref
from corner import quantile
import os
from prosp_dutils import asym_errors

dpi = 150

def collate_data(alldata):

    ### BPT information
    oiii_hb = np.zeros(shape=(len(alldata),3))
    nii_ha = np.zeros(shape=(len(alldata),3))
    linenames = alldata[0]['residuals']['emlines']['em_name']
    ha_em = linenames == 'H$\\alpha$'
    hb_em = linenames == 'H$\\beta$'
    oiii_em = linenames == '[OIII] 5007'
    nii_em = linenames == '[NII] 6583'

    ### model parameters
    model_pars = {}
    pnames = ['fagn', 'agn_tau']
    for p in pnames: model_pars[p] = []
    parnames = alldata[0]['pquantiles']['parnames']
    objname = []

    #### load information
    for ii, dat in enumerate(alldata):

        #### model parameters
        objname.append(dat['objname'])
        for key in model_pars.keys():
            model_pars[key].append(dat['pquantiles']['q50'][parnames==key][0])

        # set errors to infinity
        if dat['residuals']['emlines'] is None:
            oiii_hb[ii,1] = np.inf
            nii_ha[ii,1] = np.inf
            continue
        
        '''
        f_oiii = dat['residuals']['emlines']['obs']['lum'][oiii_em]
        f_hb = dat['residuals']['emlines']['obs']['lum'][hb_em]
        err_oiii = (dat['residuals']['emlines']['obs']['lum_errup'][oiii_em] - dat['residuals']['emlines']['obs']['lum_errdown'][oiii_em])/2.
        err_hb = (dat['residuals']['emlines']['obs']['lum_errup'][hb_em] - dat['residuals']['emlines']['obs']['lum_errdown'][hb_em])/2.
        oiii_hb[ii,0] = f_oiii / f_hb
        oiii_hb[ii,1] = oiii_hb[ii,0] * np.sqrt((err_oiii/f_oiii)**2+(err_hb/f_hb)**2)

        f_nii = dat['residuals']['emlines']['obs']['lum'][nii_em]
        f_ha = dat['residuals']['emlines']['obs']['lum'][ha_em]
        err_nii = (dat['residuals']['emlines']['obs']['lum_errup'][nii_em] - dat['residuals']['emlines']['obs']['lum_errdown'][nii_em])/2.
        err_ha = (dat['residuals']['emlines']['obs']['lum_errup'][ha_em] - dat['residuals']['emlines']['obs']['lum_errdown'][ha_em])/2.
        nii_ha[ii,0] = f_nii / f_ha
        nii_ha[ii,1] = nii_ha[ii,0] * np.sqrt((err_nii/f_nii)**2+(err_ha/f_ha)**2)
        '''

        #### calculate ratios from the chain
        ratio = np.log10(dat['residuals']['emlines']['obs']['lum_chain'][:,oiii_em].squeeze() / dat['residuals']['emlines']['obs']['lum_chain'][:,hb_em].squeeze())
        oiii_hb[ii,:] = np.percentile(ratio, [50, 84, 16])
            
        ratio = np.log10(dat['residuals']['emlines']['obs']['lum_chain'][:,nii_em].squeeze() / dat['residuals']['emlines']['obs']['lum_chain'][:,ha_em].squeeze())
        nii_ha[ii,:] = np.percentile(ratio, [50, 84, 16])

    #### numpy arrays
    for key in model_pars.keys(): model_pars[key] = np.array(model_pars[key])

    out = {}
    out['model_pars'] = model_pars
    out['objname'] = objname
    out['oiii_hb'] = oiii_hb
    out['nii_ha'] = nii_ha
    return out

def plot_bpt(agn_evidence,runname='brownseds_agn',alldata=None,outfolder=None,idx=None,**popts):

    #### load alldata
    if alldata is None:
        alldata = brown_io.load_alldata(runname=runname)

    #### make output folder if necessary
    if outfolder is None:
        outfolder = os.getenv('APPS')+'/threedhst_bsfh/plots/'+runname+'/agn_plots/'
        if not os.path.isdir(outfolder):
            os.makedirs(outfolder)

    #### collate data
    pdata = collate_data(alldata)

    ### BPT PLOT
    fig,ax = plot_scatterplot(pdata,colorpar='fagn',colorparlabel=r'log(f$_{\mathrm{MIR}}$)',
                                     log_cpar=True, cpar_range=[-2,0], idx=idx, **popts)
    add_kewley_classifications(ax)
    plt.tight_layout()
    plt.savefig(outfolder+'bpt_fagn.png',dpi=dpi)
    plt.close()

    agn_evidence['bpt_type'] = return_bpt_type(pdata)
    agn_evidence['bpt_use_flag'] = bpt_cuts(pdata)
    return agn_evidence

def return_bpt_type(pdata):

    bpt_flag = np.chararray(pdata['oiii_hb'].shape[0],itemsize=5)

    sf_line1 = 0.61 / (pdata['nii_ha'][:,0] - 0.05) + 1.3
    sf_line2 = 0.61 / (pdata['nii_ha'][:,0] - 0.47) + 1.19
    composite = (pdata['oiii_hb'][:,0] > sf_line1) & (pdata['oiii_hb'][:,0] < sf_line2)
    agn = pdata['oiii_hb'][:,0] > sf_line2

    bpt_flag[:] = 'sf'
    bpt_flag[composite] = 'comp'
    bpt_flag[agn] = 'agn'

    return bpt_flag

def add_kewley_classifications(ax):

    #### plot bpt dividers
    # Kewley+06
    # log(OIII/Hbeta) < 0.61 /[log(NII/Ha) - 0.05] + 1.3 (star-forming to the left and below)
    # log(OIII/Hbeta) < 0.61 /[log(NII/Ha) - 0.47] + 1.19 (between AGN and star-forming)
    # x = 0.61 / (y-0.47) + 1.19
    x1 = np.linspace(-2.2,0.0,num=50)
    x2 = np.linspace(-2.2,0.35,num=50)
    ax.plot(x1,0.61 / (x1 - 0.05) + 1.3 , linestyle='--',color='0.5',lw=1.5)
    ax.plot(x2,0.61 / (x2-0.47) + 1.19, linestyle='--',color='0.5',lw=1.5)

def bpt_cuts(pdata):

    #### only select those with good BPT measurements
    sncut = 0.2  # in dex
    good = ((pdata['oiii_hb'][:,1]-pdata['oiii_hb'][:,2])/2. < sncut) & \
           ((pdata['nii_ha'][:,1]-pdata['nii_ha'][:,2])/2. < sncut) & \
           np.isfinite(pdata['nii_ha'][:,0]) & np.isfinite(pdata['oiii_hb'][:,0])
    return good

def plot_scatterplot(pdata,colorpar=None,colorparlabel=None,log_cpar=False,cpar_range=None,idx=None,**popts):
    '''
    plots a color-color BPT scatterplot
    '''
    good = bpt_cuts(pdata)

    cidx = np.ones_like(good,dtype=bool)
    cidx[idx] = False
    cidx = cidx[good]

    #### generate x, y values
    xerr = asym_errors(pdata['nii_ha'][good,0],pdata['nii_ha'][good,1],pdata['nii_ha'][good,2])
    yerr = asym_errors(pdata['oiii_hb'][good,0],pdata['oiii_hb'][good,1],pdata['oiii_hb'][good,2])
    xplot = pdata['nii_ha'][good,0]
    yplot = pdata['oiii_hb'][good,0]

    #### generate color mapping
    cpar_plot = np.array(pdata['model_pars'][colorpar][good])
    if log_cpar:
        cpar_plot = np.log10(cpar_plot)
    if cpar_range is not None:
        cpar_plot = np.clip(cpar_plot,cpar_range[0],cpar_range[1])

    #### plot photometry
    fig, ax = plt.subplots(1,1, figsize=(8, 6))

    ax.errorbar(xplot, yplot, yerr=yerr, xerr=xerr,
                fmt='o', ecolor='0.2', capthick=0.6,elinewidth=0.6,ms=0.0,alpha=0.5,zorder=-5)
    pts = ax.scatter(xplot[cidx], yplot[cidx], marker=popts['nofmir_shape'], c=cpar_plot[cidx], vmin=cpar_plot.min(), vmax=cpar_plot.max(),
                     cmap=plt.cm.plasma,s=75,zorder=10)
    pts = ax.scatter(xplot[~cidx], yplot[~cidx], marker=popts['fmir_shape'], c=cpar_plot[~cidx], vmin=cpar_plot.min(), vmax=cpar_plot.max(),
                     cmap=plt.cm.plasma,s=75,zorder=10)

    ax.set_xlabel(r'log([NII 6583]/H$_{\alpha}$)')
    ax.set_ylabel(r'log([OIII 5007]/H$_{\beta}$)')
    axlim = (-1.5,0.5,-0.8,1.0)
    ax.axis(axlim)

    #### label and add colorbar
    cb = fig.colorbar(pts, ax=ax, aspect=10)
    cb.set_label(colorparlabel)
    cb.solids.set_rasterized(True)
    cb.solids.set_edgecolor("face")
    return fig, ax















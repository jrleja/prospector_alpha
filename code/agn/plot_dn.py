import numpy as np
import os
import matplotlib.pyplot as plt
import magphys_plot_pref
from prosp_dutils import smooth_spectrum

def collate_data(alldata,alldata_noagn):

    out = {'dn':[],'dnup':[],'dndo':[],'spec':[],'obs':[]}
    out_noagn = {'dn':[],'dnup':[],'dndo':[],'spec':[]}
    objname = []
    for dat, dat_noagn in zip(alldata,alldata_noagn):
        out['dn'].append(float(dat['spec_info']['dn4000']['q50']))
        out['dnup'].append(float(dat['spec_info']['dn4000']['q84']))
        out['dndo'].append(float(dat['spec_info']['dn4000']['q16']))
        out_noagn['dn'].append(float(dat_noagn['spec_info']['dn4000']['q50']))
        out_noagn['dnup'].append(float(dat_noagn['spec_info']['dn4000']['q84']))
        out_noagn['dndo'].append(float(dat_noagn['spec_info']['dn4000']['q16']))


        out['spec'].append(dat['model_spec'])
        out_noagn['spec'].append(dat_noagn['model_spec'])
        if dat['residuals']['emlines'] is not None:
            out['obs'].append(dat['residuals']['emlines']['obs']['dn4000'])
        else:
            out['obs'].append(np.nan)
        objname.append(dat['objname'])
    return {'agn': out, 'noagn': out_noagn, 'wave': alldata[0]['model_spec_lam'], 'objname': objname}

def plot_dn(idx_plot=None,outfolder=None,
            runname='brownseds_agn',runname_noagn='brownseds_np',alldata=None,alldata_noagn=None, **popts):

    #### load alldata
    if alldata is None:
        import brown_io

        alldata = brown_io.load_alldata(runname=runname)
        alldata_noagn = brown_io.load_alldata(runname=runname_noagn)

    #### make output folder if necessary
    if outfolder is None:
        outfolder = os.getenv('APPS')+'/threedhst_bsfh/plots/'+runname+'/agn_plots/'
        if not os.path.isdir(outfolder):
            os.makedirs(outfolder)
    
    ### collate data
    pdata = collate_data(alldata,alldata_noagn)

    ### plot data
    fig, ax = plt.subplots(4,4, figsize=(14, 14))
    ax = np.ravel(ax)
    w = pdata['wave']
    for n,idx in enumerate(idx_plot):
        
        agn_spec = pdata['agn']['spec'][idx]
        noagn_spec = pdata['noagn']['spec'][idx]

        agn_spec = smooth_spectrum(w,agn_spec,200.0,minlam=3e3,maxlam=8e3)
        noagn_spec = smooth_spectrum(w,noagn_spec,200.0,minlam=3e3,maxlam=8e3)

        _ = oneplot(w,agn_spec,ax=ax[n],ytxt=0.05,color=popts['agn_color'])
        _ = oneplot(w,noagn_spec,ax=ax[n],ytxt=0.10,color=popts['noagn_color'])

        ax[n].text(0.95,0.9,pdata['objname'][idx],ha='right',fontsize=10,transform=ax[n].transAxes)

        ax[n].text(0.96,0.15, 'D$_{n}$4000 meas='+"{:.2f}".format(pdata['agn']['dn'][idx]) + \
                    '('+"{:.2f}".format(pdata['agn']['dndo'][idx])+','+"{:.2f}".format(pdata['agn']['dnup'][idx])+')', 
                    transform = ax[n].transAxes,ha='right', color=popts['agn_color'],fontsize=7)
        ax[n].text(0.96,0.2, 'D$_{n}$4000 meas='+"{:.2f}".format(pdata['noagn']['dn'][idx]) + \
                    '('+"{:.2f}".format(pdata['noagn']['dndo'][idx])+','+"{:.2f}".format(pdata['noagn']['dnup'][idx])+')', 
                    transform = ax[n].transAxes,ha='right', color=popts['noagn_color'],fontsize=7)
        ax[n].text(0.05,0.9, 'D$_{n}$4000 obs='+"{:.2f}".format(pdata['agn']['obs'][idx]), 
                    transform = ax[n].transAxes,fontsize=7)

        ylim = ax[n].get_ylim()
        ax[n].set_ylim(ylim[0]*0.5,ylim[1]*2)

    for i in xrange(n+1,ax.shape[0]): ax[i].axis('off')

    plt.tight_layout(w_pad=0.5,h_pad=0.3)
    plt.savefig(outfolder+'dn_comparison.png',dpi=200)
    plt.close()

def oneplot(lam,flux,ax=None,ytxt=0.05,color='k'):
    ''' defined as average flux ratio between
    [4050,4250] and [3750,3950] (Bruzual 1983; Hamilton 1985)
    blue: 3850-3950 . . . 4000-4100 (Balogh 1999)
    '''
    blue = (lam > 3850) & (lam < 3950)
    red  = (lam > 4000) & (lam < 4100)
    dn4000 = np.mean(flux[red])/np.mean(flux[blue])

    if ax is not None:
        plt_lam_idx = (lam > 3800) & (lam < 4150)

        ax.plot(lam[plt_lam_idx],flux[plt_lam_idx],color='black',drawstyle='steps-mid')
        ax.plot(lam[blue],flux[blue],color=color,drawstyle='steps-mid',lw=2)
        ax.plot(lam[red],flux[red],color=color,drawstyle='steps-mid',lw=2)

        ax.text(0.96,ytxt, 'D$_{n}$4000 spec='+"{:.2f}".format(dn4000), transform = ax.transAxes,ha='right', color=color,fontsize=8)

        ax.set_xlim(3800,4150)
        ax.yaxis.get_major_formatter().set_powerlimits((0, 1))

        [l.set_rotation(45) for l in ax.get_xticklabels()]


    return dn4000















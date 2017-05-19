import numpy as np
from prosp_diagnostic_plots import add_sfh_plot
import os
import matplotlib.pyplot as plt
import magphys_plot_pref
import matplotlib as mpl
from magphys_plots import median_by_band
import prosp_dutils
from astropy import constants
from matplotlib.ticker import MaxNLocator

'''
plot RMS for spectral quantities as a function of f_agn
'''

minorFormatter = magphys_plot_pref.jLogFormatter(base=10, labelOnlyBase=False)
majorFormatter = magphys_plot_pref.jLogFormatter(base=10, labelOnlyBase=True)

ebaropts = {
         'fmt':'o',
         'ecolor':'k',
         'capthick':0.4,
         'elinewidth':0.4,
         'alpha':0.55
        } 

def collate_data(alldata, alldata_noagn):

    '''
    return observed + model Halpha + Hbeta luminosities, Balmer decrement, and Dn4000, + errors
    also return f_agn
    '''

    ### package up information
    the_data = [alldata, alldata_noagn]
    data_label = ['agn','no_agn']
    output = {}
    for ii in xrange(2):

        ### build containers
        rms = {}
        labels = ['halpha','hbeta','bdec','dn4000']
        obs_eline_names = alldata[0]['residuals']['emlines']['em_name']
        mod_eline_names = alldata[0]['model_emline']['emnames']
        mod_extra_names = alldata[0]['pextras']['parnames']
        for d in ['obs','mod']:
            rms[d] = {}
            for l in labels: rms[d][l] = []

        #### model parameters
        objname = []
        model_pars = {}
        pnames = ['fagn', 'agn_tau']
        for p in pnames: 
            model_pars[p] = {'q50':[],'q84':[],'q16':[]}
        parnames = alldata[0]['pquantiles']['parnames']

        #### load model information
        for dat in the_data[ii]:

            #### model parameters [NEW MODEL ONLY]
            objname.append(dat['objname'])
            if data_label[ii] == 'agn':
                for key in model_pars.keys():
                    model_pars[key]['q50'].append(dat['pquantiles']['q50'][parnames==key][0])
                    model_pars[key]['q84'].append(dat['pquantiles']['q84'][parnames==key][0])
                    model_pars[key]['q16'].append(dat['pquantiles']['q16'][parnames==key][0])

            # if we don't measure spectral parameters, put a confusing filler value
            if dat['residuals']['emlines'] is None:
                rms['obs']['halpha'].append(np.array([np.nan]))
                rms['obs']['hbeta'].append(np.array([np.nan]))
                rms['obs']['bdec'].append(np.array([np.nan]))
                rms['obs']['dn4000'].append(np.array([np.nan]))

                rms['mod']['halpha'].append(np.array([np.nan]))
                rms['mod']['hbeta'].append(np.array([np.nan]))
                rms['mod']['bdec'].append(np.array([np.nan]))
                rms['mod']['dn4000'].append(np.array([np.nan]))
                continue

            #### pull the chain for spectral quantities
            rms['obs']['halpha'].append(np.log10(dat['residuals']['emlines']['obs']['lum_chain'][:,obs_eline_names=='H$\\alpha$'].squeeze() / constants.L_sun.cgs.value))
            rms['obs']['hbeta'].append(np.log10(dat['residuals']['emlines']['obs']['lum_chain'][:,obs_eline_names=='H$\\beta$'].squeeze() / constants.L_sun.cgs.value))
            rms['obs']['bdec'].append(prosp_dutils.bdec_to_ext(10**rms['obs']['halpha'][-1]/10**rms['obs']['hbeta'][-1]))
            rms['obs']['dn4000'].append(np.array(dat['residuals']['emlines']['obs']['dn4000']))

            rms['mod']['halpha'].append(np.log10(dat['model_emline']['flux']['chain'][:,mod_eline_names=='Halpha'].squeeze()))
            rms['mod']['hbeta'].append(np.log10(dat['model_emline']['flux']['chain'][:,mod_eline_names=='Hbeta'].squeeze()))
            rms['mod']['bdec'].append(prosp_dutils.bdec_to_ext(dat['pextras']['flatchain'][:,mod_extra_names=='bdec_calc'].squeeze()))
            rms['mod']['dn4000'].append(dat['spec_info']['dn4000']['chain'].squeeze())

        #### numpy arrays
        for key in model_pars.keys(): 
            for key2 in model_pars[key].keys():
                model_pars[key][key2] = np.array(model_pars[key][key2])

        output[data_label[ii]] = {}
        output[data_label[ii]]['objname'] = objname
        output[data_label[ii]]['model_pars'] = model_pars
        output[data_label[ii]]['rms'] = rms

    return output

def plot_comparison(runname='brownseds_agn',runname_noagn='brownseds_np',alldata=None,alldata_noagn=None,outfolder=None,idx=Ellipsis,**popts):

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
    ### choose galaxies with largest 10 F_AGN
    pdata = collate_data(alldata,alldata_noagn)

    plot_rms(pdata,outfolder,agn_idx=idx,**popts)

def drawArrow(A, B, ax, scale=1):
    ax.arrow(A[0], A[1], B[0] - A[0], B[1] - A[1],
              head_width=0.025*scale, width=0.005*scale,length_includes_head=True,color='grey',alpha=0.6)


def plot_rms(pdata,outfolder,agn_idx=None,**popts):

    #### plot geometry
    fig, ax = plt.subplots(2,2, figsize=(10, 10))
    fig2, ax2 = plt.subplots(2,2, figsize=(11, 11))

    ax = ax.ravel()
    ax2 = ax2.ravel()
    red = '#FF3D0D'
    blue = '#1C86EE' 

    ### titles
    trans = {
             'halpha': r'log(H$_{\alpha}$ flux)',
             'hbeta': r'log(H$_{\beta}$ flux)',
             'dn4000': r'D$_{\mathrm{n}}$4000',
             'bdec':r'log(F/F$_0$)$_{\mathrm{H}\alpha}$ - log(F/F$_0$)$_{\mathrm{H}\beta}$'
            }
    lims = [(-2,2), (-2,2), (-0.3,0.3), (-0.15,0.15)]
    lims2 = [(6,9.3),(5,8.5),(-0.1,0.6),(1.0,1.45)]
    sn_cut = [0.2, 0.2, 0.1, 0.1]

    ### for each observable, pull out RMS
    ndraw = int(1e5)
    fagn = np.log10(pdata['agn']['model_pars']['fagn']['q50'])
    ordered_keys = ['halpha','hbeta','bdec','dn4000']
    for ii,key in enumerate(ordered_keys):
        ### for each galaxy
        q50, q84, q16 = [], [], []
        obs_agn, obs_noagn, mod_agn, mod_noagn = [], [], [], []
        ngal = len(pdata['no_agn']['rms']['mod'][key])
        for jj in xrange(ngal):
            obs_agn.append(np.random.choice(np.atleast_1d(pdata['agn']['rms']['obs'][key][jj]),size=ndraw))
            mod_agn.append(np.random.choice(pdata['agn']['rms']['mod'][key][jj],size=ndraw))
            rms_agn = np.sqrt((obs_agn[-1]-mod_agn[-1])**2)

            obs_noagn.append(np.random.choice(np.atleast_1d(pdata['no_agn']['rms']['obs'][key][jj]),size=ndraw))
            mod_noagn.append(np.random.choice(pdata['no_agn']['rms']['mod'][key][jj],size=ndraw))
            rms_noagn = np.sqrt((obs_noagn[-1]-mod_noagn[-1])**2)

            ### cut on errors
            diff = rms_noagn-rms_agn
            if (np.isnan(diff).sum() > diff.shape[0]/6.):
                cent,up,down = np.nan,np.nan,np.nan
            else:
                cent,up,down = np.nanpercentile(diff,[50.0,84.0,16.0])
            q50.append(cent)
            q84.append(up)
            q16.append(down)
        
        q50, q84, q16 = np.array(q50), np.array(q84), np.array(q16)
        idx = (np.isfinite(q50))

        ### plot
        errs = prosp_dutils.asym_errors(q50[idx], q84[idx], q16[idx])
        ax[ii].errorbar(fagn[idx],q50[idx],yerr=errs,fmt='o', ecolor=blue, capthick=1,elinewidth=1,ms=8,alpha=0.5,zorder=-5)
        ax[ii].set_title(trans[key])
        ax[ii].set_xlabel(r'log(f$_{\mathrm{MIR}}$)')
        ax[ii].set_ylabel(r'RMS(NO AGN) - RMS(AGN)')
        ax[ii].axhline(0, linestyle='--', color='0.2')
        ax[ii].xaxis.set_major_locator(MaxNLocator(5))

        ### dynamic ylimits
        #ymax = np.abs(np.concatenate((q16[idx],q84[idx]))).max()*1.05
        #ax[ii].set_ylim(-ymax,ymax)
        ax[ii].set_ylim(lims[ii][0],lims[ii][1])

        ### running median
        x, y = prosp_dutils.running_median(fagn[idx],q50[idx],nbins=9,weights=2./(q84[idx]-q16[idx])**2,avg=True)
        ax[ii].plot(x,y,color=red,lw=4,alpha=0.6)
        ax[ii].plot(x,y,color=red,lw=4,alpha=0.6)

        ####### observed vs model, AGN only, with arrows
        q16o, q50o, q84o, q16m, q50m, q84m, q16o_no, q50o_no, q84o_no, q16m_no, q50m_no, q84m_no = [np.zeros(agn_idx.shape[0]) for i in range(12)]
        for jj in xrange(agn_idx.shape[0]):
            q50o[jj],q84o[jj],q16o[jj] = np.nanpercentile(obs_agn[agn_idx[jj]],[50.0,84.0,16.0])
            q50m[jj],q84m[jj],q16m[jj] = np.nanpercentile(mod_agn[agn_idx[jj]],[50.0,84.0,16.0])
            q50o_no[jj],q84o_no[jj],q16o_no[jj] = np.nanpercentile(obs_noagn[agn_idx[jj]],[50.0,84.0,16.0])
            q50m_no[jj],q84m_no[jj],q16m_no[jj] = np.nanpercentile(mod_noagn[agn_idx[jj]],[50.0,84.0,16.0])

        good = ((np.isfinite(q50o)) & (np.isfinite(q50m)) & (np.isfinite(q50o_no)) & (np.isfinite(q50m_no)) &
               ((q84o-q16o)/2. < sn_cut[ii]))

        alpha = 0.7
        ax2[ii].scatter(q50o_no[good], q50m_no[good], marker='o', color=popts['noagn_color'],s=70,zorder=10,alpha=alpha,edgecolors='k')
        ax2[ii].scatter(q50o[good], q50m[good], marker='o', color=popts['agn_color'],s=70,zorder=10,alpha=alpha,edgecolors='k')
        
        errs_obs = prosp_dutils.asym_errors(q50o_no[good], q84o_no[good], q16o_no[good])
        errs_mod = prosp_dutils.asym_errors(q50m_no[good], q84m_no[good], q16m_no[good])

        ax2[ii].errorbar(q50o_no[good], q50m_no[good], 
                         yerr=errs_mod, xerr=errs_obs,
                         fmt='o', ecolor='0.5', capthick=0.5,elinewidth=0.5,
                         ms=0,alpha=0.4,zorder=-5)

        errs_obs = prosp_dutils.asym_errors(q50o[good], q84o[good], q16o[good])
        errs_mod = prosp_dutils.asym_errors(q50m[good], q84m[good], q16m[good])

        ax2[ii].errorbar(q50o[good], q50m[good], 
                         yerr=errs_mod, xerr=errs_obs,
                         fmt='o', ecolor='0.5', capthick=0.5,elinewidth=0.5,
                         ms=0,alpha=0.4,zorder=-5)

        ### draw in some SICK arrows
        for kk in xrange(len(good)):
            if not good[kk]:
                continue
            old = (q50o_no[kk],q50m_no[kk])
            new = (q50o[kk],q50m[kk])
            drawArrow(old,new,ax2[ii],scale=lims2[ii][1]-lims2[ii][0])

        ### labels and lines
        ax2[ii].set_xlabel('observed '+trans[key])
        ax2[ii].set_ylabel('model '+trans[key])
        ax2[ii].xaxis.set_major_locator(MaxNLocator(5))
        ax2[ii].yaxis.set_major_locator(MaxNLocator(5))
        ax2[ii].set_xlim(lims2[ii][0],lims2[ii][1])
        ax2[ii].set_ylim(lims2[ii][0],lims2[ii][1])
        ax2[ii].plot(lims2[ii],lims2[ii],'--',color='0.5',alpha=0.5,zorder=-15)

        off_agn,scat_agn = prosp_dutils.offset_and_scatter(q50o[good],q50m[good],biweight=True)
        off_noagn,scat_noagn = prosp_dutils.offset_and_scatter(q50o_no[good],q50m_no[good],biweight=True)
        topts = {'transform':ax2[ii].transAxes,'fontsize':13,'verticalalignment':'top'}
        ax2[ii].text(0.04,0.95,'AGN-off', color=popts['noagn_color'],weight='bold', **topts)
        ax2[ii].text(0.04,0.9,'scatter, offset=', color=popts['noagn_color'], **topts)
        ax2[ii].text(0.04,0.85,"{:.2f}".format(scat_noagn)+", "+"{:.2f}".format(off_noagn), color=popts['noagn_color'], **topts)
        ax2[ii].text(0.04,0.77,'AGN-on', color=popts['agn_color'], weight='semibold', **topts)
        ax2[ii].text(0.04,0.72,'scatter, offset=', color=popts['agn_color'], **topts)
        ax2[ii].text(0.04,0.67,"{:.2f}".format(scat_agn)+", "+"{:.2f}".format(off_agn), color=popts['agn_color'], **topts)

        print key
        print 'change in observables:'
        for k,idx in enumerate(agn_idx): print '\t'+pdata['agn']['objname'][idx]+': '+"{:.2f}".format(q50o_no[k]-q50o[k])

    fig.tight_layout()
    fig.savefig(outfolder+'delta_spec_rms.png',dpi=120)

    fig2.tight_layout()
    fig2.savefig(outfolder+'obs_vs_mod_arrows.png',dpi=120)

    plt.close()


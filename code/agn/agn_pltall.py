import optical_color_color,bpt,plot_delta_pars,agn_residuals,xray_luminosity,plot_spec_rms,wise_colors
import wise_gradients, plot_dn, agn_evidence_hist, sf_seq
import delta_mass_met,os,prospector_io
import numpy as np
import matplotlib.pyplot as plt
from astropy import constants
from matplotlib.ticker import MaxNLocator

def plot(runname='brownseds_agn',runname_noagn='brownseds_np',
         alldata=None,alldata_noagn=None,open_all=True,outfolder=None,fmir=True):

    #### load alldata
    if alldata is None:
        alldata = prospector_io.load_alldata(runname=runname)
    if alldata_noagn is None:
        alldata_noagn = prospector_io.load_alldata(runname=runname_noagn)

    #### outfolder
    outfolder = os.getenv('APPS')+'/prospector_alpha/plots/'+runname+'/agn_plots/'
    if not os.path.isdir(outfolder):
        os.makedirs(outfolder)

    from copy import deepcopy
    if fmir:
        alldata_sub = deepcopy(alldata)
        for i,dat in enumerate(alldata_sub):
            fidx = dat['pquantiles']['parnames'] == 'fagn'
            midx = dat['pextras']['parnames'] == 'fmir'
            lidx = dat['pextras']['parnames'] == 'l_agn'

            ### FAGN ---> FMIR
            dat['pquantiles']['q50'][fidx] = dat['pextras']['q50'][midx].squeeze()
            dat['pquantiles']['q84'][fidx] = dat['pextras']['q84'][midx].squeeze()
            dat['pquantiles']['q16'][fidx] = dat['pextras']['q16'][midx].squeeze()
            dat['pquantiles']['sample_chain'][:,fidx] = dat['pextras']['flatchain'][:,midx]

    else:
        alldata_sub = deepcopy(alldata)

    ### what are we calling "AGN" ?
    #twosigma_fmir = np.array([np.percentile(dat['pquantiles']['sample_chain'][:,fidx].squeeze(),16) for dat in alldata_sub])
    #agn_idx = np.where(twosigma_fmir > 0.05)[0]
    twosigma_fmir = np.array([np.percentile(dat['pquantiles']['sample_chain'][:,fidx].squeeze(),50) for dat in alldata_sub])
    agn_idx = np.where(twosigma_fmir > 0.1)[0]


    ### resort from largest to smallest
    fmir = np.array([dat['pquantiles']['q50'][fidx][0] for dat in alldata_sub])
    agn_idx = agn_idx[fmir[agn_idx].argsort()]

    print 'the following are identified AGN: '
    names = [alldata[idx]['objname'] for idx in agn_idx]
    print ', '.join(names)

    print fmir[agn_idx]

    ### global plot color scheme
    popts = {
            'agn_color': '#9400D3',
            'noagn_color': '#FF420E',
            'fmir_shape': '^',
            'nofmir_shape': 'o',
            'fmir_alpha': 0.8,
            'nofmir_alpha': 0.4,
            'fmir_color': 'blue',
            'nofmir_color': '0.4',
            'cmap': plt.cm.plasma
            }

    #### PLOT ALL
    '''
    plot_dn.plot_dn(idx_plot=agn_idx,runname=runname,runname_noagn=runname_noagn,
                    alldata=alldata_sub,alldata_noagn=alldata_noagn,outfolder=outfolder,**popts)
    '''
    agn_evidence = {}
    print 'PLOTTING MASS-METALLICITY DIAGRAM'
    delta_mass_met.plot_comparison(runname=runname,alldata=alldata_sub,alldata_noagn=alldata_noagn,outfolder=outfolder,plt_idx=agn_idx,**popts)  
    print 1/0
    print 'PLOTTING WISE GRADIENTS'
    agn_evidence = wise_gradients.plot_all(agn_evidence,runname=runname,runname_noagn=runname_noagn,alldata=alldata_sub,
                                           alldata_noagn=alldata_noagn,agn_idx=agn_idx,regenerate=False,outfolder=outfolder, **popts)
    print 'PLOTTING XRAY LUMINOSITY'
    agn_evidence = xray_luminosity.make_plot(agn_evidence,runname=runname,alldata=alldata_sub,outfolder=outfolder,idx=agn_idx,**popts)
    print 'PLOTTING BPT DIAGRAM'
    agn_evidence = bpt.plot_bpt(agn_evidence,runname=runname,alldata=alldata_sub,outfolder=outfolder,idx=agn_idx,**popts)
    print 'PLOTTING AGN EVIDENCE HISTOGRAM'
    agn_evidence_hist.plot(agn_evidence,alldata, outfolder, agn_idx=agn_idx, **popts)
    print 'PLOTTING WISE COLORS'
    wise_colors.plot_mir_colors(runname=runname,alldata=alldata_sub,outfolder=outfolder,idx=agn_idx,**popts)
    print 'PLOTTING DELTA OBSERVABLES'
    plot_spec_rms.plot_comparison(runname=runname,alldata=alldata_sub,alldata_noagn=alldata_noagn,outfolder=outfolder,idx=agn_idx,**popts)
    print 'PLOTTING DELTA PARS'
    plot_delta_pars.plot(runname=runname,runname_noagn=runname_noagn,alldata=alldata_sub,alldata_noagn=alldata_noagn,outfolder=outfolder,idx=agn_idx,**popts)
    print 'PLOTTING AGN RESIDUALS'
    agn_residuals.plot_comparison(idx_plot=agn_idx,runname=runname,runname_noagn=runname_noagn,
                                        alldata=alldata_sub,alldata_noagn=alldata_noagn,outfolder=outfolder,**popts)
    print 'PLOTTING SF SEQ'
    sf_seq.plot(runname=runname,alldata=alldata_sub,outfolder=outfolder,**popts) 
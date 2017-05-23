import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def assemble_flags(edict):
    '''return flags based on binary EVIDENCE or NO EVIDENCE
    '''

    agn_flag = np.zeros(129,dtype=bool)
    has_measurement = np.zeros(129,dtype=bool)

    ### BPT AGN
    # AGN + COMPOSITE (for now)
    good = np.where( ((edict['bpt_type'] == 'agn') | (edict['bpt_type'] == 'compo')) & edict['bpt_use_flag'])[0]
    #good = np.where( (edict['bpt_type'] == 'agn') & edict['bpt_use_flag'])[0]
    agn_flag[good] = True

    ### WISE GRADIENT AGN
    good = np.where( (edict['wise_gradient'] < -.15) & edict['wise_gradient_flag'])[0]
    agn_flag[good] = True

    ### XRAY AGN
    good = np.where( edict['xray_luminosity'] > 1e42)[0]
    agn_flag[good] = True
    
    ### WHAT HAS A MEASUREMENT
    flag = (edict['xray_luminosity'] > 0) & (edict['wise_gradient_flag']) & (edict['bpt_use_flag'])
    has_measurement[flag] = True

    return agn_flag, has_measurement

def plot(agn_evidence, alldata, outfolder,**popts):

    ### get flag
    flag, measure_flag = assemble_flags(agn_evidence)
    objname = np.array([dat['objname'] for dat in alldata])

    ### get fmir
    fmir_idx = alldata[0]['pextras']['parnames'] == 'fmir'
    fmir = np.log10([dat['pextras']['q50'][fmir_idx][0] for dat in alldata])

    ### split into groups
    fmir_all = fmir[(flag) | (measure_flag)]
    fmir_agn = fmir[flag]
    fmir_noagn = fmir[(measure_flag) & (~flag)]

    ### make histogram
    fig, ax = plt.subplots(1,1, figsize=(7, 7))
    nbins = 8
    alpha = 0.7
    lw = 3.5

    hist, bins = np.histogram(fmir_all,bins=nbins,density=False) # get the bins for all
    hist_agn, _ = np.histogram(fmir_agn,bins=bins,density=False)
    hist_noagn, _ = np.histogram(fmir_noagn,bins=bins,density=False)
    
    ### build plottable histograms
    # make bins into xplot
    # add 0-padding to bins and data
    delta_bin = bins[1]-bins[0]
    bins_mid = (bins[1:]+bins[:-1])/2.
    bins_mid = np.array([[bins_mid[0]-delta_bin]+bins_mid.tolist()+[bins_mid[-1]+delta_bin]]).squeeze()
    hist_agn = np.array([0]+hist_agn.tolist()+[0])
    hist_noagn = np.array([0]+hist_noagn.tolist()+[0])

    ### plot
    ax.plot(bins_mid,hist_agn,color=popts['agn_color'],drawstyle='steps-mid',alpha=alpha,lw=lw)
    ax.plot(bins_mid,hist_noagn,color=popts['noagn_color'],drawstyle='steps-mid',alpha=alpha,lw=lw)

    ax.set_ylabel('N')
    ax.set_xlabel(r'log(f$_{\mathrm{MIR}}$)')
    ax.set_ylim(0,ax.get_ylim()[1]+3)
    plt.show()
    print 1/0






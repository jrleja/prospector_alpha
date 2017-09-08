import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from astropy.table import Table

def assemble_flags(edict, composite=True):
    '''return flags based on binary EVIDENCE or NO EVIDENCE
    '''

    agn_flag = np.zeros(129,dtype=bool)
    has_measurement = np.zeros(129,dtype=bool)

    ### BPT AGN
    # AGN + COMPOSITE (for now)
    if composite:
        good = np.where( ((edict['bpt_type'] == 'AGN') | (edict['bpt_type'] == 'composite')) & edict['bpt_use_flag'])[0]
    else:
        good = np.where( (edict['bpt_type'] == 'AGN') & edict['bpt_use_flag'])[0]
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

def plot(agn_evidence, alldata, outfolder,agn_idx=None,**popts):

    ### get flag
    flag, measure_flag = assemble_flags(agn_evidence, composite=False)
    compflag_agn, _ = assemble_flags(agn_evidence, composite=True)
    objname = np.array([dat['objname'] for dat in alldata])

    ### get fmir
    fmir_idx = alldata[0]['pextras']['parnames'] == 'fmir'
    fmir = np.log10([dat['pextras']['q50'][fmir_idx][0] for dat in alldata])

    agn_evidence['fmir'] = fmir
    agn_evidence['fmir_up'] = np.log10([dat['pextras']['q84'][fmir_idx][0] for dat in alldata]) - fmir
    agn_evidence['fmir_down'] = fmir - np.log10([dat['pextras']['q16'][fmir_idx][0] for dat in alldata])

    ### create master histogram, plus bins
    nbins = 6
    all_flag = (flag) | (measure_flag)
    hist, bins = np.histogram(fmir[all_flag],bins=nbins,density=False) # get the bins for all
    delta_bin = bins[1]-bins[0]
    bins_mid = (bins[1:]+bins[:-1])/2.
    bins_mid = np.array([[bins_mid[0]-delta_bin]+bins_mid.tolist()+[bins_mid[-1]+delta_bin]]).squeeze()

    ### split into groups
    agn_flag = flag
    noagn_flag = ((measure_flag) & (~flag))# & (~compflag_agn))
    #comp_flag = ((measure_flag) & (~flag) & (compflag_agn))
    flags = [agn_flag, noagn_flag]#, comp_flag]
    colors = [popts['agn_color'], popts['noagn_color']]#, 'orange']
    labels = ['evidence for AGN', 'no evidence for AGN']#, 'BPT composite']
    linestyles = ['-',':']

    ### make histogram
    fig, ax = plt.subplots(1,1, figsize=(6.5, 6.5))
    alpha = 0.7
    lw = 3.5
    for label, color, flag, ls in zip(labels,colors,flags,linestyles):
        hist_flag, _ = np.histogram(fmir[flag],bins=bins,density=False)
        hist = np.array([0]+hist_flag.tolist()+[0])
        ax.plot(bins_mid,hist,color=color,drawstyle='steps-mid',alpha=alpha,lw=lw, label=label,linestyle=ls)

    ax.set_ylabel('N')
    ax.set_xlabel(r'log(f$_{\mathrm{AGN,MIR}}$)')
    ax.set_ylim(0,ax.get_ylim()[1]+3)

    ax.legend(loc=0,prop={'size':15},frameon=False)

    plt.tight_layout()
    plt.savefig(outfolder+'agn_evidence_histogram.png',dpi=150)
    plt.close()

    output_table(agn_evidence, agn_flag, noagn_flag, agn_idx)

def format_output(i,data,name,edict,idx):

    if data[i][name] == -99.0:
        return '---'

    elif name == r'log(f$_{\mathrm{MIR,AGN}}$)':
        fmt = "{{0:{0}}}".format(".2f").format
        title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
        return title.format(fmt(data[i][name]), fmt(edict['fmir_up'][idx][i]), fmt(edict['fmir_down'][idx][i]))

    elif name == r'log(L$_X$)':
        return "{:.2f}".format(np.log10(data[i][name]))

    elif name == r'$\nabla$(W1-W2)':
        fmt = "{{0:{0}}}".format(".4f").format
        title = r"${{{0}}}\pm{{{1}}}$"
        return title.format(fmt(data[i][name]), fmt(edict['wise_gradient_err'][idx][i]))

    else:
        return data[i][name]

def output_table(edict, agn_flag, noagn_flag,agn_idx):

    outtable = '/Users/joel/my_papers/agn_dust/table.tex'

    ### load literature notes
    loc = '/Users/joel/my_papers/agn_dust/agn_literature.txt'
    litdat = np.loadtxt(loc, comments = '#', delimiter='|', dtype = {'names':(['name','notes']),'formats':(np.concatenate((np.atleast_1d('S20'),np.atleast_1d('S200'))))})
    litdat['name'] = np.char.strip(litdat['name'])

    ### what goes in the table?
    # anything with (A POSITIVE AGN INDICATOR) | (THREE NEGATIVE AGN INDICATORS) | (PROSPECTOR AGN FLAG)
    idx = np.where(agn_flag | noagn_flag)[0]
    idx = np.unique(np.concatenate((idx,agn_idx))) # add in Prospector-AGN
    idx = idx[edict['fmir'][idx].argsort()][::-1] # sort by decreasing F_MIR
    ngal = idx.shape[0]

    ### generate data
    ordered_namelist = ('Name', r'log(f$_{\mathrm{MIR,AGN}}$)', 'BPT', r'$\nabla$(W1-W2)', r'log(L$_X$)', 'Literature Notes')
    data = Table({
                  'Name':  [str(np.array(edict['objname'])[idx][i]) for i in xrange(ngal)],
                  r'log(f$_{\mathrm{MIR,AGN}}$)': [edict['fmir'][idx][i] for i in xrange(ngal)],
                  'BPT': [edict['bpt_type'][idx][i] if edict['bpt_use_flag'][idx][i] else '---' for i in xrange(ngal)],
                  r'$\nabla$(W1-W2)': [edict['wise_gradient'][idx][i] if edict['wise_gradient_flag'][idx][i] else -99.0 for i in xrange(ngal)],
                  r'log(L$_X$)': [edict['xray_luminosity'][idx][i] if edict['xray_luminosity'][idx][i] > 0 else -99.0 for i in xrange(ngal)],
                  r'Literature Notes': [litdat['notes'][litdat['name'] == np.array(edict['objname'])[idx][i]][0] if np.array(edict['objname'])[idx][i] in litdat['name'] else '---' for i in xrange(ngal)]
                })
    data = data[ordered_namelist]

    units = {
             r'log(L$_X$)': 'erg/s',
             r'$\nabla$(W1-W2)': r"mag/kpc",
             'Name': ' ',
             'BPT': ' ',
             r'log(f$_{\mathrm{MIR,AGN}}$)': ' ',
             r'Literature Notes': ' '
            }

    with open(outtable, 'w') as f:
        f.write('\\begin{deluxetable*}{cccccp{40mm}}\n')
        f.write('\\begin{center}\n')
        f.write('\\tablecaption{Summary of AGN Evidence}\n')
        f.write('\\tablehead{' + "& ".join(["\colhead{"+name+"} " for name in ordered_namelist])+ '\\\\ ')
        f.write(" & ".join(["\colhead{"+units[name]+"} " for name in ordered_namelist])+ '} \n')
        f.write('\\startdata\n')
        for i in xrange(ngal):
            f.write(" & ".join([format_output(i,data,name,edict,idx) for name in ordered_namelist]) + '\\\\\n')
        f.write('\\enddata\n')
        f.write('\\end{center}\n')
        f.write('\\end{deluxetable*}')

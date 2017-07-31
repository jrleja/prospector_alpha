import copy
import matplotlib.pyplot as plt
import agn_plot_pref
import numpy as np
import matplotlib as mpl
import os
import observe_agn_templates
from scipy.spatial import ConvexHull
from prosp_dutils import asym_errors
import pickle
from matplotlib.patches import Polygon

np.random.seed(2)
blue = '#1C86EE' 
dpi = 150

def vega_conversions(fname):

    # mVega = mAB-delta_m
    # Table 5, http://wise2.ipac.caltech.edu/docs/release/prelim/expsup/sec4_3g.html#PhotometricZP
    if fname.lower().replace(' ','_')=='wise_w1':
        return -2.683
    if fname.lower().replace(' ','_')=='wise_w2':
        return -3.319
    if fname.lower().replace(' ','_')=='wise_w3':
        return -5.242
    if fname.lower().replace(' ','_')=='wise_w4':
        return -6.604

def collate_data(alldata):

    #### generate containers
    # photometry
    obs_phot, model_phot = {}, {}
    filters = ['wise_w1', 'wise_w2', 'wise_w3', 'wise_w4',
               'spitzer_irac_ch1','spitzer_irac_ch2','spitzer_irac_ch3','spitzer_irac_ch4']

    for f in filters: 
        obs_phot[f] = []
        model_phot[f] = []

    # model parameters
    objname = []
    model_pars = {}
    pnames = ['fagn', 'agn_tau','duste_qpah']
    for p in pnames: 
        model_pars[p] = {'q50':[],'q84':[],'q16':[]}
    parnames = alldata[0]['pquantiles']['parnames']

    #### load information
    for dat in alldata:
        objname.append(dat['objname'])

        #### model parameters
        for key in model_pars.keys():
            try:
                model_pars[key]['q50'].append(dat['pquantiles']['q50'][parnames==key][0])
                model_pars[key]['q84'].append(dat['pquantiles']['q84'][parnames==key][0])
                model_pars[key]['q16'].append(dat['pquantiles']['q16'][parnames==key][0])

            except IndexError:
                print key + ' not in model!'
                continue
        
        #### photometry
        for key in obs_phot.keys():
            match = dat['filters'] == key
            if match.sum() == 1:
                obs_phot[key].append(dat['obs_maggies'][match][0])
                model_phot[key].append(np.median(dat['model_maggies'][match]))
            elif match.sum() == 0:
                obs_phot[key].append(0)
                model_phot[key].append(0)

    #### numpy arrays
    for key in obs_phot.keys(): obs_phot[key] = np.array(obs_phot[key])
    for key in model_phot.keys(): model_phot[key] = np.array(model_phot[key])
    for key in model_pars.keys(): 
        for key2 in model_pars[key].keys():
            model_pars[key][key2] = np.array(model_pars[key][key2])

    out = {}
    out['model_phot'] = model_phot
    out['obs_phot'] = obs_phot
    out['model_pars'] = model_pars
    out['objname'] = objname
    return out

def plot_mir_colors(runname='brownseds_agn',alldata=None,outfolder=None, vega=True, idx=None, **opts):

    #### load alldata
    if alldata is None:
        import brown_io
        alldata = brown_io.load_alldata(runname=runname)

    #### make output folder if necessary
    if outfolder is None:
        outfolder = os.getenv('APPS')+'/prospector_alpha/plots/'+runname+'/agn_plots/'
        if not os.path.isdir(outfolder):
            os.makedirs(outfolder)

    #### collate data
    pdata = collate_data(alldata)

    #### magnitude system?
    system = '(AB)'
    if vega:
        system = '(Vega)'

    #### colored scatterplots
    ### IRAC
    cpar_range = [-2,0]
    xfilt, yfilt = ['spitzer_irac_ch3','spitzer_irac_ch4'], ['spitzer_irac_ch1','spitzer_irac_ch2']
    fig,ax = plot_color_scatterplot(pdata,xfilt=xfilt,yfilt=yfilt,
                           xlabel='IRAC [5.8]-[8.0] (AB)', ylabel='IRAC [3.6]-[4.5] (AB)',
                           colorpar='fagn',colorparlabel=r'log(f$_{\mathrm{AGN,MIR}}$)',log_cpar=True, cpar_range=cpar_range,
                           idx=idx,**opts)
    plot_nenkova_templates(ax, xfilt=xfilt,yfilt=yfilt)
    plt.savefig(outfolder+'irac_colors.png',dpi=dpi)
    plt.close()

    ### WISE hot
    xfilt, yfilt = ['wise_w2','wise_w3'], ['wise_w1','wise_w2']
    fig, ax = plot_color_scatterplot(pdata,xfilt=xfilt,yfilt=yfilt,
                                     xlabel='WISE [4.6]-[12] '+system,ylabel='WISE [3.4]-[4.6] '+system,
                                     colorpar='fagn',colorparlabel=r'log(f$_{\mathrm{AGN,MIR}}$)',
                                     log_cpar=True, cpar_range=cpar_range,vega=vega,
                                     idx=idx,**opts)
    plot_nenkova_templates(ax, xfilt=xfilt,yfilt=yfilt,vega=vega)

    plot_prospector_templates(ax, xfilt=xfilt,yfilt=yfilt,outfolder=outfolder,vega=vega)
    outstring = 'wise_hotcolors'
    if vega:
        outstring += '_vega'
    plot_stern_cuts(ax)
    plt.savefig(outfolder+outstring+'.png',dpi=dpi)
    plt.close()

    plot_color_vs_fmir(pdata,xfilt=yfilt,xlabel='WISE [3.4]-[4.6] '+system,
                           outname=outfolder+'w1_w2_fmir.png', vega=True, **opts)

    ### WISE warm
    xfilt, yfilt = ['wise_w2','wise_w3'],['wise_w3','wise_w4']
    outstring = 'wise_warmcolors'
    if vega:
        outstring += '_vega'

    fig,ax = plot_color_scatterplot(pdata,xfilt=xfilt,yfilt=yfilt,
                           xlabel='WISE [4.6]-[12] '+system,ylabel='WISE [12]-[22] '+system,
                           colorpar='fagn',colorparlabel=r'log(f$_{\mathrm{AGN,MIR}}$)',log_cpar=True, cpar_range=cpar_range,vega=vega,
                           idx=idx,**opts)
    plot_nenkova_templates(ax, xfilt=xfilt,yfilt=yfilt,vega=vega)

    plt.savefig(outfolder+outstring+'.png',dpi=dpi)
    plt.close()

def plot_nenkova_templates(ax, xfilt=None,yfilt=None,vega=False):

    modcolor = '0.3'

    filts = xfilt + yfilt
    templates = observe_agn_templates.observe(filts)

    xp, yp = [], []
    for key in templates.keys():
        xp.append(templates[key][0] - templates[key][1])
        yp.append(templates[key][2]-templates[key][3])

    ### convert to vega magnitudes
    if vega:
        xp = np.array(xp) + vega_conversions(xfilt[0]) - vega_conversions(xfilt[1])
        yp = np.array(yp) + vega_conversions(yfilt[0]) - vega_conversions(yfilt[1])

    yp = np.array(yp)[np.array(xp).argsort()]
    xp = np.array(xp)
    xp.sort()

    ax.plot(xp,yp,'o',alpha=0.8,ms=11,color=modcolor,mew=2.2)
    ax.plot(xp,yp,' ',alpha=0.6,color=modcolor, linestyle='-',lw=3)

    ax.text(0.05,0.89,'Nenkova+08 \nAGN Templates',transform=ax.transAxes,fontsize=13,color=modcolor, weight='semibold')

def plot_stern_cuts(ax):
    xlim = ax.get_xlim()
    print xlim
    ax.axhline(0.8, linestyle='--', color='k',lw=2,zorder=-1)
    ax.text(xlim[0]+0.2, 0.84, 'Stern+12 AGN cut', fontsize=13, weight='semibold', color='k')

def plot_prospector_templates(ax, xfilt=None, yfilt=None, outfolder=None, vega=False,multiple=True):

    '''
    strings = '' is continuous SFH
    _sfrX is all SFR in bin X
    '''

    prospcolor = ['#c264ff','#1C86EE','#ff4c4c']

    xp, yp = [], []
    strings = ['_sfr1','_sfr2','_sfr6']
    label = ['young stars','~0.5 Gyr stars', 'old stars']
    for string in strings:
        prosp = load_dl07_models(outfolder=outfolder,string=string)
        if multiple:
            xp.append(prosp[" ".join(xfilt)])
            yp.append(prosp[" ".join(yfilt)])
        else:
            xp += prosp[" ".join(xfilt)]
            yp += prosp[" ".join(yfilt)]
    
    i = 0
    for x,y in zip(xp, yp):
        points = np.array([x,y]).transpose()
        
        ### convert to vega magnitudes
        if vega:
            points[:,0] += vega_conversions(xfilt[0]) - vega_conversions(xfilt[1])
            points[:,1] += vega_conversions(yfilt[0]) - vega_conversions(yfilt[1])

        hull = ConvexHull(points)

        cent = np.mean(points, 0)
        pts = []
        for pt in points[hull.simplices]:
            pts.append(pt[0].tolist())
            pts.append(pt[1].tolist())

        pts.sort(key=lambda p: np.arctan2(p[1] - cent[1],
                                        p[0] - cent[0]))
        pts = pts[0::2]  # Deleting duplicates
        pts.insert(len(pts), pts[0])

        poly = Polygon((np.array(pts)- cent) + cent,
                       facecolor=prospcolor[i], alpha=0.45,zorder=-35)
        poly.set_capstyle('round')
        plt.gca().add_patch(poly)

        ax.text(0.05,0.805-0.09*i,'Prospector-$\\alpha$ \n'+label[i],transform=ax.transAxes,fontsize=13,color=prospcolor[i],weight='semibold')
        i += 1

def plot_color_vs_fmir(pdata,xfilt=None,xlabel=None,
                       outname=None, vega=True, idx=None, **popts):


    #### only select those with good photometry
    good = (pdata['obs_phot'][xfilt[0]] != 0) & \
           (pdata['obs_phot'][xfilt[1]] != 0)

    xplot = -2.5*np.log10(pdata['obs_phot'][xfilt[0]][good])+2.5*np.log10(pdata['obs_phot'][xfilt[1]][good])
    
    ### convert to vega magnitudes
    if vega:
        xplot += vega_conversions(xfilt[0]) - vega_conversions(xfilt[1])

    ### make that plot
    fig, ax = plt.subplots(1,1,figsize=(7,7))

    yerr = asym_errors(pdata['model_pars']['fagn']['q50'][good],
                       pdata['model_pars']['fagn']['q84'][good],
                       pdata['model_pars']['fagn']['q16'][good],log=True)

    ax.scatter(xplot, np.log10(pdata['model_pars']['fagn']['q50'][good]), 
               marker='o', color=blue,s=45,zorder=11,alpha=0.9,edgecolors='k')
    ax.errorbar(xplot, np.log10(pdata['model_pars']['fagn']['q50'][good]), 
                yerr=yerr, zorder=-5,ms=0.0, ecolor='k', linestyle=' ',
                capthick=0.8, elinewidth=0.4, alpha=0.7)

    #### label and add colorbar
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r'log(f$_{\mathrm{AGN,MIR}}$)')

    plt.savefig(outname, dpi=150)
    plt.close()

def plot_color_scatterplot(pdata,xfilt=None,yfilt=None,xlabel=None,ylabel=None,
                           colorpar=None,colorparlabel=None,log_cpar=False,cpar_range=None,
                           outname=None, vega=False, idx=None, **popts):
    '''
    plots a color-color scatterplot in AB magnitudes

    '''

    #### only select those with good photometry
    good = (pdata['obs_phot'][xfilt[0]] != 0) & \
           (pdata['obs_phot'][xfilt[1]] != 0) & \
           (pdata['obs_phot'][yfilt[0]] != 0) & \
           (pdata['obs_phot'][yfilt[1]] != 0)

    #### generate x, y values
    xplot = -2.5*np.log10(pdata['obs_phot'][xfilt[0]][good])+2.5*np.log10(pdata['obs_phot'][xfilt[1]][good])
    yplot = -2.5*np.log10(pdata['obs_phot'][yfilt[0]][good])+2.5*np.log10(pdata['obs_phot'][yfilt[1]][good])
    
    cidx = np.ones_like(good,dtype=bool)
    cidx[idx] = False
    cidx = cidx[good]

    ### convert to vega magnitudes
    if vega:
        xplot += vega_conversions(xfilt[0]) - vega_conversions(xfilt[1])
        yplot += vega_conversions(yfilt[0]) - vega_conversions(yfilt[1])

    #### generate color mapping
    cpar_plot = np.array(pdata['model_pars'][colorpar]['q50'][good])
    if log_cpar:
        cpar_plot = np.log10(cpar_plot)
    if cpar_range is not None:
        cpar_plot = np.clip(cpar_plot,cpar_range[0],cpar_range[1])

    #### plot photometry
    fig, ax = plt.subplots(1,1, figsize=(8, 6))
    ax.scatter(xplot[cidx], yplot[cidx], marker=popts['nofmir_shape'], c=cpar_plot[cidx],
               vmin=cpar_plot.min(), vmax=cpar_plot.max(), cmap=plt.cm.plasma,s=70,alpha=0.9)
    ax.scatter(xplot[~cidx], yplot[~cidx], marker=popts['fmir_shape'], c=cpar_plot[~cidx], 
               vmin=cpar_plot.min(), vmax=cpar_plot.max(), cmap=plt.cm.plasma,s=70,alpha=0.9)
    pts = ax.scatter(xplot, yplot, marker='o', c=cpar_plot, cmap=plt.cm.plasma,s=0.0,alpha=0.9)

    #pts = ax.scatter(xplot, yplot, marker='o', color='0.4',s=50,alpha=0.4, linewidths=1.4,edgecolors='black')
    #idx = (np.array(pdata['objname'])[good] == 'UGC 05101') | (np.array(pdata['objname'])[good] == 'IC 5298')
    #pts = ax.scatter(xplot[idx], yplot[idx], marker='o', color='red',s=80, linewidths=1.4,edgecolors='black')

    #### label and add colorbar
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    cb = fig.colorbar(pts, ax=ax, aspect=10)
    cb.set_label(colorparlabel)
    cb.solids.set_rasterized(True)
    cb.solids.set_edgecolor("face")
    
    #### text
    #ax.text(0.05,0.92,'N='+str(good.sum()),transform=ax.transAxes,fontsize=16)
    
    plt.tight_layout()
    return fig, ax

def load_dl07_models(outfolder=None,string=''):

    try:
        with open(outfolder+'prospector_template_colors'+string+'.pickle', "rb") as f:
            outdict=pickle.load(f)
    except IOError as e:
        print e
        print 'generating DL07 models'
        outdict = generate_dl07_models(outfolder=outfolder,string=string)

    return outdict

def generate_dl07_models(outfolder='/Users/joel/code/python/prospector_alpha/plots/brownseds_agn/agn_plots/',string=None):

    import bseds_test as nonparam

    #### load test model, build sps, build important variables ####
    sps = nonparam.load_sps(**nonparam.run_params)
    model = nonparam.load_model(**nonparam.run_params)
    obs = nonparam.load_obs(**nonparam.run_params)
    sps.update(**model.params)

    #### pull out boundaries
    ngrid = 10
    to_vary = ['duste_gamma','duste_qpah','duste_umin','logzsol']
    grid = []
    for l in to_vary:
        bounds = model.theta_bounds()[model.theta_index[l]][0]
        grid.append(np.linspace(bounds[0],bounds[1],ngrid))

    outdict = {}
    colors = [['spitzer_irac_ch3','spitzer_irac_ch4'], 
              ['spitzer_irac_ch1','spitzer_irac_ch2'],
              ['wise_w3','wise_w4'], 
              ['wise_w2','wise_w3'], 
              ['wise_w1','wise_w2']]
    for c in colors: outdict[" ".join(c)] = []

    theta = copy.deepcopy(model.initial_theta)
    pnames = model.theta_labels()
    fnames = [f.name for f in obs['filters']]

    ### custom model setup
    theta[pnames.index('fagn')] = 0.0
    indices = [i for i, s in enumerate(pnames) if ('sfr_fraction' in s)]
    theta[indices] = 0.0
    sfr_idx = [idx for idx in indices if string[-1] in pnames[idx]]
    theta[sfr_idx] = 1.0
    theta[pnames.index('dust2')] = 0.3
    theta[pnames.index('dust1')] = 0.3

    for logzsol in grid[3]:
        for gamma in grid[0]:
            for qpah in grid[1]:
                for umin in grid[2]:
                    theta[pnames.index('duste_gamma')] = gamma
                    theta[pnames.index('duste_qpah')] = qpah
                    theta[pnames.index('duste_umin')] = umin
                    theta[pnames.index('logzsol')] = logzsol
                    #sps.ssp.params.dirtiness = 1
                    spec,mags,sm = model.mean_model(theta, obs, sps=sps)
                    for c in colors: outdict[" ".join(c)].append(-2.5*np.log10(mags[fnames.index(c[0])])+2.5*np.log10(mags[fnames.index(c[1])]))
        
    # now do it again with no dust
    try: 
        test = sfr_idx[0]
    except IndexError:
        test = 6
    if test != 1:
        theta[pnames.index('dust2')] = 0.0
        theta[pnames.index('dust1')] = 0.0
        for logzsol in grid[3]:
            theta[pnames.index('logzsol')] = logzsol
            #sps.ssp.params.dirtiness = 1
            spec,mags,sm = model.mean_model(theta, obs, sps=sps)
            for c in colors: outdict[" ".join(c)].append(-2.5*np.log10(mags[fnames.index(c[0])])+2.5*np.log10(mags[fnames.index(c[1])]))
        
    pickle.dump(outdict,open(outfolder+'prospector_template_colors_sfr'+string[-1]+'.pickle', "wb"))
    return outdict











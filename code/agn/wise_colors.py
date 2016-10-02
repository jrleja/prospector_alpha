import copy
import matplotlib.pyplot as plt
import agn_plot_pref
import numpy as np
import matplotlib as mpl
import os
import observe_agn_templates
from scipy.spatial import ConvexHull
import pickle
from matplotlib.patches import Polygon

np.random.seed(2)

dpi = 150

def vega_conversions(fname):

	# mVega = mAB-delta_m
	# Table 5, http://wise2.ipac.caltech.edu/docs/release/prelim/expsup/sec4_3g.html#PhotometricZP
	if fname=='wise_w1':
		return -2.683
	if fname=='wise_w2':
		return -3.319
	if fname=='wise_w3':
		return -5.242
	if fname=='wise_w4':
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
	return out

def plot_mir_colors(runname='brownseds_agn',alldata=None,outfolder=None, vega=True):

	#### load alldata
	if alldata is None:
		import brown_io
		alldata = brown_io.load_alldata(runname=runname)

	#### make output folder if necessary
	if outfolder is None:
		outfolder = os.getenv('APPS')+'/threedhst_bsfh/plots/'+runname+'/agn_plots/'
		if not os.path.isdir(outfolder):
			os.makedirs(outfolder)

	#### collate data
	pdata = collate_data(alldata)

	#### magnitude system?
	system = '(AB)'
	if vega:
		system = '(Vega)'


	'''
	### temporary plot
	from threed_dutils import asym_errors
	xfilt = ['wise_w2','wise_w3']
	good = (pdata['obs_phot'][xfilt[0]] != 0) & (pdata['obs_phot'][xfilt[1]] != 0)
	xplot = -2.5*np.log10(pdata['obs_phot'][xfilt[0]][good])+2.5*np.log10(pdata['obs_phot'][xfilt[1]][good])
	yplot = pdata['model_pars']['duste_qpah']['q50'][good]
	yerr = asym_errors(yplot,pdata['model_pars']['duste_qpah']['q84'][good],pdata['model_pars']['duste_qpah']['q16'][good])

	opts = {
	        'color': '#1C86EE',
	        'mew': 1.5,
	        'alpha': 0.6,
	        'fmt': 'o'
	       }

	plt.errorbar(xplot,yplot,yerr=yerr,**opts)
	plt.xlabel('WISE [4.6]-[12] (AB)')
	plt.ylabel(r'Q$_{\mathrm{PAH}}$')

	plt.ylim(-0.1,7)
	plt.show()
	'''

	#### colored scatterplots
	### IRAC
	cpar_range = [-2,0]
	xfilt, yfilt = ['spitzer_irac_ch3','spitzer_irac_ch4'], ['spitzer_irac_ch1','spitzer_irac_ch2']
	fig,ax = plot_color_scatterplot(pdata,xfilt=xfilt,yfilt=yfilt,
		                   xlabel='IRAC [5.8]-[8.0] (AB)', ylabel='IRAC [3.6]-[4.5] (AB)',
		                   colorpar='fagn',colorparlabel=r'log(f$_{\mathrm{AGN}}$)',log_cpar=True, cpar_range=cpar_range)
	plot_nenkova_templates(ax, xfilt=xfilt,yfilt=yfilt)
	plot_prospector_templates(ax, xfilt=xfilt,yfilt=yfilt,outfolder=outfolder)
	plt.savefig(outfolder+'irac_colors.png',dpi=dpi)
	plt.close()

	### WISE hot
	xfilt, yfilt = ['wise_w2','wise_w3'], ['wise_w1','wise_w2']
	fig, ax = plot_color_scatterplot(pdata,xfilt=xfilt,yfilt=yfilt,
		                             xlabel='WISE [4.6]-[12] (AB)',ylabel='WISE [3.4]-[4.6] '+system,
		                             colorpar='fagn',colorparlabel=r'log(f$_{\mathrm{AGN}}$)',
		                             log_cpar=True, cpar_range=cpar_range,vega=vega)
	plot_nenkova_templates(ax, xfilt=xfilt,yfilt=yfilt,vega=vega)
	plot_prospector_templates(ax, xfilt=xfilt,yfilt=yfilt,outfolder=outfolder,vega=vega)
	outstring = 'wise_hotcolors'
	if vega:
		outstring += '_vega'
	plt.savefig(outfolder+outstring+'.png',dpi=dpi)
	plt.close()

	### WISE warm
	xfilt, yfilt = ['wise_w2','wise_w3'],['wise_w3','wise_w4']
	outstring = 'wise_warmcolors'
	if vega:
		outstring += '_vega'

	fig,ax = plot_color_scatterplot(pdata,xfilt=xfilt,yfilt=yfilt,
		                   xlabel='WISE [4.6]-[12] (AB)',ylabel='WISE [12]-[22] '+system,
		                   colorpar='fagn',colorparlabel=r'log(f$_{\mathrm{AGN}}$)',log_cpar=True, cpar_range=cpar_range,vega=vega)
	plot_nenkova_templates(ax, xfilt=xfilt,yfilt=yfilt,vega=vega)
	plot_prospector_templates(ax, xfilt=xfilt,yfilt=yfilt,outfolder=outfolder,vega=vega)

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

	ax.plot(xp,yp,'o',alpha=0.7,ms=11,color=modcolor,mew=2.2)
	ax.plot(xp,yp,' ',alpha=0.5,color=modcolor, linestyle='-',lw=3)

	ax.text(0.05,0.82,'Nenkova+08 \n AGN Templates',transform=ax.transAxes,fontsize=16,color=modcolor,weight='bold',alpha=0.6)

def plot_prospector_templates(ax, xfilt=None, yfilt=None, outfolder=None, vega=False,multiple=True):

	'''
	strings = '' is continuous SFH
	_sfrX is all SFR in bin X
	'''

	prospcolor = ['#FFB6C1','#c264ff']

	xp, yp = [], []
	strings = ['', '_sfr1','_sfr2','_sfr4','_sfr6']
	strings = ['_sfr1','_sfr6']
	label = [' young', ' old']
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
		               facecolor=prospcolor[i], alpha=0.8,zorder=-35)
		poly.set_capstyle('round')
		plt.gca().add_patch(poly)

		ax.text(0.05,0.72-0.11*i,'Prospector-$\\alpha$ \n'+label[i]+' models',transform=ax.transAxes,fontsize=16,color=prospcolor[i],weight='bold')
		i += 1

def plot_color_scatterplot(pdata,xfilt=None,yfilt=None,xlabel=None,ylabel=None,
	                       colorpar=None,colorparlabel=None,log_cpar=False,cpar_range=None,
	                       outname=None, vega=False):
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
	pts = ax.scatter(xplot, yplot, marker='o', c=cpar_plot, cmap=plt.cm.jet,s=70,alpha=0.6)

	#### label and add colorbar
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	cb = fig.colorbar(pts, ax=ax, aspect=10)
	cb.set_label(colorparlabel)
	cb.solids.set_rasterized(True)
	cb.solids.set_edgecolor("face")

	#### text
	ax.text(0.05,0.92,'N='+str(good.sum()),transform=ax.transAxes,fontsize=16)

	return fig, ax

def load_dl07_models(outfolder=None,string=''):

	try:
		with open(outfolder+'prospector_template_colors'+string+'.pickle', "rb") as f:
			outdict=pickle.load(f)
	except IOError as e:
		print e
		print 'generating DL07 models'
		outdict = generate_dl07_models(outfolder=outfolder)

	return outdict


def generate_dl07_models(outfolder='/Users/joel/code/python/threedhst_bsfh/plots/brownseds_agn/agn_plots/'):

	import brownseds_agn_params as nonparam

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
		start,end = model.theta_index[l]
		bounds = model.theta_bounds()[start:end][0]
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
	theta[pnames.index('sfr_fraction_4')] = 1.0
	print theta

	for logzsol in grid[3]:
		for gamma in grid[0]:
			for qpah in grid[1]:
				for umin in grid[2]:
					theta[pnames.index('duste_gamma')] = gamma
					theta[pnames.index('duste_qpah')] = qpah
					theta[pnames.index('duste_umin')] = umin
					theta[pnames.index('logzsol')] = logzsol
					sps.ssp.params.dirtiness = 1
					spec,mags,sm = model.mean_model(theta, obs, sps=sps)
					for c in colors: outdict[" ".join(c)].append(-2.5*np.log10(mags[fnames.index(c[0])])+2.5*np.log10(mags[fnames.index(c[1])]))
		
	# now do it again with no dust
	theta[pnames.index('dust1')] = 0.0
	theta[pnames.index('dust2')] = 0.0
	for logzsol in grid[3]:
		theta[pnames.index('logzsol')] = logzsol
		sps.ssp.params.dirtiness = 1
		spec,mags,sm = model.mean_model(theta, obs, sps=sps)
		for c in colors: outdict[" ".join(c)].append(-2.5*np.log10(mags[fnames.index(c[0])])+2.5*np.log10(mags[fnames.index(c[1])]))



	pickle.dump(outdict,open(outfolder+'prospector_template_colors_sfr4.pickle', "wb"))
	return outdict











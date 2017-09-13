import numpy as np
import prospector_io
import matplotlib.pyplot as plt
import agn_plot_pref
from prosp_dutils import asym_errors
import os

dpi = 150

def collate_data(alldata):

	'''
	alldata['uvj'] = {}
	alldata['uvj']['uv'] = np.percentile(uv,[50.0,84.0,16.0])
	alldata['uvj']['vj'] = np.percentile(vj,[50.0,84.0,16.0])
	'''

	### ssfr
	eparnames = alldata[0]['pextras']['parnames']
	idx = eparnames == 'ssfr_100'
	ssfr_100 = np.array([dat['pextras']['q50'][idx] for dat in alldata])
	#ssfr_100_errup = np.array([dat['pextras']['q84'][idx] for dat in alldata])
	#ssfr_100_errdown = np.array([dat['pextras']['q16'][idx] for dat in alldata])
	#ssfr_err = asym_errors(ssfr_100,ssfr_100_errup-ssfr_100,ssfr_100-ssfr_100_errdown,log=True)
	ssfr = np.log10(ssfr_100)

	### UVJ
	uv = np.array([dat['mphot']['uv'][0] for dat in alldata])
	uv_up = np.array([dat['mphot']['uv'][1] for dat in alldata])
	uv_do = np.array([dat['mphot']['uv'][2] for dat in alldata])
	uv_err = asym_errors(uv,uv_up,uv_do,log=False)

	vj = np.array([dat['mphot']['vj'][0] for dat in alldata])
	vj_up = np.array([dat['mphot']['vj'][1] for dat in alldata])
	vj_do = np.array([dat['mphot']['vj'][2] for dat in alldata])
	vj_err = asym_errors(vj,vj_up,vj_do,log=False)

	### u g J
	color_names = ['u-g','g-J','u-r','r-J']
	fnames = [['sdss_u0','sdss_g0'],['sdss_g0','twomass_J'],
	          ['sdss_u0','sdss_r0'],['sdss_r0','twomass_J']]

	phot = {}
	for f in color_names: phot[f] = {'q50':[],'q84':[],'q16':[],'obs':[]}

	for dat in alldata:
		filters = dat['filters']
		for f,c in zip(fnames,color_names):
			f1_idx = filters == f[0]
			f2_idx = filters == f[1]

			if f1_idx.sum()+f2_idx.sum() != 2:
				c50, c84, c16, cobs = 0.0, 0.0, 0.0, 0.0
			else:
				color =  -2.5*np.log10(dat['model_maggies'][f1_idx,:])+2.5*np.log10(dat['model_maggies'][f2_idx,:])
				c50, c84, c16 = np.percentile(color,[50.0,84.0,16.0])
				cobs = -2.5*np.log10(dat['obs_maggies'][f1_idx])+2.5*np.log10(dat['obs_maggies'][f2_idx])

			phot[c]['q50'].append(c50)
			phot[c]['q84'].append(c84)
			phot[c]['q16'].append(c16)
			phot[c]['obs'].append(cobs[0])

	for key in phot.keys():
		for key2 in phot[key].keys():
			phot[key][key2] = np.array(phot[key][key2])

	out = {}
	out['ssfr'] = ssfr
	for c in color_names:
		out[c] = phot[c]['q50']
		out[c+'_err'] = asym_errors(phot[c]['q50'],phot[c]['q84'],phot[c]['q16'],log=False)
		out[c+'_obs'] = phot[c]['obs']
	out['u-v'] = uv
	out['u-v_err'] = uv_err
	out['v-j'] = vj
	out['v-j_err'] = vj_err
	return out

def plot(runname='brownseds_np',alldata=None,outfolder=None):

	#### load alldata
	if alldata is None:
		alldata = prospector_io.load_alldata(runname=runname)

	#### make output folder if necessary
	if outfolder is None:
		outfolder = os.getenv('APPS')+'/prospector_alpha/plots/'+runname+'/pcomp/'
		if not os.path.isdir(outfolder):
			os.makedirs(outfolder)

	#### collate data
	pdata = collate_data(alldata)

	#### UVJ
	cpar_range = [-13,-8]
	fig,ax = plot_color_scatterplot(pdata, xlabel='V-J (AB)', ylabel='U-V (AB)', 
									xpar = 'v-j', ypar = 'u-v', obs=False,
		                            colorparlabel=r'log(sSFR/yr$^{-1}$)', cpar_range=cpar_range)
	add_uvj(ax)
	ax.set_xlim(-0.6,2.5)
	ax.set_ylim(-0.5,2.5)
	plt.savefig(outfolder+'uvj.png',dpi=dpi)
	plt.close()

	#### ugJ [model]
	xlim = (-1.0,3.0)
	ylim = (0.0,2.0)
	fig,ax = plot_color_scatterplot(pdata, xlabel='g-J [model] (AB)', ylabel='u-g [model] (AB)', 
									xpar = 'g-J', ypar = 'u-g', obs=False,
		                            colorparlabel=r'log(sSFR/yr$^{-1}$)', cpar_range=cpar_range)
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	plt.savefig(outfolder+'ugJ_model.png',dpi=dpi)
	plt.close()

	#### ugJ [obs]
	fig,ax = plot_color_scatterplot(pdata, xlabel='g-J [obs] (AB)', ylabel='u-g [obs] (AB)', 
									xpar = 'g-J', ypar = 'u-g', obs=True,
		                            colorparlabel=r'log(sSFR/yr$^{-1}$)', cpar_range=cpar_range)
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	plt.savefig(outfolder+'ugJ_obs.png',dpi=dpi)
	plt.close()

	#### urJ [model]
	xlim = (-1.0,2.0)
	ylim = (0.0,3.0)
	fig,ax = plot_color_scatterplot(pdata, xlabel='r-J [model] (AB)', ylabel='u-r [model] (AB)', 
									xpar = 'r-J', ypar = 'u-r', obs=False,
		                            colorparlabel=r'log(sSFR/yr$^{-1}$)', cpar_range=cpar_range)
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	plt.savefig(outfolder+'urJ_model.png',dpi=dpi)
	plt.close()

	#### urJ [obs]
	fig,ax = plot_color_scatterplot(pdata, xlabel='r-J [obs] (AB)', ylabel='u-r [obs] (AB)', 
									xpar = 'r-J', ypar = 'u-r', obs=True,
		                            colorparlabel=r'log(sSFR/yr$^{-1}$)', cpar_range=cpar_range)
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	plt.savefig(outfolder+'urJ_obs.png',dpi=dpi)
	plt.close()

def add_uvj(ax):

	##### plot UVJ dividers from van der Wel+14
	ax.plot([-20,0.92], [1.3, 1.3] , linestyle='--',color='0.5',lw=1.5)
	xline = np.array([0.92,1.6])
	yline = xline*0.8+0.7
	ax.plot(xline,yline , linestyle='--',color='0.5',lw=1.5)
	ax.plot([xline[1],xline[1]],[yline[-1],100] , linestyle='--',color='0.5',lw=1.5)

def plot_color_scatterplot(pdata,xlabel=None,ylabel=None,xpar=None,ypar=None,obs=False,
	                       colorparlabel=None,cpar_range=None):
	'''
	plots a color-color scatterplot
	'''

	#### generate x, y values
	if obs:
		xplot,yplot = pdata[xpar+'_obs'],pdata[ypar+'_obs']
		xerr, yerr = None, None
	else:
		xplot = pdata[xpar]
		xerr = pdata[xpar+'_err']
		yplot = pdata[ypar]
		yerr = pdata[ypar+'_err']

	#### generate color mapping
	cpar_plot = pdata['ssfr']
	if cpar_range is not None:
		cpar_plot = np.clip(cpar_plot,cpar_range[0],cpar_range[1])

	#### plot photometry
	fig, ax = plt.subplots(1,1, figsize=(8, 6))
	ax.errorbar(xplot, yplot, yerr=yerr, xerr=xerr,
	            fmt='o', ecolor='k', capthick=1.5,elinewidth=1.5,ms=0.0,alpha=0.5,zorder=-5)

	pts = ax.scatter(xplot, yplot, marker='o', c=cpar_plot, cmap=plt.cm.jet,s=70,alpha=0.6)

	#### label and add colorbar
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	cb = fig.colorbar(pts, ax=ax, aspect=10)
	cb.set_label(colorparlabel)
	cb.solids.set_rasterized(True)
	cb.solids.set_edgecolor("face")

	return fig, ax















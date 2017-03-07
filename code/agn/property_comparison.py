import numpy as np
from threedhst_diag import add_sfh_plot
import os
import matplotlib.pyplot as plt
import magphys_plot_pref
import matplotlib as mpl
from magphys_plots import median_by_band
from threed_dutils import running_median

'''
plot 10 largest f_agn spectral comparisons, with two SFHs and two best-fit photometries! 
(look specifically at Akari and Spitzer residuals, may be some improvement)

Maybe separate panels with SFHs, spec residuals, photos residuals? fagn labeled?
'''

minorFormatter = magphys_plot_pref.jLogFormatter(base=10, labelOnlyBase=False)
majorFormatter = magphys_plot_pref.jLogFormatter(base=10, labelOnlyBase=True)

def collate_data(alldata, alldata_noagn):

	### package up information
	the_data = [alldata, alldata_noagn]
	data_label = ['agn','no_agn']
	output = {}
	for ii in xrange(2):

		#### generate containers
		# SFH
		sfh = {
			   'perc':[],
		       't_sfh':[],
		      }

		# residuals
		residuals = {}
		labels = ['Optical','Akari','Spitzer IRS']
		for l in labels: residuals[l] = {'lam':[],'resid':[]}

		# phot residuals
		phot_residuals = {'resid':[],'lam_obs':[]}

		# model parameters
		objname = []
		model_pars = {}
		pnames = ['fagn', 'agn_tau','duste_qpah']
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

			#### SFH
			sfh['t_sfh'].append(dat['pextras']['t_sfh'])
			perc = np.zeros(shape=(len(sfh['t_sfh'][-1]),3))
			for jj in xrange(perc.shape[0]): perc[jj,:] = np.percentile(dat['pextras']['sfh'][jj,:],[16.0,50.0,84.0])
			sfh['perc'].append(perc)

			#### photometric residuals
			phot_residuals['resid'].append(dat['residuals']['phot']['frac_prosp'])
			phot_residuals['lam_obs'].append(dat['residuals']['phot']['lam_obs'])

			#### spectral residuals
			for key in residuals.keys():
				if key in dat['residuals'].keys():
					residuals[key]['lam'].append(dat['residuals'][key]['obs_restlam'])
					residuals[key]['resid'].append(dat['residuals'][key]['prospector_resid'])
				else:
					residuals[key]['lam'].append(None)
					residuals[key]['resid'].append(None)

		#### numpy arrays
		for key in residuals.keys(): 
			for key2 in residuals[key].keys():
				residuals[key][key2] = np.array(residuals[key][key2])
		for key in sfh.keys(): sfh[key] = np.array(sfh[key])
		for key in phot_residuals.keys(): phot_residuals[key] = np.array(phot_residuals[key])
		for key in model_pars.keys(): 
			for key2 in model_pars[key].keys():
				model_pars[key][key2] = np.array(model_pars[key][key2])

		output[data_label[ii]] = {}
		output[data_label[ii]]['objname'] = objname
		output[data_label[ii]]['model_pars'] = model_pars
		output[data_label[ii]]['sfh'] = sfh
		output[data_label[ii]]['residuals'] = residuals
		output[data_label[ii]]['phot_residuals'] = phot_residuals

	return output

def plot_comparison(runname='brownseds_agn',runname_noagn='brownseds_np',alldata=None,alldata_noagn=None,outfolder=None):

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
	idx_plot = pdata['agn']['model_pars']['fagn']['q50'].argsort()[-10:]
	colors = ['#9400D3','#FF420E']
	for i,key in enumerate(pdata): pdata[key]['color'] = colors[i]

	plot_residuals(pdata,idx_plot,outfolder)
	plot_sfh(pdata,idx_plot,outfolder)

def add_txt(ax,pdata,fs=12,x=0.05,y=0.88,dy=0.075,ha='left',**extras):

	for i,key in enumerate(pdata.keys()):
		ax.text(x,y-i*dy,key.replace('_',' ').upper(),fontsize=fs,transform=ax.transAxes,ha=ha,color=pdata[key]['color'],**extras)

def add_identifier(ax,idx,pdata,fs=12,x=0.98,y=0.88,dy=0.08,weight='bold'):

	ax.text(x,y,pdata['agn']['objname'][idx],fontsize=fs,transform=ax.transAxes,ha='right',weight=weight)

	mid = pdata['agn']['model_pars']['fagn']['q50'][idx]
	fmt = "{{0:{0}}}".format(".2f").format
	text = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
	text = text.format(fmt(mid), \
		               fmt(mid-pdata['agn']['model_pars']['fagn']['q16'][idx]), \
		               fmt(pdata['agn']['model_pars']['fagn']['q84'][idx]-mid))
	text = "{0} = {1}".format(r'f$_{\mathrm{MIR}}$=', text)

	ax.text(x,y-dy,text,fontsize=fs,transform=ax.transAxes,ha='right')

def plot_sfh(pdata,idx_plot,outfolder):

	### open figure
	fig, ax = plt.subplots(5,2, figsize=(7, 15))

	ax = np.ravel(ax)
	fs = 10
	idx_plot = idx_plot[::-1]

	### begin loop
	for ii,idx in enumerate(idx_plot):

		pmin,pmax = np.inf, -np.inf
		for key in pdata.keys():

			#### load SFH properties
			t = pdata[key]['sfh']['t_sfh'][idx]
			perc = pdata[key]['sfh']['perc'][idx]

			### plot SFH
			ax[ii].plot(t, perc[:,1],'-',color=pdata[key]['color'],lw=2.5,alpha=0.9)
			ax[ii].fill_between(t, perc[:,0], perc[:,2], color=pdata[key]['color'], alpha=0.15)

			### labels and ranges
			pmin,pmax = np.min([pmin,perc.min()]), np.max([pmax,perc.max()])
			ax[ii].set_ylabel(r'SFR [M$_{\odot}$/yr]',fontsize=fs*1.5)
			ax[ii].set_xlabel('lookback time [Gyr]',fontsize=fs*1.5)

			ax[ii].set_xscale('log',nonposx='clip',subsx=([1]))
			ax[ii].xaxis.set_minor_formatter(minorFormatter)
			ax[ii].xaxis.set_major_formatter(majorFormatter)

			ax[ii].set_yscale('log',nonposy='clip',subsy=([1]))
			ax[ii].yaxis.set_minor_formatter(minorFormatter)
			ax[ii].yaxis.set_major_formatter(majorFormatter)

			add_identifier(ax[ii],idx,pdata, fs=fs,weight='bold')
			add_txt(ax[ii],pdata,fs=fs,weight='bold')

		ax[ii].set_ylim(pmin*0.2,pmax*8)
		ax[ii].set_xlim(t.min()*30,t.max())

	plt.tight_layout(w_pad=0.5,h_pad=0.3)
	plt.savefig(outfolder+'sfh_comparison.png',dpi=150)
	plt.close()

def plot_residuals(pdata,idx_plot,outfolder):

	#### plot geometry
	fig, ax = plt.subplots(4,1, figsize=(7, 15))
	plt.subplots_adjust(right=0.93,top=0.98,left=0.17,hspace=0.19,bottom=0.1)
	#plt.subplots_adjust(right=0.75,top=0.98,left=0.17,hspace=0.17)
	#cmap_ax = fig.add_axes([0.88,0.05,0.1,0.9])
	ax = ax.ravel()
	fs = 18

	#### color by F_AGN
	cquant = np.log10(pdata['agn']['model_pars']['fagn']['q50'][idx_plot])
	vmin,vmax = cquant.min(),cquant.max()
	cmap = mpl.colors.LinearSegmentedColormap('jet', mpl.cm.revcmap(mpl.cm.jet._segmentdata))
	norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
	scalarMap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

	#### colorbar
	'''
	cb1 = mpl.colorbar.ColorbarBase(cmap_ax, cmap=cmap,
                                	norm=norm,
                                	orientation='vertical')
	cb1.set_label(r'log(f$_{\mathrm{MIR}}$)')
	cb1.ax.yaxis.set_ticks_position('left')
	cb1.ax.yaxis.set_label_position('left')
	'''

	#### plot options
	single_opts = {'alpha':0.5,'lw':1,'color':'0.5'}
	median_opts = {'alpha':0.8,'lw':4,'color':'k'}

	#### data holders
	ymin,xmin = [np.repeat(np.inf,4) for i in range(2)]
	ymax,xmax = [np.repeat(-np.inf,4) for i in range(2)]

	photx, photy = [],[]
	specx, specy = [[],[],[]], [[],[],[]]

	### begin loop
	for ii,idx in enumerate(idx_plot):

		color = scalarMap.to_rgba(cquant[ii])

		xdat = pdata['agn']['phot_residuals']['lam_obs'][idx]/1e4
		ydat = pdata['no_agn']['phot_residuals']['resid'][idx]**2-pdata['agn']['phot_residuals']['resid'][idx]**2
		photx += xdat.tolist()
		photy += ydat.tolist()

		ax[0].plot(xdat,ydat,**single_opts)
		ymin[0],ymax[0] = np.min([ydat.min(),ymin[0]]),np.max([ydat.max(),ymax[0]])
		xmin[0],xmax[0] = np.min([xdat.min(),xmin[0]]),np.max([xdat.max(),xmax[0]])

		# prospector_resid = np.log10(observed_flux) - np.log10(model_flux)
		for i,key in enumerate(pdata['agn']['residuals'].keys()):
			if pdata['agn']['residuals'][key]['resid'][idx] is not None:
				ydat = np.sqrt(pdata['no_agn']['residuals'][key]['resid'][idx]**2) - np.sqrt(pdata['agn']['residuals'][key]['resid'][idx]**2)
				xdat = pdata['agn']['residuals'][key]['lam'][idx]
				specx[i] += xdat.tolist()
				specy[i] += ydat.tolist()

				ax[i+1].plot(xdat,ydat,**single_opts)
				ymin[i+1],ymax[i+1] = np.nanmin([ydat.min(),ymin[i+1]]),np.nanmax([ydat.max(),ymax[i+1]])
				xmin[i+1],xmax[i+1] = np.nanmin([xdat.min(),xmin[i+1]]),np.nanmax([xdat.max(),xmax[i+1]])

	#### plot medians
	photx_median, photy_median = median_by_band(np.array(photx),np.array(photy))
	ax[0].plot(photx_median, photy_median,**median_opts)

	for i, key in enumerate(pdata['agn']['residuals'].keys()):
		x, y = running_median(np.array(specx[i]),np.array(specy[i]),nbins=100)
		ax[i+1].plot(x, y, **median_opts)

	#### labels and scales
	ymin, ymax = np.repeat(-0.3,4), np.repeat(0.3,4)
	labelpad=-1.5
	ax[0].set_xlabel(r'observed wavelength [$\mu$m]',labelpad=labelpad)
	ax[0].set_ylabel(r'RMS$_{\mathrm{no-AGN}}$-RMS$_{\mathrm{MIR}}$ [dex]')
	ax[0].set_ylim(ymin[0],ymax[0])
	ax[0].set_xlim(xmin[0],xmax[0])

	ax[0].text(0.05,0.9,'photometry',fontsize=fs,transform=ax[0].transAxes)
	ax[0].set_xscale('log',nonposx='clip',subsx=(1,2,4))
	ax[0].xaxis.set_minor_formatter(minorFormatter)
	ax[0].xaxis.set_major_formatter(majorFormatter)

	for i, key in enumerate(pdata['agn']['residuals'].keys()):
		ax[i+1].set_ylabel(r'RMS$_{\mathrm{no-AGN}}$-RMS$_{\mathrm{MIR}}$ [dex]')
		ax[i+1].set_xlabel(r'observed wavelength [$\mu$m]',labelpad=labelpad)
		ax[i+1].set_ylim(ymin[i+1],ymax[i+1])
		ax[i+1].set_xlim(xmin[i+1],xmax[i+1])
		ax[i+1].text(0.05,0.9,key,fontsize=fs,transform=ax[i+1].transAxes)

		ax[i+1].set_xscale('log',nonposx='clip',subsx=(1,2,3,4,5,6))
		ax[i+1].xaxis.set_minor_formatter(minorFormatter)
		ax[i+1].xaxis.set_major_formatter(majorFormatter)

 	plt.savefig(outfolder+'residuals.png',dpi=150)
 	plt.close()










import numpy as np
import brown_io
import matplotlib.pyplot as plt
import threed_dutils
import os
from astropy.cosmology import WMAP9
import magphys_plot_pref
from corner import quantile
from matplotlib.ticker import MaxNLocator

dpi = 150

minorFormatter = magphys_plot_pref.jLogFormatter(base=10, labelOnlyBase=False)
majorFormatter = magphys_plot_pref.jLogFormatter(base=10, labelOnlyBase=True)

plotopts = {
		 'fmt':'o',
		 'ecolor':'k',
		 'capthick':0.4,
		 'elinewidth':0.4,
		 'alpha':0.55
        } 

def get_cmap(N):
	'''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
	RGB color.'''

	import matplotlib.cm as cmx
	import matplotlib.colors as colors

	color_norm  = colors.Normalize(vmin=0, vmax=N-1)
	scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='plasma') 
	def map_index_to_rgb_color(index):
		return scalar_map.to_rgba(index)
	return map_index_to_rgb_color

def collate_data(alldata, **extras):

	### preliminary stuff
	parnames = alldata[0]['pquantiles']['parnames']
	eparnames = alldata[0]['pextras']['parnames']
	xr_idx = eparnames == 'xray_lum'
	xray = brown_io.load_xray_cat(xmatch = True, **extras)
	nsamp = 100000 # for newly defined variables

	#### for each object
	fagn, fagn_up, fagn_down, mass, lir_luv, lir_luv_up, lir_luv_down, xray_lum, xray_lum_err, database, observatory = [[] for i in range(11)]
	fagn_obs, fagn_obs_up, fagn_obs_down, lmir_lbol, lmir_lbol_up, lmir_lbol_down = [[] for i in range(6)]
	lagn, lagn_up, lagn_down, lsfr, lsfr_up, lsfr_down = [], [], [], [], [], []
	sfr, sfr_up, sfr_down, ssfr, ssfr_up, ssfr_down, d2, d2_up, d2_down = [[] for i in range(9)]
	fmir, fmir_up, fmir_down = [], [], []
	for ii, dat in enumerate(alldata):
		
		#### mass, SFR, sSFR, dust2
		mass.append(dat['pextras']['q50'][eparnames=='stellar_mass'][0])
		sfr.append(dat['pextras']['q50'][eparnames=='sfr_100'][0])
		sfr_up.append(dat['pextras']['q84'][eparnames=='sfr_100'][0])
		sfr_down.append(dat['pextras']['q16'][eparnames=='sfr_100'][0])
		ssfr.append(dat['pextras']['q50'][eparnames=='ssfr_100'][0])
		ssfr_up.append(dat['pextras']['q84'][eparnames=='ssfr_100'][0])
		ssfr_down.append(dat['pextras']['q16'][eparnames=='ssfr_100'][0])
		d2.append(dat['pquantiles']['q50'][parnames=='dust2'][0])
		d2_up.append(dat['pquantiles']['q84'][parnames=='dust2'][0])
		d2_down.append(dat['pquantiles']['q16'][parnames=='dust2'][0])

		#### model f_agn, l_agn, fmir
		fagn.append(dat['pquantiles']['q50'][parnames=='fagn'][0])
		fagn_up.append(dat['pquantiles']['q84'][parnames=='fagn'][0])
		fagn_down.append(dat['pquantiles']['q16'][parnames=='fagn'][0])
		lagn.append(dat['pextras']['q50'][eparnames=='l_agn'][0])
		lagn_up.append(dat['pextras']['q84'][eparnames=='l_agn'][0])
		lagn_down.append(dat['pextras']['q16'][eparnames=='l_agn'][0])
		fmir.append(dat['pextras']['q50'][eparnames=='fmir'][0])
		fmir_up.append(dat['pextras']['q84'][eparnames=='fmir'][0])
		fmir_down.append(dat['pextras']['q16'][eparnames=='fmir'][0])

		#### L_UV / L_IR, LMIR/LBOL
		cent, eup, edo = quantile(np.random.choice(dat['lir'],nsamp) / np.random.choice(dat['luv'],nsamp), [0.5, 0.84, 0.16])

		lir_luv.append(cent)
		lir_luv_up.append(eup)
		lir_luv_down.append(edo)

		cent, eup, edo = quantile(np.random.choice(dat['lmir'],nsamp) / np.random.choice(dat['pextras']['flatchain'][:,dat['pextras']['parnames'] == 'lbol'].squeeze(),nsamp), [0.5, 0.84, 0.16])

		lmir_lbol.append(cent)
		lmir_lbol_up.append(eup)
		lmir_lbol_down.append(edo)

		#### x-ray fluxes
		# match
		idx = xray['objname'] == dat['objname']
		if idx.sum() != 1:
			print 1/0
		xflux = xray['flux'][idx][0]
		xflux_err = xray['flux_err'][idx][0]

		# flux is in ergs / cm^2 / s, convert to erg /s 
		pc2cm =  3.08568E18
		dfactor = 4*np.pi*(dat['residuals']['phot']['lumdist']*1e6*pc2cm)**2
		xray_lum.append(xflux * dfactor)
		xray_lum_err.append(xflux_err * dfactor)

		#### CALCULATE F_AGN_OBS
		# take advantage of the already-computed conversion between FAGN (model) and LAGN (model)
		fagn_chain = dat['pquantiles']['sample_chain'][:,parnames=='fagn']
		lagn_chain = dat['pextras']['flatchain'][:,eparnames == 'l_agn']
		conversion = (lagn_chain / fagn_chain).squeeze()

		### calculate L_AGN chain
		scale = xray_lum_err[-1]
		if scale <= 0:
			lagn_chain = np.repeat(xray_lum[-1], conversion.shape[0])
		else: 
			lagn_chain = np.random.normal(loc=xray_lum[-1], scale=scale, size=conversion.shape[0])
		obs_fagn_chain = lagn_chain / conversion
		cent, eup, edo = quantile(obs_fagn_chain, [0.5, 0.84, 0.16])

		fagn_obs.append(cent)
		fagn_obs_up.append(eup)
		fagn_obs_down.append(edo)

		##### L_OBS / L_SFR(MODEL)
		# sample from the chain, assume gaussian errors for x-ray fluxes
		chain = dat['pextras']['flatchain'][:,xr_idx].squeeze()

		if scale <= 0:
			subchain =  np.repeat(xray_lum[-1], nsamp) / \
			            np.random.choice(chain,nsamp)
		else:
			subchain =  np.random.normal(loc=xray_lum[-1], scale=scale, size=nsamp) / \
			            np.random.choice(chain,nsamp)

		cent, eup, edo = quantile(subchain, [0.5, 0.84, 0.16])

		lsfr.append(cent)
		lsfr_up.append(eup)
		lsfr_down.append(edo)

		#### database and observatory
		database.append(str(xray['database'][idx][0]))
		try:
			observatory.append(str(xray['observatory'][idx][0]))
		except KeyError:
			observatory.append(' ')

	out = {}
	out['database'] = database
	out['observatory'] = observatory
	out['mass'] = mass
	out['sfr'] = sfr
	out['sfr_up'] = sfr_up
	out['sfr_down'] = sfr_down
	out['ssfr'] = ssfr
	out['ssfr_up'] = ssfr_up
	out['ssfr_down'] = ssfr_down
	out['d2'] = d2
	out['d2_up'] = d2_up
	out['d2_down'] = d2_down
	out['lir_luv'] = lir_luv
	out['lir_luv_up'] = lir_luv_up
	out['lir_luv_down'] = lir_luv_down
	out['lmir_lbol'] = lmir_lbol
	out['lmir_lbol_up'] = lmir_lbol_up
	out['lmir_lbol_down'] = lmir_lbol_down
	out['fagn'] = fagn
	out['fagn_up'] = fagn_up
	out['fagn_down'] = fagn_down
	out['fmir'] = fmir
	out['fmir_up'] = fmir_up
	out['fmir_down'] = fmir_down
	out['fagn_obs'] = fagn_obs
	out['fagn_obs_up'] = fagn_obs_up
	out['fagn_obs_down'] = fagn_obs_down
	out['lagn'] = lagn 
	out['lagn_up'] = lagn_up
	out['lagn_down'] = lagn_down
	out['lsfr'] = lsfr
	out['lsfr_up'] = lsfr_up
	out['lsfr_down'] = lsfr_down
	out['xray_luminosity'] = xray_lum
	out['xray_luminosity_err'] = xray_lum_err

	for key in out.keys(): out[key] = np.array(out[key])

	#### ADD WISE PHOTOMETRY
	from wise_colors import collate_data as wise_phot
	from wise_colors import vega_conversions
	wise = wise_phot(alldata)

	#### generate x, y values
	w1w2 = -2.5*np.log10(wise['obs_phot']['wise_w1'])+2.5*np.log10(wise['obs_phot']['wise_w2'])
	w1w2 += vega_conversions('wise_w1') - vega_conversions('wise_w2')
	out['w1w2'] = w1w2

	return out

def make_plot(runname='brownseds_agn',alldata=None,outfolder=None,maxradius=30,idx=Ellipsis,**popts):

	#### load alldata
	if alldata is None:
		alldata = brown_io.load_alldata(runname=runname)

	#### make output folder if necessary
	if outfolder is None:
		outfolder = os.getenv('APPS')+'/threedhst_bsfh/plots/'+runname+'/agn_plots/'
		if not os.path.isdir(outfolder):
			os.makedirs(outfolder)

	#### collate data
	pdata = collate_data(alldata, maxradius=maxradius)
	cbd = False
	cbo = False
	cbw = False

	### PLOT VERSUS OBSERVED X-RAY FLUX
	outname = 'xray_lum_fagn_model.png'
	fig,ax = plot(pdata,color_by_observatory=cbo,color_by_database=cbd,color_by_wise=cbw,
		          ypar='fagn',ylabel = r'f$_{\mathrm{MIR}}$',sf_flag=True, idx=idx, **popts)
	plt.savefig(outfolder+outname,dpi=dpi)
	plt.close()

	### PLOT VERSUS 'TRUE' X-RAY FLUX
	outname = 'xray_lum_sfrcorr_fagn_model.png'
	fig,ax = plot(pdata,color_by_observatory=cbo,color_by_database=cbd,color_by_wise=cbw,
		          ypar='fagn',ylabel = r'f$_{\mathrm{MIR}}$',
		          xpar='lsfr',xlabel = r'L$_{\mathrm{X}}$(observed)/L$_{\mathrm{XRB}}$(model)',
		          idx = idx, **popts)
	plt.savefig(outfolder+outname,dpi=dpi)
	plt.close()

	'''
	outname = 'xray_lum_sfrcorr_lagn.png'
	fig,ax = plot(pdata,color_by_observatory=cbo,color_by_database=cbd,color_by_wise=cbw,
		          ypar='lagn',ylabel = r'model L$_{\mathrm{MIR}}$ [erg/s]',
		          xpar='lsfr',xlabel = r'L$_{\mathrm{X}}$(observed)/L$_{\mathrm{XRB}}$(model)')
	plt.savefig(outfolder+outname,dpi=dpi)
	plt.close()
	'''

	### SFR, sSFR, dust2, LUV/LIR versus FAGN
	fig,ax = plot_model_corrs(pdata,idx=idx,**popts)
	#plt.tight_layout()
	plt.savefig(outfolder+'fagn_versus_galaxy_properties.png',dpi=dpi)
	plt.close()

def plot_model_corrs(pdata,color_by=None,idx=None,**popts):

	fig, ax = plt.subplots(2,2, figsize=(11, 10))
	cb_ax = fig.add_axes([0.88, 0.15, 0.05, 0.7])
	fig.subplots_adjust(right=0.85,wspace=0.3,hspace=0.3,left=0.12)
	ax = np.ravel(ax)

	#### fagn labeling
	xlabel = r'log(f$_{\mathrm{MIR}}$)'
	x = np.log10(pdata['fagn'])
	xerr =  threed_dutils.asym_errors(pdata['fagn'], 
		                              pdata['fagn_up'],
		                              pdata['fagn_down'],log=True)

	#### y-axis
	ypar = ['sfr','ssfr','lmir_lbol','lir_luv']
	ylabels = [r'log(SFR) [M$_{\odot}$/yr]', r'log(sSFR) [yr$^{-1}$]',
	           r'log(L$_{\mathrm{MIR}}$/L$_{\mathrm{bol}}$)',r'log(L$_{\mathrm{IR}}$/L$_{\mathrm{UV}}$)']
	cb = pdata['d2']
	for ii, yp in enumerate(ypar):

		log = True
		y = np.log10(pdata[yp])

		# find the complement
		cidx = np.ones_like(y,dtype=bool)
		cidx[idx] = False

		yerr =  threed_dutils.asym_errors(pdata[yp], 
			                              pdata[yp+'_up'],
			                              pdata[yp+'_down'],log=log)
		ax[ii].errorbar(x,y,yerr=yerr, xerr=xerr, ms=0.0,zorder=-2,**plotopts)

		vmin, vmax = cb.min(), cb.max()
		ax[ii].scatter(x[cidx],y[cidx], marker=popts['nofmir_shape'],alpha=popts['nofmir_alpha'],c=cb[cidx], cmap=popts['cmap'],
			           s=50,zorder=10, vmin=vmin, vmax=vmax)
		pts = ax[ii].scatter(x[idx],y[idx], marker=popts['fmir_shape'],alpha=popts['fmir_alpha'],c=cb[idx], cmap=popts['cmap'],
			                 s=50,zorder=10, vmin=vmin, vmax=vmax)

		ax[ii].set_xlabel(xlabel)
		ax[ii].set_ylabel(ylabels[ii])
		ax[ii].xaxis.set_major_locator(MaxNLocator(5))

	cb = fig.colorbar(pts, cax=cb_ax)
	cb.ax.set_title(r'$\tau_{v}$', fontdict={'fontweight':'bold','verticalalignment':'bottom'})
	cb.solids.set_rasterized(True)
	cb.solids.set_edgecolor("face")

	return fig, ax

def plot(pdata,color_by_observatory=False,color_by_database=False,color_by_wise=False,
	     ypar=None, ylabel=None, 
	     xpar='xray_luminosity', xlabel=r'L$_{\mathrm{X}}$(observed) [erg/s]',idx=Ellipsis,
	     sf_flag=False, **popts):
	'''
	plots a color-color BPT scatterplot
	'''

	red = '#FF3D0D'
	blue = '#1C86EE'

	xplot = pdata[xpar]
	yplot = pdata[ypar]

	if xpar == 'xray_luminosity':
		xmin, xmax = 1e36,1e44
		#xplot = np.clip(pdata['xray_luminosity'],xmin,np.inf)
		xerr_1d = pdata['xray_luminosity_err']
	elif xpar == 'lsfr':
		xmin, xmax = 1e36,1e45
		xmin, xmax = 5e-3,1e3
		#xplot = np.clip(pdata[xpar],xmin,xmax)
	else:
		xmin, xmax = 5e-6,1e-1
		#xplot = np.clip(pdata[xpar],xmin,xmax)

	#### plot photometry
	if color_by_wise:
		fig, ax = plt.subplots(1,1, figsize=(9.5, 8))
	else:
		fig, ax = plt.subplots(1,1, figsize=(8, 8))
	'''
	if color_by_observatory:
		observatories = np.unique(pdata['observatory'])
		cmap = get_cmap(observatories.shape[0])
		for i,obs in enumerate(observatories):
			idx = pdata['observatory'] == obs

			yerr =  threed_dutils.asym_errors(pdata[ypar], 
				                              pdata[ypar+'_up'],
				                              pdata[ypar+'_down'])
			if xpar == 'xray_luminosity':
				xerr = xerr_1d
			else:
				xerr =  threed_dutils.asym_errors(np.clip(pdata[xpar],xmin,xmax), 
					                              np.clip(pdata[xpar+'_up'],xmin,xmax),
					                              np.clip(pdata[xpar+'_down'],xmin,xmax))
			ax.errorbar(xplot, yplot, yerr=yerr, xerr=xerr, label=obs, color=cmap(i),
			            **plotopts)
	elif color_by_database:
		database = np.unique(pdata['database'])
		cmap = get_cmap(database.shape[0])
		for i,data in enumerate(database):
			idx = pdata['database'] == data

			yerr =  threed_dutils.asym_errors(pdata[ypar], 
				                              pdata[ypar+'_up'],
				                              pdata[ypar+'_down'])

			if xpar == 'xray_luminosity':
				xerr = xerr_1d
			else:
				xerr =  threed_dutils.asym_errors(np.clip(pdata[xpar],xmin,xmax), 
					                              np.clip(pdata[xpar+'_up'],xmin,xmax),
					                              np.clip(pdata[xpar+'_down'],xmin,xmax))

			ax.errorbar(xplot, yplot, yerr=yerr, xerr=xerr, label=data,color=cmap(i),
			            **plotopts)

	elif color_by_wise:

		yerr =  threed_dutils.asym_errors(pdata[ypar], 
			                              pdata[ypar+'_up'],
			                              pdata[ypar+'_down'])

		if xpar == 'xray_luminosity':
			xerr = xerr_1d
		else:
			xerr =  threed_dutils.asym_errors(np.clip(pdata[xpar],xmin,xmax), 
				                              np.clip(pdata[xpar+'_up'],xmin,xmax),
				                              np.clip(pdata[xpar+'_down'],xmin,xmax))

		ax.errorbar(xplot, yplot, yerr=yerr, xerr=xerr,ms=0.0,zorder=-5,
		            **plotopts)
		pts = ax.scatter(xplot, yplot, marker='o', c=pdata['w1w2'], cmap=plt.cm.jet,s=70,zorder=10)

		#### label and add colorbar
		cb = fig.colorbar(pts, ax=ax, aspect=10)
		cb.set_label('W1 - W2 [Vega]')
		cb.solids.set_rasterized(True)
		cb.solids.set_edgecolor("face")

	else: 
	'''
	yerr =  threed_dutils.asym_errors(pdata[ypar], 
		                              pdata[ypar+'_up'],
		                              pdata[ypar+'_down'])

	if xpar == 'xray_luminosity':
		xerr = xerr_1d
	else:
		xerr =  threed_dutils.asym_errors(pdata[xpar],
			                              pdata[xpar+'_up'],
			                              pdata[xpar+'_down'])

	cidx = np.ones_like(pdata[xpar],dtype=bool)
	cidx[idx] = False

	if sf_flag:

		for ind, shape in zip([cidx,~cidx],[popts['nofmir_shape'],popts['fmir_shape']]):
			lower_sigma = pdata['lsfr_down'][ind]
			significant = lower_sigma > 1

			ax.errorbar(xplot[ind][significant], yplot[ind][significant], 
				        yerr=[yerr[0][ind][significant],yerr[1][ind][significant]], xerr=xerr[ind][significant],
				        zorder=-5,ms=0.0,
			            **plotopts)
			ax.scatter(xplot[ind][significant], yplot[ind][significant], marker=shape, color=red,s=70,zorder=11,alpha=0.9)
			ax.errorbar(xplot[ind][~significant], yplot[ind][~significant], 
				        yerr=[yerr[0][ind][~significant],yerr[1][ind][~significant]], xerr=xerr[ind][~significant],
				        zorder=-5,ms=0.0,
			            **plotopts)
			ax.scatter(xplot[ind][~significant], yplot[ind][~significant], marker=shape, color=blue,s=70,zorder=10,alpha=0.9)

			ax.text(0.98,0.18,r'L$_{\mathrm{X}}$ consistent',transform=ax.transAxes,color=blue,ha='right')
			ax.text(0.98,0.14,'with XRBs',transform=ax.transAxes,color=blue,ha='right')
			ax.text(0.98,0.10,r'L$_{\mathrm{X}}$ inconsistent',transform=ax.transAxes,color=red,ha='right')
			ax.text(0.98,0.06,'with XRBs',transform=ax.transAxes,color=red,ha='right')
	else:
		ax.errorbar(xplot, yplot, yerr=yerr, xerr=xerr,ms=0.0,zorder=-2,
		            **plotopts)
		ax.scatter(xplot[cidx], yplot[cidx], marker=popts['nofmir_shape'], color=blue,s=70,zorder=10,alpha=0.9)
		ax.scatter(xplot[~cidx], yplot[~cidx], marker=popts['fmir_shape'], color=blue,s=70,zorder=10,alpha=0.9)


	ax.set_ylabel(ylabel)
	ax.set_xlabel(xlabel)
	ax.set_xlim(xmin*0.5,xmax*2)
	if xpar == 'xray_luminosity':
		ax.set_xscale('log',nonposx='clip',subsx=([1]))
		loc = 2
	else:
		ax.set_xscale('log',nonposx='clip',subsx=([1]))
		loc = 4
	try:
		ax.set_yscale('log',nonposy='clip',subsy=([1]))
	except ValueError:
		print 1/0

	if color_by_observatory or color_by_database:
		if color_by_observatory:
			title = 'Observatory'
		else:
			title = 'Database'
		ax.legend(title=title,loc=loc,prop={'size':8},)
	ax.text(0.05,0.05, r'N$_{\mathrm{match}}$='+str((pdata['xray_luminosity'] > xmin).sum()), transform=ax.transAxes)

	return fig, ax















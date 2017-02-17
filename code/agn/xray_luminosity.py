import numpy as np
import brown_io
import matplotlib.pyplot as plt
import threed_dutils
import os
from astropy.cosmology import WMAP9
import magphys_plot_pref
from corner import quantile

dpi = 150

minorFormatter = magphys_plot_pref.jLogFormatter(base=10, labelOnlyBase=False)
majorFormatter = magphys_plot_pref.jLogFormatter(base=10, labelOnlyBase=True)

popts = {
		 'fmt':'o',
		 'ecolor':'k',
		 'capthick':2,
		 'elinewidth':2,
		 'alpha':0.5
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

	#### for each object
	fagn, fagn_up, fagn_down, mass, luv_lir, xray_lum, xray_lum_err, database, observatory = [], [], [], [], [], [], [], [], []
	fagn_obs, fagn_obs_up, fagn_obs_down = [], [], []
	lagn, lagn_up, lagn_down, lsfr, lsfr_up, lsfr_down = [], [], [], [], [], []
	sfr, sfr_up, sfr_down, ssfr, ssfr_up, ssfr_down, d2, d2_up, d2_down = [[] for i in range(9)]
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

		#### model f_agn
		fagn.append(dat['pquantiles']['q50'][parnames=='fagn'][0])
		fagn_up.append(dat['pquantiles']['q84'][parnames=='fagn'][0])
		fagn_down.append(dat['pquantiles']['q16'][parnames=='fagn'][0])

		#### L_UV / L_IR
		luv_lir.append(dat['bfit']['luv']/dat['bfit']['lir'])

		#### model L_AGN
		lagn.append(dat['pextras']['q50'][eparnames=='l_agn'][0])
		lagn_up.append(dat['pextras']['q84'][eparnames=='l_agn'][0])
		lagn_down.append(dat['pextras']['q16'][eparnames=='l_agn'][0])

		#### x-ray fluxes
		# match
		idx = xray['objname'] == dat['objname']
		if idx.sum() != 1:
			print 1/0
		xflux = xray['flux'][idx][0]
		xflux_err = xray['flux_err'][idx][0]

		# flux is in ergs / cm^2 / s, convert to erg /s 
		z = dat['residuals']['phot']['z']
		dfactor = 4*np.pi*(WMAP9.luminosity_distance(z).cgs.value)**2
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
		nsamp = 10000
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
	out['luv_lir'] = luv_lir
	out['fagn'] = fagn
	out['fagn_up'] = fagn_up
	out['fagn_down'] = fagn_down
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

def make_plot(runname='brownseds_agn',alldata=None,outfolder=None,maxradius=30):

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
	cbw = True

	### PLOT VERSUS OBSERVED X-RAY FRACTION
	outname = 'fagn_obs_fagn_model.png'
	fig,ax = plot(pdata,color_by_observatory=cbo,color_by_database=cbd,color_by_wise=cbw,
		          ypar='fagn',ylabel = r'model L$_{\mathrm{AGN}}$ (IR) / L$_{\mathrm{galaxy}}$ (bolometric)',
		          xpar='fagn_obs',xlabel = r'observed L$_{\mathrm{AGN}}$ (X-ray) / L$_{\mathrm{galaxy}}$ (bolometric)')
	plt.savefig(outfolder+outname,dpi=dpi)
	plt.close()

	outname = 'fagn_obs_lagn.png'
	fig,ax = plot(pdata,color_by_observatory=cbo,color_by_database=cbd,color_by_wise=cbw,
		          ypar='lagn',ylabel = r'model L$_{\mathrm{AGN}}$ [erg/s]',
		          xpar='fagn_obs',xlabel = r'observed L$_{\mathrm{AGN}}$ (X-ray) / L$_{\mathrm{galaxy}}$ (bolometric)')
	plt.savefig(outfolder+outname,dpi=dpi)
	plt.close()

	### PLOT VERSUS OBSERVED X-RAY FLUX
	outname = 'xray_lum_fagn_model.png'
	fig,ax = plot(pdata,color_by_observatory=cbo,color_by_database=cbd,color_by_wise=cbw,
		          ypar='fagn',ylabel = r'model L$_{\mathrm{AGN}}$ (IR) / L$_{\mathrm{galaxy}}$ (bolometric)')
	plt.savefig(outfolder+outname,dpi=dpi)
	plt.close()

	outname = 'xray_lum_lagn.png'
	fig,ax = plot(pdata,color_by_observatory=cbo,color_by_database=cbd,color_by_wise=cbw,
		          ypar='lagn',ylabel = r'model L$_{\mathrm{AGN}}$ [erg/s]')
	plt.savefig(outfolder+outname,dpi=dpi)
	plt.close()

	### PLOT VERSUS 'TRUE' X-RAY FLUX
	outname = 'xray_lum_sfrcorr_fagn_model.png'
	fig,ax = plot(pdata,color_by_observatory=cbo,color_by_database=cbd,color_by_wise=cbw,
		          ypar='fagn',ylabel = r'model L$_{\mathrm{AGN}}$ (IR) / L$_{\mathrm{galaxy}}$ (bolometric)',
		          xpar='lsfr',xlabel = '(observed X-ray) / (expected SFR X-ray) [erg/s]')
	plt.savefig(outfolder+outname,dpi=dpi)
	plt.close()

	outname = 'xray_lum_sfrcorr_lagn.png'
	fig,ax = plot(pdata,color_by_observatory=cbo,color_by_database=cbd,color_by_wise=cbw,
		          ypar='lagn',ylabel = r'model L$_{\mathrm{AGN}}$ [erg/s]',
		          xpar='lsfr',xlabel = '(observed X-ray) / (expected SFR X-ray) [erg/s]')
	plt.savefig(outfolder+outname,dpi=dpi)
	plt.close()

	### SFR, sSFR, dust2, LUV/LIR versus FAGN
	fig,ax = plot_model_corrs(pdata)
	plt.tight_layout()
	plt.savefig(outfolder+'fagn_versus_galaxy_properties.png',dpi=dpi)
	plt.close()

def plot_model_corrs(pdata):

	fig, ax = plt.subplots(2,2, figsize=(10, 10))
	ax = np.ravel(ax)

	#### fagn labeling
	xlabel = r'log(f$_{\mathrm{AGN}}$)'
	x = np.log10(pdata['fagn'])
	xerr =  threed_dutils.asym_errors(pdata['fagn'], 
		                              pdata['fagn_up'],
		                              pdata['fagn_down'],log=True)

	#### y-axis
	ypar = ['sfr','ssfr','d2']
	ylabels = [r'log(SFR) [M$_{\odot}$/yr]', r'log(sSFR) [yr$^{-1}$]', 'diffuse dust optical depth']
	for ii, yp in enumerate(ypar):

		if 'sfr' in yp:
			log = True
			y = np.log10(pdata[yp])
		else:
			log = False
			y = pdata[yp]

		yerr =  threed_dutils.asym_errors(pdata[yp], 
			                              pdata[yp+'_up'],
			                              pdata[yp+'_down'],log=log)
		ax[ii].errorbar(x,y,yerr=yerr, xerr=xerr, ms=0.0,zorder=-2,**popts)
		ax[ii].plot(x,y, 'o',alpha=0.9,color = '#1C86EE')
		ax[ii].set_xlabel(xlabel)
		ax[ii].set_ylabel(ylabels[ii])

	#### luv / lir (no errors!)
	y = 1/pdata['luv_lir']
	ax[-1].errorbar(x, np.log10(y), xerr=xerr, ms=0.0, zorder=-2, **popts)
	ax[-1].plot(x,np.log10(y), 'o',alpha=0.9,color = '#1C86EE')
	ax[-1].set_xlabel(xlabel)
	ax[-1].set_ylabel(r'log(L$_{\mathrm{IR}}$/L$_{\mathrm{UV}}$)')

	return fig, ax

def plot(pdata,color_by_observatory=False,color_by_database=False,color_by_wise=False,
	     ypar=None, ylabel=None, 
	     xpar='xray_luminosity', xlabel=r'observed X-ray luminosity [erg/s]'):
	'''
	plots a color-color BPT scatterplot
	'''

	#### generate x, y values
	yplot = pdata[ypar]

	if xpar == 'xray_luminosity':
		xmin, xmax = 1e36,1e44
		xplot = np.clip(pdata['xray_luminosity'],xmin,np.inf)
		xerr_1d = pdata['xray_luminosity_err']
	elif xpar == 'lsfr':
		xmin, xmax = 1e36,1e45
		xmin, xmax = 5e-3,1e3
		xplot = np.clip(pdata[xpar],xmin,xmax)
	else:
		xmin, xmax = 5e-6,1e-1
		xplot = np.clip(pdata[xpar],xmin,xmax)

	#### plot photometry
	if color_by_wise:
		fig, ax = plt.subplots(1,1, figsize=(9.5, 8))
	else:
		fig, ax = plt.subplots(1,1, figsize=(8, 8))

	if color_by_observatory:
		observatories = np.unique(pdata['observatory'])
		cmap = get_cmap(observatories.shape[0])
		for i,obs in enumerate(observatories):
			idx = pdata['observatory'] == obs

			yerr =  threed_dutils.asym_errors(pdata[ypar][idx], 
				                              pdata[ypar+'_up'][idx],
				                              pdata[ypar+'_down'][idx])
			if xpar == 'xray_luminosity':
				xerr = xerr_1d[idx]
			else:
				xerr =  threed_dutils.asym_errors(np.clip(pdata[xpar][idx],xmin,xmax), 
					                              np.clip(pdata[xpar+'_up'][idx],xmin,xmax),
					                              np.clip(pdata[xpar+'_down'][idx],xmin,xmax))
			ax.errorbar(xplot[idx], yplot[idx], yerr=yerr, xerr=xerr, label=obs, color=cmap(i),
			            **popts)
	elif color_by_database:
		database = np.unique(pdata['database'])
		cmap = get_cmap(database.shape[0])
		for i,data in enumerate(database):
			idx = pdata['database'] == data

			yerr =  threed_dutils.asym_errors(pdata[ypar][idx], 
				                              pdata[ypar+'_up'][idx],
				                              pdata[ypar+'_down'][idx])

			if xpar == 'xray_luminosity':
				xerr = xerr_1d[idx]
			else:
				xerr =  threed_dutils.asym_errors(np.clip(pdata[xpar][idx],xmin,xmax), 
					                              np.clip(pdata[xpar+'_up'][idx],xmin,xmax),
					                              np.clip(pdata[xpar+'_down'][idx],xmin,xmax))

			ax.errorbar(xplot[idx], yplot[idx], yerr=yerr, xerr=xerr, label=data,color=cmap(i),
			            **popts)

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
		            **popts)
		pts = ax.scatter(xplot, yplot, marker='o', c=pdata['w1w2'], cmap=plt.cm.jet,s=70,zorder=10)

		#### label and add colorbar
		cb = fig.colorbar(pts, ax=ax, aspect=10)
		cb.set_label('W1 - W2 [Vega]')
		cb.solids.set_rasterized(True)
		cb.solids.set_edgecolor("face")

	else: 
		yerr =  threed_dutils.asym_errors(pdata[ypar], 
			                              pdata[ypar+'_up'],
			                              pdata[ypar+'_down'])

		if xpar == 'xray_luminosity':
			xerr = xerr_1d
		else:
			xerr =  threed_dutils.asym_errors(np.clip(pdata[xpar],xmin,xmax), 
				                              np.clip(pdata[xpar+'_up'],xmin,xmax),
				                              np.clip(pdata[xpar+'_down'],xmin,xmax))

		ax.errorbar(xplot, yplot, yerr=yerr, xerr=xerr,
		            **popts)

	ax.set_ylabel(ylabel)
	ax.set_xlabel(xlabel)
	ax.set_xlim(xmin*0.5,xmax*2)
	if xpar == 'xray_luminosity':
		ax.set_xscale('log',nonposx='clip',subsx=([1]))
		loc = 2
	else:
		ax.set_xscale('log',nonposx='clip',subsx=([1]))
		loc = 4
	ax.set_yscale('log',nonposy='clip',subsy=([1]))

	if color_by_observatory or color_by_database:
		if color_by_observatory:
			title = 'Observatory'
		else:
			title = 'Database'
		ax.legend(title=title,loc=loc,prop={'size':8},)
	ax.text(0.05,0.05, r'N$_{\mathrm{match}}$='+str((pdata['xray_luminosity'] > xmin).sum()), transform=ax.transAxes)

	return fig, ax















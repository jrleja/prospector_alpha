import numpy as np
import brown_io
import matplotlib.pyplot as plt
import threed_dutils
import os
from astropy.cosmology import WMAP9
import magphys_plot_pref

dpi = 150

minorFormatter = magphys_plot_pref.jLogFormatter(base=10, labelOnlyBase=False)
majorFormatter = magphys_plot_pref.jLogFormatter(base=10, labelOnlyBase=True)

def collate_data(alldata):

	### preliminary stuff
	parnames = alldata[0]['pquantiles']['parnames']
	xray = brown_io.load_xray_mastercat(xmatch = True)

	#### for each object
	fagn, fagn_up, fagn_down, mass, xray_lum = [], [], [], [], []
	for ii, dat in enumerate(alldata):
		
		#### photometric f_agn
		fagn.append(dat['pquantiles']['q50'][parnames=='fagn'][0])
		fagn_up.append(dat['pquantiles']['q84'][parnames=='fagn'][0])
		fagn_down.append(dat['pquantiles']['q16'][parnames=='fagn'][0])
		mass.append(10**dat['pquantiles']['q50'][parnames=='logmass'][0])
		
		#### x-ray fluxes
		# match
		idx = xray['objname'] == dat['objname']
		if idx.sum() != 1:
			print 1/0
		xflux = xray['flux'][idx][0]

		# flux is in ergs / cm^2 / s, convert to Lsun
		lsun  = 3.839E33
		z = dat['residuals']['phot']['z']
		dfactor = 4*np.pi*(WMAP9.luminosity_distance(z).cgs.value)**2
		xray_lum.append(xflux * dfactor)

	out = {}
	out['mass'] = mass
	out['fagn'] = fagn
	out['fagn_up'] = fagn_up
	out['fagn_down'] = fagn_down
	out['xray_luminosity'] = xray_lum

	for key in out.keys(): out[key] = np.array(out[key])

	return out

def make_plot(runname='brownseds_agn',alldata=None,outfolder=None):

	#### load alldata
	if alldata is None:
		alldata = brown_io.load_alldata(runname=runname)

	#### make output folder if necessary
	if outfolder is None:
		outfolder = os.getenv('APPS')+'/threedhst_bsfh/plots/'+runname+'/agn_plots/'
		if not os.path.isdir(outfolder):
			os.makedirs(outfolder)

	#### collate data
	pdata = collate_data(alldata)

	### PLOT
	fig,ax = plot(pdata)
	plt.savefig(outfolder+'xray_lum_fagn.png',dpi=dpi)
	plt.close()
	os.system('open '+outfolder+'xray_lum_fagn.png')

def plot(pdata):
	'''
	plots a color-color BPT scatterplot
	'''

	#### generate x, y values
	xmin = 1e34
	yerr =  threed_dutils.asym_errors(pdata['fagn']*pdata['mass'], 
			                          pdata['fagn_up']*pdata['mass'],
			                          pdata['fagn_down']*pdata['mass'])
	yplot = pdata['fagn']*pdata['mass']
	xplot = np.clip(pdata['xray_luminosity'],xmin,np.inf)

	#### plot photometry
	fig, ax = plt.subplots(1,1, figsize=(10, 10))

	ax.errorbar(xplot, yplot, yerr=yerr,
	            fmt='o', ecolor='k', capthick=2,elinewidth=2,alpha=0.5,zorder=-5)

	ax.set_ylabel(r'log(f$_{\mathrm{AGN}}$)')
	ax.set_xlabel(r'X-ray luminosity [erg/s]')

	ax.set_xscale('log',nonposx='clip',subsx=([1]))

	ax.set_yscale('log',nonposy='clip',subsy=([1]))
	ax.yaxis.set_minor_formatter(minorFormatter)
	ax.yaxis.set_major_formatter(majorFormatter)

	return fig, ax















from matplotlib import pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.data import download_file
from astropy.stats import sigma_clip
import os
import numpy as np
from astropy.coordinates import SkyCoord  # High-level coordinates
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
from astropy.coordinates import Angle, Latitude, Longitude  # Angles
import astropy.units as u
from reproject import reproject_exact
import copy
from astropy.convolution import convolve, convolve_fft
from wise_colors import vega_conversions
from magphys_plot_pref import jLogFormatter
import brown_io
from corner import quantile
from prosp_dutils import smooth_spectrum

plt.ioff()

c = 3e18   # angstroms per second
minorFormatter = jLogFormatter(base=10, labelOnlyBase=False)
majorFormatter = jLogFormatter(base=10, labelOnlyBase=True)

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
	pnames = ['fagn', 'agn_tau']
	for p in pnames: 
		model_pars[p] = {'q50':[],'q84':[],'q16':[]}
	parnames = alldata[0]['pquantiles']['parnames']

	#### load information
	for dat in alldata:
		objname.append(dat['objname'])

		#### model parameters
		for key in model_pars.keys():
			model_pars[key]['q50'].append(dat['pquantiles']['q50'][parnames==key][0])
			model_pars[key]['q84'].append(dat['pquantiles']['q84'][parnames==key][0])
			model_pars[key]['q16'].append(dat['pquantiles']['q16'][parnames==key][0])

	#### X-ray information
	# match
	eparnames = alldata[0]['pextras']['parnames']
	xr_idx = eparnames == 'xray_lum'
	xray = brown_io.load_xray_cat(xmatch = True)

	lsfr, lsfr_up, lsfr_down, xray_lum, xray_lum_err = [], [], [], [], []
	for dat in alldata:
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

		##### L_OBS / L_SFR(MODEL)
		# sample from the chain, assume gaussian errors for x-ray fluxes
		nsamp = 10000
		chain = dat['pextras']['flatchain'][:,xr_idx].squeeze()
		scale = xray_lum_err[-1]
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

	#### numpy arrays
	for key in model_pars.keys(): 
		for key2 in model_pars[key].keys():
			model_pars[key][key2] = np.array(model_pars[key][key2])

	out = {'pars':model_pars,'objname':objname}

	out['lsfr'] = np.array(lsfr)
	out['lsfr_up'] = np.array(lsfr_up)
	out['lsfr_down'] = np.array(lsfr_down)
	out['xray_luminosity'] = np.array(xray_lum)
	out['xray_luminosity_err'] = np.array(xray_lum_err)
	out['bpt'] = brown_io.return_agn_str(np.ones_like(alldata,dtype=bool),string=True)

	return out

def collate_spectra(alldata,alldata_noagn,idx_plot,pdata,runname,contours):

	'''
	check for AGN on and AGN off in alldata
	if it doesn't exist, make it
	if it does exist, use it.
	'''

	filters = ['wise_w1', 'wise_w2']
	agn_on_colors, agn_off_colors, obs_colors, agn_on_spec, agn_off_spec, lam, spit_flux, spit_lam, ak_flux, ak_lam, objname = [],[],[],[],[],[],[],[],[],[],[]
	for ii,idx in enumerate(idx_plot):
		dat = alldata[idx]
		dat_noagn = alldata_noagn[idx]
		objname.append(dat['objname'])
		
		f1_idx = dat['filters'] == filters[0]
		f2_idx = dat['filters'] == filters[1]

		if (f1_idx.sum() == 0) | (f2_idx.sum() == 0):
			noagn = np.nan
			agn = np.nan
			obs = np.nan
		else:
			vega_conv = (vega_conversions(contours[0]) - vega_conversions(contours[1]))
			agn_color = np.median(-2.5*np.log10(dat['model_maggies'][f1_idx,:])+2.5*np.log10(dat['model_maggies'][f2_idx,:]))+vega_conv
			noagn_color = np.median(-2.5*np.log10(dat_noagn['model_maggies'][f1_idx,:])+2.5*np.log10(dat_noagn['model_maggies'][f2_idx,:]))+vega_conv
			obs_color = -2.5*np.log10(dat['obs_maggies'][f1_idx])+2.5*np.log10(dat['obs_maggies'][f2_idx])+vega_conv

		agn_on_colors.append(agn_color)
		agn_off_colors.append(noagn_color)
		obs_colors.append(obs_color[0])

		### spectra
		agn_on_spec.append(dat['model_spec'])
		agn_off_spec.append(dat_noagn['model_spec'])
		lam.append(dat['model_spec_lam'])

		### observed spectra
		obs_spec = brown_io.load_spectra(dat['objname'])
		spit_idx = obs_spec['source'] == 3
		if spit_idx.sum() > 0:
			spit_lam.append(obs_spec['rest_lam'][spit_idx])
			modspec = smooth_spectrum(spit_lam[-1],obs_spec['flux'][spit_idx],2500)
			spit_flux.append(modspec * spit_lam[-1]/c)
		else:
			spit_lam.append(np.nan)
			spit_flux.append(np.nan)
		ak_idx = obs_spec['source'] == 2
		if ak_idx.sum() > 0:
			ak_lam.append(obs_spec['rest_lam'][ak_idx])
			ak_flux.append(obs_spec['flux'][ak_idx] * ak_lam[-1]/c)
		else:
			ak_lam.append(np.nan)
			ak_flux.append(np.nan)

	out = {}
	out['agn_on_mag'] = agn_on_colors
	out['agn_off_mag'] = agn_off_colors
	out['obs_mag'] = obs_colors
	out['agn_on_spec'] = agn_on_spec
	out['agn_off_spec'] = agn_off_spec
	out['wave'] = lam
	out['filters'] = filters
	out['spit_lam'] = spit_lam
	out['spit_flux'] = spit_flux
	out['ak_lam'] = ak_lam
	out['ak_flux'] = ak_flux
	out['objname'] = objname
	pdata['observables'] = out
	return pdata

def plot_all(runname='brownseds_agn',runname_noagn='brownseds_np',alldata=None,alldata_noagn=None,outfolder=None):

	#### load alldata
	if alldata is None:
		alldata = brown_io.load_alldata(runname=runname)
		alldata_noagn = brown_io.load_alldata(runname=runname_noagn)

	#### make output folder if necessary
	if outfolder is None:
		outfolder = os.getenv('APPS')+'/threedhst_bsfh/plots/'+runname+'/agn_plots/sdss_overlays'
		if not os.path.isdir(outfolder):
			os.makedirs(outfolder)

	#### collate data
	pdata = collate_data(alldata)

	#### select data to plot. generate SEDs with no AGN contribution if necessary.
	# 10 most massive AGN
	contours = ['WISE W1','WISE W2']
	idx_plot = pdata['pars']['fagn']['q50'].argsort()[:10][::-1]
	#idx_plot = pdata['pars']['fagn']['q50'].argsort()[-22:][::-1]
	pdata = collate_spectra(alldata,alldata_noagn,idx_plot,pdata,runname,contours)

	#### plot data
	plot_composites(pdata,idx_plot,outfolder,contours)

def plot_composites(pdata,idx_plot,outfolder,contours,contour_colors=True,calibration_plot=True):

	### open figure
	#fig, ax = plt.subplots(5,2, figsize=(7, 15))
	#fig, ax = plt.subplots(1,1, figsize=(7, 7),subplot_kw={'projection': ccrs.PlateCarree()})
	#ax = np.ravel(ax)

	### image qualities
	fs = 10
	maxlim = 0.05

	### filters
	#filters = ['SDSS u','SDSS g','SDSS i']
	#fcolors = ['Blues','Greens','Reds']
	#ftext = ['blue','green','red']
	filters = ['SDSS i']
	fcolors = ['Greys']
	ftext = ['black']

	### contour color limits (customized for W1-W2)
	color_limits = [-1.0,2.6]

	kernel = None

	### begin loop
	for ii,idx in enumerate(idx_plot):

		### load object information
		objname = pdata['objname'][idx]
		fagn = pdata['pars']['fagn']['q50'][idx]
		ra, dec = load_coordinates(objname)
		phot_size = load_structure(objname,long_axis=True) # in arcseconds

		### set up figure
		fig, ax = None, None
		xs, ys, dely = 0.05,0.9, 0.07

		for kk,filt in enumerate(filters):
			hdu = load_image(objname,filt)

			#### if it's the first filter,
			#### set up WCS using this information
			if fig == None:

				### grab WCS information, create figure + axis
				wcs = WCS(hdu.header)
				fig, ax = plt.subplots(2,3, figsize=(18, 18))
				plt.subplots_adjust(top=0.95,bottom=0.33)
				sedax = fig.add_axes([0.3,0.05,0.4,0.25])
				ax = np.ravel(ax)

				### translate object location into pixels using WCS coordinates
				pix_center = wcs.all_world2pix([[ra[0],dec[0]]],1)
				size = calc_dist(wcs, pix_center, phot_size, hdu.data.shape)
				hdu_original = copy.deepcopy(hdu.header)
				data_to_plot = hdu.data

				### build image extents
				# first calculate pixel location of image left, image bottom
				center_pix = np.atleast_2d([(size[0]+size[1])/2.,(size[2]+size[3])/2.])
				center_left_pix = np.atleast_2d([size[0],center_pix[0][1]])
				center_bottom_pix = np.atleast_2d([center_pix[0][0],size[2]])

				# now wcs location
				center_left_wcs = wcs.all_pix2world(center_left_pix,0)
				center_bottom_wcs = wcs.all_pix2world(center_bottom_pix,0)
				center_wcs = wcs.all_pix2world(center_pix,0)

				# now calculate distance
				center = SkyCoord(ra=center_wcs[0][0]*u.degree,dec=center_wcs[0][1]*u.degree)
				center_left = SkyCoord(ra=center_left_wcs[0][0]*u.degree,dec=center_left_wcs[0][1]*u.degree)
				center_bottom = SkyCoord(ra=center_bottom_wcs[0][0]*u.degree,dec=center_bottom_wcs[0][1]*u.degree)
				ydist = center.separation(center_bottom).arcsec
				xdist = center.separation(center_left).arcsec

				extent = [-xdist,xdist,-ydist,ydist]

			#### if it's not the first filter,
			#### project into WCS of first filter
			# see reprojection https://reproject.readthedocs.io/en/stable/
			else:
				data_to_plot, footprint = reproject_exact(hdu, hdu_original)

			plot_image(ax[5],data_to_plot[size[2]:size[3],size[0]:size[1]],size,cmap=fcolors[kk],extent=extent)
			ax[5].text(xs, ys, filters[kk]+'-band',color=ftext[kk],transform=ax[5].transAxes)
			ys -= dely

			### draw 6" line
			wise_psf = 6 # in arcseconds
			start = -0.85*xdist
			ax[5].plot([start,start+wise_psf],[start,start],lw=2,color='k')
			ax[5].text(start+wise_psf/2.,start+1, '6"', ha='center')
			ax[5].set_xlim(-xdist,xdist) # reset plot limits b/c of text stuff
			ax[5].set_ylim(-ydist,ydist)

		ax[5].set_xlabel('arcseconds')
		ax[5].set_ylabel('arcseconds')

		#### load up HDU, subtract background and convert to physical units
		# also convolve to W2 resolution
		hdu = load_image(objname,contours[0])
		hdu.data *= 1.9350E-06 ### convert from DN to flux in Janskies, from this table: http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec2_3f.html
		hdu.data -= np.median(hdu.data) ### subtract background as median
		data1_noconv, footprint = reproject_exact(hdu, hdu_original)
		data_convolved, kernel = match_resolution(hdu.data,contours[0],contours[1],kernel=kernel,data1_res=hdu.header['PXSCAL1']) # convolve to W2 resolution
		hdu.data = data_convolved
		data1, footprint = reproject_exact(hdu, hdu_original)

		### load up HDU2, subtract background, convert to physical units
		hdu = load_image(objname,contours[1])
		hdu.data -= np.median(hdu.data) ### subtract background as median
		hdu.data *= 2.7048E-06 ### convert from DN to flux in Janskies, from this table: http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec2_3f.html

		#### put onto same scale
		data2, footprint = reproject_exact(hdu, hdu_original)

		### plot the main result
		data1_slice = data1[size[2]:size[3],size[0]:size[1]]
		data2_slice = data2[size[2]:size[3],size[0]:size[1]]
		plot_color_contour(ax[5],data1_slice, data2_slice, contours[0],contours[1], maxlim=maxlim, color_limits=color_limits)

		#ax[5].text(xs, ys, 'contours:' +contours[0]+'-'+contours[1],transform=ax[5].transAxes)
		ys -= dely

		### labels and limits
		ax[5].text(0.98,0.93,objname,transform=ax[5].transAxes,ha='right')
		ax[5].text(0.98,0.88,r'f$_{\mathrm{MIR}}$='+"{:.2f}".format(fagn),transform=ax[5].transAxes, ha='right')
		ax[5].set_title('WISE colors on\nSDSS imaging')

		#### CALIBRATION PLOT
		flux_color = convert_to_color(data1_slice, data2_slice,contours[0],contours[1],minflux=1e-10)

		img = ax[0].imshow(data1_noconv[size[2]:size[3],size[0]:size[1]], origin='lower',extent=extent)
		cbar = fig.colorbar(img, ax=ax[0])
		cbar.formatter.set_powerlimits((0, 0))
		cbar.update_ticks()
		ax[0].set_title(contours[0]+', \n raw')

		img = ax[1].imshow(data1_slice, origin='lower',extent=extent)
		cbar = fig.colorbar(img, ax=ax[1])
		cbar.formatter.set_powerlimits((0, 0))
		cbar.update_ticks()
		ax[1].set_title(contours[0]+', \n convolved to W2 PSF')

		img = ax[2].imshow(data2_slice, origin='lower',extent=extent)
		cbar = fig.colorbar(img, ax=ax[2])
		cbar.formatter.set_powerlimits((0, 0))
		cbar.update_ticks()
		ax[2].set_title(contours[1]+', \n raw')

		img = ax[3].imshow(flux_color, origin='lower',extent=extent,vmin=color_limits[0],vmax=color_limits[1])
		cbar = fig.colorbar(img, ax=ax[3])
		ax[3].set_title(contours[0]+'-'+contours[1]+', \n raw')

		### don't trust anything less than X times the max!
		max1 = np.nanmax(data1_slice)
		max2 = np.nanmax(data2_slice)
		background = (data1_slice < max1*maxlim) | (data2_slice < max2*maxlim)
		flux_color[background] = np.nan


		img = ax[4].imshow(flux_color, origin='lower',extent=extent,vmin=color_limits[0],vmax=color_limits[1])
		cbar = fig.colorbar(img, ax=ax[4])
		cbar.formatter.set_powerlimits((0, 0))
		cbar.update_ticks()
		ax[4].set_title(contours[0]+'-'+contours[1]+', \n background removed')

		ax[4].plot([start,start+wise_psf],[start,start],lw=2,color='k')
		ax[4].text(start+wise_psf/2.,start+1, '6"', ha='center')
		ax[4].set_xlim(-xdist,xdist) # reset plot limits b/c of text stuff
		ax[4].set_ylim(-ydist,ydist)

		#### now plot the SED
		agn_color, noagn_color = '#FF3D0D', '#1C86EE'
		wavlims = (1,30)
		wav_idx = (pdata['observables']['wave'][ii]/1e4 > wavlims[0]) & (pdata['observables']['wave'][ii]/1e4 < wavlims[1])
		sedax.plot(pdata['observables']['wave'][ii][wav_idx]/1e4,pdata['observables']['agn_on_spec'][ii][wav_idx], lw=2.5, alpha=0.5, color=agn_color)
		sedax.plot(pdata['observables']['wave'][ii][wav_idx]/1e4,pdata['observables']['agn_off_spec'][ii][wav_idx], lw=2.5, alpha=0.5, color=noagn_color)
		if type(pdata['observables']['spit_lam'][ii]) is np.ndarray:
			wav_idx = (pdata['observables']['spit_lam'][ii]/1e4 > wavlims[0]) & (pdata['observables']['spit_lam'][ii]/1e4 < wavlims[1])
			sedax.plot(pdata['observables']['spit_lam'][ii][wav_idx]/1e4,pdata['observables']['spit_flux'][ii][wav_idx], lw=2.5, alpha=0.5, color='black')
		if type(pdata['observables']['ak_lam'][ii]) is np.ndarray:
			wav_idx = (pdata['observables']['ak_lam'][ii]/1e4 > wavlims[0]) & (pdata['observables']['ak_lam'][ii]/1e4 < wavlims[1])
			sedax.plot(pdata['observables']['ak_lam'][ii][wav_idx]/1e4,pdata['observables']['ak_flux'][ii][wav_idx], lw=2.5, alpha=0.5, color='black')

		### write down Vega colors
		sedax.text(0.95,0.1,'W1-W2(AGN ON)='+'{:.2f}'.format(pdata['observables']['agn_on_mag'][ii]),transform=sedax.transAxes,color=agn_color,ha='right')
		sedax.text(0.95,0.16,'W1-W2(AGN OFF)='+'{:.2f}'.format(pdata['observables']['agn_off_mag'][ii]),transform=sedax.transAxes,color=noagn_color,ha='right')
		sedax.text(0.95,0.22,'W1-W2(OBS)='+'{:.2f}'.format(pdata['observables']['obs_mag'][ii]),transform=sedax.transAxes,color='black',ha='right')

		lsfr = pdata['lsfr'][idx]
		if lsfr > 0:
			sedax.text(1.15,0.5,r'L$_{\mathrm{X}}$(obs)/L$_{\mathrm{X}}$(SFR)='+'{:.2f}'.format(lsfr),transform=sedax.transAxes,color='black',fontsize=18,weight='bold')
		else: 
			sedax.text(1.15,0.5,r'No X-ray information',transform=sedax.transAxes,color='black',fontsize=18,weight='bold')
		bpt = pdata['bpt'][idx]
		if bpt == 'None':
			sedax.text(1.15,0.42,'No BPT measurement',transform=sedax.transAxes,color='black',fontsize=18,weight='bold')
		else: 
			sedax.text(1.15,0.42,'BPT: '+bpt,transform=sedax.transAxes,color='black',fontsize=18,weight='bold')

		### scaling and labels
		sedax.set_yscale('log',nonposx='clip',subsx=(1,2,4))
		sedax.set_xscale('log',nonposx='clip',subsx=(1,2,4))
		sedax.xaxis.set_minor_formatter(minorFormatter)
		sedax.xaxis.set_major_formatter(majorFormatter)

		sedax.set_xlabel(r'wavelength $\mu$m')
		sedax.set_ylabel(r'f$_{\nu}$')

		sedax.axvline(3.4, linestyle='--', color='0.5',lw=1.5,alpha=0.8,zorder=-5)
		sedax.axvline(4.6, linestyle='--', color='0.5',lw=1.5,alpha=0.8,zorder=-5)

		sedax.set_xlim(wavlims)

		padding = ''
		if ii <= 9:
			padding='0'

		plt.savefig(outfolder+'/'+padding+str(ii)+'_'+objname+'.png',dpi=150)
		plt.close()

def calc_dist(wcs, pix_center, size, im_shape):

	'''
	calculate distance from 
	'''

	### define center coordinates
	center_coordinates = SkyCoord.from_pixel(pix_center[0][0],pix_center[0][1],wcs)
	out = np.empty(4)

	### find X UP, X DOWN
	calc_dist = 0.0
	pix = np.floor(pix_center[0][0])
	while (calc_dist < size) and (pix != 0):
		pix -=1
		new_coordinates = SkyCoord.from_pixel(pix,pix_center[0][1],wcs)
		calc_dist = new_coordinates.separation(center_coordinates).arcsec
	out[0] = pix

	calc_dist = 0.0
	pix = np.ceil(pix_center[0][0])
	while (calc_dist < size) and (pix != im_shape[0]-1):
		pix +=1
		new_coordinates = SkyCoord.from_pixel(pix,pix_center[0][1],wcs)
		calc_dist = new_coordinates.separation(center_coordinates).arcsec
	out[1] = pix

	### FIND Y UP, Y DOWN
	calc_dist = 0.0
	pix = np.floor(pix_center[0][1])
	while (calc_dist < size) and (pix != 0):
		pix -=1
		new_coordinates = SkyCoord.from_pixel(pix_center[0][0],pix,wcs)
		calc_dist = new_coordinates.separation(center_coordinates).arcsec
	out[2] = pix

	calc_dist = 0.0
	pix = np.ceil(pix_center[0][1])
	while (calc_dist < size) and (pix != im_shape[0]-1):
		pix +=1
		new_coordinates = SkyCoord.from_pixel(pix_center[0][0],pix,wcs)
		calc_dist = new_coordinates.separation(center_coordinates).arcsec
	out[3] = pix

	return out


def load_image(objname,filter):

	'''
	loads Brown+14 cutouts in a given filter
	'''

	folder = os.getenv('APPS')+'/threedhst_bsfh/data/brownseds_data/fits/'

	filter = filter.replace(' ','_')
	objname = objname.replace(' ','_')
	fits_file = folder + objname+'_'+filter+'.fits'
	hdu = fits.open(fits_file)[0]

	return hdu

def load_structure(objname,long_axis=False):

	'''
	loads structure information from Brown+14 catalog
	'''

	loc = os.getenv('APPS')+'/threedhst_bsfh/data/brownseds_data/photometry/structure.dat'

	with open(loc, 'r') as f: hdr = f.readline().split()[1:]
	dat = np.loadtxt(loc, comments = '#', delimiter=' ',
	 	             dtype = {'names':([n for n in hdr]),\
	  	                      'formats':('S40','S40','S40','S40','S40','S40','S40','S40','S40','S40')})

	match = dat['Name'] == objname.replace(' ','_')
	phot_size = np.array(dat['phot_size'][match][0].split('_'),dtype=np.float)

	if long_axis:
		phot_size = phot_size.max()

	return phot_size

def load_coordinates(objname):

	ra,dec,objnames = brown_io.load_coordinates()
	match = objname == objnames
	
	return ra[match], dec[match]

def plot_contour(ax,data,size,ncontours=20,color='white'):

	'''
	FIRST SHIFT MINIMUM FLUX TO ZERO (?)
	'''

	box_min = np.nanmin(data[size[2]:size[3],size[0]:size[1]])
	box_max = np.nanmax(data[size[2]:size[3],size[0]:size[1]])
	data = np.clip(data,box_min,np.inf)
	data[np.isnan(data)] = box_max

	bmax = box_min*100
	levels = np.linspace(box_min, bmax, ncontours)
	levels = 10**np.linspace(np.log10(box_min), np.log10(bmax), ncontours)

	ax.contour(data, levels=levels, colors=color)

def plot_color_contour(ax,flux1,flux2,filter1,filter2,ncontours=25,color='white',vega_conv=0.0,maxlim=None,color_limits=None):

	#### calculate color
	flux_color = convert_to_color(flux1,flux2,filter1,filter2)

	#### set background equal to NaN
	max1 = np.nanmax(flux1)
	max2 = np.nanmax(flux2)
	background = (flux1 < max1*maxlim) | (flux2 < max2*maxlim)
	flux_color[background] = np.nan

	#### need x,y arrays
	xlim = ax.get_xlim()
	ylim = ax.get_ylim()
	x = np.linspace(xlim[0],xlim[1],flux_color.shape[1])
	y = np.linspace(ylim[0],ylim[1],flux_color.shape[0])

	#### set levels
	levels = np.arange(color_limits[0],color_limits[1],step=0.1)
	cs = ax.contour(x,y,flux_color,levels=levels, linewidths=2,zorder=2)
	ax.clabel(cs, inline=1, fontsize=12)

def convert_to_color(flux1,flux2,filter1,filter2,minflux=1e-10):

	'''
	convert fluxes in fnu into Vega-scale colors
	'''

	flux1_clip = np.clip(flux1,minflux,np.inf)
	flux2_clip = np.clip(flux2,minflux,np.inf)
	vega_conv = (vega_conversions(filter1) - vega_conversions(filter2))

	flux_color = -2.5*np.log10(flux1_clip/flux2_clip)+vega_conv

	return flux_color

def plot_image(ax,data,size,alpha=0.5,cmap='cubehelix',extent=None):

	sigclip = sigma_clip(data,6)
	sigclip = sigclip.filled(np.nanmax(sigclip)) # fill clipped values with max

	'''
	### shift all numbers so that minimum = 0
	try:
		minimum = sigclip[size[0]:size[1],size[2]:size[3]].min()
	except ValueError:
		print 1/0
	if sigclip.min() < 0:
		sigclip += np.abs(minimum)
	else:
		sigclip -= np.abs(minimum)
	'''

	'''
	### put into units of standard deviations
	### start linear scale at 3sigma
	std = np.std(sigclip[size[2]:size[3],size[0]:size[1]])
	sigclip /= std
	sigclip = np.clip(sigclip,1.0,np.inf)
	
	minimum = np.nanmin(sigclip)
	sigclip = sigclip - minimum
	'''
	### show image
	ax.imshow(sigclip, origin='lower', cmap=cmap,alpha=alpha,extent=extent)

def load_wise_psf(filter):
	'''
	given WISE filter name, load a WISE PSF
	downloaded from here: http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4c.html#coadd_psf
	note that this is APPROXIMATE, and best to measure directly from IMAGE
	a more exact version is done here: https://arxiv.org/abs/1106.5065
	this is done in 9 different places on the CCD (use if there's obvious PSF effects)
	'''

	### extract filter number
	fnumber = [s for s in filter if s.isdigit()][0]

	### load PSF
	location = '/Users/joel/code/python/threedhst_bsfh/data/brownseds_data/fits/PSF_W'+fnumber+'.V4.fits'
	hdu = fits.open(location)[0]

	resolution = hdu.header['PSCALE']

	return hdu.data, resolution

def psf_match(f1,f2, test=False, data1_res=1.375):

	# following tutorial here:
	# http://photutils.readthedocs.io/en/latest/photutils/psf_matching.html
	# WARNING: this is sensitive to the windowing!
	# how to properly choose smoothing?

	from astropy.modeling.models import Gaussian2D
	from photutils import create_matching_kernel, CosineBellWindow, TopHatWindow

	#### how large do we want our kernel?
	# images are 1200 x 1200 pixels
	# each pixel is 0.25", WISE PSF FWHM is ~6"
	# take 52" x 52" here for maximum safety
	limits = (500,700)

	#### generate PSFs
	psf1, res1 = load_wise_psf(f1)
	psf2, res2 = load_wise_psf(f2)

	if res1 != res2:
		print 1/0

	#### shrink
	psf1 = psf1[limits[0]:limits[1],limits[0]:limits[1]]
	psf2 = psf2[limits[0]:limits[1],limits[0]:limits[1]]

	### rebin to proper pixel scale
	# following http://scipy-cookbook.readthedocs.io/items/Rebinning.html
	from congrid import congrid
	xdim = np.round(psf1.shape[0]/(data1_res/res1))
	ydim = np.round(psf1.shape[1]/(data1_res/res1))
	psf1 = congrid(psf1,[xdim,ydim])

	xdim = np.round(psf2.shape[0]/(data1_res/res2))
	ydim = np.round(psf2.shape[1]/(data1_res/res2))
	psf2 = congrid(psf2,[xdim,ydim])

	### normalize
	psf1 /= psf1.sum()
	psf2 /= psf2.sum()

	#window = CosineBellWindow(alpha=1.5)
	window = TopHatWindow(beta=0.7) #0.42
	kernel = create_matching_kernel(psf1, psf2,window=window)
	
	if test == True:
		fig, ax = plt.subplots(2,3, figsize=(15, 10))
		ax = np.ravel(ax)

		### plot PSFs
		img = ax[5].imshow(psf1/psf1.max(), cmap='Greys_r', origin='lower')
		plt.colorbar(img,ax=ax[5])
		ax[5].set_title(f1+' PSF')

		convolved_psf1 = convolve_fft(psf1, kernel,interpolate_nan=False)
		img = sedax.imshow(convolved_psf1/convolved_psf1.max(), cmap='Greys_r', origin='lower')
		plt.colorbar(img,ax=sedax)
		sedax.set_title(f1+' PSF convolved')

		img = ax[2].imshow(psf2/psf2.max(), cmap='Greys_r', origin='lower')
		plt.colorbar(img,ax=ax[2])
		ax[2].set_title(f2+' PSF')

		### plot kernel
		img = ax[3].imshow(kernel, cmap='Greys_r', origin='lower')
		plt.colorbar(img,ax=ax[3])
		ax[3].set_title('Convolution Kernel')

		### plot unconvolved residual
		img = ax[4].imshow((psf1-psf2)/psf2, cmap='Greys_r', origin='lower',vmin=-0.05,vmax=0.05)
		cbar = plt.colorbar(img,ax=ax[4])
		cbar.ax.set_title('percent deviation')
		ax[4].set_title('[f1-f2]/f2')

		### plot residual
		img = ax[5].imshow((convolved_psf1-psf2)/psf2, cmap='Greys_r', origin='lower',vmin=-0.05,vmax=0.05)
		cbar = plt.colorbar(img,ax=ax[5])
		cbar.ax.set_title('percent deviation')
		ax[5].set_title('[f1(convolved)-f2]/f2')

		plt.show()

	return kernel, res1

def match_resolution(image,f1,f2,kernel=None,data1_res=None):

	import time

	### grab kernel
	if kernel == None:
		kernel, resolution = psf_match(f1,f2,data1_res=data1_res)

	t1 = time.time()
	convolved_image = convolve_fft(image, kernel,interpolate_nan=False)
	d1 = time.time() - t1
	print('convolution took {0}s'.format(d1))

	return convolved_image, kernel

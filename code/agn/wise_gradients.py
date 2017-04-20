from matplotlib import pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
import os
import numpy as np
from astropy.coordinates import SkyCoord  # High-level coordinates
import astropy.units as u
from reproject import reproject_exact
import copy
from astropy.convolution import convolve, convolve_fft
from wise_colors import vega_conversions
from magphys_plot_pref import jLogFormatter
import brown_io
from astropy.cosmology import WMAP9
from corner import quantile
from photutils import SkyCircularAnnulus, CircularAnnulus, find_peaks

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
		z = dat['residuals']['phot']['z']
		dfactor = 4*np.pi*(WMAP9.luminosity_distance(z).cgs.value)**2
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

def plot_all(runname='brownseds_agn',runname_noagn='brownseds_np',alldata=None,
	         alldata_noagn=None,agn_idx=None,outfolder=None):

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

	#### plot data
	plot_composites(pdata,agn_idx,outfolder,['WISE W1','WISE W2'])

def plot_composites(pdata,idx_plot,outfolder,contours,contour_colors=True,calibration_plot=True):

	### image qualities
	fs = 10 # fontsize
	maxlim = 0.01 # limit of maximum

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
		fig, ax = plt.subplots(1,2, figsize=(12,6))
		ax = np.ravel(ax)

		### load image and WCS
		hdu = load_image(objname,contours[0])
		wcs = WCS(hdu.header)

		### translate object location into pixels using WCS coordinates
		pix_center = wcs.all_world2pix([[ra[0],dec[0]]],1)
		size = calc_dist(wcs, pix_center, phot_size, hdu.data.shape)
		data_to_plot = hdu.data

		### build image extents
		extent = image_extent(size,pix_center,wcs)

		#### load up HDU for WISE 1 + convert to physical units
		# also convolve to W2 resolution
		hdu.data *= 1.9350E-06 ### convert from DN to flux in Janskies, from this table: http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec2_3f.html
		#hdu.data -= np.median(hdu.data) ### subtract background as median
		data_convolved, kernel = match_resolution(hdu.data,contours[0],contours[1],kernel=kernel,data1_res=hdu.header['PXSCAL1']) # convolve to W2 resolution

		### load up HDU for WISE 2 + convert to physical units
		hdu2 = load_image(objname,contours[1])
		#hdu2.data -= np.median(hdu2.data) ### subtract background as median
		hdu2.data *= 2.7048E-06 ### convert from DN to flux in Janskies, from this table: http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec2_3f.html

		#### put onto same scale, and grab slices
		data2, footprint = reproject_exact(hdu2, hdu.header)
		data1_slice = data_convolved[size[2]:size[3],size[0]:size[1]]
		data2_slice = data2[size[2]:size[3],size[0]:size[1]]

		#### calculate the color
		flux_color = convert_to_color(data1_slice, data2_slice,contours[0],contours[1],minflux=1e-10)

		### don't trust anything less than X times the max!
		max1 = np.nanmax(data1_slice)
		max2 = np.nanmax(data2_slice)
		background = (data1_slice < max1*maxlim) | (data2_slice < max2*maxlim)
		flux_color[background] = np.nan

		### plot colormap
		img = ax[0].imshow(flux_color, origin='lower',extent=extent,vmin=color_limits[0],vmax=color_limits[1])
		cbar = fig.colorbar(img, ax=ax[0])
		cbar.formatter.set_powerlimits((0, 0))
		cbar.update_ticks()
		ax[0].set_title(contours[0]+'-'+contours[1]+' color')

		### plot W2 contours
		plot_contour(ax[0],np.log10(data2_slice),ncontours=20)

		### add in WISE PSF
		wise_psf = 6 # in arcseconds
		start = 0.85*extent[0]

		ax[0].plot([start,start+wise_psf],[start,start],lw=2,color='k')
		ax[0].text(start+wise_psf/2.,start+1, '6"', ha='center')
		ax[0].set_xlim(extent[0],extent[1]) # reset plot limits b/c of text stuff
		ax[0].set_ylim(extent[2],extent[3])

		### gradient
		measure_gradient(data1_slice,data2_slice, ax, pix_center)

		plt.savefig(outfolder+'/'+objname+'.png',dpi=150)
		plt.close()
		print 1/0

def measure_gradient(flux1, flux2, ax, center):
	# http://photutils.readthedocs.io/en/stable/api/photutils.aperture.SkyCircularAnnulus.html#photutils.aperture.SkyCircularAnnulus
	# http://photutils.readthedocs.io/en/stable/photutils/aperture.html

	### need errors
	### need to confirm pixscale 
	### need to recenter image
	pixscale = 0.5 # 0.5" / pixel (approximately) (RIGHT?)

	### subtract background from both images
	mean1, median1, std1 = sigma_clipped_stats(flux1, sigma=3.0)
	flux1 -= median1
	mean2, median2, std2 = sigma_clipped_stats(flux2, sigma=3.0)
	flux2 -= median2

	### find image center in W2 image, and mark it
	# do this by finding the source closest to center
	threshold = median2 + (20 * std2) # peak threshold, @ 20 sigma
	tbl = find_peaks(flux2, threshold, box_size=5)
	center = np.array(flux1.shape)/2.
	idx = ((center[0]-tbl['x_peak'])**2 + (center[1]-tbl['y_peak'])**2).argmin()
	ax[0].scatter(tbl['x_peak'][idx],tbl['y_peak'][idx],color='red',marker='x',s=50)

	### define circular (hm) annuli for calculation of gradient
	# do this in pixel space because the sky coordinate transformations are insane
	# and change the distance calculation by << 1%
	r_in = np.arange(1,201)
	r_out = r_in+1
	apertures = [CircularAnnulus([tbl['x_peak'][idx],tbl['y_peak'][idx]],r_in=ri,r_out=ro) for ri,ro in zip(r_in,r_out)]

	### photometer
	from photutils import aperture_photometry
	phot1 = aperture_photometry(flux1,apertures)
	phot2 = aperture_photometry(flux2,apertures)
	f1 = np.array([phot1['aperture_sum_'+str(i)][0] for i in xrange(r_in.shape[0])])
	f2 = np.array([phot2['aperture_sum_'+str(i)][0] for i in xrange(r_in.shape[0])])

	### turn into gradient
	color = convert_to_color(f1,f2,'WISE W1', 'WISE W2',minflux=1e-20)
	r_avg = (r_in+r_out)/2.
	gradient = (color[1:]-color[:-1]) / ((r_avg[1:]-r_avg[:-1])*0.5)

	ax[1].plot(r_avg[:20],gradient[:20])
	ax[1].set_xlabel('arcseconds from center')
	ax[1].set_ylabel(r'$\nabla$(W1-W2) [magnitude/arcsecond]')
	plt.show()
	print 1/0

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

def image_extent(size,center_pix,wcs):

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
	
	return [-xdist,xdist,-ydist,ydist]


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

def plot_contour(ax,cont,ncontours=10):

	#### need x,y arrays
	xlim = ax.get_xlim()
	ylim = ax.get_ylim()
	x = np.linspace(xlim[0],xlim[1],cont.shape[1])
	y = np.linspace(ylim[0],ylim[1],cont.shape[0])

	#### set levels
	cs = ax.contour(x,y, cont, linewidths=0.7,zorder=2, ncontours=ncontours, cmap='Greys')

def convert_to_color(flux1,flux2,filter1,filter2,minflux=1e-10):

	'''
	convert fluxes in fnu into Vega-scale colors
	'''

	flux1_clip = np.clip(flux1,minflux,np.inf)
	flux2_clip = np.clip(flux2,minflux,np.inf)
	vega_conv = (vega_conversions(filter1) - vega_conversions(filter2))

	flux_color = -2.5*np.log10(flux1_clip/flux2_clip)+vega_conv

	return flux_color

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
		img = ax[0].imshow(psf1/psf1.max(), cmap='Greys_r', origin='lower')
		plt.colorbar(img,ax=ax[0])
		ax[0].set_title(f1+' PSF')

		convolved_psf1 = convolve_fft(psf1, kernel,interpolate_nan=False)
		img = sedax[0].imshow(convolved_psf1/convolved_psf1.max(), cmap='Greys_r', origin='lower')
		plt.colorbar(img,ax=sedax)
		sedax[0].set_title(f1+' PSF convolved')

		img = ax[2].imshow(psf2/psf2.max(), cmap='Greys_r', origin='lower')
		plt.colorbar(img,ax=ax[2])
		ax[2].set_title(f2+' PSF')

		### plot kernel
		img = ax[3].imshow(kernel, cmap='Greys_r', origin='lower')
		plt.colorbar(img,ax=ax[3])
		ax[3].set_title('Convolution Kernel')

		### plot unconvolved residual
		img = ax[0].imshow((psf1-psf2)/psf2, cmap='Greys_r', origin='lower',vmin=-0.05,vmax=0.05)
		cbar = plt.colorbar(img,ax=ax)
		cbar.ax[0].set_title('percent deviation')
		ax[0].set_title('[f1-f2]/f2')

		### plot residual
		img = ax[0].imshow((convolved_psf1-psf2)/psf2, cmap='Greys_r', origin='lower',vmin=-0.05,vmax=0.05)
		cbar = plt.colorbar(img,ax=ax[0])
		cbar.ax[0].set_title('percent deviation')
		ax[0].set_title('[f1(convolved)-f2]/f2')

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

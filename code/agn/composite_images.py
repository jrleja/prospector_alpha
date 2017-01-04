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
from reproject import reproject_interp
import copy
from astropy.convolution import convolve, convolve_fft
from wise_colors import vega_conversions
plt.ioff()

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

	#### numpy arrays
	for key in model_pars.keys(): 
		for key2 in model_pars[key].keys():
			model_pars[key][key2] = np.array(model_pars[key][key2])

	out = {'pars':model_pars,'objname':objname}
	return out

def collate_spectra(alldata,idx_plot,pdata,runname):

	'''
	check for AGN on and AGN off in alldata
	if it doesn't exist, make it
	if it does exist, use it.
	'''

	filters = ['wise_w1', 'wise_w2', 'wise_w3', 'wise_w4']
	agn_on_mag, agn_off_mag, agn_on_spec, agn_off_spec = [],[],[],[]
	wave = None
	sps = None
	for ii,idx in enumerate(idx_plot):
		dat = alldata[idx]

		### first, if AGN_off spectrum isn't there...
		if 'no_agn' not in dat.keys():
			
			from prospect.models import model_setup
			from threed_dutils import generate_basenames
			
			#### LOAD OBJECT PARAMETER FILE
			filebase,parm,ancilname = generate_basenames(runname)
			idx = np.array([True if dat['objname'] in name else False for name in filebase])
			pfile = np.array(parm)[idx][0]
			parmfile = model_setup.import_module_from_file(pfile)

			##### MAKE THE THINGS
			sps = model_setup.load_sps(param_file=pfile,**parmfile.run_params)
			model = model_setup.load_model(param_file=pfile,**parmfile.run_params)
			obs = model_setup.load_obs(param_file=pfile,**parmfile.run_params)

			#### USE BESTFIT, TURN AGN STUFF OFF
			theta = dat['bfit']['maxprob_params']
			theta[dat['pquantiles']['parnames'] == 'fagn'] = 0.0
			spec_noagn,mags_noagn,sm = model.mean_model(theta, obs, sps=sps)

			dat['no_agn'] = {}
			dat['no_agn']['spec'] = spec_noagn
			dat['no_agn']['mags'] = mags_noagn
			dat['no_agn']['wave'] = sps.wavelengths
			dat['no_agn']['phot_mask'] = obs['phot_mask']

		### maggies
		mags, mags_noagn = [], []
		for f in filters:
			match = dat['filters'] == f

			if match.sum() == 0:
				noagn = np.nan
				agn = np.nan
			else:
				noagn = dat['no_agn']['mags'][dat['no_agn']['phot_mask']][match][0]
				agn = dat['bfit']['mags'][dat['no_agn']['phot_mask']][match][0]

			mags.append(agn)
			mags_noagn.append(noagn)
		agn_on_mag.append(mags)
		agn_off_mag.append(mags_noagn)

		### spectra
		agn_on_spec.append(dat['bfit']['spec'])
		agn_off_spec.append(dat['no_agn']['spec'])

	### if we generated spectra, save alldata
	if sps is not None:
		import brown_io
		brown_io.save_alldata(alldata,runname=runname)

	out = {}
	out['agn_on_mag'] = agn_on_mag
	out['agn_off_mag'] = agn_off_mag
	out['agn_on_spec'] = agn_on_spec
	out['agn_off_spec'] = agn_off_spec
	out['wave'] = dat['no_agn']['wave']

	pdata['observables'] = out
	return pdata

def plot_all(runname='brownseds_agn',alldata=None,outfolder=None):

	#### load alldata
	if alldata is None:
		alldata = brown_io.load_alldata(runname=runname)

	#### make output folder if necessary
	if outfolder is None:
		outfolder = os.getenv('APPS')+'/threedhst_bsfh/plots/'+runname+'/agn_plots/sdss_overlays'
		if not os.path.isdir(outfolder):
			os.makedirs(outfolder)

	#### collate data
	pdata = collate_data(alldata)

	#### select data to plot. generate SEDs with no AGN contribution if necessary.
	# 10 most massive AGN
	#idx_plot = pdata['pars']['fagn']['q50'].argsort()[-10:][::-1]
	idx_plot = pdata['pars']['fagn']['q50'].argsort()[:10][::-1]
	pdata = collate_spectra(alldata,idx_plot,pdata,runname)

	#### plot data
	plot_composites(pdata,idx_plot,outfolder)

def plot_composites(pdata,idx_plot,outfolder,contour_colors=True,calibration_plot=True):

	### open figure
	#fig, ax = plt.subplots(5,2, figsize=(7, 15))
	#fig, ax = plt.subplots(1,1, figsize=(7, 7),subplot_kw={'projection': ccrs.PlateCarree()})
	#ax = np.ravel(ax)

	### image qualities
	fs = 10

	### filters
	filters = ['SDSS u','SDSS g','SDSS i']
	fcolors = ['Blues','Greens','Reds']
	ftext = ['blue','green','red']

	### contours
	contours = ['WISE W1','WISE W2']
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
		xs, ys, dely = 0.05,0.95, 0.05

		for kk,filt in enumerate(filters):
			hdu = load_image(objname,filt)

			#### if it's the first filter,
			#### set up WCS using this information
			if fig == None:

				### grab WCS information, create figure + axis
				wcs = WCS(hdu.header)
				fig = plt.figure(figsize=(14, 7))
				ax1 = fig.add_axes([0.1,0.1,0.35,0.8])#,projection=wcs)
				ax2 = fig.add_axes([0.55,0.1,0.35,0.8])
				ax = [ax1,ax2]
				#fig, ax = plt.subplots(1,1, figsize=(14, 7),subplot_kw={'projection': wcs})

				### translate object location into pixels using WCS coordinates
				pix_center = wcs.all_world2pix([[ra[0],dec[0]]],1)
				size = calc_dist(wcs, pix_center, phot_size, hdu.data.shape)

				hdu_original = copy.deepcopy(hdu.header)
				data_to_plot = hdu.data

			#### if it's not the first filter,
			#### project into WCS of first filter
			# see reprojection https://reproject.readthedocs.io/en/stable/
			else:
				data_to_plot, footprint = reproject_interp(hdu, hdu_original)

			plot_image(ax[0],data_to_plot,size,cmap=fcolors[kk])
			ax[0].text(xs, ys, filters[kk],color=ftext[kk],transform=ax[0].transAxes)
			ys -= dely

			### labels and limits
			ax[0].set_xlim(size[0],size[1])
			ax[0].set_ylim(size[2],size[3])

			ax[0].set_xlabel('RA')
			ax[0].set_ylabel('Dec')

		if contour_colors:


			### Don't show WISE colors for pixels below BACKGROUND_FILTER*MEDIAN(CUTOUT)
			background_filter = 20.0

			#### load up HDU, subtract background and convert to physical units
			# also convolve to W2 resolution
			hdu = load_image(objname,contours[0])
			hdu.data *= 1.9350E-06 ### convert from DN to flux in Janskies, from this table: http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec2_3f.html
			hdu.data -= np.median(hdu.data) ### subtract background as median
			data1_noconv, footprint = reproject_interp(hdu, hdu_original)
			data_convolved, kernel = match_resolution(hdu.data,contours[0],contours[1],kernel=kernel,data1_res=hdu.header['PXSCAL1']) # convolve to W2 resolution
			hdu.data = data_convolved
			data1, footprint = reproject_interp(hdu, hdu_original)

			### load up HDU2, subtract background, convert to physical units
			hdu = load_image(objname,contours[1])
			hdu.data -= np.median(hdu.data) ### subtract background as median
			hdu.data *= 2.7048E-06 ### convert from DN to flux in Janskies, from this table: http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec2_3f.html

			#### put onto same scale
			data2, footprint = reproject_interp(hdu, hdu_original)

			#### PLOTTING
			if calibration_plot:
				figcal, axcal = plt.subplots(2,3, figsize=(15, 10))
				axcal = np.ravel(axcal)

				data1_slice = data1[size[2]:size[3],size[0]:size[1]]
				data2_slice = data2[size[2]:size[3],size[0]:size[1]]

				flux_color = convert_to_color(data1_slice, data2_slice,contours[0],contours[1],minflux=1e-10)

				img = axcal[0].imshow(data1_noconv[size[2]:size[3],size[0]:size[1]], origin='lower')
				plt.colorbar(img,ax=axcal[0])
				axcal[0].set_title(contours[0]+' flux, unconvolved')

				img = axcal[1].imshow(data1_slice, origin='lower')
				plt.colorbar(img,ax=axcal[1])
				axcal[1].set_title(contours[0]+' flux')

				img = axcal[2].imshow(data2_slice, origin='lower')
				plt.colorbar(img,ax=axcal[2])
				axcal[2].set_title(contours[1]+' flux')

				img = axcal[3].imshow(flux_color, origin='lower')
				plt.colorbar(img,ax=axcal[3])
				axcal[3].set_title(contours[0]+'-'+contours[1])

				'''
				background1 = np.nanmedian(data1_slice)
				background2 = np.nanmedian(data2_slice)
				background = (data1_slice < background1*background_filter) & (data2_slice < background2*background_filter)
				flux_color[background] = np.nan
				'''
				### don't trust anything less than 0.001 the max!
				maxlim = 0.08
				max1 = np.nanmax(data1_slice)
				max2 = np.nanmax(data2_slice)
				background = (data1_slice < max1*maxlim) | (data2_slice < max2*maxlim)
				flux_color[background] = np.nan


				img = axcal[4].imshow(flux_color, origin='lower')
				plt.colorbar(img,ax=axcal[4])
				axcal[4].set_title(contours[0]+'-'+contours[1]+', background filter')

				figcal.savefig(outfolder+'/'+objname+'_calibration.png',dpi=150)
				plt.close(figcal)

			plot_color_contour(ax[0],data1,data2,contours[0],contours[1],size,background_filter=background_filter)

			ax[0].text(xs, ys, contours[0]+'-'+contours[1],transform=ax[0].transAxes)
			ys -= dely

		else:
			for kk, cont in enumerate(contours):
				hdu = load_image(objname,cont)
				data_to_plot, footprint = reproject_interp(hdu, hdu_original)

				plot_contour(ax,data_to_plot,size,color=ccolors[kk])
				ax[0].text(xs, ys, contours[kk],color=ctext[kk],transform=ax[0].transAxes)
				ys -= dely

		### labels and limits
		ax[0].set_xlim(size[0],size[1])
		ax[0].set_ylim(size[2],size[3])

		ax[0].set_xlabel('RA')
		ax[0].set_ylabel('Dec')

		ax[0].text(0.98,0.95,objname,transform=ax[0].transAxes,ha='right')
		ax[0].text(0.98,0.9,r'f$_{\mathrm{AGN}}$='+"{:.2f}".format(fagn),transform=ax[0].transAxes, ha='right')

		#### now plot the SED
		plt.savefig(outfolder+'/'+objname+'.png',dpi=150)
		plt.close()

def calc_dist(wcs, pix_center, size, im_shape):

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

	folder = os.getenv('APPS')+'/threedhst_bsfh/data/brownseds_data/fits/'

	filter = filter.replace(' ','_')
	objname = objname.replace(' ','_')
	fits_file = folder + objname+'_'+filter+'.fits'
	hdu = fits.open(fits_file)[0]

	return hdu

def load_structure(objname,long_axis=False):

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

	from brown_io import load_coordinates

	ra,dec,objnames = load_coordinates()
	match = objname == objnames
	
	return ra[match], dec[match]

def plot_contour(ax,data,size,ncontours=20,color='white'):

	'''
	FIRST SHIFT MINIMUM FLUX TO ZERO (?)
	'''

	box_min = np.nanmin(data[size[0]:size[1],size[2]:size[3]])
	box_max = np.nanmax(data[size[0]:size[1],size[2]:size[3]])
	data = np.clip(data,box_min,np.inf)
	data[np.isnan(data)] = box_max

	bmax = box_min*100
	levels = np.linspace(box_min, bmax, ncontours)
	levels = 10**np.linspace(np.log10(box_min), np.log10(bmax), ncontours)

	ax.contour(data, levels=levels, colors=color)

def plot_color_contour(ax,flux1,flux2,filter1,filter2,size,ncontours=25,color='white',vega_conv=0.0,background_filter=1.5):

	#### calculate color
	flux_color = convert_to_color(flux1,flux2,filter1,filter2,minflux=1e-10)

	#### set background equal to NaN
	# only in slice
	# define background as median in slice * background_filter
	flux1_slice = flux1[size[2]:size[3],size[0]:size[1]]
	flux2_slice = flux2[size[2]:size[3],size[0]:size[1]]

	'''
	background1 = np.nanmedian(flux1_slice)
	background2 = np.nanmedian(flux2_slice)
	background = (flux1 < background1*background_filter) & (flux2 < background2*background_filter)
	flux_color[background] = np.nan
	'''
	maxlim = 0.08
	max1 = np.nanmax(flux1_slice)
	max2 = np.nanmax(flux2_slice)
	background = (flux1 < max1*maxlim) | (flux2 < max2*maxlim)
	flux_color[background] = np.nan

	#### set levels
	#vmin = np.nanmin(flux_color[size[2]:size[3],size[0]:size[1]])
	#vmin = np.nanmax(flux_color[size[2]:size[3],size[0]:size[1]])
	levels = np.arange(-1.0,2.6,step=0.1)
	cs = ax.contour(flux_color,levels=levels, linewidths=2)
	ax.clabel(cs, inline=1, fontsize=12)

def convert_to_color(flux1,flux2,filter1,filter2,minflux=1e-10):

	'''
	convert fluxes in fnu into Vega-scale colors
	'''

	flux1_clip = np.clip(flux1,minflux,np.inf)
	flux2_clip = np.clip(flux2,minflux,np.inf)
	vega_conv = (vega_conversions(filter1) - vega_conversions(filter2))

	flux_color = -2.5*np.log10(flux1/flux2)+vega_conv

	return flux_color

def plot_image(ax,data,size,alpha=0.5,cmap='cubehelix'):

	sigclip = sigma_clip(data,3)
	sigclip = sigclip.filled(np.nanmax(sigclip)) # fill clipped values with max

	### shift all numbers so that minimum = 0
	try:
		minimum = sigclip[size[0]:size[1],size[2]:size[3]].min()
	except ValueError:
		print 1/0
	if sigclip.min() < 0:
		sigclip += np.abs(minimum)
	else:
		sigclip -= np.abs(minimum)

	### set the maximum to be maximum in the data box
	maximum = np.nanmax(sigclip[size[0]:size[1],size[2]:size[3]])
	sigclip = np.clip(sigclip,-np.inf,maximum)

	### put into units of standard deviations
	### start linear scale at 3sigma
	std = np.std(sigclip[size[0]:size[1],size[2]:size[3]])
	sigclip /= std
	sigclip = np.clip(sigclip,3.0,np.inf)
	
	minimum = np.nanmin(sigclip)
	sigclip = sigclip - minimum

	### show image
	ax.imshow(sigclip, origin='lower', cmap=cmap,alpha=alpha)

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
		img = ax[1].imshow(convolved_psf1/convolved_psf1.max(), cmap='Greys_r', origin='lower')
		plt.colorbar(img,ax=ax[1])
		ax[1].set_title(f1+' PSF convolved')

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

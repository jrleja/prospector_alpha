from matplotlib import pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from astropy.modeling import models, fitting
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
from corner import quantile
from photutils import CircularAperture, CircularAnnulus, find_peaks, aperture_photometry
from astropy.cosmology import WMAP9
import pickle
from prosp_dutils import asym_errors
from scipy.optimize import brentq

plt.ioff()

c = 3e18   # angstroms per second
minorFormatter = jLogFormatter(base=10, labelOnlyBase=False)
majorFormatter = jLogFormatter(base=10, labelOnlyBase=True)

# https://arxiv.org/pdf/1603.05664.pdf
px_scale = 2.75
outfile = '/Users/joel/code/python/threedhst_bsfh/data/brownseds_data/fits/unWISE/gradients.pickle'
wise_psf = 6

def test_z(x,c):
	# c = true lumdist
	# x = redshift
	return WMAP9.luminosity_distance(x).value-c

def collate_data(alldata,alldata_noagn):

	#### generate containers
	# photometry
	obs_phot, model_phot = {}, {}
	filters = ['wise_w1', 'wise_w2', 'wise_w3', 'wise_w4',
	           'spitzer_irac_ch1','spitzer_irac_ch2','spitzer_irac_ch3','spitzer_irac_ch4']

	for f in filters: 
		obs_phot[f] = []
		model_phot[f] = []

	# model parameters
	z,objname = [], []
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
	for i, dat in enumerate(alldata):
		idx = xray['objname'] == dat['objname']
		if idx.sum() != 1:
			print 1/0
		xflux = xray['flux'][idx][0]
		xflux_err = xray['flux_err'][idx][0]

		#### convert lumdist to redshift for distance calculations
		# only in alldata_noagn for now...
		lumdist = alldata_noagn[i]['residuals']['phot']['lumdist']
		zred = brentq(test_z, 0, 0.2, args=(lumdist), rtol=1.48e-08, maxiter=1000)
		z.append(zred)

		# flux is in ergs / cm^2 / s, convert to erg /s 
		pc2cm =  3.08568E18
		dfactor = 4*np.pi*(lumdist*1e6*pc2cm)**2
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

	out['z'] = np.array(z)
	out['lsfr'] = np.array(lsfr)
	out['lsfr_up'] = np.array(lsfr_up)
	out['lsfr_down'] = np.array(lsfr_down)
	out['xray_luminosity'] = np.array(xray_lum)
	out['xray_luminosity_err'] = np.array(xray_lum_err)
	out['bpt'] = brown_io.return_agn_str(np.ones_like(alldata,dtype=bool),string=True)

	return out

def plot_all(runname='brownseds_agn',runname_noagn='brownseds_np',alldata=None,
	         alldata_noagn=None,agn_idx=None,outfolder=None,regenerate=False, **popts):

	#### load alldata
	if alldata is None:
		alldata = brown_io.load_alldata(runname=runname)
		alldata_noagn = brown_io.load_alldata(runname=runname_noagn)

	#### make output folder if necessary
	outfolder_overlays = os.getenv('APPS')+'/threedhst_bsfh/plots/'+runname+'/agn_plots/sdss_overlays'
	if not os.path.isdir(outfolder_overlays):
		os.makedirs(outfolder_overlays)

	#### collate data
	pdata = collate_data(alldata,alldata_noagn)

	#### plot data
	if regenerate:
		plot_composites(pdata,outfolder_overlays,['WISE W1','WISE W2'])
	plot_summary(pdata,outfolder,agn_idx=agn_idx,**popts)

def plot_summary(pdata, outfolder, agn_idx=Ellipsis, **popts):

	'''
	(2) evaluate: 1 kpc gradient ok?
	(3) manually check for point sources, remove [NGC 1068, NGC 3690, UGC 05101]
	'''

	#### load data
	with open(outfile, "rb") as f:
		outdict=pickle.load(f)

	#### plot options
	blue = '#1C86EE' 
	opts = {
	        'color': blue,
	        'mew': 1.5,
		 	'capthick':0.6,
		 	'elinewidth':0.6,
		 	'ecolor':'0.2',
		 	'ms': 12
        } 

	fig, ax = plt.subplots(1,1, figsize=(7.5, 7))
	idx = 1 # 1 kpc. change to 1 ---> 2 kpc.
	arcsec_lim = wise_psf # one PSF FWHM

	#### pick out good measurements, and figure out AGN indexes
	# could be more aggressive
	resolved = (outdict['arcsec'][:,idx] > 3) & \
	           (outdict['gradient_error'][:,idx] < 0.25) & \
	           (outdict['obj_size_brown_kpc'] > (idx+1)*2)
	cidx = np.ones_like(resolved,dtype=bool)
	cidx[agn_idx] = False
	cidx = cidx[resolved]

	#### plot
	xp = np.log10(pdata['pars']['fagn']['q50'][resolved])
	xerr = asym_errors(pdata['pars']['fagn']['q50'][resolved],
		               pdata['pars']['fagn']['q84'][resolved],
		               pdata['pars']['fagn']['q16'][resolved],log=True) 
	yp = outdict['gradient'][resolved,idx].squeeze()
	yerr = outdict['gradient_error'][resolved,idx].squeeze()

	ax.errorbar(xp[cidx],yp[cidx], 
		        xerr=[xerr[0][cidx],xerr[1][cidx]],
		        yerr=yerr[cidx],
		        zorder=-3, fmt=popts['nofmir_shape'],
		        alpha=popts['nofmir_alpha'],**opts)
	ax.errorbar(xp[~cidx],yp[~cidx], 
		        xerr=[xerr[0][~cidx],xerr[1][~cidx]],
		        yerr=yerr[~cidx],
		        zorder=-3, fmt=popts['fmir_shape'],
		        alpha=popts['fmir_alpha'],**opts)

	ax.set_xlabel(r'log(f$_{\mathrm{MIR}}$)')
	ax.set_ylabel(r'$\nabla$(W1-W2) at r=2 kpc [mag/kpc]')

	ax.axhline(0, linestyle='--', color='0.2',lw=2,zorder=-1,alpha=0.4)
	#ax.set_ylim(-.2,.2)
	ax.set_xlim(-4,1)

	plt.tight_layout()
	plt.savefig(outfolder+'wise_gradient.png',dpi=150)

def plot_composites(pdata,outfolder,contours,contour_colors=True,
	                calibration_plot=True,brown_data=False):

	### image qualities
	fs = 10 # fontsize
	maxlim = 0.01 # limit of maximum

	### contour color limits (customized for W1-W2)
	color_limits = [-1.0,2.6]
	kernel = None

	### output blobs
	gradient, gradient_error, arcsec, obj_size,objname_out,obj_size_gauss, obj_size_kpc = [], [], [], [], [], [], []

	### begin loop
	for idx in xrange(len(pdata['objname'])):

		### load object information
		objname = pdata['objname'][idx]
		fagn = pdata['pars']['fagn']['q50'][idx]
		ra, dec = load_coordinates(objname)
		phot_size = load_structure(objname,long_axis=True) # in arcseconds

		### load image and WCS
		try:
			if brown_data:
				### convert from DN to flux in Janskies, from this table: 
				# http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec2_3f.html
				img1, noise1 = load_image(objname,contours[0]), None
				img1, noise1 = img1*1.9350E-06, noise1*(1.9350e-06)**-2
				img2, noise2 = load_image(objname,contours[1]), None
				img2, noise2 = img2*2.7048E-06, noise2*(2.7048E-06)**-2

				### translate object location into pixels using WCS coordinates
				wcs = WCS(img1.header)
				pix_center = wcs.all_world2pix([[ra[0],dec[0]]],1)
			else:
				img1, noise1 = load_wise_data(objname,contours[0].split(' ')[1])
				img2, noise2 = load_wise_data(objname,contours[1].split(' ')[1])

				### translate object location into pixels using WCS coordinates
				wcs = WCS(img1.header)
				pix_center = wcs.all_world2pix([[ra[0],dec[0]]],1)

				if (pix_center.squeeze()[0]-4 > img1.shape[1]) or \
					(pix_center.squeeze()[1]-4 > img1.shape[0]) or \
					(np.any(pix_center < 4)):
					print 'object not in image, checking for additional image'
					print pix_center, img1.shape
					img1, noise1 = load_wise_data(objname,contours[0].split(' ')[1],load_other = True)
					img2, noise2 = load_wise_data(objname,contours[1].split(' ')[1],load_other = True)

					wcs = WCS(img1.header)
					pix_center = wcs.all_world2pix([[ra[0],dec[0]]],1)
					print pix_center, img1.shape
		except:
			print 1/0
			gradient.append(None)
			gradient_error.append(None)
			arcsec.append(None)
			kpc.append(None)
			objname_out.append(None)
			continue

		size = calc_dist(wcs, pix_center, phot_size, img1.data.shape)
		print pix_center, img1.data.shape

		### convert inverse variance to noise
		noise1.data = (1./noise1.data)**0.5
		noise2.data = (1./noise2.data)**0.5

		### build image extents
		extent = image_extent(size,pix_center,wcs)

		### convolve W1 to W2 resolution
		w1_convolved, kernel = match_resolution(img1.data,contours[0],contours[1],
													kernel=kernel,data1_res=px_scale)
		w1_convolved_noise, kernel = match_resolution(noise1.data,contours[0],contours[1],
													  kernel=kernel,data1_res=px_scale)

		#### put onto same scale, and grab slices
		data2, footprint = reproject_exact(img2, img1.header)
		noise2, footprint = reproject_exact(noise2, img1.header)

		img1_slice = w1_convolved[size[2]:size[3],size[0]:size[1]]
		img2_slice = data2[size[2]:size[3],size[0]:size[1]]
		noise1_slice = w1_convolved_noise[size[2]:size[3],size[0]:size[1]]
		noise2_slice = noise2[size[2]:size[3],size[0]:size[1]]

		### subtract background from both images
		mean1, median1, std1 = sigma_clipped_stats(w1_convolved, sigma=3.0,iters=10)
		img1_slice -= median1
		mean2, median2, std2 = sigma_clipped_stats(data2, sigma=3.0, iters=10)
		img2_slice -= median2

		#### calculate the color
		flux_color  = convert_to_color(img1_slice, img2_slice,None,None,contours[0],contours[1],
			                           minflux=-np.inf, vega_conversions=brown_data)

		### don't show anything with S/N < 2!
		sigmask = 2
		background = (img2_slice/noise2_slice < sigmask) | (img1_slice/noise1_slice < sigmask)
		flux_color[background] = np.nan

		### plot colormap
		fig, ax = plt.subplots(1,2, figsize=(12,6))
		ax = np.ravel(ax)
		img = ax[0].imshow(flux_color, origin='lower',extent=extent,vmin=color_limits[0],vmax=color_limits[1])

		cbar = fig.colorbar(img, ax=ax[0])
		cbar.formatter.set_powerlimits((0, 0))
		cbar.update_ticks()
		ax[0].set_title(contours[0]+'-'+contours[1]+' color')

		### plot W2 contours
		plot_contour(ax[0],np.log10(img2_slice),ncontours=20)

		### find image center in W2 image, and mark it
		# do this by finding the source closest to center
		tbl = []
		nthresh, box_size = 20, 5
		fake_noise1_error = copy.copy(noise1_slice)
		bad = np.logical_or(np.isinf(noise1_slice),np.isnan(noise1_slice))
		fake_noise1_error[bad] = fake_noise1_error[~bad].max()
		while len(tbl) < 1:
			threshold = nthresh * std1 # peak threshold, @ 20 sigma
			tbl = find_peaks(img1_slice, threshold, box_size=box_size, subpixel=True, border_width=4, error = fake_noise1_error)
			nthresh -=1
		
			if nthresh < 2:
				nthresh = 20
				box_size = 3

		### find size of biggest one
		center = np.array(img2_slice.shape)/2.
		idxmax = ((center[0]-tbl['x_centroid'])**2 + (center[1]-tbl['y_centroid'])**2).argmin()
		ginit = models.Gaussian2D(amplitude=tbl['fit_peak_value'][idxmax],  x_mean=tbl['x_centroid'][idxmax],  y_mean=tbl['y_centroid'][idxmax], x_stddev=1,y_stddev=1)
		fit_g = fitting.LevMarLSQFitter()
		tx, ty = np.mgrid[:img1_slice.shape[0], :img1_slice.shape[1]]
		ginit.x_mean.fixed=True
		ginit.y_mean.fixed=True
		ginit.amplitude.fixed=True
		g = fit_g(ginit, tx, ty, img1_slice)
		size_twosig = np.sqrt(g.x_stddev.value**2+g.y_stddev.value**2)*2*px_scale

		### find center in arcseconds
		xarcsec = (extent[1]-extent[0])*(tbl['x_centroid'][idxmax])/float(img2_slice.shape[0]) + extent[0]
		yarcsec = (extent[3]-extent[2])*(tbl['y_centroid'][idxmax])/float(img2_slice.shape[1]) + extent[2]
		ax[0].scatter(xarcsec,yarcsec,color='black',marker='x',s=50,linewidth=2)

		### add in WISE PSF
		wise_psf = 6 # in arcseconds
		start = 0.85*extent[0]

		ax[0].plot([start,start+wise_psf],[start,start],lw=2,color='k')
		ax[0].text(start+wise_psf/2.,start+1, '6"', ha='center')
		ax[0].set_xlim(extent[0],extent[1]) # reset plot limits b/c of text stuff
		ax[0].set_ylim(extent[2],extent[3])

		### gradient
		phys_scale = float(1./WMAP9.arcsec_per_kpc_proper(pdata['z'][idx]).value)
		grad, graderr, x_arcsec, size_arcsec = measure_gradient(img1_slice,img2_slice, 
								  noise1_slice, noise2_slice, 
								  ax, 
								  (tbl['x_centroid'][idxmax], tbl['y_centroid'][idxmax]),
								  tbl['fit_peak_value'][idxmax], (xarcsec,yarcsec), size_twosig,
								  phys_scale)

		ax[1].text(0.05,0.06,r'f$_{\mathrm{MIR}}$='+"{:.2f}".format(pdata['pars']['fagn']['q50'][idx])+\
								' ('+"{:.2f}".format(pdata['pars']['fagn']['q84'][idx]) +
								') ('+"{:.2f}".format(pdata['pars']['fagn']['q16'][idx])+')',
								transform=ax[1].transAxes,color='black',fontsize=9)

		obj_size_phys = phot_size*phys_scale
		ax[1].axvline(phot_size, linestyle='--', color='0.2',lw=2,zorder=-1)

		gradient.append(grad)
		gradient_error.append(graderr)
		arcsec.append(x_arcsec)
		obj_size.append(size_arcsec)
		obj_size_gauss.append(size_twosig)
		obj_size_kpc.append(obj_size_phys)
		objname_out.append(objname)

		plt.tight_layout()
		plt.savefig(outfolder+'/'+objname+'.png',dpi=150)
		plt.close()

	out = {
			'gradient': np.array(gradient),
			'gradient_error': np.array(gradient_error),
			'arcsec': np.array(arcsec),
			'obj_size': np.array(obj_size),
			'obj_size_gauss': np.array(obj_size_gauss),
			'obj_size_brown_kpc': np.array(obj_size_kpc),
			'objname': objname_out
		  }

	pickle.dump(out,open(outfile, "wb"))

def measure_gradient(flux1, flux2, noise1, noise2, ax, center, peakflux, center_arcsec, gauss_size, phys_scale):
	# http://photutils.readthedocs.io/en/stable/api/photutils.aperture.SkyCircularAnnulus.html#photutils.aperture.SkyCircularAnnulus
	# http://photutils.readthedocs.io/en/stable/photutils/aperture.html

	### define circular (hm) annuli for calculation of gradient
	# do this in pixel space because the sky coordinate transformations are insane
	# and change the distance calculation by << 1%
	r_in = np.arange(1,50)
	r_out = r_in+1
	apertures = [CircularAnnulus(center,r_in=ri,r_out=ro) for ri,ro in zip(r_in,r_out)]
	pdict = {'r_in':r_in,'r_out':r_out,'apertures':apertures}

	# add most important apertures: 0-1 kpc, 1-2 kpc (so gradient at 2kpc)
	r_in_calc = np.array([0.,1.,0.,2.])/phys_scale/px_scale
	r_out_calc = np.array([1.,2.,2.,4.])/phys_scale/px_scale
	apertures_calc = [CircularAperture(center,r_out_calc[0]), 
					  CircularAnnulus(center,r_in_calc[1],r_out=r_out_calc[1]),
					  CircularAperture(center,r_out_calc[2]),
					  CircularAnnulus(center,r_in_calc[3],r_out=r_out_calc[3])]
	calcdict = {'r_in':r_in_calc,'r_out':r_out_calc,'apertures':apertures_calc}

	for dic in [pdict, calcdict]:

		### photometer
		phot1 = aperture_photometry(flux1,dic['apertures'],error=noise1, mask=np.logical_or(np.isinf(noise1),np.isnan(noise1)))
		phot2 = aperture_photometry(flux2,dic['apertures'],error=noise2, mask=np.logical_or(np.isinf(noise2),np.isnan(noise2)))

		nap = len(dic['apertures'])
		f1 = np.array([phot1['aperture_sum_'+str(i)][0] for i in xrange(nap)])
		e1 = np.array([phot1['aperture_sum_err_'+str(i)][0] for i in xrange(nap)])
		f2 = np.array([phot2['aperture_sum_'+str(i)][0] for i in xrange(nap)])
		e2 = np.array([phot2['aperture_sum_err_'+str(i)][0] for i in xrange(nap)])

		### turn into gradient
		color, err = convert_to_color(f1,f2,e1,e2,'WISE W1', 'WISE W2',minflux=1e-20,vega_conversions=False)
		r_avg = (dic['r_in']+dic['r_out'])/2.
		gradient = (color[1:]-color[:-1]) / ((r_avg[1:]-r_avg[:-1])*px_scale)
		gradient_err = np.sqrt(err[1:]**2+err[:-1]**2) / ((r_avg[1:]-r_avg[:-1])*px_scale)

		### plotting
		if nap > 10:

			### what to plot?
			grad_plot = gradient
			graderr_plot = gradient_err
			x_arcsec = dic['r_out'][:-1]*px_scale

			### limit. when you first dip below s/n = 10, don't plot the remainder
			sn_lim = 10
			good = ((f1/e1) > sn_lim) & ((f2/e2) > sn_lim)
			good[np.where(good == False)[0].min():] = False
			### account for gradient
			# require BOTH ENDS of gradient have S/N > x
			good = good[1:]

			### plot
			ax[1].errorbar(x_arcsec[good],grad_plot[good],yerr=graderr_plot[good], 
				           fmt='o',ms=8,elinewidth=1.5,color='k',ecolor='k',linestyle='-')

			### show object size
			# find FIRST TIME it dips below 5% of peak
			avg_pix_flux = f1 / (np.pi*(r_out**2 - r_in**2))
			lim = np.where(avg_pix_flux / peakflux < 0.05)[0].min()+1

			obj_size = np.interp(0.1,avg_pix_flux[:lim]/peakflux,r_avg[:lim]*px_scale)

			circ=plt.Circle(center_arcsec, radius=gauss_size, fill=False,lw=2,color='red',alpha=0.8,zorder=5)
			ax[0].add_patch(circ)
			ax[1].text(0.95,0.05,r'size='+"{:.2f}".format(obj_size)+'"',transform=ax[1].transAxes,color='red',ha='right',fontsize=9)
			ax[1].text(0.95,0.1,r'gauss size='+"{:.2f}".format(gauss_size)+'"',transform=ax[1].transAxes,color='red',ha='right',fontsize=9)

		else:

			### what to plot?
			grad_calc = [gradient[0], gradient[2]]
			graderr_calc = [gradient_err[0], gradient_err[2]]
			arcsec_calc = np.array([dic['r_out'][0],dic['r_out'][2]])*px_scale

			### what to save?
			grad_kpc = np.array([color[1]-color[0],(color[3]-color[2])/2.])
			graderr_kpc = np.array([np.sqrt(err[1]**2+err[0]**2),
				                    np.sqrt(err[2]**2+err[3]**2)/2.])

			### add in important gradients
			ax[1].errorbar(arcsec_calc,grad_calc,yerr=graderr_calc,
				           fmt='o',ms=10,elinewidth=2.0,color='red',ecolor='red',linestyle=' ')

			### add text
			ax[1].text(0.05,0.93,r'$\nabla$(1 kpc)='+"{:.3f}".format(grad_kpc[0])+r'$\pm$'+"{:.3f}".format(graderr_kpc[0]),
								transform=ax[1].transAxes,color='red')
			ax[1].text(0.05,0.86,r'$\nabla$(2 kpc)='+"{:.3f}".format(grad_kpc[1])+r'$\pm$'+"{:.3f}".format(graderr_kpc[1]),
								transform=ax[1].transAxes,color='red')

	### labels
	ax[1].axhline(0, linestyle='--', color='0.2',lw=2,zorder=-1)
	ax[1].set_xlabel('arcseconds from center')
	ax[1].set_ylabel(r'$\nabla$(W1-W2) [magnitude/arcsecond]')

	### symmetric y-axis
	ymax = np.abs(ax[1].get_ylim()).max()
	ax[1].set_ylim(-ymax,ymax)

	### second axis
	y1, y2 = ax[1].get_ylim()
	x1, x2 = ax[1].get_xlim()
	ax2 = ax[1].twiny()
	ax2.set_xlim(x1*phys_scale, x2*phys_scale)
	ax2.set_xlabel(r'kpc from center')
	ax2.set_ylim(y1, y2)
	
	return grad_kpc, graderr_kpc, arcsec_calc, obj_size

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

def download_wise_data(alldata, idx=Ellipsis):

	# unWISE images (http://unwise.me/imgsearch/)

	dir = '/Users/joel/code/python/threedhst_bsfh/data/brownseds_data/fits/unWISE/'
	url = '"http://unwise.me/cutout_fits?version=neo1&size=60&bands=12&file_img_m=on&file_invvar_m=on&'
	ra,dec,objnames = brown_io.load_coordinates()
	with open(dir+'download.sh', 'w') as f:

		for dat in np.array(alldata)[idx]:
			match = dat['objname'] == objnames

			folder = dir+dat['objname'].replace(' ','_')
			if not os.path.isdir(folder):
				os.makedirs(folder)

			ra_string = 'ra={0}'.format(float(ra[match]))
			dec_string =  'dec={0}'.format(float(dec[match]))
			str = url + ra_string + '&' + dec_string
			filename = dat['objname'].replace(' ','_')+'.tar.gz'
			f.write('wget '+str+'" -O '+filename+'\n')
			f.write('mv '+filename+' '+folder+'/'+'\n')
			f.write('tar -xvf '+folder+'/'+filename+' -C '+folder+'\n')
			f.write('rm '+folder+'/'+filename+'\n')

def	load_wise_data(objname,filter,load_other=False):

	dir = '/Users/joel/code/python/threedhst_bsfh/data/brownseds_data/fits/unWISE/'+objname.replace(' ','_')+'/'
	files = [f for f in os.listdir(dir) if filter.lower() in f]
	if load_other:
		files = files[2:]

	img = fits.open(dir+files[0])[0]
	ivar = fits.open(dir+files[1])[0]

	return img, ivar

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

def convert_to_color(flux1,flux2,err1,err2,filter1,filter2,minflux=1e-10,vega_conversions=False):

	'''
	convert fluxes in fnu into Vega-scale colors
	'''

	flux1_clip = np.clip(flux1,minflux,np.inf)
	flux2_clip = np.clip(flux2,minflux,np.inf)
	if vega_conversions:
		vega_conv = (vega_conversions(filter1) - vega_conversions(filter2))
	else:
		vega_conv = 0.0

	x = flux1_clip/flux2_clip
	flux_color = -2.5*np.log10(x)+vega_conv

	dc_df1 = (-2.5 / (np.log(10)*flux1_clip))
	dc_df2 = (-2.5 / (np.log(10)*flux2_clip))
	if err1 is not None:
		flux_color_err = np.sqrt( dc_df1**2 * (err1)**2 + dc_df2**2 * (err2)**2 )
	else:
		return flux_color
	return flux_color, flux_color_err

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

import threed_dutils
import os
import numpy as np
import matplotlib.pyplot as plt
import magphys_plot_pref

def plot(runname='brownseds_np'):

	filebase,params,ancilname=threed_dutils.generate_basenames(runname)
	ngals = len(filebase)

	nfail = 0

	# name output
	outname = os.getenv('APPS')+'/threedhst_bsfh/plots/'+runname+'/pcomp/ml_vs_color.png'
	outname_2 = os.getenv('APPS')+'/threedhst_bsfh/plots/'+runname+'/pcomp/twoml_vs_color.png'

	# filler arrays
	br_color = np.zeros(ngals)
	ml_b     = np.zeros(ngals)
	ml_k     = np.zeros(ngals)

	# calulate optical and K-band mass-to-light ratios for each galaxy
	sps = None
	for jj in xrange(ngals):

		try:
			sample_results, powell_results, model = threed_dutils.load_prospector_data(filebase[jj])
		except:
			print 1/0

		if sps is None:
			# generate SPS model
			from prospect.models import model_setup
			sps = model_setup.load_sps(**sample_results['run_params'])
			wave = sps.wavelengths

		# grab maximum probability spectrum at z = 0.0
		thetas = sample_results['bfit']['maxprob_params']
		model.params['zred'] = np.array([0.0])
		specmax,magsmax,_ = model.mean_model(thetas, sample_results['obs'], sps=sps)

		# calculate magnitudes
		# spectra are in AB maggies
		# need to be in Lsun / Hz
		specmax *= 3631 * 1e-23 / 3.846e33 * (4*np.pi*(10*3.08568e18)**2)
		bmag,blum=threed_dutils.integrate_mag(wave,specmax,'b_cosmos')
		rmag,rlum=threed_dutils.integrate_mag(wave,specmax,'r_cosmos')
		kmag,klum=threed_dutils.integrate_mag(wave,specmax,'k_cosmos')

		bmag_sun = 5.47
		kmag_sun = 3.33

		ab_to_vega_b = 0.09
		ab_to_vega_r = -0.21
		ab_to_vega_k = -1.9

		mass = 10**thetas[0]
		ml_b[jj] = mass / (10**((bmag_sun-(bmag+ab_to_vega_b))/2.5))
		ml_k[jj] = mass / (10**((kmag_sun-(kmag+ab_to_vega_k))/2.5))
		br_color[jj] = (bmag+ab_to_vega_b) - (rmag+ab_to_vega_r)

		print ml_b[jj], ml_k[jj]
	
	xlim = [0.2,1.6]
	alpha = 0.5
	mew=2.2
	ms=10
	color = '0.2'
	fig, ax = plt.subplots(2,1,figsize=(6,10))
	plt.subplots_adjust(hspace=0.0,top=0.95,bottom=0.05)
	ax[0].plot(br_color,np.log10(ml_b),'o',alpha=alpha,color=color,ms=ms,mew=mew)
	ax[0].set_ylim(-1.6,1.4)
	ax[0].set_xlim(xlim)
	ax[0].set_ylabel(r'log(M/L$_\mathrm{B}$)')
	ax[0].set_xticklabels([])

	ax[1].plot(br_color,np.log10(ml_k),'o',alpha=alpha,color=color,mew=mew,ms=ms)
	ax[1].set_ylim(-1.6,0.5)
	ax[1].set_xlim(xlim)
	ax[1].set_xlabel('B-R (mag)')
	ax[1].set_ylabel(r'log(M/L$_\mathrm{K}$)')

	plt.tight_layout()
	plt.savefig(outname,dpi=150)
	os.system('open '+outname)
	plt.close()
	print 1/0
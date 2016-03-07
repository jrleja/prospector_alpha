import threed_dutils
import os
import numpy as np
import matplotlib.pyplot as plt
import magphys_plot_pref

def plot(runname='brownseds_tightbc'):

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

	# generate SPS model
	sps = threed_dutils.setup_sps(zcontinuous=2)
	wave = sps.wavelengths

	# calulate optical and K-band mass-to-light ratios for each galaxy
	for jj in xrange(ngals):

		sample_results, powell_results, model = threed_dutils.load_prospector_data(filebase[jj])

		# grab maximum probability spectrum at z=zobs
		thetas = sample_results['bfit']['maxprob_params']
		model.params['zred'] = np.atleast_1d(0.0)
		specmax,magsmax,_ = model.mean_model(thetas, sample_results['obs'], sps=sps)

		# calculate magnitudes
		bmag,blum=threed_dutils.integrate_mag(wave,specmax,'b_cosmos')
		rmag,rlum=threed_dutils.integrate_mag(wave,specmax,'r_cosmos')
		kmag,klum=threed_dutils.integrate_mag(wave,specmax,'k_cosmos')

		bmag_sun = 5.47
		kmag_sun = 3.33

		ab_to_vega_b = 0.09
		ab_to_vega_r = -0.21
		ab_to_vega_k = -1.9

		mass = thetas[0]
		ml_b[jj] = mass / (10**((bmag_sun-(bmag+ab_to_vega_b))/2.5))
		ml_k[jj] = mass / (10**((kmag_sun-(kmag+ab_to_vega_k))/2.5))
		br_color[jj] = (bmag+ab_to_vega_b) - (rmag+ab_to_vega_r)

		print ml_b[jj], ml_k[jj]
	
	xlim = [0.2,1.6]
	alpha = 0.7
	fig, ax = plt.subplots(2,1,figsize=(7.5,13))
	plt.subplots_adjust(hspace=0.0,top=0.95,bottom=0.05)
	ax[0].plot(br_color,np.log10(ml_b),'o',alpha=alpha,color='#1C86EE')
	ax[0].set_ylim(-1.6,1.4)
	ax[0].set_xlim(xlim)
	ax[0].set_ylabel(r'log(M/L$_\mathrm{B}$)')
	ax[0].set_xticklabels([])

	ax[1].plot(br_color,np.log10(ml_k),'o',alpha=alpha,color='#1C86EE')
	ax[1].set_ylim(-1.6,0.5)
	ax[1].set_xlim(xlim)
	ax[1].set_xlabel('B-R (mag)')
	ax[1].set_ylabel(r'log(M/L$_\mathrm{K}$)')

	plt.savefig(outname,dpi=150)
	os.system('open '+outname)
	print 1/0
	plt.close()
import numpy as np
import fsps, prosp_dutils, os
from bsfh import model_setup
import matplotlib.pyplot as plt
import read_sextractor
from astropy.cosmology import WMAP9 as cosmo
import matplotlib.pyplot as plt


'''
remove from logspace, add errors
compare to johannes' plots
plot control sample
'''

def av_to_dust2(av):

	# FSPS
	# dust2: opacity at 5500 
	# e.g., tau1 = (lam/5500)**dust_index)*dust1, and flux = flux*np.exp(-tautot)
	# 
	# Calzetti
	# A_V = Rv (A_B - A_V)
	# A_V = Rv*A_B / (1+Rv)
	# Rv = 4.05
	# 0.63um < lambda < 2.20um
	#k = 2.659(-1.857+1.040/lam) + Rv
	# 0.12um < lambda < 0.63um
	#k = 2.659*(-2.156+1.509/lam - 0.198/(lam**2)+0.11/(lam**3))+Rv
	# F(lambda) = Fobs(lambda)*10**(0.4*E(B-V)*k)

	# eqn: 10**(0.4*E(B-V)*k) = np.exp(-tau), solve for opacity(Av)

	# http://webast.ast.obs-mip.fr/hyperz/hyperz_manual1/node10.html
	# A_5500 = k(lambda)/Rv * Av

	# first, calculate Calzetti extinction at 5500 Angstroms
	#lam = 5500/1e4   # in microns
	#Rv = 4.05        # Calzetti R_v
	#k_over_rv = (2.659*(-2.156+1.509/lam - 0.198/(lam**2)+0.11/(lam**3))+Rv) / Rv
	#A_5500 = av*k_over_rv

	# now convert from extinction (fobs = fint * 10 ** -0.4*extinction)
	# into opacity (fobs = fint * exp(-opacity)) [i.e, dust2]
	#tau = -np.log(10**(-0.4*A_5500))
	#return av
	tau = -np.log(10**(-0.4*av))
	return tau

def main(control=False):

	ids = np.array([142,7225,12186,24639,37089,43597,
	            1237,7384,12196,26175,38169,44007,
	            1409,8067,14356,27441,38679,47430,
	            3215,8256,14781,33940,38781,48089,
	            4431,9058,16574,36659,39376,48474,
	            7207,10530,20515,37035,42560,49550])
	if control:
		ids = np.array([24270,18933,40462,781,20129,43959,
			             4394,17362,8650,34645,44758,18137,
			             24717,4048,4747,341,36017,4747,
			             40152,4394,12578,38171,45883,42503,
			             46442,40462,45691,22568,34753,29984,
			             28743,17559,29765,7024,44985,7584])

	# setup stellar populations
	sps = fsps.StellarPopulation(zcontinuous=1, compute_vega_mags=False)
	custom_filter_keys = os.getenv('APPS')+'/threedhst_bsfh/filters/filter_keys_threedhst.txt'
	fsps.filters.FILTERS = model_setup.custom_filter_dict(custom_filter_keys)

	# load model
	pfile = '/Users/joel/code/python/threedhst_bsfh/parameter_files/fast_mimic/fast_mimic.py'
	model = model_setup.load_model(param_file=pfile)
	parnames = np.array(model.theta_labels())

	# load FAST fits, photometry
	fastcat = read_sextractor.load_fast_v41('GOODS-S')
	photcat = read_sextractor.load_phot_v41('GOODS-S')
	phot = []
	fast = []
	for idnum in ids:
		index = fastcat['id'] == idnum
		fast.append(fastcat[index])
		phot.append(photcat[index])

	# set up filters
	keys = np.loadtxt(os.getenv('APPS')+'/threedhst_bsfh/filters/filter_keys_threedhst.txt', dtype='S20',usecols=[1])
	filterlist = [x for x in keys if x[-7:] == 'GOODS-S']
	obs = {'filters':filterlist,'wavelength':None}
	lamefflist = [x.lower() for x in filterlist]
	lameff     = prosp_dutils.return_mwave_custom(lamefflist)
	modellam   = np.log10(np.array(lameff))

	# set up figure
	fig, axes = plt.subplots(6, 6, figsize=(12, 12))
	fig.subplots_adjust(wspace=0.000,hspace=0.000)

	# loop over galaxies
	ngal = len(ids)
	clipped_tage = np.zeros(ngal)

	for kk in xrange(ngal):

		ind1 = kk % 6
		ind2 = kk / 6

		# set up in loop
		# note: for FSPS, 0.01 < tau < 100, and 0.1 < lage
		model.params['zred']                   = np.array(fast[kk]['z'][0])
		model.initial_theta[parnames=='tage']  = (10**fast[kk]['lage'][0])/1e9
		model.initial_theta[parnames=='tau']   = (10**fast[kk]['ltau'][0])/1e9
		model.initial_theta[parnames=='dust2'] = av_to_dust2(fast[kk]['Av'][0])
		model.initial_theta[parnames=='mass']  = 10**fast[kk]['lmass'][0]

		if model.initial_theta[parnames=='tage'] < 0.101:
			model.initial_theta[parnames=='tage'] = 0.101
			clipped_tage[kk] = 1

		# get fiducial model
		spec, mags, w = model.mean_model(model.initial_theta, obs, sps=sps, norm_spec=True)

		# get dusty model
		model.params['add_dust_emission']  = np.array(True)
		model.params['add_agb_dust_model'] = np.array(True)
		spec_dust, mags_dust, w = model.mean_model(model.initial_theta, obs, sps=sps, norm_spec=True)
		model.params['add_dust_emission']  = np.array(False)
		model.params['add_agb_dust_model'] = np.array(False)

		# get obs [irac only]
		irflux  = np.array([phot[kk][0][i] for i in xrange(len(phot[kk][0])) if phot[kk][0].dtype.names[i][:6] == 'f_IRAC' ])
		irerrs  = np.array([phot[kk][0][i] for i in xrange(len(phot[kk][0])) if phot[kk][0].dtype.names[i][:6] == 'e_IRAC' ])
		
		# get obs
		irflux  = np.array([phot[kk][0][i] for i in xrange(len(phot[kk][0])) if phot[kk][0].dtype.names[i][:2] == 'f_' ])
		irerrs  = np.array([phot[kk][0][i] for i in xrange(len(phot[kk][0])) if phot[kk][0].dtype.names[i][:2] == 'e_' ])
		names   = [phot[kk][0].dtype.names[i][2:]+'_GOODS-S' for i in xrange(len(phot[kk][0])) if phot[kk][0].dtype.names[i][:2] == 'e_' ]
		names   = np.array([x.lower() for x in names])
		names_irac1 = names == 'irac1_goods-s'
		iraclam = prosp_dutils.return_mwave_custom(names) 

		# remove non-detections
		good    = irflux > 0
		irflux  = irflux[good]
		irerrs  = irerrs[good]
		iraclam = np.log10(np.array(iraclam))[good]

		# find obs offset
		irflux = irflux/1e10
		irerrs = irerrs/1e10
		offset = np.log10(mags[-5])-np.log10(irflux[names_irac1[good]])

		# plot
		axes[ind1,ind2].plot(modellam,np.log10(mags),'bo', alpha=0.75,ms=5.5)
		axes[ind1,ind2].plot(np.log10(w),np.log10(spec),'b-')

		axes[ind1,ind2].plot(modellam,np.log10(mags_dust),'ro', alpha=0.75,ms=5.5)
		axes[ind1,ind2].plot(np.log10(w),np.log10(spec_dust),'r-')

		#axes[ind1,ind2].errorbar(iraclam,irflux+offset,yerr=irerrs,fmt='ko', alpha=0.75,ms=5.5)
		axes[ind1,ind2].plot(iraclam,np.log10(irflux)+offset,'ko', alpha=0.75,ms=5.5)
		axes[ind1,ind2].set_xlim(np.min(iraclam)*0.98,np.max(iraclam)*1.03)
		axes[ind1,ind2].set_xlim(4.4,5.0)
		axes[ind1,ind2].set_ylim(np.min(np.log10(mags_dust[-5:-1]))*1.1,np.max(np.log10(mags_dust))*0.98)

		# text
		if clipped_tage[kk]>0:
			axes[ind1,ind2].text(0.06,0.92, 'tage clipped',transform = axes[ind1,ind2].transAxes, fontsize=6)
		#axes[ind1,ind2].text(0.06,0.74, 'offset='+"{:.2f}".format(offset/mags[0]*100)+'%',transform = axes[ind1,ind2].transAxes, fontsize=6)
		#axes[ind1,ind2].text(0.06,0.74, str(fast[kk]['Av'][0]) + ' ' + str(sps.params['dust2']),transform = axes[ind1,ind2].transAxes, fontsize=12)
		#axes[ind1,ind2].text(0.06,0.66, 'ID: '+str(ids[kk]),transform = axes[ind1,ind2].transAxes, fontsize=6)

		todisplay = ['id','z','ltau','lage','Av','lmass']
		ytext     = [0.87,0.82,0.77,0.72,0.67,0.62]
		for mm in xrange(len(fast[kk].dtype)):
			var = fast[kk].dtype.names[mm]
			if var in todisplay:
				axes[ind1,ind2].text(0.04,ytext[todisplay.index(var)],
					 var+'='+str(fast[kk][var][0]),
					 transform = axes[ind1,ind2].transAxes, fontsize=6)


		# no x,y-labels
		axes[ind1,ind2].set_xticklabels([])
		axes[ind1,ind2].set_yticklabels([])

	if control:
		plt.savefig('control.png',dpi=300)
	else:
		plt.savefig('agncandidates.png',dpi=300)











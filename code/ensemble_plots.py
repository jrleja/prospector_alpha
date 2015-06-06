import os, threed_dutils, pickle, extra_output, triangle, pylab
from bsfh import read_results
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.cosmology import WMAP9
from astropy import constants
from matplotlib import cm
from scipy.interpolate import interp1d

# minimum flux: no model emission line has strength of 0!
pc2cm = 3.08567758e18
minmodel_flux = 2e-1

def age_vs_mass():
	'''
	one-off plot, delta(mass) versus delta(t_half)
	'''

	outname = os.getenv('APPS')+'/threedhst_bsfh/results/dtau_ha_zperr/dtau_ha_zperr_ensemble.pickle'
	with open(outname, "rb") as f:
		ensemble_zperr=pickle.load(f)

	outname = os.getenv('APPS')+'/threedhst_bsfh/results/dtau_ha_plog/dtau_ha_plog_ensemble.pickle'
	with open(outname, "rb") as f:
		ensemble_plog=pickle.load(f)

	valid_comp = (ensemble_zperr['z'] > 0) & (ensemble_plog['z'] > 0)

	mass_plog = np.log10(ensemble_plog['q50'][ensemble_plog['parname'] == 'totmass'][0,valid_comp])
	mass_zperr = np.log10(ensemble_zperr['q50'][ensemble_zperr['parname'] == 'totmass'][0,valid_comp])

	sfr_plog = np.log10(ensemble_plog['q50'][ensemble_plog['parname'] == 'sfr_100'][0,valid_comp])
	sfr_zperr = np.log10(ensemble_zperr['q50'][ensemble_zperr['parname'] == 'sfr_100'][0,valid_comp])

	thalf_plog = ensemble_plog['q50'][ensemble_plog['parname'] == 'half_time'][0,valid_comp]
	thalf_zperr = ensemble_zperr['q50'][ensemble_zperr['parname'] == 'half_time'][0,valid_comp]

	plt.plot(mass_plog-mass_zperr,thalf_plog-thalf_zperr,'bo',alpha=0.5)
	plt.xlabel(r'log(M$_{\rm ltau}$)-log(M)')
	plt.ylabel(r't$_{\rm half,ltau}$-t$_{\rm half}$ [Gyr]')
	plt.hlines(0,plt.xlim()[0],plt.xlim()[1],linestyle='--')
	plt.vlines(0,plt.ylim()[0],plt.ylim()[1],linestyle='--')
	plt.savefig('/Users/joel/code/python/threedhst_bsfh/plots/mass_vs_halftime.png',dpi=300)
	plt.close()

	plt.plot(sfr_plog-sfr_zperr,thalf_plog-thalf_zperr,'bo',alpha=0.5)
	plt.xlabel(r'log(SFR$_{\rm ltau})$-log(SFR)')
	plt.ylabel(r't$_{\rm half,ltau}$-t$_{\rm half}$ [Gyr]')
	plt.hlines(0,plt.xlim()[0],plt.xlim()[1],linestyle='--')
	plt.vlines(0,plt.ylim()[0],plt.ylim()[1],linestyle='--')
	plt.savefig('/Users/joel/code/python/threedhst_bsfh/plots/sfr_vs_halftime.png',dpi=300)
	plt.close()

def asym_errors(center, up, down, log=False):

	if log:
		errup = np.log10(up)-np.log10(center)
		errdown = np.log10(center)-np.log10(down)
		errarray = [errdown,errup]
	else:
		errarray = [center-down,up-center]

	return errarray

def restore_mips_info(sample_results):
	''' temporary function'''

	from copy import copy
	nsamp_mc = 20

	# initialize sps
	# check to see if we want zcontinuous=2 (i.e., the MDF)
	if np.sum([1 for x in sample_results['model'].config_list if x['name'] == 'pmetals']) > 0:
		sps = threed_dutils.setup_sps()
	else:
		sps = threed_dutils.setup_sps(zcontinuous=1)

	# set up MIPS + fake L_IR filter
	mips_flux = np.zeros(nsamp_mc)
	mips_index = [i for i, s in enumerate(sample_results['obs']['filters']) if 'mips' in s]
	botlam = np.atleast_1d(8e4-1)
	toplam = np.atleast_1d(1000e4+1)
	edgetrans = np.atleast_1d(0)
	lir_filter = [[np.concatenate((botlam,np.linspace(8e4, 1000e4, num=100),toplam))],
	              [np.concatenate((edgetrans,np.ones(100),edgetrans))]]

	# first randomize
    # use flattened and thinned chain for random posterior draws
	flatchain = copy(sample_results['flatchain'])
	lir      = np.zeros(nsamp_mc)
	np.random.shuffle(flatchain)
	z = np.atleast_1d(sample_results['model'].params['zred'])

	for jj in xrange(nsamp_mc):
		thetas = flatchain[jj,:]

		# calculate redshifted magnitudes
		sample_results['model'].params['zred'] = np.atleast_1d(z)
		spec_neboff,mags_neboff,w = sample_results['model'].mean_model(thetas, sample_results['obs'], sps=sps,norm_spec=False)

		# add mips flux info
		mips_flux[jj] = mags_neboff[mips_index][0]*1e10 # comes out in maggies, convert to flux such that AB zeropoint is 25 mags

		# now calculate z=0 magnitudes
		sample_results['model'].params['zred'] = np.atleast_1d(0.00)
		spec_neboff,mags_neboff,w = sample_results['model'].mean_model(thetas, sample_results['obs'], sps=sps,norm_spec=False)

		# calculate L_IR intrinsic
		_,lir[jj]     = threed_dutils.integrate_mag(w,spec_neboff,lir_filter, z=None, alt_file=None) # comes out in ergs/s
		lir[jj]       = lir[jj] / 3.846e33 #  convert to Lsun

	###### FORMAT MIPS OUTPUT
	mips = {'mips_flux':mips_flux,'L_IR':lir}
	sample_results['mips'] = mips
	return sample_results

def collate_output(runname,outname):

	'''
	prototype for generating useful output balls
	currently saves q16, q50, q84, maxprob, and parmlist for each galaxy

	in the future, want it to include:
	maximum likelihood fit (both parameters and likelihood)
	mean acceptance fraction
	also output information about the powell minimization process
	pri[0].x = min([p.fun for p in pr])   <--- if this is true, then the minimization process chose the processor with initial conditions
	np.array([p.x for p in pr])
	'''

	filebase,params,ancilname=threed_dutils.generate_basenames(runname)
	ngals = len(filebase)

	nfail = 0
	for jj in xrange(ngals):

		# find most recent output file
		# with the objname
		folder = "/".join(filebase[jj].split('/')[:-1])
		filename = filebase[jj].split("/")[-1]
		files = [f for f in os.listdir(folder) if "_".join(f.split('_')[:-2]) == filename]	
		times = [f.split('_')[-2] for f in files]

		# if we found no files, skip this object
		if len(times) == 0:
			print 'Failed to find any files in '+folder+' of type ' +filename+' to extract times'
			nfail+=1
			continue

		# load results
		mcmc_filename=filebase[jj]+'_'+max(times)+"_mcmc"
		model_filename=filebase[jj]+'_'+max(times)+"_model"

		try:
			sample_results, powell_results, model = read_results.read_pickles(mcmc_filename, model_file=model_filename,inmod=None)
		except (ValueError,EOFError,KeyError):
			print mcmc_filename + ' failed during output writing'
			nfail+=1
			continue
		except IOError:
			print mcmc_filename + ' does not exist!'
			nfail+=1
			continue

		# check for existence of extra information
		try:
			sample_results['quantiles']
		except:
			print 'Generating extra information for '+mcmc_filename+', '+model_filename
			extra_output.post_processing(params[jj])
			sample_results, powell_results, model = read_results.read_pickles(mcmc_filename, model_file=model_filename,inmod=None)

		# initialize output arrays if necessary
		ntheta = len(sample_results['initial_theta'])+len(sample_results['extras']['parnames'])
		try:
			q_16
		except:
			q_16, q_50, q_84 = (np.zeros(shape=(ntheta,ngals))+np.nan for i in range(3))
			thetamax = np.zeros(shape=(ntheta-len(sample_results['extras']['parnames']),ngals))
			z, mips_sn = (np.zeros(ngals) for i in range(2))
			mips_flux, L_IR = (np.zeros(shape=(3,ngals)) for i in range(2))
			output_name = np.empty(0,dtype='object')
			obs,model_emline,ancildat = [],[],[]

		# insert percentiles
		q_16[:,jj], q_50[:,jj], q_84[:,jj] = np.concatenate((sample_results['quantiles']['q16'],sample_results['extras']['q16'])),\
		              						 np.concatenate((sample_results['quantiles']['q50'],sample_results['extras']['q50'])),\
		              						 np.concatenate((sample_results['quantiles']['q84'],sample_results['extras']['q84']))
		
		# miscellaneous output
		z[jj] = np.atleast_1d(sample_results['model_params'][0]['init'])[0]
		mips_sn[jj] = sample_results['obs']['maggies'][-1]/sample_results['obs']['maggies_unc'][-1]
		output_name=np.append(output_name,filename)

		# MIPS information
		try:
			q_16_mips, q_50_mips, q_84_mips = triangle.quantile(sample_results['mips']['mips_flux'], [0.16, 0.5, 0.84])
			q_16_lir, q_50_lir, q_84_lir = triangle.quantile(sample_results['mips']['L_IR'], [0.16, 0.5, 0.84])
			mips_flux[:,jj] = np.array([q_16_mips, q_50_mips, q_84_mips])
			L_IR[:,jj] = np.array([q_16_lir, q_50_lir, q_84_lir])
		except:
			print 1/0

		# grab best-fitting model
		thetamax[:,jj] = sample_results['quantiles']['maxprob_params']

		# save dictionary lists
		obs.append(sample_results['obs'])
		model_emline.append(sample_results['model_emline'])
		try:
			ancildat.append(threed_dutils.load_ancil_data(os.getenv('APPS')+'/threedhst_bsfh/data/'+ancilname,
			            							  sample_results['run_params']['objname']))
		except TypeError:
			pass

		print jj

	print 'total galaxies: {0}, successful loads: {1}'.format(ngals,ngals-nfail)
	print 'saving in {0}'.format(outname)
	fastname = sample_results['run_params']['fastname']

	output = {'outname': output_name,\
			  'fastname': fastname,\
			  'parname': np.concatenate([sample_results['model'].theta_labels(),sample_results['extras']['parnames']]),\
		      'q16': q_16,\
		      'q50': q_50,\
		      'q84': q_84,\
		      'maxprob': thetamax,\
		      'mips_sn': mips_sn,\
		      'mips_flux': mips_flux,\
		      'L_IR': L_IR,
		      'z':z,
		      'obs':obs,
		      'model_emline':model_emline,
		      'ancildat':ancildat}

	pickle.dump(output,open(outname, "wb"))
		
def plot_driver(runname):
	
	#runname = "neboff"
	#runname = 'photerr'

	outname = os.getenv('APPS')+'/threedhst_bsfh/results/'+runname+'/'+runname+'_ensemble.pickle'

	# if the save file doesn't exist, make it
	if not os.path.isfile(outname):
		collate_output(runname,outname)

	with open(outname, "rb") as f:
		ensemble=pickle.load(f)
	
	# get SFR_observed
	sfrmin = 1e-2
	sfrmax = 1e4
	sfr_obs = np.array([x['sfr'][0] for x in ensemble['ancildat']])
	sfr_obs_uv = np.array([x['sfr_UV'][0] for x in ensemble['ancildat']])
	nodet = sfr_obs == -99
	sfr_obs[nodet] = sfr_obs_uv[nodet]
	z_sfr = np.array([x['z_sfr'] for x in ensemble['ancildat']])
	valid_comp = ensemble['z'] > 0

	# load FAST stuff
	filename = '/threedhst_bsfh'+ensemble['fastname'].split('/threedhst_bsfh')[1]
	fast,values = threed_dutils.load_fast_3dhst(os.getenv('APPS')+filename, None)
	fastmass = fast[:,np.array(values)=='lmass'].reshape(fast.shape[0])
	fastsfr  = 10**fast[:,np.array(values)=='lsfr'].reshape(fast.shape[0])
	fastav   = fast[:,np.array(values)=='Av'].reshape(fast.shape[0])
	
	# get FAST halphas
	fhalpha = np.zeros(len(fastsfr))
	for kk in xrange(len(fastsfr)):
		synth_emlines = threed_dutils.synthetic_emlines(10**fastmass[kk],fastsfr[kk],0.0,fastav[kk],None)
		fhalpha[kk]   = np.log10(synth_emlines['flux'][0])

	# format FAST
	fastsfr = np.log10(np.clip(fastsfr,sfrmin,sfrmax))

	# calculate tuniv
	tuniv=WMAP9.age(ensemble['z'][valid_comp]).value

	if np.sum(z_sfr-ensemble['z'][valid_comp]) != 0:
		print "you got some redshift mismatches yo"

	# get observed emission line strength
	obs_ha = np.array([x['Ha_flux'][0] for x in ensemble['ancildat']])
	obs_ha_err = np.array([x['Ha_error'][0] for x in ensemble['ancildat']])

	# convert to luminosity
	pc2cm = 3.08567758e18
	distances = WMAP9.luminosity_distance(ensemble['z'][valid_comp]).value*1e6*pc2cm
	lobs_halpha = obs_ha*(4*np.pi*distances**2)*1e-17
	lobs_halpha_err = obs_ha_err*(4*np.pi*distances**2)*1e-17


	# Create parameters that 
	# vary between models
	# currently, total mass, dust1, dust2, tburst, tau
	try:
		mass = np.log10(ensemble['q50'][ensemble['parname'] == 'mass'][0,valid_comp])
		masserrs = asym_errors(ensemble['q50'][ensemble['parname'] == 'mass'][0,valid_comp],ensemble['q84'][ensemble['parname'] == 'mass'][0,valid_comp], ensemble['q16'][ensemble['parname'] == 'mass'][0,valid_comp],log=True)
	except:
		mass = np.log10(ensemble['q50'][ensemble['parname'] == 'totmass'][0,valid_comp])
		masserrs = asym_errors(ensemble['q50'][ensemble['parname'] == 'totmass'][0,valid_comp],ensemble['q84'][ensemble['parname'] == 'totmass'][0,valid_comp], ensemble['q16'][ensemble['parname'] == 'totmass'][0,valid_comp],log=True)

	logzsol = ensemble['q50'][ensemble['parname'] == 'logzsol'][0,valid_comp]
	logzsolerrs = asym_errors(ensemble['q50'][ensemble['parname'] == 'logzsol'][0,valid_comp],ensemble['q84'][ensemble['parname'] == 'logzsol'][0,valid_comp],ensemble['q16'][ensemble['parname'] == 'logzsol'][0,valid_comp])

	sfr_100 = np.log10(np.clip(ensemble['q50'][ensemble['parname'] == 'sfr_100'][0],sfrmin,sfrmax))
	sfr_100errs = np.array(asym_errors(np.clip(ensemble['q50'][ensemble['parname'] == 'sfr_100'][0],sfrmin,sfrmax),
		                               np.clip(ensemble['q84'][ensemble['parname'] == 'sfr_100'][0],sfrmin,sfrmax),
		                               np.clip(ensemble['q16'][ensemble['parname'] == 'sfr_100'][0],sfrmin,sfrmax),log=True))

	try:
		tau = np.log10(np.clip(tuniv-ensemble['q50'][ensemble['parname'] == 'tau'][0,valid_comp],1e-2,1e50))
		tauerrs = asym_errors(np.clip(tuniv-ensemble['q50'][ensemble['parname'] == 'tau'][0,valid_comp],1e-4,1e50),np.clip(tuniv-ensemble['q84'][ensemble['parname'] == 'tau'][0,valid_comp],1e-4,1e50),np.clip(tuniv-ensemble['q16'][ensemble['parname'] == 'tau'][0,valid_comp],1e-4,1e50),log=True)
	except:
		tau = np.log10(np.clip(tuniv-ensemble['q50'][ensemble['parname'] == 'tau_1'][0,valid_comp],1e-2,1e50))
		tauerrs = asym_errors(np.clip(tuniv-ensemble['q50'][ensemble['parname'] == 'tau_1'][0,valid_comp],1e-4,1e50),np.clip(tuniv-ensemble['q84'][ensemble['parname'] == 'tau_1'][0,valid_comp],1e-4,1e50),np.clip(tuniv-ensemble['q16'][ensemble['parname'] == 'tau_1'][0,valid_comp],1e-4,1e50),log=True)

	x_data = [mass,\
			  logzsol,\
	          logzsol,\
	          logzsol,\
	          sfr_100,\
	          sfr_100,\
	          mass,\
	          np.log10(ensemble['q50'][ensemble['parname'] == 'ssfr_100'][0]),\
	          np.log10(sfr_obs),\
	          np.log10(sfr_obs),\
	          np.log10(np.clip(np.array([x['q50'][x['name'] == 'Halpha'] for x in ensemble['model_emline']]),minmodel_flux,1e50)),
	          np.log10(lobs_halpha),\
	          fastmass[valid_comp],\
	          fastsfr[valid_comp],
	          fhalpha[valid_comp]\
	          ]

	x_err  = [masserrs,\
			  logzsolerrs,\
			  logzsolerrs,\
			  logzsolerrs,\
			  sfr_100errs,\
			  sfr_100errs,\
			  masserrs,\
			  asym_errors(ensemble['q50'][ensemble['parname'] == 'ssfr_100'][0],ensemble['q84'][ensemble['parname'] == 'ssfr_100'][0],ensemble['q16'][ensemble['parname'] == 'sfr_100'][0],log=True),\
			  None,\
			  None,\
			  asym_errors(np.clip(np.array([x['q50'][x['name'] == 'Halpha'] for x in ensemble['model_emline']]),minmodel_flux,1e50),np.clip(np.array([x['q84'][x['name'] == 'Halpha'] for x in ensemble['model_emline']]),minmodel_flux,1e50),np.clip(np.array([x['q16'][x['name'] == 'Halpha'] for x in ensemble['model_emline']]),minmodel_flux,1e50),log=True),\
			  asym_errors(lobs_halpha,lobs_halpha+lobs_halpha_err,lobs_halpha-lobs_halpha_err,log=True),\
			  None,\
			  None,\
			  None\
			 ]

	x_labels = [r'log(M) [M$_{\odot}$]',
				r'log(Z$_{\odot}$)',
				r'log(Z$_{\odot}$)',
				r'log(Z$_{\odot}$)',
	            r'log(SFR_{100}) [M$_{\odot}$/yr]',
	            r'log(SFR_{100}) [M$_{\odot}$/yr]',
	            r'log(M) [M$_{\odot}$]',
	            r'log(sSFR_{100}) [yr$^{-1}$]',
	            r'log(SFR$_{obs}$) [M$_{\odot}$/yr]',
	            r'log(SFR$_{obs}$) [M$_{\odot}$/yr]',
	            r'log(H$\alpha$ flux) [model]',
	            r'log(H$\alpha$ lum) [obs]',
	            r'log(M$_{FAST}$) [M$_{\odot}$]',
	            r'log(SFR$_{FAST}$) [M$_{\odot}$/yr]',
	            r'log(H$_{\alpha,FAST}$) [cgs]'
	            ]

	y_data = [logzsol,\
			  sfr_100[valid_comp],\
			  ensemble['q50'][ensemble['parname'] == 'dust_index'][0,valid_comp],\
	          ensemble['q50'][ensemble['parname'] == 'half_time'][0,valid_comp],\
	          ensemble['q84'][ensemble['parname'] == 'half_time'][0]-ensemble['q16'][ensemble['parname'] == 'half_time'][0],\
	          ensemble['q50'][ensemble['parname'] == 'half_time'][0],\
	          sfr_100[valid_comp],\
	          ensemble['q84'][ensemble['parname'] == 'half_time'][0]-ensemble['q16'][ensemble['parname'] == 'half_time'][0],\
	          sfr_100[valid_comp],\
	          np.log10(ensemble['q50'][ensemble['parname'] == 'sfr_1000'][0,valid_comp]),\
	          tau,\
	          np.log10(sfr_obs),\
	          mass,\
	          sfr_100[valid_comp],\
	          np.log10(lobs_halpha)
	          ]

	y_err  = [logzsolerrs,\
			  sfr_100errs[:,valid_comp],\
			  asym_errors(ensemble['q50'][ensemble['parname'] == 'dust_index'][0,valid_comp],ensemble['q84'][ensemble['parname'] == 'dust_index'][0,valid_comp],ensemble['q16'][ensemble['parname'] == 'dust_index'][0,valid_comp],log=False),\
			  None,\
			  None,\
			  asym_errors(ensemble['q50'][ensemble['parname'] == 'half_time'][0],ensemble['q84'][ensemble['parname'] == 'half_time'][0],ensemble['q16'][ensemble['parname'] == 'half_time'][0]),\
			  sfr_100errs[:,valid_comp],\
			  None,\
			  sfr_100errs[:,valid_comp],\
			  asym_errors(ensemble['q50'][ensemble['parname'] == 'sfr_1000'][0,valid_comp],ensemble['q84'][ensemble['parname'] == 'sfr_1000'][0,valid_comp],ensemble['q16'][ensemble['parname'] == 'sfr_1000'][0,valid_comp],log=True),\
			  tauerrs,\
			  None,\
			  masserrs,\
			  sfr_100errs[:,valid_comp],\
			  asym_errors(lobs_halpha,lobs_halpha+lobs_halpha_err,lobs_halpha-lobs_halpha_err,log=True)\
			 ]

	y_labels = [r'log(Z$_{\odot}$)',
	            r'log(SFR$_{100}$) [M$_{\odot}$/yr]',
	            r'dust index',
	            r't$_{half}$ [Gyr]',
	            r'$\sigma$ (t$_{half}$) [Gyr]',
	            r't$_{half}$ [Gyr]',
	            r'log(SFR_{100}) [M$_{\odot}$/yr]',
	            r'$\sigma$ (t$_{half}$) [Gyr]',
	            r'log(SFR$_{mcmc,100}$) [M$_{\odot}$/yr]',
	            r'log(SFR$_{mcmc,1000}$) [M$_{\odot}$/yr]',
	            r'log($\tau$/Gyr)',
	            r'log(SFR$_{obs}$) [M$_{\odot}$/yr]',
	            r'log(M$_{PROSP}$) [M$_{\odot}$]',
	            r'log(SFR$_{PROSP}$) [M$_{\odot}$/yr]',
	            r'log(H$_{\alpha,obs}$) [M$_{\odot}$/yr]'
	            ]

	plotname = ['mass_metallicity',
				'logzsol_sfr100',
				'logzsol_dustindex',
				'logzsol_halftime',
				'sfr100_deltahalftime',
				'sfr100_halftime',
				'mass_sfr100',
				'ssfr_deltahalftime',
				'sfrobs_sfrmcmc100',
				'sfrobs_sfrmcmc1000',
				'modelemline_tau',
				'obsemline_sfrobs',
				'fastmass_comp',
				'fastsfr_comp',
				'fasthalpha_comp'
				]

	assert len(plotname) == len(y_labels) == len(y_err) == len(y_data), 'improper number of y data'
	assert len(plotname) == len(x_labels) == len(x_err) == len(x_data), 'improper number of x data'

	for jj in xrange(len(x_data)):
		outname = os.getenv('APPS') + '/threedhst_bsfh/plots/ensemble_plots/'+runname+'/'+plotname[jj]+'.png'
		
		fig = plt.figure()
		ax = fig.add_subplot(111)
		
		if len(x_data[jj]) != len(y_data[jj]):
			print 1/0

		ax.errorbar(x_data[jj],y_data[jj], 
			        fmt='bo', linestyle=' ', alpha=0.7)
		ax.errorbar(x_data[jj],y_data[jj], 
			        fmt=' ', ecolor='0.75', alpha=0.5,
			        yerr=y_err[jj], xerr=x_err[jj],linestyle=' ',
			        zorder=-32)

		ax.set_xlabel(x_labels[jj])
		ax.set_ylabel(y_labels[jj])

		# set plot limits to be slightly outside max values
		dynx, dyny = (np.nanmax(x_data[jj])-np.nanmin(x_data[jj]))*0.05,\
		             (np.nanmax(y_data[jj])-np.nanmin(y_data[jj]))*0.05
		
		ax.axis((np.nanmin(x_data[jj])-dynx,
			     np.nanmax(x_data[jj])+dynx,
			     np.nanmin(y_data[jj])-dyny,
			     np.nanmax(y_data[jj])+dyny,
			    ))

		# make sure comparison plot is 1:1
		if x_labels[jj] == r'log(SFR$_{obs}$) [M$_{\odot}$/yr]' or \
		   x_labels[jj] == r'log(sSFR$_{obs}$) [yr$^{-1}$]' or \
		   x_labels[jj] == r'log(M$_{FAST}$) [M$_{\odot}$]' or \
		   x_labels[jj] == r'log(SFR$_{FAST}$) [M$_{\odot}$/yr]' or \
		   x_labels[jj] == r'log(H$_{\alpha,FAST}$) [cgs]':

			if np.nanmin(x_data[jj])-dynx > np.nanmin(y_data[jj])-dyny:
				min = np.nanmin(y_data[jj])-dyny*3
			else:
				min = np.nanmin(x_data[jj])-dynx*3
			if np.nanmax(x_data[jj])+dynx > np.nanmax(y_data[jj])+dyny:
				max = np.nanmax(x_data[jj])+dynx*3
			else:
				max = np.nanmax(y_data[jj])+dyny*3

			ax.axis((min,max,min,max))
			ax.errorbar([-1e3,1e3],[-1e3,1e3],linestyle='--',color='0.1',alpha=0.8)

			n = np.sum(np.isfinite(y_data[jj]))
			detections = x_data[jj] > -1
			mean_offset = np.sum(y_data[jj][detections]-x_data[jj][detections])/n
			scat=np.sqrt(np.sum((y_data[jj][detections]-x_data[jj][detections]-mean_offset)**2.)/(n-2))
			ax.text(0.04,0.95, 'scatter='+"{:.2f}".format(scat)+' dex',transform = ax.transAxes)
			ax.text(0.04,0.9, 'mean offset='+"{:.2f}".format(mean_offset)+' dex',transform = ax.transAxes)


		print 'saving '+outname
 		plt.savefig(outname, dpi=300)
		plt.close()

def offset_and_scatter(x,y,biweight=True):

	n = len(x)
	mean_offset = np.sum(x-y)/n

	if biweight:
		diff = y-x
		Y0  = np.median(diff)

		# calculate MAD
		MAD = np.median(np.abs(diff-Y0))/0.6745

		# biweighted value
		U   = (diff-Y0)/(6.*MAD)
		UU  = U*U
		Q   = UU <= 1.0
		if np.sum(Q) < 3:
			print 'distribution is TOO WEIRD, returning -1'
			scat=-1

		N = len(diff)
		numerator = np.sum( (diff[Q]-Y0)**2 * (1-UU[Q])**4)
		den1      = np.sum( (1.-UU[Q])*(1.-5.*UU[Q]))
		siggma    = N*numerator/(den1*(den1-1.))

		scat      = np.sqrt(siggma)

	else:
		scat=np.sqrt(np.sum((x-y-mean_offset)**2.)/(n-2))

	return mean_offset,scat

def ml_vs_color(runname):

	filebase,params,ancilname=threed_dutils.generate_basenames(runname)
	ngals = len(filebase)

	nfail = 0

	# name output
	outname = os.getenv('APPS')+'/threedhst_bsfh/plots/ensemble_plots/'+runname+'/ml_vs_color.png'
	outname_2 = os.getenv('APPS')+'/threedhst_bsfh/plots/ensemble_plots/'+runname+'/ml_vs_color_twoml.png'

	# filler arrays
	br_color = np.zeros(ngals)
	ml_b     = np.zeros(ngals)
	ml_k     = np.zeros(ngals)

	for jj in xrange(ngals):

		# find most recent output file
		# with the objname
		folder = "/".join(filebase[jj].split('/')[:-1])
		filename = filebase[jj].split("/")[-1]
		files = [f for f in os.listdir(folder) if "_".join(f.split('_')[:-2]) == filename]	
		times = [f.split('_')[-2] for f in files]

		# if we found no files, skip this object
		if len(times) == 0:
			print 'Failed to find any files in '+folder+' of type ' +filename+' to extract times'
			nfail+=1
			continue

		# load results
		mcmc_filename=filebase[jj]+'_'+max(times)+"_mcmc"
		model_filename=filebase[jj]+'_'+max(times)+"_model"

		try:
			sample_results, powell_results, model = read_results.read_pickles(mcmc_filename, model_file=model_filename,inmod=None)
		except (ValueError,EOFError,KeyError):
			print mcmc_filename + ' failed during output writing'
			nfail+=1
			continue
		except IOError:
			print mcmc_filename + ' does not exist!'
			nfail+=1
			continue

		# check for existence of extra information
		try:
			sample_results['quantiles']
		except:
			print 'Generating extra information for '+mcmc_filename+', '+model_filename
			extra_output.post_processing(params[jj])
			sample_results, powell_results, model = read_results.read_pickles(mcmc_filename, model_file=model_filename,inmod=None)

		# check to see if we want zcontinuous=2 (i.e., the MDF)
		if np.sum([1 for x in sample_results['model'].config_list if x['name'] == 'pmetals']) > 0:
			sps = threed_dutils.setup_sps(zcontinuous=2)
			print 'using the MDF'
		else:
			sps = threed_dutils.setup_sps(zcontinuous=1)
			print 'using interpolated metallicities'

		# grab best-fit parameters at z=0
		thetas = sample_results['quantiles']['maxprob_params']		
		specmax,magsmax,w = sample_results['model'].mean_model(thetas, sample_results['obs'], sps=sps,norm_spec=False)

		bmag,blum=threed_dutils.integrate_mag(w,specmax,'b_cosmos')
		rmag,rlum=threed_dutils.integrate_mag(w,specmax,'r_cosmos')
		kmag,klum=threed_dutils.integrate_mag(w,specmax,'k_cosmos')

		bmag_sun = 5.47
		kmag_sun = 3.33

		ab_to_vega_b = 0.09
		ab_to_vega_r = -0.21
		ab_to_vega_k = -1.9

		ml_b[jj] = (thetas[0]+thetas[1]) / (10**((bmag_sun-(bmag+ab_to_vega_b))/2.5))
		ml_k[jj] = (thetas[0]+thetas[1]) / (10**((kmag_sun-(kmag+ab_to_vega_k))/2.5))
		br_color[jj] = (bmag+ab_to_vega_b) - (rmag+ab_to_vega_r)
	
	plt.plot(br_color,np.log10(ml_b),'bo',alpha=0.5)
	plt.ylim(-0.4,1.4)
	plt.xlim(0.0,2.22)
	plt.savefig(outname,dpi=300)
	plt.close()
	plt.plot(br_color,np.log10(ml_b),'bo',alpha=0.5)
	plt.plot(br_color,np.log10(ml_k),'ko',alpha=0.5)
	plt.ylim(-0.7,0.8)
	plt.xlim(0.5,1.7)
	plt.savefig(outname_2,dpi=300)
	plt.close()
	print 1/0




def av_to_sfr(runname, scale=False):

	'''
	this was an attempt to see whether the two-tau model naturally 
	added extra extinction towards the tau component with the bulk 
	the star formation

	has many deprecated functions as of 6/5/15
	'''

	inname = os.getenv('APPS')+'/threedhst_bsfh/results/'+runname+'/'+runname+'_ensemble.pickle'
	outname_cent = os.getenv('APPS') + '/threedhst_bsfh/plots/ensemble_plots/'+runname+'/av_to_sfr_comp.png'

	# if the save file doesn't exist, make it
	if not os.path.isfile(inname):
		collate_output(runname,inname)

	with open(inname, "rb") as f:
		ensemble=pickle.load(f)

	# pull out info
	valid_comp = ensemble['z'] > 0
	mass1 = ensemble['q50'][ensemble['parname'] == 'mass_1'][0,valid_comp]
	mass2 = ensemble['q50'][ensemble['parname'] == 'mass_2'][0,valid_comp]

	tau1 = ensemble['q50'][ensemble['parname'] == 'tau_1'][0,valid_comp]
	tau2 = ensemble['q50'][ensemble['parname'] == 'tau_2'][0,valid_comp]

	sfstart1 = ensemble['q50'][ensemble['parname'] == 'sf_start_1'][0,valid_comp]
	sfstart2 = ensemble['q50'][ensemble['parname'] == 'sf_start_2'][0,valid_comp]

	dust2_1 = ensemble['q50'][ensemble['parname'] == 'dust2_1'][0,valid_comp]
	dust2_2 = ensemble['q50'][ensemble['parname'] == 'dust2_2'][0,valid_comp]

	dustindex = ensemble['q50'][ensemble['parname'] == 'dust_index'][0,valid_comp]
	tage = WMAP9.age(ensemble['z'][valid_comp]).value

	# calculate SFRs
	deltat = 0.1
	sfr_100_1 = threed_dutils.integrate_sfh(tage-deltat,tage,mass1,tage,tau1,
                                            sfstart1)*mass1/(deltat*1e9)
	sfr_100_2 = threed_dutils.integrate_sfh(tage-deltat,tage,mass2,tage,tau2,
                                            sfstart2)*mass2/(deltat*1e9)

	sfr_100_1 = np.log10(sfr_100_1)
	sfr_100_2 = np.log10(sfr_100_2)

	# sort
	big_dust    = dust2_1 > dust2_2
	little_dust = dust2_1 < dust2_2

	# plot
	fig = plt.figure()
	ax = fig.add_subplot(111)
	axmin = -3
	axmax = 3
	vmin  = -3.0
	vmax  = -0.4
	jcmap = cm.seismic

	cobar=ax.scatter(np.clip(dust2_1[big_dust] - dust2_2[big_dust],axmin,axmax), 
		    np.clip(sfr_100_2[big_dust] - sfr_100_1[big_dust],axmin,axmax), 
		    alpha=0.8, c=dustindex[big_dust],cmap=jcmap,vmin=vmin,vmax=vmax,s=50.0)

	ax.scatter(np.clip(dust2_2[little_dust] - dust2_1[little_dust],axmin,axmax), 
		    np.clip(sfr_100_1[little_dust] - sfr_100_2[little_dust],axmin,axmax), 
		    alpha=0.8, c=dustindex[little_dust],cmap=jcmap,vmin=vmin,vmax=vmax,s=50.0)

	cbar = fig.colorbar(cobar, pad=0.1)
	cbar.set_label('dust index')

	ax.axis((0,axmax,axmin,axmax))
	ax.set_xlabel(r'dust2$_1$-dust2$_2$')
	ax.set_ylabel(r'log(SFR$_1$)-log(SFR$_2$)')

	ax.hlines(0.0,ax.get_xlim()[0],ax.get_xlim()[1], linestyle='--',colors='k')


	plt.savefig(outname_cent,dpi=300)
	plt.close()


def dynsamp_plot(runname, scale=False):

	inname = os.getenv('APPS')+'/threedhst_bsfh/results/'+runname+'/'+runname+'_ensemble.pickle'
	outname_cent = os.getenv('APPS') + '/threedhst_bsfh/plots/ensemble_plots/'+runname+'/dyn_comp.png'

	# if the save file doesn't exist, make it
	if not os.path.isfile(inname):
		collate_output(runname,inname)

	with open(inname, "rb") as f:
		ensemble=pickle.load(f)

	# identify rachel's stellar masses, uvj flag
	r_smass= np.array([x['logM'][0] for x in ensemble['ancildat']])
	uvj    = np.array([x['uvj_flag'][0] for x in ensemble['ancildat']])
	red  = uvj == 3
	blue = uvj != 3

	# get structural parameters (km/s, kpc)
	sigmaRe = np.array([x['sigmaRe'][0] for x in ensemble['ancildat']])
	e_sigmaRe = np.array([x['e_sigmaRe'][0] for x in ensemble['ancildat']])
	Re      = np.array([x['Re'][0] for x in ensemble['ancildat']])*1e3
	nserc   = np.array([x['n'][0] for x in ensemble['ancildat']])
	G      = 4.302e-3 # pc Msun**-1 (km/s)**2

	# dynamical masses
	# bezanson 2014, eqn 13+14
	k              = 5.0
	mdyn_cnst      = k*Re*sigmaRe**2/G
	mdyn_cnst_err  = 2*k*Re*sigmaRe*e_sigmaRe/G
	mdyn_cnst_lerr = asym_errors(np.log10(mdyn_cnst),np.log10(mdyn_cnst+mdyn_cnst_err)-np.log10(mdyn_cnst),np.log10(mdyn_cnst)-np.log10(mdyn_cnst-mdyn_cnst_err))
	
	k              = 8.87 - 0.831*nserc + 0.0241*nserc**2
	mdyn_serc      = k*Re*sigmaRe**2/G
	mdyn_serc_err  = 2*k*Re*sigmaRe*e_sigmaRe/G
	mdyn_serc_lerr = asym_errors(np.log10(mdyn_serc),np.log10(mdyn_serc+mdyn_serc_err)-np.log10(mdyn_serc),np.log10(mdyn_serc)-np.log10(mdyn_serc-mdyn_serc_err))

	#mdyn_serc = mdyn_cnst
	#mdyn_serc_letter = mdyn_cnst_lerr
	
	# Prospector stellar masses
	valid_comp = ensemble['z'] > 0
	mass = np.log10(ensemble['q50'][ensemble['parname'] == 'totmass'][0,valid_comp])
	masserrs = asym_errors(ensemble['q50'][ensemble['parname'] == 'totmass'][0,valid_comp],
		                   ensemble['q84'][ensemble['parname'] == 'totmass'][0,valid_comp], 
		                   ensemble['q16'][ensemble['parname'] == 'totmass'][0,valid_comp],log=True)

	gs = gridspec.GridSpec(1,2)
	fig = plt.figure(figsize = (9,4.5))
	gs.update(wspace=0.0, hspace=0.35)

	##### rachel masses #####
	ax_rachel = plt.subplot(gs[0])
	ax_rachel.errorbar(r_smass[blue],np.log10(mdyn_serc[blue]), 
					   yerr=[mdyn_serc_lerr[0][blue],mdyn_serc_lerr[1][blue]],
		               fmt='bo', linestyle=' ', alpha=0.7)
	ax_rachel.errorbar(r_smass[red],np.log10(mdyn_serc[red]), 
					   yerr=[mdyn_serc_lerr[0][red],mdyn_serc_lerr[1][red]],
		               fmt='ro', linestyle=' ', alpha=0.7)

	# equality line + axis limits
	ax_rachel.errorbar([9,13],[9,13],linestyle='--',color='0.1',alpha=0.8)
	ax_rachel.axis((9.7,11.7,9.7,11.7))

	# labels
	ax_rachel.set_ylabel(r'log(M$_{dyn}$/M$_{\odot}$)')
	ax_rachel.set_xlabel(r'log(M$_{FAST}$/M$_{\odot}$)')

	# offset and scatter
	mean_offset,scat = offset_and_scatter(np.log10(mdyn_serc[red]),r_smass[red])
	ax_rachel.text(0.96,0.20, 'scatter='+"{:.2f}".format(scat)+' dex',transform = ax_rachel.transAxes,color='red',horizontalalignment='right')
	ax_rachel.text(0.96,0.15, 'mean offset='+"{:.2f}".format(mean_offset)+' dex',transform = ax_rachel.transAxes,color='red',horizontalalignment='right')

	mean_offset,scat = offset_and_scatter(np.log10(mdyn_serc[blue]),r_smass[blue])
	ax_rachel.text(0.96,0.10, 'scatter='+"{:.2f}".format(scat)+' dex',transform = ax_rachel.transAxes,color='blue',horizontalalignment='right')
	ax_rachel.text(0.96,0.05, 'mean offset='+"{:.2f}".format(mean_offset)+' dex',transform = ax_rachel.transAxes,color='blue',horizontalalignment='right')

	#### prospectr masses ####
	ax_prosp = plt.subplot(gs[1])
	ax_prosp.errorbar(mass[blue],np.log10(mdyn_serc[blue]),
					  yerr=[masserrs[0][blue],masserrs[1][blue]],
		              fmt='bo', linestyle=' ', alpha=0.7)
	ax_prosp.errorbar(mass[red],np.log10(mdyn_serc[red]), 
					  yerr=[masserrs[0][red],masserrs[1][red]],
		              fmt='ro', linestyle=' ', alpha=0.7)
	
	# equality line + axis limits
	ax_prosp.errorbar([9,13],[9,13],linestyle='--',color='0.1',alpha=0.8)
	ax_prosp.axis((9.7,11.7,9.7,11.7))

	# labels
	ax_prosp.set_yticklabels([])
	ax_prosp.set_xlabel(r'log(M$_{prosp}$/M$_{\odot}$)')

	# offset and scatter
	mean_offset,scat = offset_and_scatter(np.log10(mdyn_serc[red]),mass[red])
	ax_prosp.text(0.96,0.20, 'scatter='+"{:.2f}".format(scat)+' dex',transform = ax_prosp.transAxes,color='red',horizontalalignment='right')
	ax_prosp.text(0.96,0.15, 'mean offset='+"{:.2f}".format(mean_offset)+' dex',transform = ax_prosp.transAxes,color='red',horizontalalignment='right')

	mean_offset,scat = offset_and_scatter(np.log10(mdyn_serc[blue]),mass[blue])
	ax_prosp.text(0.96,0.10, 'scatter='+"{:.2f}".format(scat)+' dex',transform = ax_prosp.transAxes,color='blue',horizontalalignment='right')
	ax_prosp.text(0.96,0.05, 'mean offset='+"{:.2f}".format(mean_offset)+' dex',transform = ax_prosp.transAxes,color='blue',horizontalalignment='right')

	plt.savefig(outname_cent,dpi=300)

def photerr_plot(runname, scale=False):

	inname = os.getenv('APPS')+'/threedhst_bsfh/results/'+runname+'/'+runname+'_ensemble.pickle'
	outname_errs = os.getenv('APPS') + '/threedhst_bsfh/plots/ensemble_plots/'+runname+'/photerr'
	outname_cent = os.getenv('APPS') + '/threedhst_bsfh/plots/ensemble_plots/'+runname+'/central_values'

	# if the save file doesn't exist, make it
	if not os.path.isfile(inname):
		collate_output(runname,inname)

	with open(inname, "rb") as f:
		ensemble=pickle.load(f)

	# number of parameters
	nparam = len(ensemble['parname'])

	# pull out photometric errors
	photerrs = np.array([float(x.split('_')[-2]) for x in ensemble['outname']])

	# initialize colors
	NUM_COLORS = nparam
	cm = pylab.get_cmap('gist_ncar')
	plt.rcParams['axes.color_cycle'] = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)] 
	axlim=[3.0,5.5,3.0,8] 

	# initialize plot
	fig = plt.figure()
	ax = fig.add_subplot(111)
	x_data = np.log10(photerrs*100)

	for jj in xrange(nparam):
		
		y_data = np.abs(ensemble['q50'][jj,:] / (ensemble['q84'][jj,:]-ensemble['q16'][jj,:]))[::-1]
		#y_data = y_data*(1.0/y_data[0])
		y_data = np.log10(y_data)

		ax[kk].plot(x_data, y_data[np.isnan(y_data) == 0], 'o', linestyle='-', alpha=0.7, label = ensemble['parname'][jj])

		ax[kk].set_ylabel('log(relative parameter error)')
		ax[kk].set_xlabel('log(photometric error) [%]')

		dynx = (np.nanmax(x_data)-np.nanmin(x_data))*0.05

		ax[kk].set_xlim([np.nanmin(x_data)-dynx,np.nanmax(x_data)+dynx])
		if scale:
			ax[kk].set_ylim(-2,2)

	ax[kk].legend(loc=0,prop={'size':6},
			  frameon=False)

	if scale:
		outname_errs = outname_errs+'_scale'
	plt.savefig(outname_errs+'.png', dpi=300)
	plt.close()

	# next plot
	gs = gridspec.GridSpec(3,nparam/3+1)
	fig = plt.figure(figsize = (16.5,9))
	gs.update(wspace=0.35, hspace=0.35)
	fs = 11
	for jj in xrange(nparam):

		is_data = np.isnan(ensemble['q50'][jj,:]) == 0
		y_data = np.abs(ensemble['q50'][jj,is_data])[::-1]
		
		y_err = asym_errors(ensemble['q50'][jj,is_data],ensemble['q84'][jj,is_data],ensemble['q16'][jj,is_data])
		ax = plt.subplot(gs[jj])

		ax[kk].errorbar(x_data,y_data, 
			        fmt='bo', ecolor='0.20', alpha=0.8,
			        yerr=y_err,linestyle='-')
		for tick in ax[kk].xaxis.get_major_ticks(): tick.label.set_fontsize(fs) 
		for tick in ax[kk].yaxis.get_major_ticks(): tick.label.set_fontsize(fs) 
		ax[kk].set_ylabel(ensemble['parname'][jj],fontsize=fs)
		ax[kk].set_xlabel('log(photometric error) [%]',fontsize=fs)

	ax[kk].legend(loc=0,prop={'size':6},
			  frameon=False)

	plt.savefig(outname_cent+'.png', dpi=300)
	plt.close()


def lir_comp(runname, lum=True, mjy=False):

	outname = os.getenv('APPS')+'/threedhst_bsfh/results/'+runname+'/'+runname+'_ensemble.pickle'
	outname_errs = os.getenv('APPS') + '/threedhst_bsfh/plots/ensemble_plots/'+runname+'/lir_comp'

	# if the save file doesn't exist, make it
	if not os.path.isfile(outname):
		collate_output(runname,outname)

	with open(outname, "rb") as f:
		ensemble=pickle.load(f)
	
	# get observed quantities from ancillary data
	# f_24: AB = 25
	f_24m = np.array([x['f24tot'][0] for x in ensemble['ancildat']])
	lir   = np.array([x['L_IR'][0] for x in ensemble['ancildat']])
	z_sfr = np.array([x['z_sfr'][0] for x in ensemble['ancildat']])
	valid_comp = ensemble['z'] > 0
	ensemble['z'] = ensemble['z'][valid_comp]

	# luminosity rather than flux
	if lum:
		distances = WMAP9.luminosity_distance(ensemble['z']).value*1e6*pc2cm
		dfactor = (4*np.pi*distances**2)/(1+ensemble['z'])
		ensemble['mips_flux'][:,valid_comp] = ensemble['mips_flux'][:,valid_comp] * dfactor
		f_24m = f_24m * dfactor

	# plot millijanskies
	if mjy:
		# put in maggies, then multiply by maggies to janskies, then to mjy
		jfactor = (1/1e10)*3631*1e3
		f_24m = f_24m *jfactor
		ensemble['mips_flux'][:,valid_comp] = ensemble['mips_flux'][:,valid_comp] * jfactor


	# create figure
	fig = plt.figure()
	ax = fig.add_subplot(111)
	
	# calculate L_IR, mips flux from model
	mips_err = asym_errors(ensemble['mips_flux'][1,:],ensemble['mips_flux'][2,:],ensemble['mips_flux'][0,:],log=True)
	L_IR_err = asym_errors(ensemble['L_IR'][1,:],ensemble['L_IR'][2,:],ensemble['L_IR'][0,:],log=True)

	# plot
	ax.errorbar(np.log10(ensemble['mips_flux'][1,:]),np.log10(ensemble['L_IR'][1,:]),
			 yerr=L_IR_err, xerr=mips_err,
			 fmt='bo',alpha=0.5)
	ax.errorbar(np.log10(f_24m),np.log10(lir),
		        fmt='ro',alpha=0.5)
	
	#testable = f_24m > 0
	#offset = np.mean(np.log10(ensemble['L_IR'][1,valid_comp])-np.log10(ensemble['mips_flux'][1,valid_comp]))
	#offset_2 = np.mean(np.log10(lir[testable])-np.log10(f_24m[testable]))

	ax.set_xlabel('MIPS flux density')
	ax.set_ylabel("L(IR) [8-1000$\mu$m]")

	plt.savefig(outname_errs+'.png', dpi=300)
	plt.close()

def emline_comparison(runname,emline_base='Halpha', chain_emlines='emp_ha'):
	inname = os.getenv('APPS')+'/threedhst_bsfh/results/'+runname+'/'+runname+'_ensemble.pickle'
	outname_errs = os.getenv('APPS') + '/threedhst_bsfh/plots/ensemble_plots/'+runname+'/emline_comp_kscalc_'+emline_base+'_'

	# will need to change to look at other emlines
	halpha_lam = 6563.0

	# if the save file doesn't exist, make it
	if not os.path.isfile(inname):
		collate_output(runname,inname)

	with open(inname, "rb") as f:
		ensemble=pickle.load(f)

	if emline_base == 'Halpha':
		obsname='Ha' # name of emission line in ancildat
		axislabel=r'H$\alpha$' # plot axis name
		cloudyname='Halpha' # name in CLOUDY data

	if emline_base == '[OIII]':
		obsname='OIII' # name of emission line in ancildat
		axislabel='[OIII]' # plot axis name
		cloudyname='[OIII]1' # name in CLOUDY data

	try:
		mass=ensemble['q50'][ensemble['parname'] == 'mass'][0]
	except IndexError:
		mass = ensemble['q50'][ensemble['parname'] == 'totmass'][0]

	# make plots for all three timescales
	sfr_str = ['10','100','1000']
	for bobo in xrange(len(sfr_str)):

		# CALCULATE EXPECTED HALPHA FLUX FROM SFH + KENNICUTT
		valid_comp = ensemble['z'] > 0
		distances = WMAP9.luminosity_distance(ensemble['z'][valid_comp]).value*1e6*pc2cm
		dfactor = (4*np.pi*distances**2)*1e-17

		# pull kennicutt halpha + errors out of chain
		index = ensemble['parname'] == chain_emlines
		f_emline = ensemble['q50'][index][0,valid_comp]/dfactor
		fluxup = ensemble['q84'][index][0,valid_comp]/dfactor
		fluxdo = ensemble['q16'][index][0,valid_comp]/dfactor
		f_err_emline = asym_errors(f_emline,fluxup,fluxdo,log=True)

		# EXTRACT EMISSION LINE FLUX FROM CLOUDY
		# include conversion from Lsun to cgs in dfactor
		dfactor = dfactor/constants.L_sun.cgs.value
		emline_q50 = np.array([x['q50'][x['name'] == cloudyname][0] for x in ensemble['model_emline']])/dfactor
		emline_q16 = np.array([x['q16'][x['name'] == cloudyname][0] for x in ensemble['model_emline']])/dfactor
		emline_q84 = np.array([x['q84'][x['name'] == cloudyname][0] for x in ensemble['model_emline']])/dfactor
		emline_errs = asym_errors(emline_q50,emline_q84,emline_q16,log=True)

		# EXTRACT OBSERVED EMISSION LINE FLUX
		# flux: 10**-17 ergs / s / cm**2
		# calculate scale
		obs_emline_lams = np.log(halpha_lam*(1+ensemble['z'][valid_comp])/1.e4)
		s0 = np.array([x['s0'][0] for x in ensemble['ancildat']])
		s1 = np.array([x['s1'][0] for x in ensemble['ancildat']])
		tilts = np.exp(s0 + s1*obs_emline_lams)

		# extract obs emlines
		obs_emline = np.array([x[obsname+'_flux'][0] for x in ensemble['ancildat']])*tilts
		obs_emline_err = np.array([x[obsname+'_error'][0] for x in ensemble['ancildat']])*tilts
		obs_emline_lerr = asym_errors(obs_emline,obs_emline+obs_emline_err,obs_emline-obs_emline_err,log=True)

		# extract stellar metallicity
		logzsol=ensemble['q50'][ensemble['parname'] == 'logzsol'][0,valid_comp]

		# SET UP PLOTTING QUANTITIES
		x_data = [np.log10(obs_emline),
				  np.log10(obs_emline),
				  np.log10(np.clip(emline_q50,minmodel_flux,1e50))
				  ]

		x_err =[obs_emline_lerr,
				obs_emline_lerr,
		        emline_errs]

		x_labels = [r'log('+axislabel+' flux) [observed]',
					r'log('+axislabel+' flux) [observed]',
					r'log('+axislabel+' flux) [cloudy]',
					]

		y_data = [np.log10(np.clip(emline_q50,minmodel_flux,1e50)),
				  np.log10(f_emline),
				  np.log10(f_emline)
				  ]

		y_err = [emline_errs,
		         None,
		         None]

		# if we have chain values
		try:
			y_err[1:] = f_err_emline
		except:
			pass

		y_labels = [r'log('+axislabel+' flux) [cloudy]',
					r'log('+axislabel+' flux) [ks]',
					r'log('+axislabel+' flux) [ks]',
					]

		fig, ax = plt.subplots(1, 3, figsize = (18, 5))
		for kk in xrange(len(x_data)):
			ax[kk].errorbar(x_data[kk],y_data[kk], 
				        fmt='bo', ecolor='0.20', alpha=0.8,
				        linestyle=' ',
				        yerr=y_err[kk],
				        xerr=x_err[kk])

			ax[kk].set_xlabel(x_labels[kk])
			ax[kk].set_ylabel(y_labels[kk])

			ax[kk].axis((-1,2.7,-1,2.7))
			ax[kk].plot([-1e3,1e3],[-1e3,1e3],linestyle='--',color='0.1',alpha=0.8)

			n = np.sum(np.isfinite(y_data[kk]))
			mean_offset = np.sum(y_data[kk]-x_data[kk])/n
			scat=np.sqrt(np.sum((y_data[kk]-x_data[kk]-mean_offset)**2.)/(n-2))
			ax[kk].text(-0.75,2.5, 'scatter='+"{:.2f}".format(scat)+' dex')
			ax[kk].text(-0.75,2.3, 'mean offset='+"{:.2f}".format(mean_offset)+' dex')

		fig.subplots_adjust(wspace=0.30,hspace=0.0)
		plt.savefig(outname_errs+'sfr'+sfr_str[bobo]+'.png',dpi=300)
		plt.close()

	# residuals with stellar metallicity
	fig_resid, ax_resid = plt.subplots(1, 3, figsize = (18, 5))
	for kk in xrange(len(x_data)):
		ax_resid[kk].plot(logzsol, x_data[kk]-y_data[kk],
					              'bo', alpha=0.8, linestyle=' ')
		ax_resid[kk].set_xlabel('logzsol')
		ax_resid[kk].set_ylabel(x_labels[kk]+'-'+y_labels[kk])
		ax_resid[kk].axis((-1,0.19,-1,1))
		ax_resid[kk].hlines(0.0,ax_resid[kk].get_xlim()[0],ax_resid[kk].get_xlim()[1], linestyle='--',colors='k')
	fig.subplots_adjust(wspace=0.30,hspace=0.0)
	plt.savefig(outname_errs+'logzsol_resid.png', dpi=300)
	plt.close()

def vary_logzsol(runname):

	filebase,params,ancilname=threed_dutils.generate_basenames(runname)
	ngals = len(filebase)

	nfail = 0

	# make output folder
	outdir = os.getenv('APPS')+'/threedhst_bsfh/plots/ensemble_plots/'+runname+'/mettests/'
	if not os.path.exists(outdir):
		os.makedirs(outdir)

	for jj in xrange(ngals):

		# find most recent output file
		# with the objname
		folder = "/".join(filebase[jj].split('/')[:-1])
		filename = filebase[jj].split("/")[-1]
		files = [f for f in os.listdir(folder) if "_".join(f.split('_')[:-2]) == filename]	
		times = [f.split('_')[-2] for f in files]

		# if we found no files, skip this object
		if len(times) == 0:
			print 'Failed to find any files in '+folder+' of type ' +filename+' to extract times'
			nfail+=1
			continue

		# load results
		mcmc_filename=filebase[jj]+'_'+max(times)+"_mcmc"
		model_filename=filebase[jj]+'_'+max(times)+"_model"

		try:
			sample_results, powell_results, model = read_results.read_pickles(mcmc_filename, model_file=model_filename,inmod=None)
		except (ValueError,EOFError,KeyError):
			print mcmc_filename + ' failed during output writing'
			nfail+=1
			continue
		except IOError:
			print mcmc_filename + ' does not exist!'
			nfail+=1
			continue

		# check for existence of extra information
		try:
			sample_results['quantiles']
		except:
			print 'Generating extra information for '+mcmc_filename+', '+model_filename
			extra_output.post_processing(params[jj])
			sample_results, powell_results, model = read_results.read_pickles(mcmc_filename, model_file=model_filename,inmod=None)

		# check to see if we want zcontinuous=2 (i.e., the MDF)
		if np.sum([1 for x in sample_results['model'].config_list if x['name'] == 'pmetals']) > 0:
			sps = threed_dutils.setup_sps(zcontinuous=2)
			print 'using the MDF'
		else:
			sps = threed_dutils.setup_sps(zcontinuous=2)
			print 'using the MDF'

		# generate figure
		fig, ax = plt.subplots(1, 1, figsize = (6, 6))
		thetas = sample_results['quantiles']['maxprob_params']
		theta_names = np.array(model.theta_labels())
		metind = theta_names == 'logzsol'

		# label best-fit params
		for i in xrange(len(theta_names)):
			if theta_names[i] == 'logzsol':
				weight = 'bold'
			else:
				weight = 'normal'
			ax.text(0.04,0.3-0.03*i, theta_names[i]+'='+"{:.2g}".format(thetas[i]),
				    transform = ax.transAxes,fontsize=8,
				    weight=weight)

		# grab and plot original spectrum
		specmax,magsmax,w = sample_results['model'].mean_model(thetas, sample_results['obs'], sps=sps,norm_spec=True)
		
		# nfnu or flam?
		#spec *= 3e18/w**2
		specmax *= (3e18)/w
		ax.plot(np.log10(w), np.log10(specmax), color='black', zorder=0)

		# logzsol dummy array
		logzsol = np.array([-1.0,-0.6,-0.2,0.0,0.19])

		# generate colors
		npoints = len(logzsol)
		cm = pylab.get_cmap('cool')
		plt.rcParams['axes.color_cycle'] = [cm(1.*i/npoints) for i in range(npoints)] 
		for kk in xrange(len(logzsol)):
			thetas[metind] = logzsol[kk]
			spec,mags,w = sample_results['model'].mean_model(thetas, sample_results['obs'], sps=sps,norm_spec=True)
			
			# nfnu or flam?
			#spec *= 3e18/w**2
			spec *= (3e18)/w

			ax.plot(np.log10(w), np.log10(spec),
					label = logzsol[kk],
					zorder=-32)

		# plot limits
		xlim = (3.3,4.8)
		ax.set_xlim(xlim)
		good = (np.array(w) > 10**xlim[0]) & (np.array(w) < 10**xlim[1])
		ax.set_ylim(np.log10(min(spec[good]))*0.96,np.log10(max(spec[good]))*1.04)

		# legend
		ax.legend(loc=1,prop={'size':10},
				  frameon=False,
				  title='logzsol')
		ax.get_legend().get_title().set_size(10)

		# plot obs + obs errs
		mask = sample_results['obs']['phot_mask']
		
		# restore irac
		good = np.logical_and((sample_results['obs']['maggies'] > -9.8e-9),
		                      (sample_results['obs']['maggies'] != sample_results['obs']['maggies_unc']))
		mask = good


		sample_results['obs']['phot_mask'][-5:-1] = True # restore IRAC
		obs_mags = sample_results['obs']['maggies'][mask]
		obs_lam  = sample_results['obs']['wave_effective'][mask]
		obs_err  = sample_results['obs']['maggies_unc'][mask]
		linerr_down = np.clip(obs_mags-obs_err, 1e-80, 1e80)
		linerr_up = np.clip(obs_mags+obs_err, 1e-80, 1e80)
		yerr = [np.log10(obs_mags) - np.log10(linerr_down),
		        np.log10(linerr_up)- np.log10(obs_mags)]

		# nfnu or flam?
		#spec *= 3e18/w**2
		pmags = obs_mags*(3e18)/obs_lam
		ax.errorbar(np.log10(obs_lam), np.log10(pmags), yerr=yerr,
					fmt='ok',ms=5)
		
		# save
		outname = sample_results['run_params']['objname']
		plt.savefig(outdir+outname+'_metsed.png', dpi=300)
		plt.close()

def plot_residuals_fixedmet(runname):

	'''
	special version for when metallicity is fixed
	'''
	runname = 'dtau_genpop_fixedmet'

	filebase,params,ancilname=threed_dutils.generate_basenames(runname)
	ngals = len(filebase)

	nfail = 0

	# make output folder
	outdir = os.getenv('APPS')+'/threedhst_bsfh/plots/ensemble_plots/'+runname+'/'
	if not os.path.exists(outdir):
		os.makedirs(outdir)

	for jj in xrange(ngals):

		# find most recent output file
		# with the objname
		folder = "/".join(filebase[jj].split('/')[:-1])
		filename = filebase[jj].split("/")[-1]
		files = [f for f in os.listdir(folder) if "_".join(f.split('_')[:-2]) == filename]	
		times = [f.split('_')[-2] for f in files]

		# if we found no files, skip this object
		if len(times) == 0:
			print 'Failed to find any files in '+folder+' of type ' +filename+' to extract times'
			nfail+=1
			continue

		# load results
		mcmc_filename=filebase[jj]+'_'+max(times)+"_mcmc"
		model_filename=filebase[jj]+'_'+max(times)+"_model"

		try:
			sample_results, powell_results, model = read_results.read_pickles(mcmc_filename, model_file=model_filename,inmod=None)
		except (ValueError,EOFError,KeyError):
			print mcmc_filename + ' failed during output writing'
			nfail+=1
			continue
		except IOError:
			print mcmc_filename + ' does not exist!'
			nfail+=1
			continue

		# check for existence of extra information
		try:
			sample_results['quantiles']
		except:
			print 'Generating extra information for '+mcmc_filename+', '+model_filename
			extra_output.post_processing(params[jj])
			sample_results, powell_results, model = read_results.read_pickles(mcmc_filename, model_file=model_filename,inmod=None)

		# check to see if we want zcontinuous=2 (i.e., the MDF)
		if np.sum([1 for x in sample_results['model'].config_list if x['name'] == 'pmetals']) > 0:
			sps = threed_dutils.setup_sps(zcontinuous=2)
			print 'using the MDF'
		else:
			sps = threed_dutils.setup_sps(zcontinuous=1)
			print 'using interpolated metallicities'

		# save metallicity, obj number
		met      = [x['init'] for x in sample_results['model'].config_list if x['name'] == 'logzsol'][0]
		obj_name = sample_results['run_params']['objname']

		# generate figure
		thetas = sample_results['quantiles']['maxprob_params']
		theta_names = np.array(model.theta_labels())
		metind = theta_names == 'logzsol'

		# grab best-fit magnitudes
		specmax,magsmax,w = sample_results['model'].mean_model(thetas, sample_results['obs'], sps=sps,norm_spec=True)

		# grab obs + obs errors
		# start with mask
		mask = sample_results['obs']['phot_mask']

		# define obs things
		obs_mags = sample_results['obs']['maggies'][mask]
		obs_lam  = sample_results['obs']['wave_effective'][mask]
		obs_err  = sample_results['obs']['maggies_unc'][mask]


		# residuals
		tempresid = (obs_mags - magsmax[mask]) / obs_err
		tempresid_perc = (obs_mags - magsmax[mask]) / obs_mags

		try:
			residuals[mask,jj]    = tempresid
			percentresid[mask,jj] = tempresid_perc
			restlam[mask,jj]      = np.log10(obs_lam/(1+sample_results['model'].params['zred']))
			bestmet[jj]           = met
			obj_names[jj]         = obj_name
			maxprob[jj]           = sample_results['quantiles']['maxprob']

		except:
			nfilters = len(sample_results['obs']['filters'])
			lam = np.log10(sample_results['obs']['wave_effective'])
			residuals    = np.zeros(shape=(nfilters,ngals)) + np.nan 
			restlam      = np.zeros(shape=(nfilters,ngals)) + np.nan
			percentresid = np.zeros(shape=(nfilters,ngals)) + np.nan
			bestmet      = np.zeros(ngals) + np.nan
			obj_names    = np.zeros(ngals) + np.nan
			maxprob      = np.zeros(ngals) + np.nan

			residuals[mask,jj]    = tempresid
			percentresid[mask,jj] = tempresid_perc
			restlam[mask,jj]      = np.log10(obs_lam/(1+sample_results['model'].params['zred']))
			bestmet[jj]           = met
			obj_names[jj]         = obj_name
			maxprob[jj]           = sample_results['quantiles']['maxprob']

		print bestmet[jj],obj_names[jj], maxprob[jj]

	# how many objects?
	objs = np.unique(obj_names)
	objs = objs[np.isfinite(objs)]

	for objname in objs:

		# which are of this object?
		good = obj_names == objname
		ngood = np.sum(good)

		# lnprob plot
		fig = plt.figure()
		ax = fig.add_subplot(111)

		ax.plot(bestmet[good],maxprob[good],'ro', linestyle='-',alpha=0.6)
		ax.set_xlabel('logzsol')
		ax.set_ylabel('max ln(probability)')

		plt.savefig(outdir+'maxprob_vs_met_'+str(int(objname))+'.png', dpi=300)
		plt.close()

		# residual plot
		fig = plt.figure()
		ax = fig.add_subplot(111)

		# initialize colors
		NUM_COLORS = ngood
		NUM_COLORS = 4
		cm = pylab.get_cmap('cool')
		plt.rcParams['axes.color_cycle'] = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)] 

		xlam    = restlam[:,good]
		yval    = residuals[:,good]

		met_to_plot = ['-1.0','-0.9','-0.6','0.0','0.19']
		for kk in xrange(ngood): 
			if str(bestmet[good][kk]) in met_to_plot:
				ax.plot(xlam[:,kk],yval[:,kk],lw=1,label=bestmet[good][kk],alpha=0.9)

		ax.set_ylabel('(obs-model)/err')
		ax.set_xlabel(r'log($\lambda_{rest}$) [$\AA$]')
		ax.set_ylim(-6.0,6.0)
		ax.set_xlim(3.0,5.5)
		ax.legend(loc=1,prop={'size':10},
                  frameon=False,
                  title='logzsol')
		ax.hlines(0.0,ax.get_xlim()[0],ax.get_xlim()[1], linestyle='--',colors='k')
		plt.savefig(outdir+'rf_resid_'+str(int(objname))+'.png', dpi=300)
		plt.close()

def plot_residuals(runname):

	filebase,params,ancilname=threed_dutils.generate_basenames(runname)
	ngals = len(filebase)

	nfail = 0

	# make output folder
	outdir = os.getenv('APPS')+'/threedhst_bsfh/plots/ensemble_plots/'+runname+'/mettests/'
	if not os.path.exists(outdir):
		os.makedirs(outdir)

	for jj in xrange(ngals):

		# find most recent output file
		# with the objname
		folder = "/".join(filebase[jj].split('/')[:-1])
		filename = filebase[jj].split("/")[-1]
		files = [f for f in os.listdir(folder) if "_".join(f.split('_')[:-2]) == filename]	
		times = [f.split('_')[-2] for f in files]

		# if we found no files, skip this object
		if len(times) == 0:
			print 'Failed to find any files in '+folder+' of type ' +filename+' to extract times'
			nfail+=1
			continue

		# load results
		mcmc_filename=filebase[jj]+'_'+max(times)+"_mcmc"
		model_filename=filebase[jj]+'_'+max(times)+"_model"

		try:
			sample_results, powell_results, model = read_results.read_pickles(mcmc_filename, model_file=model_filename,inmod=None)
		except (ValueError,EOFError,KeyError):
			print mcmc_filename + ' failed during output writing'
			nfail+=1
			continue
		except IOError:
			print mcmc_filename + ' does not exist!'
			nfail+=1
			continue

		# check for existence of extra information
		try:
			sample_results['quantiles']
		except:
			print 'Generating extra information for '+mcmc_filename+', '+model_filename
			extra_output.post_processing(params[jj])
			sample_results, powell_results, model = read_results.read_pickles(mcmc_filename, model_file=model_filename,inmod=None)

		# check to see if we want zcontinuous=2 (i.e., the MDF)
		if np.sum([1 for x in sample_results['model'].config_list if x['name'] == 'pmetals']) > 0:
			sps = threed_dutils.setup_sps(zcontinuous=2)
			print 'using the MDF'
		else:
			sps = threed_dutils.setup_sps(zcontinuous=1)
			print 'using interpolated metallicities'

		# where is metallicity
		met_ind = np.array(sample_results['model'].theta_labels()) == 'logzsol'
		ssfr_ind = np.array(sample_results['extras']['parnames']) == 'ssfr_100'

		# generate figure
		thetas = sample_results['quantiles']['maxprob_params']
		theta_names = np.array(model.theta_labels())
		metind = theta_names == 'logzsol'

		# grab best-fit magnitudes
		specmax,magsmax,w = sample_results['model'].mean_model(thetas, sample_results['obs'], sps=sps,norm_spec=True)

		# grab obs + obs errors
		# start with mask
		mask = sample_results['obs']['phot_mask']

		# define obs things
		obs_mags = sample_results['obs']['maggies'][mask]
		obs_lam  = sample_results['obs']['wave_effective'][mask]
		obs_err  = sample_results['obs']['maggies_unc'][mask]


		# residuals
		tempresid = (obs_mags - magsmax[mask]) / obs_err
		tempresid_perc = (obs_mags - magsmax[mask]) / obs_mags

		try:
			residuals[mask,jj]    = tempresid
			percentresid[mask,jj] = tempresid_perc
			restlam[mask,jj]      = np.log10(obs_lam/(1+sample_results['model'].params['zred']))
			bestmet[jj]           = thetas[met_ind]
			bestssfr[jj]          = sample_results['extras']['q50'][ssfr_ind]

		except:
			nfilters = len(sample_results['obs']['filters'])
			lam = np.log10(sample_results['obs']['wave_effective'])
			residuals    = np.zeros(shape=(nfilters,ngals)) + np.nan 
			restlam      = np.zeros(shape=(nfilters,ngals)) + np.nan
			percentresid = np.zeros(shape=(nfilters,ngals)) + np.nan
			bestmet      = np.zeros(ngals) + np.nan
			bestssfr     = np.zeros(ngals) + np.nan

			residuals[mask,jj]    = tempresid
			percentresid[mask,jj] = tempresid_perc
			restlam[mask,jj]      = np.log10(obs_lam/(1+sample_results['model'].params['zred']))
			bestmet[jj]           = thetas[met_ind]
			bestssfr[jj]          = sample_results['extras']['q50'][ssfr_ind]

		print bestmet[jj],bestssfr[jj]

	# take mean
	meanresid = np.nanmean(residuals,axis=1)
	meanpercent = np.nanmean(percentresid,axis=1)

	stdresid = np.nanstd(residuals,axis=1)
	stdpercent = np.nanstd(percentresid,axis=1)

	# plot normalized by errors
	fig = plt.figure()
	ax = fig.add_subplot(111)

	# initialize colors
	NUM_COLORS = nfilters
	cm = pylab.get_cmap('gist_rainbow')
	plt.rcParams['axes.color_cycle'] = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)] 

	for kk in xrange(len(lam)): ax.errorbar(lam[kk],meanresid[kk],yerr=stdresid[kk],fmt='o',color=cm(1.*kk/NUM_COLORS))
	for kk in xrange(len(lam)): ax.text(0.04,0.9-0.02*kk,sample_results['obs']['filters'][kk],color=cm(1.*kk/NUM_COLORS),transform = ax.transAxes, fontsize=7)

	ax.set_ylabel('(obs-model)/errors')
	ax.set_xlabel('observed wavelength')
	ax.set_ylim(-2.0,2.0)
	ax.set_xlim(3.0,5.5)
	ax.hlines(0.0,ax.get_xlim()[0],ax.get_xlim()[1], linestyle='--',colors='k')
	plt.savefig(outdir+'residuals_by_err.png', dpi=300)
	plt.close()

	# plot normalized by errors
	fig = plt.figure()
	ax = fig.add_subplot(111)

	# initialize colors
	NUM_COLORS = nfilters
	cm = pylab.get_cmap('gist_rainbow')
	plt.rcParams['axes.color_cycle'] = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)] 

	for kk in xrange(len(lam)): ax.errorbar(lam[kk],meanpercent[kk],yerr=stdpercent[kk],fmt='o',color=cm(1.*kk/NUM_COLORS))
	for kk in xrange(len(lam)): ax.text(0.04,0.9-0.02*kk,sample_results['obs']['filters'][kk],color=cm(1.*kk/NUM_COLORS),transform = ax.transAxes, fontsize=7)

	ax.set_ylabel('(obs-model)/obs')
	ax.set_xlabel('observed wavelength')
	ax.set_ylim(-1.5,1.5)
	ax.set_xlim(3.0,5.5)
	ax.hlines(0.0,ax.get_xlim()[0],ax.get_xlim()[1], linestyle='--',colors='k')

	plt.savefig(outdir+'residuals_by_obs.png', dpi=300)
	plt.close()

	# average by metallicity
	# take mean
	topmet = bestmet > 0.1
	ntopmet = np.sum(topmet)
	botmet = bestmet <= -0.6
	nbotmet = np.sum(botmet)
	meanresid_topmet = np.nanmean(residuals[:,topmet],axis=1)
	meanpercent_topmet = np.nanmean(percentresid[:,topmet],axis=1)

	stdresid_topmet = np.nanstd(residuals[:,topmet],axis=1)
	stdpercent_topmet = np.nanstd(percentresid[:,topmet],axis=1)

	meanresid_botmet = np.nanmean(residuals[:,botmet],axis=1)
	meanpercent_botmet = np.nanmean(percentresid[:,botmet],axis=1)

	stdresid_botmet = np.nanstd(residuals[:,botmet],axis=1)
	stdpercent_botmet = np.nanstd(percentresid[:,botmet],axis=1)

	# plot normalized by errors
	fig = plt.figure()
	ax = fig.add_subplot(111)

	ax.errorbar(lam,meanresid_topmet,fmt='-o',color='red')
	ax.errorbar(lam,meanresid_botmet,fmt='-o',color='blue')

	ax.set_ylabel('(obs-model)/errors')
	ax.set_xlabel('observed wavelength')
	ax.set_ylim(-7.0,7.0)
	ax.set_xlim(3.0,5.5)
	ax.hlines(0.0,ax.get_xlim()[0],ax.get_xlim()[1], linestyle='--',colors='k')
	plt.savefig(outdir+'residuals_by_err_metsplit.png', dpi=300)
	plt.close()

	# plot normalized by obs
	fig = plt.figure()
	ax = fig.add_subplot(111)

	xlam    = restlam.reshape(ngals*nfilters)
	yval    = percentresid.reshape(ngals*nfilters)

	good = np.isfinite(xlam) == True
	xlam = xlam[good]
	yval = yval[good]

	bins, median = threed_dutils.running_median(xlam,yval,nbins=20)

	ax.scatter(xlam,yval, marker='o', facecolor='black', alpha=0.2,lw=0,s=8.0)
	plt.plot(bins,median,color='red',lw=2)

	ax.set_ylabel('(obs-model)/obs')
	ax.set_xlabel('rest-frame wavelength')
	ax.set_ylim(-1.0,1.0)
	ax.set_xlim(3.0,5.5)
	ax.hlines(0.0,ax.get_xlim()[0],ax.get_xlim()[1], linestyle='--',colors='k')
	plt.savefig(outdir+'residuals_by_obs_restframe.png', dpi=300)
	plt.close()

	# average by metallicity, in rest-frame
	# plot normalized by errors
	fig = plt.figure()
	ax = fig.add_subplot(111)
	topmet_resid = residuals[:,topmet].reshape(ntopmet*nfilters)
	topmet_restlam = restlam[:,topmet].reshape(ntopmet*nfilters)

	nonan = np.isfinite(topmet_resid)
	topmet_resid = topmet_resid[nonan]
	topmet_restlam = topmet_restlam[nonan]

	plt.scatter(topmet_restlam, topmet_resid, marker='o', facecolor='red', alpha=0.2,lw=0,s=8.0)

	bins, median = threed_dutils.running_median(topmet_restlam,topmet_resid,nbins=20)
	plt.plot(bins,median,color='red',lw=5)

	botmet_resid = residuals[:,botmet].reshape(nbotmet*nfilters)
	botmet_restlam = restlam[:,botmet].reshape(nbotmet*nfilters)

	nonan = np.isfinite(botmet_resid)
	botmet_resid = botmet_resid[nonan]
	botmet_restlam = botmet_restlam[nonan]

	plt.scatter(botmet_restlam, botmet_resid, marker='o', facecolor='blue', alpha=0.2,lw=0,s=8.0)

	bins, median = threed_dutils.running_median(botmet_restlam,botmet_resid,nbins=20)
	plt.plot(bins,median,color='blue',lw=5)

	ax.set_ylabel('(obs-model)/errors')
	ax.set_xlabel('rest-frame wavelength')
	ax.set_ylim(-7.0,7.0)
	ax.set_xlim(3.0,5.5)
	ax.hlines(0.0,ax.get_xlim()[0],ax.get_xlim()[1], linestyle='--',colors='k')
	plt.savefig(outdir+'residuals_by_err_metsplit_rf.png', dpi=300)
	plt.close()

	# average by sSFR
	# take mean
	meanssfr = np.nanmean(bestssfr)
	topssfr = bestssfr > meanssfr
	ntopssfr = np.sum(topssfr)
	botssfr = bestssfr <= meanssfr
	nbotssfr = np.sum(botssfr)
	meanresid_topssfr = np.nanmean(residuals[:,topssfr],axis=1)
	meanpercent_topssfr = np.nanmean(percentresid[:,topssfr],axis=1)

	stdresid_topssfr = np.nanstd(residuals[:,topssfr],axis=1)
	stdpercent_topssfr = np.nanstd(percentresid[:,topssfr],axis=1)

	meanresid_botssfr = np.nanmean(residuals[:,botssfr],axis=1)
	meanpercent_botssfr = np.nanmean(percentresid[:,botssfr],axis=1)

	stdresid_botssfr = np.nanstd(residuals[:,botssfr],axis=1)
	stdpercent_botssfr = np.nanstd(percentresid[:,botssfr],axis=1)

	# average by sSFR, in rest-frame
	# plot normalized by errors
	fig = plt.figure()
	ax = fig.add_subplot(111)
	topssfr_resid = residuals[:,topssfr].reshape(ntopssfr*nfilters)
	topssfr_perc  = percentresid[:,topssfr].reshape(ntopssfr*nfilters)
	topssfr_restlam = restlam[:,topssfr].reshape(ntopssfr*nfilters)

	nonan = np.isfinite(topssfr_resid)
	topssfr_resid = topssfr_resid[nonan]
	topssfr_perc  = topssfr_perc[nonan]
	topssfr_restlam = topssfr_restlam[nonan]

	plt.scatter(topssfr_restlam, topssfr_resid, marker='o', facecolor='blue', alpha=0.2,lw=0,s=8.0)

	bins, median = threed_dutils.running_median(topssfr_restlam,topssfr_resid,nbins=20)
	plt.plot(bins,median,color='blue',lw=5)

	botssfr_resid = residuals[:,botssfr].reshape(nbotssfr*nfilters)
	botssfr_perc  = percentresid[:,botssfr].reshape(nbotssfr*nfilters)
	botssfr_restlam = restlam[:,botssfr].reshape(nbotssfr*nfilters)

	nonan = np.isfinite(botssfr_resid)
	botssfr_resid = botssfr_resid[nonan]
	botssfr_perc  = botssfr_perc[nonan]
	botssfr_restlam = botssfr_restlam[nonan]

	plt.scatter(botssfr_restlam, botssfr_resid, marker='o', facecolor='red', alpha=0.2,lw=0,s=8.0)

	bins, median = threed_dutils.running_median(botssfr_restlam,botssfr_resid,nbins=20)
	plt.plot(bins,median,color='red',lw=5)

	ax.set_ylabel('(obs-model)/errors')
	ax.set_xlabel('rest-frame wavelength')
	ax.set_ylim(-7.0,7.0)
	ax.set_xlim(3.0,5.5)
	ax.hlines(0.0,ax.get_xlim()[0],ax.get_xlim()[1], linestyle='--',colors='k')
	ax.text(0.04,0.95, 'sSFR>'+"{:.2e}".format(meanssfr),transform = ax.transAxes,color='blue')
	ax.text(0.04,0.90, 'sSFR<'+"{:.2e}".format(meanssfr),transform = ax.transAxes,color='red')
	plt.savefig(outdir+'residuals_by_err_ssfrsplit_rf.png', dpi=300)
	plt.close()

	# average by sSFR, in rest-frame
	# plot normalized by observations
	fig = plt.figure()
	ax = fig.add_subplot(111)

	plt.scatter(topssfr_restlam, topssfr_perc, marker='o', facecolor='blue', alpha=0.2,lw=0,s=8.0)

	bins, median = threed_dutils.running_median(topssfr_restlam,topssfr_perc,nbins=10)
	plt.plot(bins,median,color='blue',lw=5)

	plt.scatter(botssfr_restlam, botssfr_perc, marker='o', facecolor='red', alpha=0.2,lw=0,s=8.0)

	bins, median = threed_dutils.running_median(botssfr_restlam,botssfr_perc,nbins=10)
	plt.plot(bins,median,color='red',lw=5)

	ax.set_ylabel('(obs-model)/obs')
	ax.set_xlabel('rest-frame wavelength')
	ax.set_ylim(-1.0,1.0)
	ax.set_xlim(3.0,5.5)
	ax.hlines(0.0,ax.get_xlim()[0],ax.get_xlim()[1], linestyle='--',colors='k')
	ax.text(0.04,0.95, 'sSFR>'+"{:.2e}".format(meanssfr),transform = ax.transAxes,color='blue')
	ax.text(0.04,0.90, 'sSFR<'+"{:.2e}".format(meanssfr),transform = ax.transAxes,color='red')
	plt.savefig(outdir+'residuals_by_obs_ssfrsplit_rf.png', dpi=300)
	plt.close()

	print 1/0


	# RESTFRAME
	# plot normalized by errors
	fig, ax = plt.subplots(1, 1, figsize = (6, 6))

	# initialize colors
	NUM_COLORS = nfilters
	cm = pylab.get_cmap('gist_rainbow')
	plt.rcParams['axes.color_cycle'] = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)] 

	plt.plot(restframe,residuals_all,'bo',lw=0.6,alpha=0.5,ms=0.5)
	ax.hlines(0.0,ax.get_xlim()[0],ax.get_xlim()[1], linestyle='--',colors='k')

	bins,running_median = threed_dutils.running_median(restframe,residuals_all)
	running_median = np.array(running_median)
	bad = np.isfinite(running_median) == 0
	running_median[bad] = 0
	plt.plot(bins,running_median,'r--',lw=2,alpha=.8)

	ax.set_ylabel('(obs-model)/err')
	ax.set_xlabel('rest-frame wavelength')
	ax.set_ylim(-0.8,0.8)
	plt.savefig(outdir+'residuals_by_err_restframe.png', dpi=300)
	plt.close()
	print 1/0


def testsed_truthplots(runname):

	'''
	prototype for generating useful output balls
	currently saves q16, q50, q84, maxprob, and parmlist for each galaxy

	in the future, want it to include:
	maximum likelihood fit (both parameters and likelihood)
	mean acceptance fraction
	also output information about the powell minimization process
	pri[0].x = min([p.fun for p in pr])   <--- if this is true, then the minimization process chose the processor with initial conditions
	np.array([p.x for p in pr])
	'''

	filebase,params,ancilname=threed_dutils.generate_basenames(runname)
	ngals = len(filebase)
	output = '/Users/joel/code/python/threedhst_bsfh/plots/ensemble_plots/'+runname+'/truthplots.png'

	nfail = 0
	for jj in xrange(ngals):

		# find most recent output file
		# with the objname
		folder = "/".join(filebase[jj].split('/')[:-1])
		filename = filebase[jj].split("/")[-1]
		files = [f for f in os.listdir(folder) if "_".join(f.split('_')[:-2]) == filename]	
		times = [f.split('_')[-2] for f in files]

		# if we found no files, skip this object
		if len(times) == 0:
			print 'Failed to find any files in '+folder+' of type ' +filename+' to extract times'
			nfail+=1
			continue

		# load results
		mcmc_filename=filebase[jj]+'_'+max(times)+"_mcmc"
		model_filename=filebase[jj]+'_'+max(times)+"_model"

		try:
			sample_results, powell_results, model = read_results.read_pickles(mcmc_filename, model_file=model_filename,inmod=None)
		except (ValueError,EOFError,KeyError):
			print mcmc_filename + ' failed during output writing'
			nfail+=1
			continue
		except IOError:
			print mcmc_filename + ' does not exist!'
			nfail+=1
			continue

		# check for existence of extra information
		try:
			sample_results['quantiles']
		except:
			print 'Generating extra information for '+mcmc_filename+', '+model_filename
			extra_output.post_processing(params[jj])
			sample_results, powell_results, model = read_results.read_pickles(mcmc_filename, model_file=model_filename,inmod=None)
			
		# load truths
		truths = threed_dutils.load_truths(os.getenv('APPS')+'/threed'+sample_results['run_params']['truename'].split('/threed')[1],
		                                   sample_results['run_params']['objname'],
		                                   sample_results)
		
		# define outputs
		try:
			nextra
		except NameError:
			# output truths vector
			nextra=len(truths['extra_parnames'])
			parnames = np.append(truths['parnames'],truths['extra_parnames'])
			fulltruths = np.zeros(shape=(ngals,len(parnames))) + np.nan
			models = np.zeros(shape=(ngals,len(parnames),3)) + np.nan
			

			# find mass, sfr vectors
			massind = sample_results['extras']['parnames'] == 'totmass'
			sfrind = sample_results['extras']['parnames'] == 'sfr_100'

		#### calculate TRUE total mass, total sfr, all TRUE params #####
		fulltruths[jj,:] = np.append(truths['plot_truths'],truths['extra_truths'])

		#### calculate MODEL total mass, total sfr, all MODEL params #####
		models[jj,:-nextra,0] = sample_results['quantiles']['q16']
		models[jj,:-nextra,1] = sample_results['quantiles']['q50']
		models[jj,:-nextra,2] = sample_results['quantiles']['q84']

		models[jj,-nextra,0] = sample_results['extras']['q16'][massind]
		models[jj,-nextra,1] = sample_results['extras']['q50'][massind]
		models[jj,-nextra,2] = sample_results['extras']['q84'][massind]

		models[jj,-nextra+1,0] = sample_results['extras']['q16'][sfrind]
		models[jj,-nextra+1,1] = sample_results['extras']['q50'][sfrind]
		models[jj,-nextra+1,2] = sample_results['extras']['q84'][sfrind]
    

	# format for plotting
	for kk in xrange(len(parnames)):

		if parnames[kk] == 'sf_start' or parnames[kk][:-2] == 'sf_start':
			models[:,kk,:] = sample_results['model'].params['tage'][0]-models[:,kk,:]

		# log parameters
		if parnames[kk] == 'mass' or parnames[kk][:-2] == 'mass' or \
           parnames[kk] == 'tau' or parnames[kk][:-2] == 'tau' or \
           parnames[kk] == 'sf_start' or parnames[kk][:-2] == 'sf_start' or \
           parnames[kk] == 'totsfr' or parnames[kk] == 'totmass':

			models[:,kk,:] = np.log10(models[:,kk,:])
			parnames[kk] = 'log('+parnames[kk]+')'

	# find 3sigma deviations
	sigcut=3

	true_totmass = fulltruths[:,-2]
	model_totmass = models[:,-2,1]
	model_totmasserrs = asym_errors(models[:,-2,1],models[:,-2,2],models[:,-2,0])
	bad_totmass = np.logical_or( (true_totmass-model_totmass) > sigcut*model_totmasserrs[1] ,\
		                         (model_totmass-true_totmass) > sigcut*model_totmasserrs[0] )

	true_totsfr = fulltruths[:,-1]
	model_totsfr = models[:,-1,1]
	model_totsfrerrs = asym_errors(models[:,-1,1],models[:,-1,2],models[:,-1,0])
	bad_totsfr = np.logical_or( (true_totsfr-model_totsfr) > sigcut*model_totsfrerrs[1] ,\
		                         (model_totsfr-true_totsfr) > sigcut*model_totsfrerrs[0] )

    # set up plot
	gs = gridspec.GridSpec(4,3)
	fig = plt.figure(figsize = (9.8,12.6))
	gs.update(wspace=0.35, hspace=0.35)

	for kk in xrange(len(parnames)):
		ax = plt.subplot(gs[kk])
		
		x = fulltruths[:,kk]
		y = models[:,kk,1]
		errs = asym_errors(models[:,kk,1],models[:,kk,2],models[:,kk,0])

		# plot
		ax.errorbar(x,y, 
			        fmt='wo', linestyle=' ', alpha=0.5)
		ax.errorbar(x[bad_totmass],y[bad_totmass], 
			        fmt='ro', linestyle=' ', alpha=0.3)
		ax.errorbar(x[bad_totsfr],y[bad_totsfr], 
			        fmt='bo', linestyle=' ', alpha=0.3)
		ax.errorbar(x,y, 
			        fmt=' ', ecolor='0.75', alpha=0.5,
			        yerr=errs,linestyle=' ',
			        zorder=-32)

		# label
		ax.set_xlabel('true '+parnames[kk])
		ax.set_ylabel('fit '+parnames[kk])

		# axis limits
		dynx, dyny = (np.nanmax(x)-np.nanmin(x))*0.05,\
		             (np.nanmax(y)-np.nanmin(y))*0.05
		if np.nanmin(x)-dynx > np.nanmin(y)-dyny:
			pmin = np.nanmin(y)-dyny*3
		else:
			pmin = np.nanmin(x)-dynx*3
		if np.nanmax(x)+dynx > np.nanmax(y)+dyny:
			pmax = np.nanmax(x)+dynx*3
		else:
			pmax = np.nanmax(y)+dyny*3
		ax.axis((pmin,pmax,pmin,pmax))

		# add 1:1 line
		ax.errorbar(ax.get_xlim(),ax.get_ylim(),linestyle='--',color='0.1',alpha=0.8)

	plt.savefig(output,dpi=300)
	os.system('open '+output)


def recover_phot_jitter(runname):

	'''
	given a run, for each galaxy, load the output, and
	construct the sum of the PDF for the photometric error terms
	'''

	filebase,params,ancilname=threed_dutils.generate_basenames(runname)
	ngals = len(filebase)
	outname = '/Users/joel/code/python/threedhst_bsfh/plots/ensemble_plots/'+runname+'/phot_jitter_pdf.png'

	nfail = 0
	gs = gridspec.GridSpec(4,5)
	for jj in xrange(ngals):

		# find most recent output file
		# with the objname
		folder = "/".join(filebase[jj].split('/')[:-1])
		filename = filebase[jj].split("/")[-1]
		files = [f for f in os.listdir(folder) if "_".join(f.split('_')[:-2]) == filename]	
		times = [f.split('_')[-2] for f in files]

		# if we found no files, skip this object
		if len(times) == 0:
			print 'Failed to find any files in '+folder+' of type ' +filename+' to extract times'
			nfail+=1
			continue

		# load results
		mcmc_filename=filebase[jj]+'_'+max(times)+"_mcmc"
		model_filename=filebase[jj]+'_'+max(times)+"_model"

		try:
			sample_results, powell_results, model = read_results.read_pickles(mcmc_filename, model_file=model_filename,inmod=None)
		except (ValueError,EOFError,KeyError):
			print mcmc_filename + ' failed during output writing'
			nfail+=1
			continue
		except IOError:
			print mcmc_filename + ' does not exist!'
			nfail+=1
			continue

		# check for existence of extra information
		try:
			sample_results['quantiles']
		except:
			print 'Generating extra information for '+mcmc_filename+', '+model_filename
			extra_output.post_processing(params[jj])
			sample_results, powell_results, model = read_results.read_pickles(mcmc_filename, model_file=model_filename,inmod=None)

		# generate output buckets for photometric terms
		try:
			parnames
		except:
			
			# load truths
			truths = threed_dutils.load_truths(os.getenv('APPS')+'/threed'+sample_results['run_params']['truename'].split('/threed')[1],
		                                       sample_results['run_params']['objname'],
		                                       sample_results)

			# define relevant parameters
			parnames = np.array(sample_results['model'].theta_labels())
			phot_params = [x for x in parnames if ('gp' in x) or ('phot' in x)]

			# define outputs
			nphot = len(phot_params)
			noutliers = len([x for x in parnames if 'gp_outlier_locs' in x])
			nbins = 50
			histbounds = []
			output = np.zeros(shape=(nbins,nphot))
			edges = np.zeros(shape=(nbins,nphot))

			# create boundaries
			for param in phot_params:
				histbounds.append(sample_results['model'].theta_bounds()[np.where(parnames==param)[0][0]])

		# fill outputs
		for ii in xrange(nphot):
			out,edge = np.histogram(sample_results['flatchain'][:,parnames==phot_params[ii]][:,0],
				                    range=histbounds[ii],bins=nbins)
			if ii == 1:
				ax = plt.subplot(gs[jj])
				ax.bar((edge[:-1]+edge[1:])/2.,out,width=edge[1]-edge[0])

				#if np.sum(out > 20000) > 0:
				#	print 1/0

			output[:,ii] += out/(np.sum(out)*1.0)
			edges[:,ii] = (edge[:-1]+edge[1:])/2.
	
	
	# if we have outliers, combine them into one plot
	vars_to_plot = [True if (x[:10] != 'gp_outlier') or (x[-1] == '1') else False for x in phot_params ]
	# make plots
	nplot = np.sum(vars_to_plot)
	gs = gridspec.GridSpec(1,nplot)
	fig = plt.figure(figsize = (nplot*6,5))
	gs.update(wspace=0.35, hspace=0.35)
	lw = 1.2
	plotted=0

	for ii in xrange(nphot):
		if vars_to_plot[ii] == True:

			# gather terms
			if 'gp_outlier' in phot_params[ii]:
				histplot = np.sum(output[:,ii:ii+noutliers],axis=1)
			else:
				histplot = output[:,ii]

			# plot
			ax = plt.subplot(gs[plotted])
			plotted+=1
			ax.bar(edges[:,ii],histplot,width=edges[1,ii]-edges[0,ii]
				  ,alpha=0.5,linewidth=0.0,edgecolor='grey',color='blue')

			# labels
			ax.set_yticklabels([])
			ax.set_xlabel(phot_params[ii])

			# truth
			if 'gp_outlier' in phot_params[ii]:
				ind = np.where(parnames==phot_params[ii])[0][0]
				for gg in xrange(noutliers):
					ptruth = truths['truths'][ind+gg]
					ax.axvline(ptruth+1,color='r',alpha=0.9,linewidth=lw)
			else:
				ptruth = truths['truths'][parnames==phot_params[ii]]
				ax.axvline(ptruth,color='r',alpha=0.9,linewidth=lw)

			# range
			ax.set_xlim(np.min(edges[:,ii]),np.max(edges[:,ii]))

			# rough percentages
			cumsum = np.cumsum(output[:,ii])/np.sum(output[:,ii])
			cumsum_interp = interp1d(cumsum,edges[:,ii], bounds_error = False, fill_value = 0)
			q16,q50,q84 = cumsum_interp(0.16),cumsum_interp(0.5),cumsum_interp(0.84)

			ax.axvline(q16, ls="dashed", color='k',linewidth=lw)
			ax.axvline(q50, color='k',linewidth=lw)
			ax.axvline(q84, ls="dashed", color='k',linewidth=lw)


	plt.savefig(outname,dpi=300)
	os.system('open '+outname)
	print 1/0

def plot_sfh_ensemble(runname):

	'''
	given a run, for each galaxy, load the output, and
	plot all SFHs versus the truth. also plot average versus
	the truth, and mass/SFR offsets.
	'''

	from threedhst_diag import add_sfh_plot

	filebase,params,ancilname=threed_dutils.generate_basenames(runname)
	ngals = len(filebase)
	outname = '/Users/joel/code/python/threedhst_bsfh/plots/ensemble_plots/'+runname+'/sfh_ensemble.png'

	nfail = 0
	fig = plt.figure()
	gs = gridspec.GridSpec(4,5)
	gs.update(hspace=0.16,wspace=0.13)

	# define outputs
	diff    = np.zeros(shape=(ngals,2))
	sigdiff = np.zeros(shape=(ngals,2))

	for jj in xrange(ngals):

		# find most recent output file
		# with the objname
		folder = "/".join(filebase[jj].split('/')[:-1])
		filename = filebase[jj].split("/")[-1]
		files = [f for f in os.listdir(folder) if "_".join(f.split('_')[:-2]) == filename]	
		times = [f.split('_')[-2] for f in files]

		# if we found no files, skip this object
		if len(times) == 0:
			print 'Failed to find any files in '+folder+' of type ' +filename+' to extract times'
			nfail+=1
			continue

		# load results
		mcmc_filename=filebase[jj]+'_'+max(times)+"_mcmc"
		model_filename=filebase[jj]+'_'+max(times)+"_model"

		try:
			sample_results, powell_results, model = read_results.read_pickles(mcmc_filename, model_file=model_filename,inmod=None)
		except (ValueError,EOFError,KeyError):
			print mcmc_filename + ' failed during output writing'
			nfail+=1
			continue
		except IOError:
			print mcmc_filename + ' does not exist!'
			nfail+=1
			continue

		# check for existence of extra information
		try:
			sample_results['quantiles']
		except:
			print 'Generating extra information for '+mcmc_filename+', '+model_filename
			extra_output.post_processing(params[jj])
			sample_results, powell_results, model = read_results.read_pickles(mcmc_filename, model_file=model_filename,inmod=None)

		# load truths
		truths = threed_dutils.load_truths(os.getenv('APPS')+'/threed'+sample_results['run_params']['truename'].split('/threed')[1],
			                                sample_results['run_params']['objname'],
			                                sample_results)

		# plot SFH
		ax_loc = plt.subplot(gs[jj],zorder=-32+jj)
		add_sfh_plot(sample_results,None,ax_loc,truths=truths,fast=None)

		# calculate mass + SFR offset
		mass_loc = sample_results['extras']['parnames'] == 'totmass'
		sfr_loc = sample_results['extras']['parnames'] == 'sfr_100'

		mass50 = np.log10(sample_results['extras']['q50'][mass_loc])
		sfr50  = np.log10(sample_results['extras']['q50'][sfr_loc])

		tmass = truths['extra_truths'][0]
		tsfr = truths['extra_truths'][1]

		# calculate sigma differences
		diff[jj,:] = np.squeeze([mass50-tmass,sfr50-tsfr])
		if diff[jj,0] > 0:
			sigdiff[jj,0] = (tmass-mass50)/(mass50-np.log10(sample_results['extras']['q16'][mass_loc]))
		else:
			sigdiff[jj,0] = (tmass-mass50)/(np.log10(sample_results['extras']['q84'][mass_loc])-mass50)
		if diff[jj,1] > 0:
			sigdiff[jj,1] = (tsfr-sfr50)/(sfr50-np.log10(sample_results['extras']['q16'][sfr_loc]))
		else:
			sigdiff[jj,1] = (tsfr-sfr50)/(np.log10(sample_results['extras']['q84'][sfr_loc])-sfr50)

		# add text
		axfontsize=4
		ax_loc.text(0.95,0.9, 'M='+'%.2f' % mass50+' ('+'%.2f' % tmass+','+'%.2f' % sigdiff[jj,0]+'$\sigma$)',
			        transform = ax_loc.transAxes, horizontalalignment='right',fontsize=axfontsize*0.9)
		ax_loc.text(0.95,0.8, 'SFR='+'%.2f' % sfr50+' ('+'%.2f' % tsfr+','+'%.2f' % sigdiff[jj,1]+'$\sigma$)',
			        transform = ax_loc.transAxes, horizontalalignment='right',fontsize=axfontsize*0.9)
	
	# define average quantities
	nsuccess = np.sum(diff[:,0] != 0.0)
	av_diff     = np.sum(diff,axis=0)/nsuccess
	av_sigdiff  = np.sum(diff**2/nsuccess,axis=0)**0.5
	plt.suptitle(runname)
	plt.text(0.16, 0.92,'$\Delta$M$_{avg}$='+'%.2f' % av_diff[0] + ' dex, RMS='+'%.2f' % av_sigdiff[0]+' dex',transform=fig.transFigure)
	plt.text(0.57, 0.92,'$\Delta$SFR$_{avg}$='+'%.2f' % av_diff[1] + ' dex, RMS='+'%.2f' % av_sigdiff[1]+' dex',transform=fig.transFigure)
	plt.savefig(outname,dpi=300)
	os.system('open '+outname)

def delta_versus_ssfr():

	'''
	given N testruns, for each galaxy, load the output, and
	plot all SFHs versus the truth. also plot average versus
	the truth, and mass/SFR offsets.
	'''

	runnames = ['testsed_quiescent','testsed_oldburst','testsed_burst','testsed_constant']
	runnames = ['testsed_oldburst']
	outname = os.getenv('APPS')+'/threedhst_bsfh/plots/ensemble_plots/'+runnames[0]+'/delta_versus_ssfr.png'
	outname = os.getenv('APPS')+'/threedhst_bsfh/plots/ensemble_plots/'+runnames[0]+'/deltam_versus_deltasfr.png'

	# define outputs
	tmass          = np.zeros(shape=(0))
	tsfr           = np.zeros(shape=(0))
	tssfr          = np.zeros(shape=(0))
	sfr            = np.zeros(shape=(0))
	mass           = np.zeros(shape=(0))
	reduced_chisq  = np.zeros(shape=(0))

	# initialize sps
	sps = threed_dutils.setup_sps()

	# grab masses and star formation rates
	for run in runnames:

		filebase,params,ancilname=threed_dutils.generate_basenames(run)
		ngals = len(filebase)

		for jj in xrange(ngals):

			# find most recent output file
			# with the objname
			folder = "/".join(filebase[jj].split('/')[:-1])
			filename = filebase[jj].split("/")[-1]
			files = [f for f in os.listdir(folder) if "_".join(f.split('_')[:-2]) == filename]	
			times = [f.split('_')[-2] for f in files]

			# if we found no files, skip this object
			if len(times) == 0:
				print 'Failed to find any files in '+folder+' of type ' +filename+' to extract times'
				continue

			# load results
			mcmc_filename=filebase[jj]+'_'+max(times)+"_mcmc"
			model_filename=filebase[jj]+'_'+max(times)+"_model"

			try:
				sample_results, powell_results, model = read_results.read_pickles(mcmc_filename, model_file=model_filename,inmod=None)
			except (ValueError,EOFError,KeyError):
				print mcmc_filename + ' failed during output writing'
				continue
			except IOError:
				print mcmc_filename + ' does not exist!'
				continue

			# check for existence of extra information
			try:
				sample_results['quantiles']
			except:
				print 'Generating extra information for '+mcmc_filename+', '+model_filename
				extra_output.post_processing(params[jj])
				sample_results, powell_results, model = read_results.read_pickles(mcmc_filename, model_file=model_filename,inmod=None)

			# load truths
			truths = threed_dutils.load_truths(os.getenv('APPS')+'/threed'+sample_results['run_params']['truename'].split('/threed')[1],
				                                sample_results['run_params']['objname'],
				                                sample_results)

			# calculate fit masses + SFRs
			mass_loc = sample_results['extras']['parnames'] == 'totmass'
			sfr_loc = sample_results['extras']['parnames'] == 'sfr_100'

			# q50 mass + SFR
			m = np.log10(sample_results['extras']['q50'][mass_loc])
			s = np.log10(sample_results['extras']['q50'][sfr_loc])

			# best-fit model mass+SFR
			#pars = sample_results['quantiles']['maxprob_params']
			#parnames = sample_results['model'].theta_labels()      
			#tempmass = pars[np.array([True if 'mass' in x else False for x in parnames])]
			#m = np.log10(np.sum(tempmass))

			#tau = pars[np.array([True if 'tau' in x else False for x in parnames])]
			#sf_start = pars[np.array([True if 'sf_start' in x else False for x in parnames])]
			#tage = np.zeros(len(tau))+sample_results['model'].params['tage'][0]
			#deltat=0.1
			#s=np.log10(threed_dutils.integrate_sfh(np.atleast_1d(tage)[0]-deltat,np.atleast_1d(tage)[0],tempmass,tage,tau,sf_start)*np.sum(tempmass)/(deltat*1e9))

			mass=np.append(mass, m)
			sfr=np.append(sfr, s)

			# calculate true masses + sfrs
			tmass=np.append(tmass,truths['extra_truths'][0])
			tsfr =np.append(tsfr,truths['extra_truths'][1])

			# calculate true ssfr
			tssfr=np.append(tssfr,np.log10(10**tsfr[-1]/10**tmass[-1]))

			# calculate reduced chi-squared
			thetas = [x for x in sample_results['quantiles']['q50']]
			#thetas = sample_results['quantiles']['maxprob_params']
			spec,mags,w = sample_results['model'].mean_model(thetas, sample_results['obs'], sps=sps,norm_spec=False)
			
			chi = (mags-sample_results['obs']['maggies'])/sample_results['obs']['maggies_unc']

			chisq=np.sum(chi**2)
			ndof = np.sum(sample_results['obs']['phot_mask']) - len(sample_results['model'].free_params)-1
			reduced_chisq = np.append(reduced_chisq,chisq/(ndof-1))

	#### set up plot parameters
	gs = gridspec.GridSpec(3,1)
	fig = plt.figure(figsize = (6.5,15.5))
	gs.update(wspace=0.0, hspace=0.0)
	ms = 8 # marker size
	lw = 1.5 # line width
	axlim = [-13.5,-7,-0.5,0.5]

	#### clip sSFRs
	tssfr = np.clip(tssfr,-13,-5)
	ssfr  = np.clip(np.log10(10**sfr/10**mass),-13,-5)

	#### make cut in chisq
	#plt.hist(reduced_chisq,range=(0,10))
	chisq_limit   = 10.0
	bad     = reduced_chisq >  chisq_limit
	good    = reduced_chisq <= chisq_limit

	#### PLOT DELTA MASS #### 
	ax = plt.subplot(gs[0])
	deltam = mass-tmass
	ax.plot(tssfr[good],deltam[good],'bo',alpha=0.5,ms=ms)
	ax.plot(tssfr[bad],deltam[bad],'ro',alpha=0.5,ms=ms)

	# plot running median
	bins,running_median = threed_dutils.running_median(tssfr[good],deltam[good],nbins=10)
	ax.plot(bins,running_median,'k-',linewidth=lw)

	# labels, width, helpful lines
	ax.axis(axlim)
	ax.hlines(0.0,ax.get_xlim()[0],ax.get_xlim()[1], linestyle='--',colors='grey',alpha=0.5)
	ax.set_ylabel('$\Delta$log(M)')

	# turn off bottom ticklabel
	ax.xaxis.get_major_ticks()[0].label1On = False

	#### PLOT DELTA SFR #### 
	ax = plt.subplot(gs[1])
	deltasfr = sfr-tsfr
	ax.plot(tssfr[good],deltasfr[good],'bo',alpha=0.5,ms=ms)
	ax.plot(tssfr[bad],deltasfr[bad],'ro',alpha=0.5,ms=ms)

	# plot running median
	bins,running_median = threed_dutils.running_median(tssfr[good],deltasfr[good],nbins=10)
	ax.plot(bins,running_median,'k-',linewidth=lw)

	# labels, width, helpful lines
	ax.axis(axlim)
	ax.hlines(0.0,ax.get_xlim()[0],ax.get_xlim()[1], linestyle='--',colors='grey',alpha=0.5)
	ax.set_ylabel('$\Delta$log(SFR)')

	# turn off bottom ticklabel
	ax.xaxis.get_major_ticks()[0].label1On = False

	#### PLOT DELTA SSFR #### 
	ax = plt.subplot(gs[2])
	deltassfr = ssfr-tssfr
	ax.plot(tssfr[good],deltassfr[good],'bo',alpha=0.5,ms=ms)
	ax.plot(tssfr[bad],deltassfr[bad],'ro',alpha=0.5,ms=ms)

	# plot running median
	bins,running_median = threed_dutils.running_median(tssfr[good],deltassfr[good],nbins=10)
	ax.plot(bins,running_median,'k-',linewidth=lw)

	# labels, width, helpful lines
	ax.axis(axlim)
	ax.hlines(0.0,ax.get_xlim()[0],ax.get_xlim()[1], linestyle='--',colors='grey',alpha=0.5)
	ax.set_ylabel('$\Delta$log(sSFR)')
	ax.set_xlabel('log(sSFR$_{true}$) [yr$^{-1}$]')

	plt.savefig(outname, dpi=300)
	os.system('open '+outname)

	#### PLOT DELTA(MIPS FLUX)
	#fig = plt.figure(figsize = (6.5,6.5))
	#ax = fig.add_subplot(111)
	#ax.plot((mips_chi-tmips_chi)/tmips_chi,deltasfr,'go')
	#ax.plot(tmips_chi,deltasfr,'bo')
	#ax.axis([-0.5,0.5,-2,2])
	#plt.show()
	print 1/0

	#### DELTAM VERSUS DELTASSFR ####
	fig = plt.figure(figsize = (6.5,6.5))
	ax = fig.add_subplot(111)
	ax.plot(deltam,deltasfr,'go')
	ax.axlim([-0.5,0.5,-0.5,0.5])
	print 1/0

def plot_ssfr_prior():

	'''
	plot distribution of ssfrs for a specific set of test model parameters
	'''

	#### load a single output. this will be used to gather
	#### tage and define theta labels
	mcmc_filename  = '/Users/joel/code/python/threedhst_bsfh/results/testsed_quiescent/testsed_quiescent_16_1431381068_mcmc'
	model_filename = '/Users/joel/code/python/threedhst_bsfh/results/testsed_quiescent/testsed_quiescent_16_1431381068_model'
	sample_results, powell_results, model = read_results.read_pickles(mcmc_filename, model_file=model_filename,inmod=None)

	#### define truths file, and ngals
	truths_file    = '/Users/joel/code/python/threedhst_bsfh/data/sfr_prior_test.dat'
	ngals          = 10000

	#### load all truths, extract masses + SFRs
	mass,sfr = [],[]
	for kk in xrange(ngals):
		truths = threed_dutils.load_truths(truths_file,
			                               str(kk),
			                               sample_results)
		mass.append(truths['extra_truths'][0])
		sfr.append(truths['extra_truths'][1])

	ssfr = 10**np.array(sfr)/10**np.array(mass)
	ssfr = np.clip(ssfr,1e-14,1e-7)
	plt.hist(np.log10(ssfr),bins=20,color = 'blue',alpha=0.5)
	plt.xlabel('log(sSFR)')
	plt.ylabel('N')
	plt.show()
	plt.savefig('sSFR_prior.png')
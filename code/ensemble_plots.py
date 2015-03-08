import os, threed_dutils, pickle, extra_output, triangle, pylab
from bsfh import read_results
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.cosmology import WMAP9
from astropy import constants

# minimum flux: no model emission line has strength of 0!
pc2cm = 3.08567758e18
minmodel_flux = 2e-1

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
		z[jj] = sample_results['model_params'][0]['init'][0]
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
		ancildat.append(threed_dutils.load_ancil_data(os.getenv('APPS')+'/threedhst_bsfh/data/'+ancilname,
			            							  sample_results['run_params']['objname']))
	
		print jj

	print 'total galaxies: {0}, successful loads: {1}'.format(ngals,ngals-nfail)
	print 'saving in {0}'.format(outname)


	output = {'outname': output_name,\
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
	sfr_obs = np.clip(np.array([x['sfr'][0] for x in ensemble['ancildat']]),1e-2,1e4)
	z_sfr = np.array([x['z_sfr'] for x in ensemble['ancildat']])
	valid_comp = ensemble['z'] > 0

	# calculate tuniv
	tuniv=WMAP9.age(ensemble['z'][valid_comp]).value*1.2

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

	try:
		tau = np.log10(np.clip(tuniv-ensemble['q50'][ensemble['parname'] == 'tau'][0,valid_comp],1e-2,1e50))
		tauerrs = asym_errors(np.clip(tuniv-ensemble['q50'][ensemble['parname'] == 'tau'][0,valid_comp],1e-4,1e50),np.clip(tuniv-ensemble['q84'][ensemble['parname'] == 'tau'][0,valid_comp],1e-4,1e50),np.clip(tuniv-ensemble['q16'][ensemble['parname'] == 'tau'][0,valid_comp],1e-4,1e50),log=True)
	except:
		tau = np.log10(np.clip(tuniv-ensemble['q50'][ensemble['parname'] == 'tau_1'][0,valid_comp],1e-2,1e50))
		tauerrs = asym_errors(np.clip(tuniv-ensemble['q50'][ensemble['parname'] == 'tau_1'][0,valid_comp],1e-4,1e50),np.clip(tuniv-ensemble['q84'][ensemble['parname'] == 'tau_1'][0,valid_comp],1e-4,1e50),np.clip(tuniv-ensemble['q16'][ensemble['parname'] == 'tau_1'][0,valid_comp],1e-4,1e50),log=True)

	x_data = [mass,\
			  ensemble['q50'][ensemble['parname'] == 'logzsol'][0,valid_comp],\
	          ensemble['q50'][ensemble['parname'] == 'logzsol'][0,valid_comp],\
	          ensemble['q50'][ensemble['parname'] == 'logzsol'][0,valid_comp],\
	          np.log10(ensemble['q50'][ensemble['parname'] == 'sfr_100'][0]),\
	          np.log10(ensemble['q50'][ensemble['parname'] == 'sfr_100'][0]),\
	          mass,\
	          np.log10(ensemble['q50'][ensemble['parname'] == 'ssfr_100'][0]),\
	          np.log10(sfr_obs),\
	          np.log10(sfr_obs),\
	          np.log10(np.clip(np.array([x['q50'][x['name'] == 'Halpha'] for x in ensemble['model_emline']]),minmodel_flux,1e50)),
	          np.log10(lobs_halpha)
	          ]

	x_err  = [masserrs,\
			  asym_errors(ensemble['q50'][ensemble['parname'] == 'logzsol'][0,valid_comp],ensemble['q84'][ensemble['parname'] == 'logzsol'][0,valid_comp],ensemble['q16'][ensemble['parname'] == 'logzsol'][0,valid_comp]),\
			  asym_errors(ensemble['q50'][ensemble['parname'] == 'logzsol'][0,valid_comp],ensemble['q84'][ensemble['parname'] == 'logzsol'][0,valid_comp],ensemble['q16'][ensemble['parname'] == 'logzsol'][0,valid_comp]),\
			  asym_errors(ensemble['q50'][ensemble['parname'] == 'logzsol'][0,valid_comp],ensemble['q84'][ensemble['parname'] == 'logzsol'][0,valid_comp],ensemble['q16'][ensemble['parname'] == 'logzsol'][0,valid_comp]),\
			  asym_errors(ensemble['q50'][ensemble['parname'] == 'sfr_100'][0],ensemble['q84'][ensemble['parname'] == 'sfr_100'][0],ensemble['q16'][ensemble['parname'] == 'sfr_100'][0],log=True),\
			  asym_errors(ensemble['q50'][ensemble['parname'] == 'sfr_100'][0],ensemble['q84'][ensemble['parname'] == 'sfr_100'][0],ensemble['q16'][ensemble['parname'] == 'sfr_100'][0],log=True),\
			  masserrs,\
			  asym_errors(ensemble['q50'][ensemble['parname'] == 'ssfr_100'][0],ensemble['q84'][ensemble['parname'] == 'ssfr_100'][0],ensemble['q16'][ensemble['parname'] == 'sfr_100'][0],log=True),\
			  None,\
			  None,\
			  asym_errors(np.clip(np.array([x['q50'][x['name'] == 'Halpha'] for x in ensemble['model_emline']]),minmodel_flux,1e50),np.clip(np.array([x['q84'][x['name'] == 'Halpha'] for x in ensemble['model_emline']]),minmodel_flux,1e50),np.clip(np.array([x['q16'][x['name'] == 'Halpha'] for x in ensemble['model_emline']]),minmodel_flux,1e50),log=True),\
			  asym_errors(lobs_halpha,lobs_halpha+lobs_halpha_err,lobs_halpha-lobs_halpha_err,log=True)\
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
	            r'log(H$\alpha$ lum) [obs]'
	            ]

	y_data = [ensemble['q50'][ensemble['parname'] == 'logzsol'][0,valid_comp],\
			  np.log10(ensemble['q50'][ensemble['parname'] == 'sfr_100'][0,valid_comp]),\
			  ensemble['q50'][ensemble['parname'] == 'dust_index'][0,valid_comp],\
	          ensemble['q50'][ensemble['parname'] == 'half_time'][0,valid_comp],\
	          ensemble['q84'][ensemble['parname'] == 'half_time'][0]-ensemble['q16'][ensemble['parname'] == 'half_time'][0],\
	          ensemble['q50'][ensemble['parname'] == 'half_time'][0],\
	          np.log10(ensemble['q50'][ensemble['parname'] == 'sfr_100'][0,valid_comp]),\
	          ensemble['q84'][ensemble['parname'] == 'half_time'][0]-ensemble['q16'][ensemble['parname'] == 'half_time'][0],\
	          np.log10(ensemble['q50'][ensemble['parname'] == 'sfr_100'][0,valid_comp]),\
	          np.log10(ensemble['q50'][ensemble['parname'] == 'sfr_1000'][0]),\
	          tau,\
	          np.log10(sfr_obs)\
	          ]

	y_err  = [asym_errors(ensemble['q50'][ensemble['parname'] == 'logzsol'][0,valid_comp],ensemble['q84'][ensemble['parname'] == 'logzsol'][0,valid_comp],ensemble['q16'][ensemble['parname'] == 'logzsol'][0,valid_comp]),\
			  asym_errors(ensemble['q50'][ensemble['parname'] == 'sfr_100'][0,valid_comp],ensemble['q84'][ensemble['parname'] == 'sfr_100'][0,valid_comp],ensemble['q16'][ensemble['parname'] == 'sfr_100'][0,valid_comp],log=True),\
			  asym_errors(ensemble['q50'][ensemble['parname'] == 'dust_index'][0,valid_comp],ensemble['q84'][ensemble['parname'] == 'dust_index'][0,valid_comp],ensemble['q16'][ensemble['parname'] == 'dust_index'][0,valid_comp],log=False),\
			  None,\
			  None,\
			  asym_errors(ensemble['q50'][ensemble['parname'] == 'half_time'][0],ensemble['q84'][ensemble['parname'] == 'half_time'][0],ensemble['q16'][ensemble['parname'] == 'half_time'][0]),\
			  asym_errors(ensemble['q50'][ensemble['parname'] == 'sfr_100'][0,valid_comp],ensemble['q84'][ensemble['parname'] == 'sfr_100'][0,valid_comp],ensemble['q16'][ensemble['parname'] == 'sfr_100'][0,valid_comp],log=True),\
			  None,\
			  asym_errors(ensemble['q50'][ensemble['parname'] == 'sfr_100'][0,valid_comp],ensemble['q84'][ensemble['parname'] == 'sfr_100'][0,valid_comp],ensemble['q16'][ensemble['parname'] == 'sfr_100'][0,valid_comp],log=True),\
			  asym_errors(ensemble['q50'][ensemble['parname'] == 'sfr_1000'][0],ensemble['q84'][ensemble['parname'] == 'sfr_1000'][0],ensemble['q16'][ensemble['parname'] == 'sfr_1000'][0],log=True),\
			  tauerrs,\
			  None\
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
				'obsemline_sfrobs'
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
		if x_labels[jj] == r'log(SFR$_{obs}$) [M$_{\odot}$/yr]' or x_labels[jj] == r'log(sSFR$_{obs}$) [yr$^{-1}$]':
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
	import pylab
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

def emline_comparison(runname,emline_base='Halpha', chain_emlines=False):
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
		except (ValueError,EOFError):
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
			sps = threed_dutils.setup_sps()
		else:
			sps = threed_dutils.setup_sps(zcontinuous=1)

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
		obs['phot_mask'][-5:-1] = True # restore IRAC

		obs_mags = sample_results['obs']['maggies'][mask]
		obs_lam  = sample_results['obs']['wave_effective'][mask]
		obs_err  = sample_results['obs']['maggies_unc'][mask]
		linerr_down = np.clip(obs_mags-obs_err, 1e-80, 1e80)
		linerr_up = np.clip(obs_mags+obs_err, 1e-80, 1e80)
		yerr = [np.log10(obs_mags) - np.log10(linerr_down),
		        np.log10(linerr_up)- np.log10(obs_mags)]

		# nfnu or flam?
		#spec *= 3e18/w**2
		obs_mags *= (3e18)/obs_lam
		ax.errorbar(np.log10(obs_lam), np.log10(obs_mags), yerr=yerr,
					fmt='ok',ms=5)
		
		# save
		outname = sample_results['run_params']['objname']
		plt.savefig('/Users/joel/code/python/threedhst_bsfh/plots/testmet/'+outname+'_metsed.png', dpi=300)
		plt.close()

















import os, threed_dutils, triangle, pickle, extra_output
from bsfh import read_results
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.cosmology import WMAP9
from astropy import constants

# minimum flux: no model emission line has strength of 0!
minmodel_flux = 1e-2

def asym_errors(center, up, down, log=False):

	if log:
		errup = np.log10(up)-np.log10(center)
		errdown = np.log10(center)-np.log10(down)
		errarray = [errdown,errup]
	else:
		errarray = [center-down,up-center]

	return errarray

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

	filebase,params=threed_dutils.generate_basenames(runname)
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
		except:
			print 'Failed to open '+ mcmc_filename +','+model_filename
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
			q_16, q_50, q_84, thetamax = (np.zeros(shape=(ntheta,ngals))+np.nan for i in range(4))
			z, mips_sn = (np.zeros(ngals) for i in range(2))
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

		# grab best-fitting model
		thetamax[:,jj] = np.concatenate((sample_results['quantiles']['maxprob_params'],
			 							 sample_results['extras']['maxprob']))

		# save dictionary lists
		obs.append(sample_results['obs'])
		model_emline.append(sample_results['model_emline'])
		ancildat.append(threed_dutils.load_ancil_data(os.getenv('APPS')+'/threedhst_bsfh/data/COSMOS_testsamp.dat',
			            							  sample_results['run_params']['objname']))
	
		print jj

	print 'total galaxies: {0}, successful loads: {1}'.format(ngals,ngals-nfail)
	print 'saving in {0}'.format(outname)

	output = {'outname': output_name,\
			  'parname': np.array(sample_results['model'].theta_labels() + ['half_time','sfr_10','sfr_100','sfr_1000','ssfr_100']),\
		      'q16': q_16,\
		      'q50': q_50,\
		      'q84': q_84,\
		      'maxprob': thetamax,\
		      'mips_sn': mips_sn,\
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

	x_data = [np.log10(ensemble['q50'][ensemble['parname'] == 'mass'][0]),\
	          ensemble['q84'][ensemble['parname'] == 'dust_index'][0]-ensemble['q16'][ensemble['parname'] == 'dust_index'][0],\
	          ensemble['q84'][ensemble['parname'] == 'dust_index'][0]-ensemble['q16'][ensemble['parname'] == 'dust_index'][0],\
	          ensemble['mips_sn'],\
	          np.log10(ensemble['q50'][ensemble['parname'] == 'sfr_100'][0]),\
	          np.log10(ensemble['q50'][ensemble['parname'] == 'sfr_100'][0]),\
	          np.log10(ensemble['q50'][ensemble['parname'] == 'sfr_100'][0]),\
	          np.log10(ensemble['q50'][ensemble['parname'] == 'mass'][0]),\
	          np.log10(ensemble['q50'][ensemble['parname'] == 'ssfr_100'][0]),\
	          np.log10(sfr_obs),\
	          np.log10(sfr_obs),\
	          np.log10(sfr_obs/ensemble['q50'][ensemble['parname'] == 'mass'][0,valid_comp]),\
	          np.log10(np.clip(np.array([x['q50'][x['name'].index('Halpha')] for x in ensemble['model_emline']]),minmodel_flux,1e50)),
	          np.log10(np.clip(np.array([x['q50'][x['name'].index('Halpha')] for x in ensemble['model_emline']]),minmodel_flux,1e50)),
	          np.log10(lobs_halpha)
	          ]

	x_err  = [asym_errors(ensemble['q50'][ensemble['parname'] == 'mass'][0],ensemble['q84'][ensemble['parname'] == 'mass'][0], ensemble['q16'][ensemble['parname'] == 'mass'][0],log=True),\
			  None,\
			  None,\
			  None,\
			  asym_errors(ensemble['q50'][ensemble['parname'] == 'sfr_100'][0],ensemble['q84'][ensemble['parname'] == 'sfr_100'][0],ensemble['q16'][ensemble['parname'] == 'sfr_100'][0],log=True),\
			  asym_errors(ensemble['q50'][ensemble['parname'] == 'sfr_100'][0],ensemble['q84'][ensemble['parname'] == 'sfr_100'][0],ensemble['q16'][ensemble['parname'] == 'sfr_100'][0],log=True),\
			  asym_errors(ensemble['q50'][ensemble['parname'] == 'sfr_100'][0],ensemble['q84'][ensemble['parname'] == 'sfr_100'][0],ensemble['q16'][ensemble['parname'] == 'sfr_100'][0],log=True),\
			  asym_errors(ensemble['q50'][ensemble['parname'] == 'mass'][0],ensemble['q84'][ensemble['parname'] == 'mass'][0],ensemble['q16'][ensemble['parname'] == 'mass'][0],log=True),\
			  asym_errors(ensemble['q50'][ensemble['parname'] == 'ssfr_100'][0],ensemble['q84'][ensemble['parname'] == 'ssfr_100'][0],ensemble['q16'][ensemble['parname'] == 'sfr_100'][0],log=True),\
			  None,\
			  None,\
			  None,\
			  asym_errors(np.clip(np.array([x['q50'][x['name'].index('Halpha')] for x in ensemble['model_emline']]),minmodel_flux,1e50),np.clip(np.array([x['q84'][x['name'].index('Halpha')] for x in ensemble['model_emline']]),minmodel_flux,1e50),np.clip(np.array([x['q16'][x['name'].index('Halpha')] for x in ensemble['model_emline']]),minmodel_flux,1e50),log=True),\
			  asym_errors(np.clip(np.array([x['q50'][x['name'].index('Halpha')] for x in ensemble['model_emline']]),minmodel_flux,1e50),np.clip(np.array([x['q84'][x['name'].index('Halpha')] for x in ensemble['model_emline']]),minmodel_flux,1e50),np.clip(np.array([x['q16'][x['name'].index('Halpha')] for x in ensemble['model_emline']]),minmodel_flux,1e50),log=True),\
			  asym_errors(lobs_halpha,lobs_halpha+lobs_halpha_err,lobs_halpha-lobs_halpha_err,log=True)\
			 ]

	x_labels = [r'log(M) [M$_{\odot}$]',
	            r'$\sigma$ (dust index)',
	            r'$\sigma$ (dust index)',
	            'MIPS S/N',
	            r'log(SFR_{100}) [M$_{\odot}$/yr]',
	            r'log(SFR_{100}) [M$_{\odot}$/yr]',
	            r'log(SFR_{100}) [M$_{\odot}$/yr]',
	            r'log(M) [M$_{\odot}$]',
	            r'log(sSFR_{100}) [yr$^{-1}$]',
	            r'log(SFR$_{obs}$) [M$_{\odot}$/yr]',
	            r'log(SFR$_{obs}$) [M$_{\odot}$/yr]',
	            r'log(sSFR$_{obs}$) [yr$^{-1}$]',
	            r'log(H$\alpha$ flux) [model]',
	            r'log(H$\alpha$ flux) [model]',
	            r'log(H$\alpha$ lum) [obs]'
	            ]

	y_data = [ensemble['q50'][ensemble['parname'] == 'logzsol'][0],\
	          ensemble['q84'][ensemble['parname'] == 'dust1'][0]-ensemble['q16'][ensemble['parname'] == 'dust1'][0],\
	          ensemble['q84'][ensemble['parname'] == 'dust2'][0]-ensemble['q16'][ensemble['parname'] == 'dust2'][0],\
	          ensemble['q84'][ensemble['parname'] == 'duste_qpah'][0]-ensemble['q16'][ensemble['parname'] == 'duste_qpah'][0],\
	          ensemble['q84'][ensemble['parname'] == 'half_time'][0]-ensemble['q16'][ensemble['parname'] == 'half_time'][0],\
	          ensemble['q50'][ensemble['parname'] == 'half_time'][0],\
	          ensemble['q84'][ensemble['parname'] == 'dust1'][0]-ensemble['q16'][ensemble['parname'] == 'dust1'][0],\
	          np.log10(ensemble['q50'][ensemble['parname'] == 'sfr_100'][0]),\
	          ensemble['q84'][ensemble['parname'] == 'half_time'][0]-ensemble['q16'][ensemble['parname'] == 'half_time'][0],\
	          np.log10(ensemble['q50'][ensemble['parname'] == 'sfr_100'][0,valid_comp]),\
	          np.log10(ensemble['q50'][ensemble['parname'] == 'sfr_1000'][0,valid_comp]),\
	          np.log10(ensemble['q50'][ensemble['parname'] == 'ssfr_100'][0,valid_comp]**-1),\
	          np.log10(np.clip(tuniv-ensemble['q50'][ensemble['parname'] == 'tburst'][0,valid_comp],1e-2,1e50)),\
	          np.log10(np.clip(tuniv-ensemble['q50'][ensemble['parname'] == 'tau'][0,valid_comp],1e-2,1e50)),\
	          np.log10(sfr_obs)\
	          ]

	y_err  = [asym_errors(ensemble['q50'][ensemble['parname'] == 'logzsol'][0],ensemble['q84'][ensemble['parname'] == 'logzsol'][0],ensemble['q16'][ensemble['parname'] == 'logzsol'][0]),\
			  None,\
			  None,\
			  None,\
			  None,\
			  asym_errors(ensemble['q50'][ensemble['parname'] == 'half_time'][0],ensemble['q84'][ensemble['parname'] == 'half_time'][0],ensemble['q16'][ensemble['parname'] == 'half_time'][0]),\
			  None,\
			  asym_errors(ensemble['q50'][ensemble['parname'] == 'sfr_100'][0],ensemble['q84'][ensemble['parname'] == 'sfr_100'][0],ensemble['q16'][ensemble['parname'] == 'sfr_100'][0],log=True),\
			  None,\
			  asym_errors(ensemble['q50'][ensemble['parname'] == 'sfr_100'][0,valid_comp],ensemble['q84'][ensemble['parname'] == 'sfr_100'][0,valid_comp],ensemble['q16'][ensemble['parname'] == 'sfr_100'][0,valid_comp],log=True),\
			  asym_errors(ensemble['q50'][ensemble['parname'] == 'sfr_1000'][0,valid_comp],ensemble['q84'][ensemble['parname'] == 'sfr_1000'][0,valid_comp],ensemble['q16'][ensemble['parname'] == 'sfr_1000'][0,valid_comp],log=True),\
			  asym_errors(ensemble['q50'][ensemble['parname'] == 'ssfr_100'][0,valid_comp]**-1,ensemble['q84'][ensemble['parname'] == 'ssfr_100'][0,valid_comp]**-1,ensemble['q16'][ensemble['parname'] == 'sfr_100'][0,valid_comp]**-1,log=True),\
			  asym_errors(np.clip(tuniv-ensemble['q50'][ensemble['parname'] == 'tburst'][0,valid_comp],1e-4,1e50),np.clip(tuniv-ensemble['q84'][ensemble['parname'] == 'tburst'][0,valid_comp],1e-4,1e50),np.clip(tuniv-ensemble['q16'][ensemble['parname'] == 'tburst'][0,valid_comp],1e-4,1e50),log=True),\
			  asym_errors(np.clip(tuniv-ensemble['q50'][ensemble['parname'] == 'tau'][0,valid_comp],1e-4,1e50),np.clip(tuniv-ensemble['q84'][ensemble['parname'] == 'tau'][0,valid_comp],1e-4,1e50),np.clip(tuniv-ensemble['q16'][ensemble['parname'] == 'tau'][0,valid_comp],1e-4,1e50),log=True),\
			  None\
			 ]

	y_labels = [r'log(Z$_{\odot}$)',
	            r'$\sigma$ (dust1)',
	            r'$\sigma$ (dust2)',
	            r'$\sigma$ (q$_{pah})$',
	            r'$\sigma$ (t$_{half}$) [Gyr]',
	            r't$_{half}$ [Gyr]',
	            r'$\sigma$ (dust1)',
	            r'log(SFR_{100}) [M$_{\odot}$/yr]',
	            r'$\sigma$ (t$_{half}$) [Gyr]',
	            r'log(SFR$_{mcmc,100}$) [M$_{\odot}$/yr]',
	            r'log(SFR$_{mcmc,1000}$) [M$_{\odot}$/yr]',
	            r'log(sSFR$_{mcmc,100}$) [yr$^{-1}$]',
	            r'log(t$_{burst}$/Gyr)',
	            r'log($\tau$/Gyr)',
	            r'log(SFR$_{obs}$) [M$_{\odot}$/yr]',
	            ]

	plotname = ['mass_metallicity',
				'deltadustindex_deltadust1',
				'deltadustindex_deltadust2',
				'mipssn_deltaqpah',
				'sfr100_deltahalftime',
				'sfr100_halftime',
				'sfr100_deltadust1',
				'mass_sfr100',
				'ssfr_deltahalftime',
				'sfrobs_sfrmcmc100',
				'sfrobs_sfrmcmc1000',
				'ssfrobs_ssfrmcmc100',
				'modelemline_tburst',
				'modelemline_tau',
				'obsemline_sfrobs'
				]

	assert len(plotname) == len(y_labels) == len(y_err) == len(y_data), 'improper number of y data'
	assert len(plotname) == len(x_labels) == len(x_err) == len(x_data), 'improper number of x data'

	for jj in xrange(len(x_data)):
		outname = os.getenv('APPS') + '/threedhst_bsfh/plots/ensemble_plots/'+runname+'/'+plotname[jj]+'.png'
		
		fig = plt.figure()
		ax = fig.add_subplot(111)
		
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

def nebcomp(runname):
	inname = os.getenv('APPS')+'/threedhst_bsfh/results/'+runname+'/'+runname+'_ensemble.pickle'
	outname_errs = os.getenv('APPS') + '/threedhst_bsfh/plots/ensemble_plots/'+runname+'/emline_comp'

	# if the save file doesn't exist, make it
	if not os.path.isfile(inname):
		collate_output(runname,inname)

	with open(inname, "rb") as f:
		ensemble=pickle.load(f)

	# extract model
	# 'luminosity' in cgs
	halpha_q50 = np.clip(np.array([x['q50'][x['name'].index('Halpha')] for x in ensemble['model_emline']]),minmodel_flux,1e50)
	halpha_q16 = np.clip(np.array([x['q16'][x['name'].index('Halpha')] for x in ensemble['model_emline']]),minmodel_flux,1e50)
	halpha_q84 = np.clip(np.array([x['q84'][x['name'].index('Halpha')] for x in ensemble['model_emline']]),minmodel_flux,1e50)
	ha_lam = np.array([x['lam'][x['name'].index('Halpha')] for x in ensemble['model_emline']])[0]
	halpha_errs = asym_errors(halpha_q50,halpha_q84,halpha_q16,log=True)
	halpha_q50 = np.log10(halpha_q50)

	# extract observations
	# flux: 10**-17 ergs / s / cm**2
	jansky_cgs=1e-23
	to_flux_density = (2.99e8/(ha_lam*1e-10))
	conv_factor = (10**(-17))/(jansky_cgs*3631*to_flux_density)
	conv_factor = 1e-17

	obs_ha = np.log10(np.array([x['Ha_flux'][0] for x in ensemble['ancildat']])*conv_factor)
	obs_ha_err = np.array([x['Ha_error'][0] for x in ensemble['ancildat']])*conv_factor
	
	fig = plt.figure()
	ax  = fig.add_subplot(111)

	###### HERE'S THE BIG PLOT ######
	ax[kk].errorbar(obs_ha,halpha_q50, 
			    fmt='bo', ecolor='0.20', alpha=0.8,
			    yerr=halpha_errs,xerr=obs_ha_err,linestyle=' ')


	#### formatting ####
	ax[kk].set_xlabel('observed Ha flux')
	ax[kk].set_ylabel('model Ha flux')

	# set plot limits to be slightly outside max values
	dynx, dyny = (np.nanmax(obs_ha)-np.nanmin(obs_ha))*0.05,\
		         (np.nanmax(halpha_q50)-np.nanmin(halpha_q50))*0.05
		
	ax[kk].axis((np.nanmin(obs_ha)-dynx,
			 np.nanmax(obs_ha)+dynx,
			 np.nanmin(halpha_q50)-dyny,
			 np.nanmax(halpha_q50)+dyny,
			 ))
	if np.nanmin(obs_ha)-dynx > np.nanmin(halpha_q50)-dyny:
		min = np.nanmin(halpha_q50)-dyny*3
	else:
		min = np.nanmin(obs_ha)-dynx*3
	if np.nanmax(obs_ha)+dynx > np.nanmax(halpha_q50)+dyny:
		max = np.nanmax(obs_ha)+dynx*3
	else:
		max = np.nanmax(halpha_q50)+dyny*3

	ax[kk].plot([-1e3,1e3],[-1e3,1e3],linestyle='--',color='0.1',alpha=0.8)
	ax[kk].axis((min,max,min,max))

	plt.savefig(outname_errs+'.png',dpi=300)
	plt.close()

def malpha_from_sfr(runname, add_dust=True):
	inname = os.getenv('APPS')+'/threedhst_bsfh/results/'+runname+'/'+runname+'_ensemble.pickle'
	outname_errs = os.getenv('APPS') + '/threedhst_bsfh/plots/ensemble_plots/'+runname+'/emline_comp_kscalc'
	if not add_dust:
		outname_errs = outname_errs+'_nodust_'

	# if the save file doesn't exist, make it
	if not os.path.isfile(inname):
		collate_output(runname,inname)

	with open(inname, "rb") as f:
		ensemble=pickle.load(f)

	sfr_str = ['10','100','1000']
	for bobo in xrange(len(sfr_str)):

		# CALCULATE EXPECTED HALPHA FLUX FROM SFH
		# ADD IN DUST DIMMING
		# eqn 2 in Kennicutt 1998, sfr [100 Myr], and redshift
		pc2cm = 3.08567758e18
		imf_fac = 1.7
		valid_comp = ensemble['z'] > 0
		l_halpha = 1.26e41*(ensemble['q50'][ensemble['parname'] == 'sfr_'+sfr_str[bobo]]*imf_fac)[0,valid_comp]
		distances = WMAP9.luminosity_distance(ensemble['z'][valid_comp]).value*1e6*pc2cm
		f_halpha = l_halpha/(4*np.pi*distances**2)/1e-17

		if add_dust:
			halpha_lam = 6562.0
			tau2 = ((halpha_lam/5500.)**ensemble['q50'][ensemble['parname'] == 'dust_index'][0,valid_comp])*ensemble['q50'][ensemble['parname'] == 'dust2'][0,valid_comp]
			tau1 = ((halpha_lam/5500.)**ensemble['q50'][ensemble['parname'] == 'dust_index'][0,valid_comp])*ensemble['q50'][ensemble['parname'] == 'dust1'][0,valid_comp]
			tautot = tau2#+tau1
			f_halpha = f_halpha*np.exp(-tautot)

		# EXTRACT MODEL HALPHA FLUX
		# 'luminosity' in cgs
		factor = (constants.L_sun.cgs.value)/(1e-17)/(4*np.pi*distances**2)
		halpha_q50 = np.clip(np.array([x['q50'][x['name'].index('Halpha')] for x in ensemble['model_emline']]),minmodel_flux,1e50)*factor
		halpha_q16 = np.clip(np.array([x['q16'][x['name'].index('Halpha')] for x in ensemble['model_emline']]),minmodel_flux,1e50)*factor
		halpha_q84 = np.clip(np.array([x['q84'][x['name'].index('Halpha')] for x in ensemble['model_emline']]),minmodel_flux,1e50)*factor
		ha_lam = np.array([x['lam'][x['name'].index('Halpha')] for x in ensemble['model_emline']])[0]
		halpha_errs = asym_errors(halpha_q50,halpha_q84,halpha_q16,log=True)

		# EXTRACT OBSERVED HALPHA FLUX
		# flux: 10**-17 ergs / s / cm**2
		obs_ha = np.array([x['Ha_flux'][0] for x in ensemble['ancildat']])
		obs_ha_err = np.array([x['Ha_error'][0] for x in ensemble['ancildat']])

		# SET UP PLOTTING QUANTITIES
		x_data = [np.log10(obs_ha),
				  np.log10(obs_ha),
				  np.log10(halpha_q50)
				  ]

		x_labels = [r'log(H$\alpha$ flux) [observed]',
					r'log(H$\alpha$ flux) [observed]',
					r'log(H$\alpha$ flux) [cloudy]',
					]

		y_data = [np.log10(halpha_q50),
				  np.log10(f_halpha),
				  np.log10(f_halpha)
				  ]

		y_labels = [r'log(H$\alpha$ flux) [cloudy]',
					r'log(H$\alpha$ flux) [ks]',
					r'log(H$\alpha$ flux) [ks]',
					]

		fig, ax = plt.subplots(1, 3, figsize = (18, 5))
		for kk in xrange(len(x_data)):
		
			ax[kk].errorbar(x_data[kk],y_data[kk], 
				        fmt='bo', ecolor='0.20', alpha=0.8,
				        linestyle=' ')

			ax[kk].set_xlabel(x_labels[kk])
			ax[kk].set_ylabel(y_labels[kk])

			# set plot limits to be slightly outside max values
			dynx, dyny = (np.nanmax(x_data[kk])-np.nanmin(x_data[kk]))*0.05,\
			         	 (np.nanmax(y_data[kk])-np.nanmin(y_data[kk]))*0.05
			
			ax[kk].axis((np.nanmin(x_data[kk])-dynx,
					 np.nanmax(x_data[kk])+dynx,
					 np.nanmin(y_data[kk])-dyny,
					 np.nanmax(y_data[kk])+dyny,
					 ))
			if np.nanmin(x_data[kk])-dynx > np.nanmin(y_data[kk])-dyny:
				min = np.nanmin(y_data[kk])-dyny*3
			else:
				min = np.nanmin(x_data[kk])-dynx*3
			if np.nanmax(x_data[kk])+dynx > np.nanmax(y_data[kk])+dyny:
				max = np.nanmax(x_data[kk])+dynx*3
			else:
				max = np.nanmax(y_data[kk])+dyny*3

			ax[kk].plot([-1e3,1e3],[-1e3,1e3],linestyle='--',color='0.1',alpha=0.8)
			ax[kk].axis((min,max,min,max))

			ratio = (10**x_data[kk])/(10**y_data[kk])
			#ax[kk].text(min+(max-min)*0.05,min+(max-min)*0.95, 'std(x/y)={0}'.format(ratio.std()))
			#ax[kk].text(min+(max-min)*0.05,min+(max-min)*0.90, 'mean(x/y)={0}'.format(ratio.mean()))

		fig.subplots_adjust(wspace=0.30,hspace=0.0)
		plt.savefig(outname_errs+'sfr'+sfr_str[bobo]+'.png',dpi=300)
		plt.close()


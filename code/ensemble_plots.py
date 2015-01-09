import os, threed_dutils, triangle, pickle
from bsfh import read_results
import numpy as np
import matplotlib.pyplot as plt

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

		# initialize output arrays if necessary
		nnew = 3
		ntheta = len(sample_results['initial_theta'])+nnew
		if jj == 0:
			q_16, q_50, q_84, thetamax = (np.zeros(shape=(ntheta,ngals))+np.nan for i in range(4))
			z, mips_sn = (np.zeros(ngals) for i in range(2))
			output_name = np.empty(0,dtype='object')

		# chop and thin the chain
		flatchain = threed_dutils.chop_chain(sample_results)

		# calculate extra quantities, add to chain
		half_time,sfr_100 = read_results.calc_extra_quantities(sample_results,flatchain)
		ssfr_100 = np.array([x/sfr_100[nn] for nn,x in enumerate(flatchain[:,np.array(sample_results['model'].theta_labels()) == 'mass'])])
		chainlen = flatchain.shape[0]
		flatchain = np.concatenate((flatchain,
			                        half_time.reshape(chainlen,1),
			                        sfr_100.reshape(chainlen,1),
									ssfr_100), axis=1)

		# insert percentiles
		for kk in xrange(ntheta): q_16[kk,jj], q_50[kk,jj], q_84[kk,jj] = triangle.quantile(flatchain[:,kk], [0.16, 0.5, 0.84])
		
		# miscellaneous output
		z[jj] = sample_results['model_params'][0]['init'][0]
		mips_sn[jj] = sample_results['obs']['maggies'][-1]/sample_results['obs']['maggies_unc'][-1]
		output_name=np.append(output_name,filename)

		# grab best-fitting model
		maxprob = np.max(sample_results['lnprobability'])
		probind = sample_results['lnprobability'] == maxprob
		thetas = sample_results['chain'][probind,:]

		# find maximum probability model in old chain
		# search for match in new chain
		maxhalf_time,maxsfr_100 = read_results.calc_extra_quantities(sample_results,thetas[0,:].reshape(1,ntheta-nnew))
		maxssfr_100 = thetas[0,np.array(sample_results['model'].theta_labels()) == 'mass']/maxsfr_100
		thetamax[:,jj] = np.concatenate((thetas[0,:],maxhalf_time,maxsfr_100,maxssfr_100))

		print jj

	print 'total galaxies: {0}, successful loads: {1}'.format(ngals,ngals-nfail)
	print 'saving in {0}'.format(outname)

	output = {'outname': output_name,\
			  'parname': np.array(sample_results['model'].theta_labels() + ['half_time','sfr','ssfr']),\
		      'q16': q_16,\
		      'q50': q_50,\
		      'q84': q_84,\
		      'maxprob': thetamax,\
		      'mips_sn': mips_sn,\
		      'z':z}

	pickle.dump(output,open(outname, "wb"))
		
def plot_driver():
	
	runname = "neboff"
	runname = 'photerr'

	outname = os.getenv('APPS')+'/threedhst_bsfh/results/'+runname+'/'+runname+'_ensemble.pickle'

	# if the save file doesn't exist, make it
	if not os.path.isfile(outname):
		collate_output(runname,outname)

	with open(outname, "rb") as f:
		ensemble=pickle.load(f)
	
	# get SFR_observed
	ngals = len(ensemble['outname'])
	sfr_obs = np.zeros(ngals)
	z_sfr = np.zeros(ngals)
	for jj in xrange(ngals):
		ancildat = threed_dutils.load_ancil_data(os.getenv('APPS')+
			                       '/threedhst_bsfh/data/COSMOS_testsamp.dat',
			                       ensemble['outname'][jj].split('_')[-1])
		sfr_obs[jj] = ancildat['sfr']
		z_sfr[jj] = ancildat['z_sfr']
	valid_comp = ensemble['z'] > 0

	if np.sum(z_sfr-ensemble['z'][valid_comp]) != 0:
		print "you got some redshift mismatches yo"

	x_data = [np.log10(ensemble['q50'][ensemble['parname'] == 'mass'][0]),\
	          ensemble['q84'][ensemble['parname'] == 'dust_index'][0]-ensemble['q16'][ensemble['parname'] == 'dust_index'][0],\
	          ensemble['q84'][ensemble['parname'] == 'dust_index'][0]-ensemble['q16'][ensemble['parname'] == 'dust_index'][0],\
	          ensemble['mips_sn'],\
	          np.log10(ensemble['q50'][ensemble['parname'] == 'sfr'][0]),\
	          np.log10(ensemble['q50'][ensemble['parname'] == 'sfr'][0]),\
	          np.log10(ensemble['q50'][ensemble['parname'] == 'sfr'][0]),\
	          np.log10(ensemble['q50'][ensemble['parname'] == 'mass'][0]),\
	          np.log10(ensemble['q50'][ensemble['parname'] == 'ssfr'][0]),\
	          np.log10(sfr_obs),\
	          ]

	x_err  = [asym_errors(ensemble['q50'][ensemble['parname'] == 'mass'][0],ensemble['q84'][ensemble['parname'] == 'mass'][0], ensemble['q16'][ensemble['parname'] == 'mass'][0],log=True),\
			  None,\
			  None,\
			  None,\
			  asym_errors(ensemble['q50'][ensemble['parname'] == 'sfr'][0],ensemble['q84'][ensemble['parname'] == 'sfr'][0],ensemble['q16'][ensemble['parname'] == 'sfr'][0],log=True),\
			  asym_errors(ensemble['q50'][ensemble['parname'] == 'sfr'][0],ensemble['q84'][ensemble['parname'] == 'sfr'][0],ensemble['q16'][ensemble['parname'] == 'sfr'][0],log=True),\
			  asym_errors(ensemble['q50'][ensemble['parname'] == 'sfr'][0],ensemble['q84'][ensemble['parname'] == 'sfr'][0],ensemble['q16'][ensemble['parname'] == 'sfr'][0],log=True),\
			  asym_errors(ensemble['q50'][ensemble['parname'] == 'mass'][0],ensemble['q84'][ensemble['parname'] == 'mass'][0],ensemble['q16'][ensemble['parname'] == 'mass'][0],log=True),\
			  asym_errors(ensemble['q50'][ensemble['parname'] == 'ssfr'][0],ensemble['q84'][ensemble['parname'] == 'ssfr'][0],ensemble['q16'][ensemble['parname'] == 'sfr'][0],log=True),\
			  None,\
			 ]

	x_labels = [r'log(M) [M$_{\odot}$]',
	            r'$\sigma$ (dust index)',
	            r'$\sigma$ (dust index)',
	            'MIPS S/N',
	            r'log(SFR) [M$_{\odot}$/yr]',
	            r'log(SFR) [M$_{\odot}$/yr]',
	            r'log(SFR) [M$_{\odot}$/yr]',
	            r'log(M) [M$_{\odot}$]',
	            r'log(sSFR) [yr$^{-1}$]',
	            r'log(SFR$_{obs}$) [M$_{\odot}$/yr]',
	            ]

	y_data = [ensemble['q50'][ensemble['parname'] == 'logzsol'][0],\
	          ensemble['q84'][ensemble['parname'] == 'dust1'][0]-ensemble['q16'][ensemble['parname'] == 'dust1'][0],\
	          ensemble['q84'][ensemble['parname'] == 'dust2'][0]-ensemble['q16'][ensemble['parname'] == 'dust2'][0],\
	          ensemble['q84'][ensemble['parname'] == 'duste_qpah'][0]-ensemble['q16'][ensemble['parname'] == 'duste_qpah'][0],\
	          ensemble['q84'][ensemble['parname'] == 'half_time'][0]-ensemble['q16'][ensemble['parname'] == 'half_time'][0],\
	          ensemble['q50'][ensemble['parname'] == 'half_time'][0],\
	          ensemble['q84'][ensemble['parname'] == 'dust1'][0]-ensemble['q16'][ensemble['parname'] == 'dust1'][0],\
	          np.log10(ensemble['q50'][ensemble['parname'] == 'sfr'][0]),\
	          ensemble['q84'][ensemble['parname'] == 'half_time'][0]-ensemble['q16'][ensemble['parname'] == 'half_time'][0],\
	          np.log10(ensemble['q50'][ensemble['parname'] == 'sfr'][0,valid_comp]),\
	          ]

	y_err  = [asym_errors(ensemble['q50'][ensemble['parname'] == 'logzsol'][0],ensemble['q84'][ensemble['parname'] == 'logzsol'][0],ensemble['q16'][ensemble['parname'] == 'logzsol'][0]),\
			  None,\
			  None,\
			  None,\
			  None,\
			  asym_errors(ensemble['q50'][ensemble['parname'] == 'half_time'][0],ensemble['q84'][ensemble['parname'] == 'half_time'][0],ensemble['q16'][ensemble['parname'] == 'half_time'][0]),\
			  None,\
			  asym_errors(ensemble['q50'][ensemble['parname'] == 'sfr'][0],ensemble['q84'][ensemble['parname'] == 'sfr'][0],ensemble['q16'][ensemble['parname'] == 'sfr'][0],log=True),\
			  None,\
			  asym_errors(ensemble['q50'][ensemble['parname'] == 'sfr'][0,valid_comp],ensemble['q84'][ensemble['parname'] == 'sfr'][0,valid_comp],ensemble['q16'][ensemble['parname'] == 'sfr'][0,valid_comp],log=True),\
			 ]

	y_labels = [r'log(Z$_{\odot}$)',
	            r'$\sigma$ (dust1)',
	            r'$\sigma$ (dust2)',
	            r'$\sigma$ (q$_{pah})$',
	            r'$\sigma$ (t$_{half}$) [Gyr]',
	            r't$_{half}$ [Gyr]',
	            r'$\sigma$ (dust1)',
	            r'log(SFR) [M$_{\odot}$/yr]',
	            r'$\sigma$ (t$_{half}$) [Gyr]',
	            r'log(SFR$_{mcmc}$) [M$_{\odot}$/yr]',
	            ]

	plotname = ['mass_metallicity',
				'deltadustindex_deltadust1',
				'deltadustindex_deltadust2',
				'mipssn_deltaqpah',
				'sfr_deltahalftime',
				'sfr_halftime',
				'sfr_deltadust1',
				'mass_sfr',
				'ssfr_deltahalftime',
				'sfrobs_sfrmcmc']

	assert len(plotname) == len(y_labels) == len(y_err) == len(y_data), 'improper number of y data'
	assert len(plotname) == len(x_labels) == len(x_err) == len(x_data), 'improper number of x data'

	for jj in xrange(len(x_data)):
		outname = os.getenv('APPS') + '/threedhst_bsfh/plots/ensemble_plots/'+plotname[jj]+'.png'
		
		fig = plt.figure()
		ax = fig.add_subplot(111)

		try:
			print len(x_data[jj]),len(y_data[jj]), len(x_err[jj]), len(y_err[jj])
		except:
			print 'None type'
		
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

		plt.savefig(outname, dpi=300)
		plt.close()


	
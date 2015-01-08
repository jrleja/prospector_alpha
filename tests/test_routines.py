import fsps,os,time,pylab
from bsfh import read_results,model_setup
import matplotlib.pyplot as plt
import numpy as np

c =2.99792458e8

''' TEST PLOT PLEASE IGNORE '''
def sed_test_plot():
	
	"""
	Plot the photometry+spectra for a variety of ages, etc
	"""
			
	sps = fsps.StellarPopulation(zcontinuous=2, compute_vega_mags=False)
	custom_filter_keys = os.getenv('APPS')+'/threedhst_bsfh/filters/filter_keys_threedhst.txt'
	fsps.filters.FILTERS = model_setup.custom_filter_dict(custom_filter_keys)

	# load custom model
	model = model_setup.setup_model(os.getenv('APPS')+'/threedhst_bsfh/threedhst_params.py', sps=sps)
	
	# setup figure
	fig, axarr = plt.subplots(2, 2, figsize = (8,8))
	fig.subplots_adjust(wspace=0.000,hspace=0.000)
	fast_lam = model.obs['wave_effective']
	init_theta = np.array([10.5, 0, 0, 0.0])
	
	# generate colors
	import pylab
	NUM_COLORS = 10
	cm = pylab.get_cmap('cool')
	plt.rcParams['axes.color_cycle'] = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)] 
	axlim=[3.0,5.5,3.0,8]
	
	# setup delta
	delta = [np.linspace(init_theta[0]-1,init_theta[0]+1, NUM_COLORS),
			 np.linspace(init_theta[1]-0.9,init_theta[1]+1, NUM_COLORS),
			 np.linspace(init_theta[2]-0.9,init_theta[2]+1, NUM_COLORS),
			 np.linspace(init_theta[3]-1,init_theta[3]+0.3, NUM_COLORS)]

	for kk in range(4):
		itone = kk % 2
		ittwo = kk > 1
	
		for jj in range(len(delta[kk])):

			# set model parms
			model_params = np.copy(init_theta)
			model_params[kk] = delta[kk][jj]

			# load data
			observables = model.mean_model(10**model_params, sps=sps)
			fast_spec, fast_mags = observables[0],observables[1]
			w, spec_throwaway = sps.get_spectrum(tage=sps.params['tage'], peraa=False)
    
			nz = fast_spec > 0
			axarr[itone,ittwo].plot(np.log10(w[nz]), np.log10(fast_spec[nz]*(c/(w[nz]/1e10))),label = "{:10.2f}".format(model_params[kk]))
	
		# beautify
		if itone == 1:
			axarr[itone,ittwo].set_xlabel('log(lam)')
		else:
			axarr[itone,ittwo].set_xticklabels([])
			axarr[itone,ittwo].yaxis.get_major_ticks()[0].label1On = False # turn off bottom ticklabel
	
		if ittwo == 0:
			axarr[itone,ittwo].set_ylabel('log(nu*fnu)')
		else:
			axarr[itone,ittwo].set_yticklabels([])
			axarr[itone,ittwo].xaxis.get_major_ticks()[0].label1On = False # turn off bottom ticklabel
	
		axarr[itone,ittwo].legend(loc=0,prop={'size':6},
								  frameon=False,
								  title='log('+str(model.theta_labels()[kk])+')')
		axarr[itone,ittwo].get_legend().get_title().set_size(8)
		axarr[itone,ittwo].axis(axlim)

def test_getmags():

	'''
	run-time test of getmags
	'''

	parm_file='/Users/joel/code/python/threedhst_bsfh/parameter_files/halpha_selected_params.py'

	# load stellar population, set up custom filters
	sps = fsps.StellarPopulation(zcontinuous=2, compute_vega_mags=False)
	custom_filter_keys = os.getenv('APPS')+'/threedhst_bsfh/filters/filter_keys_threedhst.txt'
	fsps.filters.FILTERS = model_setup.custom_filter_dict(custom_filter_keys)

	# load parameter file
	model = model_setup.setup_model(parm_file, sps=sps)

	# test
	ntest=10
	tot_time=0.0
	for jj in xrange(ntest):
		d1 = time.time()
		mu, phot, x = model.mean_model(model.initial_theta, sps = sps)
		print len(mu)
		f1 = time.time()-d1
		print f1
		tot_time = tot_time+f1
	print tot_time/ntest
	
def mips_flux_err(field='COSMOS'):

	dataloc = os.getenv('APPS')+'/threedhst_bsfh/data/MIPS/'+field.lower()+'_3dhst.v4.1.4.sfr'
	
	with open(dataloc, 'r') as f:
		for jj in range(1): hdr = f.readline().split()
	dat = np.loadtxt(dataloc, comments = '#',dtype = np.dtype([(n, np.float) for n in hdr[1:]]))
	good = dat['f24tot'] > 0
	print np.sum(good)*1.0/len(dat)
	dat = dat[good]

	fig, ax = plt.subplots(1)
	ax.plot(np.log10(dat['f24tot']), np.log10(dat['ef24tot']),
	        'o',linestyle=' ',alpha=0.3,ms=2,markeredgewidth=0.0)
	ax.axis([-1,3.5,0.9,1.75])
	ax.set_ylabel(r'log(err$_{MIPS-24}$)')
	ax.set_xlabel(r'log(flux$_{MIPS-24}$)')
	ax.plot(np.linspace(-50,50),np.linspace(-50,50),linestyle='--',color='black')
	
	plt.savefig(os.getenv('APPS')+'/threedhst_bsfh/tests/testfigs/mips_flux_err.png', bbox_inches='tight',dpi=300)
	plt.close()

def dust_emission_test():
	
	# setup stellar populations
	sps = fsps.StellarPopulation(zcontinuous=2, compute_vega_mags=False)
	custom_filter_keys = os.getenv('APPS')+'/threedhst_bsfh/filters/filter_keys_threedhst.txt'
	fsps.filters.FILTERS = model_setup.custom_filter_dict(custom_filter_keys)

	# load custom model
	model = model_setup.setup_model(os.getenv('APPS')+'/threedhst_bsfh/threedhst_params.py', sps=sps)

	# setup plot
	fig, axarr = plt.subplots(2, 2, figsize = (8,8))
	fig.subplots_adjust(wspace=0.000,hspace=0.000)
	
	# generate colors
	npoints = 10
	cm = pylab.get_cmap('cool')
	plt.rcParams['axes.color_cycle'] = [cm(1.*i/npoints) for i in range(npoints)] 
	axlim=[4.8,7.4,7.5,9.0]
	
	# set up delta
	params_to_vary = ['duste_gamma', 'duste_umin', 'duste_qpah', 'tau']
	vary_in_log    = [True, True, False, True]
	for jj in xrange(len(params_to_vary)):
		
		# set plot parameters
		itone = jj % 2
		ittwo = jj > 1
		
		# extract min/max
		priors=[x['prior_args'] for x in model.config_list if x['name'] == params_to_vary[jj]][0]
		
		# set values in linear or logspace
		if vary_in_log[jj] == True:
			param_values = 10**np.linspace(np.log10(np.array(priors['mini']).clip(min=0.01)),
			                               np.log10(priors['maxi']),
			                               npoints)
		else:
			param_values = np.linspace(priors['mini'],
			                           priors['maxi'], 
			                           npoints)
		
		if params_to_vary[jj] in model.free_params:
			parmindex = model.theta_labels().index(params_to_vary[jj])
			save_initial = model.initial_theta[parmindex]
		elif params_to_vary[jj] in model.fixed_params:
			save_initial = model.params[params_to_vary[jj]]
		else:
			print 'ERROR:' +params_to_vary[jj]+' is not a parameter!'
			sys.exit()
		
		for kk in xrange(npoints):
			
			# set model parameters
			if params_to_vary[jj] in model.free_params:
				model.initial_theta[parmindex] = param_values[kk]
			else:
				model.params[params_to_vary[jj]]=np.array([param_values[kk]])
			
			# load and plot data
			spec,mags,w = model.mean_model(model.initial_theta, sps=sps)
    
			nz = spec > 0
			axarr[itone,ittwo].plot(np.log10(w[nz]), 
			                        np.log10(spec[nz]*(c/(w[nz]/1e10))),
			                        label = "{:10.2f}".format(param_values[kk]),
			                        linewidth=0.4)
		
		if params_to_vary[jj] in model.free_params:
			model.initial_theta[parmindex] = save_initial
		else:
			model.params[params_to_vary[jj]] = save_initial
		
		# beautify
		if itone == 1:
			axarr[itone,ittwo].set_xlabel(r'log($\lambda$) [$\AA$]')
		else:
			axarr[itone,ittwo].set_xticklabels([])
			axarr[itone,ittwo].yaxis.get_major_ticks()[0].label1On = False # turn off bottom ticklabel
	
		if ittwo == 0:
			axarr[itone,ittwo].set_ylabel(r'log($\nu f_{\nu}$)')
		else:
			axarr[itone,ittwo].set_yticklabels([])
		
		# add horizontal line	
		axarr[itone,ittwo].axvline(x=np.log10(24*1e4), linestyle='--', color='grey',alpha=0.6)

		# make nice legend
		axarr[itone,ittwo].legend(loc=1,prop={'size':6},
								  frameon=False,
								  title=params_to_vary[jj])
		axarr[itone,ittwo].get_legend().get_title().set_size(8)
		axarr[itone,ittwo].axis(axlim)
	outname=os.getenv('APPS')+'/threedhst_bsfh/tests/testfigs/dustem_params.png'
	plt.savefig(outname, bbox_inches='tight',dpi=300)
	os.system('open '+outname)

















	
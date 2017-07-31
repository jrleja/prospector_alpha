import fsps,os,time,pylab
import prosp_dutils
from bsfh import read_results,model_setup
import matplotlib.pyplot as plt
import numpy as np

c =2.99792458e8

def simha_sfh_test():
	
	# test the SFH=5 prescription
	# setup stellar populations
	sps = prosp_dutils.setup_sps(zcontinuous=2)

	# load custom model
	param_file=os.getenv('APPS')+'/prospector_alpha/parameter_files/testsed_simha/testsed_simha_params.py'
	model = model_setup.load_model(param_file=param_file, sps=sps)
	obs = model_setup.load_obs(param_file=param_file)

	# generate colors
	npoints = 8
	cm = pylab.get_cmap('cool')
	plt.rcParams['axes.color_cycle'] = [cm(1.*i/npoints) for i in range(npoints)] 
	axlim=[3.5,6.0,2.0,6.5]
	
	# set model information
	model.params['tau'] = np.array(10.0)
	model.params['tage'] = np.array(8.0)
	parmindex_sftheta = model.theta_labels().index('sf_theta')
	parmindex_sftrunc = model.theta_labels().index('sf_slope')
	model.initial_theta[parmindex_sftheta] = -5.0

	# set up delta
	sf_slope = [1.5, 5.0, 6.0, 6.5, 7.0, 7.5, 8.0, 10.0]
	sf_theta = np.linspace(-np.pi/3,np.pi/2,8)
	fig, ax = plt.subplots(2, 1, figsize = (6,12))

	# here we iterate over "param_iterable"
	for ll in xrange(len(sf_trunc)):

		model.initial_theta[parmindex_sftrunc] = sf_slope[ll]

		# load and plot data
		spec,mags,w = model.mean_model(model.initial_theta, obs, sps=sps)

		nz = spec > 0
		ax[0].plot(np.log10(w[nz]), 
		        np.log10(spec[nz]*(c/(w[nz]/1e10))),
		        label = "{:10.2f}".format(sf_trunc[ll]),
		        linewidth=0.6)

	# make nice legend
	ax[0].legend(loc=4,prop={'size':6},
			  frameon=False,
			  title='sf_trunc')
	ax[0].get_legend().get_title().set_size(8)
	ax[0].axis(axlim)
	ax[0].text(0.98,0.9,'tau='+"{:.2f}".format(model.params['tau'][0]),horizontalalignment='right',transform = ax[0].transAxes)
	ax[0].text(0.98,0.85,'tage='+"{:.2f}".format(float(model.params['tage'])),horizontalalignment='right',transform = ax[0].transAxes)
	ax[0].text(0.98,0.8,'sf_theta='+"{:.2f}".format(model.initial_theta[parmindex_sftheta]),horizontalalignment='right',transform = ax[0].transAxes)

	ax[0].set_xlabel(r'log($\lambda$) [$\AA$]')
	ax[0].set_ylabel(r'log($\nu f_{\nu}$)')
	model.initial_theta[parmindex_sftrunc] = 6.0

	# here we iterate over "param_iterable"
	for ll in xrange(len(sf_theta)):

		model.initial_theta[parmindex_sftheta] = sf_theta[ll]

		# load and plot data
		spec,mags,w = model.mean_model(model.initial_theta, obs, sps=sps)

		nz = spec > 0
		ax[1].plot(np.log10(w[nz]), 
		        np.log10(spec[nz]*(c/(w[nz]/1e10))),
		        label = "{:10.2f}".format(sf_theta[ll]),
		        linewidth=0.6)

	# make nice legend
	ax[1].legend(loc=4,prop={'size':6},
			  frameon=False,
			  title='sf_theta')
	ax[1].get_legend().get_title().set_size(8)
	ax[1].axis(axlim)
	ax[1].text(0.98,0.9,'tau='+"{:.2f}".format(model.params['tau'][0]),horizontalalignment='right',transform = ax[1].transAxes)
	ax[1].text(0.98,0.85,'tage='+"{:.2f}".format(float(model.params['tage'])),horizontalalignment='right',transform = ax[1].transAxes)
	ax[1].text(0.98,0.8,'sf_trunc='+"{:.2f}".format(model.initial_theta[parmindex_sftrunc]),horizontalalignment='right',transform = ax[1].transAxes)

	ax[1].set_xlabel(r'log($\lambda$) [$\AA$]')
	ax[1].set_ylabel(r'log($\nu f_{\nu}$)')

	outname=os.getenv('APPS')+'/prospector_alpha/tests/testfigs/simha_sfh_test.png'
	plt.savefig(outname, bbox_inches='tight',dpi=500)
	plt.close()
	os.system('open '+outname)
	print 1/0


def new_metals_test():
	
	# setup stellar populations
	sps = fsps.StellarPopulation(zcontinuous=2, compute_vega_mags=False)
	custom_filter_keys = os.getenv('APPS')+'/prospector_alpha/filters/filter_keys_threedhst.txt'
	fsps.filters.FILTERS = model_setup.custom_filter_dict(custom_filter_keys)

	# load custom model
	param_file=os.getenv('APPS')+'/prospector_alpha/parameter_files/testsed_nonoise_fast/testsed_nonoise_fast_params.py'
	model = model_setup.load_model(param_file=param_file, sps=sps)
	obs = model_setup.load_obs(param_file=param_file)
	
	# set up delta
	logzsol = np.array([-1.9,-1.1,-0.3,0.19])
	fig, ax = plt.subplots(1, 1, figsize = (4,4))

	# generate colors
	npoints = len(logzsol)
	cm = pylab.get_cmap('cool')
	plt.rcParams['axes.color_cycle'] = [cm(1.*i/npoints) for i in range(npoints)] 
	axlim=[3.0,5.0,3.6,6.1]

	# here we iterate over "param_iterable"
	for ll in xrange(len(logzsol)):

		parmindex = model.theta_labels().index('logzsol')
		model.initial_theta[parmindex] = logzsol[ll]

		# load and plot data
		spec,mags,w = model.mean_model(model.initial_theta, obs, sps=sps)

		nz = spec > 0
		ax.plot(np.log10(w[nz]), 
		        np.log10(spec[nz]*(c/(w[nz]/1e10))),
		        label = "{:10.2f}".format(logzsol[ll]),
		        linewidth=0.8)

	# make nice legend
	ax.legend(loc=8,prop={'size':12},
			  frameon=False,
			  title='logzsol')
	#ax.get_legend().get_title().set_size(12)
	ax.axis(axlim)

	ax.set_xlabel(r'log($\lambda$) [$\AA$]')
	ax.set_ylabel(r'log($\nu f_{\nu}$)')

	outname=os.getenv('APPS')+'/prospector_alpha/tests/testfigs/new_metals.png'
	plt.savefig(outname, bbox_inches='tight',dpi=500)
	plt.close()
	os.system('open '+outname)

def interp_metals():
	
	# INTERPOLATED METALLICITY
	# setup stellar populations
	sps = fsps.StellarPopulation(zcontinuous=1, compute_vega_mags=False)
	custom_filter_keys = os.getenv('APPS')+'/prospector_alpha/filters/filter_keys_threedhst.txt'
	fsps.filters.FILTERS = model_setup.custom_filter_dict(custom_filter_keys)

	# load custom model
	param_file=os.getenv('APPS')+'/prospector_alpha/parameter_files/dtau_nebon/dtau_nebon_params.py'
	model = model_setup.load_model(param_file=param_file, sps=sps)
	obs = model_setup.load_obs(param_file=param_file)

	# generate colors
	npoints = 8
	cm = pylab.get_cmap('cool')
	plt.rcParams['axes.color_cycle'] = [cm(1.*i/npoints) for i in range(npoints)] 
	axlim=[2.2,5.0,0,7.0]
	
	# set up delta
	logzsol = np.array([-1.0,-0.8,-0.6,-0.4,-0.25,-0.1,0.00,0.15])
	fig, ax = plt.subplots(1, 1, figsize = (8,8))

	# here we iterate over "param_iterable"
	for ll in xrange(len(logzsol)):

		parmindex = model.theta_labels().index('logzsol')
		model.initial_theta[parmindex] = logzsol[ll]

		# load and plot data
		spec,mags,w = model.mean_model(model.initial_theta, obs, sps=sps)

		nz = spec > 0
		ax.plot(np.log10(w[nz]), 
		        np.log10(spec[nz]*(c/(w[nz]/1e10))),
		        label = "{:10.2f}".format(logzsol[ll]),
		        linewidth=0.6, alpha=0.3)

	# make nice legend
	ax.legend(loc=4,prop={'size':6},
			  frameon=False,
			  title='logzsol')
	ax.get_legend().get_title().set_size(8)
	ax.axis(axlim)

	# USE PMETALS
	# setup stellar populations
	sps = fsps.StellarPopulation(zcontinuous=2, compute_vega_mags=False)
	custom_filter_keys = os.getenv('APPS')+'/prospector_alpha/filters/filter_keys_threedhst.txt'
	fsps.filters.FILTERS = model_setup.custom_filter_dict(custom_filter_keys)

	# load custom model
	param_file=os.getenv('APPS')+'/prospector_alpha/parameter_files/dtau_nebon/dtau_nebon_params.py'
	model = model_setup.load_model(param_file=param_file, sps=sps)
	obs = model_setup.load_obs(param_file=param_file)

	# generate colors
	npoints = 8
	cm = pylab.get_cmap('cool')
	plt.rcParams['axes.color_cycle'] = [cm(1.*i/npoints) for i in range(npoints)] 
	model.params['pmetals']=np.array(40)

	# here we iterate over "param_iterable"
	for ll in xrange(len(logzsol)):

		parmindex = model.theta_labels().index('logzsol')
		model.initial_theta[parmindex] = logzsol[ll]

		# load and plot data
		spec,mags,w = model.mean_model(model.initial_theta, obs, sps=sps)

		nz = spec > 0
		ax.plot(np.log10(w[nz]), 
		        np.log10(spec[nz]*(c/(w[nz]/1e10))),
		        label = "{:10.2f}".format(logzsol[ll]),
		        linewidth=0.6,linestyle='--')
		



	ax.set_xlabel(r'log($\lambda$) [$\AA$]')
	ax.set_ylabel(r'log($\nu f_{\nu}$)')

	outname=os.getenv('APPS')+'/prospector_alpha/tests/testfigs/interp_metals.png'
	plt.savefig(outname, bbox_inches='tight',dpi=500)
	plt.close()
	os.system('open '+outname)



def test_metals():
	
	# setup stellar populations
	sps = fsps.StellarPopulation(zcontinuous=2, compute_vega_mags=False)
	custom_filter_keys = os.getenv('APPS')+'/prospector_alpha/filters/filter_keys_threedhst.txt'
	fsps.filters.FILTERS = model_setup.custom_filter_dict(custom_filter_keys)

	# load custom model
	param_file=os.getenv('APPS')+'/prospector_alpha/parameter_files/testsed_nonoise_fast/testsed_nonoise_fast_params.py'
	model = model_setup.load_model(param_file=param_file, sps=sps)
	obs = model_setup.load_obs(param_file=param_file)
	
	# fixed interpolated metallicity
	sps_fixed = fsps.StellarPopulation(zcontinuous=1, compute_vega_mags=False)
	fsps.filters.FILTERS = model_setup.custom_filter_dict(custom_filter_keys)
	model_fixed = model_setup.load_model(param_file=param_file, sps=sps_fixed)
	metals_index = model_fixed.theta_labels().index('logzsol')
	model_fixed.initial_theta[metals_index] = -0.25
	spec_solar,mags_solar,w_solar = model_fixed.mean_model(model_fixed.initial_theta, obs, sps=sps_fixed)
	nz_solar = spec_solar > 0

	# generate colors
	npoints = 8
	cm = pylab.get_cmap('cool')
	plt.rcParams['axes.color_cycle'] = [cm(1.*i/npoints) for i in range(npoints)] 
	axlim=[2.2,5.0,0,7.0]
	
	# set up delta
	pmetals = np.array([0.5,1,2,3,4,5,10,100])
	logzsol = np.array([-1.0,-0.8,-0.6,-0.4,-0.25,-0.1,0.00,0.15])
	param_iterablename= ['logzsol','pmetals']
	
	# make two plots, one for pmetals, one for logzsol
	for jj in xrange(len(param_iterablename)):
		
		# setup plot
		fig, axarr = plt.subplots(2, 4, figsize = (12,6))
		fig.subplots_adjust(wspace=0.000,hspace=0.000)

		# setup parameter to vary
		if param_iterablename[jj] == 'pmetals':
			param_iterable = pmetals
			param_fixed = logzsol
			param_fixedname = 'logzsol'
		else:
			param_iterable = logzsol
			param_fixed = pmetals
			param_fixedname = 'pmetals'
			#model.params['pmetals'] = pmetals[jj]			
		
		# for the fixed parameter in each plot
		# here we iterate over "param_fixed"
		for kk in xrange(len(param_fixed)):
			
			# set plot parameters
			itone = kk / 4
			ittwo = kk % 4

			# set model parameters for fixed parameter
			if param_fixedname in model.free_params:
				parmindex = model.theta_labels().index(param_fixedname)
				model.initial_theta[parmindex] = param_fixed[kk]
			else:
				model.params[param_fixedname]=np.array([param_fixed[kk]])
			
			# here we iterate over "param_iterable"
			for ll in xrange(len(param_iterable)):

				# set model parameters for fixed parameter
				if param_iterablename[jj] in model.free_params:
					parmindex = model.theta_labels().index(param_iterablename[jj])
					model.initial_theta[parmindex] = param_iterable[ll]
				else:
					model.params[param_iterablename[jj]]=np.array([param_iterable[ll]])

				# load and plot data
				spec,mags,w = model.mean_model(model.initial_theta, obs, sps=sps)
	    
				nz = spec > 0
				axarr[itone,ittwo].plot(np.log10(w[nz]), 
				                        np.log10(spec[nz]*(c/(w[nz]/1e10))),
				                        label = "{:10.2f}".format(param_iterable[ll]),
				                        linewidth=0.4)

				axarr[itone,ittwo].plot(np.log10(w_solar[nz]), 
				                        np.log10(spec_solar[nz]*(c/(w_solar[nz]/1e10))),
				                        linewidth=0.6,
				                        color='k',
				                        linestyle='--')
		
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
			axarr[itone,ittwo].legend(loc=4,prop={'size':6},
									  frameon=False,
									  title=param_iterablename[jj])
			axarr[itone,ittwo].get_legend().get_title().set_size(8)
			axarr[itone,ittwo].axis(axlim)
			axarr[itone,ittwo].text(2.3,6.5, param_fixedname+'='+"{:.2f}".format(param_fixed[kk]),fontsize=8)
	
		outname=os.getenv('APPS')+'/prospector_alpha/tests/testfigs/mettest_vary_'+param_iterablename[jj]+'.png'
		plt.savefig(outname, bbox_inches='tight',dpi=500)
		plt.close()
		os.system('open '+outname)































	
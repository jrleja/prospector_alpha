import fsps,os,time,pylab
from bsfh import read_results,model_setup
import matplotlib.pyplot as plt
import numpy as np
pylab.rc('font', family='serif', size=40)
c =2.99792458e8

def uv_ir_sfrs():
	
	# INTERPOLATED METALLICITY
	# setup stellar populations
	sps = fsps.StellarPopulation(zcontinuous=1, compute_vega_mags=False)
	custom_filter_keys = os.getenv('APPS')+'/threedhst_bsfh/filters/filter_keys_threedhst.txt'
	fsps.filters.FILTERS = model_setup.custom_filter_dict(custom_filter_keys)

	# load custom model
	param_file=os.getenv('APPS')+'/threedhst_bsfh/parameter_files/dtau_nebon/dtau_nebon_params.py'
	model = model_setup.load_model(param_file=param_file, sps=sps)
	obs = model_setup.load_obs(param_file=param_file)

	# set second component to zero, make star-forming
	parmindex = model.theta_labels().index('mass_2')
	model.initial_theta[parmindex] = 0.0
	parmindex = model.theta_labels().index('tau_1')
	model.initial_theta[parmindex] = 0.105

	# generate colors
	npoints = 4
	cm = pylab.get_cmap('cool')
	plt.rcParams['axes.color_cycle'] = [cm(1.*i/npoints) for i in range(npoints)] 
	axlim=[2.9,6.3,-2,5.5]
	
	# set up delta
	dust2 = np.array([0.0,0.1,0.4,2.0])
	fig, ax = plt.subplots(1, 1, figsize = (8,8))

	# here we iterate over "param_iterable"
	for ll in xrange(len(dust2)):

		parmindex = model.theta_labels().index('dust2_1')
		model.initial_theta[parmindex] = dust2[ll]

		# load and plot data
		spec,mags,w = model.mean_model(model.initial_theta, obs, sps=sps)

		nz = spec > 0
		ax.plot(np.log10(w[nz]), 
		        np.log10(spec[nz]*(c/(w[nz]/1e10))),
		        label = "{:10.2f}".format(dust2[ll]),
		        linewidth=1.5, alpha=0.9)
	
	ax.vlines(3.08,-2,5.5, linestyle='--',colors='blue',linewidth=1.5,alpha=0.7)
	ax.vlines(3.47,-2,5.5, linestyle='--',colors='blue',linewidth=1.5,alpha=0.7)
	ax.fill_between([3.08,3.47], axlim[2], axlim[3], facecolor='blue', interpolate=True, alpha=0.3, zorder=-32)
	ax.text(3.13,5.2,r'L$_{UV}$',fontsize=20)

	ax.vlines(4.9,-2,5.5, linestyle='--',colors='red',linewidth=1.5,alpha=0.7)
	ax.vlines(6.0,-2,5.5, linestyle='--',colors='red',linewidth=1.5,alpha=0.7)
	ax.fill_between([4.9,6.0], axlim[2], axlim[3], facecolor='red', interpolate=True, alpha=0.3, zorder=-32)
	ax.text(5.35,5.25,r'L$_{IR}$',fontsize=20)


	# make nice legend
	ax.legend(loc=4,prop={'size':16},
			  frameon=False,
			  title='Av')
	ax.get_legend().get_title().set_size(16)
	ax.axis(axlim)

	ax.set_xlabel(r'log($\lambda$) [$\AA$]')
	ax.set_ylabel(r'log($\nu f_{\nu}$)')

	outname=os.getenv('APPS')+'/threedhst_bsfh/tests/testfigs/uv_ir_sfrs.png'
	plt.savefig(outname, bbox_inches='tight',dpi=200)
	plt.close()
	os.system('open '+outname)

def uv_ir_sfrs():
	
	# INTERPOLATED METALLICITY
	# setup stellar populations
	sps = fsps.StellarPopulation(zcontinuous=1, compute_vega_mags=False)
	custom_filter_keys = os.getenv('APPS')+'/threedhst_bsfh/filters/filter_keys_threedhst.txt'
	fsps.filters.FILTERS = model_setup.custom_filter_dict(custom_filter_keys)

	# load custom model
	param_file=os.getenv('APPS')+'/threedhst_bsfh/parameter_files/dtau_nebon/dtau_nebon_params.py'
	model = model_setup.load_model(param_file=param_file, sps=sps)
	obs = model_setup.load_obs(param_file=param_file)

	# set second component to zero, make star-forming
	parmindex = model.theta_labels().index('mass_2')
	model.initial_theta[parmindex] = 0.0
	parmindex = model.theta_labels().index('tau_1')
	model.initial_theta[parmindex] = 0.105
	parmindex = model.theta_labels().index('dust2_1')
	model.initial_theta[parmindex] = 0.4
	
	# set up delta
	fig, ax = plt.subplots(1, 1, figsize = (8,8))

	# load and plot data
	spec,mags,w = model.mean_model(model.initial_theta, obs, sps=sps)

	nz = spec > 0
	ax.plot(np.log10(w[nz]), 
	        np.log10(spec[nz]*(c/(w[nz]/1e10))),
	        color='blue',
	        linewidth=1.5, alpha=0.9)

	# load and plot data
	model.params['add_dust_emission'] = np.array(False)
	spec,mags,w = model.mean_model(model.initial_theta, obs, sps=sps)

	nz = spec > 0
	ax.plot(np.log10(w[nz]), 
	        np.log10(spec[nz]*(c/(w[nz]/1e10))),
	        color='red',
	        linewidth=1.5, alpha=0.9)

	ax.set_xlabel(r'log($\lambda$) [$\AA$]')
	ax.set_ylabel(r'log($\nu f_{\nu}$)')

	ax.axis([3,8,0.5,5.5])

	outname=os.getenv('APPS')+'/threedhst_bsfh/tests/testfigs/duston_dustoff.png'
	plt.savefig(outname, bbox_inches='tight',dpi=200)
	plt.close()
	os.system('open '+outname)
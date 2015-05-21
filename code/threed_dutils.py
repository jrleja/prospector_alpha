import numpy as np
import os, fsps
import matplotlib.pyplot as plt
from bsfh import model_setup
from scipy.interpolate import interp1d
from scipy.integrate import simps
from calc_ml import load_filter_response
from bsfh.likelihood import LikelihoodFunction


def test_likelihood(param_file=None, sps=None, model=None, obs=None, thetas=None):

	'''
	skeleton:
	load up some model, instantiate an sps, and load some observations
	generate spectrum, compare to observations, assess likelihood
	can be run in different environments as a test
	'''

	if param_file is None:
		param_file = os.getenv('APPS')+'/threedhst_bsfh/parameter_files/dtau_intmet/dtau_intmet_params_66.py'

	if sps is None:
		# load stellar population, set up custom filters
		sps = fsps.StellarPopulation(zcontinuous=2, compute_vega_mags=False)
		custom_filter_keys = os.getenv('APPS')+'/threedhst_bsfh/filters/filter_keys_threedhst.txt'
		fsps.filters.FILTERS = model_setup.custom_filter_dict(custom_filter_keys)

	if model is None:
		model = model_setup.load_model(param_file)
	
	if obs is None:
		run_params = model_setup.get_run_params(param_file=param_file)
		obs = model_setup.load_obs(**run_params)

	if thetas is None:
		thetas = np.array(model.initial_theta)

	# setup gp
	from bsfh import gp
	gp_phot = gp.PhotOutlier()
	try:
		s, a, l = model.phot_gp_params(obs=obs)
		print 'phot gp parameters'
		print s, a, l
		gp_phot.kernel = np.array( list(a) + list(l) + [s])
	except(AttributeError):
		#There was no phot_gp_params method
		pass

	likefn = LikelihoodFunction(obs=obs, model=model)
	mu, phot, x = model.mean_model(thetas, obs, sps = sps)
	lnp_phot = likefn.lnlike_phot(phot, obs=obs, gp=gp_phot)
	lnp_prior = model.prior_product(thetas)

	return lnp_phot + lnp_prior

def setup_sps(zcontinuous=2,compute_vega_magnitudes=False):

	'''
	easy way to define an SPS
	'''

	# load stellar population, set up custom filters
	sps = fsps.StellarPopulation(zcontinuous=zcontinuous, compute_vega_mags=compute_vega_magnitudes)
	custom_filter_keys = os.getenv('APPS')+'/threedhst_bsfh/filters/filter_keys_threedhst.txt'
	fsps.filters.FILTERS = model_setup.custom_filter_dict(custom_filter_keys)

	return sps

def synthetic_emlines(mass,sfr,dust1,dust2,dust_index):

	'''
	SFR in Msun/yr
	mass in Msun
	'''

	# wavelength in angstroms
	emlines = np.array(['Halpha','Hbeta','Hgamma','[OIII]', '[NII]','[OII]'])
	lam     = np.array([6563,4861,4341,5007,6583,3727])
	flux    = np.zeros(shape=(len(lam),len(np.atleast_1d(mass))))

	# calculate Halpha luminosity from KS relationship
	# comes out in units of [ergs/s]
	# correct from Chabrier to Salpeter with a factor of 1.7
	flux[0,:] = 1.26e41 * (sfr*1.7)

	# Halpha: Hbeta: Hgamma = 2.8:1.0:0.47 (Miller 1974)
	flux[1,:] = flux[0,:]/2.8
	flux[2,:] = flux[1,:]*0.47

	# [OIII] from Fig 8, http://arxiv.org/pdf/1401.5490v2.pdf
	# assume [OIII] is a singlet for now
	# this calculates [5007], add 4959 manually
	y_ratio=np.array([0.3,0.35,0.5,0.8,1.0,1.3,1.6,2.0,2.7,\
	       3.0,4.10,4.9,4.9,6.2,6.2])
	x_ssfr=np.array([1e-10,2e-10,3e-10,4e-10,5e-10,6e-10,7e-10,9e-10,1.2e-9,\
	       2e-9,3e-9,5e-9,8e-9,1e-8,2e-8])
	ratio=np.interp(1.0/sfr,x_ssfr,y_ratio)

	# 5007/4959 = 2.88
	flux[3,:] = ratio*flux[1,:]*(1+1/2.88)

	# from Leja et al. 2013
	# log[NII/Ha] = -5.36+0.44log(M)
	lnii_ha = -5.36+0.44*np.log10(mass)
	flux[4,:] = (10**lnii_ha)*flux[0,:]

	# [Ha / [OII]] vs [NII] / Ha from Hayashi et al. 2013, fig 6
	# evidence in discussion suggests should add reddening
	# corresponding to extinction of A(Ha) = 0.35
	# also should change with metallicity, oh well
	nii_ha_x = np.array([0.13,0.2,0.3,0.4,0.5])
	ha_oii_y = np.array([1.1,1.3,2.0,2.9,3.6])
	ratio = np.interp(flux[4,:]/flux[0,:],nii_ha_x,ha_oii_y)
	flux[5,:] = (1.0/ratio)*flux[0,:]

	# correct for dust
	# if dust_index == None, use Calzetti
	if dust_index is not None:
		tau2 = ((lam.reshape(len(lam),1)/5500.)**dust_index)*dust2
		tau1 = ((lam.reshape(len(lam),1)/5500.)**dust_index)*dust1
		tautot = tau2+tau1
		flux = flux*np.exp(-tautot)
	else:
		Rv   = 4.05
		klam = 2.659*(-2.156+1.509/(lam/1e4)-0.198/(lam/1e4)**2+0.011/(lam/1e4)**3)+Rv
		A_lam = klam/Rv*dust2

		flux = flux[:,0]*10**(-0.4*A_lam)

	# comes out in ergs/s
	output = {'name': emlines,
	          'lam': lam,
	          'flux': flux}
	return output

def load_truths(truthname,objname,sample_results):

	'''
	loads truths
	generates plotvalues
	'''

	# load truths
	nextra = 2
	truth = np.loadtxt(truthname)
	truths = truth[int(objname)-1,:]
	parnames = np.array(sample_results['model'].theta_labels())

	# create totmass and totsfr        
	mass = truths[np.array([True if 'mass' in x else False for x in parnames])]
	totmass = np.log10(np.sum(mass))

	tau = truths[np.array([True if 'tau' in x else False for x in parnames])]
	sf_start = truths[np.array([True if 'sf_start' in x else False for x in parnames])]
	tage = np.zeros(len(tau))+sample_results['model'].params['tage'][0]
	deltat=0.1
	totsfr=np.log10(integrate_sfh(np.atleast_1d(tage)[0]-deltat,np.atleast_1d(tage)[0],mass,tage,tau,sf_start)*np.sum(mass)/(deltat*1e9))

	# convert truths to plotting parameters
	plot_truths = truths+0.0
	for kk in xrange(len(parnames)):
		# reset age
		if parnames[kk] == 'sf_start' or parnames[kk][:-2] == 'sf_start':
			plot_truths[kk] = sample_results['model'].params['tage'][0]-plot_truths[kk]

		# log parameters
		if parnames[kk] == 'mass' or parnames[kk][:-2] == 'mass' or \
           parnames[kk] == 'tau' or parnames[kk][:-2] == 'tau':

			plot_truths[kk] = np.log10(plot_truths[kk])

    
	truths_dict = {'parnames':parnames,
				   'truths':truths,
				   'plot_truths':plot_truths,
				   'extra_parnames':np.array(['totmass','totsfr']),
				   'extra_truths':np.array([totmass,totsfr])}
	return truths_dict

def running_median(x,y,nbins=10,avg=False):

	bins = np.linspace(x.min(),x.max(), nbins)
	delta = bins[1]-bins[0]
	idx  = np.digitize(x,bins)
	if avg == False:
		running_median = np.array([np.median(y[idx-1==k]) for k in range(nbins)])
	else:
		running_median = np.array([np.mean(y[idx-1==k]) for k in range(nbins)])
	bins = bins-delta/2.

	# remove empty
	empty = np.isnan(running_median) == 1
	running_median[empty] = 0

	return bins,running_median

def generate_basenames(runname):

	filebase=[]
	parm=[]
	ancilname='COSMOS_testsamp.dat'

	if runname == 'testsed_nonoise':

		id_list = os.getenv('APPS')+"/threedhst_bsfh/data/testsed_nonoise.ids"
		ids = np.loadtxt(id_list, dtype='|S20')
		ngals = len(ids)

		basename = "testsed_nonoise"
		parm_basename = "testsed_nonoise_params"
		ancilname=None

		for jj in xrange(ngals):
			filebase.append(os.getenv('APPS')+"/threedhst_bsfh/results/"+runname+'/'+basename+'_'+ids[jj])
			parm.append(os.getenv('APPS')+"/threedhst_bsfh/parameter_files/"+runname+'/'+parm_basename+'_'+str(jj+1)+'.py')	

	elif runname == 'testsed_all':

		id_list = os.getenv('APPS')+"/threedhst_bsfh/data/testsed.ids"
		ids = np.loadtxt(id_list, dtype='|S20')
		ngals = len(ids)

		basename = "testsed_all"
		parm_basename = "testsed_all_params"
		ancilname=None

		for jj in xrange(ngals):
			filebase.append(os.getenv('APPS')+"/threedhst_bsfh/results/"+runname+'/'+basename+'_'+ids[jj])
			parm.append(os.getenv('APPS')+"/threedhst_bsfh/parameter_files/"+runname+'/'+parm_basename+'_'+str(jj+1)+'.py')	

	elif runname == 'testsed_tlink':

		id_list = os.getenv('APPS')+"/threedhst_bsfh/data/testsed.ids"
		ids = np.loadtxt(id_list, dtype='|S20')
		ngals = len(ids)

		basename = "testsed_tlink"
		parm_basename = "testsed_tlink_params"
		ancilname=None

		for jj in xrange(ngals):
			filebase.append(os.getenv('APPS')+"/threedhst_bsfh/results/"+runname+'/'+basename+'_'+ids[jj])
			parm.append(os.getenv('APPS')+"/threedhst_bsfh/parameter_files/"+runname+'/'+parm_basename+'_'+str(jj+1)+'.py')	

	elif runname == 'testsed_linked':

		id_list = os.getenv('APPS')+"/threedhst_bsfh/data/testsed.ids"
		ids = np.loadtxt(id_list, dtype='|S20')
		ngals = len(ids)

		basename = "testsed_linked"
		parm_basename = "testsed_linked_params"
		ancilname=None

		for jj in xrange(ngals):
			filebase.append(os.getenv('APPS')+"/threedhst_bsfh/results/"+runname+'/'+basename+'_'+ids[jj])
			parm.append(os.getenv('APPS')+"/threedhst_bsfh/parameter_files/"+runname+'/'+parm_basename+'_'+str(jj+1)+'.py')	

	elif runname == 'testsed_outliers':

		id_list = os.getenv('APPS')+"/threedhst_bsfh/data/testsed.ids"
		ids = np.loadtxt(id_list, dtype='|S20')
		ngals = len(ids)

		basename = "testsed_outliers"
		parm_basename = "testsed_outliers_params"
		ancilname=None

		for jj in xrange(ngals):
			filebase.append(os.getenv('APPS')+"/threedhst_bsfh/results/"+runname+'/'+basename+'_'+ids[jj])
			parm.append(os.getenv('APPS')+"/threedhst_bsfh/parameter_files/"+runname+'/'+parm_basename+'_'+str(jj+1)+'.py')	

	elif runname == 'dtau_ha_plog':

		id_list = os.getenv('APPS')+"/threedhst_bsfh/data/COSMOS_testsamp.ids"
		ids = np.loadtxt(id_list, dtype='|S20')
		ngals = len(ids)

		basename = "dtau_ha_plog"
		parm_basename = "dtau_ha_plog_params"

		for jj in xrange(ngals):
			ancildat = load_ancil_data(os.getenv('APPS')+
			                           '/threedhst_bsfh/data/COSMOS_testsamp.dat',
			                           ids[jj])
			heqw_txt = "%04d" % int(ancildat['Ha_EQW_obs']) 
			filebase.append(os.getenv('APPS')+"/threedhst_bsfh/results/"+runname+'/'+basename+'_'+heqw_txt+'_'+ids[jj])
			parm.append(os.getenv('APPS')+"/threedhst_bsfh/parameter_files/"+runname+'/'+parm_basename+'_'+str(jj+1)+'.py')	

	elif runname == 'testsed_new':

		id_list = os.getenv('APPS')+"/threedhst_bsfh/data/testsed.ids"
		ids = np.loadtxt(id_list, dtype='|S20')
		ngals = len(ids)

		basename = "testsed_new"
		parm_basename = "testsed_new_params"
		ancilname=None

		for jj in xrange(ngals):
			filebase.append(os.getenv('APPS')+"/threedhst_bsfh/results/"+runname+'/'+basename+'_'+ids[jj])
			parm.append(os.getenv('APPS')+"/threedhst_bsfh/parameter_files/"+runname+'/'+parm_basename+'_'+str(jj+1)+'.py')	

	elif runname == 'testsed':

		id_list = os.getenv('APPS')+"/threedhst_bsfh/data/testsed.ids"
		ids = np.loadtxt(id_list, dtype='|S20')
		ngals = len(ids)

		basename = "testsed"
		parm_basename = "testsed_params"
		ancilname=None

		for jj in xrange(ngals):
			filebase.append(os.getenv('APPS')+"/threedhst_bsfh/results/"+runname+'/'+basename+'_'+ids[jj])
			parm.append(os.getenv('APPS')+"/threedhst_bsfh/parameter_files/"+runname+'/'+parm_basename+'_'+str(jj+1)+'.py')	

	elif runname == 'dtau_genpop_zperr':

		id_list = os.getenv('APPS')+"/threedhst_bsfh/data/COSMOS_gensamp.ids"
		ids = np.loadtxt(id_list, dtype='|S20')
		ngals = len(ids)

		basename = "dtau_genpop_zperr"
		parm_basename = "dtau_genpop_zperr_params"
		ancilname='COSMOS_gensamp.dat'

		for jj in xrange(ngals):
			ancildat = load_ancil_data(os.getenv('APPS')+
			                           '/threedhst_bsfh/data/COSMOS_gensamp.dat',
			                           ids[jj])
			heqw_txt = "%04d" % int(ancildat['Ha_EQW_obs']) 
			filebase.append(os.getenv('APPS')+"/threedhst_bsfh/results/"+runname+'/'+basename+'_'+heqw_txt+'_'+ids[jj])
			parm.append(os.getenv('APPS')+"/threedhst_bsfh/parameter_files/"+runname+'/'+parm_basename+'_'+str(jj+1)+'.py')	

	elif runname == 'dtau_nonir':

		id_list = os.getenv('APPS')+"/threedhst_bsfh/data/COSMOS_testsamp.ids"
		ids = np.loadtxt(id_list, dtype='|S20')
		ngals = len(ids)

		basename = "dtau_nonir"
		parm_basename = "dtau_nonir_params"

		for jj in xrange(ngals):
			ancildat = load_ancil_data(os.getenv('APPS')+
			                           '/threedhst_bsfh/data/COSMOS_testsamp.dat',
			                           ids[jj])
			heqw_txt = "%04d" % int(ancildat['Ha_EQW_obs']) 
			filebase.append(os.getenv('APPS')+"/threedhst_bsfh/results/"+runname+'/'+basename+'_'+heqw_txt+'_'+ids[jj])
			parm.append(os.getenv('APPS')+"/threedhst_bsfh/parameter_files/"+runname+'/'+parm_basename+'_'+str(jj+1)+'.py')	

	elif runname == 'dtau_dynsamp':

		id_list = os.getenv('APPS')+"/threedhst_bsfh/data/twofield_dynsamp.ids"
		ids = np.loadtxt(id_list, dtype='|S20')
		ngals = len(ids)

		basename = "dtau_dynsamp"
		parm_basename = "dtau_dynsamp_params"
		ancilname='twofield_dynsamp.dat'

		for jj in xrange(ngals):
			ancildat = load_ancil_data(os.getenv('APPS')+
			                           '/threedhst_bsfh/data/twofield_dynsamp.dat',
			                           ids[jj])
			heqw_txt = "%04d" % int(ancildat['Ha_EQW_obs']) 
			filebase.append(os.getenv('APPS')+"/threedhst_bsfh/results/"+runname+'/'+basename+'_'+heqw_txt+'_'+ids[jj])
			parm.append(os.getenv('APPS')+"/threedhst_bsfh/parameter_files/"+runname+'/'+parm_basename+'_'+str(jj+1)+'.py')	

	elif runname == 'dtau_genpop_fixedmet':

		ids = np.array(['12658','22801'])
		met_list = os.getenv('APPS')+"/threedhst_bsfh/parameter_files/"+runname+"/met.dat"
		mets = np.loadtxt(met_list, dtype='|S20')
		ngals = len(mets)*len(ids)

		basename = "dtau_genpop_fixedmet"
		parm_basename = "dtau_genpop_fixedmet_params"

		for nn in xrange(len(ids)):
			for mm in xrange(len(mets)/2):
				logzsol_txt = mets[mm]
				num = nn*len(mets)/2+mm+1
				filebase.append(os.getenv('APPS')+"/threedhst_bsfh/results/"+runname+'/'+basename+'_'+logzsol_txt+'_'+ids[nn])
				parm.append(os.getenv('APPS')+"/threedhst_bsfh/parameter_files/"+runname+'/'+parm_basename+'_'+str(num)+'.py')	

	elif runname == 'dtau_genpop_nonir':

		id_list = os.getenv('APPS')+"/threedhst_bsfh/data/COSMOS_gensamp.ids"
		ids = np.loadtxt(id_list, dtype='|S20')
		ngals = len(ids)

		basename = "dtau_genpop_nonir"
		parm_basename = "dtau_genpop_nonir_params"
		ancilname='COSMOS_gensamp.dat'

		for jj in xrange(ngals):
			ancildat = load_ancil_data(os.getenv('APPS')+
			                           '/threedhst_bsfh/data/COSMOS_gensamp.dat',
			                           ids[jj])
			heqw_txt = "%04d" % int(ancildat['Ha_EQW_obs']) 
			filebase.append(os.getenv('APPS')+"/threedhst_bsfh/results/"+runname+'/'+basename+'_'+heqw_txt+'_'+ids[jj])
			parm.append(os.getenv('APPS')+"/threedhst_bsfh/parameter_files/"+runname+'/'+parm_basename+'_'+str(jj+1)+'.py')	

	elif runname == 'dtau_ha_zp':

		id_list = os.getenv('APPS')+"/threedhst_bsfh/data/COSMOS_testsamp_zp.ids"
		ids = np.loadtxt(id_list, dtype='|S20')
		ngals = len(ids)

		basename = "dtau_ha_zp"
		parm_basename = "dtau_ha_zp_params"
		ancilname='COSMOS_testsamp_zp.dat'

		for jj in xrange(ngals):
			ancildat = load_ancil_data(os.getenv('APPS')+
			                           '/threedhst_bsfh/data/COSMOS_testsamp_zp.dat',
			                           ids[jj])
			heqw_txt = "%04d" % int(ancildat['Ha_EQW_obs']) 
			filebase.append(os.getenv('APPS')+"/threedhst_bsfh/results/"+runname+'/'+basename+'_'+heqw_txt+'_'+ids[jj])
			parm.append(os.getenv('APPS')+"/threedhst_bsfh/parameter_files/"+runname+'/'+parm_basename+'_'+str(jj+1)+'.py')	


	elif runname == 'dtau_ha_zperr':

		id_list = os.getenv('APPS')+"/threedhst_bsfh/data/COSMOS_testsamp.ids"
		ids = np.loadtxt(id_list, dtype='|S20')
		ngals = len(ids)

		basename = "dtau_ha_zperr"
		parm_basename = "dtau_ha_zperr_params"

		for jj in xrange(ngals):
			ancildat = load_ancil_data(os.getenv('APPS')+
			                           '/threedhst_bsfh/data/COSMOS_testsamp.dat',
			                           ids[jj])
			heqw_txt = "%04d" % int(ancildat['Ha_EQW_obs']) 
			filebase.append(os.getenv('APPS')+"/threedhst_bsfh/results/"+runname+'/'+basename+'_'+heqw_txt+'_'+ids[jj])
			parm.append(os.getenv('APPS')+"/threedhst_bsfh/parameter_files/"+runname+'/'+parm_basename+'_'+str(jj+1)+'.py')	




	elif runname == 'dtau_genpop':

		id_list = os.getenv('APPS')+"/threedhst_bsfh/data/COSMOS_gensamp.ids"
		ids = np.loadtxt(id_list, dtype='|S20')
		ngals = len(ids)

		basename = "dtau_genpop"
		parm_basename = "dtau_genpop_params"
		ancilname='COSMOS_gensamp.dat'

		for jj in xrange(ngals):
			ancildat = load_ancil_data(os.getenv('APPS')+
			                           '/threedhst_bsfh/data/COSMOS_gensamp.dat',
			                           ids[jj])
			heqw_txt = "%04d" % int(ancildat['Ha_EQW_obs']) 
			filebase.append(os.getenv('APPS')+"/threedhst_bsfh/results/"+runname+'/'+basename+'_'+heqw_txt+'_'+ids[jj])
			parm.append(os.getenv('APPS')+"/threedhst_bsfh/parameter_files/"+runname+'/'+parm_basename+'_'+str(jj+1)+'.py')	


	elif runname == 'dtau_calzetti':

		id_list = os.getenv('APPS')+"/threedhst_bsfh/data/COSMOS_testsamp.ids"
		ids = np.loadtxt(id_list, dtype='|S20')
		ngals = len(ids)

		basename = "dtau_calzetti"
		parm_basename = "dtau_calzetti_params"

		for jj in xrange(ngals):
			ancildat = load_ancil_data(os.getenv('APPS')+
			                           '/threedhst_bsfh/data/COSMOS_testsamp.dat',
			                           ids[jj])
			heqw_txt = "%04d" % int(ancildat['Ha_EQW_obs']) 
			filebase.append(os.getenv('APPS')+"/threedhst_bsfh/results/"+runname+'/'+basename+'_'+heqw_txt+'_'+ids[jj])
			parm.append(os.getenv('APPS')+"/threedhst_bsfh/parameter_files/"+runname+'/'+parm_basename+'_'+str(jj+1)+'.py')	

	elif runname == 'stau':

		id_list = os.getenv('APPS')+"/threedhst_bsfh/data/COSMOS_testsamp.ids"
		ids = np.loadtxt(id_list, dtype='|S20')
		ngals = len(ids)

		basename = "stau"
		parm_basename = "stau_params"

		for jj in xrange(ngals):
			ancildat = load_ancil_data(os.getenv('APPS')+
			                           '/threedhst_bsfh/data/COSMOS_testsamp.dat',
			                           ids[jj])
			heqw_txt = "%04d" % int(ancildat['Ha_EQW_obs']) 
			filebase.append(os.getenv('APPS')+"/threedhst_bsfh/results/"+runname+'/'+basename+'_'+heqw_txt+'_'+ids[jj])
			parm.append(os.getenv('APPS')+"/threedhst_bsfh/parameter_files/"+runname+'/'+parm_basename+'_'+str(jj+1)+'.py')	


	elif runname == 'stau_iracoff':

		id_list = os.getenv('APPS')+"/threedhst_bsfh/data/COSMOS_testsamp.ids"
		ids = np.loadtxt(id_list, dtype='|S20')
		ngals = len(ids)

		basename = "stau_iracoff"
		parm_basename = "stau_iracoff_params"

		for jj in xrange(ngals):
			ancildat = load_ancil_data(os.getenv('APPS')+
			                           '/threedhst_bsfh/data/COSMOS_testsamp.dat',
			                           ids[jj])
			heqw_txt = "%04d" % int(ancildat['Ha_EQW_obs']) 
			filebase.append(os.getenv('APPS')+"/threedhst_bsfh/results/"+runname+'/'+basename+'_'+heqw_txt+'_'+ids[jj])
			parm.append(os.getenv('APPS')+"/threedhst_bsfh/parameter_files/"+runname+'/'+parm_basename+'_'+str(jj+1)+'.py')	

	elif runname == 'stau_intmet':

		id_list = os.getenv('APPS')+"/threedhst_bsfh/data/COSMOS_testsamp.ids"
		ids = np.loadtxt(id_list, dtype='|S20')
		ngals = len(ids)

		basename = "stau_intmet"
		parm_basename = "stau_intmet_params"

		for jj in xrange(ngals):
			ancildat = load_ancil_data(os.getenv('APPS')+
			                           '/threedhst_bsfh/data/COSMOS_testsamp.dat',
			                           ids[jj])
			heqw_txt = "%04d" % int(ancildat['Ha_EQW_obs']) 
			filebase.append(os.getenv('APPS')+"/threedhst_bsfh/results/"+runname+'/'+basename+'_'+heqw_txt+'_'+ids[jj])
			parm.append(os.getenv('APPS')+"/threedhst_bsfh/parameter_files/"+runname+'/'+parm_basename+'_'+str(jj+1)+'.py')	

	elif runname == 'dtau_intmet':

		id_list = os.getenv('APPS')+"/threedhst_bsfh/data/COSMOS_testsamp.ids"
		ids = np.loadtxt(id_list, dtype='|S20')
		ngals = len(ids)

		basename = "dtau_intmet"
		parm_basename = "dtau_intmet_params"

		for jj in xrange(ngals):
			ancildat = load_ancil_data(os.getenv('APPS')+
			                           '/threedhst_bsfh/data/COSMOS_testsamp.dat',
			                           ids[jj])
			heqw_txt = "%04d" % int(ancildat['Ha_EQW_obs']) 
			filebase.append(os.getenv('APPS')+"/threedhst_bsfh/results/"+runname+'/'+basename+'_'+heqw_txt+'_'+ids[jj])
			parm.append(os.getenv('APPS')+"/threedhst_bsfh/parameter_files/"+runname+'/'+parm_basename+'_'+str(jj+1)+'.py')	

	elif runname == 'dtau_neboff':

		id_list = os.getenv('APPS')+"/threedhst_bsfh/data/COSMOS_testsamp.ids"
		ids = np.loadtxt(id_list, dtype='|S20')
		ngals = len(ids)

		basename = "dtau_neboff"
		parm_basename = "dtau_neboff_params"

		for jj in xrange(ngals):
			ancildat = load_ancil_data(os.getenv('APPS')+
			                           '/threedhst_bsfh/data/COSMOS_testsamp.dat',
			                           ids[jj])
			heqw_txt = "%04d" % int(ancildat['Ha_EQW_obs']) 
			filebase.append(os.getenv('APPS')+"/threedhst_bsfh/results/"+runname+'/'+basename+'_'+heqw_txt+'_'+ids[jj])
			parm.append(os.getenv('APPS')+"/threedhst_bsfh/parameter_files/"+runname+'/'+parm_basename+'_'+str(jj+1)+'.py')		

	elif runname == 'dtau_nebon':

		id_list = os.getenv('APPS')+"/threedhst_bsfh/data/COSMOS_testsamp.ids"
		ids = np.loadtxt(id_list, dtype='|S20')
		ngals = len(ids)

		basename = "dtau_nebon"
		parm_basename = "dtau_nebon_params"

		for jj in xrange(ngals):
			ancildat = load_ancil_data(os.getenv('APPS')+
			                           '/threedhst_bsfh/data/COSMOS_testsamp.dat',
			                           ids[jj])
			heqw_txt = "%04d" % int(ancildat['Ha_EQW_obs']) 
			filebase.append(os.getenv('APPS')+"/threedhst_bsfh/results/"+runname+'/'+basename+'_'+heqw_txt+'_'+ids[jj])
			parm.append(os.getenv('APPS')+"/threedhst_bsfh/parameter_files/"+runname+'/'+parm_basename+'_'+str(jj+1)+'.py')	

	elif runname == 'neboff_oiii':

		id_list = os.getenv('APPS')+"/threedhst_bsfh/data/COSMOS_oiii_em.ids"
		ids = np.loadtxt(id_list, dtype='|S20')
		ngals = len(ids)

		basename = "neboff_oiii"
		parm_basename = "neboff_oiii_params"
		ancilname='COSMOS_oiii_em.dat'

		for jj in xrange(ngals):
			ancildat = load_ancil_data(os.getenv('APPS')+
			                           '/threedhst_bsfh/data/COSMOS_oiii_em.dat',
			                           ids[jj])
			heqw_txt = "%04d" % int(ancildat['Ha_EQW_obs']) 
			filebase.append(os.getenv('APPS')+"/threedhst_bsfh/results/"+runname+'/'+basename+'_'+heqw_txt+'_'+ids[jj])
			parm.append(os.getenv('APPS')+"/threedhst_bsfh/parameter_files/"+runname+'/'+parm_basename+'_'+str(jj+1)+'.py')		

	elif runname == 'nebon':

		id_list = os.getenv('APPS')+"/threedhst_bsfh/data/COSMOS_testsamp.ids"
		ids = np.loadtxt(id_list, dtype='|S20')
		ngals = len(ids)

		basename = "ha_selected_nebon"
		parm_basename = "halpha_selected_nebon_params"
		
		for jj in xrange(ngals):
			ancildat = load_ancil_data(os.getenv('APPS')+
			                           '/threedhst_bsfh/data/COSMOS_testsamp.dat',
			                           ids[jj])
			heqw_txt = "%04d" % int(ancildat['Ha_EQW_obs']) 
			filebase.append(os.getenv('APPS')+"/threedhst_bsfh/results/"+basename+'_'+heqw_txt+'_'+ids[jj])
			parm.append(os.getenv('APPS')+"/threedhst_bsfh/parameter_files/"+parm_basename+'_'+str(jj+1)+'.py')

	elif runname == 'neboff':

		id_list = os.getenv('APPS')+"/threedhst_bsfh/data/COSMOS_testsamp.ids"
		ids = np.loadtxt(id_list, dtype='|S20')
		ngals = len(ids)

		basename = "ha_selected_neboff"
		parm_basename = "halpha_selected_params"

		for jj in xrange(ngals):
			ancildat = load_ancil_data(os.getenv('APPS')+
			                           '/threedhst_bsfh/data/COSMOS_testsamp.dat',
			                           ids[jj])
			heqw_txt = "%04d" % int(ancildat['Ha_EQW_obs']) 
			filebase.append(os.getenv('APPS')+"/threedhst_bsfh/results/"+runname+'/'+basename+'_'+heqw_txt+'_'+ids[jj])
			parm.append(os.getenv('APPS')+"/threedhst_bsfh/parameter_files/"+runname+'/'+parm_basename+'_'+str(jj+1)+'.py')

	elif runname == 'photerr':
		
		id = '19723'
		basename = 'photerr/photerr'
		errnames = np.loadtxt(os.getenv('APPS')+'/threedhst_bsfh/parameter_files/photerr/photerr.txt')

		for jj in xrange(len(errnames)): 
			filebase.append(os.getenv('APPS')+"/threedhst_bsfh/results/"+basename+'_'+str(errnames[jj])+'_'+id)
			parm.append(os.getenv('APPS')+"/threedhst_bsfh/parameter_files/photerr/photerr_params_"+str(jj+1)+'.py')

	elif 'testsed' in runname:
		id_list = os.getenv('APPS')+"/threedhst_bsfh/data/testsed.ids"
		ids = np.loadtxt(id_list, dtype='|S20')
		ngals = len(ids)

		basename = runname
		parm_basename = runname+'_params'
		ancilname=None

		for jj in xrange(ngals):
			filebase.append(os.getenv('APPS')+"/threedhst_bsfh/results/"+runname+'/'+basename+'_'+ids[jj])
			parm.append(os.getenv('APPS')+"/threedhst_bsfh/parameter_files/"+runname+'/'+parm_basename+'_'+str(jj+1)+'.py')	

	return filebase,parm,ancilname

def chop_chain(chain):
	'''
	simple placeholder
	will someday replace with a test for convergence to determine where to chop
	JRL 1/5/14
	'''
	nchop=1.66

	flatchain = chain[:,int(chain.shape[1]/nchop):,:]
	flatchain = flatchain.reshape(flatchain.shape[0] * flatchain.shape[1],
                                  flatchain.shape[2])

	return flatchain


def return_mwave_custom(filters):

	"""
	returns effective wavelength based on filter names
	"""

	loc = os.getenv('APPS')+'/threedhst_bsfh/filters/'
	key_str = 'filter_keys_threedhst.txt'
	lameff_str = 'lameff_threedhst.txt'
	
	lameff = np.loadtxt(loc+lameff_str)
	keys = np.loadtxt(loc+key_str, dtype='S20',usecols=[1])
	keys = keys.tolist()
	keys = np.array([keys.lower() for keys in keys], dtype='S20')
	
	lameff_return = [[lameff[keys == filters[i]]][0] for i in range(len(filters))]
	lameff_return = [item for sublist in lameff_return for item in sublist]
	assert len(filters) == len(lameff_return), "Filter name is incorrect"

	return lameff_return

def load_zp_offsets(field):

	filename = os.getenv('APPS')+'/threedhst_bsfh/data/zp_offsets_tbl11_skel14.txt'
	with open(filename, 'r') as f:
		for jj in range(1): hdr = f.readline().split()
	dtype = [np.dtype((str, 35)),np.dtype((str, 35)),np.float,np.float]
	dat = np.loadtxt(filename, comments = '#',dtype = np.dtype([(hdr[n+1], dtype[n]) for n in xrange(len(hdr)-1)]))

	if field is not None:
		good = dat['Field'] == field
		if np.sum(good) == 0:
			print 'Not an acceptable field name! Returning None'
			return None
		else:
			dat = dat[good]

	return dat

def load_ancil_data(filename,objnum):

	'''
	loads ancillary plotting information
	'''
	
	with open(filename, 'r') as f:
		for jj in range(1): hdr = f.readline().split()
	dat = np.loadtxt(filename, comments = '#',dtype = np.dtype([(n, np.float) for n in hdr[1:]]))

	if objnum:
		objdat = dat[dat['id'] == float(objnum)]
		return objdat

	return dat

def load_mips_data(filename,objnum=None):
	
	with open(filename, 'r') as f:
		for jj in range(1): hdr = f.readline().split()
	dat = np.loadtxt(filename, comments = '#',dtype = np.dtype([(n, np.float) for n in hdr[1:]]))
	
	if objnum is not None:
		objdat = dat[dat['id'] == float(objnum)]
		return objdat

	return dat

def load_obs_3dhst(filename, objnum, mips=None, min_error = None, abs_error=False,zperr=False):
	"""
	Load 3D-HST photometry file, return photometry for a particular object.
	min_error: set the minimum photometric uncertainty to be some fraction
	of the flux. if not set, use default errors.
	"""
	obs ={}

	with open(filename, 'r') as f:
		hdr = f.readline().split()
	dat = np.loadtxt(filename, comments = '#',
					 dtype = np.dtype([(n, np.float) for n in hdr[1:]]))
	obj_ind = np.where(dat['id'] == int(objnum))[0][0]
	
	# extract fluxes+uncertainties for all objects and all filters
	flux_fields = [f for f in dat.dtype.names if f[0:2] == 'f_']
	unc_fields = [f for f in dat.dtype.names if f[0:2] == 'e_']
	filters = [f[2:] for f in flux_fields]

	# extract fluxes for particular object, converting from record array to numpy array
	flux = dat[flux_fields].view(float).reshape(len(dat),-1)[obj_ind]
	unc  = dat[unc_fields].view(float).reshape(len(dat),-1)[obj_ind]

	# define all outputs
	wave_effective = np.array(return_mwave_custom(filters))
	phot_mask = np.logical_or(np.logical_or((flux != unc),(flux > 0)),flux != -99.0)
	maggies = flux/(10**10)
	maggies_unc = unc/(10**10)

	# set minimum photometric error
	if min_error is not None:
		if abs_error:
			maggies_unc = min_error*maggies
		else:
			under = maggies_unc < min_error*maggies
			maggies_unc[under] = min_error*maggies[under]
	
	if zperr is True:
		zp_offsets = load_zp_offsets(None)
		band_names = np.array([x['Band'].lower()+'_'+x['Field'].lower() for x in zp_offsets])
		
		for kk in xrange(len(filters)):
			match = band_names == filters[kk]
			if np.sum(match) > 0:
				maggies_unc[kk] = ( (maggies_unc[kk]**2) + (maggies[kk]*(1-zp_offsets[match]['Flux-Correction'][0]))**2 ) **0.5

	# sort outputs based on effective wavelength
	points = zip(wave_effective,filters,phot_mask,maggies,maggies_unc)
	sorted_points = sorted(points)

	# build output dictionary
	obs['wave_effective'] = np.array([point[0] for point in sorted_points])
	obs['filters'] = np.array([point[1] for point in sorted_points])
	obs['phot_mask'] =  np.array([point[2] for point in sorted_points])
	obs['maggies'] = np.array([point[3] for point in sorted_points])
	obs['maggies_unc'] =  np.array([point[4] for point in sorted_points])
	obs['wavelength'] = None
	obs['spectrum'] = None

	return obs

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
	lam = 5500/1e4   # in microns
	Rv   = 4.05
	klam = 2.659*(-2.156+1.509/(lam)-0.198/(lam)**2+0.011/(lam)**3)+Rv
	A_lam = klam/Rv*av

	# now convert from extinction (fobs = fint * 10 ** -0.4*extinction)
	# into opacity (fobs = fint * exp(-opacity)) [i.e, dust2]
	#tau = -np.log(10**(-0.4*A_5500))
	#return av
	tau = av/1.086
	tau = av/1.17
	return tau

	#return av

def return_fast_sed(fastname,objname, sps=None, obs=None, dustem = False):

	'''
	give the fast parameters straight from the FAST out file
	return observables with best-fit FAST parameters, main difference hopefully being stellar population models
	'''

	# load fast parameters
	fast, fields = load_fast_3dhst(fastname, objname)
	fields = np.array(fields)

	# load fast model
	param_file = os.getenv('APPS')+'/threedhst_bsfh/parameter_files/fast_mimic/fast_mimic.py'
	model = model_setup.load_model(param_file)
	parnames = np.array(model.theta_labels())

	# feed parameters into model
	model.params['zred']                   = np.array(fast[fields == 'z'])
	model.initial_theta[parnames=='tage']  = np.clip(np.array((10**fast[fields == 'lage'])/1e9),0.101,10000)
	model.initial_theta[parnames=='tau']   = np.array((10**fast[fields == 'ltau'])/1e9)
	model.initial_theta[parnames=='dust2'] = np.array(av_to_dust2(fast[fields == 'Av']))
	model.initial_theta[parnames=='mass']  = np.array(10**fast[fields == 'lmass'])

	print 'z,tage,tau,dust2,mass'
	print model.params['zred'],model.initial_theta[parnames=='tage'],model.initial_theta[parnames=='tau'],model.initial_theta[parnames=='dust2'],model.initial_theta[parnames=='mass']

	# get dust emission, if desired
	if dustem:
		model.params['add_dust_emission']  = np.array(True)
		model.params['add_agb_dust_model'] = np.array(True)


	spec, mags, w = model.mean_model(model.initial_theta, obs, sps=sps, norm_spec=True)

	return spec,mags,w,fast,fields


def load_fast_3dhst(filename, objnum):
	"""
	Load FAST output for a particular object
	Returns a dictionary of inputs for BSFH
	"""

	# filter through header junk, load data
	with open(filename, 'r') as f:
		for jj in range(1): hdr = f.readline().split()
	dat = np.loadtxt(filename, comments = '#',dtype = np.dtype([(n, np.float) for n in hdr[1:]]))

	# extract field names, search for ID, pull out object info
	fields = [f for f in dat.dtype.names]
	
	
	if objnum is None:
		values = dat[fields].view(float).reshape(len(dat),-1)
	else:
		values = dat[fields].view(float).reshape(len(dat),-1)
		id_ind = fields.index('id')
		obj_ind = [int(x[id_ind]) for x in dat].index(int(objnum))
		values = values[obj_ind]

	return values, fields

def integrate_mag(spec_lam,spectra,filter, z=None, alt_file=None):

	'''
	borrowed from calc_ml
	given a filter name and spectrum, calculate magnitude/luminosity in filter (see alt_file for filter names)
	INPUT: 
		SPEC_LAM: must be in angstroms. if redshift is specified, this should ALREADY be corrected for reddening.
		SPECTRA: must be in Lsun/Hz (FSPS standard). if redshift is specified, the normalization will be taken care of.
	OUTPUT:
		LUMINOSITY: comes out in erg/s
		MAG: comes out as absolute magnitude
			NOTE: if redshift is specified, INSTEAD RETURN apparent magnitude and flux [erg/s/cm^2]
	'''

	if type(filter) == str:
		resp_lam, res = load_filter_response(filter, 
		                                 	 alt_file='/Users/joel/code/fsps/data/allfilters_threedhst.dat')
	else:
		resp_lam = filter[0][0]
		res      = filter[1][0]

	# calculate effective width
	#dlam = (resp_lam[1:]-resp_lam[:-1])/2.
	#width = np.array([dlam[ii]+dlam[ii+1] for ii in xrange(len(resp_lam)-2)])
	#width = np.concatenate((np.atleast_1d(dlam[0]*2),width,np.atleast_1d(dlam[-1]*2)))
	#effective_width = np.sum(res*width)
	
	# physical units, in CGS, from sps_vars.f90 in the SPS code
	pc2cm = 3.08568E18
	lsun  = 3.839E33
	c     = 2.99E10

	# interpolate filter response onto spectral lambda array
	# when interpolating, set response outside wavelength range to be zero.
	response_interp_function = interp1d(resp_lam,res, bounds_error = False, fill_value = 0)
	resp_interp = response_interp_function(spec_lam)
	
	# integrate spectrum over filter response
	# first calculate luminosity: convert to flambda (factor of c/lam^2, with c converted to AA/s)
	# then integrate over flambda [Lsun/AA] to get Lsun
	spec_flam = spectra*(c*1e8/(spec_lam**2))
	luminosity = simps(spec_flam*resp_interp,spec_lam)
	
	# now calculate luminosity density [erg/s/Hz] in filter
	# this involves normalizing the filter response by integrating over wavelength
	norm = simps(resp_interp/spec_lam,spec_lam)
	luminosity_density = simps(spectra*(resp_interp/norm)/spec_lam,spec_lam)

	# if redshift is specified, convert to flux and apparent magnitude
	if z:
		from astropy.cosmology import WMAP9
		dfactor = (WMAP9.luminosity_distance(z).value*1e5)**(-2)*(1+z)
		luminosity = luminosity*dfactor
		luminosity_density = luminosity_density*dfactor

	# convert luminosity density to flux density
	# the units of the spectra are Lsun/Hz; convert to
	# erg/s/cm^2/Hz, at 10pc for absolute mags
	flux_density = luminosity_density*lsun/(4.0*np.pi*(pc2cm*10)**2)
	luminosity   = luminosity*lsun

	# convert flux density to magnitudes in AB system
	mag = -2.5*np.log10(flux_density)-48.60

	#print 'maggies: {0}'.format(10**(-0.4*mag)*1e10)
	return mag, luminosity

def integrate_sfh(t1,t2,mass,tage,tau,sf_start):
	
	'''
	integrate a delayed tau SFH between t1 and t2
	'''
	t1 = t1-sf_start
	t2 = t2-sf_start
	tage = tage-sf_start

	# sanitize inputs
	ndim = len(np.atleast_1d(mass))
	if len(np.atleast_1d(t2)) != ndim:
		t2 = np.zeros(ndim)+t2
	if len(np.atleast_1d(t1)) != ndim:
		t1 = np.zeros(ndim)+t1

	# if we're outside of the time boundaries, clip to boundary values
	t1 = np.clip(t1,0,tage)
	t2 = np.clip(t2,0,tage)

	# add tau model
	intsfr =  np.exp(-t1/tau)*(1+t1/tau) - \
	          np.exp(-t2/tau)*(1+t2/tau)
	norm =    1.0- np.exp(-tage    /tau)*(1+tage/tau)
	intsfr = intsfr/norm

	# return sum of SFR components
	tot_sfr = np.sum(intsfr*mass)/np.sum(mass)
	return tot_sfr

def measure_emline_lum(sps, model = None, obs = None, thetas = None, measure_ir = False, saveplot = False):
	
	'''
	takes spec(on)-spec(off) to measure emission line luminosity
	sideband is defined for each emission line after visually 
	inspecting the spectral sampling density around each line
	'''

    # define emission lines
	emline = np.array(['[OII]','Hbeta','[OIII]1','[OIII]2','Halpha','[SII]'])
	wavelength = np.array([3728,4861.33,4959,5007,6562,6732.71])
	sideband   = [(3723,3736),(4857,4868),(4954,4968),(5001,5015),(6556,6573),(6710,6728)]
	nline = len(emline)

	# get spectrum
	if model:

		# save redshift
		z      = model.params.get('zred', np.array(0.0))+0.0
		model.params['zred'] = np.array(0.0)

		# nebon
		model.params['add_neb_emission'] = np.array(True)
		model.params['add_neb_continuum'] = np.array(True)
		spec,mags,w = model.mean_model(thetas, obs, sps=sps, norm_spec=False)
		
		# switch to flam
		factor = 3e18 / w**2
		spec *= factor

		model.params['zred'] = z

	else:
		sps.params['add_neb_emission'] = True
		sps.params['add_neb_continuum'] = True
		w, spec = sps.get_spectrum(tage=sps.params['tage'], peraa=True)

	emline_flux = np.zeros(len(wavelength))

	for jj in xrange(len(wavelength)):
		integrate_lam = (w > sideband[jj][0]) & (w < sideband[jj][1])
		baseline      = spec[np.abs(w-sideband[jj][0]) == np.min(np.abs(w-sideband[jj][0]))][0]
		emline_flux[jj] = np.trapz(spec[integrate_lam]-baseline, w[integrate_lam])

		if saveplot and jj==4:
			plt.plot(w,spec,'ro',linestyle='-')
			plt.plot(w[integrate_lam], spec[integrate_lam], 'bo',linestyle=' ')
			plt.xlim(wavelength[jj]-40,wavelength[jj]+40)
			plt.ylim(-np.max(spec[integrate_lam])*0.2,np.max(spec[integrate_lam])*1.2)
			
			plotlines=['[SIII]','[NII]','Halpha','[SII]']
			plotlam  =np.array([6312,6583,6563,6725])
			for kk in xrange(len(plotlam)):
				plt.vlines(plotlam[kk],plt.ylim()[0],plt.ylim()[1],color='0.5',linestyle='--')
				plt.text(plotlam[kk],(plt.ylim()[0]+plt.ylim()[1])/2.*(1.0-kk/6.0),plotlines[kk])
			plt.savefig(os.getenv('APPS')+'/threedhst_bsfh/plots/testem/emline_'+str(saveplot)+'.png',dpi=300)
			plt.close()

	if measure_ir:

		# set up filters
		mips_index = [i for i, s in enumerate(obs['filters']) if 'mips' in s]
		botlam = np.atleast_1d(8e4-1)
		toplam = np.atleast_1d(1000e4+1)
		edgetrans = np.atleast_1d(0)
		lir_filter = [[np.concatenate((botlam,np.linspace(8e4, 1000e4, num=100),toplam))],
		              [np.concatenate((edgetrans,np.ones(100),edgetrans))]]

		# calculate z=0 magnitudes
		spec_neboff,mags_neboff,w = model.mean_model(thetas, obs, sps=sps,norm_spec=False)

		_,lir     = integrate_mag(w,spec_neboff,lir_filter, z=None, alt_file=None) # comes out in ergs/s
		lir       = lir / 3.846e33 #  convert to Lsun

		# revert to proper redshift, calculate redshifted mips magnitudes
		model.params['zred'] = np.atleast_1d(z)
		
		# if no MIPS flux...
		try:
			mips = mags_neboff[mips_index][0]*1e10 # comes out in maggies, convert to flux such that AB zeropoint is 25 mags
		except:
			mips = np.nan

		out = {'emline_flux': emline_flux,
		       'emline_name': emline,
		       'lir': lir,
		       'mips': mips}
	else:
		out = {'emline_flux': emline_flux,
			   'emline_name': emline}

	return out












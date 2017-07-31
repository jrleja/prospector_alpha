import read_sextractor, read_data, random, os, prosp_dutils
import numpy as np
u.random.seed(25001)
	
def return_test_sfhs(test_sfhs):

	'''
	return special constraints on sf_start and tau
	custom-built for SFH recovery tests
	'''

	# always implement mass priors
	if test_sfhs != 0:
		mass_bounds      = np.array([[1e10,1e11],[1e10,1e11]])

	if test_sfhs == 1:
		tau_bounds      = np.array([[50,100],[50,100]])
		sf_start_bounds = np.array([[0,4],[0,4]])
		dust_bounds     = np.array([[0.0,4.0],[0.0,4.0]])
		descriptor      = 'constant sfh'

	elif test_sfhs == 2:
		tau_bounds      = np.array([[0.1,3],[0.1,3]])
		sf_start_bounds = np.array([[0,4],[0,4]])
		dust_bounds     = np.array([[0.0,4.0],[0.0,4.0]])
		descriptor      = 'quiescent'		

	elif test_sfhs == 3:
		tau_bounds      = np.array([[0.1,3],[0.1,3]])
		sf_start_bounds = np.array([[12,14],[12,14]])
		dust_bounds     = np.array([[0.0,4.0],[0.0,4.0]])
		descriptor      = 'burst'

	elif test_sfhs == 4:
		tau_bounds      = np.array([[0.1,7],[0.1,3]])
		sf_start_bounds = np.array([[0,3],[12,14]])
		dust_bounds     = np.array([[0.0,4.0],[0.0,4.0]])
		descriptor      = 'old+burst'

	elif test_sfhs == 5:
		tau_bounds      = np.array([[4.0,20.0],[0.1,3]])
		sf_start_bounds = np.array([[0,12],[12.5,14]])
		dust_bounds     = np.array([[0.0,4.0],[3.0,4.0]])
		descriptor      = 'dusty_burst'

	parnames = np.array(['mass','tau','sf_start','dust2'])
	bounds   = np.array([mass_bounds,tau_bounds,sf_start_bounds,dust_bounds])

	return bounds,parnames,descriptor

def return_bounds(parname,model,i,test_sfhs=False):

	'''
	returns parameter boundaries
	if test_sfhs is on, puts special constraints on certain variables
	these special constraints are defined in return_test_sfhs
	'''

	bounds = model.theta_bounds()[i]
	if test_sfhs != False:
		bounds,parnames,descriptor=return_test_sfhs(test_sfhs)
		if parname[:-2] in parnames:
			bounds = bounds[parnames == parname[:-2]][0][int(parname[-1])-1]

	return bounds[0],bounds[1]

def parname_strip(parname):

	try:
		int(parname[-1])
		return parname[-2:]
	except:
		return parname

def build_sample_constrained(basename,outname=None,add_zp_err=False):

	'''
	Generate model SEDs and add noise
	same as below, but only vary tage, sf_trunc, sf_slope, and dust2
	'''

	from bsfh import model_setup

	#### output names ####
	if outname == None:
		outname = '/Users/joel/code/python/prospector_alpha/data/'+basename
	parmfile='/Users/joel/code/python/prospector_alpha/parameter_files/'+basename+'/'+basename+'_params.py'

	#### load test model, build sps  ####
	model = model_setup.load_model(parmfile)
	obs   = model_setup.load_obs(parmfile)
	sps = prosp_dutils.setup_sps()

	#### basic parameters ####
	ngals_per_model     = 70
	noise               = 0.05            # perturb fluxes
	reported_noise      = 0.05            # reported noise
	test_sfhs           = [0]
	ntest               = len(test_sfhs)
	ngals               = ntest*ngals_per_model
	
	#### parameters to vary ####
	to_vary = ['logzsol']

	#### band-specific noise ####
	if 'gp_filter_amps' in model.free_params:
		band_specific_noise = [0.0,0.15,0.25] # add band-specific noise?

	#### outlier noise ####
	if 'gp_outlier_locs' in model.free_params:
		outliers_noise      = 0.5             # add outlier noise
		outliers_bands      = [5,22,29]
	else:
		outliers_bands=[]

	#### generate random model parameters ####
	nparams = len(model.initial_theta)
	testparms = np.zeros(shape=(ngals,nparams))
	parnames = np.array(model.theta_labels())

	for jj in xrange(ntest):
		for ii in xrange(nparams):

			# random in logspace for mass
			# also enforce priors
			if parnames[ii] not in to_vary:
				
				for kk in xrange(jj*ngals_per_model,(jj+1)*ngals_per_model): testparms[kk,ii] = model.params[parnames[ii]] 
				print model.params[parnames[ii]] 

			#### generate specific SFHs if necessary ####
			elif parname_strip(parnames[ii]) == 'tage':
				min,max = return_bounds(parnames[ii],model,ii,test_sfhs=test_sfhs[jj])
				for kk in xrange(jj*ngals_per_model,(jj+1)*ngals_per_model): testparms[kk,ii] = random.random()*(max-min)+min
				print min,max

			#### enforce SFH priors ####
			elif parname_strip(parnames[ii]) == 'delt_trunc':
				min,max = return_bounds(parnames[ii],model,ii,test_sfhs=test_sfhs[jj])
				for kk in xrange(jj*ngals_per_model,(jj+1)*ngals_per_model): testparms[kk,ii] = random.random()*(max-min)+min
				print min,max

			elif parname_strip(parnames[ii]) == 'sf_slope':
				min,max = return_bounds(parnames[ii],model,ii,test_sfhs=test_sfhs[jj])
				for kk in xrange(jj*ngals_per_model,(jj+1)*ngals_per_model): 
					if random.random() > 0.5:
						testparms[kk,ii] = random.random()*max
					else:
						testparms[kk,ii] = random.random()*np.abs(min)+min

				print min,max

			#### tone down the dust a bit-- flat in prior means lots of Av = 2.0 galaxies ####
			elif parnames[ii] == 'dust2':
				min = model.theta_bounds()[ii][0]
				max = model.theta_bounds()[ii][1]
				for kk in xrange(jj*ngals_per_model,(jj+1)*ngals_per_model): testparms[kk,ii] = np.clip(random.gauss(0.5, 0.5),min,max)
				print min,max

			#### not randomized
			elif parnames[ii] == 'logzsol':

				min,max = return_bounds(parnames[ii],model,ii,test_sfhs=test_sfhs[jj])
				min = -0.6
				for kk in xrange(jj*ngals_per_model,(jj+1)*ngals_per_model): testparms[kk,ii] = 0.19-0.01*kk
			
			

	#### make sure priors are satisfied
	for ii in xrange(ngals):
		assert np.isfinite(model.prior_product(testparms[ii,:]))

	#### write out thetas ####
	with open(outname+'.dat', 'w') as f:
		
		### header ###
		f.write('# ')
		for theta in model.theta_labels():
			f.write(theta+' ')
		f.write('\n')

		### data ###
		for ii in xrange(ngals):
			for kk in xrange(nparams):
				f.write(str(testparms[ii,kk])+' ')
			f.write('\n')

	#### set up photometry output ####
	nfilters = len(obs['filters'])
	maggies     = np.zeros(shape=(ngals,nfilters))
	maggies_unc = np.zeros(shape=(ngals,nfilters))

	#### generate photometry, add noise ####
	for ii in xrange(ngals):
		_,maggies[ii,:],_ = model.mean_model(testparms[ii,:], obs, sps=sps)
		print maggies[ii,:]
		#### record noise ####
		maggies_unc[ii,:] = maggies[ii,:]*reported_noise

		#### add noise ####
		for kk in xrange(nfilters): 
			
			###### general noise
			tnoise = noise
			
			##### linked filter noise
			filtlist = model.params.get('gp_filter_locs',[])			
			for mm in xrange(len(filtlist)):
				if obs['filters'][kk].lower() in filtlist[mm]:
					tnoise = (tnoise**2+band_specific_noise[mm]**2)**0.5

			##### outlier noise
			if kk in outliers_bands:
				tnoise = (tnoise**2+outliers_noise**2)**0.5
			add_noise = random.gauss(0, tnoise)
			print obs['filters'][kk].lower()+': ' + "{:.2f}".format(add_noise)
			maggies[ii,kk] += add_noise*maggies[ii,kk]

	#### output ####
	#### ids first ####
	ids =  np.arange(ngals)+1
	with open(outname+'.ids', 'w') as f:
	    for id in ids:
	        f.write(str(id)+'\n')

	#### photometry ####
	with open(outname+'.cat', 'w') as f:
		
		### header ###
		f.write('# id ')
		for filter in obs['filters']:
			f.write('f_'+filter+' e_' +filter+' ')
		f.write('\n')

		### data ###
		for ii in xrange(ngals):
			f.write(str(ids[ii])+' ')
			for kk in xrange(nfilters):
				f.write(str(maggies[ii,kk])+' '+str(maggies_unc[ii,kk]) + ' ')
			f.write('\n')

def build_sample_test(basename,outname=None,add_zp_err=False):

	'''
	Generate model SEDs and add noise
	IMPORTANT: linked+outlier noise will NOT be added if those variables are not free 
	parameters in the passed parameter file!
	'''

	from bsfh import model_setup

	#### output names ####
	if outname == None:
		outname = '/Users/joel/code/python/prospector_alpha/data/'+basename
	parmfile='/Users/joel/code/python/prospector_alpha/parameter_files/'+basename+'/'+basename+'_params.py'

	#### load test model, build sps  ####
	model = model_setup.load_model(parmfile)
	obs   = model_setup.load_obs(parmfile)
	sps = prosp_dutils.setup_sps(custom_filter_key=model['run_params'].get('custom_filter_key',None))

	#### basic parameters ####
	ngals_per_model     = 500
	noise               = 0.05            # perturb fluxes
	reported_noise      = 0.05            # reported noise
	test_sfhs           = [1,2,3,4,5]     # which test sfhs to use?
	test_sfhs           = [0]
	ntest               = len(test_sfhs)
	ngals               = ntest*ngals_per_model
	
	#### band-specific noise ####
	if 'gp_filter_amps' in model.free_params:
		band_specific_noise = [0.0,0.15,0.25] # add band-specific noise?

	#### outlier noise ####
	if 'gp_outlier_locs' in model.free_params:
		outliers_noise      = 0.5             # add outlier noise
		outliers_bands      = [5,22,29]
	else:
		outliers_bands=[]

	#### generate random model parameters ####
	nparams = len(model.initial_theta)
	testparms = np.zeros(shape=(ngals,nparams))
	parnames = np.array(model.theta_labels())

	for jj in xrange(ntest):
		for ii in xrange(nparams):
			
			# random in logspace for mass
			# also enforce priors
			if parname_strip(parnames[ii]) == 'mass':
				
				min,max = np.log10(return_bounds(parnames[ii],model,ii,test_sfhs=test_sfhs[jj]))

				# if double tau, ensure that later priors can be enforced
				if parnames[ii] == 'mass_1': max = np.log10(10**max/20)
				if parnames[ii] == 'mass_2':
					max = np.log10(testparms[:,parnames == 'mass_1']*20)
					for kk in xrange(jj*ngals_per_model,(jj+1)*ngals_per_model): testparms[kk,ii] = 10**(random.random()*(max[kk]-min)+min)
				else:
					for kk in xrange(jj*ngals_per_model,(jj+1)*ngals_per_model): testparms[kk,ii] = 10**(random.random()*(max-min)+min)
			
			#### generate specific SFHs if necessary ####
			elif parname_strip(parnames[ii]) == 'tage':
				min,max = return_bounds(parnames[ii],model,ii,test_sfhs=test_sfhs[jj])
				for kk in xrange(jj*ngals_per_model,(jj+1)*ngals_per_model): testparms[kk,ii] = random.random()*(max-min)+min

			#### enforce SFH priors ####
			elif parname_strip(parnames[ii]) == 'sf_trunc':
				min,max = return_bounds(parnames[ii],model,ii,test_sfhs=test_sfhs[jj])
				max = testparms[:,parnames == 'tage']-0.05
				for kk in xrange(jj*ngals_per_model,(jj+1)*ngals_per_model): testparms[kk,ii] = random.random()*(max[kk]-min)+min

			elif parname_strip(parnames[ii]) == 'sf_slope':
				min,max = return_bounds(parnames[ii],model,ii,test_sfhs=test_sfhs[jj])
				for kk in xrange(jj*ngals_per_model,(jj+1)*ngals_per_model): 
					if random.random() > 0.5:
						testparms[kk,ii] = random.random()*max
					else:
						testparms[kk,ii] = random.random()*np.abs(min)+min

			#### tone down the dust a bit-- flat in prior means lots of Av = 2.0 galaxies ####
			elif parnames[ii] == 'dust2':
				if test_sfhs[jj] != 5:
					min = model.theta_bounds()[ii][0]
					max = model.theta_bounds()[ii][1]
					for kk in xrange(jj*ngals_per_model,(jj+1)*ngals_per_model): testparms[kk,ii] = np.clip(random.gauss(0.5, 0.5),min,max)
				else:
					min,max = return_bounds(parnames[ii],model,ii,test_sfhs=test_sfhs[jj])
					for kk in xrange(jj*ngals_per_model,(jj+1)*ngals_per_model): testparms[kk,ii] = random.random()*(max-min)+min

			#### apply dust_index prior! ####
			elif parname_strip(parnames[ii]) == 'dust_index':
				min = model.theta_bounds()[ii][0]
				max = model.theta_bounds()[ii][1]
				for kk in xrange(jj*ngals_per_model,(jj+1)*ngals_per_model): testparms[kk,ii] = np.clip(random.gauss(-0.7, 0.5),min,max)

			#### general photometric jitter ####
			elif parname_strip(parnames[ii])  == 'phot_jitter':
				for kk in xrange(jj*ngals_per_model,(jj+1)*ngals_per_model): testparms[kk,ii] = 0.0
		
			#### linked filter noise ####
			elif parname_strip(parnames[ii]) == 'gp_filter_amps':
				for kk in xrange(jj*ngals_per_model,(jj+1)*ngals_per_model): testparms[kk,ii] = band_specific_noise[int(parnames[ii][-1])-1]

			#### outliers ####
			elif parname_strip(parnames[ii]) == 'gp_outlier_amps':
				for kk in xrange(jj*ngals_per_model,(jj+1)*ngals_per_model): testparms[kk,ii] = outliers_noise
			elif parname_strip(parnames[ii]) == 'gp_outlier_locs':
				for kk in xrange(jj*ngals_per_model,(jj+1)*ngals_per_model): testparms[kk,ii] = outliers_bands[int(parnames[ii][-1])-1]

			else:
				min = model.theta_bounds()[ii][0]
				max = model.theta_bounds()[ii][1]
				for kk in xrange(jj*ngals_per_model,(jj+1)*ngals_per_model): testparms[kk,ii] = random.random()*(max-min)+min
			print parname_strip(parnames[ii])
			print min,max

	#### make sure priors are satisfied
	for ii in xrange(ngals):
		assert np.isfinite(model.prior_product(testparms[ii,:]))

	#### write out thetas ####
	with open(outname+'.dat', 'w') as f:
		
		### header ###
		f.write('# ')
		for theta in model.theta_labels():
			f.write(theta+' ')
		f.write('\n')

		### data ###
		for ii in xrange(ngals):
			for kk in xrange(nparams):
				f.write(str(testparms[ii,kk])+' ')
			f.write('\n')

	#### set up photometry output ####
	nfilters = len(obs['filters'])
	maggies     = np.zeros(shape=(ngals,nfilters))
	maggies_unc = np.zeros(shape=(ngals,nfilters))

	#### generate photometry, add noise ####
	for ii in xrange(ngals):
		model.initial_theta = testparms[ii,:]
		_,maggiestemp,_ = model.mean_model(model.initial_theta, obs, sps=sps,norm_spec=False)
		maggies[ii,:] = maggiestemp

		#### record noise ####
		maggies_unc[ii,:] = maggies[ii,:]*reported_noise

		#### add noise ####
		for kk in xrange(nfilters): 
			
			###### general noise
			tnoise = noise
			
			##### linked filter noise
			filtlist = model.params.get('gp_filter_locs',[])			
			for mm in xrange(len(filtlist)):
				if obs['filters'][kk].lower() in filtlist[mm]:
					tnoise = (tnoise**2+band_specific_noise[mm]**2)**0.5

			##### outlier noise
			if kk in outliers_bands:
				tnoise = (tnoise**2+outliers_noise**2)**0.5
			add_noise = random.gauss(0, tnoise)
			print obs['filters'][kk].lower()+': ' + "{:.2f}".format(add_noise)
			maggies[ii,kk] += add_noise*maggies[ii,kk]

	#### output ####
	#### ids first ####
	ids =  np.arange(ngals)+1
	with open(outname+'.ids', 'w') as f:
	    for id in ids:
	        f.write(str(id)+'\n')

	#### photometry ####
	with open(outname+'.cat', 'w') as f:
		
		### header ###
		f.write('# id ')
		for filter in obs['filters']:
			f.write('f_'+filter+' e_' +filter+' ')
		f.write('\n')

		### data ###
		for ii in xrange(ngals):
			f.write(str(ids[ii])+' ')
			for kk in xrange(nfilters):
				f.write(str(maggies[ii,kk])+' '+str(maggies_unc[ii,kk]) + ' ')
			f.write('\n')
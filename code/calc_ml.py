# loads FSPS filter response curves
def load_filter_response(filter, alt_file=None):
	'''READS FILTER RESPONSE CURVES FOR FSPS'''
	
	
	from astropy.io import ascii
	import numpy as np
	
	if not alt_file:
		filter_response_curve = '/Users/joel/code/fsps/data/allfilters.dat'
	else:
		filter_response_curve = alt_file

	# initialize output arrays
	lam,res = (np.zeros(0) for i in range(2))

	# upper case?
	if filter.lower() == filter:
		lower_case = True
	else:
		lower_case = False

	# open file
	with open(filter_response_curve, 'r') as f:
    	# Skips text until we find the correct filter
		for line in f:
			if lower_case:
				if line.lower().find(filter) != -1:
					break
			else:
				if line.find(filter) != -1:
					break
		# Reads text until we hit the next filter
		for line in f:  # This keeps reading the file
			if line.find('#') != -1:
				break
			# read line, extract data
			data = line.split()
			lam = np.append(lam,float(data[0]))
			res = np.append(res,float(data[1]))

	if len(lam) == 0:
		print "Couldn't find filter " + filter + ': STOPPING'
		print 1/0

	return lam, res

# loads .mags outputs from FSPS
def load_mag_output(filename):

	from astropy.io import ascii

	# first read in filter names
	filters = []
	with open('/Users/joel/code/fsps/data/FILTER_LIST', 'r') as f:
		for line in f:
			if line.find('(') == -1:
				str = line[line.find('\t')+1:line.rfind('\n')]
				filters.append(str.strip())
			else:
				str = line[line.find('\t')+1:line.rfind('(')]
				filters.append(str.strip())
	names = ['log_age', 'log_mass','log_lbol', 'log_SFR']+filters
	
	x = ascii.read(filename,names=names)

	return x

# loads .spec outputs from FSPS
def load_spec_output(filename):
	
	import numpy as np
	
	# output arrays
	log_age, log_mass,log_lbol, log_sfr, spectra = (np.zeros(0) for i in range(5))
	
	# open file
	# comments are in lines 1-9 (i 0-8)
	with open(filename,'r') as f:
		for i,line in enumerate(f):
			
			if i == 9:
				# wavelength vector
				lam = np.array(map(float, line.split()))
			elif (i % 2 == 0) and (i > 9):
				# ages, masses, luminosities, SFRs
				data = line.split()
				log_age = np.append(log_age,float(data[0]))
				log_mass = np.append(log_mass,float(data[1]))
				log_lbol = np.append(log_lbol,float(data[2]))
				log_sfr = np.append(log_sfr,float(data[3]))
			elif (i % 2 == 1) and (i > 9):
				# initialize
				if i == 11:
					spectra = np.array(map(float, line.split()))
				else: 
					spectra = np.vstack([spectra,np.array(map(float, line.split()))])

	return lam, spectra, log_age, log_mass, log_lbol, log_sfr

# convert from FSPS stellar pop synthesis output (filebase+'.spec') to observed AB magnitude in a given filter (filter)
def sed_to_mag(filebase, filter, z=False, spectra = False, spec_lam = False, alt_file=None):
	
	import numpy as np
	from scipy.interpolate import interp1d
	from scipy.integrate import simps
	
	# load SED, response curves
	if not np.sum(spectra):
		spec_lam, spectra, log_age, log_mass,log_lbol, log_sfr = load_spec_output(filebase+'.spec')
	resp_lam, res = load_filter_response(filter, alt_file=alt_file)
	
	# redshift?
	if z != 0:
		spectra_interp = interp1d(spec_lam*(1+z), spectra, bounds_error = False, fill_value = 0)
		spectra = spectra_interp(spec_lam)
		spectra[spectra<0] = 0
	
	# physical units, in CGS, from sps_vars.f90 in the SPS code
	pc2cm = 3.08568E18
	lsun  = 3.839E33
	c     = 2.99E10

	# interpolate filter response onto spectral lambda array
	# when interpolating, set response outside wavelength range to be zero.
	response_interp = interp1d(resp_lam,res, bounds_error = False, fill_value = 0)

	# normalize [THIS STEP I DON'T UNDERSTAND... BUT IT IS NECESSARY]
	norm = simps(response_interp(spec_lam)/spec_lam,spec_lam)

	# integrate filter for all timesteps
	# luminosity density in filter is calculated
	# luminosity is also calculated: multiply by extra factor of (c/lambda) in sum, and convert one lambda to CGS
	luminosity = np.zeros(spectra.shape[0]) 
	luminosity_density = np.zeros(spectra.shape[0]) 
	for ii in range(0, spectra.shape[0]):
		luminosity[ii] = simps(spectra[ii,:]*(response_interp(spec_lam)/norm)*(c/(spec_lam**2*1E-8)),spec_lam)
		luminosity_density[ii] = simps(spectra[ii,:]*(response_interp(spec_lam)/norm)/spec_lam,spec_lam)

	# convert luminosity density to flux density
	# the units of the spectra are Lsun/Hz; convert to
	# erg/s/cm^2/Hz, at 10pc for absolute mags
	flux_density = luminosity_density*lsun/(4.0*np.pi*(pc2cm*10)**2)

	# convert flux density to magnitudes in AB system
	mag = -2.5*np.log10(flux_density)-48.60

	return mag, luminosity
	
def plot_model_ml():

	import numpy as np
	import matplotlib.pyplot as plt

	# where is the data?
	model_loc = '/Users/joel/code/fsps/OUTPUTS/fast_mimic/co11_pr_ch_0.02_'
	ltau = np.arange(8.0,10.1,0.1)
	ltau = np.arange(8.0,10.1,0.5)
	
	# what redshift(s) and what filters do we want to plot?
	z = [0.25, 0.75, 1.25, 1.75, 2.25, 2.75]
	filters = ['F606W', 'F160W']

	# plot info
	colors = ['Aqua', 'DodgerBlue', 'Blue', 'DarkBlue', 'Black']
	fig, axarr = plt.subplots(ncols=3, nrows=2, figsize=(12,8))
	plt.subplots_adjust(wspace=0.000,hspace=0.000)
	axlim = [-0.9,2.9,-1.3,1.9]

	for ii in range(0,len(z)):
		
		# iteration values
		itone = int(ii/3)
		ittwo = ii % 3
		
		for jj in range(0,len(ltau)):
			
			# calculate magnitudes and colors as a function of age at fixed tau, redshift [MODIFY TO INCLUDE REDSHIFT]
			mag1, luminosity1 = sed_to_mag(model_loc+str(ltau[jj]), filters[0],z=z[ii])
			mag2, luminosity2 = sed_to_mag(model_loc+str(ltau[jj]), filters[1],z=z[ii])
			color = mag1-mag2
			
			# calculate mass to light here
			spec_lam, spectra, log_age, log_mass,log_lbol, log_sfr = load_spec_output(model_loc+str(ltau[jj])+'.spec')
			mass2light = np.log10(10**log_mass/luminosity2)
			
			# only plot ages that we use in FAST
			LOG_AGE_MIN    = 7.6
			LOG_AGE_MAX    = 10.1
			good_age = (log_age >= LOG_AGE_MIN) & (log_age <= LOG_AGE_MAX)
			
			# define plot quantities
			plot_color = color[good_age]
			plot_mass2light = mass2light[good_age]
			
			# plot
			axarr[itone,ittwo].plot(plot_color, plot_mass2light,'-o',color=colors[jj],alpha=0.7, lw=2.5, zorder=1)	
			axarr[itone,ittwo].axis(axlim)
			
			# labels (once) (taus)
			xt = axlim[1]-(axlim[1]-axlim[0])*0.07
			if ii == 0:
				axarr[itone,ittwo].text(xt,axlim[3]-(axlim[3]-axlim[2])*(0.05*(jj+1)+0.07),r'log($\tau$)='+str(ltau[jj]),color=colors[jj],horizontalalignment='right')
			
			if jj == 0:
				# labels (repeat) (redshift)
				axarr[itone,ittwo].text(xt,axlim[3]-(axlim[3]-axlim[2])*0.07,'z='+str(z[ii]), weight='semibold',horizontalalignment='right')
		
		# axis labels
		if (itone > 0):
			axarr[itone,ittwo].set_xlabel(r'm$_{\mathrm{'+filters[0]+'}})$-m$_{\mathrm{'+filters[1]+'}})$')
		else:
			axarr[itone,ittwo].set_xticklabels([])
		if (ii == 0 or ii == 3):
			axarr[itone,ittwo].set_ylabel(r'log(M/L$_{\mathrm{'+filters[1]+'}})$')
		else:
			axarr[itone,ittwo].set_yticklabels([])
	
	plt.savefig('/Users/joel/code/python/mass_function/figures/aperture_color/m2l_model/model_ml_'+filters[0]+'_'+filters[1]+'.png', bbox_inches='tight',dpi=300)
	plt.close()

def plot_delta_ml(match_size = False):

	import read_sextractor
	import astropy.table
	import numpy as np
	import matplotlib.pyplot as plt
	from scipy.interpolate import interp1d

	# where is the data?
	model_loc = '/Users/joel/code/fsps/OUTPUTS/fast_mimic/co11_pr_ch_0.02_'
	ltau = np.arange(8.0,10.1,0.1)
	ltau = np.arange(8.0,10.1,0.5)
	
	# what redshift(s) and what filters do we want to plot?
	z = [0.25, 0.75, 1.25, 1.75, 2.25, 2.75]
	zbins = np.array([[0.2,0.5],[0.5,0.75],[0.75,1.0],[1.0,1.25],[1.25,1.5],[1.5,2.0],[2.0,2.5],[2.5,3.0]])
	filters = ['F606W', 'F160W']

	# plot info
	colors = ['Aqua', 'DodgerBlue', 'Blue', 'DarkBlue', 'Black']
	fig, axarr = plt.subplots(ncols=3, nrows=2, figsize=(12,8))
	plt.subplots_adjust(wspace=0.000,hspace=0.000)
	axlim = [-0.5,0.5,-0.9,0.9]

	# load photometric catalog, FAST output (for redshifts)
	fields = ['AEGIS', 'COSMOS', 'GOODS-N', 'GOODS-S']
	for kk in range(0,len(fields)):
		if kk == 0:
			fast = read_sextractor.load_fast_v41(fields[kk])
			phot = read_sextractor.load_phot_v41(fields[kk])
			if match_size == True:
				size = read_sextractor.read_morphology(fields[kk],'F160W')
		else:
			temp_phot = read_sextractor.load_phot_v41(fields[kk])
			phot = astropy.table.operations.vstack([phot,temp_phot])
			temp_fast = read_sextractor.load_fast_v41(fields[kk])
			fast = astropy.table.operations.vstack([fast,temp_fast])
			if match_size == True:
				temp_size = read_sextractor.read_morphology(fields[kk],'F160W')
				size = astropy.table.operations.vstack([size,temp_size])
	
	# cut photometric catalog
	# no cuts on S/N in filter b/c don't want to bias measurement of population median
	goodphot = (phot['use_phot'] == 1)
	if match_size == True:
			goodphot = (phot['use_phot'] == 1) & (phot['f_'+filters[0]]/phot['e_'+filters[1]] >= 10) & (phot['f_'+filters[1]]/phot['e_'+filters[1]] >= 10) & (size['re']/size['dre'] >= 1) & (size['re'] > 0.4)
	phot = phot[goodphot]
	fast = fast[goodphot]

	for ii in range(0,len(z)):
		
		# iteration values
		itone = int(ii/3)
		ittwo = ii % 3
		
		# calculate median color
		# set fluxes < 0 equal to 1E-60. this is fine in calculating median, unless 
		# more than half of the galaxies had flux < 0!
		in_zbin = (fast['z'] >= zbins[ii,0]) & (fast['z'] < zbins[ii,1])
		obs_flux1 = phot['f_'+filters[0]][in_zbin]
		obs_flux1[obs_flux1 < 0] = 1E-30
		obs_flux2 = phot['f_'+filters[1]][in_zbin]
		obs_flux2[obs_flux1 < 0] = 1E-30
		
		obs_median_color = np.median((-2.5*np.log10(obs_flux1))-(-2.5*np.log10(obs_flux2)))
		
		for jj in range(0,len(ltau)):
			
			# calculate magnitudes and colors as a function of age at fixed tau, redshift [MODIFY TO INCLUDE REDSHIFT]
			mag1, luminosity1 = sed_to_mag(model_loc+str(ltau[jj]), filters[0],z=z[ii])
			mag2, luminosity2 = sed_to_mag(model_loc+str(ltau[jj]), filters[1],z=z[ii])
			color = mag1-mag2
			
			# calculate mass to light here
			spec_lam, spectra, log_age, log_mass,log_lbol, log_sfr = load_spec_output(model_loc+str(ltau[jj])+'.spec')
			mass2light = np.log10(10**log_mass/luminosity2)
			
			# only plot ages that we use in FAST
			LOG_AGE_MIN    = 7.6
			LOG_AGE_MAX    = 10.1
			good_age = (log_age >= LOG_AGE_MIN) & (log_age <= LOG_AGE_MAX)
			
			# put into delta-space
			# color first
			median_color = color[good_age]-obs_median_color
			
			# now M/L
			interp_models = interp1d(color[good_age],mass2light[good_age])
			# if median M/L is not covered in tau model, don't plot the model
			if (np.min(color[good_age]) <= obs_median_color) and (np.max(color[good_age]) >= obs_median_color):
				zero_ml = interp_models(obs_median_color)
			
				# plot
				axarr[itone,ittwo].plot(median_color, mass2light[good_age]-zero_ml,'-o',color=colors[jj],alpha=0.7, lw=2.5, zorder=1)	
				axarr[itone,ittwo].axis(axlim)
			
			# labels (once) (taus)
			xt = axlim[1]-(axlim[1]-axlim[0])*0.07
			if ii == 0:
				axarr[itone,ittwo].text(xt,axlim[3]-(axlim[3]-axlim[2])*(0.05*(jj+1)+0.07),r'log($\tau$)='+str(ltau[jj]),color=colors[jj],horizontalalignment='right')
			
			if jj == 0:
				# labels (repeat) (redshift)
				axarr[itone,ittwo].text(xt,axlim[3]-(axlim[3]-axlim[2])*0.07,'z='+str(z[ii]), weight='semibold',horizontalalignment='right')
				axarr[itone,ittwo].text(axlim[0]+(axlim[1]-axlim[0])*0.05, axlim[3]-(axlim[3]-axlim[2])*0.07, 'median(color obs)='+str(obs_median_color)[:4],size='smaller')
		
		# axis labels
		if (itone > 0):
			axarr[itone,ittwo].set_xlabel(r'$\Delta$(m$_{\mathrm{'+filters[0]+'}}$-m$_{\mathrm{'+filters[1]+'}})$')
		else:
			axarr[itone,ittwo].set_xticklabels([])
		if (ii == 0 or ii == 3):
			axarr[itone,ittwo].set_ylabel(r'$\Delta$log(M/L$_{\mathrm{'+filters[1]+'}})$')
		else:
			axarr[itone,ittwo].set_yticklabels([])
	
	if match_size == True:
		plt.savefig('/Users/joel/code/python/mass_function/figures/aperture_color/m2l_model/delta_ml_'+filters[0]+'_'+filters[1]+'_match_size.png', bbox_inches='tight',dpi=300)
	else:	
		plt.savefig('/Users/joel/code/python/mass_function/figures/aperture_color/m2l_model/delta_ml_'+filters[0]+'_'+filters[1]+'.png', bbox_inches='tight',dpi=300)
	plt.close()
	
	
	
	
	
	
	
	
	
	
	
	
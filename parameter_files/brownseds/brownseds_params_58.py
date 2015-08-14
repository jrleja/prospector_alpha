import numpy as np
import os
from bsfh import priors, sedmodel
from astropy.cosmology import WMAP9
from astropy.io import fits
tophat = priors.tophat
logarithmic = priors.logarithmic

#############
# RUN_PARAMS
#############

run_params = {'verbose':True,
              'outfile':os.getenv('APPS')+'/threedhst_bsfh/results/brownseds/brownseds',
              'ftol':0.5e-5, 
              'maxfev':5000,
              'nwalkers':496,
              'nburn':[32,32,64], 
              'niter': 1000,
              'initial_disp':0.1,
              'debug': False,
              'spec': False, 
              'phot':True,
              'datname':os.getenv('APPS')+'/threedhst_bsfh/data/brownseds_data/photometry/table1.fits',
              'photname':os.getenv('APPS')+'/threedhst_bsfh/data/brownseds_data/photometry/table3.fits',
              'extinctname':os.getenv('APPS')+'/threedhst_bsfh/data/brownseds_data/photometry/table4.fits',
              'herschname':os.getenv('APPS')+'/threedhst_bsfh/data/brownseds_data/photometry/kingfish.brownapertures.flux.fits',
              'objname':'NGC 4365',
              }

############
# OBS
#############

def translate_filters(bfilters):
    '''
    translate filter names to FSPS standard
    this is ALREADY a mess, clean up soon!
    suspect there are smarter routines to do this in python-fsps
    '''

    # this is necessary for my code
    # to calculate effective wavelength
    # in threed_dutils
    translate = {
    'FUV': 'GALEX FUV',
    'UVW2': 'UVOT w2',
    'UVM2': 'UVOT m2',
    'NUV': 'GALEX NUV',
    'UVW1': 'UVOT w1',
    'Umag': np.nan,    # [11.9/15.7]? Swift/UVOT U AB band magnitude
    'umag': 'SDSS Camera u Response Function, airmass = 1.3 (June 2001)',
    'gmag': 'SDSS Camera g Response Function, airmass = 1.3 (June 2001)',
    'Vmag': np.nan,    # [10.8/15.6]? Swift/UVOT V AB band magnitude
    'rmag': 'SDSS Camera r Response Function, airmass = 1.3 (June 2001)',
    'imag': 'SDSS Camera i Response Function, airmass = 1.3 (June 2001)',
    'zmag': 'SDSS Camera z Response Function, airmass = 1.3 (June 2001)',
    'Jmag': '2MASS J filter (total response w/atm)',
    'Hmag': '2MASS H filter (total response w/atm)',
    'Ksmag': '2MASS Ks filter (total response w/atm)',
    'W1mag': 'WISE W1',
    '[3.6]': 'IRAC Channel 1',
    '[4.5]': 'IRAC Channel 2',
    'W2mag': 'WISE W2',
    '[5.8]': 'IRAC Channel 3',
    '[8.0]': 'IRAC CH4',
    'W3mag': 'WISE W3',
    'PUIB': np.nan,    # [8.2/15.6]? Spitzer/IRS Blue Peak Up Imaging channel (13.3-18.7um) AB magnitude
    'W4mag': np.nan,    # two WISE4 magnitudes, what is the correction?
    "W4'mag": 'WISE W4',
    'PUIR': np.nan,    # Spitzer/IRS Red Peak Up Imaging channel (18.5-26.0um) AB magnitude
    '[24]': 'MIPS 24um',
    'pacs70': 'Herschel PACS 70um',
    'pacs100': 'Herschel PACS 100um',
    'pacs160': 'Herschel PACS 160um',
    'spire250': 'Herschel SPIRE 250um',
    'spire350': 'Herschel SPIRE 350um',
    'spire500': 'Herschel SPIRE 500um'
    }

    # this translates filter names
    # to names that FSPS recognizes
    translate_pfsps = {
    'FUV': 'GALEX_FUV',
    'UVW2': 'UVOT_W2',
    'UVM2': 'UVOT_M2',
    'NUV': 'GALEX_NUV',
    'UVW1': 'UVOT_W1',
    'Umag': np.nan,    # [11.9/15.7]? Swift/UVOT U AB band magnitude
    'umag': 'SDSS_u',
    'gmag': 'SDSS_g',
    'Vmag': np.nan,    # [10.8/15.6]? Swift/UVOT V AB band magnitude
    'rmag': 'SDSS_r',
    'imag': 'SDSS_i',
    'zmag': 'SDSS_z',
    'Jmag': '2MASS_J',
    'Hmag': '2MASS_H',
    'Ksmag': '2MASS_Ks',
    'W1mag': 'WISE_W1',
    '[3.6]': 'IRAC_1',
    '[4.5]': 'IRAC_2',
    'W2mag': 'WISE_W2',
    '[5.8]': 'IRAC_3',
    '[8.0]': 'IRAC_4',
    'W3mag': 'WISE_W3',
    'PUIB': np.nan,    # [8.2/15.6]? Spitzer/IRS Blue Peak Up Imaging channel (13.3-18.7um) AB magnitude
    'W4mag': np.nan,    # two WISE4 magnitudes, what is the correction?
    "W4'mag": 'WISE_W4',
    'PUIR': np.nan,    # Spitzer/IRS Red Peak Up Imaging channel (18.5-26.0um) AB magnitude
    '[24]': 'MIPS_24',
    'pacs70': 'PACS_70',
    'pacs100': 'PACS_100',
    'pacs160': 'PACS_160',
    'spire250': 'SPIRE_250',
    'spire350': 'SPIRE_350',
    'spire500': 'SPIRE_500'
    }

    return np.array([translate[f] for f in bfilters]), np.array([translate_pfsps[f] for f in bfilters])

def load_obs_brown(photname, extinctname, herschname, objname):
    """
    let's do this
    """
    obs ={}

    # load photometry
    hdulist = fits.open(photname)

    # find object
    if objname is not None:
        idx = hdulist[1].data['Name'] == objname
    else:
        idx = np.ones(len(hdulist[1].data['Name']),dtype=bool)
    
    # extract fluxes+uncertainties for all objects
    mag_fields = [f for f in hdulist[1].columns.names if (f[0:2] != 'e_') and (f != 'Name')]
    magunc_fields = [f for f in hdulist[1].columns.names if f[0:2] == 'e_']

    # extract fluxes for particular object
    mag = np.array([np.squeeze(hdulist[1].data[f][idx]) for f in mag_fields])
    magunc  = np.array([np.squeeze(hdulist[1].data[f][idx]) for f in magunc_fields])

    # extinctions
    extinct = fits.open(extinctname)
    extinctions = np.array([np.squeeze(extinct[1].data[f][idx]) for f in extinct[1].columns.names if f != 'Name'])

    # adjust fluxes for extinction
    # then convert to maggies
    mag_adj = mag - extinctions
    flux = 10**((-2./5)*mag_adj)

    # convert uncertainty to maggies
    unc = magunc*flux/1.086

    #### Herschel photometry
    herschel = fits.open(herschname)
    
    # find interesting fields
    hflux_fields = [f for f in herschel[1].columns.names if (('pacs' in f) or ('spire' in f)) and f[-3:] != 'unc']
    hunc_fields = [f for f in herschel[1].columns.names if (('pacs' in f) or ('spire' in f)) and f[-3:] == 'unc']

    # different versions if objname is passed or no
    if objname is not None:
        match = herschel[1].data['Name'] == objname.lower().replace(' ','')
        
        hflux = np.array([np.squeeze(herschel[1].data[match][hflux_fields[i]]) for i in xrange(len(hflux_fields))])
        hunc = np.array([np.squeeze(herschel[1].data[match][f]) for f in hunc_fields])
    else:
        optnames = hdulist[1].data['Name']
        hnames   = herschel[1].data['Name']

        # non-pythonic, i know, but why change it if it works?
        hflux,hunc = np.zeros(shape=(len(hflux_fields),len(hnames))), np.zeros(shape=(len(hflux_fields),len(hnames)))
        for ii in xrange(len(optnames)):
            match = hnames == optnames[ii].lower().replace(' ','')
            for kk in xrange(len(hflux_fields)):
                hflux[kk,ii] = herschel[1].data[match][hflux_fields[kk]]
                hunc[kk,ii]  = herschel[1].data[match][hunc_fields[kk]]

    # 5% error floor
    hunc = np.clip(hunc, hflux*0.05, np.inf)

    #### combine with brown catalog
    # convert from Jy to maggies
    flux = np.concatenate((flux,hflux/3631.))   
    unc = np.concatenate((unc, hunc/3631.))
    mag_fields = np.append(mag_fields,hflux_fields)   

    # phot mask
    phot_mask_brown = mag != 0
    phot_mask_hersch = hflux != 0
    phot_mask = np.concatenate((phot_mask_brown,phot_mask_hersch))

    # map brown filters to FSPS filters
    # and remove fields where we don't have filter definitions
    filters,fsps_filters = translate_filters(mag_fields)
    have_definition = np.array(filters) != 'nan'

    filters = filters[have_definition]
    fsps_filters = fsps_filters[have_definition]
    flux = flux[have_definition]
    unc = unc[have_definition]
    phot_mask = phot_mask[have_definition]

    # load wave_effective
    from translate_filter import calc_lameff_for_fsps
    wave_effective = calc_lameff_for_fsps(filters)

    # build output dictionary
    obs['wave_effective'] = wave_effective
    obs['filters'] = fsps_filters
    obs['phot_mask'] = phot_mask
    obs['maggies'] = flux
    obs['maggies_unc'] =  unc
    obs['wavelength'] = None
    obs['spectrum'] = None

    if objname is None:
        obs['hnames'] = herschel[1].data['Name']
        obs['names'] = hdulist[1].data['Name']

    # tidy up
    hdulist.close()
    extinct.close()
    herschel.close()

    return obs

obs = load_obs_brown(run_params['photname'], run_params['extinctname'], 
                     run_params['herschname'],run_params['objname'])

#############
# MODEL_PARAMS
#############

def transform_sftanslope_to_sfslope(sf_slope=None,sf_tanslope=None,**extras):

    return np.tan(sf_tanslope)

def transform_delt_to_sftrunc(tage=None, delt_trunc=None, **extras):

    return tage*delt_trunc

def transform_logtau_to_tau(tau=None, logtau=None, **extras):

    return 10**logtau

def add_dust1(dust2=None, **extras):

    return 0.86*dust2

class BurstyModel(sedmodel.CSPModel):

    def theta_disps(self, thetas, initial_disp=0.1):
        """Get a vector of dispersions for each parameter to use in
        generating sampler balls for emcee's Ensemble sampler.

        :param initial_disp: (default: 0.1)
            The default dispersion to use in case the `init_disp` key
            is not provided in the parameter configuration.  This is
            in units of the parameter, so e.g. 0.1 will result in a
            smpler ball with a dispersion that is 10% of the central
            parameter value.
        """
        disp = np.zeros(self.ndim) + initial_disp
        for par, inds in self.theta_index.iteritems():
            
            # fractional dispersion
            if par == 'mass' or \
               par == 'tage':
                disp[inds[0]:inds[1]] = self._config_dict[par].get('init_disp', initial_disp) * thetas[inds[0]:inds[1]]

            # constant (log) dispersion
            if par == 'logtau' or \
               par == 'metallicity' or \
               par == 'sf_tanslope' or \
               par == 'delt_trunc' or \
               par == 'duste_umin' or \
               par == 'duste_qpah' or \
               par == 'duste_gamma':
                disp[inds[0]:inds[1]] = self._config_dict[par].get('init_disp', initial_disp)

            # fractional dispersion with artificial floor
            if par == 'dust2' or \
               par == 'dust_index':
                disp[inds[0]:inds[1]] = (self._config_dict[par].get('init_disp', initial_disp) * thetas[inds[0]:inds[1]]**2 + \
                                         0.1**2)**0.5
            
        return disp

    def theta_disp_floor(self, thetas):
        """Get a vector of dispersions for each parameter to use as
        a floor for the walker-calculated dispersions.
        """
        disp = np.zeros(self.ndim)
        for par, inds in self.theta_index.iteritems():
            
            # constant 5% floor
            if par == 'mass':
                disp[inds[0]:inds[1]] = 0.05 * thetas[inds[0]:inds[1]]

            # constant 0.05 floor (log space, sf_slope, dust_index)
            if par == 'logzsol':
                disp[inds[0]:inds[1]] = 0.25

            if par == 'logtau':
                disp[inds[0]:inds[1]] = 0.25

            if par == 'sf_tanslope':
                disp[inds[0]:inds[1]] = 0.3

            if par == 'dust2' or \
               par == 'dust_index':
                disp[inds[0]:inds[1]] = 0.15

            if par == 'duste_umin':
                disp[inds[0]:inds[1]] = 4.5

            if par == 'duste_qpah':
                disp[inds[0]:inds[1]] = 3.0

            if par == 'duste_gamma':
                disp[inds[0]:inds[1]] = 0.2

            # 20% floor
            if par == 'tage':
                disp[inds[0]:inds[1]] = 0.2 * thetas[inds[0]:inds[1]]

            if par == 'delt_trunc':
                disp[inds[0]:inds[1]] = 0.1
            
        return disp

    def prior_product(self, theta):
        """
        Return a scalar which is the ln of the product of the prior
        probabilities for each element of theta.  Requires that the
        prior functions are defined in the theta descriptor.

        :param theta:
            Iterable containing the free model parameter values.

        :returns lnp_prior:
            The log of the product of the prior probabilities for
            these parameter values.
        """  
        lnp_prior = 0

        # implement uniqueness of outliers
        if 'gp_outlier_locs' in self.theta_index:
            start,end = self.theta_index['gp_outlier_locs']
            outlier_locs = theta[start:end]
            if len(np.unique(np.round(outlier_locs))) != len(outlier_locs):
                return -np.inf

        for k, v in self.theta_index.iteritems():
            start, end = v
            lnp_prior += np.sum(self._config_dict[k]['prior_function']
                                (theta[start:end], **self._config_dict[k]['prior_args']))
        return lnp_prior

##### if we have KINGFISH imaging,
##### leave dust parameters free
if 'SPIRE_500' in obs['filters']:
    dust_variable = True
else:
    dust_variable = False


#### SET SFH PRIORS #####
###### REDSHIFT ######
hdulist = fits.open(run_params['datname'])
idx = hdulist[1].data['Name'] == run_params['objname']
zred =  hdulist[1].data['cz'][idx][0] / 3e5
hdulist.close()

#### TUNIV #####
tuniv = WMAP9.age(zred).value
run_params['tuniv']       = tuniv

#### TAGE #####
tage_maxi = tuniv
tage_init = tuniv/2.
tage_mini  = 0.11      # FSPS standard

model_type = BurstyModel
model_params = []

param_template = {'name':'', 'N':1, 'isfree': False,
                  'init':0.0, 'units':'', 'label':'',
                  'prior_function_name': None, 'prior_args': None}

###### BASIC PARAMETERS ##########
model_params.append({'name': 'zred', 'N': 1,
                        'isfree': False,
                        'init': zred,
                        'units': '',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.0, 'maxi':4.0}})

model_params.append({'name': 'add_igm_absorption', 'N': 1,
                        'isfree': False,
                        'init': 1,
                        'units': None,
                        'prior_function': None,
                        'prior_args': None})

model_params.append({'name': 'add_agb_dust_model', 'N': 1,
                        'isfree': False,
                        'init': True,
                        'units': None,
                        'prior_function': None,
                        'prior_args': None})
                        
model_params.append({'name': 'mass', 'N': 1,
                        'isfree': True,
                        'init': 1e10,
                        'init_disp': 0.25,
                        'units': r'M_\odot',
                        'prior_function': tophat,
                        'prior_args': {'mini':1e7,'maxi':1e14}})

model_params.append({'name': 'pmetals', 'N': 1,
                        'isfree': False,
                        'init': -99,
                        'units': '',
                        'prior_function': None,
                        'prior_args': {'mini':-3, 'maxi':-1}})

model_params.append({'name': 'logzsol', 'N': 1,
                        'isfree': True,
                        'init': -0.1,
                        'init_disp': 0.15,
                        'units': r'$\log (Z/Z_\odot)$',
                        'prior_function': tophat,
                        'prior_args': {'mini':-1.98, 'maxi':0.19}})
                        
###### SFH   ########
model_params.append({'name': 'sfh', 'N': 1,
                        'isfree': False,
                        'init': 5,
                        'units': 'type',
                        'prior_function_name': None,
                        'prior_args': None})

model_params.append({'name': 'logtau', 'N': 1,
                        'isfree': True,
                        'init': 0.0,
                        'init_disp': 0.25,
                        'units': 'log(Gyr)',
                        'prior_function': tophat,
                        'prior_args': {'mini':-1.0,
                                       'maxi':2.0}})

model_params.append({'name': 'tau', 'N': 1,
                        'isfree': False,
                        'init': 1.0,
                        'depends_on': transform_logtau_to_tau,
                        'units': 'Gyr',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.1,
                                       'maxi':100}})

model_params.append({'name': 'tage', 'N': 1,
                        'isfree': True,
                        'init': tage_init,
                        'init_disp': 0.25,
                        'units': 'Gyr',
                        'prior_function': tophat,
                        'prior_args': {'mini':tage_mini, 'maxi':tage_maxi}})

model_params.append({'name': 'tburst', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
                        'init_disp': 1.0,
                        'units': '',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.0, 'maxi':10.0}})

model_params.append({'name': 'fburst', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
                        'init_disp': 0.5,
                        'units': '',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.0, 'maxi':0.2}})

model_params.append({'name': 'fconst', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
                        'units': '',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.0, 'maxi':1.0}})

model_params.append({'name': 'sf_start', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
                        'units': 'Gyr',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.0,'maxi':14.0}})

model_params.append({'name': 'delt_trunc', 'N': 1,
                        'isfree': True,
                        'init': 0.5,
                        'init_disp': 0.1,
                        'units': '',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.0, 'maxi': 1.0}})

model_params.append({'name': 'sf_trunc', 'N': 1,
                        'isfree': False,
                        'init': 1.0,
                        'units': '',
                        'depends_on': transform_delt_to_sftrunc,
                        'prior_function': tophat,
                        'prior_args': {'mini':0, 'maxi':16}})

model_params.append({'name': 'sf_tanslope', 'N': 1,
                        'isfree': True,
                        'init': 0.0,
                        'init_disp': 0.25,
                        'units': '',
                        'prior_function': tophat,
                        'prior_args': {'mini':-np.pi/2., 'maxi': np.pi/3.}})

model_params.append({'name': 'sf_slope', 'N': 1,
                        'isfree': False,
                        'init': -1.0,
                        'units': '',
                        'depends_on': transform_sftanslope_to_sfslope,
                        'prior_function': tophat,
                        'prior_args': {'mini':-np.inf,'maxi':5.0}})

########    IMF  ##############
model_params.append({'name': 'imf_type', 'N': 1,
                             'isfree': False,
                             'init': 1, #1 = chabrier
                             'units': None,
                             'prior_function_name': None,
                             'prior_args': None})

######## Dust Absorption ##############
model_params.append({'name': 'dust_type', 'N': 1,
                        'isfree': False,
                        'init': 4,
                        'units': 'index',
                        'prior_function_name': None,
                        'prior_args': None})
                        
model_params.append({'name': 'dust1', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
                        'depends_on': add_dust1,
                        'units': '',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.0, 'maxi':8.0}})

model_params.append({'name': 'dust2', 'N': 1,
                        'isfree': True,
                        'init': 1.0,
                        'units': '',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.0,'maxi':4.0}})

model_params.append({'name': 'dust_index', 'N': 1,
                        'isfree': True,
                        'init': -0.7,
                        'units': '',
                        'prior_function': priors.normal_clipped,
                        'prior_args': {'mini':-3.0, 'maxi': -0.4,'mean':-0.7,'sigma':0.5}})

model_params.append({'name': 'dust1_index', 'N': 1,
                        'isfree': False,
                        'init': -1.0,
                        'units': '',
                        'prior_function': tophat,
                        'prior_args': {'mini':-1.5, 'maxi':-0.5}})

model_params.append({'name': 'dust_tesc', 'N': 1,
                        'isfree': False,
                        'init': 7.0,
                        'units': 'log(Gyr)',
                        'prior_function_name': None,
                        'prior_args': None})

###### Dust Emission ##############
model_params.append({'name': 'add_dust_emission', 'N': 1,
                        'isfree': False,
                        'init': 1,
                        'units': None,
                        'prior_function': None,
                        'prior_args': None})

model_params.append({'name': 'duste_gamma', 'N': 1,
                        'isfree': dust_variable,
                        'init': 0.01,
                        'init_disp': 0.2,
                        'units': None,
                        'prior_function': tophat,
                        'prior_args': {'mini':0.0, 'maxi':1.0}})

model_params.append({'name': 'duste_umin', 'N': 1,
                        'isfree': dust_variable,
                        'init': 1.0,
                        'init_disp': 5.0,
                        'units': None,
                        'prior_function': tophat,
                        'prior_args': {'mini':0.1, 'maxi':25.0}})

model_params.append({'name': 'duste_qpah', 'N': 1,
                        'isfree': dust_variable,
                        'init': 3.0,
                        'init_disp': 3.0,
                        'units': 'percent',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.0, 'maxi':10.0}})

###### Nebular Emission ###########
model_params.append({'name': 'add_neb_emission', 'N': 1,
                        'isfree': False,
                        'init': 2,
                        'units': r'log Z/Z_\odot',
                        'prior_function_name': None,
                        'prior_args': None})

model_params.append({'name': 'add_neb_continuum', 'N': 1,
                        'isfree': False,
                        'init': True,
                        'units': r'log Z/Z_\odot',
                        'prior_function_name': None,
                        'prior_args': None})
                        
model_params.append({'name': 'gas_logz', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
                        'units': r'log Z/Z_\odot',
                        'prior_function': tophat,
                        'prior_args': {'mini':-2.0, 'maxi':0.5}})

model_params.append({'name': 'gas_logu', 'N': 1,
                        'isfree': False,
                        'init': -2.0,
                        'units': '',
                        'prior_function': tophat,
                        'prior_args': {'mini':-4, 'maxi':-1}})


####### Calibration ##########
model_params.append({'name': 'phot_jitter', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
                        'init_disp': 0.5,
                        'units': 'fractional maggies (mags/1.086)',
                        'prior_function':tophat,
                        'prior_args': {'mini':0.0, 'maxi':0.5}})

# name outfile
run_params['outfile'] = run_params['outfile']+'_'+run_params['objname']

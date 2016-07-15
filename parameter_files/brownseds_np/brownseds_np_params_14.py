import numpy as np
import os
from prospect.models import priors, sedmodel
from prospect.sources import StepSFHBasis
from sedpy import observate
from astropy.cosmology import WMAP9
from astropy.io import fits
tophat = priors.tophat
logarithmic = priors.logarithmic

#############
# RUN_PARAMS
#############

run_params = {'verbose':True,
              'debug': False,
              'outfile': os.getenv('APPS')+'/threedhst_bsfh/results/brownseds_np/brownseds_np',
              'nofork': True,
              # Optimizer params
              'ftol':0.5e-5, 
              'maxfev':5000,
              # MCMC params
              'nwalkers':546,
              'nburn':[150,200,400], 
              'niter': 2000,
              # Model info
              'zcontinuous': 2,
              'compute_vega_mags': False,
              'initial_disp':0.1,
              'interp_type': 'logarithmic',
              'agelims': [0.0,8.0,8.5,9.0,9.5,10.0],
              # Data info
              'datname':os.getenv('APPS')+'/threedhst_bsfh/data/brownseds_data/photometry/table1.fits',
              'photname':os.getenv('APPS')+'/threedhst_bsfh/data/brownseds_data/photometry/table3.fits',
              'extinctname':os.getenv('APPS')+'/threedhst_bsfh/data/brownseds_data/photometry/table4.fits',
              'herschname':os.getenv('APPS')+'/threedhst_bsfh/data/brownseds_data/photometry/kingfish.brownapertures.flux.fits',
              'objname':'NGC 1068',
              }
run_params['outfile'] = run_params['outfile']+'_'+run_params['objname']

############
# OBS
#############

def translate_filters(bfilters, full_list = False):
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
    'FUV': 'galex_FUV',
    'UVW2': 'uvot_w2',
    'UVM2': 'uvot_m2',
    'NUV': 'galex_NUV',
    'UVW1': 'uvot_w1',
    'Umag': np.nan,    # [11.9/15.7]? Swift/UVOT U AB band magnitude
    'umag': 'sdss_u0',
    'gmag': 'sdss_g0',
    'Vmag': np.nan,    # [10.8/15.6]? Swift/UVOT V AB band magnitude
    'rmag': 'sdss_r0',
    'imag': 'sdss_i0',
    'zmag': 'sdss_z0',
    'Jmag': 'twomass_J',
    'Hmag': 'twomass_H',
    'Ksmag': 'twomass_Ks',
    'W1mag': 'wise_w1',
    '[3.6]': 'spitzer_irac_ch1',
    '[4.5]': 'spitzer_irac_ch2',
    'W2mag': 'WISE_W2',
    '[5.8]': 'spitzer_irac_ch3',
    '[8.0]': 'spitzer_irac_ch4',
    'W3mag': 'WISE_W3',
    'PUIB': np.nan,    # [8.2/15.6]? Spitzer/IRS Blue Peak Up Imaging channel (13.3-18.7um) AB magnitude
    'W4mag': np.nan,    # two WISE4 magnitudes, this one is "native" and must be corrected
    "W4'mag": 'WISE_W4',
    'PUIR': np.nan,    # Spitzer/IRS Red Peak Up Imaging channel (18.5-26.0um) AB magnitude
    '[24]': 'spitzer_mips_24',
    'pacs70': 'herschel_pacs_70',
    'pacs100': 'herschel_pacs_100',
    'pacs160': 'herschel_pacs_160',
    'spire250': 'herschel_spire_250',
    'spire350': 'herschel_spire_350',
    'spire500': 'herschel_spire_500'
    }

    if full_list:
        return translate.values()
    else:
        return np.array([translate[f] for f in bfilters]), np.array([translate_pfsps[f] for f in bfilters])

def load_obs(photname='', extinctname='', herschname='', objname='', **extras):
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
    mag_adj = mag - extinctions

    # add correction to MIPS magnitudes (only MIPS 24 right now!)
    mips_corr = np.array([-0.03542,-0.07669,-0.03807]) # 24, 70, 160
    mag_adj[mag_fields.index('[24]') ] += mips_corr[0]

    # then convert to maggies
    flux = 10**((-2./5)*mag_adj)

    # convert uncertainty to maggies
    unc = magunc*flux/1.086

    #### Herschel photometry
    # find fluxes + errors
    herschel = fits.open(herschname)
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

        # non-pythonic, but why change if it works?
        hflux,hunc = np.zeros(shape=(len(hflux_fields),len(hnames))), np.zeros(shape=(len(hflux_fields),len(hnames)))
        for ii in xrange(len(optnames)):
            match = hnames == optnames[ii].lower().replace(' ','')
            for kk in xrange(len(hflux_fields)):
                hflux[kk,ii] = herschel[1].data[match][hflux_fields[kk]]
                hunc[kk,ii]  = herschel[1].data[match][hunc_fields[kk]]

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
    # and remove fluxes where we don't have filter definitions
    filters,fsps_filters = translate_filters(mag_fields)
    have_definition = np.array(filters) != 'nan'

    filters = filters[have_definition]
    fsps_filters = fsps_filters[have_definition]
    flux = flux[have_definition]
    unc = unc[have_definition]
    phot_mask = phot_mask[have_definition]

    # implement error floor
    unc = np.clip(unc, flux*0.05, np.inf)

    # build output dictionary
    obs['filters'] = observate.load_filters(fsps_filters)
    obs['wave_effective'] = np.array([filt.wave_effective for filt in obs['filters']])
    obs['phot_mask'] = phot_mask
    obs['maggies'] = flux
    obs['maggies_unc'] =  unc
    obs['wavelength'] = None
    obs['spectrum'] = None
    obs['logify_spectrum'] = False

    if objname is None:
        obs['hnames'] = herschel[1].data['Name']
        obs['names'] = hdulist[1].data['Name']

    # tidy up
    hdulist.close()
    extinct.close()
    herschel.close()
    return obs

def expsfh(agelims, tau=1e5, power=1, **extras):
    """
    Calculate the mass in a set of step functions that is equivalent to an
    exponential SFH.  That is, \int_amin^amax \, dt \, e^(-t/\tau) where
    amin,amax are the age limits of the bins making up the step function.
    """
    from scipy.special import gamma, gammainc
    tage = 10**np.max(agelims) / 1e9
    t = tage - 10**np.array(agelims)/1e9
    nb = len(t)
    mformed = np.zeros(nb-1)
    t = np.insert(t, 0, tage)
    for i in range(nb-1):
        t1, t2 = t[i+1], t[i]
        normalized_times = (np.array([t1, t2, tage])[:, None]) / tau
        mass = gammainc(power, normalized_times)
        intsfr = (mass[1,...] - mass[0,...]) / mass[2,...]
        mformed[i] = intsfr
    return mformed * 1e3
        
######################
# GENERATING FUNCTIONS
######################
def transform_logmass_to_mass(mass=None, logmass=None, **extras):

    return 10**logmass

def load_gp(**extras):
    return None, None

def add_dust1(dust2=None, **extras):

    return 0.86*dust2

def tie_gas_logz(logzsol=None, **extras):

    return logzsol

#############
# MODEL_PARAMS
#############

model_params = []

###### BASIC PARAMETERS ##########
model_params.append({'name': 'zred', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
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

model_params.append({'name': 'pmetals', 'N': 1,
                        'isfree': False,
                        'init': -99,
                        'units': '',
                        'prior_function': None,
                        'prior_args': {'mini':-3, 'maxi':-1}})

model_params.append({'name': 'logzsol', 'N': 1,
                        'isfree': True,
                        'init': -0.5,
                        'init_disp': 0.25,
                        'disp_floor': 0.2,
                        'units': r'$\log (Z/Z_\odot)$',
                        'prior_function': tophat,
                        'prior_args': {'mini':-1.98, 'maxi':0.19}})
                        
###### SFH   ########
model_params.append({'name': 'sfh', 'N':1,
                        'isfree': False,
                        'init': 0,
                        'units': None})

model_params.append({'name': 'logmass', 'N': 1,
                        'isfree': True,
                        'init': 10.0,
                        'units': 'Msun',
                        'prior_function': priors.tophat,
                        'prior_args':{'mini':1.0, 'maxi':14.0}})

model_params.append({'name': 'mass', 'N': 1,
                        'isfree': False,
                        'init': 1e10,
                        'depends_on': transform_logmass_to_mass,
                        'units': 'Msun',
                        'prior_function': priors.tophat,
                        'prior_args':{'mini':1e1, 'maxi':1e14}})

model_params.append({'name': 'agebins', 'N': 1,
                        'isfree': False,
                        'init': [],
                        'units': 'log(yr)',
                        'prior_function': priors.tophat,
                        'prior_args':{'mini':0.1, 'maxi':15.0}})

model_params.append({'name': 'sfr_fraction', 'N': 1,
                        'isfree': True,
                        'init': [],
                        'units': 'Msun',
                        'prior_function': priors.tophat,
                        'prior_args':{'mini':0.0, 'maxi':1.0}})

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
                        'isfree': True,
                        'init': 1.0,
                        'init_disp': 0.8,
                        'disp_floor': 0.5,
                        'units': '',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.0, 'maxi':4.0}})

model_params.append({'name': 'dust2', 'N': 1,
                        'isfree': True,
                        'init': 1.0,
                        'init_disp': 0.25,
                        'disp_floor': 0.15,
                        'units': '',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.0,'maxi':4.0}})

model_params.append({'name': 'dust_index', 'N': 1,
                        'isfree': True,
                        'init': 0.0,
                        'init_disp': 0.25,
                        'disp_floor': 0.15,
                        'units': '',
                        'prior_function': tophat,
                        'prior_args': {'mini':-2.2, 'maxi': 0.4}})

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
                        'isfree': True,
                        'init': 0.01,
                        'init_disp': 0.2,
                        'disp_floor': 0.15,
                        'units': None,
                        'prior_function': tophat,
                        'prior_args': {'mini':0.0, 'maxi':1.0}})

model_params.append({'name': 'duste_umin', 'N': 1,
                        'isfree': True,
                        'init': 1.0,
                        'init_disp': 5.0,
                        'disp_floor': 4.5,
                        'units': None,
                        'prior_function': tophat,
                        'prior_args': {'mini':0.1, 'maxi':25.0}})

model_params.append({'name': 'duste_qpah', 'N': 1,
                        'isfree': True,
                        'init': 3.0,
                        'init_disp': 3.0,
                        'disp_floor': 3.0,
                        'units': 'percent',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.0, 'maxi':10.0}})

###### Nebular Emission ###########
model_params.append({'name': 'add_neb_emission', 'N': 1,
                        'isfree': False,
                        'init': True,
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
                        'depends_on': tie_gas_logz,
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

####### Units ##########
model_params.append({'name': 'peraa', 'N': 1,
                     'isfree': False,
                     'init': False})

model_params.append({'name': 'mass_units', 'N': 1,
                     'isfree': False,
                     'init': 'mstar'})

#### resort list of parameters 
#### so that major ones are fit first
parnames = [m['name'] for m in model_params]
fit_order = ['logmass','sfr_fraction','dust2', 'logzsol', 'dust_index', 'dust1', 'duste_qpah', 'duste_gamma', 'duste_umin']
tparams = [model_params[parnames.index(i)] for i in fit_order]
for param in model_params: 
    if param['name'] not in fit_order:
        tparams.append(param)
model_params = tparams

###### REDEFINE MODEL FOR MY OWN NEFARIOUS PURPOSES ######
class BurstyModel(sedmodel.SedModel):

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

        # dust1/dust2 ratio
        if 'dust1' in self.theta_index:
            if 'dust2' in self.theta_index:
                start,end = self.theta_index['dust1']
                dust1 = theta[start:end]
                start,end = self.theta_index['dust2']
                dust2 = theta[start:end]
                if dust1/1.5 > dust2:
                    return -np.inf
                '''
                if dust1 < 0.5*dust2:
                    return -np.inf
                '''

        # sum of SFH fractional bins <= 1.0
        if 'sfr_fraction' in self.theta_index:
            start,end = self.theta_index['sfr_fraction']
            sfr_fraction = theta[start:end]
            if np.sum(sfr_fraction) > 1.0:
                return -np.inf

        for k, v in self.theta_index.iteritems():
            start, end = v
            this_prior = np.sum(self._config_dict[k]['prior_function']
                                (theta[start:end], **self._config_dict[k]['prior_args']))

            if (not np.isfinite(this_prior)):
                print('WARNING: ' + k + ' is out of bounds')
            lnp_prior += this_prior
        return lnp_prior

class FracSFH(StepSFHBasis):
    
    @property
    def all_ssp_weights(self):
        # Cache age bins and relative weights.  This means params['agebins']
        # *must not change* without also setting _ages = None
        if getattr(self, '_ages', None) is None:
            self._ages = self.params['agebins']
            nbin, nssp = len(self._ages), len(self.logage) + 1
            self._bin_weights = np.zeros([nbin, nssp])
            self._time_per_bin = np.zeros(nbin)
            for i, (t1, t2) in enumerate(self._ages):
                # These *should* sum to one (or zero) for each bin
                self._bin_weights[i,:] = self.bin_weights(t1, t2)
                self._time_per_bin[i] = 10**t2-10**t1

        # Now normalize the weights in each bin by the massfrac parameter, and sum
        # over bins.
        bin_masses = np.array(self.params['sfr_fraction'])
        bin_masses = np.append(bin_masses,(1-np.sum(self.params['sfr_fraction'])))*self._time_per_bin
        if np.all(self.params.get('mass_units', 'mstar') == 'mstar'):
            # Convert from mstar to mformed for each bin.  We have to do this
            # here as well as in get_spectrum because the *relative*
            # normalization in each bin depends on the units, as well as the
            # overall normalization.
            bin_masses /= self.bin_mass_fraction
        w = (bin_masses[:, None] * self._bin_weights).sum(axis=0)

        return w

def load_sps(**extras):

    sps = FracSFH(**extras)
    return sps

def load_model(objname='',datname='', agelims=[], **extras):

    ###### REDSHIFT ######
    hdulist = fits.open(datname)
    idx = hdulist[1].data['Name'] == objname
    zred =  hdulist[1].data['cz'][idx][0] / 3e5
    hdulist.close()

    #### CALCULATE TUNIV #####
    tuniv = WMAP9.age(zred).value

    #### NONPARAMETRIC SFH ######
    agelims[-1] = np.log10(tuniv*1e9)
    agebins = np.array([agelims[:-1], agelims[1:]])
    ncomp = len(agelims) - 1
    mass_init =  expsfh(agelims, **extras)*1e5

    #### ADJUST MODEL PARAMETERS #####
    n = [p['name'] for p in model_params]

    #### SET UP AGEBINS
    model_params[n.index('agebins')]['N'] = ncomp
    model_params[n.index('agebins')]['init'] = agebins.T

    #### FRACTIONAL MASS
    # N-1 bins, last is set by x = 1 - np.sum(sfr_fraction)
    model_params[n.index('sfr_fraction')]['N'] = ncomp-1
    model_params[n.index('sfr_fraction')]['init'] = mass_init[:-1] / np.sum(mass_init)
    model_params[n.index('sfr_fraction')]['prior_args'] = {
                                                           'maxi':np.full(ncomp-1,1.0), 
                                                           'mini':np.full(ncomp-1,0.0),
                                                           'alpha':1.0,
                                                           'alpha_sum':ncomp 
                                                           # NOTE: ncomp instead of ncomp-1 makes the prior take into account the implicit Nth variable too
                                                          }
    model_params[n.index('sfr_fraction')]['init_disp'] = 0.15

    #### INSERT REDSHIFT INTO MODEL PARAMETER DICTIONARY ####
    zind = n.index('zred')
    model_params[zind]['init'] = zred

    #### CREATE MODEL
    model = BurstyModel(model_params)

    return model

model_type = BurstyModel


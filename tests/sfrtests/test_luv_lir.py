import numpy as np
import fsps
from scipy.integrate import simps
from scipy.interpolate import interp1d

def integrate_spec(spec_lam,spectra,filter):

    resp_lam = filter[0][0]
    res      = filter[1][0]

    # interpolate filter response onto spectral lambda array
    # when interpolating, set response outside wavelength range to be zero.
    response_interp_function = interp1d(resp_lam,res, bounds_error = False, fill_value = 0)
    resp_interp = response_interp_function(spec_lam)
    
    # integrate spectrum over filter response
    luminosity = simps(spectra*resp_interp,spec_lam)

    return luminosity

sps = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=0)
w = sps.wavelengths

sps.params['sfh'] = 1 #composite
sps.params['imf_type'] = 1 # Chabrier
sps.params['const'] = 1.0 #all constant
sps.params['tage'] = 1.0 #1 Gyr old
sps.params['add_dust_emission'] = True

if len(sps.zlegend) > 10:
    # We are using BaSeL
    sps.params['zmet'] = 22
    sps.params['logzsol'] = 0.0
else:
    # we are using MILES
    sps.params['zmet'] = 4
    sps.params['logzsol'] = 0.0

ns = 1000 # set number of samples
gd = np.random.uniform(0, 4, ns) # random dust2 values

# output values
luv = np.zeros(ns)
lir = np.zeros(ns)
sfr_ratio = np.zeros(ns)

# make fake filters
# first, L_IR filter
# over 8-1000 microns
botlam = np.atleast_1d(8e4)
toplam = np.atleast_1d(1000e4)
edgetrans = np.atleast_1d(0)
lir_filter = [[np.concatenate((botlam-1,np.linspace(botlam, toplam, num=100),toplam+1))],
              [np.concatenate((edgetrans,np.ones(100),edgetrans))]]

# now UV filter
# over 1216-3000 angstroms
botlam = np.atleast_1d(1216)
toplam = np.atleast_1d(3000)
luv_filter =  [[np.concatenate((botlam-1,np.linspace(botlam, toplam, num=100),toplam+1))],
               [np.concatenate((edgetrans,np.ones(100),edgetrans))]]

# get the spectrum for a variety of dust extinctions
# then compute SFR, SFR_KATE
for i, u in enumerate(gd):
    
    # set dust to random value
    sps.params['dust2'] = u

    # returns spectrum in Lsun/AA
    w, spec = sps.get_spectrum(tage=sps.params['tage'], peraa=True)
    
    # calculate luminosities [returns erg/s]
    luv[i]     = integrate_spec(w,spec,luv_filter)
    lir[i]     = integrate_spec(w,spec,lir_filter)

    # calculate implied SFRs
    # sfr_kate from eqn 1 in Whitaker+14
    sfr = 1/(1e9 * sps.params['tage'])
    sfr_kate = 1.09e-10*(lir[i] + 2.2*luv[i])

    sfr_ratio[i] = sfr/sfr_kate

print sfr_ratio.mean(), sfr_ratio.std()
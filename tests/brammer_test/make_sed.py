import numpy as np
import fsps

def av_to_tau(av):

    # optical depth
    return av/1.086

# setup model, sps
sps = fsps.StellarPopulation(zcontinuous=1, compute_vega_mags=False)
sps.params['logzsol'] = 0.0 # log(Z/Zsun), interpolated
sps.params['sfh'] = 4 # delayed tau
sps.params['imf_type'] = 1 # chabrier
sps.params['tage'] = 2.5 # Gyr

# get magnitudes, spectrum
w, spec = sps.get_spectrum(tage=sps.params['tage'], peraa=False)
mags = sps.get_mags(tage=sps.params['tage'],bands='V')

# add some dust
sps.params['dust_type'] = 2 # Calzetti
sps.params['dust2'] = av_to_tau(2.5)
sps.params['dust1'] = 0.0

# get magnitudes, spectrum
w, spec_dust = sps.get_spectrum(tage=sps.params['tage'], peraa=False)
mags_dust = sps.get_mags(tage=sps.params['tage'],bands='V')

'''
The way it *should* work is that the total stellar mass formed (i.e. the integral of the SFR) 
is always 1 M_sun (as long as tage is non-zero), so sps.stellar_mass is naturally the ratio of 
the existing stellar mass  (i.e. not including gas returned to the ISM by dead stars, but including 
remnants if sps.params['add_stellar_remnants'] == 1) to the total mass formed.
'''

# remove dust
sps.params['dust2'] = 0.0

# calculate M/L for range of tage
solar_absmag = 4.83
tage = np.linspace(0.11,13.0,num=50)
ml = np.zeros_like(tage)
for ii,age in enumerate(tage): 
    vband = sps.get_mags(tage=age,bands='V')
    v_norm = 10**(-0.4*vband) / 10**(-0.4*solar_absmag)
    ml[ii] = sps.stellar_mass/v_norm

#### write out
with open('spec_nodust.dat', 'w') as f:
    for wave in w: f.write("{:.1f}".format(wave) + ' ')
    f.write('\n')
    for sp in spec: f.write("{:.3e}".format(sp)+' ')

with open('spec_dust.dat', 'w') as f:
    for wave in w: f.write("{:.1f}".format(wave) + ' ')
    f.write('\n')
    for sp in spec_dust: f.write("{:.3e}".format(sp)+' ')

with open('ml_v.dat', 'w') as f:
    for age in tage: f.write("{:.3f}".format(age) + ' ')
    for m in ml: f.write("{:.3f}".format(m) + ' ')

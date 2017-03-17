    
from prospect.sources import SSPBasis
import numpy as np

'''
class EmlineBasis(SSPBasis):

    @property
    def emline_wavelengths(self):
        return self.ssp.emline_wavelengths

    def get_nebline_luminosity(self, **params):
        """Update parameters, then multiply SSP weights by SSP emission line luminosities, and sum.

        :returns eline_lum:
            Emission line luminosities in units of Lsun
        """
        self.update(**params)

        weights = self.all_ssp_weights

        # Get the SSP emission line luminosities, adding an extra
        # copy of the t=0 array for consistency with spectral treatment
        ssp_eline_lum = self.ssp.emline_luminosity
        ssp_eline_lum = np.vstack([ssp_eline_lum[0, :], ssp_eline_lum])
        
        eline_lum = np.dot(weights,ssp_eline_lum) / weights.sum()

        return eline_lum
'''

### create SPS
sps = EmlineBasis(compute_vega_mags=False, zcontinuous=1)

### SSPs, nebular on and nebular off
sps.params['sfh'] = 0 #composite
sps.params['tage'] = 0
sps.params['add_neb_emission'] = False
sps.params['add_neb_continuum'] = False
sps.update(**sps.params)
wav, ssp_spectra_noem = sps.ssp.get_spectrum(peraa=True)
sps.params['add_neb_emission'] = True
sps.params['add_neb_continuum'] = True
sps.ssp.dirtiness = 2
sps.update(**sps.params)
wav, ssp_spectra_withem = sps.ssp.get_spectrum(peraa=True)

emline_lum = sps.ssp.emline_luminosity
emline_wav = sps.ssp.emline_wavelengths

#### find halpha in wavelength array
sps.params['sigma_smooth'] = 0.0
ha_spec_idx = (wav > 6556) & (wav < 6573)

### find halpha in print array
ha_lum_idx = (emline_wav > 6564) & (emline_wav < 6566)

#### integrate and compare
print 'calculating [Halpha(spec) - Halpha(print)]/Halpha[print] for different SSPs'
for i in xrange(ssp_spectra_noem.shape[0]):
    specval = np.trapz((ssp_spectra_withem[i,:] - ssp_spectra_noem[i,:])[ha_spec_idx], wav[ha_spec_idx])
    printval = emline_lum[i,ha_lum_idx].squeeze()
    print "age={0} Myr, {1}".format(10**sps.ssp.log_age[i]/1e6,(specval-printval)/printval)

#wav, spec, sm = sps.get_spectrum(**sps.params)
#wav, ssp_spectra = self.ssp.get_spectrum(tage=0, peraa=False)

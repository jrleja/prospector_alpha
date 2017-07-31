import numpy as np
import hickle
from astropy import constants

fnames = {
          'IC 4051': '/Users/joel/code/python/prospector_alpha/results/brownseds_test/brownseds_test_IC 4051_1497340731_post',
          'NGC 3198': '/Users/joel/code/python/prospector_alpha/results/brownseds_test/brownseds_test_NGC 3198_1497340669_post',
          '3767': '/Users/joel/code/python/prospector_alpha/results/camilla/camilla_3767_1497435370_post',
          '20168': '/Users/joel/code/python/prospector_alpha/results/camilla/camilla_20168_1497434851_post'
         }
wavelim = (3000,7000) # Lsun / Hz / cm^2
maggies_to_lsun = 3631*1e-23/constants.L_sun.cgs.value
c = 2.99e18

def load_extra_output(fname):
    with open(fname, "r") as f:
        extra_output=hickle.load(f)
    return extra_output

def main():

    parout = ['stellar_mass','sfr_100','half_time','logzsol','dust2']
    parnames = ['stellar_mass', 'sfr', 't_half', 'zstellar', 'tau_dust']
    with open('parameters.txt', 'w') as f:

        ### parameters header
        f.write('# objname ')
        for p in parnames:
            f.write(p + ' ' + p + '_errup ' + p + '_errdown ')
        f.write('\n')

        for key in fnames.keys():
            output = load_extra_output(fnames[key])
            idx = np.where((output['observables']['lam_obs'] > wavelim[0]) & (output['observables']['lam_obs'] < wavelim[1]))[0]
            pnames = output['quantiles']['parnames']
            epnames = output['extras']['parnames']

            ### writing spectra
            with open(key.replace(' ','_')+'_modspec.txt', 'w') as file:
                for i in idx:
                    wav = output['observables']['lam_obs'][i]
                    sp, sp_up, sp_do = np.percentile(output['observables']['spec'][i,:]*maggies_to_lsun*(c/wav**2), [50,84,16])
                    file.write( 
                                "{:.3e}".format(wav) + ' ' + \
                                "{:.3e}".format(sp)  + ' ' + \
                                "{:.3e}".format(sp_up)  + ' ' + \
                                "{:.3e}".format(sp_do)  + '\n'
                              )
            ### write SFH
            with open(key.replace(' ','_')+'_sfh.txt', 'w') as file:
                sfh = output['extras']['sfh']
                times = output['extras']['t_sfh']
                file.write('# time[Gyr] sfr sfr_errup sfr_errdown\n')
                for i,time in enumerate(times):
                    sfh_c, sfh_up, sfh_do = np.percentile(sfh[i,:],[16,50,84])
                    sfh_up -= sfh_c
                    sfh_do = sfh_c-sfh_do
                    file.write(
                            "{:.3e}".format(time) + ' ' + \
                            "{:.3e}".format(sfh_c)  + ' ' + \
                            "{:.3e}".format(sfh_up)  + ' ' + \
                            "{:.3e}".format(sfh_do)  + '\n'
                           )

            ### writing parameters
            f.write(key.replace(' ','_'))
            for par in parout:
                if par in pnames:
                    idx = pnames == par
                    cent = output['quantiles']['q50'][idx][0]
                    eup = output['quantiles']['q84'][idx][0] - output['quantiles']['q50'][idx][0]
                    edo = output['quantiles']['q50'][idx][0] - output['quantiles']['q16'][idx][0]
                else:
                    idx = epnames == par
                    cent = output['extras']['q50'][idx][0]
                    eup = output['extras']['q84'][idx][0] - output['extras']['q50'][idx][0]
                    edo = output['extras']['q50'][idx][0] - output['extras']['q16'][idx][0]

                if par == 'logzsol':
                    cent = (10**cent) * 0.019
                    eup = (10**(cent+eup)) * 0.019 - cent
                    edo = cent - (10**(cent-edo)) * 0.019

                f.write(' ' + "{:.3e}".format(cent) + \
                        ' ' + "{:.3e}".format(eup) + \
                        ' ' + "{:.3e}".format(edo))
            f.write('\n')












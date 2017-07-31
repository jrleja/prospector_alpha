import numpy as np
import os
from astropy.io import fits
from sedpy import observate
from brownseds_np_params import load_obs
from brown_io import load_spectra

datname = os.getenv('APPS')+'/prospector_alpha/data/brownseds_data/photometry/table1.fits'
photname = os.getenv('APPS')+'/prospector_alpha/data/brownseds_data/photometry/table3.txt'
extinctname = os.getenv('APPS')+'/prospector_alpha/data/brownseds_data/photometry/table4.fits'
herschname = os.getenv('APPS')+'/prospector_alpha/data/brownseds_data/photometry/kingfish.brownapertures.flux.fits'
name=['IC 4051', 'NGC 3198']

def write_photometry():

    with open('phot.txt', 'w') as file:
        for j in xrange(len(name)):
            obs = load_obs(photname=photname, extinctname=extinctname, herschname = herschname, datname = datname, objname = name[j])
            fnames = [filt.name for filt in obs['filters']]


            ### rewrite phot mask: NOTHING REDDER THAN K-BAND
            for i,filt in enumerate(fnames): 
                if obs['wave_effective'][i] > 2.5*1e4:
                    obs['phot_mask'][i] = False

            ### header ###
            if j == 0:
                hdr = '# objname '
                for i,filt in enumerate(fnames): 
                    if obs['phot_mask'][i]:
                        hdr += filt+' '+filt+'_err '
                file.write(hdr+'\n')
            file.write(name[j].replace(' ','_')+' ')
            for i,filt in enumerate(fnames):
                if obs['phot_mask'][i]:
                    file.write( "{:.3e}".format(obs['maggies'][i]) +' ' +  "{:.3e}".format(obs['maggies_unc'][i])  + ' ')
            file.write('\n')

    for j in xrange(len(name)):
        hdulist = fits.open(datname)
        idx = hdulist[1].data['Name'] == name[j]
        zred =  hdulist[1].data['cz'][idx][0] / 3e5
        lumdist = hdulist[1].data['Dist'][idx][0]

        print zred
        print lumdist

    for j in xrange(len(name)):
        spec = load_spectra(name[j], nufnu=True)
        idx = spec['source'] == 1

        outwave, outflux = spec['obs_lam'][idx], spec['flux_lsun'][idx]

        with open(name[j].replace(' ','_')+'_spec.txt', 'w') as file:
            file.write('# wavelength [observed, Angstroms] flux [Lsun/cm^2/AA]\n')
            for w, f in zip(outwave,outflux):
                file.write( "{:.3e}".format(w) +' ' +  "{:.3e}".format(f)  + '\n')




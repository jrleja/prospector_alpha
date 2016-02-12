import os
from astropy.io import fits
import numpy as np

'''
quick script to create a list of matching IDs
for galaxies in both the Brown and Kingfish sample
'''

#### filenames
brownlist = os.getenv('APPS')+'/threedhst_bsfh/data/brownseds_data/photometry/namelist.txt'
hfile = os.getenv('APPS')+'/threedhst_bsfh/data/brownseds_data/photometry/kingfish.brownapertures.flux.fits'
outname = os.getenv('APPS')+'/threedhst_bsfh/data/herschel_names.txt'

#### open names
herschel = fits.open(hfile)
kingnames = herschel[1].data['name']
ids = np.loadtxt(brownlist, dtype='|S20',delimiter=',')

#### match
matchlist = []
for id in ids:
	match = kingnames == id.lower().replace(' ','')
	if herschel[1].data['spire350'][match] > 0.0:
		matchlist.append(id)

#### write out thetas
with open(outname, 'w') as f:
	for name in matchlist:
		f.write(name+'\n')

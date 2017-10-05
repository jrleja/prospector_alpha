import td_io, os, prosp_dutils
import numpy as np
from astropy.io import ascii
import astropy.coordinates as coords
from astropy import units as u
from astropy.cosmology import WMAP9

apps = os.getenv('APPS')

def remove_zp_offsets(field,phot,no_zp_correction=None):
    # by default, don't correct the space-based photometry
    zp_offsets = td_io.load_zp_offsets(field)
    nbands     = len(zp_offsets)
    if no_zp_correction is None:
        no_zp_correction = ['irac1','irac2','irac3','irac4','f435w','f606w','f606wcand','f775w','f814w',
                            'f814wcand','f850lp','f850lpcand','f125w','f140w','f160w']
    for kk in xrange(nbands):
        filter = zp_offsets[kk]['Band'].lower()+'_'+field.lower()
        if zp_offsets[kk]['Band'].lower() in no_zp_correction:
            phot['f_'+filter] = phot['f_'+filter]/zp_offsets[kk]['Flux-Correction']
            phot['e_'+filter] = phot['e_'+filter]/zp_offsets[kk]['Flux-Correction']

    return phot

def select_massive(phot=None,fast=None,zbest=None,**extras):
    # consider removing zbest['use_zgrism'] cut in the future!! no need for good grism data really
    return (phot['use_phot'] == 1) & (fast['lmass'] > 11) & (zbest['use_zgrism'] == 1)

def select_ha(phot=None,fast=None,zbest=None,gris=None,**extras):
    np.random.seed(2)
    idx = np.where((phot['use_phot'] == 1) & (zbest['use_zgrism'] == 1)  & (gris['Ha_FLUX']/gris['Ha_FLUX_ERR'] > 10) & (gris['Ha_EQW'] > 10))[0]
    return np.random.choice(idx,40,replace=False)

def select_td(phot=None,fast=None,zbest=None,gris=None,**extras):
    np.random.seed(2)
    idx = np.where((phot['use_phot'] == 1) & 
                   ((zbest['z_best_u68'] - zbest['z_best_l68'])/2. < 0.1) & \
                   (zbest['z_best'] - fast['z'] < 0.01)) # this SHOULDN'T matter but some of the FAST z's are not zbest!
    return idx

def td_cut(out):
    """ select galaxies spaced evenly in z, mass, sSFR
    """
    n_per_bin = 30
    nfield = len(out['fast'])
    zbins = np.linspace(0.5,3,6)
    ssfrbins = np.array([-np.inf,1e-11,1e-10,1e-9,5e-9,1e-8])
    massbins = np.array([9.0,9.5,10.,10.5,11.,11.5])

    # grab quantities from all fields
    mass, ssfr, z = [], [], []
    for i in range(nfield):
        mass += np.array(out['fast'][i]['lmass']).tolist()
        ssfr += np.clip((out['zbest'][i]['sfr']/10**out['fast'][i]['lmass']),1e-14,np.inf).tolist()
        z += np.array(out['zbest'][i]['z_best']).tolist()
    mass, ssfr, z = np.array(mass), np.array(ssfr), np.array(z)

    # select in bins
    idx = []
    for i in xrange(len(zbins)-1):
        for j in xrange(len(ssfrbins)-1):
            for k in xrange(len(massbins)-1):
                good = np.where((mass > massbins[k]) & \
                                (mass < massbins[k+1]) & \
                                (z > zbins[i]) & \
                                (z < zbins[i+1]) & \
                                (ssfr > ssfrbins[j]) & \
                                (ssfr < ssfrbins[j+1]))
                good = good[0]
                nsamp = len(good)
                if nsamp < n_per_bin:
                    print '{0} galaxies for bin {1} < logM < {2}, {3} < log(sSFR) < {4}, {5} < z < {6}'.format(\
                          nsamp,massbins[k],massbins[k+1],ssfrbins[j],ssfrbins[j+1],zbins[i],zbins[i+1])
                    idx += good.tolist()
                else:
                    np.random.seed(2)
                    idx += np.random.choice(good,replace=False,size=n_per_bin).tolist()

    print '{0} total galaxies selected (out of {1})'.format(len(idx),len(mass))
    idx = np.array(idx)

    # this removes galaxies in each field that don't make the cuts
    ncut = 0
    out['ids'] = np.array(out['ids'])[idx].tolist()
    for i in range(nfield):
        ngals = len(out['fast'][i])
        in_field = (idx < (ngals+ncut)) & (idx >= ncut)

        field_idx = idx[in_field]-ncut
        out['fast'][i] = out['fast'][i][field_idx]
        out['zbest'][i] = out['zbest'][i][field_idx]
        out['phot'][i] = out['phot'][i][field_idx]

        ncut += ngals

    return out

td_sample = {
             'selection_function': select_td,
             'runname': 'td',
             'rm_zp_offsets': True,
             'master_cut': td_cut
            }

massive_sample = {
                  'selection_function': select_massive,
                  'runname': 'td_massive',
                  'rm_zp_offsets': True
                  }

ha_sample = {
             'selection_function': select_ha,
             'runname': 'td_ha',
             'rm_zp_offsets': True
            }

dynamic_sample = {
             'selection_function': None,
             'runname': 'td_dynamic',
             'rm_zp_offsets': True
            }

def build_sample(sample=None):
    """general function to select a sample of galaxies from the 3D-HST catalogs
    each sample is defined by a series of keywords in a dictionary
    """
    ### output
    fields = ['AEGIS','COSMOS','GOODSN','GOODSS','UDS']
    id_str_out = '/Users/joel/code/python/prospector_alpha/data/3dhst/'+sample['runname']+'.ids'
    out = {
           'fast': [],
           'zbest': [],
           'phot': [],
           'ids': []
          }

    for field in fields:

        # load data
        print 'loading '+field
        phot = td_io.load_phot_v41(field)
        fast = td_io.load_fast_v41(field)
        zbest = td_io.load_zbest(field)
        mips = td_io.load_mips_data(field)
        gris = td_io.load_grism_dat(field,process=True)

        # make catalog cuts
        good = sample['selection_function'](fast=fast,phot=phot,zbest=zbest,gris=gris)

        phot = phot[good]
        fast = fast[good]
        zbest = zbest[good]
        mips = mips[good]
        gris = gris[good]

        # add in mips
        phot['f_MIPS_24um'] = mips['f24tot']
        phot['e_MIPS_24um'] = mips['ef24tot']

        # save UV+IR SFRs + emission line info in .dat file
        for name in mips.dtype.names:
            if (name != 'z') & (name != 'id') & (name != 'z_best'):
                zbest[name] = mips[name]
            elif (name == 'z') or (name == 'z_best'):
                zbest['z_sfr'] = mips[name]
        for name in gris.dtype.names: zbest[name] = gris[name]

        # rename bands in photometric catalogs
        for column in phot.colnames:
            if column[:2] == 'f_' or column[:2] == 'e_':
                phot.rename_column(column, column.lower()+'_'+field.lower())    

        # are we removing the zeropoint offsets? (don't do space photometry!)
        if sample['rm_zp_offsets']:
            phot = remove_zp_offsets(field,phot)

        # save for writing out
        out['phot'].append(phot)
        out['fast'].append(fast)
        out['zbest'].append(zbest)
        out['ids'] += [field+'_'+str(id) for id in phot['id']]

    if 'master_cut' in sample.keys():
        out = sample['master_cut'](out) 

    # write out for each field
    for i,field in enumerate(fields):
        outbase = '/Users/joel/code/python/prospector_alpha/data/3dhst/'+field+'_'+sample['runname']
        ascii.write(out['phot'][i], output=outbase+'.cat', 
                    delimiter=' ', format='commented_header',overwrite=True)
        ascii.write(out['fast'][i], output=outbase+'.fout', 
                    delimiter=' ', format='commented_header',
                    include_names=out['fast'][i].keys()[:11],overwrite=True)
        ascii.write(out['zbest'][i], output=outbase+'.dat', 
                    delimiter=' ', format='commented_header',overwrite=True)

    ascii.write([out['ids']], output=id_str_out, Writer=ascii.NoHeader,overwrite=True)

def build_sample_dynamics(sample=dynamic_sample,print_match=True):
    """finds Rachel's galaxies in the threedhst catalogs
    matches on distance < 3 arcseconds and logM < 0.4 dex
    this will FAIL if we have two identical IDs in UDS/COSMOS... unlikely but if used
    in the future as a template for multi-field runs, beware!!!
    """

    # load catalog, define matching
    bez = load_rachel_sample()
    matching_radius = 3 # in arcseconds
    matching_mass = 0.4 # in dex

    # output formatting
    fields = ['COSMOS','UDS']
    id_str_out = '/Users/joel/code/python/prospector_alpha/data/3dhst/'+sample['runname']+'.ids'
    ids = []

    # let's go
    for field in fields:
        outbase = '/Users/joel/code/python/prospector_alpha/data/3dhst/'+field+'_'+sample['runname']

        # load data
        phot = td_io.load_phot_v41(field)
        fast = td_io.load_fast_v41(field)
        zbest = td_io.load_zbest(field)
        mips = td_io.load_mips_data(field)
        gris = td_io.load_grism_dat(field,process=True)
        morph = td_io.load_morphology(field,'F160W')

        # note: converting to physical sizes, not comoving. I assume matches Rachel's catalog
        arcsec_size = bez['Re']*WMAP9.arcsec_per_kpc_proper(bez['z']).value

        # match
        match, bezmatch_flag, dist = [], [], []
        threed_cat = coords.SkyCoord(ra=phot['ra']*u.degree, 
                                 dec=phot['dec']*u.degree)
        for i in range(len(bez)):
            bzcat = coords.SkyCoord(ra=bez['RAdeg'][i]*u.deg,dec=bez['DEdeg'][i]*u.deg)  
            close_mass = np.where((np.abs(fast['lmass']-bez[i]['logM']) < matching_mass) & \
                         (fast['z'] < 2.0) & \
                         (phot['use_phot'] == 1))[0]
                         
            idx, d2d, d3d = bzcat.match_to_catalog_sky(threed_cat[close_mass])
            if d2d.value*3600 < matching_radius:
                match.append(int(close_mass[idx]))
                bezmatch_flag.append(True)
                dist.append(d2d.value[0]*3600)
            else:
                bezmatch_flag.append(False)
        print '{0} galaxies matched in {1}'.format(len(match),field)
        if len(match) == 0:
            continue
        # full match
        bezmatch_flag = np.array(bezmatch_flag,dtype=bool)
        phot = phot[match]
        fast = fast[match]
        zbest = zbest[match]
        mips = mips[match]
        morph = morph[match]
        gris = gris[match]

        # print out relevant parameters for visual inspection
        if print_match:
            for kk,cat in enumerate(bez[bezmatch_flag]):
                print 'dM:{0},\t dz:{1},\t dr:{2},\t dist:{3}'.format(\
                      fast[kk]['lmass']-bez[kk]['logM'], fast[kk]['z']-bez[kk]['z'], morph[kk]['re']-arcsec_size[kk],dist[kk])
        
        # assemble ancillary data
        phot['f_MIPS_24um'] = mips['f24tot']
        phot['e_MIPS_24um'] = mips['ef24tot']

        # save UV+IR SFRs + emission line info in .dat file
        for name in mips.dtype.names:
            if name != 'z' and name != 'id':
                zbest[name] = mips[name]
            elif name == 'z':
                zbest['z_sfr'] = mips[name]
        for name in gris.dtype.names: zbest[name] = gris[name]

        # rename bands in photometric catalogs
        for column in phot.colnames:
            if column[:2] == 'f_' or column[:2] == 'e_':
                phot.rename_column(column, column.lower()+'_'+field.lower())    

        # are we removing the zeropoint offsets? (don't do space photometry!)
        if sample['rm_zp_offsets']:
            phot = remove_zp_offsets(field,phot)

        # include bezanson catalog info
        not_to_save = ['z', 'ID', 'Filter']
        for i,name in enumerate(bez.colnames):
            if name not in not_to_save:
                zbest[name] = bez[bezmatch_flag][name]
            elif name == 'z':
                zbest['z_bez'] = bez[bezmatch_flag][name]

        # write out
        ascii.write(phot, output=outbase+'.cat', 
                    delimiter=' ', format='commented_header',overwrite=True)
        ascii.write(fast, output=outbase+'.fout', 
                    delimiter=' ', format='commented_header',
                    include_names=fast.keys()[:11],overwrite=True)
        ascii.write(zbest, output=outbase+'.dat', 
                    delimiter=' ', format='commented_header',overwrite=True)
        ### save IDs
        ids = ids + [field+'_'+str(id) for id in phot['id']]

    print 'number of matched galaxies: {0} (out of {1})'.format(len(ids),len(bez))
    ascii.write([ids], output=id_str_out, Writer=ascii.NoHeader,overwrite=True)
























def calc_uvj_flag(rf):
    """calculate a UVJ flag
    0 = quiescent, 1 = starforming"""
    import numpy as np

    umag = 25 - 2.5*np.log10(rf['L153'])
    vmag = 25 - 2.5*np.log10(rf['L155'])
    jmag = 25 - 2.5*np.log10(rf['L161'])
    
    u_v = umag-vmag
    v_j = vmag-jmag  
    
    # initialize flag to 3, for quiescent
    uvj_flag = np.zeros(len(umag))+3
    
    # star-forming
    sfing = (u_v < 1.3) | (v_j >= 1.5)
    uvj_flag[sfing] = 1
    sfing = (v_j >= 0.92) & (v_j <= 1.6) & (u_v <= 0.8*v_j+0.7)
    uvj_flag[sfing] = 1
    
    # dusty star-formers
    dusty_sf = (uvj_flag == 1) & (u_v >= -1.2*v_j+2.8)
    uvj_flag[dusty_sf] = 2
    
    # outside box
    # from van der wel 2014
    outside_box = (u_v < 0) | (u_v > 2.5) | (v_j > 2.3) | (v_j < -0.3)
    uvj_flag[outside_box] = 0
    
    return uvj_flag


def build_sample_onekrun(rm_zp_offsets=True):

    '''
    selects a sample of galaxies for 1k run
    evenly spaced in five redshift bins out to z=3
    '''

    zbins = [(0.4,0.8),(0.8,1.2),(1.2,1.6),(1.6,2.0),(2.0,2.5)]

    # output
    field = 'COSMOS'
    basename = 'onek'
    fast_str_out = '/Users/joel/code/python/prospector_alpha/data/'+field+'_'+basename+'.fout'
    ancil_str_out = '/Users/joel/code/python/prospector_alpha/data/'+field+'_'+basename+'.dat'
    phot_str_out = '/Users/joel/code/python/prospector_alpha/data/'+field+'_'+basename+'.cat'
    id_str_out   = '/Users/joel/code/python/prospector_alpha/data/'+field+'_'+basename+'.ids'

    # load data
    # use grism redshift
    phot = read_sextractor.load_phot_v41(field)
    fast = read_sextractor.load_fast_v41(field)
    rf = read_sextractor.load_rf_v41(field)
    lineinfo = load_linelist()
    mips = prosp_dutils.load_mips_data(field)
    
    # remove junk
    # 153, 155, 161 are U, V, J
    good = (phot['use_phot'] == 1) & \
           (phot['f_IRAC4'] < 1e7) & (phot['f_IRAC3'] < 1e7) & (phot['f_IRAC2'] < 1e7) & (phot['f_IRAC1'] < 1e7) & \
           (fast['lmass'] > 10.3)

    phot = phot[good]
    fast = fast[good]
    rf = rf[good]
    lineinfo = lineinfo[good]
    mips = mips[good]
    
    # define UVJ flag, S/N, HA EQW
    uvj_flag = calc_uvj_flag(rf)
    sn_F160W = phot['f_F160W']/phot['e_F160W']
    Ha_EQW_obs = lineinfo['Ha_EQW_obs']
    lineinfo.rename_column('zgris' , 'z')
    lineinfo['uvj_flag'] = uvj_flag
    lineinfo['sn_F160W'] = sn_F160W

    # mips
    phot['f_MIPS_24um'] = mips['f24tot']
    phot['e_MIPS_24um'] = mips['ef24tot']

    for i in xrange(len(mips.dtype.names)):
        tempname=mips.dtype.names[i]
        if tempname != 'z_best' and tempname != 'id':
            lineinfo[tempname] = mips[tempname]
        elif tempname == 'z_best':
            lineinfo['z_sfr'] = mips[tempname]
    
    # split into bins
    ngal = 1000
    n_per_bin = ngal / len(zbins)
    
    for z in zbins:
        selection = np.where((lineinfo['zbest'] >= z[0]) & (lineinfo['zbest'] < z[1]))[0]

        if np.sum(selection) < n_per_bin:
            print 'ERROR: Not enough galaxies in bin!'
            print np.sum(selection),n_per_bin
                
        # choose random set of indices
        random_index = random.sample(xrange(len(selection)), n_per_bin)
        if z[0] != zbins[0][0]:
            fast_out = vstack([fast_out,fast[selection[random_index]]])
            phot_out = vstack([phot_out,phot[selection[random_index]]])
            lineinfo_out = vstack([lineinfo_out,lineinfo[selection[random_index]]])
        else:
            fast_out = fast[selection[random_index]]
            phot_out = phot[selection[random_index]]
            lineinfo_out = lineinfo[selection[random_index]]
    
    # rename bands in photometric catalogs
    for column in phot_out.colnames:
        if column[:2] == 'f_' or column[:2] == 'e_':
            phot_out.rename_column(column, column.lower()+'_'+field.lower())    

    if rm_zp_offsets:
        phot_out = remove_zp_offsets(field,phot_out)

    ascii.write(phot_out, output=phot_str_out, 
                delimiter=' ', format='commented_header')
    ascii.write(fast_out, output=fast_str_out, 
                delimiter=' ', format='commented_header',
                include_names=fast.keys()[:11])
    ascii.write(lineinfo_out, output=ancil_str_out, 
                delimiter=' ', format='commented_header')
    ascii.write([np.array(phot_out['id'],dtype='int')], output=id_str_out, Writer=ascii.NoHeader)

def build_sample_general():

    '''
    selects a sample of galaxies "randomly"
    to add: output for the EAZY parameters, so I can include p(z) [or whatever I need for that]
    '''

    # output
    field = 'COSMOS'
    basename = 'gensamp'
    fast_str_out = '/Users/joel/code/python/prospector_alpha/data/'+field+'_'+basename+'.fout'
    ancil_str_out = '/Users/joel/code/python/prospector_alpha/data/'+field+'_'+basename+'.dat'
    phot_str_out = '/Users/joel/code/python/prospector_alpha/data/'+field+'_'+basename+'.cat'
    id_str_out   = '/Users/joel/code/python/prospector_alpha/data/'+field+'_'+basename+'.ids'

    # load data
    # use grism redshift
    phot = read_sextractor.load_phot_v41(field)
    fast = read_sextractor.load_fast_v41(field)
    rf = read_sextractor.load_rf_v41(field)
    lineinfo = load_linelist()
    mips = prosp_dutils.load_mips_data(os.getenv('APPS')+'/prospector_alpha/data/MIPS/cosmos_3dhst.v4.1.4.sfr')
    
    # remove junk
    # 153, 155, 161 are U, V, J
    good = (phot['use_phot'] == 1) & \
           (phot['f_IRAC4'] < 1e7) & (phot['f_IRAC3'] < 1e7) & (phot['f_IRAC2'] < 1e7) & (phot['f_IRAC1'] < 1e7) & \
           (phot['f_F160W']/phot['e_F160W'] > 100) & (fast['lmass'] > 10)

    phot = phot[good]
    fast = fast[good]
    rf = rf[good]
    lineinfo = lineinfo[good]
    mips = mips[good]
    
    # define UVJ flag, S/N, HA EQW
    uvj_flag = calc_uvj_flag(rf)
    sn_F160W = phot['f_F160W']/phot['e_F160W']
    Ha_EQW_obs = lineinfo['Ha_EQW_obs']
    lineinfo.rename_column('zgris' , 'z')
    lineinfo['uvj_flag'] = uvj_flag
    lineinfo['sn_F160W'] = sn_F160W

    # mips
    phot['f_MIPS_24um'] = mips['f24tot']
    phot['e_MIPS_24um'] = mips['ef24tot']

    for i in xrange(len(mips.dtype.names)):
        tempname=mips.dtype.names[i]
        if tempname != 'z_best' and tempname != 'id':
            lineinfo[tempname] = mips[tempname]
        elif tempname == 'z_best':
            lineinfo['z_sfr'] = mips[tempname]
    
    # split into bins
    selection = np.arange(0,np.sum(good))
    random_index = random.sample(xrange(len(selection)), 108)
    fast_out = fast[selection][random_index]
    phot_out = phot[selection][random_index]
    lineinfo = lineinfo[selection][random_index]
    
    # rename bands in photometric catalogs
    for column in phot_out.colnames:
        if column[:2] == 'f_' or column[:2] == 'e_':
            phot_out.rename_column(column, column.lower()+'_'+field.lower())    

    ascii.write(phot_out, output=phot_str_out, 
                delimiter=' ', format='commented_header')
    ascii.write(fast_out, output=fast_str_out, 
                delimiter=' ', format='commented_header',
                include_names=fast.keys()[:11])
    ascii.write(lineinfo, output=ancil_str_out, 
                delimiter=' ', format='commented_header')
    ascii.write([np.array(phot['id'],dtype='int')], output=id_str_out, Writer=ascii.NoHeader)
    print 1/0

def load_rachel_sample():

    loc = os.getenv('APPS')+'/prospector_alpha/data/bezanson_2015_disps.txt'
    data = ascii.read(loc,format='cds') 
    return data


import td_io, os, prosp_dutils, hickle
import numpy as np
from astropy.io import ascii
import astropy.coordinates as coords
from astropy import units as u
from astropy.cosmology import WMAP9
from astropy.table import Table
from scipy.interpolate import interp1d

apps = os.getenv('APPS')

def remove_zp_offsets(field,phot,no_zp_correction=None):
    # by default, don't correct the space-based photometry
    zp_offsets = td_io.load_zp_offsets(field)
    nbands     = len(zp_offsets)
    if no_zp_correction is None:
        no_zp_correction = ['f435w','f606w','f606wcand','f775w','f814w',
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
    idx = np.where((phot['use_phot'] == 1) & 
                   ((zbest['z_best_u68'] - zbest['z_best_l68'])/2. < 0.1) & \
                   (zbest['z_best'] - fast['z'] < 0.01)) # this SHOULDN'T matter but some of the FAST z's are not zbest!
    return idx

def select_huge(phot=None,fast=None,zbest=None,gris=None,**extras):
    idx = np.where((phot['use_phot'] == 1) & 
                   ((zbest['z_best_u68'] - zbest['z_best_l68'])/2. < 0.1) & \
                   (phot['f_F160W'] / phot['e_F160W'] > 10) & \
                   (zbest['z_best'] - fast['z'] < 0.01)) # this SHOULDN'T matter but some of the FAST z's are not zbest!
    return idx

def select_huge_supp(phot=None,fast=None,zbest=None,gris=None,**extras):
    # pick out td_huge selection and new selection
    # find where they diverge
    idx_huge = select_huge(phot=phot,fast=fast,zbest=zbest,gris=gris,**extras)
    idx = np.where((phot['use_phot'] == 1) & \
                   ((zbest['z_best_u68'] - zbest['z_best_l68'])/2. < 0.25) & \
                   (phot['f_F160W'] / phot['e_F160W'] > 10) & \
                   (zbest['z_best'] >= 0.5) & (zbest['z_best'] <= 2.5))
    #idx_supp = idx[0][~np.in1d(idx[0],idx_huge[0])]
    return np.unique(np.concatenate((idx_huge[0],idx[0])))

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

huge_sample = {
             'selection_function': select_huge,
             'runname': 'td_huge',
             'rm_zp_offsets': True,
              }

huge_supp_sample = {
                    'selection_function': select_huge_supp,
                    'runname': 'td_huge',
                    'rm_zp_offsets': True,
                    }

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
        rf = td_io.load_rf_v41(field)

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

        # save UVJ cut in .dat file
        zbest['uvj'] = calc_uvj_flag(rf[good])

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

    # count galaxies
    count = 0
    for i in range(len(out['fast'])): count += len(out['fast'][i])
    print 'Total of {0} galaxies'.format(count)

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

def load_master_sample():
    """builds a 'master' sample with a very simple photometric cut
    this can be used for comparison to subsamples of this sample
    """
    fields = ['AEGIS','COSMOS','GOODSN','GOODSS','UDS']
    out = {key: [] for key in ['zbest','uvir_sfr','fast_logmass','id','z_best_u68','z_best_l68','f_F160W','e_F160W','fast_z']}

    for field in fields:

        # load data
        print 'loading '+field
        
        # define a master catalog
        # let's keep it simple here (and FULL OF CRAP of course!)
        phot = td_io.load_phot_v41(field)
        fast = td_io.load_fast_v41(field)
        good = (phot['use_phot'] == 1) & np.isfinite(fast['lmass'])
        phot = phot[good]
        fast = fast[good]

        # load the rest of the data
        zbest = td_io.load_zbest(field)[good]
        mips = td_io.load_mips_data(field)[good]

        # fill the data
        out['zbest'] += np.array(zbest['z_best']).tolist()
        out['uvir_sfr'] += np.array(mips['sfr']).tolist()
        out['fast_logmass'] += np.array(fast['lmass']).tolist()
        out['fast_z'] += np.array(fast['z']).tolist()
        out['id'] += [field+'_'+str(name) for name in phot['id']]
        out['z_best_u68'] += np.array(zbest['z_best_u68']).tolist()
        out['z_best_l68'] += np.array(zbest['z_best_l68']).tolist()
        out['f_F160W'] += np.array(phot['f_F160W']).tolist()
        out['e_F160W'] += np.array(phot['e_F160W']).tolist()

    return out 

def build_sample_lyc():
    """function to select LyC candidates from the 3D-HST catalogs
    """
    ### output
    fields = ['GOODSS','GOODSN']
    id_str_out = '/Users/joel/code/python/prospector_alpha/data/3dhst/td_lyc.ids'
    out = {
           'fast': [],
           'zbest': [],
           'phot': [],
           'ids': []
          }

    # list of candidates
    # 3dhst_id, zred
    lyc_list = [os.getenv('APPS')+'/prospector_alpha/data/td_lyc/' + s for s in ['goodss.txt','goodsn.txt']]
    lyc_cands = [np.genfromtxt(l,dtype=[('id',int),('zred',float)],delimiter=',') for l in lyc_list]

    for (cand, field) in zip(lyc_cands,fields):

        # load data
        print 'loading '+field
        phot = td_io.load_phot_v41(field)
        fast = td_io.load_fast_v41(field)
        zbest = td_io.load_zbest(field)
        mips = td_io.load_mips_data(field)
        gris = td_io.load_grism_dat(field,process=True)
        rf = td_io.load_rf_v41(field)

        # match to lyc candidates
        good = (np.in1d(phot['id'],cand['id'])) & (np.array(phot['use_phot'],dtype=bool))

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

        # replace z_best with MUSE redshifts
        for i, id in enumerate(phot['id']):
            match = cand['id'] == id
            try:
                zbest['z_best'][i] = cand['zred'][match]
            except:
                print 1/0
            print zbest['z_best'][i], id

        # save UVJ cut in .dat file
        zbest['uvj'] = calc_uvj_flag(rf[good])

        # rename bands in photometric catalogs
        for column in phot.colnames:
            if column[:2] == 'f_' or column[:2] == 'e_':
                phot.rename_column(column, column.lower()+'_'+field.lower())    

        # treat ZP offsets (don't do space photometry!)
        phot = remove_zp_offsets(field,phot)

        # save for writing out
        out['phot'].append(phot)
        out['fast'].append(fast)
        out['zbest'].append(zbest)
        out['ids'] += [field+'_'+str(id) for id in phot['id']]

    # count galaxies
    count = 0
    for i in range(len(out['fast'])): count += len(out['fast'][i])
    print 'Total of {0} galaxies'.format(count)

    # write out for each field
    for i,field in enumerate(fields):
        outbase = '/Users/joel/code/python/prospector_alpha/data/3dhst/'+field+'_td_lyc'
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

def build_sample_hstacks():
    """select a sample of galaxies from the 3D-HST catalogs for Herschel stacking
    here we take anything with log(sSFR) > -10.8, as measured by Prospector 
    """

    ### output
    fields = ['AEGIS','COSMOS','GOODSN','GOODSS','UDS']
    outfolder = '/Users/joel/code/python/prospector_alpha/data/3dhst/hstacks/'
    if not os.path.isdir(outfolder):
        os.makedirs(outfolder)

    ### hack to pick out the ancillary data
    with open('/Users/joel/code/python/prospector_alpha/plots/td_huge/fast_plots/data/fastcomp.h5', "r") as f:
        ancil = hickle.load(f)
    
    ### pull out IDs and field names
    # this is for the sSFR cut
    ssfr_min = -10.8
    fnames, fids = [], []
    for name in ancil['objname']:
        temp = name.split('_')
        fnames += [temp[0]]
        fids += [temp[1]]
    fnames, fids = np.array(fnames), np.array(fids,dtype=int)

    ### iterate over each field
    names = ['ID','ra','dec','zred','log_mass','log_sfr']
    for field in fields:

        # load data
        print 'loading '+field
        phot = td_io.load_phot_v41(field)
        fast = td_io.load_fast_v41(field)
        zbest = td_io.load_zbest(field)
        mips = td_io.load_mips_data(field)
        gris = td_io.load_grism_dat(field,process=True)

        # only get galaxies which we have prospector measurements for!
        in_field = fnames == field
        matches = np.in1d(phot['id'],fids[in_field])

        phot = phot[matches]
        fast = fast[matches]
        zbest = zbest[matches]
        mips = mips[matches]
        gris = gris[matches]

        # now make sSFR cuts
        starforming = ancil['prosp']['ssfr_100']['q50'][in_field] > ssfr_min
        phot = phot[starforming]
        fast = fast[starforming]
        zbest = zbest[starforming]
        mips = mips[starforming]
        gris = gris[starforming]

        # save outputs
        # IDs, masses, redshifts, UV+IR SFRs, coordinates 
        out = {}
        out['ID'] = np.array([field + '_' + str(i) for i in fast['id']])
        out['zred'] = np.array(zbest['z_best'])
        out['log_mass'] = (ancil['prosp']['stellar_mass']['q50'][in_field])[starforming]
        out['log_sfr'] = (ancil['prosp']['sfr_100']['q50'][in_field])[starforming]
        out['ra'] = phot['ra']
        out['dec'] = phot['dec']
        
        print field+' has {0} eligible galaxies'.format(len(out['ID']))
        outname = outfolder + field + '.cat'
        ascii.write([out[key] for key in names], output=outname, names=names,
                     delimiter=' ', format='commented_header',overwrite=True)


    """
    # format output
    for key in out.keys(): out[key] = np.array(out[key])
    print 'Total of {0} galaxies'.format(len(out['ID']))

    # now subdivide into mass, Z bins
    mbins = np.array([9,9.8,10.6,12])
    zbins = np.array([0.5,1.0,1.5,2.0,2.5,3.0])
    n_mbins, n_zbins = mbins.shape[0]-1, zbins.shape[0]-1
    for i in range(n_mbins):
        for j in range(n_zbins):
            idx = (out['log_mass'] > mbins[i]) & (out['log_mass'] <= mbins[i+1]) & \
                  (out['zred'] > zbins[j]) & (out['zred'] <= zbins[j+1])
            print '{0} < logM < {1}, {2} < z < {3}: {4}'.format(mbins[i],mbins[i+1],zbins[j],zbins[j+1],idx.sum())

            outname = outfolder + 'logm_{:1.2f}_{:1.2f}_z_{:1.2f}_{:1.2f}.cat'.format(mbins[i],mbins[i+1],zbins[j],zbins[j+1])
            ascii.write([out[key][idx] for key in ordered_names], output=outname, names=ordered_names,
                        delimiter=' ', format='commented_header',overwrite=True)
    """
def sfr_ms(z,logm):
    """ returns the SFR of the star-forming sequence from Whitaker+14
    as a function of mass and redshift
    note that this is only valid over 0.5 < z < 2.5.
    we use the quadratic form (as opposed to the broken power law form)
    """

    # sanitize logm
    logm = np.atleast_2d(logm).T
    z = np.atleast_2d(z).T
    sfr_out = np.zeros_like(logm)

    # parameters from whitaker+14
    zwhit = np.array([0.75, 1.25, 1.75, 2.25])
    a = np.array([-27.4,-26.03,-24.04,-19.99])
    b = np.array([5.02, 4.62, 4.17, 3.44])
    c = np.array([-0.22, -0.19, -0.16, -0.13])

    # generate SFR(M) at all redshifts 
    log_sfr = a + b*logm + c*logm**2

    # interpolate to proper redshift
    for i in range(sfr_out.shape[0]): 
        tmp = interp1d(zwhit, log_sfr[i,:],fill_value='extrapolate')
        sfr_out[i] = 10**tmp(z[i])

    return sfr_out.squeeze()

def calc_uvj_flag(rf):
    """calculate a UVJ flag
    0 = quiescent, 1 = starforming
    taken from Whitaker et al. 2012
    U-V > 0.8(V-J) + 0.7, U-V > 1.3, V-J < 1.5
    """

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
    sfing = (v_j >= 0.75) & (v_j <= 1.5) & (u_v <= 0.8*v_j+0.7)
    uvj_flag[sfing] = 1
    
    # dusty star-formers
    dusty_sf = (uvj_flag == 1) & (u_v >= 1.3)
    uvj_flag[dusty_sf] = 2
    
    # outside box
    # from van der wel 2014
    #outside_box = (u_v < 0) | (u_v > 2.5) | (v_j > 2.3) | (v_j < -0.3)
    #uvj_flag[outside_box] = 0
    
    return uvj_flag

def load_rachel_sample():

    loc = os.getenv('APPS')+'/prospector_alpha/data/bezanson_2015_disps.txt'
    data = ascii.read(loc,format='cds') 
    return data


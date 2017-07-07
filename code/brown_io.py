import pickle
import numpy as np
import os

#### where do alldata pickle files go?
outpickle = '/Users/joel/code/magphys/data/pickles'

def save_spec_cal(spec_cal,runname='brownseds'):
    output = outpickle+'/spec_calibration.pickle'
    pickle.dump(spec_cal,open(output, "wb"))

def load_spec_cal(runname='brownseds'):
    with open(outpickle+'/spec_calibration.pickle', "rb") as f:
        spec_cal=pickle.load(f)
    return spec_cal

def load_alldata(runname='brownseds'):

    output = outpickle+'/'+runname+'_alldata.pickle'
    with open(output, "rb") as f:
        alldata=pickle.load(f)
    return alldata

def save_alldata(alldata,runname='brownseds'):

    output = outpickle+'/'+runname+'_alldata.pickle'
    with open(outname, "wb") as out:
        pickle.dump(model_store, out)

def return_agn_str(idx, string=False):

    # NEW VERSION, WITH MY FLUXES
    with open(os.getenv('APPS')+'/threedhst_bsfh/data/brownseds_data/photometry/joel_bpt.pickle', "rb") as f:
        agn_str=pickle.load(f)

    agn_str = agn_str[idx]
    sfing = (agn_str == 'SF') | (agn_str == 'None')
    composite = (agn_str == 'SF/AGN')
    agn = agn_str == 'AGN'

    if string:
        return agn_str
    else:
        return sfing, composite, agn

def load_prospector_data(filebase,no_sample_results=False,objname=None,runname=None,hdf5=True,load_extra_output=True):

    '''
    loads Prospector chains
    filebase: string describing the location + objname. automatically finds 
    no_sample_results: only load the Powell results and the model
    objname and runname: if both of these are supplied, don't need to supply filebase
    hdf5: load HDF5 output
    '''

    from prospect.io import read_results

    #### shortcut: pass None to filebase and objname + runname keywords
    if (objname is not None) & (runname is not None):
        filebase = os.getenv('APPS')+'/threedhst_bsfh/results/'+runname+'/'+runname+'_'+objname
    mcmc_filename, model_filename, extra_name = create_prosp_filename(filebase)

    if not hdf5:
        mcmc_filename = mcmc_filename[-3:]

    if load_extra_output:
        extra_output = load_prospector_extra(filebase,objname=objname,runname=runname)
    else:
        extra_output = None

    if no_sample_results:
        model, powell_results = read_results.read_model(model_filename)
        sample_results = None
    else:
        try:
            sample_results, powell_results, model = read_results.results_from(mcmc_filename, model_file=model_filename,inmod=None)
        except KeyError:
            print 'failed to load file '+str(mcmc_filename)+' for object '+filebase.split('/')[-1]
            sample_results, powell_results, model = None, None, None

    return sample_results, powell_results, model, extra_output

def load_prospector_extra(filebase,objname=None,runname=None):

    import hickle

    #### shortcut: pass None to filebase and objname + runname keywords
    if (objname is not None) & (runname is not None):
        filebase = os.getenv('APPS')+'/threedhst_bsfh/results/'+runname+'/'+runname+'_'+objname
    mcmc_filename, model_filename, extra_name = create_prosp_filename(filebase)

    with open(extra_name, "r") as f:
        extra_output=hickle.load(f)

    return extra_output

def create_prosp_filename(filebase):

    # find most recent output file
    # with the objname
    folder = "/".join(filebase.split('/')[:-1])
    filename = filebase.split("/")[-1]
    files = [f for f in os.listdir(folder) if "_".join(f.split('_')[:-2]) == filename]  
    times = [f.split('_')[-2] for f in files]

    # if we found no files, skip this object
    if len(times) == 0:
        print 'Failed to find any files to extract times in ' + folder + ' of form ' + filename
        return None,None,None

    # generate output
    mcmc_filename=filebase+'_'+max(times)+"_mcmc.h5"
    model_filename=filebase+'_'+max(times)+"_model"
    postname = mcmc_filename[:-7]+'post'

    if not os.path.isfile(model_filename):
        print 'no model file for ' + mcmc_filename
        model_filename = None
    if not os.path.isfile(mcmc_filename):
        print 'no sampling file for ' + model_filename
        mcmc_filename = None

    return mcmc_filename, model_filename, postname


def load_moustakas_data(objnames = None):

    '''
    specifically written to load optical emission line fluxes, of the "radial strip" variety
    this corresponds to the aperture used in the Brown sample

    if we pass a list of object names, return a sorted, matched list
    otherwise return everything

    returns in units of 10^-15^erg/s/cm^2
    '''

    #### load data
    # arcane vizier formatting means I'm using astropy tables here
    from astropy.io import ascii
    foldername = os.getenv('APPS')+'/threedhst_bsfh/data/Moustakas+10/'
    filename = 'table3.dat'
    readme = 'ReadMe'
    table = ascii.read(foldername+filename, readme=foldername+readme)

    #### filter to only radial strips
    accept = table['Spectrum'] == 'Radial Strip'
    table = table[accept.data]

    #####
    if objnames is not None:
        outtable = []
        for name in objnames:
            match = table['Name'] == name
            if np.sum(match.data) == 0:
                outtable.append(None)
                continue
            else:
                outtable.append(table[match.data])
    else:
        outtable = table

    return outtable

def load_moustakas_newdat(objnames = None):

    '''
    access (new) Moustakas line fluxes, from email in Jan 2016

    if we pass a list of object names, return a sorted, matched list
    otherwise return everything

    returns in units of erg/s/cm^2
    '''

    #### load data
    from astropy.io import fits
    filename = os.getenv('APPS')+'/threedhst_bsfh/data/Moustakas_new/atlas_specdata_solar_drift_v1.1.fits'
    hdulist = fits.open(filename)

    ##### match
    if objnames is not None:
        outtable = []
        objnames = np.core.defchararray.replace(objnames, ' ', '')  # strip spaces
        for name in objnames:
            match = hdulist[1].data['GALAXY'] == name
            if np.sum(match.data) == 0:
                outtable.append(None)
                continue
            else:
                outtable.append(hdulist[1].data[match])
    else:
        outtable = hdulist.data

    return outtable

def save_alldata(alldata,runname='brownseds'):

    output = outpickle+'/'+runname+'_alldata.pickle'
    pickle.dump(alldata,open(output, "wb"))

def write_results(alldata,outfolder):
    '''
    create table for Prospector-Alpha paper, write out in AASTeX format
    '''

    data, errup, errdown, names, fmts, objnames = [], [], [], [], [], []
    objnames = [dat['objname'] for dat in alldata]

    #### gather regular parameters
    par_to_write = ['logmass','dust2','logzsol']
    theta_names = alldata[0]['pquantiles']['parnames']
    names.extend([r'log(M/M$_{\odot}$)',r'$\tau_{\mathrm{diffuse}}$',r'log(Z/Z$_{\odot}$)'])
    fmts.extend(["{:.2f}","{:.2f}","{:.2f}"])
    for p in par_to_write: 
        idx = theta_names == p
        data.append([dat['pquantiles']['q50'][idx][0] for dat in alldata])
        errup.append([dat['pquantiles']['q84'][idx][0]-dat['pquantiles']['q50'][idx][0] for dat in alldata])
        errdown.append([dat['pquantiles']['q50'][idx][0]-dat['pquantiles']['q16'][idx][0] for dat in alldata])

    #### gather error parameters
    epar_to_write = ['sfr_100','ssfr_100','half_time']
    theta_names = alldata[0]['pextras']['parnames']
    names.extend([r'log(SFR)',r'log(sSFR)',r'log(t$_{\mathrm{half}}$)'])
    fmts.extend(["{:.2f}","{:.2f}","{:.2f}"])
    for p in epar_to_write: 
        idx = theta_names == p
        data.append([np.log10(dat['pextras']['q50'][idx][0]) for dat in alldata])
        errup.append([np.log10(dat['pextras']['q84'][idx][0]) - np.log10(dat['pextras']['q50'][idx][0]) for dat in alldata])
        errdown.append([np.log10(dat['pextras']['q50'][idx][0])-np.log10(dat['pextras']['q16'][idx][0]) for dat in alldata])

    #### write formatted data (for putting into the above)
    nobj = len(objnames)
    ncols = len(data)
    with open(outfolder+'results.dat', 'w') as f:
        for i in xrange(nobj):
            f.write(objnames[i])
            for j in xrange(ncols):
                string = ' & $'+fmts[j].format(data[j][i])+'^{+'+fmts[j].format(errup[j][i])+'}_{-'+fmts[j].format(errdown[j][i])+'}$'
                f.write(string)
            f.write(' \\\ \n')

def load_spectra(objname, nufnu=True):
    
    # flux is read in as ergs / s / cm^2 / Angstrom
    # the source key is:
    # 0 = model
    # 1 = optical spectrum
    # 2 = Akari
    # 3 = Spitzer IRS

    foldername = '/Users/joel/code/python/threedhst_bsfh/data/brownseds_data/spectra/'
    rest_lam, flux, obs_lam, source = np.loadtxt(foldername+objname.replace(' ','_')+'_spec.dat',comments='#',unpack=True)

    lsun = 3.846e33  # ergs/s
    flux_lsun = flux / lsun #

    # convert to flam * lam
    flux = flux * obs_lam

    # convert to janskys, then maggies * Hz
    flux = flux * 1e23 / 3631

    out = {}
    out['rest_lam'] = rest_lam
    out['flux'] = flux
    out['flux_lsun'] = flux_lsun
    out['obs_lam'] = obs_lam
    out['source'] = source

    return out

def load_coordinates(dec_in_string=False):

    from astropy.io import fits

    location = '/Users/joel/code/python/threedhst_bsfh/data/brownseds_data/photometry/table1.fits'
    hdulist = fits.open(location)

    ### convert from hours to degrees
    ra, dec = [], []
    for i, x in enumerate(hdulist[1].data['RAm']):
        r = hdulist[1].data['RAh'][i] * (360./24) + hdulist[1].data['RAm'][i] * (360./(24*60)) + hdulist[1].data['RAs'][i] * (360./(24*60*60))
        
        if dec_in_string:
            d = str(hdulist[1].data['DE-'][i])+str(hdulist[1].data['DEd'][i])+' '+str(hdulist[1].data['DEm'][i])+' '+"{:.1f}".format(float(hdulist[1].data['DEs'][i]))
        else:
            d = hdulist[1].data['DEd'][i] + hdulist[1].data['DEm'][i] / 60. + hdulist[1].data['DEs'][i] / 3600.
            if str(hdulist[1].data['DE-'][i]) == '-':
                d = -d
        ra.append(r)
        dec.append(d)

    return np.array(ra),np.array(dec),hdulist[1].data['Name'] 

def write_coordinates():

    outloc = '/Users/joel/code/python/threedhst_bsfh/data/brownseds_data/photometry/coords.dat'
    ra, dec, name = load_coordinates(dec_in_string=True)
    with open(outloc, 'w') as f:
        for r, d in zip(ra,dec):
            f.write(str(r)+', '+d+'; ')

def agn_str_match(dat, bcoords, objname):
    '''used to be simple string matching but issue popped up with floating point accuracy (?)
    now matches based on declination string
    '''

    ### add match
    match = []
    bco_dec = np.array([cord.split(', ')[-1] for cord in bcoords])
    for query in dat['_Search_Offset']:
        match_str = (query.split('('))[1].split(')')[0].split(', ')[-1]
        idx = bco_dec == match_str
        if idx.sum() != 1:
            print 1/0
        match.append(objname[idx][0])
    match = np.array(match)
    dat = np.lib.recfunctions.append_fields(dat, 'match', data=match)

    return dat

def load_csc(bcoords,objname):

    location = '/Users/joel/code/python/threedhst_bsfh/data/brownseds_data/photometry/xray/csc_table.dat'

    #### extract headers
    with open(location, 'r') as f:
        for line in f:
            if line[:5] != '|name':
                continue
            else:
                hdr = line
                hdr = hdr.replace(' ', '').split('|')[:-1]
                break

    #### load
    # names = ('', 'name', 'ra', 'dec', 'significance', 'fb_flux_ap', 'fb_flux_ap_upper', 'fb_flux_ap_lower', 'mb_flux_ap', 
    #          'mb_flux_ap_upper', 'mb_flux_ap_lower', 'hb_flux_ap', 'sb_flux_ap', 'extent_flag', 'hb_flux_ap_upper', 
    #          'sb_flux_ap_upper', 'hb_flux_ap_lower', 'sb_flux_ap_lower', '_Search_Offset')

    dat = np.loadtxt(location, comments = '#', delimiter='|',skiprows=5, dtype = {'names':([str(n) for n in hdr]), \
                     'formats':(['S40','S40','S40','S40','f16','f16','f16','f16','f16','f16','f16','f16','f16','S40','f16','f16','f16','f16','S40'])})
    offset = gather_offset(dat['_Search_Offset'])
    dat = np.lib.recfunctions.append_fields(dat, 'offset', data=offset) 
    dat = agn_str_match(dat,bcoords,objname)

    return dat

def load_cxo(bcoords,objname):

    location = '/Users/joel/code/python/threedhst_bsfh/data/brownseds_data/photometry/xray/cxoxassist_table.dat'

    #### extract headers
    with open(location, 'r') as f:
        for line in f:
            if line[:5] != '|name':
                continue
            else:
                hdr = line
                hdr = hdr.replace(' ', '').split('|')[:-1]
                break

    #### load
    # names = ('', 'name', 'ra', 'dec', 'count_rate', 'count_rate_error', 'flux', 'database_table', 'observatory','error_radius', 'exposure', 'class', 'hardness_ratio', 'hardness_ratio_err', '_Search_Offset')
    dat = np.loadtxt(location, comments = '#', delimiter='|',skiprows=5,
                     dtype = {'names':([str(n) for n in hdr]),\
                              'formats':(['S40','S40','S40','S40','S40','f16','f16','f16','f16','S40','f16','f16','S40'])})
    
    offset = gather_offset(dat['_Search_Offset'])
    dat = np.lib.recfunctions.append_fields(dat, 'offset', data=offset)
    dat = agn_str_match(dat,bcoords,objname)

    return dat

def load_chng(bcoords,objname):

    location = '/Users/joel/code/python/threedhst_bsfh/data/brownseds_data/photometry/xray/chngpscliu_table.dat'

    #### extract headers
    with open(location, 'r') as f:
        for line in f:
            if line[:5] != '|name':
                continue
            else:
                hdr = line
                hdr = hdr.replace(' ', '').split('|')[:-1]
                break

    #### load
    # names = ('', 'name', 'ra', 'dec', 'count_rate', 'count_rate_error', 'flux', 'database_table', 'observatory','error_radius', 'exposure', 'class', '_Search_Offset')
    dat = np.loadtxt(location, comments = '#', delimiter='|',skiprows=5,
                     dtype = {'names':([str(n) for n in hdr]),\
                              'formats':(['S40','S40','S40','S40','S40','f16','f16','f16','f16','f16','S40','S40'])})
    
    ### add offset
    offset = gather_offset(dat['_Search_Offset'])
    dat = np.lib.recfunctions.append_fields(dat, 'offset', data=offset)
    dat = agn_str_match(dat,bcoords,objname)

    return dat

def gather_offset(offset_in):

    # transform offset
    offset = []
    for query in offset_in:
        n = 0
        offset_float = None
        while offset_float == None:
            try: 
                offset_float = float(query.split(' ')[n])
            except ValueError:
                n+=1
        offset.append(offset_float)
    offset = np.array(offset)

    return offset

def load_xray_cat(xmatch = True,maxradius=30):

    '''
    returns flux in (erg/cm^2/s) and object name from brown catalog
    use three large catalogs
    '''

    ### get brown coordinates
    ra, dec, objname = load_coordinates(dec_in_string=True)
    bcoords = []
    for r, d in zip(ra,dec): bcoords.append(str(r)+', '+d)

    ### load up each catalog
    csc = load_csc(bcoords,objname)
    cxo = load_cxo(bcoords,objname)
    chng = load_chng(bcoords,objname)

    #### match based on query string
    if xmatch == True:
        
        #### load Brown positional data
        #### mock up as query parameters

        ### take brightest X-ray detection per object
        flux, flux_err, hardness, hardness_err, database = [], [], [], [], []
        for i, name in enumerate(objname):

            ### find matches in the query with nonzero flux entries within MAXIMUM radius in arcseconds (max is 1')
            idx_csc = (csc['match'] == name) & \
                      (csc['fb_flux_ap'] > 0.0) & \
                      (csc['offset'] < maxradius/30.) & \
                      (np.core.defchararray.strip(csc['extent_flag']) == 'F')
            
            idx_cxo = (cxo['match'] == name) & \
                      (cxo['flux'] > 0.0) & \
                      (cxo['offset'] < maxradius/30.) & \
                      (np.core.defchararray.strip(cxo['extent_flag']) == 'F')

            idx_chng = (chng['match'] == name) & \
                       (chng['flux'] > 0.0) & \
                       (chng['offset'] < maxradius/30.)

            ### PREFER CXO > CSC > CHNG
            if idx_cxo.sum() > 0:
                ### take brightest
                idx = cxo['flux'][idx_cxo].argmax()
                fac = correct_for_window('CXOXASSIST', targlow = 0.5, targhigh = 8)
                #hardness.append(cxo['hardness_ratio'][idx_cxo][idx])
                #hardness_err.append(cxo['hardness_ratio_error'][idx_cxo][idx])
                hardness.append(None)
                hardness_err.append(None)

                flux.append(cxo['flux'][idx_cxo][idx]*fac)
                flux_err.append(fac*cxo['flux'][idx_cxo][idx] * (cxo['counts_error'][idx_cxo][idx]/cxo['counts'][idx_cxo][idx]))
                database.append('CXO')
            elif idx_csc.sum() > 0:
                ### take brightest
                idx = csc['fb_flux_ap'][idx_csc].argmax()
                fac = correct_for_window('CSC', targlow = 0.5, targhigh = 8)
                S = csc['sb_flux_ap'][idx_csc][idx] + csc['mb_flux_ap'][idx_csc][idx]
                H = csc['hb_flux_ap'][idx_csc][idx]
                hardness.append((H-S)/(H+S))
                hardness_err.append(None)               
                flux.append(csc['fb_flux_ap'][idx_csc][idx]*fac)
                flux_err.append((csc['fb_flux_ap_upper'][idx_csc][idx]-csc['fb_flux_ap_lower'][idx_csc][idx])/2.*fac)
                database.append('CSC')
            elif idx_chng.sum() > 0:
                idx = chng['flux'][idx_chng].argmax()
                fac = correct_for_window('CHNGPSCLIU', targlow = 0.5, targhigh = 8)
                hardness.append(None)
                hardness_err.append(None)
                flux.append(chng['flux'][idx_chng][idx]*fac)
                flux_err.append(0.0)
                database.append('CHNG')
            ### if no detections, give it a dummy number
            else:
                flux.append(-99)
                flux_err.append(0.0)
                hardness.append(None)
                hardness_err.append(None)               
                database.append('no match')
                continue 

        out = {'objname':objname,
               'flux':np.array(flux),
               'flux_err':np.array(flux_err),
               'hardness':np.array(hardness),
               'hardness_err':np.array(hardness_err),
               'database':np.array(database)}
        return out
    else:
        return dat

def load_xray_mastercat(xmatch = True,maxradius=30):

    '''
    returns flux in (erg/cm^2/s) and object name from brown catalog
    flux is the brightest x-ray source within 1'
    by taking the brightest over multiple tables, we are biasing high (also blended sources?
        prefer observatory with highest resolution (Chandra ?) to avoid blending
        think about a cut on location (e.g., within 10'', or 30'')
        would like to do ERROR_RADIUS cut but most entries don't have it. similar with EXPOSURE.
        if we could translate COUNT_RATE into FLUX for ARBITRARY TELESCOPE AND DATA TABLE then we could include many more sources
    '''

    location = '/Users/joel/code/python/threedhst_bsfh/data/brownseds_data/photometry/xray/xray_mastercat.dat'

    #### extract headers
    with open(location, 'r') as f:
        for line in f:
            if line[:5] != '|name':
                continue
            else:
                hdr = line
                hdr = hdr.replace(' ', '').split('|')[:-1]
                break

    #### load
    # names = ('', 'name', 'ra', 'dec', 'count_rate', 'count_rate_error', 'flux', 'database_table', 'observatory','error_radius', 'exposure', 'class', '_Search_Offset')
    dat = np.loadtxt(location, comments = '#', delimiter='|',skiprows=5,
                     dtype = {'names':([str(n) for n in hdr]),\
                              'formats':(['S40','S40','S40','S40','f16','f16','f16','S40','S40','S40','S40','S40','S40','S40'])})

    ### remove whitespace from strings
    for i in xrange(dat.shape[0]):
        dat['database_table'][i] = str(np.core.defchararray.strip(dat['database_table'][i]))
        dat['observatory'][i] = str(np.core.defchararray.strip(dat['observatory'][i]))

    #### match based on query string
    if xmatch == True:
        
        #### load Brown positional data
        #### mock up as query parameters
        ra, dec, objname = load_coordinates(dec_in_string=True)
        bcoords = []
        for r, d in zip(ra,dec):
            bcoords.append(str(r)+', '+d)
        match, offset = [], []
        for query in dat['_Search_Offset']:
            match_str = (query.split('('))[1].split(')')[0]
            match.append(objname[bcoords.index(match_str)]) 

            n = 0
            offset_float = None
            while offset_float == None:
                try: 
                    offset_float = float(query.split(' ')[n])
                except ValueError:
                    n+=1
            offset.append(offset_float)

        offset = np.array(offset)

        ### take brightest X-ray detection per object
        flux, flux_err, observatory, database = [], [], [], []
        count = 0
        for i, name in enumerate(objname):

            ### find matches in the query with nonzero flux entries within MAXIMUM radius in arcseconds (max is 1')
            # forbidden datatables either use bandpasses above 10 keV or flux definition is unclear
            # SFGALHMXB is removed because it's a high-mass x-ray binary catalog!
            idx = (np.array(match) == name) & (dat['flux'] != 0.0) & (offset < maxradius/60.) & \
                  (dat['database_table'] != 'INTAGNCAT') & (dat['database_table'] != 'INTIBISAGN') & \
                  (dat['database_table'] != 'BMWHRICAT') & (dat['database_table'] != 'IBISCAT4') & \
                  (dat['database_table'] != 'INTIBISASS') & (dat['database_table'] != 'ULXRBCAT') & \
                  (dat['database_table'] != 'SFGALHMXB')
            ### if no detections, give it a dummy number
            if idx.sum() == 0: #or 'CHANDRA' not in dat['observatory'][idx]:
                print dat['observatory'][idx]
                flux.append(-99)
                flux_err.append(0.0)
                observatory.append('no match')
                database.append('no match')
                continue 

            ### choose the detection to keep
            # prefer chandra
            # of chandra, take the observation closest to centere
            ch_idx = np.core.defchararray.strip(dat['observatory'][idx]) == 'CHANDRA'
            if ch_idx.sum() > 0:
                idx_keep = np.where((offset[idx] == offset[idx][ch_idx].min()) & \
                                    (np.core.defchararray.strip(dat['observatory'][idx]) == 'CHANDRA'))[0][0]
            else:
                idx_keep = dat['flux'][idx].argmax()
            idx_keep = dat['flux'][idx].argmax()

            ### fill out data
            cfactor = correct_for_window(dat['database_table'][idx][idx_keep])
            flux.append(dat['flux'][idx][idx_keep]*cfactor)
            fractional_count_err = dat['count_rate_error'][idx][idx_keep]/dat['count_rate'][idx][idx_keep]
            if np.isnan(fractional_count_err):
                fractional_count_err = 0.0
            flux_err.append(flux[-1] * fractional_count_err)
            observatory.append(dat['observatory'][idx][idx_keep])
            database.append(dat['database_table'][idx][idx_keep])
        print count
        out = {'objname':objname,
               'flux':np.array(flux),
               'flux_err':np.array(flux_err),
               'observatory':np.array(observatory),
               'database':np.array(database)}
        return out
    else:
        return dat

def table_window(table):

    if table == 'ASCAGIS':
        low, high = 0.7, 7.0
    elif table == 'CHNGPSCLIU':
        low, high = 0.3, 8.0
    elif table == 'CSC':
        low, high = 0.5, 7.0
    elif table == 'CXOXASSIST':
        low, high = 0.3, 8.0
    elif table == 'EINGALCAT':
        low, high = 0.2, 4.0
    elif table == 'ETGALXRAY': #CAREFUL
        low, high = 0.088, 17.25
    elif table == 'RASSBSCPGC':
        low, high = 0.1, 2.4
    elif table == 'RASSDSSAGN':
        low, high = 0.1, 2.4
    elif table == 'RBSCNVSS':
        low, high = 0.1, 2.4
    elif table == 'ROSATRQQ':
        low, high = 0.1, 2.4
    elif table == 'ROXA':
        low, high = 0.1, 2.4
    elif table == 'SACSTPSCAT': 
        low, high = 0.5, 8.0
    elif table == 'TARTARUS':
        low, high = (2,7.5), (5,10)
    elif table == 'ULXNGCAT':
        low, high = 0.3, 10
    elif table == 'WGACAT':
        low, high = 0.05, 2.5
    elif table == 'XMMSLEWCLN':
        low, high = 0.2, 12
    elif table == 'XMMSSC':
        low, high = 0.2, 12
    elif table == 'XMMSSCLWBS':
        low, high = 0.2, 12
    else:
        print 1/0

    # return list if it's not a tuple
    try:
        len(low)
    except TypeError: 
        low = [low]
        high = [high]

    return low, high


def correct_for_window(table, targlow = 0.5, targhigh = 8):

    # Want 0.5-8 keV
    # n(E)dE proportional to E^-gamma dE
    # n(E)dE = c E^-gamma dE where c is a constant
    # integrate
    # Etot = int_Elow^Ehigh (c E^-gamma dE)
    # Etot = (1./(-gamma+1)) c E^(-gamma+1) |_Elow^Ehigh
    # Etot = (1./(-gamma+1)) c [Ehigh^(-gamma+1) - Elow^(-gamma+1)]

    factor = 0.0
    gamma = -1.8

    low, high = table_window(table)

    for l, h in zip(low, high):
        factor += h**(gamma+1) - l**(gamma+1)

    fscale = (targhigh**(1+gamma) - targlow**(1+gamma)) / factor

    return fscale

def plot_brown_coordinates():

    '''
    plot the coordinates above
    ''' 

    import matplotlib.pyplot as plt

    ra, dec = load_coordinates()

    plt.plot(ra, dec, 'o', linestyle=' ', mew=2, alpha = 0.8, ms = 10)
    plt.xlabel('Right Ascension [degrees]')
    plt.ylabel('Declination [degrees]')

    plt.show()

def write_spectrum(sample_results,outname='best_fit_spectrum.dat'):


    with open(outname,'w') as f:

            f.write('# First line is wavelength in Angstroms, second line is best-fit flux in maggies (multiply by 3631 to get to Jy)\n')
            for lam in sample_results['observables']['lam_obs']: f.write("{:.1f}".format(lam)+' ')
            f.write('\n')
            for spec in sample_results['bfit']['spec']: f.write("{:.3e}".format(spec)+' ')

def write_villar_data():
    
    from prosp_dutils import generate_basenames

    # All I need is UV (intrinsic and observed), and mid-IR fluxes, sSFRs, SFRs, Mstars, and tau_V  
    filebase, parm_basename, ancilname = generate_basenames('villar')
    ngals = len(filebase)
    mass, sfr100, ssfr100, tauv, met = [np.zeros(shape=(3,ngals)) for i in xrange(5)]
    names = []
    for jj in xrange(ngals):
        
        try:
            extra_output = load_prospector_extra(filebase[jj])
        except TypeError:
            print 'galaxy {0} named '.format(jj)+filebase[jj].split('_')[-1]+' failed to load.'
            print 'not adding this to the output.'
            continue

        if jj == 0:
            mass_idx = np.array(extra_output['extras']['parnames']) == 'stellar_mass'
            tauv_idx = np.array(extra_output['quantiles']['parnames']) == 'dust2'
            met_idx = np.array(extra_output['quantiles']['parnames']) == 'logzsol'
            sfr100_idx = extra_output['extras']['parnames'] == 'sfr_100'
            ssfr100_idx = extra_output['extras']['parnames'] == 'ssfr_100'

        mass[:,jj] = [np.log10(extra_output['extras']['q50'][mass_idx])[0],
                      np.log10(extra_output['extras']['q84'][mass_idx])[0],
                      np.log10(extra_output['extras']['q16'][mass_idx])[0]]

        sfr100[:,jj] = [extra_output['extras']['q50'][sfr100_idx],
                        extra_output['extras']['q84'][sfr100_idx],
                        extra_output['extras']['q16'][sfr100_idx]]
        
        ssfr100[:,jj] = [extra_output['extras']['q50'][ssfr100_idx],
                         extra_output['extras']['q84'][ssfr100_idx],
                         extra_output['extras']['q16'][ssfr100_idx]]
        
        tauv[:,jj] = [extra_output['quantiles']['q50'][tauv_idx],
                      extra_output['quantiles']['q84'][tauv_idx],
                      extra_output['quantiles']['q16'][tauv_idx]]

        met[:,jj] = [extra_output['quantiles']['q50'][met_idx],
                        extra_output['quantiles']['q84'][met_idx],
                        extra_output['quantiles']['q16'][met_idx]]

        names.append(filebase[jj].split('_')[-1])

    outmod = os.getenv('APPS')+'/threedhst_bsfh/data/ashley/ashley_out.dat'

    outdict ={
              'mass': mass,
              'sfr100': sfr100,
              'ssfr100': ssfr100,
              'tauv': tauv,
              'logzsol': met,
             }

    # write out model parameters
    # logmass is in log(M/Msun), Chabrier IMF
    # sfr100 is in Msun/yr
    # ssfr100 is in yr^-1
    # tauv, tautot are in optical depths
    # luv, lirs are in LSUN
    with open(outmod, 'w') as f:
        
        ### header ###
        hdr = '# objname '
        for key in outdict.keys(): hdr += key+' '+key+'_84th '+key+'_16th '
        f.write(hdr+'\n')

        ngals_loaded = len(names)
        for jj in xrange(ngals_loaded):
            f.write(names[jj]+' ')
            for key in outdict.keys():
                fmt = "{:.6e}"
                f.write(fmt.format(outdict[key][0,jj]) + ' ' + fmt.format(outdict[key][1,jj]) + ' ' + fmt.format(outdict[key][2,jj])+' ')
            f.write('\n')

def write_eufrasio_data():

    from prosp_dutils import generate_basenames

    # All I need is UV (intrinsic and observed), and mid-IR fluxes, sSFRs, SFRs, Mstars, and tau_V  
    filebase, parm_basename, ancilname = generate_basenames('brownseds_np')
    ngals = len(filebase)
    mass, sfr100, ssfr100, tauv, tautot, luv, lir, luv0 = [np.zeros(shape=(3,ngals)) for i in xrange(8)]
    names = []
    for jj in xrange(ngals):
        sample_results, powell_results, model = load_prospector_data(filebase[jj])
        
        if jj == 0:
            mass_idx = np.array(sample_results['quantiles']['parnames']) == 'logmass'
            tauv_idx = np.array(sample_results['quantiles']['parnames']) == 'dust2'
            sfr100_idx = sample_results['extras']['parnames'] == 'sfr_100'
            ssfr100_idx = sample_results['extras']['parnames'] == 'ssfr_100'
            tautot_idx = sample_results['extras']['parnames'] == 'total_ext5500'

            nmag = len(sample_results['observables']['mags'])
            fnames = [f.name for f in sample_results['obs']['filters']]
            mags, mags_nodust = [np.zeros(shape=(3,nmag,ngals)) for i in xrange(2)]
            magsobs = np.zeros(shape=(nmag,ngals))

        for kk in xrange(nmag):
            mags[:,kk,jj] = np.percentile(sample_results['observables']['mags'][kk,:],[50.0,84.0,16.0]).tolist()
            mags_nodust[:,kk,jj] = np.percentile(sample_results['observables']['mags_nodust'][kk,:],[50.0,84.0,16.0]).tolist()
            magsobs[kk,jj] = sample_results['obs']['maggies'][kk]

        mass[:,jj] = [sample_results['quantiles']['q50'][mass_idx],
                      sample_results['quantiles']['q84'][mass_idx],
                      sample_results['quantiles']['q16'][mass_idx]]
        
        sfr100[:,jj] = [sample_results['extras']['q50'][sfr100_idx],
                        sample_results['extras']['q84'][sfr100_idx],
                        sample_results['extras']['q16'][sfr100_idx]]
        
        ssfr100[:,jj] = [sample_results['extras']['q50'][ssfr100_idx],
                         sample_results['extras']['q84'][ssfr100_idx],
                         sample_results['extras']['q16'][ssfr100_idx]]
        
        tauv[:,jj] = [sample_results['quantiles']['q50'][tauv_idx],
                      sample_results['quantiles']['q84'][tauv_idx],
                      sample_results['quantiles']['q16'][tauv_idx]]

        tautot[:,jj] = [sample_results['extras']['q50'][tautot_idx],
                        sample_results['extras']['q84'][tautot_idx],
                        sample_results['extras']['q16'][tautot_idx]]

        luv[:,jj] = np.percentile(sample_results['observables']['L_UV'][:],[50.0,84.0,16.0]).tolist()
        lir[:,jj] = np.percentile(sample_results['observables']['L_IR'][:],[50.0,84.0,16.0]).tolist()
        luv0[:,jj] = np.percentile(sample_results['observables']['L_UV_INTRINSIC'][:],[50.0,84.0,16.0]).tolist()

        names.append("_".join(sample_results['run_params']['objname'].split(' ')))
        print tauv[:,jj]
    outmod = '/Users/joel/code/python/threedhst_bsfh/data/eufrasio_model_parameters.txt'
    outdat = '/Users/joel/code/python/threedhst_bsfh/data/eufrasio_photometry_'

    outdict ={
              'mass': mass,
              'sfr100': sfr100,
              'ssfr100': ssfr100,
              'tauv': tauv,
              'tautot': tautot,
              'luv': luv,
              'lir': lir,
              'luv0': luv0,
             }

    outdict_mags =  { 
                    'mags': mags,
                    'mags_nodust': mags_nodust,
                    'magsobs': magsobs
                    }

    # write out model parameters
    # logmass is in log(M/Msun), Chabrier IMF
    # sfr100 is in Msun/yr
    # ssfr100 is in yr^-1
    # tauv, tautot are in optical depths
    # luv, lirs are in LSUN
    with open(outmod, 'w') as f:
        
        ### header ###
        hdr = '# objname '
        for key in outdict.keys(): hdr += key+' '+key+'_84th '+key+'_16th '
        f.write(hdr+'\n')

        for jj in xrange(ngals):
            f.write(names[jj]+' ')
            for key in outdict.keys():
                fmt = "{:.2e}"
                f.write(fmt.format(outdict[key][0,jj]) + ' ' + fmt.format(outdict[key][1,jj]) + ' ' + fmt.format(outdict[key][2,jj])+' ')
            f.write('\n')

    # write out observables
    # fluxes are in maggies (show link to SDSS thing)
    # I set the observed flux errors equal to 5% of the flux density for my work
    # An observed value of 0.0 means that there is no available photometry
    # I include MODEL_WITH_DUST, MODEL_NODUST, and OBSERVED
    with open(outdat+'model.txt', 'w') as f:
        
        ### header ###
        hdr = '# objname '
        for filt in fnames: hdr += filt+' '+filt+'_84th '+filt+'_16th '
        f.write(hdr+'\n')
        key = 'mags'

        # mags shape = (3,nmag,ngals)
        for jj in xrange(ngals):
            f.write(names[jj]+' ')
            for i,filt in enumerate(fnames):
                fmt = "{:.3e}"
                f.write(fmt.format(outdict_mags[key][0,i,jj]) + ' ' + fmt.format(outdict_mags[key][1,i,jj]) +' '+ fmt.format(outdict_mags[key][2,i,jj])+' ')

    with open(outdat+'model_intrinsic.txt', 'w') as f:
        
        ### header ###
        hdr = '# objname '
        for filt in fnames: hdr += filt+' '+filt+'_84th '+filt+'_16th '
        f.write(hdr+'\n')
        key = 'mags_nodust'

        # mags shape = (3,nmag,ngals)
        for jj in xrange(ngals):
            f.write(names[jj]+' ')
            for i,filt in enumerate(fnames):
                fmt = "{:.3e}"
                f.write(fmt.format(outdict_mags[key][0,i,jj]) + ' ' + fmt.format(outdict_mags[key][1,i,jj]) +' '+ fmt.format(outdict_mags[key][2,i,jj])+' ')

    with open(outdat+'observed.txt', 'w') as f:
        
        ### header ###
        hdr = '# objname '
        for filt in fnames: hdr += filt+' '
        f.write(hdr+'\n')
        key = 'magsobs'

        # mags shape = (3,nmag,ngals)
        for jj in xrange(ngals):
            f.write(names[jj]+' ')
            for i,filt in enumerate(fnames):
                fmt = "{:.3e}"
                f.write(fmt.format(outdict_mags[key][i,jj])+' ')

def write_bestfit_photometry():

    import prosp_dutils

    objnames = ['NGC 0628', 'NGC 2798', 'NGC 4559', 'NGC 4579', 'NGC 7331']

    # write out observables
    with open('bfit_phot.dat', 'w') as f:

        for obj in objnames:
            sample_results, powell_results, model = load_prospector_data(None,objname=obj,runname='brownseds_np')
        
            fnames = [filt.name for filt in sample_results['obs']['filters']]
            bfit_maggies = sample_results['bfit']['mags']

            f.write('# '+obj+'\n')
            f.write('# '+" ".join(fnames)+'\n')
            for mag in bfit_maggies: f.write("{:.3e}".format(mag)+' ')
            f.write('\n')


def write_kinney_txt():
    
    filebase, parm_basename, ancilname=prosp_dutils.generate_basenames('virgo')
    ngals = len(filebase)
    mass, sfr10, sfr100, cloudyha, dmass = [np.zeros(shape=(3,ngals)) for i in xrange(5)]
    names = []
    for jj in xrange(ngals):
        sample_results, powell_results, model = load_prospector_data(filebase[jj])
        
        if jj == 0:
            sfr10_ind = sample_results['extras']['parnames'] == 'sfr_10'
            sfr100_ind = sample_results['extras']['parnames'] == 'sfr_100'
            mass_ind = sample_results['quantiles']['parnames'] == 'mass'
            ha_ind = sample_results['model_emline']['name'] == 'Halpha'

            nwav = len(sample_results['observables']['lam_obs'])
            nmag = len(sample_results['observables']['mags'])
            lam_obs = np.zeros(shape=(nwav,ngals))
            spec_obs = np.zeros(shape=(4,nwav,ngals)) # q50, q84, q16, best-fit
            mag_obs = np.zeros(shape=(4,nmag,ngals)) # q50, q84, q16, best-fit


        lam_obs[:,jj] = sample_results['observables']['lam_obs']
        for kk in xrange(nwav):
            spec_obs[:,kk,jj] = np.percentile(sample_results['observables']['spec'][kk,:],[5.0,50.0,95.0]).tolist()+\
                                [sample_results['observables']['spec'][kk,0]]
        for kk in xrange(nmag):
            mag_obs[:,kk,jj] = np.percentile(sample_results['observables']['mags'][kk,:],[5.0,50.0,95.0]).tolist()+\
                                [sample_results['observables']['mags'][kk,0]]

        mass[:,jj] = [sample_results['quantiles']['q50'][mass_ind],
                      sample_results['quantiles']['q84'][mass_ind],
                      sample_results['quantiles']['q16'][mass_ind]]
        
        sfr10[:,jj] = [sample_results['extras']['q50'][sfr10_ind],
                      sample_results['extras']['q84'][sfr10_ind],
                      sample_results['extras']['q16'][sfr10_ind]]
        
        sfr100[:,jj] = [sample_results['extras']['q50'][sfr100_ind],
                      sample_results['extras']['q84'][sfr100_ind],
                      sample_results['extras']['q16'][sfr100_ind]]
        
        dmass[:,jj] = [sample_results['extras']['q50'][dmass_ind],
                      sample_results['extras']['q84'][dmass_ind],
                      sample_results['extras']['q16'][dmass_ind]]

        cloudyha[:,jj] = [sample_results['model_emline']['q50'][ha_ind],
                      sample_results['model_emline']['q84'][ha_ind],
                      sample_results['model_emline']['q16'][ha_ind]]
        names.append(sample_results['run_params']['objname'])

    outobs = '/Users/joel/code/python/threedhst_bsfh/data/virgo/observables.txt'
    outpars = '/Users/joel/code/python/threedhst_bsfh/data/virgo/parameters.txt'

    # write out observables
    with open(outpars, 'w') as f:
        
        ### header ###
        f.write('# name mass mass_errup mass_errdown sfr10 sfr10_errup sfr10_errdown sfr100 sfr100_errup sfr100_errdown dustmass dustmass_errup dustmass_errdown halpha_flux halpha_flux_errup halpha_flux_errdown')
        f.write('\n')

        ### data ###
        for jj in xrange(ngals):
            f.write(names[jj]+' ')
            for kk in xrange(3): f.write("{:.2f}".format(mass[kk,jj])+' ')
            for kk in xrange(3): f.write("{:.2e}".format(sfr10[kk,jj])+' ')
            for kk in xrange(3): f.write("{:.2e}".format(sfr100[kk,jj])+' ')
            for kk in xrange(3): f.write("{:.2f}".format(dmass[kk,jj])+' ')
            for kk in xrange(3): f.write("{:.2e}".format(cloudyha[kk,jj])+' ')
            f.write('\n')

    # write out observables
    with open(outobs, 'w') as f:
        
        ### header ###
        f.write('# lambda, best-fit spectrum, median spectrum, 84th percentile, 16th percentile, best-fit fluxes, median fluxes, 84th percentile flux, 16th percentile flux')
        for jj in xrange(ngals):
            for kk in xrange(nwav): f.write("{:.1f}".format(lam_obs[kk,jj])+' ')
            f.write('\n')
            for kk in xrange(nwav): f.write("{:.3e}".format(spec_obs[3,kk,jj])+' ')
            f.write('\n')
            for kk in xrange(nwav): f.write("{:.3e}".format(spec_obs[0,kk,jj])+' ')
            f.write('\n')
            for kk in xrange(nwav): f.write("{:.3e}".format(spec_obs[1,kk,jj])+' ')
            f.write('\n')
            for kk in xrange(nwav): f.write("{:.3e}".format(spec_obs[2,kk,jj])+' ')
            f.write('\n')
            
            for kk in xrange(nmag): f.write("{:.3e}".format(mag_obs[3,kk,jj])+' ')
            f.write('\n')
            for kk in xrange(nmag): f.write("{:.3e}".format(mag_obs[0,kk,jj])+' ')
            f.write('\n')
            for kk in xrange(nmag): f.write("{:.3e}".format(mag_obs[1,kk,jj])+' ')
            f.write('\n')
            for kk in xrange(nmag): f.write("{:.3e}".format(mag_obs[2,kk,jj])+' ')
            f.write('\n')



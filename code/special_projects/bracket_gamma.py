import numpy as np
from astropy.io import fits
import os, prosp_dutils, hickle
from prospector_io import load_prospector_data
from astropy import constants
from astropy import units as u
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import astropy.coordinates as coord
from astropy.time import Time
from regions import PolygonSkyRegion as poly_region
from regions import write_ds9

plt.ioff()

plotopts = {
         'fmt':'o',
         'ecolor':'k',
         'capthick':0.4,
         'elinewidth':0.4,
         'alpha':0.5,
         'ms':0.0,
         'zorder':-2
        } 

region_files = os.getenv('APPS') + '/prospector_alpha/data/brownseds_data/region_files/'

def is_observable(pdat):
    """are objects observable between SUNSET and SUNRISE at PALOMAR on XXX to YYY?
    where "observable" means ZZZ
    follows http://www.astropy.org/astropy-tutorials/Coordinates.html
    returns minimum airmass
    """

    # create Coordinate object for catalog
    names = pdat['Name']
    ra, dec = [], []
    for i in range(len(names)):
        ra += [str(pdat['RAh'][i])+'h '+str(pdat['RAm'][i])+'m '+str(pdat['RAs'][i])+'s']
        dec += [str(pdat['DE-'][i])+str(pdat['DEd'][i])+'d '+str(pdat['DEm'][i])+'m '+str(pdat['DEs'][i])+'s']
    galcoords = coord.SkyCoord(ra, dec, frame='fk5')

    # location of observatory
    observing_location = coord.EarthLocation(lat='33.3563d', lon='-116.8648d', height=1712*u.m)

    # what times are 18-degree twilight?
    # create array of sun altitude starting @ at 6pm on Friday March 30
    observing_time = Time('2018-3-30 1:00')

    # find start time
    full_night_times = observing_time + np.linspace(0, 6, 100)*u.hour
    full_night_aa_frames = coord.AltAz(location=observing_location, obstime=full_night_times)
    full_night_sun_coos = coord.get_sun(full_night_times).transform_to(full_night_aa_frames)
    starttime = full_night_times[np.abs(full_night_sun_coos.alt.deg+18).argmin()]

    # find end time
    full_night_times = observing_time + np.linspace(6, 14, 100)*u.hour
    full_night_aa_frames = coord.AltAz(location=observing_location, obstime=full_night_times)
    full_night_sun_coos = coord.get_sun(full_night_times).transform_to(full_night_aa_frames)
    endtime = full_night_times[np.abs(full_night_sun_coos.alt.deg+18).argmin()]

    # now calculate MINIMUM AIR MASS between those times
    # first generate time array
    full_night_times = observing_time + np.linspace(0, 14, 250)*u.hour
    idx = (full_night_times > starttime) & (full_night_times < endtime)
    full_night_times = np.array(full_night_times)[idx].tolist()

    # now calculate airmass for all objects
    full_night_aa_frames = coord.AltAz(location=observing_location, obstime=full_night_times)
    min_airmass = []
    for co in galcoords:

        # require that they be above the horizon
        x = co.transform_to(full_night_aa_frames)
        idx = x.alt > 0
        if idx.sum() > 0:
            min_airmass += [np.min(x[idx].secz).value]
        else:
            min_airmass += [np.nan]

    return np.array(min_airmass)

def sky_vector(c1,unit_vector,distance):
    """given a unit vector in RA, DEC and an angular distance in arcseconds
    return the proper coordinates
    """

    # check for the null case
    if distance == 0:
        return c1

    # check for negatives
    if distance < 0:
        distance = np.abs(distance)
        unit_vector *= -1

    # accuracy set by grid spacing
    destinations = np.linspace(-distance.to(u.arcsecond).value/5.,distance.to(u.arcsecond).value*4.,4001)*u.arcsecond
    coords = coord.SkyCoord(c1.ra+unit_vector[0]*(distance+destinations), c1.dec+unit_vector[1]*(distance+destinations),frame='fk5')

    distances = c1.separation(coords).to(u.arcsecond)
    idx = np.abs(distances-distance).argmin()
    if (idx == 0) | (idx == len(destinations)-1):
        print 1/0
    return coords[idx]

def calc_new_position(gcoord,pa,r):
    
    # sanitize inputs
    if (pa < 0):
        pa += 2*np.pi*u.radian
    ra, dec = gcoord.ra.to(u.radian), gcoord.dec.to(u.radian),

    # let's do this numerically (what the actual fuck)
    pa_array = np.linspace(pa-np.pi*u.radian,pa+np.pi*u.radian,20001)

    # this is the DISTANCE in the RA direction (NOT the change in RA coordinates)
    dra = np.arcsin(np.sin(pa_array)*np.sin(r))
    #ra_out = ra + np.arccos((np.cos(dra) - np.cos(np.pi/2.*u.rad - dec)**2) / (np.sin(np.pi/2.*u.rad-dec)**2))
    ra_out = dra/np.cos(dec) + ra

    ddec = np.arctan(np.cos(pa_array)*np.tan(r))
    dec_out = dec+ddec

    coords = coord.SkyCoord(ra_out, dec_out,frame='fk5')
    position_angles = gcoord.position_angle(coords)
    
    idx = np.abs(position_angles - pa).argmin()
    #if (idx == 0) or (idx == len(pa_array)-1):
    #    print 1/0

    return coords[idx]

def make_master_catalog(dat,outfolder,remake_catalog=False,norates=True):
    """makes master catalog for selecting objects

    CATALOG 1 (FOR ME): Objname, RA, DEC, PA, box dimensions, expected Br gamma flux + EW, recessional velocity, RGB,
    nearby OH lines + strengths, sSFR, metallicity, Herschel photometry flag
    CATALOG 2 (CSV): Objname, RA, DEC, equinox (fk5), non-sidereal tracking rate
        -- this is 2' per 10 minutes in direction perpendicular to PA, in [d(RA), d(DEC)]
    CATALOG 3 (CSV): (nearby bright stars)?

    order by brightness, for galaxies with no strong OH lines
    """

    # input catalog DAT has mass, SFR, sSFR, (Br gamma / Pa alpha / H-alpha) (EW/flux/luminosity)

    # additional inputs
    # dictionary of RA, DEC, recessional velocity
    pdat = load_positional_data()
    # list of objects with Herschel information
    hnames = load_herschel_data()
    # PA, box dimensions
    box_data = load_slit_box()

    # remove unobservable galaxies
    lowest_allowed_airmass = 1.6
    min_airmass = is_observable(pdat)
    obs_idx = np.isfinite(min_airmass) & (min_airmass < lowest_allowed_airmass)
    names, min_airmass = pdat['Name'][obs_idx], min_airmass[obs_idx]

    # important observing quantities
    # need exposure time to calculate tracking rate
    slitlength = (30*u.arcsec).to(u.degree)
    exposure_time = 5*u.minute

    # sort by (line flux / photometric area) (surface brightness?)
    # also remove unobservable galaxies here!
    brg_flux = [np.median(dat['Br gamma 21657']['flux'][dat['names'].index(name),:]) for name in names]
    area = []
    for name in names:
        p1, p2 = box_data['phot_size'][box_data['Name']==name.replace(' ','_')][0].split('_')
        longax = np.max([int(p1),int(p2)])
        area += [30*longax]
    sb = brg_flux/np.array(area)
    nidx = sb.argsort()[::-1]
    names, min_airmass = names[nidx], min_airmass[nidx]

    # Part 1: Rogue's gallery.
    # figure info
    if remake_catalog:
        figsize =  (20,20)
        fs = 10
        dx_txt, dy_txt = 0.1, 0.008
        xs_start, ys_start = 0.01,0.98
        xbox_size, ybox_size = 0.24, 0.11
        dx_box = 0.005

        # already observed
        obsed = ['NGC 3690','NGC 3310', 'NGC 4194', 'NGC 4254', 'NGC 4536', 'NGC 5731', 'IC 4553', 
                 'NGC 6052', 'NGC 5653', 'UGCA 166', 'NGC 2798', 'Mrk 33', 'NGC 3627', 'NGC 4321',
                 'NGC 6090', 'Mrk 1490', 'IRAS 17208-0014']
        # these were observed with potentially bad PA definitions
        # check by hand with old version: are they clearly wrong?
        # NGC 6052 should be fine (50x60 vs 60x50)
        # Mrk 33 might be fine as well; it's small and the opposite sense may have covered it
        # can also check to see if there's signal in the observations
        maybe_bad = ['NGC 6052', 'Mrk 33']

        # initialize and start
        xs, ys = xs_start, ys_start
        catnum = 1
        fig = plt.figure(figsize = figsize)
        for i, name in enumerate(names):

            if name in obsed:
                continue

            # RA, dec, cz, hflag
            pidx = pdat['Name'] == name
            ra = str(pdat['RAh'][pidx][0])+'h '+str(pdat['RAm'][pidx][0])+'m '+str(pdat['RAs'][pidx][0])+'s'
            dec = str(pdat['DE-'][pidx][0])+str(pdat['DEd'][pidx][0])+'d '+str(pdat['DEm'][pidx][0])+'m '+str(pdat['DEs'][pidx][0])+'s'
            cz_str = str(pdat['cz'][pidx][0])+ ' km/s'
            hflag = 'has Herschel photometry'
            if name not in hnames:
                hflag = ''

            # PA, box dimensions
            bidx = box_data['Name'] == name.replace(' ','_')
            phot_pa = box_data['phot_pa'][bidx][0]
            aperture = box_data['phot_size'][bidx][0].replace('_',"'x")+"'"

            # physical properties
            didx = np.array(dat['names']) == name
            brg_flux = np.median(dat['Br gamma 21657']['flux'][didx])
            brg_ew = np.median(dat['Br gamma 21657']['ew'][didx])
            mass = np.median(dat['stellar_mass'][didx])
            ssfr = np.log10(np.median(dat['ssfr'][didx]))

            # OH lines
            cz = pdat['cz'][pidx][0]
            lam_cent = 21657*(1+(cz/(3e5)))
            lamlist, strlist = oh_lines(lam_cent,cz)
            lamlist = ", ".join([str(int(lam-lam_cent)) for lam in lamlist])
            strlist = ", ".join([str(int(stren)) for stren in strlist])

            # put it all together
            xt = xs + dx_txt + 0.005
            fig.text(xt,ys+0.001,name,fontsize=fs+2,weight='bold')
            strs = ['RA='+ra,
                    'DEC='+dec,
                    r'log(f$_{\mathrm{Br}\gamma}$)='+'{:.1e}'.format(brg_flux)+r' erg/s/cm$^{2}$',
                    r' EW$_{\mathrm{Br}\gamma}$='+'{:.1f}'.format(brg_ew)+r' $\AA$',
                    r'log(M/M$_{\odot}$)='+'{:.1f}'.format(mass),
                    r' log(sSFR/yr$^{-1}$)='+'{:.2f}'.format(ssfr),
                    r'$\lambda_{\mathrm{Br}\gamma}$='+str(int(lam_cent))+r' $\AA$',
                    r'OH $\Delta \lambda$: '+lamlist+r'$\AA$',
                    'OH strength: '+strlist,
                    'aperture: '+phot_pa+r'$^{\circ}$ PA'+', '+aperture,
                    'min(airmass): '+'{:.2f}'.format(min_airmass[i]),
                    hflag
                    ]
            for i, s in enumerate(strs): fig.text(xt,ys-dy_txt*(i+1),s,fontsize=fs)

            # RGB
            rgb = load_rgb_png(name).swapaxes(0,1)
            ax = fig.add_axes([xs,ys-dx_txt,dx_txt,dx_txt])
            ax.imshow(rgb)
            ax.set_facecolor('white')
            ax.set_axis_off()

            # increment
            xs += xbox_size+dx_box

            # check for right edge of page
            if (xs + xbox_size) > 1:
                ys = ys - ybox_size-dx_box
                xs = xs_start

            # check for off-page
            if (ys-ybox_size) < 0:
                plt.savefig(outfolder+'catalog'+str(catnum)+'.png',dpi=150)
                plt.close()
                fig = plt.figure(figsize = figsize)
                catnum += 1
                xs, ys = xs_start, ys_start

    # Part 2: target list
    # Objname, RA, DEC, equinox (fk5), non-sidereal tracking rate
    # RA: Specified in sexagesimal hours, minutes and seconds.  Internal fields are separated by spaces.
    # DEC:   Specified in sexagesimal degrees, minutes and seconds.
    # equinox: J2000
    # RA_track: arcseconds / second
    # DEC_track: arcseconds / second
    objname_list, ra_list, dec_list, ra_track_list, dec_track_list, slit_pa = [[] for i in range(6)]
    for i, name in enumerate(names):
        
        # create coord.SkyCoord object
        # this represents the center of the box
        pidx = pdat['Name'] == name
        ra = str(pdat['RAh'][pidx][0])+'h '+str(pdat['RAm'][pidx][0])+'m '+str(pdat['RAs'][pidx][0])+'s'
        dec = str(pdat['DE-'][pidx][0])+str(pdat['DEd'][pidx][0])+'d '+str(pdat['DEm'][pidx][0])+'m '+str(pdat['DEs'][pidx][0])+'s'
        galcoords = coord.SkyCoord(ra, dec, frame='fk5')

        # grab PA
        bidx = box_data['Name'] == name.replace(' ','_')
        phot_pa = float(box_data['phot_pa'][bidx][0])

        # grab aperture, translate to sky box sizes
        # the given PA describes the position of the FIRST axis
        # if this is not the LONGEST axis, redefine by adding 90 to PA
        # this means we'll need to output a list of PAs (put it in the tracking output)
        aperture = box_data['phot_size'][bidx][0]
        ap1, ap2 = aperture.split('_')
        aps = np.array([float(ap1),float(ap2)])/3600.*u.deg # from arcsec to degrees
        if aps[0] < aps[1]:
            phot_pa -= 90
            if (phot_pa < 0):
                phot_pa += 360
            print name
        shortax, longax = aps.min(), aps.max() 
        shortax = 30*u.arcsecond
        phot_pa_rad = np.pi/180. * phot_pa * u.radian

        # generate starting position
        # check that it has proper distance and PA
        bot_mid = calc_new_position(galcoords, phot_pa_rad, longax.to(u.radian)/2.)
        np.testing.assert_almost_equal(galcoords.separation(bot_mid).to(u.arcsec).value,longax.to(u.arcsec).value/2,decimal=0)
        np.testing.assert_almost_equal(galcoords.position_angle(bot_mid).to(u.degree).value,phot_pa,decimal=1)
        """
        # create .region file with proper scan positions
        pclose = [calc_new_position(bot_mid, phot_pa_rad+np.pi/2.*u.radian,slitlength.to(u.radian)/2.),
                  calc_new_position(bot_mid, phot_pa_rad-np.pi/2.*u.radian,slitlength.to(u.radian)/2.)]
        pfar = [calc_new_position(pclose[1], -phot_pa_rad,longax.to(u.radian)),
                calc_new_position(pclose[0], -phot_pa_rad,longax.to(u.radian))]
        points = [pfar+pclose]
        sregions = [poly_region(point) for point in points]
        write_ds9(sregions, region_files+name.replace(' ','_')+'.reg')
        """
        # non sidereal tracking rate, must be output in RA and DEC (arcseconds/hour)
        # we travel LONGAX in EXPOSURE_TIME, in the -PA_CAT_PARALLEL direction
        # calculate this for one slit, spherical geometry negligible
        # far_point = sky_vector(x_slits[0],(-1)*pa_cat_parallel,longax)
        # rate = [(far_point.ra-x_slits[0].ra)/exposure_time,(far_point.dec-x_slits[0].dec)/exposure_time]
        pa_cat_parallel = np.array([np.sin(phot_pa_rad),np.cos(phot_pa_rad)])
        dist = (-1)*pa_cat_parallel*longax
        rate = dist/exposure_time
        rate = [x.to(u.arcsecond/u.hour) for x in rate]

        # create .region file with proper scan positions
        pclose = [calc_new_position(bot_mid, phot_pa_rad+np.pi/2.*u.radian,slitlength.to(u.radian)/2.),
                  calc_new_position(bot_mid, phot_pa_rad-np.pi/2.*u.radian,slitlength.to(u.radian)/2.)]
        pfar = [coord.SkyCoord(pclose[1].ra+dist[0]/np.cos(pclose[0].dec.to(u.rad)),pclose[1].dec+dist[1]),
                coord.SkyCoord(pclose[0].ra+dist[0]/np.cos(pclose[0].dec.to(u.rad)),pclose[0].dec+dist[1])]
        points = [pfar+pclose]
        sregions = [poly_region(point) for point in points]
        write_ds9(sregions, region_files+name.replace(' ','_')+'.reg')


        # add to lists for output
        objname_list += [name.replace(' ','_')]
        ra,dec = bot_mid.to_string('hmsdms').split(' ')
        ra_list += [ra.split('h')[0]  + ' ' + ra.split('m')[0].split('h')[-1] + ' ' + ra.split('m')[-1][:-1]]
        dec_list += [dec.split('d')[0]  + ' ' + dec.split('m')[0].split('d')[-1] + ' ' + dec.split('m')[-1][:-1]]
        ra_track_list += [rate[0].value]
        dec_track_list += [rate[1].value]
        slit_pa += [phot_pa-90]

    # sky
    sky_objname, sky_ra, sky_dec = sky_lists()

    # write to CSV
    if norates:
        outloc = outfolder+'target_list_norates.csv'
        with open(outloc, 'w') as f:
                for i in range(len(objname_list)):
                    f.write(objname_list[i]+','+ra_list[i]+','+dec_list[i]+',J2000')#,{:.2f},{:.2f}'.format(
                            #ra_track_list[i],dec_track_list[i]))
                    f.write('\n')

                # write marla objects
                mnames = ['2M00444105+8351358','2M14390944+4953029']
                mcoords = [coord.SkyCoord(11.171077*u.degree, 83.859955*u.degree, frame='fk5'),
                           coord.SkyCoord(219.789356*u.degree, 49.884155*u.degree, frame='fk5')]
                for i in range(2):
                    ra, dec = mcoords[i].to_string('hmsdms').split(' ')
                    ra_str = ra.split('h')[0]  + ' ' + ra.split('m')[0].split('h')[-1] + ' ' + ra.split('m')[-1][:-1]
                    dec_str = dec.split('d')[0]  + ' ' + dec.split('m')[0].split('d')[-1] + ' ' + dec.split('m')[-1][:-1]
                    f.write(mnames[i]+','+ra_str+','+dec_str+',J2000')#,0.0,0.0')
                    f.write('\n')    
                
                for i in range(len(sky_objname)):
                    f.write(sky_objname[i].replace(' ','_')+'_sky,'+sky_ra[i]+','+sky_dec[i]+',J2000')
                    f.write('\n')

    else:
        outloc = outfolder+'target_list.csv'
        with open(outloc, 'w') as f:
            for i in range(len(objname_list)):
                f.write(objname_list[i]+','+ra_list[i]+','+dec_list[i]+',J2000,{:.2f},{:.2f},{:.1f}'.format(
                        ra_track_list[i],dec_track_list[i],slit_pa[i]))
                f.write('\n')

            # write marla objects
            mnames = ['2M00444105+8351358','2M14390944+4953029']
            mcoords = [coord.SkyCoord(11.171077*u.degree, 83.859955*u.degree, frame='fk5'),
                       coord.SkyCoord(219.789356*u.degree, 49.884155*u.degree, frame='fk5')]
            for i in range(2):
                ra, dec = mcoords[i].to_string('hmsdms').split(' ')
                ra_str = ra.split('h')[0]  + ' ' + ra.split('m')[0].split('h')[-1] + ' ' + ra.split('m')[-1][:-1]
                dec_str = dec.split('d')[0]  + ' ' + dec.split('m')[0].split('d')[-1] + ' ' + dec.split('m')[-1][:-1]
                f.write(mnames[i]+','+ra_str+','+dec_str+',J2000,0.0,0.0,0.0')
                f.write('\n')    
                   
            for i in range(len(sky_objname)):
                f.write(sky_objname[i].replace(' ','_')+'_sky,'+sky_ra[i]+','+sky_dec[i]+',J2000,0,0,0')
                f.write('\n')

def sky_lists():

    objname = ['NGC 3690','UGCA 166','Mrk 1450','UM 461','UGCA 219', 'UGC 06850','UGCA 410', 'Mrk 0475','UGC 08335 SE',
               'Haro 06','IRAS 17208-0014', 'IRAS 08572+3915', 'NGC 6090','Mrk 1490','UGC 06665', 'UGCA 208','Mrk 33',
               'UGC 08696', 'NGC 5992','UGC 08335 NW','IC 0691','NGC 2798','CGCG 049-057','IC 0883','NGC 3773',
               'NGC 2798','NGC 4631', 'NGC 5713','NGC 5194','NGC 4321','NGC 5055','NGC 3627','NGC 3351','NGC 4088',
               'NGC 3521','NGC 3049', 'NGC 5953', 'NGC 2388','NGC 5256','NGC 4826','NGC 5257','IC 0860',
               'NGC 4569','NGC 4670','NGC 3938','NGC 3079','NGC 4385','NGC 6240','NGC 4559','NGC 5258']
    ra = ['11 28 08.484','9 33 55.309','11 38 40.451','11 51 33.252','10 49 07.986', '11 52 37.432', '15 37 15.438', '14 39 10.461', '13 15 28.182',
               '12 15 14.875', '17 23 27.374', '9 00 34.228', '16 11 47.564','14 19 41.461','11 42 12.003', '10 16 32.266','10 32 32.765',
                '13 44 35.440', '15 44 30.270','13 15 28.238','11 26 40.704','9 17 14.413', '15 13 12.813', '13 20 38.710', '11 38 19.983',
                '9 17 23.532', '12 42 19.984','14 40 17.889','13 29 29.540','12 23 08.371','13 16 01.884','11 20 01.888','10 44 10.322','12 05 15.515',
                '11 06 2.342','9 54 56.896','15 34 35.440','7 29 01.122','13 38 21.793','12 56 36.661','13 39 43.696','13 15 07.441',
                '12 36 39.144','12 45 08.361','11 52 36.498','10 01 25.471','12 25 37.081','16 53 02.566','12 35 46.870','13 40 03.454']
    dec = ['+58 33 19.101', '+55 14 37.616','+57 51 00.489','-2 23 56.088','+52 18 33.347', '-2 26 28.869', '+55 15 51.308', '+36 47 03.112','+62 06 29.627',
                '+5 47 03.598', '-0 16 30.916', '+39 03 21.140', '+52 28 42.638','+49 12 28.036','+0 22 00.197', '+45 20 46.299','+54 26 09.373',
                '+55 55 02.501', '+41 05 33.818','+62 06 33.191', '+59 07 48.174','+42 01 16.876', '+7 15 28.045', '+34 09 36.047', '+12 06 49.807',
                '+42 02 20.191', '+32 29 46.809', '-0 18 46.472','+47 13 46.870','+15 47 06.804','+41 57 57.059','+13 00 19.842','+11 40 24.295','+50 32 53.076',
                '-0 05 04.918','+9 15 34.584','+15 10 08.991','+33 49 39.794','+48 15 27.673','+21 45 03.135','+0 51 13.157','+24 38 28.736',
                '+13 11 30.781','+27 07 05.086','+44 09 18.894','+55 40 36.241','+0 33 16.739','+2 24 59.332','+27 57 52.155','+0 50 52.450']

    return objname, ra, dec

def load_rgb_png(name):

    #### load png
    imgname=os.getenv('APPS')+'/prospector_alpha/data/brownseds_data/rgb/'+name.replace(' ','_')+'.png'
    img=mpimg.imread(imgname)

    return img

def oh_lines(lam,v,dv=500):
    """ data from Oliva+15 http://adsabs.harvard.edu/abs/2015A%26A...581A..47O
    return list of lines + strengths for a given lambda (Angstrom), velocity, and velocity width (km/s)
    strengths are normalized such that strongest line is 10^4
    """

    dloc = '/Users/joel/data/triplespec/oh_lines/'

    d1 = np.loadtxt(dloc+'table1.dat', comments = '#', delimiter=' ', 
                    dtype = {'formats':('f16','S40','f16','S40','f16'),'names':('wav1','nam1','wav2','name2','strength')})
    d2 = np.loadtxt(dloc+'table2.dat', comments = '#', dtype = {'formats':('f16','S40','f16'),'names':('wav','nam','strength')})

    dlam = (dv/3e5)*lam

    w1 = np.abs(d1['wav1']-lam) < dlam
    w2 = np.abs(d2['wav']-lam) < dlam

    lamlist = d1['wav1'][w1].tolist() + d2['wav'][w2].tolist()
    strengthlist = d1['strength'][w1].tolist() + d2['strength'][w2].tolist()

    return lamlist, strengthlist

def make_plots(dat,outfolder=None,errs=True):

    # set it up
    fig, ax = plt.subplots(1,3, figsize=(16, 5))
    fs = 18 # font size
    ms = [8,7] # marker size
    alpha = [0.9,0.65]
    zorder = [1,-1]
    sfr_min = 0.01
    colors = ['#FF420E', '#545454','#31A9B8']
    plotlines = ['Br gamma 21657','H alpha 6563']
    plotlabels = [r'Br-$\gamma$ 21657',r'H$\alpha$ 6563']

    # make plots
    for i, line in enumerate(plotlines):
        xplot_low, xplot, xplot_high = np.percentile(dat[line]['ew'],[16,50,84],axis=1)  
        yplot_low, yplot, yplot_high = np.percentile(dat[line]['flux'],[16,50,84],axis=1)
        xplot_err = prosp_dutils.asym_errors(xplot,xplot_high,xplot_low)
        yplot_err = prosp_dutils.asym_errors(yplot,yplot_high,yplot_low)
        if errs:
            ax[0].errorbar(xplot, yplot, xerr=xplot_err, yerr=yplot_err,
                           **plotopts)
        ax[0].plot(xplot, yplot, 'o', linestyle=' ', 
                label=plotlabels[i], alpha=alpha[i], markeredgecolor='k',
                color=colors[i],ms=ms[i],zorder=zorder[i])
    
    # plot geometry
    ax[0].set_xlabel(r'equivalent width [\AA]',fontsize=fs)
    ax[0].set_ylabel(r'flux [erg/s/cm$^{2}$]',fontsize=fs)
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].xaxis.set_tick_params(labelsize=fs)
    ax[0].yaxis.set_tick_params(labelsize=fs)
    ax[0].legend(prop={'size':fs*0.8},frameon=False)

    # make plots
    xplot_low, xplot, xplot_high = np.percentile(dat['sfr'],[16,50,84],axis=1)
    idx = xplot > sfr_min  
    xplot_low, xplot, xplot_high = xplot_low[idx], xplot[idx], xplot_high[idx]
    xplot_err = prosp_dutils.asym_errors(xplot,xplot_high,xplot_low)
    for i, line in enumerate(plotlines):
        yplot_low, yplot, yplot_high = np.percentile(np.log10(dat[line]['lum'][idx]),[16,50,84],axis=1)
        yplot_err = prosp_dutils.asym_errors(yplot,yplot_high,yplot_low)
        if errs:
            ax[1].errorbar(xplot, yplot, xerr=xplot_err, yerr=yplot_err,
                           **plotopts)
        ax[1].plot(xplot, yplot, 'o', linestyle=' ', 
                label=plotlabels[i], alpha=alpha[i], markeredgecolor='k',
                color=colors[i],ms=ms[i])
    
    # plot geometry
    ax[1].set_xlabel(r'SFR [M$_{\odot}$/yr]',fontsize=fs)
    ax[1].set_xscale('log')
    ax[1].set_ylabel(r'log(L/L$_{\odot}$) [dust attenuated]',fontsize=fs)
    ax[1].xaxis.set_tick_params(labelsize=fs)
    ax[1].yaxis.set_tick_params(labelsize=fs)
    ax[1].legend(prop={'size':fs*0.8},frameon=False)

    # make plots
    yplot_low, yplot, yplot_high = np.percentile(np.log10(np.array(dat[plotlines[0]]['flux']) / np.array(dat[plotlines[1]]['flux']))[idx],[16,50,84],axis=1)
    yplot_err = prosp_dutils.asym_errors(yplot,yplot_high,yplot_low)
    if errs:
        ax[2].errorbar(xplot, yplot, xerr=xplot_err, yerr=yplot_err,
                       **plotopts)
    ax[2].plot(xplot, yplot, 'o', linestyle=' ', 
            label=line, alpha=alpha[i], markeredgecolor='k',
            color=colors[1],ms=ms[i],zorder=zorder[i])
    
    # plot geometry
    ax[2].set_xlabel(r'SFR [M$_{\odot}$/yr]',fontsize=fs)
    ax[2].set_xscale('log')
    ax[2].set_ylabel(r'log(F$_{\mathrm{Br-}\gamma}$/F$_{\mathrm{H}\alpha}$)',fontsize=fs)
    ax[2].xaxis.set_tick_params(labelsize=fs)
    ax[2].yaxis.set_tick_params(labelsize=fs)
    ax[2].set_ylim(-2.2,-1.15)

    ax[2].axhline(yplot.min(), linestyle='--', color='k',lw=2,zorder=-3)
    ax[2].text(np.median(xplot)*1.5,yplot.min()+0.015,'atomic ratio',fontsize=14,weight='semibold')

    ax[2].arrow(sfr_min*2.5, -1.6, 0.0, 0.1,
                head_width=0.01, width=0.002,color='#FF3D0D')
    ax[2].text(sfr_min*2.5,-1.65,'dust',color='#FF3D0D',ha='center',fontsize=14)

    plt.tight_layout()
    fig.savefig(outfolder+'brgamma_halpha.png',dpi=150)
    plt.close()

def do_all(runname='brownseds_agn', regenerate=False,errs=True,**opts):
    # I/O folder
    outfolder = os.getenv('APPS')+'/prospector_alpha/plots/'+runname+'/bracket_gamma/'
    if not os.path.isdir(outfolder):
        os.makedirs(outfolder)
    dat = get_galaxy_properties(regenerate=regenerate,runname=runname,outfolder=outfolder)
    make_master_catalog(dat,outfolder,**opts)
    make_plots(dat,outfolder=outfolder,errs=errs)

def get_galaxy_properties(runname='brownseds_agn', regenerate=False, outfolder=None):
    """Loads output, runs post-processing.
    Measure luminosity, fluxes, EW for (Br_gamma, Paschen_alpha, H_alpha)
    Measure mass, SFR, sSFR
    currently best-fit only!
    """

    # skip processing if we don't need new data
    filename = outfolder+'bracket_gamma.hickle'
    if os.path.isfile(filename) and regenerate == False:
        with open(filename, "r") as f:
            outdict = hickle.load(f)
        return outdict

    # build output dictionary
    outlines = ['Br gamma 21657','Pa alpha 18752','H alpha 6563']
    outpars = ['stellar_mass', 'ssfr', 'sfr','logzsol']
    outdict = {'names':[]}
    ngal, nsamp = 129, 300
    for par in outpars: outdict[par] = np.zeros(shape=(ngal,nsamp))
    for line in outlines:
        outdict[line] = {}
        outdict[line]['lum'] = np.zeros(shape=(ngal,nsamp))
        outdict[line]['flux'] = np.zeros(shape=(ngal,nsamp))
        outdict[line]['ew'] = np.zeros(shape=(ngal,nsamp))

    # interface with FSPS lines
    loc = os.getenv('SPS_HOME')+'/data/emlines_info.dat'
    dat = np.loadtxt(loc, delimiter=',', dtype = {'names':('lam','name'),'formats':('f16','S40')})
    line_idx = [dat['name'].tolist().index(line) for line in outlines]

    # iterate over sample
    basenames, _, _ = prosp_dutils.generate_basenames(runname)
    for i, name in enumerate(basenames):
        sample_results, powell_results, model, eout = load_prospector_data(name,hdf5=True,load_extra_output=True)
        outdict['names'].append(str(sample_results['run_params']['objname']))

        # instatiate sps, make fake obs dict
        if i == 0:
            import brownseds_agn_params
            sps = brownseds_agn_params.load_sps(**sample_results['run_params'])
            fake_obs = {'maggies': None, 
                        'phot_mask': None,
                        'wavelength': None, 
                        'filters': []}

        # generate model
        for k in xrange(nsamp):

            spec,mags,sm = model.mean_model(eout['quantiles']['sample_chain'][k,:] , fake_obs, sps=sps)
            mformed = float(10**eout['quantiles']['sample_chain'][k,0])
            
            for j, line in enumerate(outlines):

                # line luminosity (Lsun / [solar mass formed])
                llum = sps.get_nebline_luminosity[line_idx[j]] * mformed
                lumdist = model.params['lumdist'][0] * u.Mpc.to(u.cm)

                # flux [erg / s / cm^2]
                lflux = llum * constants.L_sun.cgs.value / (4*np.pi*lumdist**2)

                # continuum [erg / s / cm^2 / AA]
                spec_in_units = spec * 3631 * 1e-23 * (3e18/sps.wavelengths**2)
                if line == 'H alpha 6563':
                    idx = (sps.wavelengths > 6400) & (sps.wavelengths < 6700)
                    continuum = np.median(spec_in_units[idx])
                else:
                    continuum = np.interp(dat['lam'][line_idx[j]], sps.wavelengths, spec_in_units)
                ew = lflux / continuum

                outdict[line]['lum'][i,k] = float(llum)
                outdict[line]['flux'][i,k] = float(lflux)
                outdict[line]['ew'][i,k] = float(ew)

            # save galaxy parameters
            outdict['stellar_mass'][i,k] = np.log10(mformed * sm)
            outdict['logzsol'][i,k] = float(eout['quantiles']['sample_chain'][k,eout['quantiles']['parnames'] == 'logzsol'])
            outdict['sfr'][i,k] = float(eout['extras']['flatchain'][k,eout['extras']['parnames'] == 'sfr_100'])
            outdict['ssfr'][i,k] = float(eout['extras']['flatchain'][k,eout['extras']['parnames'] == 'ssfr_100'])
        print i

    hickle.dump(outdict,open(filename, "w"))
    return outdict

def load_slit_box():
    """ returns slit dimensions and PA
    """

    dloc = '/Users/joel/code/python/prospector_alpha/data/brownseds_data/photometry/structure.dat'
    hdr = ['Name','phot_size','phot_pa','spectrum','akari_size','akari_pa','spitzer_sl_size','spitzer_sl_pa','spitzer_ll_size','spitzer_ll_pa']
    dtype = {'names':([n for n in hdr]),'formats':('S40','S40','S40','S40','S40','S40','S40','S40','S40','S40')}
    dat = np.loadtxt(dloc,dtype=dtype)
    return {d:dat[d] for d in ['Name','phot_size','phot_pa']}

def load_positional_data():
    """ must return:
    RAh, RAm, RAs, DE-, DEd, DEm, DEs, cz from table1
    """

    # open FITS file
    datname = os.getenv('APPS')+'/prospector_alpha/data/brownseds_data/photometry/table1.fits'
    hdulist = fits.open(datname)

    # create output and exit
    outarrs = ['Name', 'RAh', 'RAm', 'RAs', 'DE-', 'DEd', 'DEm', 'DEs', 'cz']
    out = {n: hdulist[1].data[n] for n in outarrs}
    return out

def load_herschel_data():
    """ must return list of galaxies with Herschel data
    """

    datname = os.getenv('APPS')+'/prospector_alpha/data/brownseds_data/photometry/table1.fits'
    hdulist = fits.open(datname)
    herschname = os.getenv('APPS')+'/prospector_alpha/data/brownseds_data/photometry/kingfish.brownapertures.flux.fits'
    herschel = fits.open(herschname)

    outnames = []
    for i, name in enumerate(herschel[1].data['Name']):

        idx = hdulist[1].data['Name'].lower().replace(' ','') == name
        if herschel[1].data['pacs70'][i] == 0:
            continue

        outnames += [hdulist[1].data['Name'][idx][0]]

    return outnames

def calculate_tracking_rate(ra,dec,PA,box_size,exp_length):
    """this takes in coordinates, PA, box size
    and calculates tracking rate for an exposure of X minutes
    """
    pass

def locations():
    """prints RA, DEC of galaxies in the Brown sample
    """

    ### locations
    datname = os.getenv('APPS')+'/prospector_alpha/data/brownseds_data/photometry/table1.fits'
    herschname = os.getenv('APPS')+'/prospector_alpha/data/brownseds_data/photometry/kingfish.brownapertures.flux.fits'
    photname = os.getenv('APPS')+'/prospector_alpha/data/brownseds_data/photometry/table3.txt'

    ### open
    herschel = fits.open(herschname)
    hdulist = fits.open(datname)

    match=0
    for i, name in enumerate(herschel[1].data['Name']):

        idx = hdulist[1].data['Name'].lower() .replace(' ','') == name
        if herschel[1].data['pacs70'][i] == 0:
            continue

        print name + ' RA:' + str(hdulist[1].data['RAh'][i]) + 'h ' + str(hdulist[1].data['RAm'][i])\
                   + 'm, Dec: ' + str(hdulist[1].data['DEd'][i]) + 'd ' + str(hdulist[1].data['DEm'][i])+'m'
        match +=1

    print match












import numpy as np
import matplotlib.pyplot as plt
import agn_plot_pref
from corner import quantile
import os
from matplotlib.ticker import MaxNLocator
import matplotlib.image as mpimg
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord  # High-level coordinates
import brown_io
from matplotlib.colors import LinearSegmentedColormap
from astropy.stats import sigma_clipped_stats
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

dpi = 150
red = '#FF3D0D'
blue = '#1C86EE' 

banned_list = ['NGC 5055', 'NGC 5953', 'UGC 08696', 'NGC 5258', 'NGC 4676 A', 'UGC 09618 N', 'NGC 1144']
banned_list = []
noise_list = ['UGC 09618','NGC 4676 A', 'UGC 08696', 'NGC 4914', 'II Zw 096', 'UGC 08335 NW']

def collate_data(alldata):

    ### extra parameters
    eparnames_all = alldata[0]['pextras']['parnames']
    eparnames = ['stellar_mass','sfr_100', 'ssfr_100', 'fmir']

    ### setup dictionary
    outq = {'objname':[]}
    for par in eparnames: outq[par] = []

    ### fill with data
    for dat in alldata:
        outq['objname'].append(dat['objname'])
        for par in eparnames:
            match = eparnames_all == par
            outq[par].append(np.log10(dat['pextras']['q50'][match][0]))

    for key in outq: outq[key] = np.array(outq[key])
    return outq

def plot(runname='brownseds_agn',alldata=None,outfolder=None,**popts):

    #### load alldata
    if alldata is None:
        alldata = brown_io.load_alldata(runname=runname)

    #### make output folder if necessary
    if outfolder is None:
        outfolder = os.getenv('APPS')+'/prospector_alpha/plots/'+runname+'/agn_plots/'
        if not os.path.isdir(outfolder):
            os.makedirs(outfolder)

    #### collate data
    pdata = collate_data(alldata)

    ### star-forming sequence plot
    plot_sfseq(pdata,outfolder=outfolder,**popts)
    
def plot_sfseq(pdata,outfolder=None,sigclip=True,name_label=False,**popts):

    #### set up figure geometry
    xlim = [9., 11.5]
    ylim = [-2.,2.]
    delta_im = 0.05
    delx, dely = (xlim[1]-xlim[0])*delta_im, (ylim[1]-ylim[0])*delta_im

    #### randomly sample subject to constraints
    logM_min, logM_max = xlim[0]+delx/2., xlim[1]-delx/2.
    logSFR_min, logSFR_max = ylim[0]+dely/2., ylim[1]-dely/2.
    overlap = 1.0
    m, sfr, fmir, name = [[] for i in range(4)]
    
    good = np.where(
                    (pdata['stellar_mass'] > logM_min) & \
                    (pdata['stellar_mass'] < logM_max) & \
                    (pdata['sfr_100'] > logSFR_min) & \
                    (pdata['sfr_100'] < logSFR_max) & \
                    np.array([obj not in banned_list for obj in pdata['objname']],dtype=bool)
                  )[0]
    print 'total number of objects in window: ' + str(len(good))

    ### allowed shifts
    scale = 1.2
    nshift = 30 # must be even
    shiftx = np.concatenate((np.linspace(0,delx*scale,nshift/2+1),np.linspace(-delx*scale/nshift/2,-delx*scale,nshift/2)))
    shifty = np.concatenate((np.linspace(0,dely*scale,nshift/2+1),np.linspace(-dely*scale/nshift/2,-dely*scale,nshift/2)))

    for i, newseed in enumerate(np.linspace(1,25,25)):

        ### use galaxies with N highest FMIR
        ### otherwise sample randomly
        np.random.seed(seed=int(newseed))
        ntop = 10
        arg = good[pdata['fmir'][good].argsort()][::-1]
        np.random.shuffle(arg[ntop:])

        ### newlist
        m.append([])
        sfr.append([])
        fmir.append([])
        name.append([])

        for idx in arg:

            ### shift until we're in the clear
            win = False
            for sx in shiftx:
                for sy in shifty:
                    stellar_mass = pdata['stellar_mass'][idx] + sx
                    sfr_100 = pdata['sfr_100'][idx] + sy
                    if ((np.abs(stellar_mass-np.array(m[i])) > delx*overlap) | \
                       (np.abs(sfr_100-np.array(sfr[i])) > dely*overlap)).all():
                        win = True
                        break
                if win:
                    break

            ### if we can't fit it, try the next one
            if not win:
                continue

            ### add it in
            m[i] = m[i] + [stellar_mass]
            sfr[i] = sfr[i] + [sfr_100]
            fmir[i] = fmir[i] + [pdata['fmir'][idx]]
            name[i] = name[i] + [pdata['objname'][idx]]

    nsize = [len(n) for n in m]
    amax = np.array(nsize).argmax()
    print str(nsize[amax]) + ' objects fit in'
    m, sfr, fmir, name = m[amax], sfr[amax], fmir[amax], name[amax]

    #### colormap scaling
    fmir = np.array(fmir)
    fmir_min, fmir_max = fmir.min(), fmir.max()
    fmir_mid = (fmir_max+fmir_min)/2.

    ### let us begin
    fig, ax = plt.subplots(1,1, figsize=(8, 6.5))
    for i in xrange(len(m)):

        ### load FITS file
        imgname = os.getenv('APPS')+'/prospector_alpha/data/brownseds_data/fits/'+name[i].replace(' ','_')+'_SDSS_i.fits'
        dat = fits.open(imgname)
        
        ### load image center + dimensions
        ra, dec = load_coordinates(name[i])
        phot_size = load_structure(name[i],long_axis=True) # in arcseconds
        if name[i] == 'IRAS 17208-0014':
            phot_size *= 0.7

        ### translate object location into pixels using WCS coordinates
        px_scale = (dat[0].header['CD1_2']**2 + dat[0].header['CD2_2']**2)**0.5 * 3600  # arcseconds per pixel
        wcs = WCS(dat[0].header)
        pix_center = wcs.all_world2pix([[ra[0],dec[0]]],1)
        size = img_extent(pix_center, px_scale, phot_size, dat[0].data.shape)
        data = dat[0].data[size[2]:size[3],size[0]:size[1]]

        ### do a little data transformation
        # take maximum to be within ~6 pixels of center
        d1, d2 = np.int(np.round(data.shape[0]/2.)), np.int(np.round(data.shape[1]/2.))
        vmax = data[(d1-3):(d1+3),(d2-3):(d2+3)].max()

        # get rid of background
        # sigclip is beautiful but slow!
        if sigclip:
            mean, median, std = sigma_clipped_stats(data, sigma=3.0,iters=20)
            if name[i] in noise_list:
                if vmax > (median+2.5*std):
                    data[data < (median+2.5*std)] = median+2.5*std
                elif vmax > (median+2*std):
                    data[data < (median+2*std)] = median+2*std
                else:
                    data[data < (median+1.5*std)] = median+1.5*std
            else:
                data[data < (median+std)] = median+std
        else:
            data[data < 0.01*vmax] = 0.01*vmax

        # log to emphasize outskirts
        data, vmax = np.log(data), np.log(vmax)

        ### create colormap
        # clip is so that things are at least a little red or blue
        # B = (0, 0, 1)
        # R = (1, 0, 0)
        if fmir[i] < fmir_mid: # blue, B = (0, 0, 1)
            blue = (fmir_mid-fmir[i]) / (fmir_mid-fmir_min)
            if (blue < 0) or (blue > 1):
                print 1/0
            colors = [(1,1,1),(0,0,blue)]
        else:
            red = (fmir[i]-fmir_mid) / (fmir_max-fmir_mid)
            if (red < 0) or (red > 1):
                print 1/0
            colors = [(1,1,1),(red,0,0)]
        cm = LinearSegmentedColormap.from_list('my_list', colors, N=50)
        ax.imshow(data, extent=[m[i]-delx/2., m[i]+delx/2., sfr[i]-dely/2., sfr[i]+dely/2.],cmap=cm,vmax=vmax)
        if name_label:
            ax.text(m[i],sfr[i]+dely/1.9, name[i],ha='center',fontsize=6)

    #### ADD SALIM+07
    salim_mass = np.linspace(7,12,40)
    ssfr_salim = -0.35*(salim_mass-10)-9.83
    salim_sfr = np.log10(10**ssfr_salim*10**salim_mass)

    ax.plot(salim_mass, salim_sfr,
            color='green',
            alpha=0.75,
            lw=1.5,
            zorder=1)
    ax.plot(salim_mass, salim_sfr-0.5,
            color='green',
            alpha=0.75,
            lw=1.5, linestyle='--',
            zorder=1)
    ax.plot(salim_mass, salim_sfr+0.5,
            color='green',
            alpha=0.75,
            lw=1.5, linestyle='--',
            zorder=1)

    ### labels and saving
    ax.set_xlabel(r'log(M$_*$) [M$_{\odot}$]')
    ax.set_ylabel(r'log(SFR) [M$_{\odot}$/yr]')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('auto')

    ### build a colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.07)

    colors = [(0, 0, 1), (0, 0, 0), (1, 0, 0)]
    cm = LinearSegmentedColormap.from_list('my_list', colors, N=200)
    norm = mpl.colors.Normalize(vmin=fmir_min, vmax=fmir_max)
    cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cm, norm=norm)
    cb1.set_label(r'log(f$_{\mathrm{AGN,MIR}}$)')

    ### save and eject
    plt.tight_layout()
    fig.savefig(outfolder+'sf_seq.png',dpi=220)
    os.system('open '+outfolder+'sf_seq.png')
    plt.close()

def img_extent(pix_center, px_scale, phot_size, im_shape):

    size = [np.clip(pix_center[0][0] - phot_size/px_scale,0,im_shape[1]-1),
            np.clip(pix_center[0][0] + phot_size/px_scale,0,im_shape[1]-1),
            np.clip(pix_center[0][1] - phot_size/px_scale,0,im_shape[0]-1),
            np.clip(pix_center[0][1] + phot_size/px_scale,0,im_shape[0]-1)
           ]

    return np.round(size).astype(int)

def load_structure(objname,long_axis=False):

    '''
    loads structure information from Brown+14 catalog
    '''

    loc = os.getenv('APPS')+'/prospector_alpha/data/brownseds_data/photometry/structure.dat'

    with open(loc, 'r') as f: hdr = f.readline().split()[1:]
    dat = np.loadtxt(loc, comments = '#', delimiter=' ',
                     dtype = {'names':([n for n in hdr]),\
                              'formats':('S40','S40','S40','S40','S40','S40','S40','S40','S40','S40')})

    match = dat['Name'] == objname.replace(' ','_')
    phot_size = np.array(dat['phot_size'][match][0].split('_'),dtype=np.float)

    if long_axis:
        phot_size = phot_size.max()

    return phot_size

def load_coordinates(objname):

    ra,dec,objnames = brown_io.load_coordinates()
    match = objname == objnames
    
    return ra[match], dec[match]



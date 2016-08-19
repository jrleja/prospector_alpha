import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
import numpy as np
import threed_dutils
from prospect.models import model_setup
import copy, math
from scipy.special import erf

#### constants
dpi = 75
c = 3e18   # angstroms per second

#### plot preferences
plt.ioff() # don't pop up a window for each plot
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
mpl.rcParams['font.sans-serif']='Geneva'

mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble']=[r"\usepackage{bm}"]
mpl.rcParams.update({'font.size': 30})
sfhsize = 40

#### parameter plot style
colors = ['blue', 'black', 'red']
alpha = 0.8
lw = 3

#### define model
param_file='/Users/joel/code/python/threedhst_bsfh/parameter_files/brownseds_np/brownseds_np_params.py'
run_params = model_setup.get_run_params(param_file=param_file)
sps = model_setup.load_sps(**run_params)
model = model_setup.load_model(**run_params)
obs = model_setup.load_obs(**run_params)

#### output locations
out1 = '/Users/joel/code/python/threedhst_bsfh/plots/brownseds_np/pcomp/model_diagram1.png'
out2 = '/Users/joel/code/python/threedhst_bsfh/plots/brownseds_np/pcomp/model_diagram2.png'

#### set initial theta
# 'logmass','sfr_fraction_1', sfr_fraction_2,
# sfr_fraction_3, sfr_fraction_4, 'sfr_fraction_5',
#  dust2', 'logzsol', 'dust_index', 
# 'dust1', 'duste_qpah', 'duste_gamma', 
# 'duste_umin'

labels = model.theta_labels()
itheta = np.array([10, 0.02, 0.1,
                   0.15, 0.2, 0.25, 
                   0.3, -1.0, 0.0, 
                   0.3, 3.0, 1e-1, 
                   10.0])
model.initial_theta = itheta

class jLogFormatter(mpl.ticker.LogFormatter):
	'''
	this changes the format from exponential to floating point.
	'''

	def __call__(self, x, pos=None):
		"""Return the format for tick val *x* at position *pos*"""
		vmin, vmax = self.axis.get_view_interval()
		d = abs(vmax - vmin)
		b = self._base
		if x == 0.0:
			return '0'
		sign = np.sign(x)
		# only label the decades
		fx = math.log(abs(x)) / math.log(b)
		isDecade = mpl.ticker.is_close_to_int(fx)
		if not isDecade and self.labelOnlyBase:
			s = ''
		elif x > 10000:
			s = '{0:.3g}'.format(x)
		elif x < 1:
			s = '{0:.3g}'.format(x)
		else:
			s = self.pprint_val(x, d)
		if sign == -1:
			s = '-%s' % s
		return self.fix_minus(s)

#### format those log plots! 
minorFormatter = jLogFormatter(base=10, labelOnlyBase=False)
majorFormatter = jLogFormatter(base=10, labelOnlyBase=True)

def make_ticklabels_invisible(ax,showx=False):
	if showx:
	    for tl in ax.get_yticklabels():tl.set_visible(False)
	else:
	    for tl in ax.get_xticklabels() + ax.get_yticklabels():
	        tl.set_visible(False)

def add_label(ax,label,par=None,par_idx=None,fmt=None,txtlabel=None,fontsize=None,secondary_text=None,init=None):

	tx = -0.65
	ty = 0.5
	dy = 0.1

	#### sandwich 'regular' spectra in between examples
	if par_idx is not None:
		par = [par[0],model.initial_theta[par_idx],par[1]]
	else:
		par = [par[0],init,par[1]]

	if secondary_text is None:
		ax.text(tx, ty, label, transform = ax.transAxes,ha='center',weight='bold',fontsize=60)
	else:
		ax.text(tx, ty+0.1, label, transform = ax.transAxes,ha='center',weight='bold',fontsize=60)
		ax.text(tx, ty-0.02, secondary_text, transform = ax.transAxes,ha='center',weight='bold',fontsize=40)

	#### text
	for ii,p in enumerate(par): ax.text(tx,ty-dy*(ii+1)-dy*0.3,txtlabel+'='+fmt.format(p),ha='center',weight='bold',transform=ax.transAxes,color=colors[ii])

	if init is not None:
		ax.text(tx,ty-dy*(ii+2)-dy*0.3,'not a free parameter',ha='center',weight='bold',transform=ax.transAxes,color=colors[ii])

def mass_xplot(ax,par,idx):

	##### sdss mass function
	##### log(M) log(phi) sigma_SV N [total, then repeat for star-forming, then quiescent]
	loc = '/Users/joel/IDLWorkspace82/ND_analytic/data/sdss-galex.txt'
	mf = np.loadtxt(loc)

	##### SDSS plot
	sdss_color = '0.6'
	ax.plot(mf[:,0],mf[:,1],color=sdss_color,lw=lw,alpha=alpha)
	ax.text(0.95,0.9,r'SDSS z$\sim$0.1',transform = ax.transAxes,color=sdss_color,ha='right')

	#### sandwich 'regular' spectra in between examples
	par = [par[0],model.initial_theta[idx],par[1]]

	##### parameter locations
	for ii,p in enumerate(par): ax.axvline(p,color=colors[ii],alpha=alpha,linewidth=lw)

	#### limits and labels
	ax.set_xlabel(r'log(M/M$_{\odot}$)')
	ax.set_ylabel(r'log(galaxy number density)')
	ax.xaxis.set_major_locator(MaxNLocator(4))

def met_xplot(ax,par,idx):

	#### sandwich 'regular' spectra in between examples
	par = [par[0],model.initial_theta[idx],par[1]]

	for ii, p in enumerate(par):
		met,mdf = met_triangular_kernel(p)
		#ax.plot(met,mdf,color=colors[ii],lw=lw,alpha=alpha)

		width=np.concatenate((met[1:] - met[:-1], np.atleast_1d(met[-1]-met[-2])))
		ax.bar(met,mdf,color=colors[ii],alpha=0.5,align='edge', width=width)


	#### limits and labels
	ax.set_xlabel(r'log(Z/Z$_{\odot}$)')
	ax.set_ylabel(r'weighting')
	ax.set_ylim(0.0,0.6)


def met_triangular_kernel(logzsol):

	'''
	FROM ZTINTERP.F90

		w1=0.25,w2=0.5,w3=0.25

	        IF (zpow.LT.0) THEN
           ! This smooths the SSPs in metallicity using a simple
           ! triangular kernel given by w1,w2,w3.  The smoothed SSPs
           ! are then interpolated to the target metallicity.  
           mdf = 0.

           # this step finds (the closest?) match to requested
           # metallicity in model metallicity. must be at least 1, at most nz

           zlo = MAX(MIN(locate(LOG10(zlegend/zsol),zpos),nz-1),1)
           dz  = (zpos-LOG10(zlegend(zlo)/zsol)) / &
                ( LOG10(zlegend(zlo+1)/zsol) - LOG10(zlegend(zlo)/zsol) )
           ! Here the weights for the mdf are a combination of the
           ! triangular kernel weights and the the linear interpolation
           ! weights.  If the kernel extends off the metallicity grid
           ! then the out-of-bounds weights are accumulated at the edges.
           mdf(MAX(zlo-1,1)) = w1*(1-dz)
           mdf(MIN(zlo+2,nz)) = w3*dz
           mdf(zlo)   = mdf(zlo) + w2*(1-dz) + w1*dz
           mdf(zlo+1) = mdf(zlo+1) + w3*(1-dz) + w2*dz
    '''

	#### available metallicities
	# in logzsol
	zsol = 0.019
	met = np.log10(sps.ssp.zlegend/zsol)

	#### define useful quantities
	nz = len(met)
	mdf = np.zeros_like(met)
	w = np.array([0.25, 0.5, 0.25])

	#### closest match
	idx = (np.abs(logzsol-met)).argmin()
	idx = np.where(met == np.min(met[(met - logzsol) > 0]))[0][0]-1

	#### calculate dz, fill in MDF
	dz = (logzsol - met[idx]) / (met[idx+1]-met[idx])
	mdf[np.clip(idx-1,0,nz-1)] = w[0]*(1-dz)
	mdf[np.clip(idx+2,0,nz-1)] = w[2]*dz
	mdf[idx]   = mdf[idx] + w[1]*(1-dz) + w[0]*dz
	mdf[idx+1] = mdf[idx+1] + w[2]*(1-dz) + w[1]*dz

	return met,mdf

def sfh_xplot(ax,par,par_idx,init=None):

	#### sandwich 'regular' spectra in between examples
	# if we're setting the implicit one...
	if par_idx is not None:
		par = [par[0],model.initial_theta[par_idx],par[1]]
	else:
		par = [par[0],init,par[1]]

	#### calculate time array
	in_years = 10**model.params['agebins']/1e9
	t = np.concatenate((np.ravel(in_years)*0.9999, np.ravel(in_years)*1.001))
	t.sort()
	t = t[1:-1] # remove older than oldest bin, younger than youngest bin
	t = np.clip(t,1e-3,np.inf) # nothing younger than 1 Myr!

	### calculate time per bin for weighting
	nbin = in_years.shape[0]
	time_per_bin = np.zeros(nbin)
	for i, (t1, t2) in enumerate(in_years): time_per_bin[i] = t2-t1
	time_sum = time_per_bin.sum()

	#### calculate and plot different SFHs
	theta = copy.copy(model.initial_theta)
	indices = [i for i, s in enumerate(labels) if ('sfr_fraction' in s) and (i != par_idx)]
	for ii,p in enumerate(par):

		theta[indices] = itheta[indices] * (1-p) / (1-par[1])
		if par_idx is not None:
			theta[par_idx] = p

		sfh_pars = threed_dutils.find_sfh_params(model,theta,obs,sps)
		sfh = threed_dutils.return_full_sfh(t, sfh_pars)

		ax.plot(t[::-1],np.log10(sfh[::-1]),color=colors[ii],lw=lw,alpha=alpha)

	#### limits and labels
	#ax.set_xlim(3.3,6.6)
	ax.set_ylim(-1.5,0.8)
	ax.set_xlabel(r'log(lookback time [Gyr])')
	ax.set_ylabel(r'log(SFR [M$_{\odot}$/yr])')
	ax.set_xlim(13,0.04)
	ax.set_xscale('log',nonposx='clip',subsx=[1])
	ax.xaxis.set_minor_formatter(minorFormatter)
	ax.xaxis.set_major_formatter(majorFormatter)

def attn_xplot(ax,par,par_idx,label):

	#### sandwich 'regular' spectra in between examples
	par = [par[0],model.initial_theta[par_idx],par[1]]

	#### plot details
	faint_color = '0.5'
	linestyle = '-'
	bc_linestyle = '--'

	#### retrieve dust index, attenuation values
	didx_diff = model.initial_theta[labels.index('dust_index')]
	didx_bc = -1.0	
	diff = model.initial_theta[labels.index('dust2')]
	bc = model.initial_theta[labels.index('dust1')]

	#### define wavelength, in log units
	lam=10**np.linspace(3,4.3,200)

	#### calculate and plot different attenuation curves
	theta = copy.copy(model.initial_theta)
	for ii,p in enumerate(par):
		if par_idx == 6: # if dust2
			diff_attn = threed_dutils.charlot_and_fall_extinction(lam,bc,p,didx_bc,didx_diff, kriek=True, nobc=True, nodiff=False)
			ax.plot(lam/1e4,-np.log(diff_attn),color=colors[ii],lw=lw,alpha=alpha,linestyle=linestyle)
		elif par_idx == 9: # if dust1
			bc_attn   = threed_dutils.charlot_and_fall_extinction(lam,p,diff,didx_bc,didx_diff, kriek=True, nobc=False, nodiff=True)
			ax.plot(lam/1e4,-np.log(bc_attn),color=colors[ii],lw=lw,alpha=alpha,linestyle=linestyle)
		elif par_idx == 8: # if dust_index
			diff_attn = threed_dutils.charlot_and_fall_extinction(lam,bc,diff,didx_bc,p, kriek=True, nobc=True, nodiff=False)
			ax.plot(lam/1e4,-np.log(diff_attn),color=colors[ii],lw=lw,alpha=alpha,linestyle=linestyle)

	#### limits and labels
	ax.set_xlim(10**-1,10**-0.2)
	ax.set_ylim(-0.1,6)
	ax.set_xlabel(r'wavelength [microns]')
	ax.set_ylabel(label + r' optical depth')

	#ax.set_xscale('log',nonposx='clip')

def dlogm_du(gamma, umin):

	'''
	eqn 23 in Draine & Li 2007

	The dust grains in a galaxy will be exposed to a wide range of
	starlight intensities. The bulk of the dust in the diffuse ISM will
	be heated by a general diffuse radiation field contributed by many
	stars. However, some dust grains will happen to be located in regions
	close to luminous stars, such as PDRs near OB stars, where
	the starlight heating the dust will be much more intense than the
	diffuse starlight illuminating the bulk of the grains.
	'''

	# "the starlight intensity distribution function"
	# recall that U is a dimensionless scaling factor 
	# which toggles the normalization of the energy density
	# per unit frequency which is shining on the dust
	# 
	# (1-gamma) is the fraction of dust mass exposed to starlight intensity umin
	# 'most' of the dust sees only the general diffuse radiation field
	# though gamma can go to 1 of course.
	# 
	# umin is the normalization of this diffuse radiation field

	# dlogM since we're setting mdust=1, effectively dividing by mdust, and dx / x = dlogx

	umax = 1e6
	alpha = 2.0
	mdust = 1.0

	u = 10**np.linspace(np.log10(umin),np.log10(umax), 200)

	# general expression
	dm_du = gamma*mdust*(alpha-1)/ (umin**(1-alpha) - umax**(1-alpha)) * u**(-alpha)

	# delta function
	dm_du[0] += (1-gamma)*mdust

	return u, dm_du

def dust_heating_xplot(ax, par, par_idx):

	#### sandwich 'regular' spectra in between examples
	par = [par[0],model.initial_theta[par_idx],par[1]]

	#### pull out initial thetas
	duste_gamma = model.initial_theta[labels.index('duste_gamma')]
	duste_umin = model.initial_theta[labels.index('duste_umin')]

	#### calculate and plot different starlight intensity distributions
	theta = copy.copy(model.initial_theta)
	for ii,p in enumerate(par):

		if par_idx == 11: # duste_gamma
			u,intensity_distr = dlogm_du(p,duste_umin)
			ax.plot(np.log10(u),np.log10(intensity_distr),color=colors[ii],lw=lw,alpha=alpha)
		elif par_idx == 12: # duste_umin
			u,intensity_distr = dlogm_du(duste_gamma,p)
			ax.plot(np.log10(u),np.log10(intensity_distr),color=colors[ii],lw=lw,alpha=alpha)
		else:
			print 1/0

	#### limits and labels
	ax.set_xlim(-0.1,6)
	#ax.set_ylim(-0.5,1.7)
	ax.set_xlabel(r'starlight intensity (U)')
	ax.set_ylabel(r'log(dlog(M$_{\mathbf{dust}}$)/dU)')

	#ax.set_xlim(ax.get_xlim()[1],ax.get_xlim()[0])

def mass_distr(qpah):

	'''
	eqn 11 in Draine & Li 2007 combined with eqn 4 in Weingartner & Draine 2001a
	volume distribution for carbonaceous grains
	(4/3)* pi * a^3 * dn / dln(a) in microns^3 / H-atom
	'''

	'''
	ftp://ftp.astro.princeton.edu/draine/dust/irem4/README.txt
	q_PAH is the fraction of the total dust mass that is contributed by
	PAH particles containing < 1000 C atoms.
	'''

	##### define size array in angstroms
	a = 10**np.linspace(-4,1,200)*1e4 # define in microns, convert to angstroms

	##### this is quantitatively wrong
	##### but it gives the correct qualitative shape
	##### to do it correctly, would integrate dn/da (M) to calculate qpah...
	# eqn taken from pg 819 of DL07
	if qpah == 0.0:
		b1_b2 = 0.0
	elif qpah == 3.0:
		b1_b2 = 0.92*60e-6*((5.-1)/6.)
	elif qpah == 6.0:
		b1_b2 = 0.92*60e-6*((10.-1)/6.)
	else:
		print 'go back to the original papers and find new b1+b2 --> qpah conversion'
		print 1/0

	# lognormal contributions
	# coefficients from Tbl 2 of Draine & Lee (2007)
	a0 = np.array([4.0, 20.0]) # Angstroms
	sigma = np.array([0.4,0.55])
	n0 = draine_li_n0(a0,sigma)*b1_b2

	ln = np.zeros(shape=(len(a),len(n0)))
	for ii in xrange(len(n0)): ln[:,ii] = n0[ii] / a * np.exp(-(np.log(a/a0[ii])**2)/(2*sigma[ii]**2))

	# nonlognormals from Equation 4 of WD01
	# these are most dust grains, and they tend to be much larger than PAHs
	nonlog = wd01_carbonaceous(a,qpah)

	#### SOLVE FOR PAH FRACTION HERE
	# we have dn/da (a), where a is the radius
	# q_PAH is the fraction of the total dust mass that is contributed by
	# PAH particles containing < 1000 C atoms.

	# total, convert to volume distribution
	# dy/dlogX = (dy/dX)'*X
	tot = np.sum(ln,axis=1) + nonlog
	tot = tot * (4*np.pi*a**3 / 3.) * a
	return a, tot


def draine_li_n0(a0,sigma):

	# return values of lognormal normalizations
	b = np.array([0.75,0.25]) # actually values of (b / [b1+b2]), must multiply by [b1+b2] in main eqn

	m_c = 1.994e-26 # mass of carbon atom in kg
	rho_c = 2.24e-3*1e-24 # kg / angstrom^3, density of graphite

	a_M = a0*np.exp(3*sigma**2)
	x = np.log(a_M / np.min(a0)) / (sigma*np.sqrt(2))

	out = 3. / ((2*np.pi) ** (3/2.))
	out *= np.exp(4.5*sigma**2) / (1. + erf(x))
	out *= m_c / (rho_c*a_M**3*sigma) * b

	return out

def wd01_carbonaceous(a,qpah):

	# a must be in angstroms!
	# return Eqn 4 from WD01, number density of carbonaceous particles
	# (1 / nh) dn / da = draine_li_no + ....
	# parameters taken from model 2 from table 1 in WD01
	
	# silicates
	#a_ts = 0.174 * 1e4 # angstroms
	#cs = 1.09e-12
	#beta_s = -10.3
	#alpha_s = -1.46

	# carbonaceous
	# pulled from Table 1, with the conversion between bc and qpah taken
	# from the text+figures in DL07 and tabulated in mass_distr
	if qpah == 0.0:
		c_g = 9.94e-11
		a_cg = 0.606e4 # angstroms
		a_tg = 0.00745e4 # angstroms
		beta_g = -0.0648
		alpha_g = -2.25
	elif qpah == 3.0:
		c_g = 4.15e-11
		a_cg = 0.499e4 # angstroms
		a_tg = 0.00837e4 # angstroms
		beta_g = -.125
		alpha_g = -1.91
	elif qpah == 6.0:
		c_g = 9.99e-12
		a_cg = 0.428e4 # angstroms
		a_tg = 0.0107e4 # angstroms
		beta_g = -0.165
		alpha_g = -1.54

	# F(a;Bg,atg)
	if beta_g >= 0:
		f = 1 + beta_g*a/a_tg
	else:
		f = (1 - beta_g*a/a_tg)**-1

	# piecewise function
	norm = np.zeros_like(a)
	ones = (a > 3.5) & (a < a_tg)
	norm[ones] = 1.0
	fnct = (a > a_tg)
	norm[fnct] = np.exp(-((a[fnct]-a_tg)/a_cg)**3)

	dn_da = (c_g / a) * (a/a_tg)**alpha_g * f * norm

	return dn_da # in angstroms


def qpah_xplot(ax, par, par_idx):

	'''
	reproduce Fig. 11 of Draine & Li (2007)
	volume distribution for carbonaceous grains
	(4/3)* pi * a^3 * dn / dln(a) in microns^3 / H-atom
	'''

	#### sandwich 'regular' spectra in between examples
	par = [par[0],model.initial_theta[par_idx],par[1]]

	#### calculate and plot different dust grain size distributions
	print par
	for ii,p in enumerate(par):
		a, tot = mass_distr(p)
		ax.plot(a,tot,lw=lw,alpha=alpha,color=colors[ii])

	ax.set_xscale('log',nonposx='clip', subsx=[1])
	ax.set_yscale('log',nonposx='clip')
	ax.set_xlim(2,1e4-1)
	ax.xaxis.set_minor_formatter(minorFormatter)
	ax.xaxis.set_major_formatter(majorFormatter)
	ax.set_ylim(1e-6,1e-2)
	ax.set_ylabel('dn/d(radius) $\times$ volume' '\n' '[carbon dust]')
	make_ticklabels_invisible(ax,showx=True)
	ax.set_xlabel(r'grain radius [$\AA$]')

def plot_sed(ax,ax_res,par_idx,par=None,txtlabel=None,fmt="{:.2e}",init=None):

	#### sandwich 'regular' spectra in between examples
	if par_idx is not None:
		par = [par[0],itheta[par_idx],par[1]]
	else:
		par = [par[0],init,par[1]]

	#### loop, plot first plot
	theta = copy.deepcopy(itheta)
	sfr_indices = [i for i, s in enumerate(labels) if 'sfr_fraction' in s]
	spec_sav = []

	for ii,p in enumerate(par):

		sps.ssp.params.dirtiness = 1
		# sfr fraction? 
		if (par_idx in sfr_indices) or (par_idx == None):
			theta[sfr_indices] = itheta[sfr_indices] * (1-p) / (1-par[1])
		if par_idx is not None:
			theta[par_idx] = p
		spec,mags,_ = model.mean_model(theta, obs, sps=sps)
		spec *= c/sps.wavelengths

		spec = threed_dutils.smooth_spectrum(sps.wavelengths,spec,200,minlam=3e3,maxlam=1e4)

		ax.plot(sps.wavelengths/1e4,np.log10(spec),color=colors[ii],lw=lw,alpha=alpha)
		spec_sav.append(spec)

	for ii,p in enumerate(par):
		ax_res.plot(sps.wavelengths/1e4,np.log10(spec_sav[ii]/spec_sav[1]),color=colors[ii],lw=lw,alpha=alpha)

	#### limits and labels [main plot]
	xlim = (10**-1.05,10**2.6)
	ylim = (6.5,9.3)
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	ax.set_xlabel(r'wavelength [microns]')
	ax.set_xscale('log',nonposx='clip',subsx=(1,2,5))
	ax.xaxis.set_minor_formatter(minorFormatter)
	ax.xaxis.set_major_formatter(majorFormatter)
	make_ticklabels_invisible(ax,showx=True)

	#### limits and labels [residual plot]
	ax_res.set_xlim(xlim)
	ax_res.set_ylim(-0.6,0.6)
	ax_res.set_xlabel(r'wavelength [microns]')
	ax_res.set_ylabel(r'relative change [dex]')
	ax_res.set_xscale('log',nonposx='clip',subsx=(1,2,5))
	ax_res.xaxis.set_minor_formatter(minorFormatter)
	ax_res.xaxis.set_major_formatter(majorFormatter)

def main_plot():

	#### I/O
	fig1_out = '/Users/joel/code/python/threedhst_bsfh/plots/brownseds/pcomp/diagram1.png'
	fig2_out = '/Users/joel/code/python/threedhst_bsfh/plots/brownseds/pcomp/diagram2.png'

	#### open figures
	fig1 = plt.figure(figsize = (37.5,37.5))

	gs1 = gridspec.GridSpec(7, 2)
	gs1.update(left=0.28, right=0.985, wspace=0.15, top=0.98,bottom=0.05)
	gs2 = gridspec.GridSpec(7, 1)
	gs2.update(left=0.14, right=0.265, top=0.98,bottom=0.05)

	#### PLOT 1 MASS
	idx = labels.index('logmass')
	par = [9.7,10.3]
	a1 = plt.subplot(gs2[0])
	add_label(a1,'stellar \n mass',par=par,par_idx=idx, txtlabel=r'log(M/M$_{\odot}$)', fmt="{:.1f}")
	mass_xplot(a1,par,idx)
	plot_sed(plt.subplot(gs1[0,0]),plt.subplot(gs1[0,1]),idx,
						 par=[9.7,10.3])

	#### PLOT 2 SFR1
	a3 = plt.subplot(gs2[1])
	idx = labels.index('sfr_fraction_1')
	par = [0.01,0.04]
	add_label(a3,r'SFH',par=par,par_idx=idx, txtlabel=r'fraction', fmt="{:.2f}",secondary_text='0-100 Myr')
	sfh_xplot(a3,par,idx)
	plot_sed(plt.subplot(gs1[1,0]),plt.subplot(gs1[1,1]),idx,
						    par=par)

	#### PLOT 3 SFR2
	a4 = plt.subplot(gs2[2])
	idx = labels.index('sfr_fraction_2')
	par = [0.05,0.2]
	add_label(a4,r'SFH',par=par,par_idx=idx, txtlabel=r'fraction', fmt="{:.2f}",secondary_text='100-300 Myr')
	sfh_xplot(a4,par,idx)
	plot_sed(plt.subplot(gs1[2,0]),plt.subplot(gs1[2,1]),idx,
						    par=par)

	#### PLOT 4 SFR3
	a5 = plt.subplot(gs2[3])
	idx = labels.index('sfr_fraction_3')
	par = [0.075,0.3]
	add_label(a5,r'SFH',par=par,par_idx=idx, txtlabel=r'fraction', fmt="{:.2f}",secondary_text='300 Myr-1 Gyr')
	sfh_xplot(a5,par,idx)
	plot_sed(plt.subplot(gs1[3,0]),plt.subplot(gs1[3,1]),idx,
						 par=par)

	#### PLOT 5 SFR4
	a6 = plt.subplot(gs2[4])
	idx = labels.index('sfr_fraction_4')
	par = [0.1,0.4]
	add_label(a6,r'SFH',par=par,par_idx=idx, txtlabel=r'fraction', fmt="{:.2f}",secondary_text='1-3 Gyr')
	sfh_xplot(a6,par,idx)
	plot_sed(plt.subplot(gs1[4,0]),plt.subplot(gs1[4,1]), idx,
						 par=par)

	#### PLOT 6 SFR5
	a6 = plt.subplot(gs2[5])
	idx = labels.index('sfr_fraction_5')
	par = [0.125,0.5]
	add_label(a6,r'SFH',par=par, par_idx=idx, txtlabel=r'fraction', fmt="{:.2f}",secondary_text='3-6 Gyr')
	sfh_xplot(a6,par,idx)
	plot_sed(plt.subplot(gs1[5,0]),plt.subplot(gs1[5,1]), idx,
						 par=par)

	#### PLOT 7 SFR6
	a7 = plt.subplot(gs2[6])
	temp_init = 0.28
	par = [0.14,0.56]
	add_label(a7,r'SFH',par=par,par_idx=None, txtlabel=r'fraction', fmt="{:.2f}",secondary_text='6-13.6 Gyr',init=temp_init)
	sfh_xplot(a7,par,None,init=temp_init)
	plot_sed(plt.subplot(gs1[6,0]),plt.subplot(gs1[6,1]), idx,
						 par=par,init=temp_init)


	plt.savefig(out1,dpi=dpi)
	plt.close()

	#### NEW PLOT
	fig2 = plt.figure(figsize = (37.5,43.75))

	gs1 = gridspec.GridSpec(7, 2)
	gs1.update(left=0.28, right=0.985, wspace=0.15, top=0.98,bottom=0.05)
	gs2 = gridspec.GridSpec(7, 1)
	gs2.update(left=0.14, right=0.265, top=0.98,bottom=0.05)

	#### PLOT 7 METALLICITY
	idx = labels.index('logzsol')
	par = [-1.8,0.1]
	a2 = plt.subplot(gs2[0])
	add_label(a2,'stellar \n metallicity',par=par,par_idx=idx, txtlabel=r'log(Z/Z$_{\odot}$)', fmt="{:.2f}")
	met_xplot(a2,par,idx)
	plot_sed(plt.subplot(gs1[0,0]),plt.subplot(gs1[0,1]),idx,
						 par=[-1.8,0.1])

	#### PLOT 8 DUST1
	a7 = plt.subplot(gs2[1])
	idx = labels.index('dust1')
	par = [0.0,1.0]
	add_label(a7,r'birth cloud' '\n' r'dust',par=par,par_idx=idx, txtlabel=r'$\tau_{\mathbf{bc}}$', fmt="{:.1f}")
	attn_xplot(a7,par,idx, 'birth cloud')
	plot_sed(plt.subplot(gs1[1,0]),plt.subplot(gs1[1,1]), idx,
						 par=par)

	#### PLOT 9 DUST2
	a8 = plt.subplot(gs2[2])
	idx = labels.index('dust2')
	par = [0.0,1.0]
	add_label(a8,r'diffuse' '\n' r'dust',par=par,par_idx=idx, txtlabel=r'$\tau_{\mathbf{diff}}$', fmt="{:.1f}")
	attn_xplot(a8,par,idx, 'diffuse')
	plot_sed(plt.subplot(gs1[2,0]),plt.subplot(gs1[2,1]), idx,
						 par=par)

	#### PLOT 10 DUST2_INDEX
	a9 = plt.subplot(gs2[3])
	idx = labels.index('dust_index')
	par = [-0.3,0.3]
	add_label(a9,r'diffuse' '\n' r'dust index',par=par,par_idx=idx, txtlabel=r'$n_{\mathbf{diff}}$', fmt="{:.1f}")
	attn_xplot(a9,par,idx, 'diffuse')
	plot_sed(plt.subplot(gs1[3,0]),plt.subplot(gs1[3,1]), idx,
						 par=par)

	#### PLOT 11 DUSTE_GAMMA
	a10 = plt.subplot(gs2[4])
	idx = labels.index('duste_gamma')
	par = [0.01,0.5]
	add_label(a10,r'dust' '\n' 'emission' '\n' r'$\bm{\gamma}$',par=par,par_idx=idx, txtlabel=r'$\gamma$', fmt="{:.1f}")
	dust_heating_xplot(a10,par,idx)
	plot_sed(plt.subplot(gs1[4,0]),plt.subplot(gs1[4,1]), idx,
						 par=par)

	#### PLOT 11 DUSTE_UMIN
	a11 = plt.subplot(gs2[5])
	idx = labels.index('duste_umin')
	par = [1,20]
	add_label(a11,r'dust' '\n' 'emission' '\n' r'U$_{\mathbf{min}}$',par=par,par_idx=idx, txtlabel=r'U$_{\mathbf{min}}$', fmt="{:.1f}")
	dust_heating_xplot(a11,par,idx)
	plot_sed(plt.subplot(gs1[5,0]),plt.subplot(gs1[5,1]), idx,
						 par=par)

	#### PLOT 12 DUSTE_QPAH
	a12 = plt.subplot(gs2[6])
	idx = labels.index('duste_qpah')
	par = [0.0,6.0]
	add_label(a12,r'dust' '\n' 'emission' '\n' r'q$_{\mathbf{PAH}}$',par=par,par_idx=idx, txtlabel=r'q$_{\mathbf{PAH}}$', fmt="{:.1f}")
	qpah_xplot(a12,par,idx)
	plot_sed(plt.subplot(gs1[6,0]),plt.subplot(gs1[6,1]), idx,
						 par=par)

	plt.savefig(out2,dpi=dpi)
	plt.close()

	print 1/0



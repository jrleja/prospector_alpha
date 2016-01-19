import numpy as np
import threed_dutils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import brown_io
import matplotlib.gridspec as gridspec

plt.ioff()

def make_plot(runname='brownseds_tightbc'):

	#### load alldata
	alldata = brown_io.load_alldata(runname=runname)

	#### create figure, plus metrics
	dpi = 135
	scale = 1.4
	figsize = (8.5*scale,11*scale)
	fig = plt.figure(figsize = figsize)
	fontsize=8
	xcurr = -0.05
	xmax = 0.9

	ycurr = 0.9
	ytrans = 0.6 # where we transition from half to full

	dy = 0.1
	dx = 0.1

	ysep = 0.06

	fig = add_mass_sfr_plot(alldata,fig)

	#### loop over alldata, add each galaxy png
	xstart = xcurr + dx
	for dat in alldata:

		#### determine location of png
		xcurr += dx
		if (xcurr > xmax) or ((ycurr > ytrans) & (xcurr > 0.5)):
			xcurr = xstart
			ycurr = ycurr - ysep

		#### if we're off the page, bail!
		if ycurr < -ysep:
			break

		#### load png
		imgname=os.getenv('APPS')+'/threedhst_bsfh/data/brownseds_data/rgb/'+dat['objname'].replace(' ','_')+'.png'
		img=mpimg.imread(imgname)

		#### background
		size = 0.01
		ax_bg = fig.add_axes([xcurr+0.0007,ycurr+0.0145,dx-0.0007,size])
		ax_bg.set_axis_bgcolor('0.83')
		ax_bg.get_xaxis().set_visible(False)
		ax_bg.get_yaxis().set_visible(False)

		#### plot png
		ax = fig.add_axes([xcurr,ycurr,dx,dy])
		ax.imshow(img)
		ax.set_axis_off()

		#### text
		ax.text(0.5,-0.15,dat['objname'],fontsize=fontsize,ha='center',transform=ax.transAxes)

	plt.savefig('/Users/joel/my_papers/prospector_brown/figures/introduce_sample.png',dpi=dpi)
	plt.close()

def add_mass_sfr_plot(alldata,fig):

	minsfr = 1e-4

	##### find prospector indexes
	parnames = alldata[0]['pquantiles']['parnames']
	idx_mass = parnames == 'mass'
	eparnames = alldata[0]['pextras']['parnames']
	idx_sfr = eparnames == 'sfr_100'

	mass = np.zeros(shape=(len(alldata),3))
	sfr = np.zeros(shape=(len(alldata),3))

	##### add axis
	ax = fig.add_axes([0.61,0.72,0.35,0.25])

	for ii,dat in enumerate(alldata):
		mass[ii,0] = np.log10(dat['pquantiles']['q50'][idx_mass])
		mass[ii,1] = np.log10(dat['pquantiles']['q84'][idx_mass])
		mass[ii,2] = np.log10(dat['pquantiles']['q16'][idx_mass])

		sfr[ii,0] = np.clip(dat['pextras']['q50'][idx_sfr],minsfr,np.inf)
		sfr[ii,1] = np.clip(dat['pextras']['q84'][idx_sfr],minsfr,np.inf)
		sfr[ii,2] = np.clip(dat['pextras']['q16'][idx_sfr],minsfr,np.inf)

	errs_mass = threed_dutils.asym_errors(mass[:,0],mass[:,1],mass[:,2],log=False)
	errs_sfr  = threed_dutils.asym_errors(mass[:,0],mass[:,1],mass[:,2],log=True)

	ax.errorbar(mass[:,1],np.log10(sfr[:,1]),
		        fmt='o', alpha=0.6,
		        color='0.5',
			    xerr=errs_mass, yerr=errs_sfr,
			    ms=6.0)

	#### ADD SALIM+07
	salim_mass = np.linspace(7,12,40)
	ssfr_salim = -0.35*(salim_mass-10)-9.83
	salim_sfr = np.log10(10**ssfr_salim*10**salim_mass)

	ax.plot(salim_mass, salim_sfr,
	          color='green',
	          alpha=0.8,
	          lw=2.5,
	          label='Salim+07',
	          zorder=-1)
	ax.fill_between(salim_mass, salim_sfr-0.5, salim_sfr+0.5, 
	                  color='green',
	                  alpha=0.3)

	ax.set_xlabel(r'log(M/M$_{\odot}$)')
	ax.set_ylabel(r'log(SFR/M$_{\odot}$/yr)')

	ax.set_ylim(-4.5,2)

	ax.text(0.05,0.05,'Salim+07',transform=ax.transAxes,color='green',fontsize=13)

	return fig




















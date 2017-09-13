import numpy as np
import prosp_dutils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import prospector_io
import matplotlib.gridspec as gridspec

plt.ioff()

def make_plot(runname='brownseds_np'):

	#### load alldata
	alldata = prospector_io.load_alldata(runname=runname)

	#### create figure, plus metrics
	dpi = 135
	scale = 1.4
	figsize = (8.5*scale,11*scale)
	fig = plt.figure(figsize = figsize)
	fontsize=18.5

	dy = 0.249
	dx = 0.249

	xcurr = -dx+0.001
	xmax = 1.0-dx # farthest right you can place an image, after transition

	ycurr = 0.8
	ytrans = 0.6 # where we transition from half to full
	xtrans = 0.4 # farthest right position to place an image above half-page

	ysep = 0.16 # distance between images

	fig = add_mass_sfr_plot(alldata,fig)

	### galaxies we want
	# limit 20
	in_list = ['NGC 0337', 'NGC 0474', 'NGC 0660', 'NGC 7331', 'NGC 1275', 'NGC 2403', 'UGC 06850', 'NGC 2798', \
	           'NGC 3079', 'NGC 3198', 'NGC 3190', 'NGC 3310', 'NGC 3351', 'Arp 256 N', 'NGC 3521', 'NGC 3627', \
	           'NGC 3938', 'NGC 4594', 'NGC 4168', 'NGC 4254']

	#### loop over alldata, add each galaxy png
	xstart = xcurr + dx
	for name in in_list:

		#### determine location of png
		xcurr += dx
		if (xcurr > xmax) or ((ycurr > ytrans) & (xcurr > xtrans)):
			xcurr = xstart
			ycurr = ycurr - ysep

		#### if we're off the page, bail!
		if ycurr < -0.05:
			break

		#### load png
		imgname=os.getenv('APPS')+'/prospector_alpha/data/brownseds_data/rgb/'+name.replace(' ','_')+'.png'
		img=mpimg.imread(imgname)

		#### background
		size = 0.04
		ax_bg = fig.add_axes([xcurr+0.0007,ycurr+0.021,dx-0.0007,size])
		ax_bg.set_axis_bgcolor('0.87')
		ax_bg.get_xaxis().set_visible(False)
		ax_bg.get_yaxis().set_visible(False)

		#### plot png
		ax = fig.add_axes([xcurr,ycurr,dx,dy])
		ax.imshow(img)
		ax.set_axis_off()

		#### text
		ax.text(0.5,-0.18,name,fontsize=fontsize,ha='center',transform=ax.transAxes,weight='semibold')

	plt.savefig('/Users/joel/my_papers/prospector_brown/figures/introduce_sample.png',dpi=dpi)
	plt.close()

def add_mass_sfr_plot(alldata,fig):

	minsfr = 1e-4

	##### find prospector indexes
	eparnames = alldata[0]['pextras']['parnames']
	idx_mass = eparnames == 'stellar_mass'
	idx_sfr = eparnames == 'sfr_100'

	mass = np.zeros(shape=(len(alldata),3))
	sfr = np.zeros(shape=(len(alldata),3))

	##### add axis
	ax = fig.add_axes([0.58,0.71,0.37,0.28])

	for ii,dat in enumerate(alldata):
		mass[ii,0] = np.log10(dat['pextras']['q50'][idx_mass])
		mass[ii,1] = np.log10(dat['pextras']['q84'][idx_mass])
		mass[ii,2] = np.log10(dat['pextras']['q16'][idx_mass])

		sfr[ii,0] = np.clip(dat['pextras']['q50'][idx_sfr],minsfr,np.inf)
		sfr[ii,1] = np.clip(dat['pextras']['q84'][idx_sfr],minsfr,np.inf)
		sfr[ii,2] = np.clip(dat['pextras']['q16'][idx_sfr],minsfr,np.inf)

	errs_mass = prosp_dutils.asym_errors(mass[:,0],mass[:,1],mass[:,2],log=False)
	errs_sfr  = prosp_dutils.asym_errors(sfr[:,0],sfr[:,1],sfr[:,2],log=True)

	ax.errorbar(mass[:,0],np.log10(sfr[:,0]),
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

	ax.legend(loc=3, prop={'size':12},
				     frameon=False)

	return fig




















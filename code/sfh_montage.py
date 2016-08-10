import matplotlib.pyplot as plt
import numpy as np
import brown_io
import os
import magphys_plot_pref
import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import math
import matplotlib.lines as lines

plt.ioff() # don't pop up a window for each plot
mpl.rcParams['xtick.major.size'] = 2
mpl.rcParams['ytick.major.size'] = 2
mpl.rcParams['xtick.major.width'] = 0.5
mpl.rcParams['ytick.major.width'] = 0.5
mpl.rcParams['xtick.labelsize'] = 8 # font size of xtick labels
mpl.rcParams['ytick.labelsize'] = 8 # font size of ytick labels
mpl.rcParams['axes.labelsize'] = 12 # font size of x+y labels
mpl.rcParams['axes.labelpad'] = 2.5 # space between label and axis

class jLogFormatter(ticker.LogFormatter):
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
		isDecade = ticker.is_close_to_int(fx)
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

def create_formatted_figure():

	### overall figure geometry
	nfig = 6
	figsize = (37.5,37.5)
	figsize = (25,12.5)
	fig = plt.figure(figsize = figsize)

	### plot spacing
	wspace_between_columns = 0.00
	hspace = 0.0
	left = 0.02
	right = 0.98
	margin = left + (1.-right)
	figsize = (1.-margin-wspace_between_columns*nfig)/nfig

	bottom = 0.06
	top = 0.98

	### create arrays
	axarr1 = np.empty(shape=(nfig,nfig),dtype=object)
	axarr2 = np.empty(shape=(nfig,nfig),dtype=object)
	for i in xrange(nfig): # over columns
		figstart = left+i*(figsize+wspace_between_columns)
		gs = gridspec.GridSpec(nfig, 2)
		gs.update(hspace=hspace,left=figstart,right=figstart+figsize,wspace=0.00,bottom=bottom,top=top)
		for j in xrange(nfig): # over rows
			axarr1[j,i] = plt.subplot(gs[j,0])
			axarr2[j,i] = plt.subplot(gs[j,1])

	return fig, np.ravel(axarr1), np.ravel(axarr2)

def add_png(ax,objname,swapaxes=False):

	# stretch axis
	# [left, bottom, width, height]
	ax.set_position([ax.get_position().x0-0.014,
		             ax.get_position().y0-0.002,
		             ax.get_position().width+0.028,
		             ax.get_position().height+0.004])
	# add RGB
	try:
		imgname=os.getenv('APPS')+'/threedhst_bsfh/data/brownseds_data/rgb/'+objname.replace(' ','_')+'.png'
		img=mpimg.imread(imgname)
		
		if swapaxes:
			img = np.swapaxes(img,0,1)
		else:
			img = img[:,115:565]
		ax.imshow(img,zorder=32)
		ax.set_axis_off()
	except IOError:
		print 'no RGB image for ' + objname

def plot_sfh(ax,t,sfh,sfrmin):

	#### colors and preferences
	median_err_color = '0.75'
	median_main_color = 'black'

	#### analyze marginalized SFH
	# clip to minimum
	perc = np.zeros(shape=(len(t),3))
	for jj in xrange(len(t)): perc[jj,:] = np.percentile(sfh[jj,:],[16.0,50.0,84.0])
	perc = np.log10(np.clip(perc,sfrmin,np.inf))

	### plot whole SFH
	ax.plot(t, perc[:,1],'-',color=median_main_color)
	ax.fill_between(t, perc[:,0], perc[:,2], color=median_err_color)

	### set up y-plotting range
	plotmax_y = np.max(perc)
	plotmin_y = np.min(perc)
	ax.set_ylim(plotmin_y-0.1, plotmax_y+0.1)

	### format x-axis
	ax.set_xlabel('t [Gyr]',weight='bold')

	ax.set_xscale('log',nonposx='clip',subsx=[1])
	ax.xaxis.set_minor_formatter(minorFormatter)
	ax.xaxis.set_major_formatter(majorFormatter)

	ax.set_xlim(13.6,0.01)

	### turn off y-axis
	for tl in ax.get_yticklabels():tl.set_visible(False)

	### add 0.3 dex line
	ylim = ax.get_ylim()[0]+0.2
	lw = 1.3
	line = lines.Line2D([0.03,0.03],[ylim,ylim+0.3],solid_capstyle='projecting',alpha=0.8,lw=lw,color='black')
	ax.add_line(line)
	line = lines.Line2D([0.027,0.033],[ylim+0.3,ylim+0.3],solid_capstyle='projecting',alpha=0.8,lw=lw,color='black')
	ax.add_line(line)
	line = lines.Line2D([0.027,0.033],[ylim,ylim],solid_capstyle='projecting',alpha=0.8,lw=lw,color='black')
	ax.add_line(line)

def main_plot(runname='brownseds_np'):

	### output folder
	outfolder = os.getenv('APPS')+'/threedhst_bsfh/plots/'+runname+'/pcomp/'

	### load data
	alldata = brown_io.load_alldata(runname=runname)

	### define t_half
	thalf_idx = alldata[0]['pextras']['parnames'] == 'half_time'
	thalf = np.array([f['pextras']['q50'][thalf_idx][0] for f in alldata])
	sorted_indices = thalf.argsort()

	### figure
	fig, ax1, ax2 = create_formatted_figure()

	### let's do this
	npage = 1
	for i, idx in enumerate(sorted_indices):

		# do we need a new page?
		if i >= npage*ax1.shape[0]:
			outname = outfolder+'sfh_montage_'+str(npage)+'.pdf'
			print 'saving as '+outname
			fig.savefig(outname)
			plt.close()
			fig, ax1, ax2 = create_formatted_figure()
			npage+=1

		### calculate minSFR as 1/10,000th of the average SFR over 10 Gyr
		sfrmin = alldata[idx]['pextras']['q50'][alldata[idx]['pextras']['parnames'] == 'totmass'] / 1e10 / 1e5

		### plots
		axidx = i-(npage-1)*ax1.shape[0]
		plot_sfh(ax1[axidx],alldata[idx]['pextras']['t_sfh'],alldata[idx]['pextras']['sfh'],sfrmin)
		add_png(ax2[axidx],alldata[idx]['objname'])

		### turn on y-axis label if necessary
		if i % 6 == 0:
			ax1[axidx].set_ylabel('log(SFR)',weight='bold',zorder=-1)

		### half-time label
		if npage > 2:
			xtext, ha = 0.97, 'right'
		else:
			xtext, ha = 0.05, 'left'
		ax1[axidx].text(xtext,0.9,r't$_{\mathbf{half}}$ = '+"{:.2f}".format(thalf[idx])+' Gyr',
			            ha=ha,fontsize=10,weight='bold',transform=ax1[axidx].transAxes)
		ax1[axidx].text(xtext,0.8,alldata[idx]['objname'],
			            ha=ha,fontsize=10,weight='bold',transform=ax1[axidx].transAxes)

	# turn the remaining axes off
	for i in xrange(axidx+1,ax1.shape[0]):
		ax1[i].axis('off')
		ax2[i].axis('off')
	outname = outfolder+'sfh_montage_'+str(npage)+'.pdf'
	print 'saving as '+outname
	fig.savefig(outname)
	plt.close()




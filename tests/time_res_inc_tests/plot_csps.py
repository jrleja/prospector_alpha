import numpy as np
import fsps, pickle
import matplotlib.pyplot as plt

with open('time_res_incr=2.pickle', "rb") as f:
	timeres2=pickle.load(f)

with open('time_res_incr=8.pickle', "rb") as f:
	timeres8=pickle.load(f)

# create figure
fig = plt.figure()
ax = fig.add_axes([0.1,0.55,0.8,0.42])
colors = ['#76FF7A', '#1CD3A2', '#1974D2', '#7442C8', '#FC2847', '#FDFC74', '#8E4585', '#FF1DCE']
for ii in xrange(len(timeres8['tage'])):

	# plot
	ratio = (timeres8['outspec'][ii]/np.array(timeres8['sm'][ii])) / (timeres2['outspec'][ii]/np.array(timeres2['sm'][ii]))
	ax.plot(np.log10(timeres8['wavelength']), np.log10(ratio), 
		    alpha=0.5,linestyle='-',label=timeres8['tage'][ii],color=colors[ii],linewidth=2)

plt.axhline(0.0,linestyle='--',color='black')

ax.set_xlabel(r'log($\lambda [\AA]$)')
ax.set_ylabel('log(f_8 / f_2)')
ax.legend(loc=1,prop={'size':10},title='tage')
ax.axis((2.4,8,-0.1,0.1))

# plot stellar mass
ax = fig.add_axes([0.1,0.1,0.35,0.35])
ratio = np.array(timeres8['sm']) / np.array(timeres2['sm'])
ax.plot(timeres8['tage'], ratio, 'bo',
	    alpha=0.5,linestyle='-',linewidth=2)

plt.axhline(1.0,linestyle='--',color='black')

ax.set_xlabel('tage')
ax.set_ylabel('mass (8) / mass (2)')
ax.axis((0,14,0.99,1.01))

# plot lbol
try:
	ax = fig.add_axes([0.6,0.1,0.35,0.35])
	ratio = 10**np.array(timeres8['lbol']) / 10**np.array(timeres2['lbol'])
	ax.plot(timeres8['tage'], ratio, 'bo',
		    alpha=0.5,linestyle='-',linewidth=2)

	plt.axhline(1.0,linestyle='--',color='black')

	ax.set_xlabel('tage')
	ax.set_ylabel('lbol (8) / lbol (2)')
	ax.axis((0,14,0.99,1.01))
except:
	pass

plt.savefig('time_res_incr_test_8_2.png',dpi=300)
plt.close()

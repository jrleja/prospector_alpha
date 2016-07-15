import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import pickle

outname1 = 'old_brownseds.pickle'
outname2 = 'new_brownseds.pickle'
outplot = 'spectral_ratio.png'

with open(outname1, "rb") as f:
    old=pickle.load(f)

with open(outname2, "rb") as f:
    new=pickle.load(f)

#### check parameter consistency
for k, v in old['param_dict'].iteritems():
	if k in new['param_dict']:
		if new['param_dict'][k] == v:
			pass
		else:
			print k+' OLD:'+str(v)+ ' NEW:'+str(new['param_dict'][k])
		new['param_dict'].pop(k)
	else:
		print k + '='+str(v)+' is not in the new sps'

for k, v in new['param_dict'].iteritems():
	print k + '='+str(v)+' is not in the old sps'

# set up plot
fig, ax = plt.subplots()
colors = ['#1974D2', '#FF1DCE']

fig, ax = plt.subplots()
ax.set_xlabel(r'log($\lambda$) [$\AA$]')
ax.set_ylabel(r'old spectrum/new spectrum')
ax.plot(np.log10(new['w']),old['spec']/new['spec'],alpha=0.5,color=colors[0])

print 'average deviation: '+ str(np.mean(old['spec']/new['spec']))

for i, f in enumerate(old['filt']):
	if new['filt'][i].name != f:
		print 'NEW: ' + new['filt'][i].name + ' OLD: '+f+' OLD/NEW: ' + "{:.2f}".format(old['phot'][i]/(new['phot'][i]))


#print old['phot']/new['phot']

plt.savefig(outplot,dpi=300)
plt.close()
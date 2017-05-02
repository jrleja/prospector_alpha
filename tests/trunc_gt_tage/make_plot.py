import numpy as np
import fsps, pickle, prosp_dutils, os
from bsfh import model_setup
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import pickle

outname1 = 'before_fix.pickle'
outname2 = 'after_fix.pickle'
outplot = 'before_after_comp.png'

with open(outname1, "rb") as f:
    before=pickle.load(f)

with open(outname2, "rb") as f:
    after=pickle.load(f)


# set up plot
fig, ax = plt.subplots()
colors = ['#1974D2', '#7442C8']

fig, ax = plt.subplots()
ax.set_xlabel(r'sftrunc/tage')
ax.set_ylabel(r'GALEX mag')
ax.plot(before['sf_trunc'],before['galex'],alpha=0.5,color=colors[0],label='Before fix')
ax.plot(after['sf_trunc'],after['galex'],alpha=0.5,color=colors[1],label='After fix')
ax.legend(loc=1,prop={'size':10})

plt.savefig(outplot,dpi=300)

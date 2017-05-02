import numpy as np
import fsps, pickle, prosp_dutils, os
from bsfh import model_setup
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import pickle

outname1 = 'tres=2.pickle'
outname2 = 'tres=10.pickle'
outplot = 'tres=2_tres=10_comp.png'

with open(outname1, "rb") as f:
    before=pickle.load(f)

with open(outname2, "rb") as f:
    after=pickle.load(f)


# set up plot
fig, ax = plt.subplots()
colors = ['#1974D2', '#FF1DCE']

fig, ax = plt.subplots()
ax.set_xlabel(r'tage')
ax.set_ylabel(r'GALEX mag')
ax.plot(before['tcalc'],before['galex'],alpha=0.5,color=colors[0],label='tres=2')
ax.plot(after['tcalc'],after['galex'],alpha=0.5,color=colors[1],label='tres=10')
ax.legend(loc=1,prop={'size':10})

plt.savefig(outplot,dpi=300)

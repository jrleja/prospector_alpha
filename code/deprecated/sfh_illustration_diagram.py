import numpy as np
import threed_dutils
from bsfh import model_setup
import os
import matplotlib.pyplot as plt
import magphys_plot_pref
import copy
import matplotlib as mpl

mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble']=[r"\usepackage{bm}"]
mpl.rcParams.update({'font.size': 18})

param_file = os.getenv('APPS') + '/threedhst_bsfh/parameter_files/brownseds_tightbc/brownseds_tightbc_params_1.py'

model = model_setup.load_model(param_file)
model.params['zred'] = np.atleast_1d(0.0)
obs   = model_setup.load_obs(param_file)
obs['filters'] = None # don't generate photometry, for speed
sps = threed_dutils.setup_sps()
parnames = np.array(model.theta_labels())

def generate_thetas():

	theta1 = model.initial_theta
	theta1[parnames=='sf_tanslope'] = -np.pi/3.2
	theta1[parnames=='logtau'] = 0.4
	theta1[parnames=='tage'] = 9
	theta1[parnames=='delt_trunc'] = 0.85

	theta2 = copy.copy(model.initial_theta)
	theta2[parnames=='sf_tanslope'] = 0.05
	theta2[parnames=='logtau'] = 0.4
	theta2[parnames=='tage'] = 10
	theta2[parnames=='delt_trunc'] = 0.6
	theta2[parnames=='mass'] = 2.2e10

	theta3 = copy.copy(model.initial_theta)
	theta3[parnames=='sf_tanslope'] = 0.05
	theta3[parnames=='logtau'] = 0.2
	theta3[parnames=='tage'] = 3
	theta3[parnames=='delt_trunc'] = 1.0
	theta3[parnames=='mass'] = 5e9

	return theta1, theta2, theta3

def main():
	'''
	script to generate, illustrate, and annotate several representative SFHs
	'''

	##### plotparameters
	alpha = 0.7
	lw = 3.0
	color1 = '#1C86EE' #blue
	color2 = '#FF3D0D' #red 
	color3 = '#4CBB17' #green
	color1_text = color1
	color2_text = color2
	color3_text = color3
	arrowcolor='0.3'
	fontsize=20
	arrowprops1=dict(arrowstyle='<|-|>',fc=color1,ec=color1,lw=3,alpha=alpha)
	arrowprops2=dict(arrowstyle='<|-|>',fc=color2,ec=color2,lw=3,alpha=alpha)
	arrowprops3=dict(arrowstyle='<|-|>',fc=color3,ec=color3,lw=3,alpha=alpha)

	##### generate thetas
	theta1,theta2,theta3 = generate_thetas()

	##### calculate SFR(t)
	t = np.linspace(0,10.0,num=400)
	sfhpars1 = threed_dutils.find_sfh_params(model,theta1,obs,sps)
	sfhpars2 = threed_dutils.find_sfh_params(model,theta2,obs,sps)
	sfhpars3 = threed_dutils.find_sfh_params(model,theta3,obs,sps)

	sfh1 = threed_dutils.return_full_sfh(t, sfhpars1)
	sfh2 = threed_dutils.return_full_sfh(t, sfhpars2)
	sfh3 = threed_dutils.return_full_sfh(t, sfhpars3)

	##### plot SFR(t)
	fig, ax = plt.subplots(1, 1, figsize = (8,8))

	ax.plot(t, sfh1, lw=lw, color=color1, alpha=alpha)
	ax.plot(t, sfh2, lw=lw, color=color2, alpha=alpha)
	ax.plot(t, sfh3, lw=lw, color=color3, alpha=alpha)

	yarr = 2.4
	xarr = 7.8
	length = 3.2
	ax.annotate(s='', xy=(xarr,yarr), xytext=(xarr-length,yarr), size=20, arrowprops=arrowprops1)
	ax.text(np.mean((xarr,xarr-length)), yarr-0.2,r'$\bm{\tau}$',fontsize=fontsize,color=color1_text,ha='center')

	yarr = 4.25
	xarr = 8.5
	length = 2.5
	ax.annotate(s='', xy=(xarr,yarr), xytext=(xarr-length,yarr), size=20, arrowprops=arrowprops2)
	ax.text(np.mean((xarr,xarr-length)), yarr-0.2,r'$\bm{\tau}$',fontsize=fontsize,color=color2_text,ha='center')

	yarr = 2.7
	xarr = 2.1
	length = 1.7
	ax.annotate(s='', xy=(xarr,yarr), xytext=(xarr-length,yarr), size=20, arrowprops=arrowprops3)
	ax.text(np.mean((xarr,xarr-length)), yarr-0.2,r'$\bm{\tau}$',fontsize=fontsize,color=color3_text,ha='center')

	yt = -0.25
	ax.text(8.85, yt,r't$_{\mathrm{\bm{age}}}$',fontsize=fontsize,color=color1_text,weight='bold',ha='center')
	ax.text(9.85, yt,r't$_{\mathrm{\bm{age}}}$',fontsize=fontsize,color=color2_text,weight='bold',ha='center')
	ax.text(2.85, yt,r't$_{\mathrm{\bm{age}}}$',fontsize=fontsize,color=color3_text,weight='bold',ha='center')

	ax.text(1.22,1.3,r't$_{\mathrm{\bm{trunc}}}$',fontsize=fontsize,color=color1_text,weight='bold',ha='center')
	ax.text(4, 2.8,r't$_{\mathrm{\bm{trunc}}}$',fontsize=fontsize,color=color2_text,weight='bold',ha='center')

	ax.text(0.76,0.65,r'sf$_{\mathrm{\bm{slope}}}$',fontsize=fontsize,color=color1_text,weight='bold',ha='center',rotation=76)
	ax.text(2, 3.4,r'sf$_{\mathrm{\bm{slope}}}$',fontsize=fontsize,color=color2_text,weight='bold',ha='center',rotation=-16)

	ax.set_xlabel('lookback time')
	ax.set_xlim(np.min(t),np.max(t)+0.2)
	ax.set_ylabel('SFR')
	ax.set_ylim(0.0,5.3)

	ax.set_xticklabels([])
	ax.set_xticks([])
	ax.set_yticklabels([])
	ax.set_yticks([])

	plt.savefig('/Users/joel/my_papers/prospector_brown/figures/sfh_illustration.png',dpi=150)













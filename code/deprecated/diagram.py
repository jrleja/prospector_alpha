import matplotlib.pyplot as plt
import matplotlib.patches as patches

def annotate_arrow(ax, from_loc, to_loc, text, tweaks,angle):
	

	x = (from_loc[0]+to_loc[0])/2. + tweaks[0]
	y = (from_loc[1]+to_loc[1])/2. + tweaks[1]

	ax.text(x,y,
	        text,
	        ha='center',
	        va='center',
	        rotation = angle,
	        **arrow_txt_args)

	return ax

outname = '/Users/joel/my_papers/prospector_brown/figures/diagram.png'

##### SETUP FIGURE
fig, ax1 = plt.subplots(1, 1)
ax1.set_axis_off()

##### LOCATIONS
inloc  =  [(.50,.93),(.80,.93)]
proloc  = [(.50,.60),(.80,.60)]
spsloc  = [(.15,.60),(.10,.88)]
samploc = [(.65,.30),(.90,.30)]
outloc  = [(.40,.15)]

##### COLORS
prosp_color = 'orange'
sps_color = 'blue'
samp_color = 'green'
out_color = "0.5"

##### GENERAL FORMATTING
bbox_args = dict(boxstyle="round", fc=out_color,alpha=0.5)
arrow_args = dict(arrowstyle="->")
txt_args = dict(fontsize=16,fontname='Helvetica',ha="center",va="center")
arrow_txt_args = dict(fontsize=9, fontname='Helvetica')


##### INPUT BOXES
in1 = ax1.annotate('model \n choices', xy=inloc[0],  xycoords='figure fraction',
                   bbox=bbox_args,
                   **txt_args
                   )

in2 = ax1.annotate('Observed \n photometry', xy=(1.0,0.5),  xycoords='figure fraction',
                   xytext=inloc[1], textcoords='figure fraction',
                   bbox=bbox_args,
                   **txt_args
                   )

##### Prospector boxes
bbox_args.update(fc=prosp_color)
pro1 = ax1.annotate('Prospector \n model', xy=proloc[0],  xycoords='figure fraction',
                   bbox=bbox_args,
                   **txt_args
                   )

pro2 = ax1.annotate('likelihood \n function', xy=(0.5,0.5),  xycoords=pro1,
                   xytext=proloc[1], textcoords='figure fraction',
                   bbox=bbox_args,
                   arrowprops=dict(patchB=pro1.get_bbox_patch(),
                                   connectionstyle="arc3,rad=0.2",
                                   **arrow_args),
                   **txt_args
                   )

pro3 = ax1.annotate('', xy=(.5, .5),  xycoords=pro2,
                   textcoords=pro1,xytext=(.5, .5),
                   bbox=bbox_args,
                   arrowprops=dict(patchA=pro1.get_bbox_patch(),
                                   patchB=pro2.get_bbox_patch(),
                                   connectionstyle="arc3,rad=0.2",
                                   **arrow_args),
                   **txt_args
                   )

ax1 = annotate_arrow(ax1, proloc[0], proloc[1], 'model photometry', (0.035,-0.045), 'horizontal')
ax1 = annotate_arrow(ax1, proloc[0], proloc[1], 'fit parameters', (0.028,0.09), 'horizontal')


# jmodel to prospector model
cnt1 = ax1.annotate('', xy=(.6,.5),  xycoords=pro1,
                   xytext=(.6,.5),  textcoords=in1,
                   bbox=bbox_args,
                   arrowprops=dict(patchA=in1.get_bbox_patch(),
                   				   patchB=pro1.get_bbox_patch(),
                                   connectionstyle="arc3,rad=0.2",
                                   **arrow_args),
                   **txt_args
                   )

ax1 = annotate_arrow(ax1, inloc[0], proloc[0], 'all parameters', (-0.05,0.08),'vertical')

# jmodel to likelihood function
cnt2 = ax1.annotate('', xy=(1.0,0.0),  xycoords=pro2,
                   xytext=(1.0,0.0),  textcoords=in1,
                   bbox=bbox_args,
                   arrowprops=dict(patchA=in1.get_bbox_patch(),
                   				   patchB=pro2.get_bbox_patch(),
                                   connectionstyle="arc3,rad=0.2",
                                   **arrow_args),
                   **txt_args
                   )

ax1 = annotate_arrow(ax1, inloc[0], proloc[1], 'priors', (0.02,0.07),-50)

# observed phot to likelihood function
cnt3 = ax1.annotate('', xy=(.4,.5),  xycoords=pro2,
                   xytext=(.4,.5),  textcoords=in2,
                   bbox=bbox_args,
                   arrowprops=dict(patchA=in2.get_bbox_patch(),
                   				   patchB=pro2.get_bbox_patch(),
                                   connectionstyle="arc3,rad=-0.2",
                                   **arrow_args),
                   **txt_args
                   )

ax1 = annotate_arrow(ax1, inloc[1], proloc[1], 'obs. photometry', (0.1,0.07),-90)







##### SPS boxes
bbox_args.update(fc=sps_color)
sps1 = ax1.annotate('python-fsps', xy=spsloc[0],  xycoords='figure fraction',
                   bbox=bbox_args,
                   **txt_args
                   )

sps2 = ax1.annotate('FSPS', xy=(0.5,0.5),  xycoords=sps1,
                   xytext=spsloc[1], textcoords='figure fraction',
                   bbox=bbox_args,
                   arrowprops=dict(patchB=sps1.get_bbox_patch(),
                                   connectionstyle="arc3,rad=0.2",
                                   **arrow_args),
                   **txt_args
                   )

sps3 = ax1.annotate('', xy=(.5, .5),  xycoords=sps2,
                   textcoords=sps1,xytext=(.5, .5),
                   bbox=bbox_args,
                   arrowprops=dict(patchA=sps1.get_bbox_patch(),
                                   patchB=sps2.get_bbox_patch(),
                                   connectionstyle="arc3,rad=0.2",
                                   **arrow_args),
                   **txt_args
                   )

ax1 = annotate_arrow(ax1, spsloc[0], spsloc[1], 'model photometry', (-.175,.07), 97.5)
ax1 = annotate_arrow(ax1, spsloc[0], spsloc[1], 'SPS parameters', (-.07,.08), -80)

# python-fsps to prospector model
cnt3 = ax1.annotate('', xy=(.4,.4),  xycoords=pro1,
                   xytext=(1.0,1.0),  textcoords=sps1,
                   bbox=bbox_args,
                   arrowprops=dict(patchA=sps1.get_bbox_patch(),
                   				   patchB=pro1.get_bbox_patch(),
                                   connectionstyle="arc3,rad=-0.2",
                                   **arrow_args),
                   **txt_args
                   )

# python-fsps to prospector model
cnt3 = ax1.annotate('', xy=(1.0,0.0),  xycoords=sps1,
                   xytext=(1.0,1.0),  textcoords=pro1,
                   bbox=bbox_args,
                   arrowprops=dict(patchA=pro1.get_bbox_patch(),
                   				   patchB=sps1.get_bbox_patch(),
                                   connectionstyle="arc3,rad=-0.2",
                                   **arrow_args),
                   **txt_args
                   )

ax1 = annotate_arrow(ax1, proloc[0], spsloc[0], 'model photometry', (-0.06,.09),0.0)
ax1 = annotate_arrow(ax1, proloc[0], spsloc[0], 'model parameters', (-0.06,-.05),0.0)






##### sampler box
bbox_args.update(fc=samp_color)

### first emcee
samp1 = ax1.annotate('emcee', xy=(0.4,0.5),  xycoords=pro2,
                   xytext=samploc[0], textcoords='figure fraction',
                   bbox=bbox_args,
                   arrowprops=dict(patchB=pro2.get_bbox_patch(),
                                   connectionstyle="arc3,rad=0.2",
                                   **arrow_args),
                   **txt_args
                   )

cnt4 = ax1.annotate('', xy=(0.5,0.5),  xycoords=samp1,
                   xytext=(0.4,0.0),  textcoords=pro2,
                   bbox=bbox_args,
                   arrowprops=dict(patchA=pro2.get_bbox_patch(),
                   				   patchB=samp1.get_bbox_patch(),
                                   connectionstyle="arc3,rad=0.2",
                                   **arrow_args),
                   **txt_args
                   )

ax1 = annotate_arrow(ax1, samploc[0], proloc[1], 'likelihood', (0.00,.01),57)
ax1 = annotate_arrow(ax1, samploc[0], proloc[1], 'fit parameters', (0.085,-.05),57)


### then Powell
samp2 = ax1.annotate('scipy \n Powell', xy=(0.5,1.0),  xycoords=pro2,
                   xytext=samploc[1], textcoords='figure fraction',
                   bbox=bbox_args,
                   arrowprops=dict(patchB=pro2.get_bbox_patch(),
                                   connectionstyle="arc3,rad=0.2",
                                   **arrow_args),
                   **txt_args
                   )

cnt4 = ax1.annotate('', xy=(0.7,0.6),  xycoords=samp2,
                   xytext=(0.6,1.0),  textcoords=pro2,
                   bbox=bbox_args,
                   arrowprops=dict(patchA=pro2.get_bbox_patch(),
                   				   patchB=samp2.get_bbox_patch(),
                                   connectionstyle="arc3,rad=0.2",
                                   **arrow_args),
                   **txt_args
                   )

ax1 = annotate_arrow(ax1, samploc[1], proloc[1], 'likelihood', (0.065,-.03),-65)
ax1 = annotate_arrow(ax1, samploc[1], proloc[1], 'fit parameters', (0.14,0.0),-75)





##### output box
bbox_args.update(fc=out_color)

out1 = ax1.annotate('output', xy=outloc[0],  xycoords='figure fraction',
                   bbox=bbox_args,
                   **txt_args
                   )

cnt5 = ax1.annotate('', xy=(0.5,0.5),  xycoords=out1,
                   xytext=(0.4,0.0),  textcoords=samp1,
                   bbox=bbox_args,
                   arrowprops=dict(patchA=samp1.get_bbox_patch(),
                   				   patchB=out1.get_bbox_patch(),
                                   connectionstyle="arc3,rad=0.2",
                                   **arrow_args),
                   **txt_args
                   )


ax1 = annotate_arrow(ax1, outloc[0], samploc[0], 'fit parameter PDFs', (-.03,-.02),25)


##### ANNOTATION
prosp_color = 'orange'
sps_color = 'blue'
samp_color = 'green'
out_color = "0.5"

ax1.add_patch(
    patches.Rectangle(
        (0.01, 0.12),
        0.24,
        0.225,
        linewidth=2,
        fill=False      # remove background
    ))

ax1.text(0.015,0.365, 'Key',transform=ax1.transAxes,weight='semibold',fontsize='x-large')
ax1.text(0.02,0.3, 'I/O',transform=ax1.transAxes,color=out_color,weight='bold',fontsize='large')
ax1.text(0.02,0.25, 'SPS code',transform=ax1.transAxes,color=sps_color,weight='bold',fontsize='large')
ax1.text(0.02,0.2, 'Prospector',transform=ax1.transAxes,color=prosp_color,weight='bold',fontsize='large')
ax1.text(0.02,0.15, 'Minimizers',transform=ax1.transAxes,color=samp_color,weight='bold',fontsize='large')



plt.savefig(outname, dpi=300)
plt.close()
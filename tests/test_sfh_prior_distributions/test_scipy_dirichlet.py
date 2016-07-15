import numpy as np
import matplotlib.pyplot as plt
import magphys_plot_pref
import math
import operator

def marginalized_dirichlet_pdf(x, alpha_one, alpha_sum):
	''' 
	returns the marginalized Dirichlet PDF (i.e., for ONE component of the distribution).
	From Wikipedia, the marginalized Dirichlet PDF is the Beta function, 
	where a = alpha_1 and b = sum(alpha)-alpha_1. Here, "alpha" are the concentration 
	parameters of the Dirichlet distribution, and alpha_1 is the dimension to marginalize over.

	input "x" is the x-vector
	'''

	return (math.gamma(alpha_one+alpha_sum) * x**(alpha_one-1) * \
			 (1-x)**(alpha_sum-1)) / \
			 (math.gamma(alpha_one)*math.gamma(alpha_sum)) 

np.random.seed(50)

###### NUMBER OF DRAWS
ndraw_dirichlet = int(1e6)
nbins = 5

##### draw dirichlet
dirichlet = np.random.dirichlet(tuple(1. for x in xrange(nbins)),ndraw_dirichlet)
dirichlet_straight = np.ravel(dirichlet)

##### scipy dirichlet
test_x = np.linspace(1e-5,1,1000,endpoint=False)
#test_dirichlet = dirichlet_pdf(test_x,np.array([1. for x in xrange(nbins)]))
test_dirichlet = marginalized_dirichlet_pdf(test_x, 1,nbins-1)
#test_dirichlet = np.log10(test_dirichlet)

##### histoplot
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8,8))
nbins = 100
alpha = 0.5
lw = 2
color = 'blue'
histtype = 'step'

##### all five bins
num, b, p = ax.hist(dirichlet_straight, bins=nbins, range=(0.0,1.0), histtype=histtype, alpha=alpha,lw=lw,color=color,normed=True,log=True)
ax.plot(test_x,test_dirichlet,color='red',lw=2,alpha=0.5)

ax.set_ylabel('density')
ax.set_xlabel('x')

ax.text(0.02,0.1, 'analytical marginalized Dirichlet distribution', transform = ax.transAxes,ha='left',color='red',weight='bold')
ax.text(0.02,0.05, 'numerical marginalized Dirichlet distribution', transform = ax.transAxes,ha='left',color=color,weight='bold')

plt.savefig('dirichlet_function_test.png',dpi=150)
plt.close()
print 1/0

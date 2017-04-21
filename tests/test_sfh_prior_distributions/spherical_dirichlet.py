import numpy as np
import operator
import math
import matplotlib.pyplot as plt

##### draw dirichlet
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

### multiplicative sum
def prod(factors):
	return reduce(operator.mul, factors, 1)

##### draw spherical dirichlet distribution
# magical transformation from https://arxiv.org/pdf/1010.3436.pdf
def draw_spherical(n,ndim):

	x = np.zeros(shape=(n,ndim))
	z = np.zeros(shape=(n,ndim-1))

	tilde_alpha = np.array([ndim-i for i in xrange(1,ndim)])

	# draw z's
	for i in xrange(ndim-1): z[:,i] = np.random.beta(tilde_alpha[i], 1, size=n)

	# convert into dirichlet distribution
	for j in xrange(n):
		x[j,0] = 1-z[j,0]
		for i in xrange(1,ndim-1):
			x[j,i] =  prod(z[j,:i])*(1-z[j,i])
		x[j,-1] = prod(z[j,:])

	return x

#### general samples
ndim = 6
dirichlet_x = np.linspace(1e-5,1,1000,endpoint=False)
dirichlet = marginalized_dirichlet_pdf(dirichlet_x, 1, ndim-1)

dirichlet_sphere = draw_spherical(int(1e6),ndim)
dirichlet_sphere = dirichlet_sphere.ravel()


##### histoplot
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8,8))
nbins = 100
alpha = 0.5
lw = 2
color = 'blue'
histtype = 'step'

##### all five bins
num, b, p = ax.hist(dirichlet_sphere, bins=nbins, range=(0.0,1.0), histtype=histtype, alpha=alpha,lw=lw,color=color,normed=True,log=True)
ax.plot(dirichlet_x,dirichlet,color='red',lw=2,alpha=0.5)
plt.show()
print 1/0


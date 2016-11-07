import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('galaxy_sed_flux2.dat',delimiter=',',dtype='str')
freq = np.loadtxt('frequencies2.dat',delimiter=',')

data = np.asarray(data[:,2:],dtype='float')

c = 3.e18
wv = c / freq


unc = data[:,1::2]
data = data[:,::2]
print 1/0


for i,my_data in enumerate(data):
    plt.plot(wv,freq*my_data,'o',markersize=10)

plt.xscale('log')
plt.yscale('log')
plt.show()




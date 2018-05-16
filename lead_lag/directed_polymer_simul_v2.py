import numpy as np
import cPickle
import matplotlib.pyplot as plt
import random
import pylab
from scipy import interpolate

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def zero_temperature_position_columnar(T=15, alpha=0) :
	# T should be odd
	lattice=np.zeros((T,2*T))
	E=10000*np.ones(2*T)

	disorder=np.random.normal(size=(2*T-1))+alpha*np.abs(np.roll(np.arange(-T+1,T),T))
	#print disorder
	## Definition of the starting point
	E[0]=0

	for time in range(1,T) :
		for x in range(time,-time-2,-2) :
			values=[E[x-1],E[x+1]]
			lattice[time,x]=2*np.argmin(values)-1
			E[x]=np.amin(values)+disorder[x]

	Polym=np.zeros(T)

	## Definition of the ending point
	# Polym is the "x" position on the lattice
	itoto=0
	Polym[T-1]=itoto

	for time in range(1,T-1):
		idelta=lattice[T-time,int(itoto)]
		itoto+=idelta
		Polym[T-1-time]=itoto

	plt.figure(5)
	plt.plot(Polym,range(T), 'r', markersize=2)
	plt.title('Path of the polymer')
	plt.xlabel('Position of the polymer'), plt.ylabel('Time')
	plt.show(block=False)
	raw_input("<Hit Enter To Close>")
	plt.close()

	disorder2=np.roll(disorder, T, axis=0)

	# fig = plt.figure(6,figsize=(12, 8))
	# ax = fig.gca(projection='3d')

	# Make data.
	X = np.arange(-T, T)
	Y = np.arange(0, T)
	X, Y= np.meshgrid(X, Y)

	disorder3=disorder2[X]

	# # Plot the surface.
	# surf = ax.plot_surface(X, Y, disorder3, cmap=cm.jet,linewidth=1, antialiased=True)

	# # Customize the z axis.
	# #ax.set_zlim(-1.01, 1.01)
	# ax.zaxis.set_major_locator(LinearLocator(10))
	# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
	# ax.set_xlabel('x')
	# ax.set_ylabel('time')
	# ax.set_zlabel('Disorder strength')

	# # Add a color bar which maps values to colors.
	# fig.colorbar(surf, shrink=0.5, aspect=5)
	# plt.show(block=False)
	# raw_input("<Hit Enter To Close>")
	# plt.close()

	#### Smoothing the surface

	xnew, ynew = np.mgrid[-T:T:0.1, 0:T:0.1]
	tck = interpolate.bisplrep(X, Y, disorder3, s=0)
	znew = interpolate.bisplev(xnew[:,0], ynew[0,:], tck)

	fig = plt.figure(7,figsize=(12,8))
	ax2 = fig.gca(projection='3d')

	surf2=ax2.plot_surface(xnew, ynew, znew, cmap=cm.jet, rstride=1, cstride=1, alpha=None, linewidth=0, antialiased=True)
	ax2.zaxis.set_major_locator(LinearLocator(10))
	ax2.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
	ax2.set_xlabel('x')
	ax2.set_ylabel('time')
	ax2.set_zlabel('Disorder strength')
	ax2.grid(b=True)

	fig.colorbar(surf2, shrink=0.5, aspect=5)
	plt.show(block=False)
	raw_input("<Hit Enter To Close>")
	plt.close()


zero_temperature_position_columnar()
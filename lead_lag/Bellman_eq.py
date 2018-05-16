import numpy as np
import cPickle
import matplotlib.pyplot as plt
import random
import pylab
from scipy import interpolate

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

random.seed(-2)
#np.random.seed(20)

T  =  101

## Free energy of the polymer
def zero_temperature_energy(T=101):
	E = 100*np.ones(2*T-1)
	E[0] = 0

	for time in range(1,T) :
	    for x in range(time,-time-2,-2) :
	        E[x] = np.minimum(E[x-1],E[x+1]) + np.random.normal()

	xt1 = np.arange(T)
	xt2 = np.arange(T-1)-T+1
	xt = np.concatenate((xt1,xt2))

	# ou xt=np.roll(np.arange(-T+1,T),T)

	plt.figure(3)
	plt.plot(xt,E, 'r.', markersize=2)
	plt.title('Energy vs position')
	plt.xlabel('Position'), plt.ylabel('Energy')
	plt.show(block=False)
	raw_input("<Hit Enter To Close>")
	plt.close()

def zero_temperature_position(T=101) :
	# T should be odd
	lattice=np.zeros((T,2*T))
	E=10000*np.ones(2*T)

	## Definition of the starting point
	E[0]=0

	for time in range(1,T) :
		for x in range(time,-time-2,-2) :
			values=[E[x-1],E[x+1]]
			lattice[time,x]=2*np.argmin(values)-1
			E[x]=np.amin(values)+np.random.normal()

	Polym=np.zeros(T)

	## Definition of the ending point
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

def temperature_partition(T=101) :
	beta=1
	Z=np.zeros(2*T-1)
	Z[0]=1
	for time in range(1,T) :
		for x in range(time,-time-2,-2) :
			Z[x]=(Z[x-1]+Z[x+1])*math.exp(-beta*np.random.normal())

def zero_temperature_position_3D(T=101) :
	# T should be odd
	lattice=np.zeros((T,2*T))
	E=10000*np.ones(2*T)

	disorder=np.random.normal(size=(T,2*T))

	fig = plt.figure(6,figsize=(12, 8))
	ax = fig.gca(projection='3d')

	# Make data.
	X = np.arange(-T, T)
	Y = np.arange(0, T)
	X, Y = np.meshgrid(X, Y)

	# Plot the surface.
	surf = ax.plot_surface(X, Y, disorder, cmap=cm.jet,linewidth=10, antialiased=False)

	# Customize the z axis.
	#ax.set_zlim(-1.01, 1.01)
	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
	ax.set_xlabel('x')
	ax.set_ylabel('time')
	ax.set_zlabel('Disorder strength')

	# Add a color bar which maps values to colors.
	fig.colorbar(surf, shrink=0.5, aspect=5)
	plt.show(block=False)
	raw_input("<Hit Enter To Close>")
	plt.close()


	## Definition of the starting point
	E[0]=0

	for time in range(1,T) :
		for x in range(np.minimum(time,T-time+1),np.maximum(-time,time-T-1)-2,-2) :
			values=[E[x-1],E[x+1]]
			lattice[time,x]=2*np.argmin(values)-1
			E[x]=np.amin(values)+disorder[time,x]

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


def zero_temperature_position_3D_2(T=101) :
	# T should be odd
	lattice=np.zeros((T,2*T))
	E=10000*np.ones(2*T)

	disorder=np.random.normal(size=(T,2*T))

	disorder2=np.roll(disorder, T, axis=1)

	fig = plt.figure(6,figsize=(12, 8))
	ax = fig.gca(projection='3d')

	# Make data.
	X = np.arange(-T, T)
	Y = np.arange(0, T)
	X, Y = np.meshgrid(X, Y)

	# Plot the surface.
	surf = ax.plot_surface(X, Y, disorder2, cmap=cm.jet,linewidth=1, antialiased=True)

	# Customize the z axis.
	#ax.set_zlim(-1.01, 1.01)
	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
	ax.set_xlabel('x')
	ax.set_ylabel('time')
	ax.set_zlabel('Disorder strength')

	# Add a color bar which maps values to colors.
	fig.colorbar(surf, shrink=0.5, aspect=5)
	plt.show(block=False)
	raw_input("<Hit Enter To Close>")
	plt.close()

	#### Smoothing the surface

	xnew, ynew = np.mgrid[-T:T:0.5, -T:T:0.5]
	tck = interpolate.bisplrep(X, Y, disorder2, s=0)
	znew = interpolate.bisplev(xnew[:,0], ynew[0,:], tck)

	fig = plt.figure(7,figsize=(12,8))
	ax2 = fig.gca(projection='3d')
	surf2=ax2.plot_surface(xnew, ynew, znew, cmap=cm.jet, rstride=1, cstride=1, alpha=None, antialiased=True)
	ax2.zaxis.set_major_locator(LinearLocator(10))
	ax2.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
	ax2.set_xlabel('x')
	ax2.set_ylabel('time')
	ax2.set_zlabel('Disorder strength')

	fig.colorbar(surf2, shrink=0.5, aspect=5)
	plt.show(block=False)
	raw_input("<Hit Enter To Close>")
	plt.close()


	## Definition of the starting point
	E[0]=0

	for time in range(1,T) :
		for x in range(np.minimum(time,T-time+2),np.maximum(-time,time-T-2)-2,-2) :
			values=[E[x-1],E[x+1]]
			lattice[time,x]=2*np.argmin(values)-1
			E[x]=np.amin(values)+disorder[time,x]

	Polym=np.zeros(T)

	## Definition of the ending point
	# Polym is the "x" position on the lattice
	itoto=0
	Polym[T-1]=itoto

	for time in range(1,T-1):
		idelta=lattice[T-time,int(itoto)]
		itoto+=idelta
		Polym[T-1-time]=itoto

	plt.figure(8)
	plt.plot(Polym,range(T), 'r', markersize=2)
	plt.title('Path of the polymer')
	plt.xlabel('Position of the polymer'), plt.ylabel('Time')
	plt.show(block=False)
	raw_input("<Hit Enter To Close>")
	plt.close()


def zero_temperature_position_columnar(T=101) :
	# T should be odd
	lattice=np.zeros((T,2*T))
	E=10000*np.ones(2*T)

	disorder=np.random.normal(size=(T+1,1))+alpha*np.roll(np.arange(-T+1,T),T)
	## Definition of the starting point
	E[0]=0

	for time in range(1,T) :
		for x in range(time,-time-2,-2) :
			values=[E[x-1],E[x+1]]
			lattice[time,x]=2*np.argmin(values)-1
			E[x]=np.amin(values)+np.random.normal()

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

#zero_temperature_position()
zero_temperature_position_3D_2(50)
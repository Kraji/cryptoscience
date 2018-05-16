import numpy as np
import cPickle
import matplotlib.pyplot as plt 
import sys
#sys.setrecursionlimit(15000)

# Transfer matrix method at non-zero temperature
## In time frame t1,t2
### FORWARD
memo2={}
def partition_forward(t1,t2,epsilon,T,N,t10,t20,t11,t21):
	if (t1<t10 or t2<t20 or t1>t11 or t2>t21):
		return 0
	else:
		if not (t1,t2) in memo2:
			if (t1==0 and t2==0):
				memo2[t1,t2]=epsilon[t1,t2]
			else:
				memo2[t1,t2]=epsilon[t1,t2] * (partition_forward(t1,t2-1,epsilon,T,N) + partition_forward(t1-1,t2,epsilon,T,N) + partition_forward(t1-1,t2-1,epsilon,T,N))
		return memo2[t1,t2]

### BACKWARD
memo3={}
def partition_backward(t1,t2,epsilon,T,N, t10, t20, t11,t21):
	if (t1>t11 or t2>t21 or t1<t10 or t2<t20):
		return 0
	else:
		if not (t1,t2) in memo3:
			if (t1==N-1 and t2==N-1):
				memo3[t1,t2]=epsilon[t1,t2]
			else:
				memo3[t1,t2]=epsilon[t1,t2] * (partition_backward(t1,t2+1,epsilon,T,N)+partition_backward(t1+1,t2,epsilon,T,N)+partition_backward(t1+1,t2+1,epsilon,T,N))
		return memo3[t1,t2]

## In rotated frame where t=t1+t2 and x=t2-t1
## Partition function
def sum_partition_rotated(partition,t,epsilon,T,N,t10,t20,t11,t21):
	sum=0
	if (t=>t10+t20 and t<=t11+t21):
		for x in range(int(np.maximum(-t,t-2*(N-1))),int(np.minimum(t,2*(N-1)-t))+1):
			if ((t-x)%2==0 and (t+x)%2==0):
				sum+=partition((t-x)/2,(t+x)/2,epsilon,T,N,t10,t20,t11,t21)
	return sum

## Average of lead/lag <x(t)>
def average_x(partition,t,epsilon,T,N,t10,t20,t11,t21):
	avr_x=0
	if (t=>t10+t20 and t<=t11+t21):
		for x in range(int(np.maximum(-t,t-2*(N-1))),int(np.minimum(t,2*(N-1)-t))+1):
			if ((t-x)%2==0 and (t+x)%2==0):
				avr_x+=x*partition((t-x)/2,(t+x)/2,epsilon,T,N,t10,t20,t11,t21)
	return avr_x/sum_partition_rotated(partition,t,epsilon,T,N,t10,t20,t11,t21)

## Cost energy
def cost_energy(partition,t,epsilon,T,N,t10,t20,t11,t21):
	cost_energy=0
	if (t=>t10+t20 and t<=t11+t21):
		for x in range(int(np.maximum(-t,t-2*(N-1))),int(np.minimum(t,2*(N-1)-t))+1):
			if ((t-x)%2==0 and (t+x)%2==0):
				cost_energy+=epsilon[(t-x)/2,(t+x)/2]*partition((t-x)/2,(t+x)/2,epsilon,T,N,t10,t20,t11,t21)
	return cost_energy/sum_partition_rotated(partition,t,epsilon,T,N,t10,t20,t11,t21)


## TOPS METHOD 
def tops_average_x(t,epsilon,T,N,t10,t20,t11,t21):
	return ( average_x(partition_forward,t,epsilon,T,N,t10,t20,t11,t21) + average_x(partition_backward,t,epsilon,T,N,t10,t20,t11,t21) )/2.

def tops_cost_energy(epsilon,T,N,t10,t20,t11,t21):
	tops_cost=0
	for t in range(t10+t20,t11+t21):
		tops_cost+=(cost_energy(partition_forward,t,epsilon,T,N)+cost_energy(partition_backward,t,epsilon,T,N))/2.
	return tops_cost/(2*N-1)

## Time series renormalization (pre-processing)

def processing(time_series):
	return (time_series-np.mean(time_series))/np.std(time_series)


## Creation of the time series
nb_points=500
X=np.zeros(nb_points+200)
X[0]=5
for k in range(1,nb_points+200):
	X[k]=0.7*X[k-1]+np.random.normal(loc= 0,scale= 0.5)

Z=X[100:nb_points+100]
ZZ=processing(Z)

Y=np.zeros(nb_points)
a=0.8
period=nb_points
amplitude=nb_points/25.

theoretical=-(amplitude*np.cos(2*np.pi*np.arange(nb_points)/period)).astype(int)
Y=a*X[100+theoretical+np.arange(nb_points)]+np.random.normal(loc= 0,scale= 0.1, size=nb_points)

YY=processing(Y)

## Definition of the metric matrix

epsilon_minus=np.square( np.tile(ZZ,(ZZ.shape[0],1))-np.tile(YY,(YY.shape[0],1)).transpose())

## Parameter of the TOP(S)
average_tops=np.zeros(2*nb_points-1)
T=1.2

boltzmann_epsilon=np.exp(-epsilon_minus/T)

for i in range(2*nb_points-1):
	average_tops[i]=tops_average_x(i,boltzmann_epsilon,T,nb_points)

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,8))
ax1.plot(ZZ), ax1.plot(YY)

TOPS=ax2.plot(np.arange(2*nb_points-1)/2., average_tops, label='TOPS')
model=ax2.plot(np.arange(nb_points), theoretical, label='Model')
ax2.legend(['TOPS','Model'])

plt.show(block=False)
raw_input("<Hit Enter To Close>")
plt.close()


memo2plot=[sum_partition_rotated(partition_forward,t,epsilon_minus,T,nb_points) for t in range(2*nb_points-1)]
memo3plot=[sum_partition_rotated(partition_backward,t,epsilon_minus,T,nb_points) for t in range(2*nb_points-1)]

plt.plot(memo2plot)
plt.plot(memo3plot)
plt.legend(['memo2','memo3'])
plt.show(block=False)
raw_input("<Hit Enter To Close>")
plt.close()

memo44=np.array([np.log10(memo2[i,i]) for i in range(nb_points)])

plt.plot(memo44)
plt.legend(['forward partition function diagonal'])
plt.show(block=False)
raw_input("<Hit Enter To Close>")
plt.close()


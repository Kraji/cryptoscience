import numpy as np
import cPickle
from box_making import box_making
import matplotlib.pyplot as plt 

# Open data
f = open('EthEur.p','rb')
A = cPickle.load(f)
f.close()

# Vwap over time interval delta_t
delta_t = 200 # in seconds

data=box_making(A, delta_t, -1, -1)

f=open('compressed_EthEur4.p','wb')
cPickle.dump(data,f)
f.close()
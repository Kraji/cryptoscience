import numpy as np
import pandas as pd

# data_input: price, volume, time
def box_making(data_input, delta_t, initial_time,final_time):
    dataShape = data_input.shape
    (nRows, nCols)= dataShape
    nb_data=nRows
    
    price=data_input[:,0]
    volume=data_input[:,1]
    time=data_input[:,2]

    if initial_time==-1:
        initial_time=time[0]

    if final_time==-1:
        final_time=time[-1]
    
    time_interval = final_time - initial_time
    nb_bins = int(time_interval // delta_t +1)
    
    data_compressed=np.zeros((nb_bins,5))
    
    vwap = np.zeros(nb_bins)
    weights = np.zeros(nb_bins)
    time_stamp=initial_time+delta_t*np.arange(nb_bins)
    variance=np.zeros(nb_bins)
    nb_transaction=np.zeros(nb_bins)

    initialization=0

    for j in range(nb_data): 
        ind = int((time[j]-initial_time)//delta_t)
        if ind > -1 :
            if (ind==0 and initialization==0):
                nb_transaction[0]=j
                initialization=1
            elif ind==ind2+1:
                nb_transaction[ind]=j
            vwap[ind] += volume[j] * price[j]
            weights[ind] += volume[j]
            ind2=ind
    for k in range(len(vwap)):
    	if weights[k] > 0.001:
            vwap[k] = vwap[k] / weights[k]
    	else:
    	    vwap[k] = vwap[k-1]
    
    for j in range(nb_data):
        ind = int((time[j]-initial_time)//delta_t)
        if ind>-1:
           variance[ind] += volume[j] * (price[j]-vwap[ind])**2
    for k in range(len(vwap)):
        if weights[k] > 0.001:
            variance[k] = variance[k] / weights[k]
        else:
            variance[k] = variance[k-1]

    data_compressed[:,0]=vwap
    data_compressed[:,1]=weights
    data_compressed[:,2]=time_stamp
    data_compressed[:,3]=variance
    data_compressed[:,4]=nb_transaction

    return data_compressed

def box_making_dataframe(data_input, delta_t, initial_time,final_time):
    data_boxed = box_making(data_input, delta_t, initial_time,final_time)
    df = pd.DataFrame(data=data_boxed[0:,0:], columns=['price','volume','time_stamp','variance','index_transaction']) 
    return df
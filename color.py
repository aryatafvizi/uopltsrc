import dst13
import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib
from sklearn.cluster import KMeans
import pickle as pkl
import math
from ternary import ternaryPlot, ternaryAxConfig
from matplotlib import animation

labs = dst13.dst_window_labels.WindowLabels(hand=True, nopos=False) # window labels                                                                        
res_ind = [i for i in range(4147) if (labs.rvec[i]>1000)] #residential window indeces
com_ind = [i for i in range(4147) if (labs.rvec[i]<=1000)] #commercial window indeces

lcs = pkl.load(open(os.path.join(os.environ['DST_WRITE'],'lcs_night_09_lcs.pkl')))  
res_lcs = np.array([lcs[i,:,:] for i in res_ind])
com_lcs = np.array([lcs[i,:,:] for i in com_ind])

fig = plt.figure(figsize=(25, 15), dpi=100)
ax = fig.add_subplot(111)
ternaryAxConfig(ax)

res_data = res_lcs[:,0,:]
res_newdata = ternaryPlot(res_data)
res_scat = plt.scatter(res_newdata[:,0], res_newdata[:,1], alpha=0.5, vmin=0, vmax=1, edgecolors='none', cmap=cm.autumn)

com_data = com_lcs[:,0,:]
com_newdata = ternaryPlot(com_data)
com_scat = plt.scatter(com_newdata[:,0], com_newdata[:,1], alpha=0.5, vmin=0, vmax=1, edgecolors='none', cmap=cm.winter)

minval = np.min(lcs)#.min(axis=2),axis=1)[:,np.newaxis,np.newaxis] #min color value to be subtracted from all values

res_lcsnorm = res_lcs - minval #subtract a constant from all rgb values to separate out the colors
com_lcsnorm = com_lcs - minval

res_normalizer = 1.001*np.max(res_lcs.sum(axis=2),axis=1)[:, np.newaxis, np.newaxis] #normalize each lightcurve so all rgb values are less than 1, and r+g+b=1 at max timestep
com_normalizer = 1.001*np.max(com_lcs.sum(axis=2),axis=1)[:, np.newaxis, np.newaxis]

res_lcsnorm = res_lcsnorm/res_normalizer
com_lcsnorm = com_lcsnorm/com_normalizer

res_lcsnorm = np.array([res_lcsnorm[i,:,:] for i in range(res_lcsnorm.shape[0]) if np.var(res_lcsnorm[i,:,:].sum(1)) > 0.05]) #pick only the high variance lightcurves
com_lcsnorm = np.array([com_lcsnorm[i,:,:] for i in range(com_lcsnorm.shape[0]) if np.var(com_lcsnorm[i,:,:].sum(1)) > 0.05]) #pick only the high variance lightcurves

def animate(i):
    global res_lcs, com_lcs, ax, scat, fig
    print i
    
    fig.suptitle('Timestep: '+str((i+1)*36))
    
    res_data = res_lcsnorm[:,(i+1)*36-36:(i+1)*36+36,:].mean(1)
    com_data = com_lcsnorm[:,(i+1)*36-36:(i+1)*36+36,:].mean(1)

    res_newdata = ternaryPlot(res_data)
    com_newdata = ternaryPlot(com_data)

    res_scat.set_offsets(res_newdata)
    com_scat.set_offsets(com_newdata)

    res_colors = cm.autumn(res_data.sum(1))
    res_colors[:,3] = res_data.sum(1) #set opacity equal to intensity (currently not working)
    res_scat.set_facecolors(res_colors)
    
    com_colors = cm.winter(com_data.sum(1))
    com_colors[:,3] = com_data.sum(1)
    com_scat.set_facecolors(com_colors)

    return res_scat, com_scat


anim = animation.FuncAnimation(fig, animate, frames=99)

anim.save('../plots/test_anim.mov', bitrate=10000, dpi=100)#, savefig_kwargs={'facecolor':'black'})

plt.show()

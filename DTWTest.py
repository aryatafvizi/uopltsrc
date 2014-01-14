import mlpy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle as pkl
import os
import numpy as np

lcs = pkl.load(open(os.path.join(os.environ['DST_WRITE'],'lcs_night_01_lcs.pkl')))
ideal = [1]*1800
ideal += [0]*1800

distances = []
for i in range(4147):

    dist = mlpy.dtw_std(ideal, lcs[i,:,0]/np.linalg.norm(lcs[i,:,0]), dist_only=True)
    #if dist < 100000:
    plt.plot(lcs[i,:,0],label=str(dist))
    plt.legend()
    plt.savefig('stepcurve'+str(i)+'.png')
    plt.clf()
    #distances.append(dist)

#plt.hist(distances, bins=100)
#plt.savefig('DTWnormhist.png')

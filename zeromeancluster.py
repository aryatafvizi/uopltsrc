import pickle as pkl
import numpy as np
import dst13
import sys
import os
import math
from sklearn import metrics
from multiprocessing import Process
from sklearn.cluster import KMeans
from dst13.py.dst_light_curves import LightCurves
import matplotlib.pyplot as plt

def calcSigmaCurve(normalized,model,i):
    diffs = []
    n = 0
    for j in range(len(normalized)):
        if model.labels_[j]==i:
            diffs.append(model.cluster_centers_[i] - normalized[j])
            n+=1
    return n, np.array([np.std([d[j] for d in diffs]) for j in range(len(diffs[0]))])



lcs = pkl.load (open(os.path.join(os.environ['DST_WRITE'],'lcs_night_00_lcs.pkl'))) 
#zeromean = [lcs[i,:,0]-np.mean(lcs[i,:,0]) for i in range(4147)]
normalized = [lcs[i,:,0]/np.linalg.norm(lcs[i,:,0]) for i in range(4147) if (np.var(lcs[i,:,0]/np.linalg.norm(lcs[i,:,0])) > 0.000005)]
print len(normalized)
#var = [np.var(lcs[i,:,0]/np.linalg.norm(lcs[i,:,0])) for i in range(4147)]
#plt.hist(var,bins=100,range=(0,0.0001))
#plt.savefig('varhist.png')

kmeans_ = KMeans(init='random',n_clusters=12,n_init=10)
kmeans_.fit(normalized) 

f, axarr = plt.subplots(3, 4)

for i in range(12):
    n, sigmaCurve = calcSigmaCurve(normalized,kmeans_,i)
    for j in range(len(normalized)):
        if j%(int(n/5)) == 0:
            if kmeans_.labels_[j]==i:
                axarr[i%3, (i-(i%3))/3 - 1].plot(normalized[j],color='gray',alpha=0.2,lw=0.1)
    axarr[i%3, (i-(i%3))/3 - 1].plot(kmeans_.cluster_centers_[i],color='black',lw=0.1, label=('N = '+str(n)+'\n'+'cluster '+str(i)))
    axarr[i%3, (i-(i%3))/3 - 1].plot(kmeans_.cluster_centers_[i] + sigmaCurve,color='red',lw=0.1)
    axarr[i%3, (i-(i%3))/3 - 1].plot(kmeans_.cluster_centers_[i] - sigmaCurve,color='red',lw=0.1)
    axarr[i%3, (i-(i%3))/3 - 1].xaxis.set_visible(False)
    axarr[i%3, (i-(i%3))/3 - 1].yaxis.set_visible(False)
    #axarr[i%3, (i-(i%3))/3 - 1].set_ylim([0.010,0.040])
    axarr[i%3, (i-(i%3))/3 - 1].set_xlim([0,3600])
    axarr[i%3, (i-(i%3))/3 - 1].legend(prop={'size':4})
#plt.suptitle('Day ' + str(day))
plt.savefig('highvar.png',dpi=1000)

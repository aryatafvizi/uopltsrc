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

def l2_norm(lcs, norm='time', rgb=False):

    """ Calculate the L2 norm of the light curves in either time                                                                                         
        (default) or space. """

        # -- calculate L2-norm                                                                                                                               
    ax = 1 if norm=='time' else 0
        # -- return transposed (for speed later)                                                                                                             
    return np.sqrt((lcs**2).sum(axis=ax)).T



def calcSigmaCurve(lcs,model,i):
    diffs = []
    n = 0
    for j in range(len(lcs)):
        if model.labels_[j]==i:
            diffs.append(model.cluster_centers_[i] - lcs[j,:,1]/np.linalg.norm(lcs[j,:,1]))
            n+=1
    return n, np.array([np.std([d[j] for d in diffs]) for j in range(len(diffs[0]))])

"""
#lcs = pkl.load(open(os.path.join(os.environ['DST_WRITE'],'lcs_night_10_lcs.pkl')))
#kmeans_ = pkl.load(open(os.path.join(os.environ['DST_WRITE'], 'lcs_night_10_kmeans.pkl')))
#l2    = l2_norm(lcs)
#kmeans_ = KMeans(init='random',n_clusters=12,n_init=10
#kmeans_.fit((lcs[:,:,1].T/l2[1]).T)
#pkl.dump(kmeans_,open(os.path.join(os.environ['DST_WRITE'],'lcs_night_10_kmeans.pkl'),'wb'))
#sigma = calcSigmaCurve(lcs, l2, kmeans_,5)
for day in range(0,22):
    print day
    lcs = pkl.load(open(os.path.join(os.environ['DST_WRITE'],'lcs_night_'+str(day).zfill(2)+'_lcs.pkl')))
    #l2    = l2_norm(lcs)
    kmeans_ = pkl.load(open(os.path.join(os.environ['DST_WRITE'], 'kmeans_12_night_' + str(day).zfill(2) + '_0.pkl')))   
    #kmeans_ = KMeans(init='random',n_clusters=12,n_init=10)
    #kmeans_.fit((lcs[:,:,1].T/l2[1]).T)
    #pkl.dump(kmeans_,open(os.path.join(os.environ['DST_WRITE'],'lcs_night_'+str(day).zfill(2)+'_kmeans.pkl'),'wb'))  

    f, axarr = plt.subplots(3, 4)

    for i in range(12):
        n, sigmaCurve = calcSigmaCurve(lcs, kmeans_,i)
        print i
        for j in range(len(lcs)):
            if j%(int(n/5)) == 0:
                if kmeans_.labels_[j]==i:
                    axarr[i%3, (i-(i%3))/3 - 1].plot(lcs[j]/np.linalg.norm(lcs[j,:,1]),color='gray',alpha=0.2,lw=0.1)
        axarr[i%3, (i-(i%3))/3 - 1].plot(kmeans_.cluster_centers_[i],color='black',lw=0.1, label=('N = '+str(n)+'\n'+'cluster '+str(i)))
        axarr[i%3, (i-(i%3))/3 - 1].plot(kmeans_.cluster_centers_[i] + sigmaCurve,color='red',lw=0.1)
        axarr[i%3, (i-(i%3))/3 - 1].plot(kmeans_.cluster_centers_[i] - sigmaCurve,color='red',lw=0.1)
        axarr[i%3, (i-(i%3))/3 - 1].xaxis.set_visible(False)
        axarr[i%3, (i-(i%3))/3 - 1].yaxis.set_visible(False)
        axarr[i%3, (i-(i%3))/3 - 1].set_ylim([0.010,0.040])
        axarr[i%3, (i-(i%3))/3 - 1].set_xlim([0,3600])
        axarr[i%3, (i-(i%3))/3 - 1].legend(prop={'size':4})
    plt.suptitle('Day ' + str(day))
    plt.savefig('sigma_'+str(day)+'.png',dpi=1000)
"""

import sys
import pickle as pkl
import os
from munkres import Munkres, print_matrix
import numpy as np
import dst13
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib
import random
from sklearn.cluster import KMeans

#let's say model1 has one more curve than model2
def get_match_index(model1, model2):
    num_clusters = model1.n_clusters
    cost_matrix = np.zeros((num_clusters,num_clusters))
    cent1 = model1.cluster_centers_
    cent2 = model2.cluster_centers_
    ln = np.min([len(cent1[0]),len(cent2[0])])
    for i in range(num_clusters):
        for j in range(num_clusters):
            if j != (num_clusters-1):
                cost_matrix[i,j] = np.linalg.norm(model1.cluster_centers_[i][0:ln] - model2.cluster_centers_[j][0:ln])
            else:
                cost_matrix[i,j] = 100
    m = Munkres()
    indeces = m.compute(cost_matrix)
    ret = [i[1] for i in indeces]
    return ret

def get_match_index2(model1, model2, model1index):
    ind = get_match_index(model1, model2)
    ind.pop(len(ind)-1)
    ret = [model1index[i] for i in ind]
    return ret

def get_cluster_size(model, i):
    n = 0
    for j in range(len(model.labels_)):
        if model.labels_[j]==i:
            n+=1
    return n


models = []
for i in range(2,18):
    m = pkl.load(open(os.path.join(os.environ['DST_WRITE'],'kmeans_'+str(i).zfill(2)+'_night_01_0.pkl')))
    models.append(m)

#models = reversed(models)

hour = 360                                                                                                                                             
htimes = [str(i%24).zfill(2) + ":00" for i in range(19,30)]                                                                                          
ticks  = [hour*j for j in range(10+1)]                                                                                                           
font = {'size'   : 5}
matplotlib.rc('font', **font)

prevmodel = pkl.load(open(os.path.join(os.environ['DST_WRITE'],'kmeans_18_night_01_0.pkl')))
prevind = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]

c=1
for m in range(len(models)-1):
    #f.suptitle('day 01 '+str(model.n_clusters) + ' clusters')
    model1 = models[m]
    model2 = models[m+1]
    index = get_match_index(model2, model1)
    f, axarr = plt.subplots(3,6)
    for i in range(model2.n_clusters):
        j = index[i]
        #n = get_cluster_size(model, i)
        axarr[i%3, (i-(i%3))/3].yaxis.set_visible(False)
        axarr[i%3, (i-(i%3))/3].set_xticklabels(htimes,rotation=70)
        axarr[i%3, (i-(i%3))/3].set_xticks(ticks)
        axarr[i%3, (i-(i%3))/3].vlines(ticks,0,0.05,linestyles='dotted',lw=0.03)
        axarr[i%3, (i-(i%3))/3].plot(model2.cluster_centers_[i] - 0.001 ,color='black',lw=0.1)
        if j < model1.n_clusters:
            axarr[i%3, (i-(i%3))/3].plot(model1.cluster_centers_[j],color='blue',lw=0.1)
        axarr[i%3, (i-(i%3))/3].set_ylim([0.010,0.040])
        axarr[i%3, (i-(i%3))/3].set_xlim([0,3600])
        axarr[i%3, (i-(i%3))/3].legend(prop={'size':4})
    k = prevind[j+1]
    c+=1
    plt.savefig('ktestmatch'+str(c)+'.png',dpi=300)

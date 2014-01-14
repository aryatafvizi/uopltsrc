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


def get_match_index(model1, model2):
    num_clusters = model1.n_clusters
    cost_matrix = np.zeros((num_clusters,num_clusters))
    ln = np.min([len(model1.cluster_centers_[0]),len(model2.cluster_centers_[0])])
    for i in range(num_clusters):
        for j in range(num_clusters):
            cost_matrix[i,j] = np.linalg.norm(model1.cluster_centers_[i][0:ln] - model2.cluster_centers_[j][0:ln])
    m = Munkres()
    indeces = m.compute(cost_matrix)
    ret = [i[1] for i in indeces]
    return ret

def get_cluster_size(model, i):
    n = 0
    for j in range(len(model.labels_)):
        if model.labels_[j]==i:
            n+=1
    return n


raw_lcs = pkl.load(open(os.path.join(os.environ['DST_WRITE'],'lcs_night_01_lcs.pkl')))
full_kmeans = pkl.load(open(os.path.join(os.environ['DST_WRITE'],'kmeans_12_night_01_0.pkl')))

lcs_list = [raw_lcs[i,:,0]/np.linalg.norm(raw_lcs[i,:,0]) for i in range(4147)]
random.shuffle(lcs_list)

chunk1 = lcs_list[0:830]
chunk2 = lcs_list[830:1660]
chunk3 = lcs_list[1660:2490]
chunk4 = lcs_list[2490:3320]
chunk5 = lcs_list[3320:4147]

kmeans1 = KMeans(init='random',n_clusters=12,n_init=5,precompute_distances=True)                                                                                                   
kmeans1.fit(chunk1)
print 'chunk1'
pkl.dump(kmeans1,open(os.path.join(os.environ['DST_WRITE'],'day_01_five_chunk_01_kmeans.pkl'),'wb'))

kmeans2 = KMeans(init='random',n_clusters=12,n_init=5,precompute_distances=True)
kmeans2.fit(chunk2)
print 'chunk2'
pkl.dump(kmeans2,open(os.path.join(os.environ['DST_WRITE'],'day_01_five_chunk_02_kmeans.pkl'),'wb'))

kmeans3 = KMeans(init='random',n_clusters=12,n_init=5,precompute_distances=True)
kmeans3.fit(chunk3)
print 'chunk3'
pkl.dump(kmeans3,open(os.path.join(os.environ['DST_WRITE'],'day_01_five_chunk_03_kmeans.pkl'),'wb'))

kmeans4 = KMeans(init='random',n_clusters=12,n_init=5,precompute_distances=True)
kmeans4.fit(chunk4)
print 'chunk4'
pkl.dump(kmeans4,open(os.path.join(os.environ['DST_WRITE'],'day_01_five_chunk_04_kmeans.pkl'),'wb'))

kmeans5 = KMeans(init='random',n_clusters=12,n_init=5,precompute_distances=True)
kmeans5.fit(chunk5)
print 'chunk5'
pkl.dump(kmeans5,open(os.path.join(os.environ['DST_WRITE'],'day_01_five_chunk_05_kmeans.pkl'),'wb'))

                                                                                                                                                  
                                                                                                                                                                                   
hour = 360                                                                                                                                                                         
htimes = [str(i%24).zfill(2) + ":00" for i in range(19,30)]                                                                                                                        
ticks  = [hour*j for j in range(10+1)]                                                                                                                                             
font = {'size'   : 5}                                                                                                                                                              
                                                                                                                                                                                   
matplotlib.rc('font', **font)                                                                                                                                                      
    
f, axarr = plt.subplots(3,4)
c = 1              
for model in [kmeans1,kmeans2, kmeans3, kmeans4, kmeans5]: 
    index = get_match_index(full_kmeans,model)
    for i in range(12):
        j = index[i]
        n = get_cluster_size(model, j)                                                                                                             
        axarr[i%3, (i-(i%3))/3].yaxis.set_visible(False)                                                                                                                               
        axarr[i%3, (i-(i%3))/3].set_xticklabels(htimes,rotation=70)                                                                                                                    
        axarr[i%3, (i-(i%3))/3].set_xticks(ticks)                                                                                                                                      
        axarr[i%3, (i-(i%3))/3].vlines(ticks,0,0.05,linestyles='dotted',lw=0.1)                                                                                                        
        axarr[i%3, (i-(i%3))/3].plot(model.cluster_centers_[j],color=cm.Spectral(1.0*c/5),lw=0.1,label=('N = '+str(n)))                                                      
    #axarr[i%3, (i-(i%3))/3].plot(model.cluster_centers_[i]+sigmaCurve,color='blue',lw=0.05)                                                                                        
    #axarr[i%3, (i-(i%3))/3].plot(model.cluster_centers_[i]-sigmaCurve,color='blue',lw=0.05)                                                                                        
        axarr[i%3, (i-(i%3))/3].set_ylim([0.010,0.040])                                                                                                                                
        axarr[i%3, (i-(i%3))/3].set_xlim([0,3600])                                                                                                                                     
        axarr[i%3, (i-(i%3))/3].legend(prop={'size':4})                                                                                                                                
    c+=1
    
model = full_kmeans
for i in range(12):
        n = get_cluster_size(model, i)
        axarr[i%3, (i-(i%3))/3].yaxis.set_visible(False)
        axarr[i%3, (i-(i%3))/3].set_xticklabels(htimes,rotation=70)
        axarr[i%3, (i-(i%3))/3].set_xticks(ticks)
        axarr[i%3, (i-(i%3))/3].vlines(ticks,0,0.05,linestyles='dotted',lw=0.1)
        axarr[i%3, (i-(i%3))/3].plot(model.cluster_centers_[i],color='black',lw=0.2,label=('N = '+str(n)))
    #axarr[i%3, (i-(i%3))/3].plot(model.cluster_centers_[i]+sigmaCurve,color='blue',lw=0.05)                                                                          
    #axarr[i%3, (i-(i%3))/3].plot(model.cluster_centers_[i]-sigmaCurve,color='blue',lw=0.05)                                                                          
        axarr[i%3, (i-(i%3))/3].set_ylim([0.010,0.040])
        axarr[i%3, (i-(i%3))/3].set_xlim([0,3600])
        axarr[i%3, (i-(i%3))/3].legend(prop={'size':4})


plt.savefig('fivechunkmatched.png',dpi=300) 
    
    #plt.clf()



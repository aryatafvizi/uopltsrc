import dst13
import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib
from sklearn.cluster import KMeans
import pickle as pkl
import math
import random

def calc_sigma_curve(lcs,model,i):                                                                                                                                                                                   
    diffs = []                                                                                                                                                                              
    n = 0                                                                                                                                                                             
    for j in range(len(lcs)):                                                                                                                                                                 
        if model.labels_[j]==i:                                                                                                                          
            diffs.append(model.cluster_centers_[i] - lcs[j]/np.linalg.norm(lcs[j]))                                                                                                 
            n+=1 
    return n, np.array([np.std([d[j] for d in diffs]) for j in range(len(diffs[0]))])

def get_cluster_size(model, i):
    n = 0
    for j in range(len(model.labels_)):
        if model.labels_[j]==i:
            n+=1
    return n

def replace_with_random(lcs, fraction):
    num_to_replace = int(len(lcs)*fraction)
    for i in range(num_to_replace):
        lcs.pop(int(random.random()*len(lcs)))
    for i in range(num_to_replace):
        r = random.random()
        l = [1]*int(VEC_LENGTH*r)
        l += [0]*(VEC_LENGTH - len(l))
        l = np.array(l)
        l = l/max(np.linalg.norm(l),1e-6)
        lcs.append(l)
    return lcs

VEC_LENGTH = 3600

for frac in [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]:

    lcs = []

    for i in range(12):
        #r = (i+1.0)/13
        r = 0.5
        l = [1]*int(VEC_LENGTH*r)
        l += [0]*(VEC_LENGTH - len(l))
        l = np.array(l)
        l = l/max(np.linalg.norm(l),1e-6)
        for j in range(250):#add 250 copies to lcs
            lcs.append(l)

    print frac
    lcs = replace_with_random(lcs, frac)
    kmeans_ = KMeans(init='random',n_clusters=12,n_init=10,precompute_distances=False)                                                                        
    kmeans_.fit(lcs)                                                                                                                     

#pkl.dump(kmeans_,open(os.path.join(os.environ['DST_WRITE'],'synth_kmeans.pkl'),'wb'))
#pkl.dump(lcs,open(os.path.join(os.environ['DST_WRITE'],'synth_lcs.pkl'),'wb'))
#kmeans_ = pkl.load(open(os.path.join(os.environ['DST_WRITE'],'synth_kmeans.pkl')))

    f, axarr = plt.subplots(3, 4)                                                                                                                           
                                                                                                                                                        
    hour = 360                                                                                                                                              
    htimes = [str(i%24).zfill(2) + ":00" for i in range(19,30)]                                                                                             
    ticks  = [hour*j for j in range(10+1)]                                                                                                                  
    font = {'size'   : 5}                                                                                                                  
    matplotlib.rc('font', **font)                                                 
    model = kmeans_
                                                                                                            
    for i in range(12):                                                                                                                                     
        n = get_cluster_size(model, i)
        axarr[i%3, (i-(i%3))/3].yaxis.set_visible(False)                                                        
        axarr[i%3, (i-(i%3))/3].set_xticklabels(htimes,rotation=70)                                                                                         
        axarr[i%3, (i-(i%3))/3].set_xticks(ticks)                                                                                                           
        axarr[i%3, (i-(i%3))/3].vlines(ticks,-0.05,0.1,linestyles='dotted',lw=0.1)                                                                             
        axarr[i%3, (i-(i%3))/3].plot(model.cluster_centers_[i],color='black',lw=0.3,label=('cluster '+str(i))+'\n'+'N = '+str(n))                           
    #axarr[i%3, (i-(i%3))/3].plot(model.cluster_centers_[i]+sigmaCurve,color='blue',lw=0.15)                                                             
    #axarr[i%3, (i-(i%3))/3].plot(model.cluster_centers_[i]-sigmaCurve,color='blue',lw=0.15)                                                             
    #axarr[i%3, (i-(i%3))/3].set_ylim([0.010,0.040])
        axarr[i%3, (i-(i%3))/3].set_xlim([0,3600])
        axarr[i%3, (i-(i%3))/3].legend(prop={'size':4})
    f.suptitle(str(int(frac*100)).zfill(3)+' percent replaced by random step functions')    
    plt.savefig('synthcluster_err2_'+str(int(frac*100)).zfill(3)+'.png',dpi=300) 
    plt.clf()            

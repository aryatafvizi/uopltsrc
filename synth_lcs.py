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


VEC_LENGTH = 3600

lcs = []

on = []
for i in range(1000):
    l = [1]*VEC_LENGTH
    l = np.array(l)
    l = l/np.linalg.norm(l)
    on.append(l)

onoff = []
for i in range(1000):
    r = random.random()
    l = [1]*int(VEC_LENGTH*r)
    l += [0]*(VEC_LENGTH - len(l))
    l = np.array(l)
    l = l/max(np.linalg.norm(l),1e-6)
    onoff.append(l)

offon = []
for i in range(1000):
    r = random.random()
    l = [0]*int(VEC_LENGTH*r)
    l += [1]*(VEC_LENGTH - len(l))
    l = np.array(l)
    l = l/max(np.linalg.norm(l),1e-6)
    onoff.append(l)

onoffon = []
for i in range(1000):
    r1_ = random.random()
    r2_ = random.random()
    r1 = min(r1_,r2_)
    r2 = max(r1_,r2_)
    l = [1]*int(VEC_LENGTH*r1)
    l += [0]*int(VEC_LENGTH*(r2-r1))
    l += [1]*(VEC_LENGTH - len(l))
    l = np.array(l)
    l = l/max(np.linalg.norm(l),1e-6)
    onoffon.append(l)

offonoff = []
for i in range(1000):
    r1_ = random.random()
    r2_ = random.random()
    r1 = min(r1_,r2_)
    r2 = max(r1_,r2_)
    l = [0]*int(VEC_LENGTH*r1)
    l += [1]*int(VEC_LENGTH*(r2-r1))
    l += [0]*(VEC_LENGTH - len(l))
    l = np.array(l)
    l = l/max(np.linalg.norm(l),1e-6)
    offonoff.append(l)

lcs += on
lcs += onoff
lcs += offon
lcs += onoffon
lcs += offonoff

kmeans_ = KMeans(init='random',n_clusters=12,n_init=5,precompute_distances=True)                                                                        
kmeans_.fit(lcs)                                                                                                                     

pkl.dump(kmeans_,open(os.path.join(os.environ['DST_WRITE'],'synth_kmeans.pkl'),'wb'))
pkl.dump(lcs,open(os.path.join(os.environ['DST_WRITE'],'synth_lcs.pkl'),'wb'))
#kmeans_ = pkl.load(open(os.path.join(os.environ['DST_WRITE'],'synth_kmeans.pkl')))

f, axarr = plt.subplots(3, 4)                                                                                                                           
                                                                                                                                                        
hour = 360                                                                                                                                              
htimes = [str(i%24).zfill(2) + ":00" for i in range(19,30)]                                                                                             
ticks  = [hour*j for j in range(10+1)]                                                                                                                  
font = {'size'   : 5}                                                                                                                                   
                                                                                                                                                        
matplotlib.rc('font', **font)         
                                        
model = kmeans_
                                                                                                            
for i in range(12):                                                                                                                                     
    n, sigmaCurve = calc_sigma_curve(lcs, model, i)
    axarr[i%3, (i-(i%3))/3].yaxis.set_visible(False)                                                        
    axarr[i%3, (i-(i%3))/3].set_xticklabels(htimes,rotation=70)                                                                                         
    axarr[i%3, (i-(i%3))/3].set_xticks(ticks)                                                                                                           
    axarr[i%3, (i-(i%3))/3].vlines(ticks,-0.05,0.1,linestyles='dotted',lw=0.1)                                                                             
    axarr[i%3, (i-(i%3))/3].plot(model.cluster_centers_[i],color='black',lw=0.3,label=('cluster '+str(i))+'\n'+'N = '+str(n))                           
    axarr[i%3, (i-(i%3))/3].plot(model.cluster_centers_[i]+sigmaCurve,color='blue',lw=0.15)                                                             
    axarr[i%3, (i-(i%3))/3].plot(model.cluster_centers_[i]-sigmaCurve,color='blue',lw=0.15)                                                             
    #axarr[i%3, (i-(i%3))/3].set_ylim([0.010,0.040])
    axarr[i%3, (i-(i%3))/3].set_xlim([0,3600])
    axarr[i%3, (i-(i%3))/3].legend(prop={'size':4})
plt.savefig('synthcluster.png',dpi=300)                                                                                                                                                         

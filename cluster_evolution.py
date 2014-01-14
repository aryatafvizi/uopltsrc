import sys
import pickle as pkl
import os
import numpy as np
import dst13
from matplotlib import pyplot as plt
import matplotlib

def getAverage(lcs):
    summ = np.array(lcs[0])
    n=1.0
    for i in range(1,len(lcs)):
        summ += np.array(lcs[i])
        n+=1
    print n
    return summ/n

model1 = pkl.load (open(os.path.join(os.environ['DST_WRITE'],'kmeans_12_night_01_0.pkl')))

orderedLabels = []
for i in range(12):
    orderedLabels.append([j for j in range(len(model1.labels_)) if model1.labels_[j] == i])
                                                                                                                                                                                   
hour = 360                                                                                                                                                                         
htimes = [str(i%24).zfill(2) + ":00" for i in range(19,30)]                                                                                                                        
ticks  = [hour*j for j in range(10+1)]                                                                                                                                             
font = {'size'   : 5}                                                                                                                                                              
                                                                                                                                                                                   
matplotlib.rc('font', **font)                                                                                                                                                      


for k in range(12):                                                                                                 
    f, axarr = plt.subplots(4, 6)
    for i in range(22):                                                                                                                                                                
    #n, sigmaCurve = calc_sigma_curve(norm_res_highvar_stack, model, i)                                                                                                             
        lcs = pkl.load(open(os.path.join(os.environ['DST_WRITE'],'lcs_night_' + str(i).zfill(2) + '_lcs.pkl')))
        clusterLcs = [np.array(lcs[j,:,0])/np.linalg.norm(np.array(lcs[j,:,0])) for j in orderedLabels[k]]
    #axarr[i%4, (i-(i%4))/4].yaxis.set_visible(False)                                                                                                                               
        axarr[i%4, (i-(i%4))/4].set_xticklabels(htimes,rotation=70)                                                                                                                    
        axarr[i%4, (i-(i%4))/4].set_xticks(ticks)                                                                                                                                      
        axarr[i%4, (i-(i%4))/4].vlines(ticks,0,0.05,linestyles='dotted',lw=0.1)                                                                                                        
        axarr[i%4, (i-(i%4))/4].plot(getAverage(clusterLcs),color='black',lw=0.1,label=('Day '+str(i)))                                                      
    #axarr[i%4, (i-(i%4))/4].plot(model.cluster_centers_[i]+sigmaCurve,color='blue',lw=0.05)                                                                                        
    #axarr[i%4, (i-(i%4))/4].plot(model.cluster_centers_[i]-sigmaCurve,color='blue',lw=0.05)                                                                                        
        axarr[i%4, (i-(i%4))/4].set_ylim([0.010,0.040])                                                                                                                                
        axarr[i%4, (i-(i%4))/4].set_xlim([0,3600])                                                                                                                                     
        axarr[i%4, (i-(i%4))/4].legend(prop={'size':4})                                                                                                                                
    plt.savefig('cluster_evolution' + str(k).zfill(2) +'.png',dpi=300)  
    plt.close()

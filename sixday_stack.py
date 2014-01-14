import dst13
import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib
from sklearn.cluster import KMeans
import pickle as pkl
import math

"""
def calc_sigma_curve(lcs,model,i):
    diffs = []
    n = 0
    for j in range(len(lcs)):
        if model.labels_[j]==i:
            diffs.append(model.cluster_centers_[i] - lcs[j]/np.linalg.norm(lcs[j]))
            n+=1
    return n, np.array([np.std([d[j] for d in diffs]) for j in range(len(diffs[0]))])


labs = dst13.dst_window_labels.WindowLabels(hand=True, nopos=False) # window labels                                                                        
res_ind = [i for i in range(4147) if (labs.rvec[i]>1000)] #res window indeces

goodweekdays=[9,10,11,16,18,19]

norm_res_lcs_stack = []

for day in goodweekdays:                                                                                                                                   
    raw_lcs = pkl.load(open(os.path.join(os.environ['DST_WRITE'],'lcs_night_'+str(day).zfill(2)+'_lcs.pkl')))                                              
    norm_res_lcs = [raw_lcs[i,:,0]/np.linalg.norm(raw_lcs[i,:,0]) for i in res_ind] 
    norm_res_lcs_stack += norm_res_lcs

norm_res_highvar_stack = []

for i in range(len(norm_res_lcs_stack)/len(goodweekdays)):
    allhighvar = True
    for day in range(len(goodweekdays)):
        if np.var(norm_res_lcs_stack[i+day*len(norm_res_lcs_stack)/len(goodweekdays)]) < 0.000005:
            allhighvar = False
    if allhighvar == True:
        for day in range(len(goodweekdays)):
            l = norm_res_lcs_stack[i+day*len(norm_res_lcs_stack)/len(goodweekdays)].tolist()
            del l[3597:]
            norm_res_highvar_stack.append(l)


kmeans_ = KMeans(init='random',n_clusters=12,n_init=5,precompute_distances=True)
kmeans_.fit(norm_res_highvar_stack)

pkl.dump(kmeans_,open(os.path.join(os.environ['DST_WRITE'],'residential_highvar_stack_kmeans.pkl'),'wb'))

"""

def entropy(data):
    entropy = 0
    labels = list(set(data))
    for l in labels:
        pl = float(data.count(l))/len(data)
        if pl > 0:
            entropy += - pl*math.log(pl, 2)
    return entropy

model = pkl.load(open(os.path.join(os.environ['DST_WRITE'],'residential_highvar_stack_kmeans.pkl'))) 
"""
samewindowlabels=[]
for i in range(487):
    labels=[]
    for j in range(6):
        labels.append(model.labels_[i+487*j])
    
    samewindowlabels.append(labels)

ent = [entropy(l) for l in samewindowlabels]
plt.hist(ent,range=(0,3),bins=30)
plt.xlabel('label entropy')
plt.savefig('sixdaystacklabelsentropy.png')
"""

font = {'size'   : 5}                                                                                                                            
matplotlib.rc('font', **font)    
#f, axarr = plt.subplots(3,4)
labelfreq=[]
for i in range(6):
    l=[]
    for label in range(12):
        l.append(model.labels_[0+i*487:487+i*487].tolist().count(label))
    labelfreq.append(l)
    #plt.bar(range(12),l, color=cm.Spectral(1.0*i/6))
#axarr[label%3,(label-(label%3))/3].plot(l,color=cm.Spectral(1.0*11/12),label='Cluster ' + str(label))
    #axarr[label%3,(label-(label%3))/3].legend(prop={'size':4})

bot=[0]*12
for i in range(6):
    bars = labelfreq[i]
    plt.bar(range(12),bars,bottom=bot,color=cm.Spectral(1.*i/12))
    bot = [bot[j] + bars[j] for j in range(12)]
#plt.xlim((0,5))
plt.savefig('sixdayclustersizebar2.png',dpi=300)


"""
f, axarr = plt.subplots(3, 4)                      

hour = 360
htimes = [str(i%24).zfill(2) + ":00" for i in range(19,30)]
ticks  = [hour*j for j in range(10+1)]                 
font = {'size'   : 5}

matplotlib.rc('font', **font)                                                                                                                                                   

for i in range(12):                                                
    n, sigmaCurve = calc_sigma_curve(norm_res_highvar_stack, model, i)
    axarr[i%3, (i-(i%3))/3].yaxis.set_visible(False)                                                                                                       
    axarr[i%3, (i-(i%3))/3].set_xticklabels(htimes,rotation=70)
    axarr[i%3, (i-(i%3))/3].set_xticks(ticks)
    axarr[i%3, (i-(i%3))/3].vlines(ticks,0,0.05,linestyles='dotted',lw=0.1)
    axarr[i%3, (i-(i%3))/3].plot(model.cluster_centers_[i],color='black',lw=0.1,label=('cluster '+str(i))+'\n'+'N = '+str(n))
    axarr[i%3, (i-(i%3))/3].plot(model.cluster_centers_[i]+sigmaCurve,color='blue',lw=0.05)
    axarr[i%3, (i-(i%3))/3].plot(model.cluster_centers_[i]-sigmaCurve,color='blue',lw=0.05)
    axarr[i%3, (i-(i%3))/3].set_ylim([0.010,0.040])                                                                                                       
    axarr[i%3, (i-(i%3))/3].set_xlim([0,3600])                                                                                                             
    axarr[i%3, (i-(i%3))/3].legend(prop={'size':4})                                                                                                        
plt.savefig('sixday_stack.png',dpi=300)
"""

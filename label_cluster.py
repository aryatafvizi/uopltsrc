import match_cluster as mc
import pickle as pkl
import os
from matplotlib import pyplot as plt
import numpy as np
import math
from sklearn.cluster import KMeans
import window_labels as wl
import dst13


def entropy(data):
    entropy = 0
    labels = list(set(data))
    for l in labels:
        pl = float(data.count(l))/len(data)
        if pl > 0:
            entropy += - pl*math.log(pl, 2)
    return entropy

def genFeatureVector(data):
    out = []
    for i in range(12):
        out.append(data.count(i))
    return out


labs = dst13.dst_window_labels.WindowLabels(hand=True, nopos=False) # window labels 
goodweekdays=[9,10,11,16,18,19]

filters = [1]*4147
dayslabelindeces=[]
for day in goodweekdays:
    raw_lcs = pkl.load(open(os.path.join(os.environ['DST_WRITE'],'lcs_night_'+str(day).zfill(2)+'_lcs.pkl')))
    norm_lcs = [raw_lcs[i,:,0]/np.linalg.norm(raw_lcs[i,:,0]) for i in range(4147)]
    filters = [((labs.rvec[i]>1000) and (np.var(norm_lcs[i])>0.000005) and filters[i]) for i in range(4147)]
    thisdaylabelindeces = [i for i in range(4147) if ((labs.rvec[i]>1000) and (np.var(norm_lcs[i])>0.000005))]
    dayslabelindeces.append(thisdaylabelindeces)
print filters.count(True)
#wl.genWindowOverlay(filters,'common_windows.png')


indeces = mc.generateUnifiedIndeces()

labelsmatrix=[]
commonlabelsmatrix=[]
for i in range(4147):
    labels=[]
    for day in range(len(goodweekdays)):
        if(i in dayslabelindeces[day]):
               labels.append(indeces[day][dayslabelindeces[day].index(i)])
        else:
               labels.append(-1)
    labelsmatrix.append(labels)
    if (labels.count(-1)==0):
        commonlabelsmatrix.append(labels)
        if (entropy(labels)<0.5):
            print labels



#plt.hist(np.array([entropy(commonlabelsmatrix[i]) for i in range(len(commonlabelsmatrix))]))
#plt.savefig('common_res_high_var_label_entropy.png')

"""
vectors=[]
for i in range(4147):
    vectors.append(genFeatureVector(indeces[i,:].tolist()))

kmeans = KMeans(n_clusters=12, n_init=10)
kmeans.fit(vectors)


for j in range(5):
    clus_indeces = [1*(kmeans.labels_[i]==j) for i in range(4147)]
    #wl.genWindowOverlay(clus_indeces,'labelclusteroverlay_'+str(j)+'.png')


f, axarr = plt.subplots(3, 4)                                                                                                                          
                                                                                                                                                       
for i in range(12):                                      
    axarr[i%3, (i-(i%3))/3 - 1].bar(range(12),kmeans.cluster_centers_[i],1,label=('N = ' + str(kmeans.labels_.tolist().count(i)) + '\n' + 'cluster  '+str(i)))
    #axarr[i%3, (i-(i%3))/3 - 1].xaxis.set_visible(False)                                                                                               
    #axarr[i%3, (i-(i%3))/3 - 1].yaxis.set_visible(False)                                                                                               
    #axarr[i%3, (i-(i%3))/3 - 1].set_ylim([0.010,0.040])                                                                                               
    #axarr[i%3, (i-(i%3))/3 - 1].set_xlim([0,3600])                                                                                                     
    axarr[i%3, (i-(i%3))/3 - 1].legend(prop={'size':4})  

plt.savefig('res_goodday_labelclusters.png',dpi=300)

"""
"""
n=0
for i in range(4147):
    if entropy(indeces[i,:].tolist()) < 2.0:
        print indeces[i,:]
        n+=1
print n
"""
"""
ent = [entropy(indeces[i,:].tolist()) for i in range(4147)]
plt.hist(ent,bins=20)
plt.xlabel('entropy')
plt.savefig('entropyhist.png')
"""

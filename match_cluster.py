import sys
import pickle as pkl
import os
from munkres import Munkres, print_matrix
import numpy as np
import dst13
from matplotlib import pyplot as plt
from matplotlib import cm

def getDayIndexes(day1model, otherdaymodel):
    cost_matrix = np.zeros((12,12))
    ln = np.min([len(day1model.cluster_centers_[0]),len(otherdaymodel.cluster_centers_[0])])
    for i in range(0,12):
        for j in range(0,12):
            cost_matrix[i,j]=np.linalg.norm(day1model.cluster_centers_[i][0:ln] - otherdaymodel.cluster_centers_[j][0:ln])
            #cost_matrix[i,j]=2.0 - np.corrcoef(day1model.cluster_centers_[i][0:ln], otherdaymodel.cluster_centers_[j][0:ln])[0][1]       
    m = Munkres()
    indexes = m.compute(cost_matrix)
    return indexes

def generateUnifiedIndeces():
    unifiedlabels=[]
    #lcs = pkl.load (open(os.path.join(os.environ['DST_WRITE'],'lcs_night_00_lcs.pkl')))
    model1 = pkl.load (open(os.path.join(os.environ['DST_WRITE'],'lcs_night_09_residential_high_var_kmeans.pkl')))
    unifiedlabels.append(model1.labels_)
    for day in range(1,len(goodweekdays)):
        model = pkl.load (open(os.path.join(os.environ['DST_WRITE'],'lcs_night_'+str(goodweekdays[day]).zfill(2)+'_residential_high_var_kmeans.pkl')))  
        indexes = getDayIndexes(model,model1)
        l=[indexes[i][1] for i in model.labels_]
        unifiedlabels.append(l)
    return unifiedlabels



goodweekdays=[9,10,11,16,18,19]
"""
model1 = pkl.load (open(os.path.join(os.environ['DST_WRITE'],'lcs_night_09_residential_high_var_kmeans.pkl')))
f, axarr = plt.subplots(3, 4)

for i in range(12):
    
    axarr[i%3, (i-(i%3))/3].plot(model1.cluster_centers_[i],color='black',lw=0.1,label=('cluster '+str(i)))
    for day in range(len(goodweekdays)):
        dayModel = pkl.load(open(os.path.join(os.environ['DST_WRITE'], 'lcs_night_'+str(goodweekdays[day]).zfill(2)+'_residential_high_var_kmeans.pkl')))
        indexes = getDayIndexes(model1, dayModel)
        axarr[i%3, (i-(i%3))/3].plot(dayModel.cluster_centers_[indexes[i][1]]+(day)*0.005,color=cm.winter(1.0*day/7),lw=0.1)
    axarr[i%3, (i-(i%3))/3].xaxis.set_visible(False)
    axarr[i%3, (i-(i%3))/3].yaxis.set_visible(False)
    #axarr[i%3, (i-(i%3))/3].set_ylim([0.010,0.040])
    axarr[i%3, (i-(i%3))/3].set_xlim([0,3600])
    axarr[i%3, (i-(i%3))/3].legend(prop={'size':4})

plt.savefig('gooddays_high_var_residential_match.png',dpi=1000)

"""

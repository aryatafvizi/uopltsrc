import pickle as pkl
import numpy
import dst13
import sys
import os
import math
from sklearn import metrics
from multiprocessing import Process
import csv
from matplotlib import pyplot as plt


"""
data=[]
reader = csv.reader(open('test2out.txt'), delimiter='\t')
for line in reader:
    data.append(line[0].split())

err=[]
avg=[]
for k in range(2,20):
    nums=[]
    for line in data:
        if int(line[1]) == k:
            nums.append(float(line[2]))
    avg.append(numpy.average(nums))
    err.append(numpy.std(nums)/math.sqrt(len(nums)))
    plt.scatter([k]*len(nums),nums,color='gray',alpha=0.5)
plt.errorbar(range(2,20),avg,err,fmt='o')
plt.xlabel('k')
plt.ylabel('DB Index')
plt.savefig('err.png')
"""


def getKModels(k):
    models = []
    for day in range(21):
        models.append(pkl.load(open(os.path.join(os.environ['DST_WRITE'],'kmeans_coarse_night_MODEL_'+str(day)+'_'+str(k)+'.pkl'))))
    return models

def getDayModels(day):
    models = []
    for k in range(2,20):
        models.append(pkl.load(open(os.path.join(os.environ['DST_WRITE'],'kmeans_coarse_night_MODEL_'+str(day)+'_'+str(k)+'.pkl'))))
    return models

#i is the cluster index
def calcInterVar(lcs,model,i):
    distance = 0
    n = 0
    for j in range(len(lcs)):
        if model.labels_[j]==i:
            distance+=numpy.linalg.norm(model.cluster_centers_[i] - (lcs[j]/numpy.linalg.norm(lcs[j])))**2
            n+=1
    return distance/(1.0*n)
        
def calcIntraVar(model):
    clusterAverage=numpy.zeros(len(model.cluster_centers_[0]))
    n=0
    for c in model.cluster_centers_:
        numpy.add(clusterAverage, c)
        n+=1
    clusterAverage = clusterAverage/n
    var=0
    for c in model.cluster_centers_:
        var+=numpy.linalg.norm(c - clusterAverage)**2
    return var/n

def calcDBScore(lcs,model):
    DBScore=0
    for i in range(len(model.cluster_centers_)):
        R=0
        for j in range(len(model.cluster_centers_)):
            if i != j:
                Rj = (math.sqrt(calcInterVar(lcs,model,i)) + math.sqrt(calcInterVar(lcs,model,j)))/(numpy.linalg.norm(model.cluster_centers_[i] - model.cluster_centers_[j]))
                if Rj > R:
                    R = Rj
        DBScore+=R
    DBScore = DBScore/len(model.cluster_centers_)
    #print DBScore, len(model.cluster_centers_)
    return DBScore
            

#def calcSilhouette(lcs,model):

"""
lcs = pkl.load (open(os.path.join(os.environ['DST_WRITE'],'lcs_night_00_lcs.pkl')))
model = pkl.load (open(os.path.join(os.environ['DST_WRITE'],'kmeans_12_night_00_0.pkl'))) 

varhtmp=numpy.zeros((12,12))
for i in range(0,12):
    for j in range(0,12):
        if i==j:
            varhtmp[i,i]=math.sqrt(calcInterVar(lcs[:,:,0],model,i))
        else:
            varhtmp[i,j]=numpy.linalg.norm(model.cluster_centers_[i] - model.cluster_centers_[j])
plt.pcolor(varhtmp,cmap='winter')
plt.colorbar()
plt.savefig('varhtmp.png')
"""
"""
for day in range(0,22):
    lcs = pkl.load (open(os.path.join(os.environ['DST_WRITE'],'lcs_night_'+str(day).zfill(2)+'_lcs.pkl')))
    for k in range(2,20):
        model = pkl.load (open(os.path.join(os.environ['DST_WRITE'],'kmeans_'+str(k).zfill(2)+'_night_'+str(day).zfill(2)+'_0.pkl')))
        print day, k, calcDBScore(lcs[:,:,0],model)

"""

"""
model = pkl.load(open(os.path.join(os.environ['DST_WRITE'],'kmeans_coarse_night_MODEL_15_5.pkl')))
print calcInterVar(lcs,model,3)
print calcIntraVar(model)
print calcDBScore(lcs,model)
"""
"""
#models = getKModels(12)
processes=[]
for day in range(1,21):
    models = getDayModels(8)
    for m in models:
        #print k, calcDBScore(lcs,m)
        p = Process(target=calcDBScore,args=(lcs,m))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
"""
"""
clusterdicts=[]
#starting from day 10 because some of the previous days have shorter centroids. should investigate why
for day in range(10,21):
    model = models[day]
    centroids_0 = models[0].cluster_centers_
    centroids_day = models[day].cluster_centers_
    print centroids_0.shape[0]
    print centroids_day.shape[0]
    for c0 in range(centroids_0.shape[0]):
        distances = []
        for cd in range(centroids_0.shape[0]): 
            distances.append(numpy.linalg.norm(centroids_0[c0,:] - centroids_0[cd,:]))
            
            print numpy.linalg.norm(centroids_0[c0,:] - centroids_0[cd,:])

"""


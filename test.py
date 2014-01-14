import dst13
from sklearn import metrics
from multiprocessing import Process
import pickle as pkl
import os
import sys

def kmeans(k,band):
    dst13.dst_kmeans.kmeans(n_clusters=k,band=band)


processes=[]
for k in range(2,20):
    p = Process(target=kmeans,args=(k, 0))
    p.start()
    processes.append(p)
for p in processes:
    p.join()

"""
def silhouette(k,start,end,day):
    lcs, km = dst13.dst_kmeans_coarse.kmeans_coarse(start,end,timerate=25,spacerate=1,k=k,verbose=False)
    lcswfile = os.path.join(os.environ['DST_WRITE'],'kmeans_coarse_night_LCS_' + str(day) + '_' + str(k) + '.pkl')
    kmwfile = os.path.join(os.environ['DST_WRITE'],'kmeans_coarse_night_MODEL_' + str(day) + '_' + str(k) + '.pkl')
    fopen = open(lcswfile,'wb')
    pkl.dump(lcs,fopen)
    fopen.close()
    fopen = open(kmwfile,'wb')
    pkl.dump(km,fopen)
    fopen.close()
    print (k,start, metrics.silhouette_score(lcs, km.labels_, metric='euclidean'))


processes = []
starts, ends = dst13.dst_night_times.night_times()
for k in range(2,20):
    for day in range(len(starts)):
        p = Process(target=silhouette,args=(k,starts[day],ends[day],day))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    
"""

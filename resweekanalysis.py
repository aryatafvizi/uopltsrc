import dst13
import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
import pickle as pkl


def calc_sigma_curve(lcs,model,i):
    diffs = []
    n = 0
    for j in range(len(lcs)):
        if model.labels_[j]==i:
            diffs.append(model.cluster_centers_[i] - lcs[j]/np.linalg.norm(lcs[j]))
            n+=1
    return n, np.array([np.std([d[j] for d in diffs]) for j in range(len(diffs[0]))])




labs = dst13.dst_window_labels.WindowLabels(hand=True, nopos=False) # window labels                                                          
res_ind = [i for i in range(4147) if (labs.rvec[i]>1000)]

goodweekdays=[9,10,11,16,18,19]


"""
#create variance histograms for good weekdays
for day in goodweekdays:
    raw_lcs = pkl.load(open(os.path.join(os.environ['DST_WRITE'],'lcs_night_'+str(day).zfill(2)+'_lcs.pkl')))
    norm_res_lcs = [raw_lcs[i,:,0]/np.linalg.norm(raw_lcs[i,:,0]) for i in res_ind]
    var = [np.var(l) for l in norm_res_lcs]
    plt.hist(var,bins=100,range=(0,0.0001))
    plt.savefig('res_norm_lcs_var_hist_'+str(day).zfill(2)+'.png')
    plt.close()
"""

###this section of the code clusters the residential lightcurves for each day into 12 clusters
for day in goodweekdays:

    raw_lcs = pkl.load(open(os.path.join(os.environ['DST_WRITE'],'lcs_night_'+str(day).zfill(2)+'_lcs.pkl')))
    norm_res_lcs = [raw_lcs[i,:,0]/np.linalg.norm(raw_lcs[i,:,0]) for i in res_ind]
    high_var = [l for l in norm_res_lcs if np.var(l)>0.000005]
    print len(high_var)
    print 'CLUSTERING...'

    kmeans_ = KMeans(init='random',n_clusters=12,n_init=10)
    kmeans_.fit(high_var)

    pkl.dump(kmeans_,open(os.path.join(os.environ['DST_WRITE'],'lcs_night_'+str(day).zfill(2)+'_residential_high_var_kmeans.pkl'),'wb'))

    f, axarr = plt.subplots(3, 4)
    for i in range(12):
        n, sigmaCurve = calc_sigma_curve(high_var, kmeans_, i)
        for j in range(len(high_var)):
            if j%(int(n/5)) == 0:
                if kmeans_.labels_[j]==i:
                    axarr[i%3, (i-(i%3))/3].plot(high_var[j]/np.linalg.norm(high_var[j]),color='gray',alpha=0.2,lw=0.1)
        axarr[i%3, (i-(i%3))/3].plot(kmeans_.cluster_centers_[i],color='black',lw=0.1, label=('N = '+str(n)+'\n'+'cluster '+str(i)))
        axarr[i%3, (i-(i%3))/3].plot(kmeans_.cluster_centers_[i] + sigmaCurve,color='red',lw=0.1)
        axarr[i%3, (i-(i%3))/3].plot(kmeans_.cluster_centers_[i] - sigmaCurve,color='red',lw=0.1)
        axarr[i%3, (i-(i%3))/3].xaxis.set_visible(False)
        axarr[i%3, (i-(i%3))/3].yaxis.set_visible(False)
        axarr[i%3, (i-(i%3))/3].set_ylim([0.010,0.040])
        axarr[i%3, (i-(i%3))/3].set_xlim([0,3600])
        axarr[i%3, (i-(i%3))/3].legend(prop={'size':4})
    plt.suptitle('Day ' + str(day))
    plt.savefig('res_weekday_high_var'+str(day).zfill(2)+'.png',dpi=1000)

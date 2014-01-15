import pickle as pkl
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import dst13

#l is lightcurve, t is list of transitions, find transition that has the highest delta
#delta defined as the average of everything before the transition minus the average of everything after the transition
def getbiggesttransition(l,t):
    deltas = []
    t2 = [transition for transition in t if transition < 0]
    for i in range(len(t2)):
        delta = np.mean(l[0:abs(t2[i])]) - np.mean(l[abs(t2[i]):len(l)+1]) 
        deltas.append(delta)
    if len(deltas)>0:
        maxindex = np.argmax(deltas)
        return t2[maxindex]
    else:
        return -1

tr = []
for i in range(1,10):
    t = pkl.load(open('/users/arya/Desktop/UO/ind_onoff_night_01_'+str(i)+'.pkl'))
    tr += t


lcs = pkl.load(open('/users/arya/Desktop/UO/lcs_night_01_lcs.pkl'))
km = pkl.load(open('/users/arya/Desktop/UO/kmeans_12_night_01_0.pkl'))
lb = km.labels_

labs = dst13.dst_window_labels.WindowLabels(hand=True, nopos=False) # window labels                                                                              

#get subseet of lightcurves belonging to plateua clusters
plateau_clusters = [1,5,6,8,10]

lres = [lcs[i,:,0] for i in range(4147) if ((lb[i] in plateau_clusters) and (labs.rvec[i]>1000))]
tres = [tr[i].tolist() for i in range(4147) if ((lb[i] in plateau_clusters) and (labs.rvec[i]>1000))]
l = [lcs[i,:,0] for i in range(4147) if ((lb[i] in plateau_clusters))]
t = [tr[i].tolist() for i in range(4147) if ((lb[i] in plateau_clusters))]

bigt = [abs(getbiggesttransition(l[i],t[i])) for i in range(len(l))]
bigtres = [abs(getbiggesttransition(lres[i],tres[i])) for i in range(len(lres))]

print len(lres)
print len(l)

plt.hist(bigt,bins=50,range=(0,3600),color=cm.Spectral(0.3),label='All Plateua Windows (N = 806)')
plt.hist(bigtres,bins=50,range=(0,3600),color=cm.Spectral(0.6),label='Residential Plateua Windows (N = 401)')

hour = 360                                                                         
htimes = [str(i%24).zfill(2) + ":00" for i in range(19,30)]                                                         
ticks  = [hour*j for j in range(10+1)]

plt.xticks(ticks, htimes, rotation=70)
plt.xlim(0,3600)
#plt.imshow(l,cmap='gray')
#plt.scatter(bigt,range(len(bigt)))
plt.legend()
plt.suptitle('Distribution of Plateua Off Times')
#plt.show()
plt.savefig('plateua_offtimes.png', dpi=300)


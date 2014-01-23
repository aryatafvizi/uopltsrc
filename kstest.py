import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import random
import numpy as np
from scipy.stats import ks_2samp, chisquare

def offdistpdf(PARAMS):
    mu1 = PARAMS[0]
    sigma1 = PARAMS[1]
    intercept = PARAMS[2]
    normfactor1 = PARAMS[3]
    normfactor2 = 0#10e3
    mu2 = 0.3
    sigma2 = 0.01

    length=38
    dist = [0]*length
    for x in range(length):
        dist[x] += intercept - 1.0*intercept*x/length
        dist[x] += (normfactor1/((sigma1*length)*math.sqrt(2*math.pi)))*math.exp(-(x-(mu1*length))**2/(2*(sigma1*length)**2))
        dist[x] += (normfactor2/((sigma2*length)*math.sqrt(2*math.pi)))*math.exp(-(x-(mu2*length))**2/(2*(sigma2*length)**2))

    normalization = sum(dist)
    ret = [v/normalization for v in dist]
    return ret


def offdistcdf(PARAMS):
    pdf = offdistpdf(PARAMS)
    ret = [sum(pdf[0:i]) for i in range(len(pdf))]
    return ret


def sqdistance(data, fit):
    return sum([(fit[i]-data[i])**2 for i in range(len(data))])

def distparam(vec):
    fit = offdistpdf(vec)
    return sqdistance(normdata, fit)

def drawfromcdf(cdf):
    r = random.random()
    for i in range(len(cdf)):
        if cdf[i]>r:
            return i-1
    return len(cdf)

def distfromrandomdraw(cdf,size):
    draws = []
    for i in range(size):
        draws.append(drawfromcdf(cdf))
    dist =  np.histogram(draws,bins=len(cdf),range=(0,len(cdf)))[0]
    dist = 1.0*dist/sum(dist)
    return dist.tolist()


data = [251, 204, 188, 190, 197, 212, 188, 240, 214, 226, 233, 307, 323, 370, 296, 356, 376, 317, 320, 263, 212, 139, 179, 145, 138,  80,  86,  93,  88,  53,  41,  38,  61,  37,  32, 29,  59,  36]
datasum = sum(data)
normdata = [1.0*d/datasum for d in data]
datacdf = [sum(normdata[0:i]) for i in range(len(normdata))]


OPTIMAL_PARAMS = [  3.31121108e-01,   2.49757095e-01,   1.00000065e+02,   1.00000000e+05]

optimalcdf = offdistcdf(OPTIMAL_PARAMS)
res = minimize(distparam, [0.5,0.1,100,1e5]) 

print res.x

normexpected = offdistpdf(OPTIMAL_PARAMS)
normexpected = [normexpected[i]*sum(data) for i in range(len(normexpected))]

cs = chisquare(np.array(data), np.array(normexpected), 4)
print cs
"""
scores = []
for i in range(1000):
    scores.append(ks_2samp(distfromrandomdraw(optimalcdf,6817),normdata)[1])

plt.hist(scores,bins=100)
plt.show()
"""

expected = offdistpdf(OPTIMAL_PARAMS)
expected = [expected[i]*sum(data) for i in range(len(expected))]
plt.plot(expected)
plt.plot(data)
#plt.plot(distfromrandomdraw(optimalcdf,100))
#plt.plot(distfromrandomdraw(optimalcdf,1000))
#plt.plot(distfromrandomdraw(optimalcdf,100000))
plt.show()

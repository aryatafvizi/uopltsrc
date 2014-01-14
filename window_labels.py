import dst13
import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib.cm as cm

def read_raw(fname, path=os.environ['DST_WRITE'], nrow=2160, ncol=4096):

    """ Read in the raw file format to a numpy array """

    # set the file name, read, and return
    infile = os.path.join(path,fname)
    img    = np.fromfile(open(infile,'rb'), dtype=np.uint8, count=-1)

    if img.size>0:
        return img.reshape(nrow,ncol,3)[:,:,::-1]
    else:
        print("DST_IO: ERROR - EMPTY FILE IN {0}!!!".format(path))
        print("DST_IO:   {0}".format(fname))
        return np.zeros([nrow,ncol,3])


def genWindowOverlay(indeces,outname):
    labs       = dst13.dst_window_labels.WindowLabels(hand=True, nopos=True) # window labels
    nrow, ncol = labs.labels.shape
    maps       = np.zeros([nrow,ncol],dtype=np.uint8)

    iwin = [i for i,j in enumerate(indeces) if j==1]

    for ii in iwin:
        a = (labs.labels==(ii+1))
        maps += (labs.labels==(ii+1))
        print ii




    bkg = np.ma.array(read_raw('oct08_2013-10-25-175504-181179.raw',
                               os.path.join(os.environ['DST_DATA'],
                                            '11/15/16.23.43')
                               )[20:-20,20:-20,:].astype(np.float).mean(2))

    bkg.mask = maps>0

    mn = bkg.min() + 0.2*np.abs(bkg.min())
    mx = bkg.max() - 0.2*np.abs(bkg.max())
    wincolor='#0099FF'
    color = cm.get_cmap('bone')
    color.set_bad(color=wincolor)

    plt.figure()

    plt.imshow(bkg,cmap=color,clim=[mn,mx])
    plt.axis('off')
    plt.savefig(outname)



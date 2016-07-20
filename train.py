from data_handler import data_handler
from model import user2vec
import pdb
import numpy as np
#n = data.shape[0]
data = data_handler("rating_with_timestamp.mat", "trust.mat", "rating_with_timestamp.mat")
data.load_matrices()
n = data.n
i = data.i
h = 100
d = 100
u2v = user2vec(n, h,d,i)
u2v.model1()

# Training for batch mode
def training_batch(batch_size):
    print "in batch mode"
    for i in xrange(0, data.shape[0], batch_size):
        X = data[i:(i+batch_size), 0:2]
        y = data[i:(i+batch_size), 2]
        #print[i:(i+batch_size)]
        #print data[i:(i+batch_size)]
        #pdb.set_trace()
        #print node2vec.debug(X)
        try:
            print node2vec.gd_batch(X, y)
        except Exception as e:
            print "in exception"
            print str(e)
            #U,V = node2vec.debug(X)
            #L = node2vec.debug1(X,y)
            #print L
            pdb.set_trace()

# Training for single example mode
def training():
    row = np.arange(n)
    np.random.shuffle(row)
    print "training u-i model"

    for r in row:
        col = data.T1[r, :].nonzero()[1]
        np.random.shuffle(col)
        for c in col:
            cost = u2v.sgd_ui([[r,c],data.T1[r,c]/float(5)])
            print cost
        pdb.set_trace()

    print "training u-u model"



if __name__ == "__main__":
    training()
    pdb.set_trace()

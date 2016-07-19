from data_handler import data_handler
from model import social2vec
import pdb
import numpy as np
#n = data.shape[0]
data = data_handler("../../trust-aware-recom/data/rating_with_timestamp.mat", "../../trust-aware-recom/data/trust.mat", "../../trust-aware-recom/data/rating_with_timestamp.mat")
data.load_matrices()
n = data.n

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



if __name__ == "__main__":
    training_batch(16)
    pdb.set_trace()

from data_handler import data_handler
from model1 import user2vec
import pdb
import numpy as np
#n = data.shape[0]
data = data_handler("rating_with_timestamp.mat", "trust.mat", "rating_with_timestamp.mat")
data.load_matrices()
n = data.n
i = data.i
h = 100
d = 50
n_epochs = 5
u2v = user2vec(n, h,d,i)
#u2v.model1()
u2v.model_batch_uu()
u2v.model_batch_ui()

# Training for batch mode
def training_batch(batch_size): 
    # U-U part
    ind = 0
    f = open('Trust.txt','r')
    batch = []

    for epoch in xrange(n_epochs):
        batch = []
        print "Initiating epoch %d"%epoch
        with open('train.txt', 'r') as f:
            for line in f:
                data1 = line.strip()
                data1 = map(lambda x:float(x), data1.split())
                batch.append(data1)
                if (ind + 1) % batch_size == 0:
                   batch = np.array(batch).astype(np.int32)
                   #pdb.set_trace()
                   try:
                       cost = u2v.uu_batch(batch[:, 0:2], batch[:, 2])
                       #cost1 = u2v.debug(batch[:, :2])
                       #cost2 = u2v.debug1(batch[:, :2], batch[:, 2])
                       print cost
                       print "\n"
                       #pdb.set_trace()
                       #if max(batch[:, 0]) >= n or max(batch[:, 1]) >= n:
                       #    print " in buggy region"
                       #    pdb.set_trace()
                       #assert max(batch[:, 0]) > n and max(batch[:, 1]) > n
                       batch = []
                   except Exception as e:
                       print str(e)
                       print "in exception, check batch"
                       #pdb.set_trace()
                ind += 1



        print "UU training completed"
        ## U-I training
        #m = len(data.T1)
        ##pdb.set_trace()
        #print m
        #for i in xrange(0, m, batch_size):
        #    batch = data.T1[i:(i + batch_size), :]
        #    #pdb.set_trace()
        #    U = batch[:, :2]
        #    Y = batch[:, 2]
        #    cost = u2v.ui_batch(U, Y)
        #    #cost = u2v.debug1(U)
        #    print cost
        #    #print u2v.debug1()
        
        # Save the model to disk
    u2v.get_params()
        

# Training for single example mode
def training():
    row = np.arange(n)
    np.random.shuffle(row)
    print "training u-i model"

    for r in row:
        col = data.T1[r, :].nonzero()[1]
        np.random.shuffle(col)
        for c in col:
            cost = u2v.sgd_ui([r,c],data.T1[r,c])
            print cost
           #print u2v.debug1([r,c])
           #print u2v.debug2([r,c], data.T1[r,c]/float(5))
        #pdb.set_trace()

    print "training u-u model"
    for r in row:
        col = data.T[r,:]
    #    print col
        for c in col:
            #print u2v.debug1([r,c], data.T[r,c])
            print u2v.sgd_uu([r,c], data.T[r,c])
    #    pdb.set_trace()

    print "training u-u model"



if __name__ == "__main__":
    #training()
    training_batch(256)
    print "Training complete,"
    #pdb.set_trace()

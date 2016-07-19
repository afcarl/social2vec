from theano import tensor as T
import theano
import numpy as np
#from preprocess import data
from theano.compile.nanguardmode import NanGuardMode
import pdb

class user2vec(object):
    def __init__(self, n_user, d, h, n_item):
        self.n_user = n_user
        self.d = d
        self.h = h
        self.n_item = n_item
        # Shared parameter (user embedding vector)
        self.Wu = theano.shared(np.random.uniform(low = - np.sqrt(6.0/float(n_user + d)),\
                                   high =  np.sqrt(6.0/float(n_user + d)),\
                                   size=(n_user,d)).astype(theano.config.floatX))
        # Item embedding matrix
        self.Wi = theano.shared(np.random.uniform(low = - np.sqrt(6.0/float(n_item + d)),\
                                   high =  np.sqrt(6.0/float(n_item + d)),\
                                   size=(n_item ,d)).astype(theano.config.floatX))

        self.W1 = self.Wu
        self.W2 = self.Wi

        # Paramters for user-user model
        self.Wm1 = theano.shared(np.random.uniform(low=-np.sqrt(6.0/float(h + d)),
                                    high = np.sqrt(6.0/float(h+d)),
                                    size=(h,d)).astype(theano.config.floatX))
        self.Wp1 = theano.shared(np.random.uniform(low= - np.sqrt(6.0/float(h + d)),
                                    high = np.sqrt(6.0/float(h+d)),
                                    size=(h,d)).astype(theano.config.floatX))
        self.b11 = theano.shared(np.zeros((h), dtype=theano.config.floatX))\
                                #, broadcastable=(False,True))
        self.b21 = theano.shared(np.zeros((2), dtype=theano.config.floatX)),
                                #broadcastable=(False, True))
        self.U1 = theano.shared(np.random.uniform(low= - np.sqrt(6.0/float(2 + h)),\
                                              high = np.sqrt(6.0/float(2 + h)),
                                              size=(2,h)).astype(theano.config.floatX))

        # Parameters for user-item model
        self.Wm2 = theano.shared(np.random.uniform(low=-np.sqrt(6.0/float(h + d)),
                                    high = np.sqrt(6.0/float(h+d)),
                                    size=(h,d)).astype(theano.config.floatX))
        self.Wp2 = theano.shared(np.random.uniform(low= - np.sqrt(6.0/float(h + d)),
                                    high = np.sqrt(6.0/float(h+d)),
                                    size=(h,d)).astype(theano.config.floatX))
        self.b12 = theano.shared(np.zeros((h), dtype=theano.config.floatX))\
                                #, broadcastable=(False,True))
        self.b22 = theano.shared(np.zeros((2), dtype=theano.config.floatX)),
                                #broadcastable=(False, True))
        self.U2 = theano.shared(np.random.uniform(low= - np.sqrt(6.0/float(2 + h)),\
                                              high = np.sqrt(6.0/float(2 + h)),
                                              size=(1,h)).astype(theano.config.floatX))

        self.params = [self.Wm1, self.Wp1, self.b11, self.b21, self.U1,self.Wm2, self.Wp2, self.b12, self.b22, self.U2]

    def model_batch(self, lr=0.01):
        # theano matrix storing node embeddings
        X = T.imatrix()
        # Target labels for input
        y = T.ivector()
        # Extract the word vectors corresponding to inputs
        U = self.W[X[:,0],:]
        V = self.W[X[:,1],:]
        self.debug = theano.function([X], [U,V])
        hLm = U * V
        hLp = abs(U - V)
        hL = T.tanh(T.dot(self.Wm, hLm.T) + T.dot(self.Wp, hLp.T) + self.b1)
        #param = [U , V]
        #params.extend(self.params)
        # Likelihood
        #l = T.nnet.softmax(T.dot(self.U, hL) + self.b2)[y, T.arange(y.shape[0])]
        l = T.nnet.softmax(T.dot(self.U, hL) + self.b21)
        cost = T.sum(T.nnet.binary_crossentropy(l, y))
        #cost = - T.sum(T.log(l + eps))
        #self.debug1 = theano.function([X,y], l)
        grad1 = T.grad(cost, [U,V])
        grads = T.grad(cost, self.params)
        #updates1 = [(self.W1, T.inc_subtensor(self.W[X[:, 0]], grads[0]))]
        #updates2 = [(self.W, T.inc_subtensor(self.W1[X[:, 1]], grads[1]))]
        self.W1 = T.set_subtensor(self.W1[X[:,0], :], self.W1[X[:,0], :] - lr * grad1[0])
        self.W1 = T.set_subtensor(self.W1[X[:,1], :], self.W1[X[:,1], :] - lr * grad1[1])
        updates1 = [(self.W, self.W1)]
        updates3 = [(param, param - lr * grad) for (param, grad) in zip(self.params, grads)]
        updates = updates1  + updates3
        self.gd_batch = theano.function([X,y], cost, updates=updates, mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))

    def model(self, lr=0.01):
        # Tuple for user-user-item
        uu = T.ivector()
        # Tuple for user-item
        #ui = T.ivector()
        yu = T.vector()
        #yi = T.scalar()

        u = self.Wu[uu[0], :]
        v = self.Wu[uu[1], :]
        i = self.Wi[uu[2], :]

        # Model for uu
        hm1 = u * v
        hp1 = abs(u - v)
        # Function to debug dimensions
        #self.debug = theano.function([uu], hm1)
        ## paramter for numerical stablility of log
        ##eps = T.scalar()
        h1 = T.tanh(T.dot(self.Wm1, hm1) + T.dot(self.Wp1, hp1) + self.b11)
        l = T.nnet.softmax(T.dot(self.U1, h1) + self.b21)
        #self.debug2 = theano.function([uu], l)
        # cost 1
        J1 = T.switch(T.eq(yu[1], 0), l[0], l[1])

        # Model for ui
        hm2 = u * i
        hp2 = abs(u - i)
        h2 = T.tanh(T.dot(self.Wm1, hm2) + T.dot(self.Wp2, hp2) + self.b12)
        l1 = T.dot(self.U2, h2)
        self.debug2 = theano.function([uu], l1)
        J2 = (l1 - yu[1]) ** 2

        cost = J1 + J2
        #cost = -T.log(l[0][y] + eps)
        grad1 = T.grad(cost, [u, v, i])
        grads = T.grad(cost, self.params)
        self.W1 = T.set_subtensor(self.W1[uu[0],:], self.W1[uu[0],:] - lr * grad1[0])
        self.W1 = T.set_subtensor(self.W1[uu[1], :], self.W1[uu[1],:] - lr * grad1[1])
        self.W2 = T.set_subtensor(self.W2[uu[2], :] - self.W2[uu[2], :] - lr * grad1[2])
        updates1 = [(self.Wu, self.W1)]
        updates2 = [(self.Wi, self.W2)]
        updates3 = [(param, param - lr * grad) for (param, grad) in zip(self.params, grads)]
        updates = updates1 + updates2 + updates3
        self.gd = theano.function([uu, yu], cost, updates=updates, mode='DebugMode')
        #self.gd = theano.function([uu,yu], cost, mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))


if __name__ == "__main__":
       u2v = user2vec(22166, 100, 100, 200)
       u2v.model()
       pdb.set_trace()


import numpy as np
from scipy.io import loadmat
import collections
import math
from collections import OrderedDict
import pdb
from scipy.sparse import coo_matrix


class data_handler():

    def __init__(self,rating_path,trust_path,time_path):
        self.rating_path = rating_path
        self.trust_path = trust_path
        self.time_path = time_path
        self.n = 0
        self.k = 0
        self.d = 0
        #BRING G BACK BEFORE RUNNING MAIN--------------------

    def load_matrices(self, test=0.2):
        #Loading matrices from data
        f1 = open(self.rating_path)
        f2 = open(self.trust_path)
        f3 = open(self.time_path)

        P_initial = loadmat(f1) #user-rating matrix
        G_raw = loadmat(f2) #trust-trust matrices
        P_initial = P_initial['rating_with_timestamp']
        G_raw = G_raw['trust']
        # Number of users
        self.n = max(P_initial[:,0])
        # number of items
        self.i = max(P_initial[:,1])
        # user and item and rating vectors from P matrix
        U = P_initial[:, 0]
        I = P_initial[:, 1]
        U = U-1
        I = I-1
        R = P_initial[:, 3]
        R = R/float(5)
        self.T1 = np.vstack((U, I, R)).T
        #pdb.set_trace()
        np.random.shuffle(self.T1)
        #pdb.set_trace()
        # Normalize ratings to (0-1) range (max-min normalization)
        #R = (1/float(4)) * (R - 5) + 1
        #data_UI = np.zeros((len(U), 3), dtype=np.float32)
        #data_UI[:, 0] = (U -1)
        #data_UI[:, 1] = (I - 1)
        #data_UI[:, 2] = R
        #T1 = np.zeros((self.n * self.n, 3), dtype=np.float32)
        #T1 = coo_matrix((R, (U-1,I-1)))
        #self.T1 = T1.tocsr()
        #pdb.set_trace()
        #self.T = np.zeros((self.n, self.n), dtype=np.float16)
        #self.T[G_raw[:,0] - 1, G_raw[:, 1] -1] = 1
        #pdb.set_trace()
        f = open('trust.txt', 'w')

        #for row in xrange(self.n):
        #    for col in xrange(self.n):
        #        f.write(str(row) + "\t" + str(col) + "\t" + str(self.T[row,col]) + '\n')


        # list of all indices
        #ind =list(np.ndindex(self.T.shape))
        #pdb.set_trace()
        test_value = self.n * test

data = data_handler("rating_with_timestamp.mat", "trust.mat", "rating_with_timestamp.mat")
data.load_matrices()









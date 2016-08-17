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
        G_raw = G_raw - 1
        # Number of users
        self.n = G_raw.max() + 1
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
        np.random.shuffle(self.T1)
        # Randomly shuffles each time, seeding for reproducibility
        np.random.seed(42)
        np.random.shuffle(self.T1)
        self.UI = coo_matrix((R, (U, I)))
        # Convert sparse to dense matrix for binary UI model
        self.ui = self.UI.todense()
        # Convert Ui to binary matrix
        self.ui[self.ui > 0.0] = 1.0
        # Write the UI matrix to file in the format "i j ui[i, j]"
        f_ui = open('UI_pos.txt', 'w')
        # writing positive entries to file
        pos_ind = np.where(self.ui == 1)
        R = pos_ind[0]
        C = pos_ind[1]
        for r,c  in zip(R, C):
                f_ui.write(str(r) + "\t" + str(c) + "\t" + str(1) + "\n")
        print("positive example written to file")
        ##pdb.set_trace()
        ## Generate training data for UI model from self.ui matrix throught negative sampling
        #U = self.ui.shape[0]
        #I = self.ui.shape[1]
        ## Array to store (u, i, label) tuples
        #data_UI = []
        #print("Writing negative examples to file")
        ## Generate negative samples
        #for u in xrange(U):
        #    ones = np.where(self.ui[u, :] == 1)[0]
        #    zeros = np.array(xrange(I))
        #    m = len(ones)
        #    zeros = np.setdiff1d(zeros, ones)
        #    neg_ind = np.random.randint(len(zeros), size=(m))
        #    neg_samples = zeros[neg_ind]
        #    for neg in neg_samples:
        #        f_ui.write(str(u) + "\t" + str(neg) + "\t" + str(0) + "\n")
        ##pdb.set_trace()
        #print("UI model's training data written to file")


if __name__ == "__main__":
    data = data_handler("rating_with_timestamp.mat", "trust.mat", "rating_with_timestamp.mat")
    data.load_matrices()









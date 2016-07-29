
import numpy as np
# Import logistic regrs
import pdb
from sklearn.metrics import accuracy_score


# Load saved params for prediction
Wu = np.load('./model/Wu.npy')
Wm1 = np.load('./model/Wm1.npy')
Wp1 = np.load('./model/Wp1.npy')
B11 = np.load('./model/B11.npy')
B21 = np.load('./model/B21.npy')
U1 = np.load('./model/U1.npy')


# Loading testing data
f = open('test.txt', 'r')
batch = []
print "loading test data"

def softmax(x):
    e = np.exp(x)
    dist = e / np.sum(e)
    return dist

for line in f:
    data = line.strip()
    data = map(lambda x:float(x), data.split())
    batch.append(data)

batch = np.array(batch).astype(np.int32)

X = batch[:, :2]
Y = batch[:, 2]


# Running the data through the model
U = Wu[X[:, 0], :]
V = Wu[X[:, 1], :]

hLm = U * V
hLp = abs(U - V)

hL = np.tanh(np.dot(Wm1, hLm.T) + np.dot(Wp1, hLp.T) + B11)
x = np.dot(U1, hL) + B21
l = softmax(x)

yp = np.argmax(l, axis=0)
print accuracy_score(Y, yp)

pdb.set_trace()
# Predict accuracy




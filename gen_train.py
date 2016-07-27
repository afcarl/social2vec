
import numpy as np
import pdb

R = []
C = []
D = []

f = open('Trust.txt','r')

for x in f.readlines():
    x = x.strip()
    r,c,d = map(lambda y:float(y), x.split())
    R.append(r)
    C.append(c)
    D.append(d)

R = np.array(R)
C = np.array(C)
D = np.array(D)
data = np.vstack((R, C, D)).astype(np.int32).T
pdb.set_trace()

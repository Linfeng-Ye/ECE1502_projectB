import numpy as np
def entropy(a):
    return np.sum(a*np.log2(1/a))

def CMI(a,b,c):
    C = (a+b+c)/3
    rtval = 0
    rtval += (a*np.log2(a/C)).sum()
    rtval += (b*np.log2(b/C)).sum()
    rtval += (c*np.log2(c/C)).sum()
    return rtval
a = np.array([1/3,1/2,1/6])
b = np.array([1/3,1/6,1/2])
c = np.array([1/6, 1/3,1/2])

print(np.mean([entropy(a), entropy(b), entropy(c)]), CMI(a,b,c))

a = np.array([0.9, 0.05, 0.05])
b = np.array([0.9,0.09,0.01])
c = np.array([0.9, 0.01,0.09])

print(np.mean([entropy(a), entropy(b), entropy(c)]), CMI(a,b,c))
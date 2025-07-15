# Automated check that you use the correct Python version
from sys import version_info

if version_info[0] < 3 or version_info[1] < 10:
    raise Exception("Must be using Python 3.10 or newer")
###########################################################
import numpy as np
import matplotlib.pyplot as plt
from ev import power_method




if __name__ == '__main__':
    c=np.array([[(1/(100*10**-9))+(1/10**-8),-1/10**-8],[-1/10**-8,(1/(47*10**-9))+(1/10**-8)]])
    l=np.array([[10**-5,0],[0,22*10**-6]])
    l_inv=np.linalg.inv(l)
    A=l_inv@c
    x0 = np.array([1, 0], dtype=complex)
    result=power_method(A,x0)
    n=result[0].size
    print("Der grösste Eigenewert ist: ",result[0][n-1])
    print("Der eigene Vektor ist: ",result[1])
    print("Die Kreisfrequenz w ist gleich: ",np.sqrt(result[0][n-1]))


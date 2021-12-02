'''
We want to know how many epochs we can do
'''

import numpy as np
import matplotlib.pyplot as plt

z_low = -50
z_high = 50
z_delta = 0.0001
z = np.arange(z_low, z_high + z_delta, z_delta)

c = 128 
lmbda = 4
batch_size = 64
N = 60000
lr = (1/N**2)

epsilon = 5
delta = 10**(-5)

sigma = 2*batch_size/(N*np.sqrt(lr)*c)
q = (N/batch_size)**(-1)
calc_delta = 1000
print("Check for sigma = " + str(sigma))

for lmbda in range(1, 32):
    mu0 = 1/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5*(z/sigma)**2)
    mu0[mu0 < delta] = 0
    mu1 = 1/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5*((z - 1)/sigma)**2)
    mu1[mu1 < delta] = 0
    mu = (1 - q)*mu0 + q*mu1
    
    v1 = np.divide(mu0, mu, out=np.zeros_like(mu0), where=mu!=0)
    v1 = np.sum(v1**lmbda*mu0*z_delta)
    v1 = np.log(v1)
    #print(v1)
    
    v2 = np.divide(mu, mu0, out=np.zeros_like(mu), where=mu0!=0)
    v2 = np.sum(v2**lmbda*mu*z_delta)
    v2 = np.log(v2)
    #print(v2)
    
    bound = q**2*lmbda*(lmbda+1)/((1 - q)*sigma**2) + q**3*lmbda**3/sigma**3
    #print(bound)

    alpha = np.max([v1, v2])

    T = (lmbda*epsilon + np.log(delta))/alpha
    if T < 0:
        continue
    else:
        print("found 1: " + str(T))
    
    tmp = np.exp(lmbda*alpha*T - lmbda*epsilon)
    if tmp < calc_delta:
        calc_delta = tmp
        print(lmbda)

print("found delta: " + str(calc_delta))

T = (lmbda*epsilon + np.log(delta))/alpha
print(T)
print(T*q)

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import csv
# from Markov import *

def BalanceDistribution_MonteCarlo_Method(P,X_0,M): #Works better when M is large
    n=len(P[0,:]); X=X_0; pi=np.zeros(n); M=M+1
    for i in range(M):        
        pi[int(X)]+=1
        X=np.random.choice(range(n), p=P[int(X), :])
    return pi/M

def print_BalanceDistribution_MonteCarlo_Method(P,X_0,M):
    n=len(P[0,:])
    pi=BalanceDistribution_MonteCarlo_Method(P,X_0,M)
    print(f'',end='\t')
    for i in range(n):
        print(f'X_{i}',end='\t')
    print('');  print(f'π = [',end='\t')
    for i in range(n):
        if i<n-1:
            print(f'{round(pi[i],5)}',end='\t')
        else:
            print(f'{round(pi[i],5)} \t]')
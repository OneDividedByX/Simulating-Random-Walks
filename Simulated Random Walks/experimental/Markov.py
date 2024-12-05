import sys
import time
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import csv
from experimental.random_walks.miscellaneous.markov_misc_func import rowtext_simulate_markov_chain
# from random_walks.miscellaneous.markov_misc_func import ExistenceInList
# import random_walks.miscellaneous.markov_misc_func as markov_misc
# ################################################################
################################################################

def simulate_markov_chain_to_nsteps(P, X_0, n_steps):
    n=len(P[0,:])
    X = np.zeros(n_steps+1)
    X[0] = X_0
    for i in range(0, n_steps):
        X[i+1] = np.random.choice(range(n), p=P[int(X[i]), :])
    return X
        
def simulate_markov_chain_procedure(P, X_0,speed):
    if speed==None:
        time_rate=1
    else:
        time_rate = float(1/speed)
    n=len(P[0,:])    
    if 0<=X_0 and X_0<=n:
        X = X_0; freq=np.linspace(0,n-1,n); prob=np.linspace(0,n-1,n)
        print('t',end='\t'); print('X_t',end='\t')
        for i in range(0,n):
            print(f'f({int(freq[i])})',end=' \t')
        for i in range(0,n):
            if i<n-1:   
                print(f'p({int(prob[i])})',end=' \t')
            else:
                print(f'p({int(prob[i])})')
        freq=np.zeros(n); prob=np.zeros(n); i=0; 
        while X>=0:
            freq[int(X)]+=1
            for j in range(n):
                prob[j]=round(freq[j]/(i+1),3)  #in this case sum of probabilities is equal to (i+1)
            rowtext_simulate_markov_chain(f't={i}',f'{int(X)}',freq,prob)
            time.sleep(time_rate)
            i=i+1; X = np.random.choice(range(n), p=P[int(X), :])
    else:
        print('Initial state out of range')
################################################################
################################################################
def print_TransitionMatrix(P):
    n=len(P[0,:])
    print('',end='\t')
    for i in range(0,n):
        print(f'X_{i}',end='\t')
    print()
    for i in range(0,n):
        print(f'X_{i}',end='\t')
        for j in range(0,n):
            if P[i,j]==0:
                print(f'*',end='\t')
            else:
                print(f'{P[i,j]:.3f}',end='\t')
        print()
        
################################################################
################################################################

def simulate_markov_chain(P, X_0, speed,n_steps):
    if n_steps==None:
        try: 
            simulate_markov_chain_procedure(P, X_0,speed)
        except KeyboardInterrupt:
            sys.exit()
    else:
        n_steps=n_steps+1
        n=len(P[0,:])
        if 0<=X_0 and X_0<=n:
            X=simulate_markov_chain_to_nsteps(P,X_0,n_steps); freq=np.linspace(0,n-1,n); prob=np.linspace(0,n-1,n)
            print('t',end='\t'); print('X_t',end='\t')
            for i in range(0,n):
                print(f'f({int(freq[i])})',end=' \t')
            for i in range(0,n):
                if i<n-1:   
                    print(f'p({int(prob[i])})',end=' \t')
                else:
                    print(f'p({int(prob[i])})')
            freq=np.zeros(n); prob=np.zeros(n)
            for i in range(n_steps):
                freq[int(X[i])]+=1
                for j in range(n):
                    prob[j]=round(freq[j]/(i+1),3)  #in this case sum of probabilities is equal to (i+1)
                rowtext_simulate_markov_chain(f't={i}',f'{int(X[i])}',freq,prob)
        else:
            print('Initial state out of range')
        
################################################################
################################################################

# def simulate_markov_chain_procedure(P, X_0,speed):
#     time_rate = float(1/speed)
#     n=len(P[0,:])    
#     if 0<=X_0 and X_0<=n:
#         X = X_0; freq=np.linspace(0,n-1,n); prob=np.linspace(0,n-1,n)
#         print('t',end='\t'); print('X_t',end='\t')
#         for i in range(0,n):
#             print(f'f({int(freq[i])})',end=' \t')
#         for i in range(0,n):
#             if i<n-1:   
#                 print(f'p({int(prob[i])})',end=' \t')
#             else:
#                 print(f'p({int(prob[i])})')
#         freq=np.zeros(n); prob=np.zeros(n); i=0; 
#         while X>=0:
#             freq[int(X)]+=1
#             for j in range(n):
#                 prob[j]=round(freq[j]/(i+1),3)  #in this case sum of probabilities is equal to (i+1)
#             rowtext_simulate_markov_chain(f't={i}',f'{int(X)}',freq,prob)
#             time.sleep(time_rate)
#             i=i+1; X = np.random.choice(range(n), p=P[int(X), :])
#     else:
#         print('Initial state out of range')
        
# def simulate_markov_chain(P, X_0,speed):
#     try: 
#         simulate_markov_chain_procedure(P, X_0,speed)
#     except KeyboardInterrupt:
#         sys.exit()

################################################################
################################################################
def update_Markov(frame,n,P,x,y,graph,state,freq_list,prob_list):
    # updating the data
    x.append(x[-1] + 1)
    Y=np.random.choice(range(n), p=P[int(y[-1]), :])
    y.append(Y)
    graph.set_xdata(x)
    graph.set_ydata(y)    
    plt.xlim(x[0], x[-1])
    if state=='discrete':        
        plt.plot(x[-1],y[-1],'ro')
    else:
        plt.plot(x[-1],y[-1])
    ###################################
    freq=freq_list[-1]; prob=prob_list[-1]
    freq[int(y[-1])]+=1
    for j in range(n):
        prob[j]=round(freq[j]/(x[-1]),3)  #in this case sum of probabilities is equal to (i+1)
    rowtext_simulate_markov_chain(f't={x[-1]}',f'{int(y[-1])}',freq,prob)

def update_Markov_ns(frame,n,P,x,y,graph,state,n_steps,freq_list,prob_list):
    if len(x)<=n_steps:
        # updating the data
        x.append(x[-1] + 1)
        Y=np.random.choice(range(n), p=P[int(y[-1]), :])
        y.append(Y)
        graph.set_xdata(x)
        graph.set_ydata(y)    
        plt.xlim(x[0], x[-1])
        if state=='discrete':        
            plt.plot(x[-1],y[-1],'ro')
        else:
            plt.plot(x[-1],y[-1])
        ###################################
        freq=freq_list[-1]; prob=prob_list[-1]
        freq[int(y[-1])]+=1
        for j in range(n):
            prob[j]=round(freq[j]/(x[-1]+1),3)  #in this case sum of probabilities is equal to (i+1)
        rowtext_simulate_markov_chain(f't={x[-1]}',f'{int(y[-1])}',freq,prob)       

def graph_simulate_markov_chain(P,X_0,speed,state,n_steps):
    n=len(P[0,:])
    if speed>0:
        rate=1/speed*1000
    else: rate=200
    ###################################
    print('t',end='\t'); print('X_t',end='\t')
    for i in range(0,n):
        print(f'f({i})',end=' \t')
    for i in range(0,n):
        if i<n-1:   
            print(f'p({i})',end=' \t')
        else:
            print(f'p({i})')
    freq=np.zeros(n); prob=np.zeros(n)
    freq[int(X_0)]=1; prob[int(X_0)]=1
    rowtext_simulate_markov_chain(f't={0}',f'{int(X_0)}',freq,prob)
    freq_list=[freq]; prob_list=[prob]
    ###################################
    if n_steps==None:
        n=len(P[0,:]);  x=[0];  y=[X_0]
        fig, ax = plt.subplots()
        graph = ax.plot(x,y,color = 'g')[0]
        # delta=(n-1)*0.05
        # plt.ylim(-delta,n-1+delta)
        plt.plot(0,X_0,'ro')
        anim = FuncAnimation(fig, update_Markov, fargs=(n, P, x, y,graph,state,freq_list,prob_list), frames=None,interval=rate,cache_frame_data=False)
        plt.grid()
        plt.show()
    else:
        n=len(P[0,:]);  x=[0];  y=[X_0]
        fig, ax = plt.subplots()
        graph = ax.plot(x,y,color = 'g')[0]
        # delta=(n-1)*0.05
        # plt.ylim(-delta,n-1+delta)
        plt.plot(0,X_0,'ro')
        anim = FuncAnimation(fig, update_Markov_ns, fargs=(n, P, x, y,graph,state,n_steps,freq_list,prob_list), frames=None,interval=rate,cache_frame_data=False)
        plt.grid()
        plt.show()
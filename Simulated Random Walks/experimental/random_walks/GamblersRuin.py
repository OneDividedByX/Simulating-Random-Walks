import sys
import time
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import csv
from experimental.random_walks.miscellaneous.markov_misc_func import rowtext_simulate_markov_chain
from experimental.Markov import *

def TransitionMatrix_GamblersRuin(n,X_0,prob_coin):
    P=np.zeros([n+1,n+1])
    P[0][0]=1; P[n][n]=1
    for i in range(1,n):
        P[i][i-1]=prob_coin[0]
        P[i][i+1]=prob_coin[1]
    return P

def print_TransitionMatrix_GamblersRuin(n,X_0,prob_coin):
    P=TransitionMatrix_GamblersRuin(n,X_0,prob_coin)
    print('',end='\t')
    for i in range(0,n+1):
        if i==X_0:
            print(f'{i} (X_0)',end='\t')
        else:
            print(f'{i}',end='\t')
    print()
    for i in range(0,n+1):
        if i==X_0:
            print(f'{i} (X_0)',end='\t')
        else:
            print(f'{i}',end='\t')
        for j in range(0,n+1):
            if P[i,j]==0:
                print(f'*',end='\t')
            else:
                print(f'{P[i,j]:.3f}',end='\t')
        print()

def simulate_GamblersRuin(n,X_0,prob_coin,speed,n_steps):
    P = TransitionMatrix_GamblersRuin(n,X_0,prob_coin)
    if n_steps==None: #Simulate to infinite loop
        try: 
            simulate_markov_chain_procedure(P, X_0,speed)
        except KeyboardInterrupt:
            sys.exit()
    elif n_steps=='stopped': #Simulate and stop when reach 0 or n
        if 0<=X_0 and X_0<=n:
            n=len(P[0,:])
            X=X_0
            freq=np.linspace(0,n-1,n); prob=np.linspace(0,n-1,n)
            print('t',end='\t'); print('X_t',end='\t')
            for i in range(0,n):
                print(f'f({int(freq[i])})',end=' \t')
            for i in range(0,n):
                if i<n-1:   
                    print(f'p({int(prob[i])})',end=' \t')
                else:
                    print(f'p({int(prob[i])})')    
            freq=np.zeros(n); prob=np.zeros(n)            
            freq[int(X)]+=1
            for j in range(n):
                prob[j]=round(freq[j]/(i+1),3)  #in this case sum of probabilities is equal to (i+1)
            rowtext_simulate_markov_chain(f't={0}',f'{int(X)}',freq,prob)
            i=0; 
            while i>=0:
                X=np.random.choice(np.linspace(0,n-1,n), p=P[int(X),:])           
                freq[int(X)]+=1
                for j in range(n):
                    prob[j]=round(freq[j]/(i+1),3)  #in this case sum of probabilities is equal to (i+1)
                rowtext_simulate_markov_chain(f't={i}',f'{int(X)}',freq,prob)
                if X==0 or X==n-1:
                    i=-1
                else:
                    i=i+1                
        else:
            print('Initial state out of range')
    else: #Simulate to n_steps 
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


def update_GamblersRuin(frame,n,P,x,y,graph,state,freq_list,prob_list):
    # updating the data
    if y[-1]>0 and y[-1]<n-1:
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

def update_GamblersRuin_ns(frame,n,P,x,y,graph,state,n_steps,freq_list,prob_list):
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

def graph_simulate_GamblersRuin(n,X_0,prob_coin,speed,state,n_steps):
    P = TransitionMatrix_GamblersRuin(n,X_0,prob_coin)
    n=n+1 #now is len of P[0,:]
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
        delta=(n-1)*0.05
        plt.ylim(-delta,n-1+delta)
        plt.plot(0,X_0,'ro')
        anim = FuncAnimation(fig, update_GamblersRuin, fargs=(n, P, x, y,graph,state,freq_list,prob_list), frames=None,interval=rate,cache_frame_data=False)
        plt.grid()
        plt.show()
    else:
        n=len(P[0,:]);  x=[0];  y=[X_0]
        fig, ax = plt.subplots()
        graph = ax.plot(x,y,color = 'g')[0]
        delta=(n-1)*0.05
        plt.ylim(-delta,n-1+delta)
        plt.plot(0,X_0,'ro')
        anim = FuncAnimation(fig, update_GamblersRuin_ns, fargs=(n, P, x, y,graph,state,n_steps,freq_list,prob_list), frames=None,interval=rate,cache_frame_data=False)
        plt.grid()
        plt.show()
        
def simulate_GamblersRuin_HittingTime(n,X_0,prob_coin, n_experiments): #Inf bound is used to test theorical results (default=0)
    n_steps=n_experiments
    P=TransitionMatrix_GamblersRuin(n,X_0,prob_coin)
    tau_list=[]
    XT_list=[]
    if 0<=X_0 and X_0<=n:
        for I in range (1,n_steps+1):        
            n=len(P[0,:])
            X=X_0
            freq=np.linspace(0,n-1,n); prob=np.linspace(0,n-1,n)
            print(f'HittingTime Experiment I={I}')
            print(f't',end='\t')
            print(f'X_t',end='\t')
            for i in range(0,n):
                print(f'f({int(freq[i])})',end=' \t')
            for i in range(0,n):
                if i<n-1:   
                    print(f'p({prob[i]})',end=' \t')
                else:
                    print(f'p({prob[i]})')
            freq=np.zeros(n); prob=np.zeros(n)            
            freq[int(X)]+=1
            for j in range(n):
                prob[j]=round(freq[j],3)  #in this case sum of probabilities is equal to (i+1)
            rowtext_simulate_markov_chain(f't={0}',f'{int(X)}',freq,prob)
            i=1; 
            while i>=0:
                X=np.random.choice(np.linspace(0,n-1,n), p=P[int(X),:])           
                freq[int(X)]+=1
                for j in range(n):
                    prob[j]=round(freq[j]/(i+1),3)  #in this case sum of probabilities is equal to (i+1)
                rowtext_simulate_markov_chain(f't={i}',f'{int(X)}',freq,prob)
                if X==0 or X==n-1:
                    tau_list.append(i)
                    XT_list.append(X)
                    i=-1
                else:
                    i=i+1
            print('\n================================')
        T=np.array(tau_list);XT=np.array(XT_list)
        T_max=T.max(); n=n-1 
        print(f'HittingTime Experiment Summary')
        print('T',end='\t'); print('f(T)',end='\t') ; print('F(T)',end='\t') ;  print(f'|X_T={n}|',end='\t') ; print(f'p_k(T={n})')
        total=0
        freq_XT=0
        for i in range(0,T_max+1):
            freq_i=0
            for j in range(0,len(T)):
                if i==T[j]:
                    freq_i+=1
                    if XT[j]==n:#condition to estimate the probability
                        freq_XT+=1
            total=total+freq_i      
            if total==0:
                print(f'{i}',end='\t'); print(f'{freq_i}',end='\t') ; print(f'{total}',end='\t') ; print(f'{freq_XT}',end='\t'); print(f'{0:.3f}')
            else:
                print(f'{i}',end='\t'); print(f'{freq_i}',end='\t') ; print(f'{total}',end='\t') ; print(f'{freq_XT}',end='\t'); print(f'{freq_XT/total:.3f}')
        print(f'T_mean: {T.mean()}')
    else:
        print('Initial state out of range')
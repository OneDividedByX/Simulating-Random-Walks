import sys
import time
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import csv
import matplotlib.animation as animation

def simulate_RandomWalkInZ(X_0,P,speed):
    if speed>0:
        rate=1/speed
        t=0
        X_t=X_0
        print('Time \t X_t')
        print(f't={t} \t {X_t}')
        while t>=0:
            X=np.array([X_t-1,X_t,X_t+1])
            X_t=np.random.choice(X,p=P)
            t=t+1        
            print(f't={t} \t {X_t}')
            time.sleep(rate)
    else:
        t=0
        X_t=X_0
        print('Time \t X_t')
        print(f't={t} \t {X_t}')
        while t>=0:
            X=np.array([X_t-1,X_t,X_t+1])
            X_t=np.random.choice(X,p=P)
            t=t+1        
            print(f't={t} \t {X_t}')
            
def update_RandomWalkInZ(frame,P,x,y,graph,state):
    # updating the data
    x.append(x[-1] + 1)
    X_t=y[-1]
    X=np.array([X_t-1,X_t,X_t+1])
    Y=np.random.choice(X, p=P)
    y.append(Y)
    graph.set_xdata(x)
    graph.set_ydata(y)    
    plt.xlim(x[0], x[-1])
    if state=='discrete':        
        plt.plot(x[-1],y[-1],'ro')
    elif state=='non_discrete':
        plt.plot(x[-1],y[-1])

def graph_RandomWalkInZ(X_0,P,speed,state):
    if speed>0:
        rate=1/speed*1000
    else: rate=200
    x=[0];  y=[X_0]
    fig, ax = plt.subplots()
    graph = ax.plot(x,y,color = 'g')[0]
    ax.axhline(y=0, lw=2, color='k')
    ax.axvline(x=0, lw=2, color='k')
    plt.plot(0,X_0,'ro')
    anim = FuncAnimation(fig, update_RandomWalkInZ, fargs=(P, x, y,graph,state), frames=None,interval=rate,cache_frame_data=False)
    plt.grid()
    plt.show()
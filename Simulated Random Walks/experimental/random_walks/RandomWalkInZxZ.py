import sys
import time
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import csv
import matplotlib.animation as animation

def ExistenceInList(element,list):
    b=0
    for i in range(len(list)):
        if (element==list[i]).all():
            b=1; i=-1
    if b==1:
        return 1
    else:
        return 0
    
def array_uniform_random_choice(array_list):
    n=len(array_list)
    r_index=np.random.choice(n)
    return array_list[r_index]

def adjacent_Points(XY):
    neighbors=[]
    for i in range(-1, 2):
        for j in range(-1, 2):
            if (i!=0 or j!=0):
                neighbors.append(np.array([XY[0]+i, XY[1]+j]))
    adjacent=[]
    for i in range(0, len(neighbors)):
        if np.linalg.norm(XY-neighbors[i],ord=1)==1:
            adjacent.append(neighbors[i])
    return adjacent

def available_adjacent_Points_1(XY,walk):    
    XY_adjacent=adjacent_Points(XY)
    available=[]
    for i in range(0,len(XY_adjacent)):
        if ExistenceInList(XY_adjacent[i], walk)==0:
            available.append(XY_adjacent[i])
    return available

def simulate_RandomWalkInZxZ_1(XY_0):
    XY=[]    
    t=0
    XY_t=XY_0
    XY.append(XY_t)
    print('Time \t XY_t')
    print(f't={t} \t {XY[-1]}')
    while t>=0:
        XY_available=available_adjacent_Points_1(XY[-1],XY)
        if len(XY_available)>0:
            XY_t=array_uniform_random_choice(XY_available)
            XY.append(XY_t)
            t=t+1
            print(f't={t} \t {XY[-1]}')
        else:
            t=-1

# def simulate_RandomWalkInZxZ_1_T(XY_0,T):
#     XY=[]    
#     t=0;T=T-1
#     XY_t=XY_0
#     XY.append(XY_t)
#     print('Time \t XY_t')
#     print(f't={t} \t {XY[-1]}')
#     while t<=T:
#         XY_available=available_adjacent_Points_1(XY[-1],XY)
#         if len(XY_available)>0:
#             XY_t=array_uniform_random_choice(XY_available)
#             XY.append(XY_t)
#             t=t+1
#             print(f't={t} \t {XY[-1]}')
#         else:
#             t=-1
            
def update_RandomWalkInZxZ_1(frame,t,x,y,XY,graph,state,annotate):
    # updating the data
    XY_available=available_adjacent_Points_1(XY[-1],XY)
    if len(XY_available)>0:
        XY_t=array_uniform_random_choice(XY_available)
        XY.append(XY_t)   
        xy=XY[-1]
        t.append(t[-1]+1)
        x.append(xy[0])
        y.append(xy[1])
        graph.set_xdata(x)
        graph.set_ydata(y)    
        # plt.xlim(-5, 5)
        if state=='discrete':        
            plt.plot(x[-1],y[-1],'ro')
        elif state=='non_discrete':
            plt.plot(x[-1],y[-1])
        if annotate==True:
            plt.annotate(f'{t[-1]}',xy) 

def graph_RandomWalkInZxZ_1(XY_0,speed,state,annotate):
    if speed>0:
        rate=1/speed*1000
    else: rate=200
    x=[XY_0[0]];  y=[XY_0[0]]; XY=[XY_0]; t=[0]
    fig, ax = plt.subplots()
    graph = ax.plot(x,y,color = 'g')[0]
    ax.axhline(y=0, lw=2, color='k')
    ax.axvline(x=0, lw=2, color='k')
    plt.plot(x,y,'co')
    plt.annotate(f'{0}',XY_0)
    # plt.ylim(y_min,y_max)
    anim = FuncAnimation(fig, update_RandomWalkInZxZ_1, fargs=(t,x, y,XY,graph,state,annotate), frames=None,interval=rate,cache_frame_data=False)
    plt.grid()
    plt.show()      
##############################################################
##############################################################

def available_adjacent_Points_2(XY,walk):

    if ExistenceInList(XY,walk)==0:        
        available=[XY-np.array([1,0]),XY+np.array([1,0])]
    else:
        available=[XY-np.array([0,1]),XY+np.array([0,1])]
    
    return available

def simulate_RandomWalkInZxZ_2_procedure(XY_0,speed):
    if speed==None:
        time_rate=1
    else:
        time_rate = float(1/speed)
    XY=[]; walk=[]
    t=0
    XY_t=XY_0
    XY.append(XY_t)
    print('Time \t XY_t')
    print(f't={t} \t {XY[-1]}')
    while t>=0:
        XY_available=available_adjacent_Points_2(XY[-1],walk)
        walk.append(XY[-1])
        XY_t=array_uniform_random_choice(XY_available)
        XY.append(XY_t)
        t=t+1
        print(f't={t} \t {XY[-1]}')
        time.sleep(time_rate)
def simulate_RandomWalkInZxZ_2(XY_0,speed,n_steps):
    if n_steps==None:
        try: 
            simulate_RandomWalkInZxZ_2_procedure(XY_0,speed)
        except KeyboardInterrupt:
            sys.exit()
    else:
        XY=[]; walk=[]
        t=0
        XY_t=XY_0
        XY.append(XY_t)
        print('Time \t XY_t')
        print(f't={t} \t {XY[-1]}')
        while t<n_steps:
            XY_available=available_adjacent_Points_2(XY[-1],walk)
            walk.append(XY[-1])
            XY_t=array_uniform_random_choice(XY_available)
            XY.append(XY_t)
            t=t+1
            print(f't={t} \t {XY[-1]}') 
def update_RandomWalkInZxZ_2(frame,t,x,y,XY,walk,graph,state,annotate):
    # updating the data
    XY_available=available_adjacent_Points_2(XY[-1],walk)
    walk.append(XY[-1])
    XY_t=array_uniform_random_choice(XY_available)
    XY.append(XY_t) 
    xy=XY[-1]
    t.append(t[-1]+1)
    x.append(xy[0])
    y.append(xy[1])
    graph.set_xdata(x)
    graph.set_ydata(y)    
    # plt.xlim(-5, 5)
    if state=='discrete': 
        plt.plot(x[-1],y[-1],'ro')
    elif state=='non_discrete':
        plt.plot(x[-1],y[-1])
    if annotate==True:
        plt.annotate(f'{t[-1]}',xy) 

def graph_RandomWalkInZxZ_2(XY_0,speed,state,annotate):
    if speed>0:
        rate=1/speed*1000
    else: rate=200
    x=[XY_0[0]];  y=[XY_0[0]]; XY=[XY_0]; t=[0]; walk=[]
    fig, ax = plt.subplots()
    graph = ax.plot(x,y,color = 'g')[0]
    ax.axhline(y=0, lw=2, color='k')
    ax.axvline(x=0, lw=2, color='k')
    plt.plot(x,y,'co')
    plt.annotate(f'{0}',XY_0)
    # plt.ylim(y_min,y_max)
    anim = FuncAnimation(fig, update_RandomWalkInZxZ_2, fargs=(t,x, y,XY,walk,graph,state,annotate), frames=None, interval=rate,cache_frame_data=False)
    plt.grid()
    plt.show() 
    
    
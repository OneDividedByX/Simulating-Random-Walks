import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import csv
from experimental.Markov import simulate_markov_chain,graph_simulate_markov_chain, print_TransitionMatrix
from experimental.MonteCarlo import *
from experimental.random_walks.RandomWalkInZ import graph_RandomWalkInZ
from experimental.random_walks.RandomWalkInZxZ import simulate_RandomWalkInZxZ_1,simulate_RandomWalkInZxZ_2
from experimental.random_walks.RandomWalkInZxZ import graph_RandomWalkInZxZ_1,graph_RandomWalkInZxZ_2
from experimental.random_walks.GamblersRuin import print_TransitionMatrix_GamblersRuin,simulate_GamblersRuin,graph_simulate_GamblersRuin,simulate_GamblersRuin_HittingTime
from experimental.random_walks.CuponCollector import simulate_CuponCollector,cupon_matrix,graph_simulate_CuponCollector
from experimental.random_walks.CuponCollector import simulate_CuponCollector_HittingTime
# from experimental.random_walks.miscellaneous.markov_misc_func import *

################################################################################
################################################################################
# Define the transition matrix
v_0=np.array([0.0,0.5,0,0.5])
v_1=np.array([0.5,0,0.5,0])
v_2=np.array([0,0.5,0,0.5])
v_3=np.array([0.5,0,0.5,0])
P=np.array([v_0,v_1,v_2,v_3])
X_0=0; n_steps=1000

prob_coin =np.array([0.5,0.5])
# print_TransitionMatrix_GamblersRuin(8,5,prob_coin)

# simulate_GamblersRuin(8,5,prob_coin,speed=1,n_steps='stopped')
# graph_simulate_GamblersRuin(15,7,prob_coin,speed=10,state='discrete',n_steps=None)
simulate_markov_chain(P, X_0, speed=1, n_steps=15)
# graph_simulate_markov_chain(P,X_0,5,'discrete',n_steps=100)

################################################################################
################################################################################
XY_0=np.array([0,0]); speed=10
# graph_RandomWalkInZxZ_1(XY_0,speed,'discrete',annotate=True)
# graph_RandomWalkInZxZ_2(XY_0,speed,'discrete',annotate=True)

################################################################################
################################################################################
# prob=np.array([0.43,0.1,0.47])
# # simulate_RandomWalkInZ(0,prob,0)
# graph_RandomWalkInZ(X_0,prob,1,'discrete')

################################################################################
################################################################################
n=10; X_0=5
prob_coin =np.array([0.5,0.5])
# simulate_GamblersRuin(n=n,X_0=X_0,prob_coin=prob_coin,speed=2,n_steps='stopped')
# graph_simulate_GamblersRuin(n=n,X_0=X_0,prob_coin=prob_coin,speed=2,state='discrete',n_steps=None)
# simulate_GamblersRuin_HittingTime(n=n,X_0=X_0,prob_coin=prob_coin,n_experiments=1000)

################################################################################
################################################################################

# simulate_CuponCollector(10, 0,1,n_steps=None)
# graph_simulate_CuponCollector(10,0,speed=10,state='discrete',n_steps=None)
# simulate_CuponCollector_HittingTime(number_of_coupons=10,X_0=0, T_inf_bound=35,n_experiments=1000)


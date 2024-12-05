import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import csv


def states(index, length):
  binary_str = bin(index)[2:].zfill(length)
  return [int(bit) for bit in binary_str]

def hypercube_matrix_DictionaryOfVertices(n):
  dictionary = {}
  for i in range(2**n):
    dictionary[i]=(states(i,n))
  return dictionary

# def states_index(vector):
#   binary_str = "".join(map(str, vector))
#   return int(binary_str, 2)
def states_index(vector):
  dict = hypercube_matrix_DictionaryOfVertices(len(vector))
  i=0
  for index in range(len(dict)):
    if (np.array(dict[index]) == vector).all():
      i=index
      index=-1
  return i

def hypercube_TransitionMatrix(n,type):
  matrix = np.zeros((2**n,2**n))
  for i in range(2**n):
    if type=='lazy':
      q = 1/2
    elif type=='not_lazy':
      q = 0
    else:
      i=-1
      return print(f'Insert a correct parameter option')
    matrix[i][i]=q
    aux = states(i,n)    
    for j in range(n):
      auxx=np.array(aux)
      if aux[j] == 0:
        auxx[j] = 1
      else:
        auxx[j] = 0
      matrix[i][states_index(auxx)] = (1-q)/n
  return matrix
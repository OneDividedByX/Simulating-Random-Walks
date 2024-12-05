import numpy as np

def rowtext_simulate_markov_chain(text_1, text_2,freq,prob):
    n=len(freq)
    print(text_1,end='\t')
    print(text_2,end='\t')
    for i in range(0,n):
        print(f'{int(freq[i])}',end=' \t')
    for i in range(0,n):
        if i<n-1:   
            print(f'{prob[i]}',end=' \t')
        else:
            print(f'{prob[i]}')
            
def ExistenceInList(element,list):
    b=0
    for i in range(len(list)):
        if (element==list[i]).all():
            b=1; i=-1
    if b==1:
        return 1
    else:
        return 0
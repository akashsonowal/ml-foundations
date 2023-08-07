import numpy as np 
from numpy import argmax 

def ctc_greedy_decoder(probs):
    T, S = probs.shape
    argmaxes = [argmax(probs[t, :]) for t in range(T)]
    decoded_output = []

    for i, args in enumerate(argmaxes): # args = 0 for blank token
        if (args != 0 and args == argmaxes[i-1] and i! = 0) or (args == 0): # collapse repeats and remove blank token
            continue
        decoded_output.append(args)
    return decoded_output
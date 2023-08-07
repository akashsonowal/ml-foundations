import numpy as np 
from numpy import argmax 

def ctc_greedy_decoder(probs):
    T, S = probs.shape
    decoded_output = []

    prev_token = -1 # Initialize prev_class to an invalid value

    for t in range(T):
        args = np.argmax(probs[t, :])

        # args = 0 for blank token
        if (args != 0 and args == prev_token) or (args == 0): # collapse repeats and remove blank token 
            continue 
        else:
            decoded_output.append(args)
            prev_token = args
            
    return decoded_output
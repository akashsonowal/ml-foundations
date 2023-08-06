
import numpy as np 
from numpy import argmax 

def greedy_decoder(probs):
    T, S = probs.shape
    return [argmax(probs[t, :]) for t in range(T)]

if __name__ == "__main__":
    np.random.seed(3)

    time = 50
    output_dim = 20

    probs = np.random.rand(time, output_dim)
    probs = probs / np.sum(probs, axis=1, keepdims=True)

    sequence = greedy_decoder(probs)
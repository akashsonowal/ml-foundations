import numpy as np 

def levenstein_distance(ref, hyp): # character matching b/ew words
    m, n = len(ref), len(hyp)
    if ref == hyp:
        return 0
    if m == 0:
        return n 
    if n == 0:
        return m 
        


if __name__ == "__main__":
    a = ['Mavs', 'Spurs', 'Lakers', 'Cavs']
    b = ['Rockets', 'Pacers', 'Warriors', 'Celtics']

    for i, k in zip(a, b):
        print(levenstein_distance(i, k))


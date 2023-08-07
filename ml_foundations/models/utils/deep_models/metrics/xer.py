import numpy as np 

def levenstein_distance(ref, hyp):
    pass 


if __name__ == "__main__":
    a = ['Mavs', 'Spurs', 'Lakers', 'Cavs']
    b = ['Rockets', 'Pacers', 'Warriors', 'Celtics']

    for i, k in zip(a, b):
        print(levenstein_distance(i, k))


import numpy as np
import math 
import collections  

NEG_INF = -float("-inf")





if __name__ == "__main__":
    np.random.seed(3)

    time = 50
    output_dim = 20

    probs = np.random.rand(time, output_dim)
    probs = probs / np.sum(probs, axis=1, keepdims=True)

    labels, score = ctc_decode(probs)
    print(f"Score {score}:.3f")

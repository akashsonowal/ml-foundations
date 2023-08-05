import numpy as np
import math 
import collections  

NEG_INF = -float("-inf")

def ctc_decode(probs, beam_size=100, blank=0):
     """
    Performs inference for the given output probabilities.
    Arguments:
        probs: The output probabilities (e.g. post-softmax) for each
            time step. Should be an array of shape (time x output dim).
        beam_size (int): Size of the beam to use during inference.
        blank (int): Index of the CTC blank label.
    Returns the output label sequence and the corresponding negative
    log-likelihood estimated by the decoder.
    """
    pass




if __name__ == "__main__":
    np.random.seed(3)

    time = 50
    output_dim = 20

    probs = np.random.rand(time, output_dim)
    probs = probs / np.sum(probs, axis=1, keepdims=True)

    labels, score = ctc_decode(probs)
    print(f"Score {score :.3f}")

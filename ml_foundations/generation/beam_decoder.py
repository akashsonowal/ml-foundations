import numpy as np
import math

def beam_search_decoder(probs, beam_size):
    T, S = probs.shape # T time steps (sequence length) and S is vocab size

    beams = [[[], 0.0]] # initialize one beam that starts with empty token and score 0 

    for t in range(T):
        new_beam = [] # new beams at curr time step

        for i in range(len(beams)): # starts from the previous beams at previous time step
            seq, score = beams[i]

            for j in range(S):  # Iterate over each possible next token
                candidate_seq = seq + [j]  # Add the next token to the sequence
                candidate_score = score - math.log(probs[t, j])  # Update the score

                new_beam.append([candidate_seq, candidate_score])

        # Sort and keep only top-k sequences based on the score
        new_beam.sort(key=lambda x: x[1])
        beams = new_beam[:beam_size]

    return beams

if __name__ == "__main__":
    np.random.seed(3)

    time = 50
    output_dim = 20

    probs = np.random.rand(time, output_dim)
    probs = probs / np.sum(probs, axis=1, keepdims=True)

    beams = beam_search_decoder(probs, 5)
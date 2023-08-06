import math 

def beam_search_decoder(probs, k):
    T, S = probs.shape

    beams = [ [list(), 0.0] ]

    for t in range(T):
        all_candidates = list()
        for i in range(len(sequences)):
            seq, score = sequences[i]

            for j in range(len(row)):
                candidate = [seq + j, score - math.log(row[j])]

            all_candidates.append(candidate)
    
    ordered = sorted(all_candidates, key=lambda tup: tup[1])
    sequences = ordered[:k]
    return sequences
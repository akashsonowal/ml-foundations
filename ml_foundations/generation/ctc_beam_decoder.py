import numpy as np
import math
import collections

# instead of keeping a list of alignments in the beam, we store the output prefixes after collapsing repeats and removing
# blank characters. At each step of the search we accumulate scores for a given prefix based on all the alignments which map to it

NEG_INF = - float("inf")


def make_new_beam():
    fn = lambda: (NEG_INF, NEG_INF)
    return collections.defaultdict(fn)


def logsumexp(*args):
    """
    Stable log sum exp. (for numerical stability)
    """
    if all(a == NEG_INF for a in args):
        return NEG_INF
    a_max = max(args)
    lsp = math.log(sum(math.exp(a - a_max) for a in args))
    return a_max + lsp


def ctc_beam_decode(probs, beam_size=5, blank=0):
    """
    Performs inference for the given output probabilities.
    Arguments:
        probs: The output probabilities (e.g. post-softmax) for each
            time step. Should be an array of shape (time x output dim).
        beam_size (int): Size of the beam to use during inference.
        blank (int): Index of the CTC blank label.
    Returns the output label sequence and the corresponding negative log-likelihood estimated by the decoder.
    """
    T, S = probs.shape
    probs = np.log(probs)
    # Elements in the beam are (prefix, (p_blank, p_no_blank))
    # Initialize the beam with the empty sequence, a probability of
    # 0 for ending in blank and negative infinity value indicates that the sequence has not been extended yet,
    # and its probability for ending in non-blank is unknown or undefined at this point. (in log space).
    beams = [(tuple(), (0.0, NEG_INF))]

    for t in range(T):  # loop over time
        # A default dictionary to store the next step (curr time) candidates.
        next_beams = make_new_beam()

        for s in range(S):  # loop over vocab
            p = probs[t, s]
            # The variables p_b and p_nb are respectively the probabilities for the prefix given that it ends in a
            # blank and does not end in a blank at this time step.
            for prefix, (p_b, p_nb) in beams:  # loop over beams at previous time step
                # If we propose a blank the prefix doesn't change. Only the probability of ending in blank gets updated.
                if s == blank:
                    n_p_b, n_p_nb = next_beams[prefix]  # n_p_b is new p_b and n_p_nb is new p_nb
                    n_p_b = logsumexp(n_p_b, p_b + p, p_nb + p)
                    next_beams[prefix] = n_p_b, n_p_nb
                    continue

                # Extend the prefix by the new character s and add it to the beam. Only the probability of
                # not ending in blank gets updated.
                end_t = prefix[-1] if prefix else None
                n_prefix = prefix + (s,)
                n_p_b, n_p_nb = next_beams[n_prefix]

                if s != end_t:
                    n_p_nb = logsumexp(n_p_nb, p_b + p, p_nb + p)
                else:
                    # We don't include the previous probability of not ending in blank (p_nb) if s is repeated at the end. The CTC
                    # algorithm merges characters not separated by a blank.
                    n_p_nb = logsumexp(n_p_nb, p_b + p)

                # *NB* this would be a good place to include an LM score.
                next_beams[n_prefix] = (n_p_b, n_p_nb)

                # If s is repeated at the end we also update the unchanged prefix. This is the merging case.
                if s == end_t:
                    n_p_b, n_p_nb = next_beams[prefix]
                    n_p_nb = logsumexp(n_p_nb, p_nb + p)
                    next_beams[prefix] = (n_p_b, n_p_nb)

        # Sort and trim the beam before moving on to the next time-step.
        beams = sorted(next_beams.items(), key=lambda x: logsumexp(*x[1]), reverse=True)
        beams = beams[:beam_size]

    best = beams[0]
    return best[0], -logsumexp(*best[1])


if __name__ == "__main__":
    np.random.seed(3)

    time = 50
    output_dim = 20

    probs = np.random.rand(time, output_dim)
    probs = probs / np.sum(probs, axis=1, keepdims=True)

    labels, score = ctc_beam_decode(probs)
    print(f"best beam: {labels}")
    print(f"Score {score :.3f}")

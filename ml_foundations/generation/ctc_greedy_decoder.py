import numpy as np

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

if __name__ == "__main__":
    np.random.seed(3)

    time = 50
    output_dim = 20

    probs = np.random.rand(time, output_dim)
    probs = probs / np.sum(probs, axis=1, keepdims=True)

    sequence = ctc_greedy_decoder(probs)
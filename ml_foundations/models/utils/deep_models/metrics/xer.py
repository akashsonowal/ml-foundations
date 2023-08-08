import numpy as np 

def levenstein_distance(source, target): # character matching b/ew words
    n, m = len(source), len(target)

    distance = np.zeros((n + 1, m + 1), dtype=np.int32)

    # when n = 0 i.e., source is empty
    for j in range(m + 1):
        distance[0][j] = j # insert

    # when m = 0 i.e., target is empty
    for  i in range(n + 1):
        distance[i][0] = i # delete
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if source[i - 1] == target[j - 1]:
                distance[i][j] = distance[i - 1][j - 1]
            else:
                distance[i][j] = 1 + min( # the 1 is cost of edit operation at current step
                    distance[i - 1][j], # insert character in source
                    distance[i][j - 1], # delete character in source
                    distance[i - 1][j - 1] # replace charcter in source
                )
    return distance[n][m]

if __name__ == "__main__":
    A = ["helo", "algorithm", "kitten", "gate"]
    B = ["hello", "rhythm", "sitting", "goat"]

    for i in range(len(A)):
        print("Levenshtein Distance between {} and {} = {}".format(A[i], B[i], levenstein_distance(A[i], B[i])))



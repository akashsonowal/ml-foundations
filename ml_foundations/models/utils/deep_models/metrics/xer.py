from .edit_distance import levenstein_distance

def wer(source_seq, target_seq, ignore_case=True):
    if ignore_case:
        source_seq = source_seq.lower()
        target_seq = target_seq.lower()
    
    source_words = source_seq.split(" ")
    target_words = target_seq.split(" ")

    edit_distance = levenstein_distance(source_words, target_words) # edit distance sums over every word pair
    return float(edit_distance) / len(source_words)

def cer(source_seq, target_seq, ignore_case=True):
    if ignore_case:
        source_seq = source_seq.lower()
        target_seq = target_seq.lower()
    
    edit_distance = levenstein_distance(source_seq, target_seq) # entire sequence is considered as a word
    return float(edit_distance) / len(source_seq)

if __name__ == "__main__":
    seq_1 = "Hi I am Akash Sonowal"
    seq_2 = "Ho I am Aakash Sonowall"

    print(f"WER and CER of {seq_1} and {seq_2} is {wer(seq_1, seq_2)} and {cer(seq_1, seq_2)}")
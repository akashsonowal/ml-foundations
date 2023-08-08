from .edit_distance import levenstein_distance

def wer(source_seq, target_seq, ignore_case=True):
    if ignore_case:
        source_seq = source_seq.lower()
        target_seq = target_seq.lower()
    
    source_words = source_seq.split(" ")
    target_words = target_seq.split(" ")

    edit_distance = levenstein_distance(source_words, target_words)
    return float(edit_distance) / len(source_words)


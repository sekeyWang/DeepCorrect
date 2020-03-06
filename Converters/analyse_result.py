import pandas as pd
from parsers import _match_AA_novor
def analyse(result_file_name):
    df = pd.read_csv(result_file_name)
    match1, match2 = 0, 0
    total1, total2 = 0, 0
    same_seq1, same_seq2 = 0, 0
    seq_num = 0
    for index, row in df.iterrows():
        if type(row['deepnovo_seq']) is not str:
            continue
        seq = row['seq'].split(',')
        novor_seq = row['novor_seq'].split(',')
        deepnovo_seq = row['deepnovo_seq'].split(',')

        seq_num += 1
        num_match1, predicted_result1 = _match_AA_novor(seq, novor_seq)
        match1 += num_match1
        total1 += len(predicted_result1)
        if num_match1 == len(predicted_result1):
            same_seq1 += 1

        num_match2, predicted_result2 = _match_AA_novor(seq, deepnovo_seq)
        match2 += num_match2
        total2 += len(predicted_result2)
        if num_match2 == len(predicted_result2):
            same_seq2 += 1
    print(match1, total1, match1/total1)
    print(same_seq1, seq_num, same_seq1 / seq_num)

    print(match2, total2, match2/total2)
    print(same_seq2, seq_num, same_seq2 / seq_num)

if __name__ == '__main__':
    analyse(result_file_name = "training_features/test.csv")
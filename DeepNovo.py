import pandas as pd
import config


def print_deepnovo_accuracy(filename):
    df = pd.read_csv(filename, sep='\t')
    correct, total = 0, 0
    for index, row in df.iterrows():
        correct += row['recall_AA']
        total += row['predicted_len']
    print(correct, total, correct / total)


def read_deepnovo_result(filename):
    df = pd.read_csv(filename, sep='\t')
    feature_id = df['feature_id']
    predicted_sequence = df['predicted_sequence']
    predicted_result = {}
    for idx in range(len(feature_id)):
        predicted_result[feature_id[idx]] = predicted_sequence[idx]
    return predicted_result


if __name__ == '__main__':
    result = read_deepnovo_result(config.Deepnovo_result_filename)
    print(type(result['F0:7259']) is not str)
    print(type(result['F0:7261']) is not str)
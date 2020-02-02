import pandas as pd
from parsers import parse_denovo_sequence, parse_raw_sequence, _match_AA_novor
def add_feature(feature_file_name, novor_result_file_name, output_file_name):
    df_feature = pd.read_csv(feature_file_name)
    df_novor = pd.read_csv(novor_result_file_name, skiprows=20, skipinitialspace=True)

    if (df_feature.shape[0] == df_novor.shape[0]):
        df_feature['predicted_seq'] = df_novor['peptide']
        df_feature['compare'] = None
        for index, row in df_feature.iterrows():
            _, target = parse_raw_sequence(row['seq'])
            _, predicted = parse_denovo_sequence(row['predicted_seq'])
            num_match, predicted_result = _match_AA_novor(target, predicted)
            df_feature['seq'][index] = ','.join(target)
            df_feature['predicted_seq'][index] = ','.join(predicted)
            df_feature['compare'][index] = predicted_result
            print(index)
        df_feature.to_csv(output_file_name)
    else:
        print(df_feature.shape[0], df_novor.shape[0])

if __name__ == '__main__':
    add_feature(feature_file_name='test_file/features.test.csv', novor_result_file_name='test_file/spectrum.mgf.csv', output_file_name = 'test_file/novor_data.csv')
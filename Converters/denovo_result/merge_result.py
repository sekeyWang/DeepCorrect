import pandas as pd
from parsers import parse_denovo_sequence, parse_raw_sequence
def add_feature(feature_file_name, novor_file_name, deepnovo_file_name, output_file_name):
    df_feature = pd.read_csv(feature_file_name)
    df_novor = pd.read_csv(novor_file_name, skiprows=20, skipinitialspace=True)
    df_deepnovo = pd.read_csv(deepnovo_file_name, sep='\t')
    deepnovo_idx = 0
    df_feature = df_feature.drop(columns=['scans','profile','feature area'])
    df_feature['novor_seq'] = df_novor['peptide']
    df_feature['deepnovo_seq'] = None
    for index, row in df_feature.iterrows():
        _, target = parse_raw_sequence(row['seq'])
        df_feature['seq'][index].loc = ','.join(target)

        _, novor_seq = parse_denovo_sequence(row['novor_seq'])
        df_feature['novor_seq'][index].loc = ','.join(novor_seq)

        if index+1 == df_deepnovo['feature_id'][deepnovo_idx]:
            if type(df_deepnovo['predicted_sequence'][deepnovo_idx]) is str:
                deepnovo_seq = df_deepnovo['predicted_sequence'][deepnovo_idx].split(',')
                df_feature['deepnovo_seq'][index].loc = ','.join(deepnovo_seq)
            deepnovo_idx += 1
        if index % 1000 == 480:
            print(index)

if __name__ == '__main__':
    add_feature(feature_file_name='../feature_files/features.csv', novor_file_name='spectrum.mgf.csv', deepnovo_file_name = 'features.csv.deepnovo_denovo', output_file_name = 'result.csv')
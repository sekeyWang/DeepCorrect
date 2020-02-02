import pandas as pd
from parsers import parse_denovo_sequence, parse_raw_sequence, _match_AA_novor

def add_feature(feature_file_name, deepnovo_result_file_name, output_file_name):
    df_feature = pd.read_csv(feature_file_name)
    df_deepnovo = pd.read_csv(deepnovo_result_file_name, sep='\t')

    df_feature['predicted_seq'] = None
    df_feature['compare'] = None

    if (df_feature.shape[0] == df_deepnovo.shape[0]):
        for index, row in df_feature.iterrows():
            _, target = parse_raw_sequence(row['seq'])
            _, predicted = parse_denovo_sequence(row['predicted_seq'])
            num_match, predicted_result = _match_AA_novor(target, predicted)
            df_feature['seq'][index] = target
            df_feature['predicted_seq'][index] = predicted
            df_feature['compare'][index] = predicted_result
            print(index)
        df_feature.to_csv(output_file_name)
    else:
        print(df_feature.shape[0], df_deepnovo.shape[0])
        new_feature_df = pd.DataFrame(columns=df_feature.columns)
        for index, row in df_deepnovo.iterrows():
            id = row['feature_id']-1
            add_row = df_feature.loc[id].copy()
            _, target = parse_raw_sequence(add_row['seq'])
            predicted = row['predicted_sequence'].split(',')
#            print(target, predicted)
            num_match, predicted_result = _match_AA_novor(target, predicted)
            add_row['seq'] = ','.join(target)
            add_row['predicted_seq'] = ','.join(predicted)
            add_row['compare'] = predicted_result
            new_feature_df = new_feature_df.append(add_row)
            print(index)
        new_feature_df.to_csv(output_file_name)


if __name__ == '__main__':
    add_feature(feature_file_name='test_file/features.test.csv', deepnovo_result_file_name='test_file/features.test.csv.deepnovo_denovo', output_file_name = 'test_file/deepnovo_data.csv')
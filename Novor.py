import pandas as pd
from data_reader import DenovoDataset
import config


def read_novor_result(novor_filename):
    df = pd.read_csv(novor_filename, skiprows=20, skipinitialspace=True)
    print(df.head(5))


Novor_input_file_name = 'input.mgf'


def make_mgf(dataset:DenovoDataset):
    output_file = open(Novor_input_file_name, 'w')
    for i in range(len(dataset)):
        print(i)
        denovo_data = dataset[i]
        feature = denovo_data.original_dda_feature
        output_file = open(Novor_input_file_name, 'a')
        output_file.writelines('BEGIN IONS\n')
        output_file.writelines('TITLE=%s,%s\n' % (feature.feature_id, feature.scan))
        output_file.writelines('PEPMASS=%f\n' % feature.mass)
        output_file.writelines('CHARGE=%d+\n' % feature.z)
        output_file.writelines('SCANS=%d\n' % i)
        output_file.writelines('RTINSECONDS=%f\n' % feature.rt_mean)
        for j in range(len(denovo_data.mz_list)):
            output_file.writelines('%f %f\n'%(denovo_data.mz_list[j], denovo_data.intensity_list[j]))
        output_file.writelines('END IONS\n\n')


if __name__ == '__main__':
    dataset = DenovoDataset(config.input_feature_file_train, config.input_spectrum_file_train)
    make_mgf(dataset)
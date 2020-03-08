from dataclasses import dataclass

#input_spectrum_file_train = "Hela/spectrums.mgf"
#input_feature_file_train = "Hela/features.csv.identified.test.nodup"
#Deepnovo_result_filename = 'Hela/features.csv.identified.test.nodup.deepnovo_denovo'
input_spectrum_file_train = "Converters/feature_files/spectrum.mgf"

input_feature_file_train = "Converters/training_features/train.csv"
input_feature_file_val = "Converters/training_features/valid.csv"
input_feature_file_test = "Converters/training_features/test.csv"

output_file_test = "Converters/training_features/test_score.csv"

mass_H = 1.0078
mass_H2O = 18.0106
mass_NH3 = 17.0265
mass_N_terminus = 1.0078
mass_C_terminus = 17.0027
mass_CO = 27.9949

mass_AA = {'_PAD': 0.0,
           '_GO': mass_N_terminus-mass_H,
           '_EOS': mass_C_terminus+mass_H,
           'A': 71.03711, # 0
           'R': 156.10111, # 1
           'N': 114.04293, # 2
           'N(Deamidation)': 115.02695,
           'D': 115.02694, # 3
           #~ 'C(Carbamidomethylation)': 103.00919, # 4
           'C(Carbamidomethylation)': 160.03065, # C(+57.02)
           #~ 'C(Carbamidomethylation)': 161.01919, # C(+58.01) # orbi
           'E': 129.04259, # 5
           'Q': 128.05858, # 6
           'Q(Deamidation)': 129.0426,
           'G': 57.02146, # 7
           'H': 137.05891, # 8
           'I': 113.08406, # 9
           'L': 113.08406, # 10
           'K': 128.09496, # 11
           'M': 131.04049, # 12
           'M(Oxidation)': 147.0354,
           'F': 147.06841, # 13
           'P': 97.05276, # 14
           'S': 87.03203, # 15
           'T': 101.04768, # 16
           'W': 186.07931, # 17
           'Y': 163.06333, # 18
           'V': 99.06841, # 19
          }
amino_acid_number = {
        'A': 0,
        'R': 1,
        'N': 2,
        'N(Deamidation)':20,
        'D': 3,
        'C': 4,
        'C(Carbamidomethylation)':21,
        'E': 5,
        'Q': 6,
        'Q(Deamidation)': 22,
        'G': 7,
        'H': 8,
        'I': 9,
        'L': 10,
        'K': 11,
        'M': 12,
        'M(Oxidation)':23,
        'F': 13,
        'P': 14,
        'S': 15,
        'T': 16,
        'W': 17,
        'Y': 18,
        'V': 19,
}
col_feature_id = "spec_group_id"
col_precursor_mz = "m/z"
col_precursor_charge = "z"
col_rt_mean = "rt_mean"
col_raw_sequence = "seq"
col_scan_list = "scans"
col_feature_area = "feature area"
col_predicted_seq = "novor_seq"

MZ_MAX = 3000.0
MAX_LEN = 50 #if args.search_denovo else 30

train_epochs = 12
splits = 5
lr = 0.01
momentum = 0.9

@dataclass
class DDAFeature:
    feature_id: str
    mz: float
    z: int
    rt_mean: float
    peptide: list
    scan: str
    mass: float
#    feature_area: str
    predicted_seq: list


@dataclass
class DenovoData:
    original_dda_feature: DDAFeature
    mz_list: list
    intensity_list: list

vocab_reverse = ['A',
                 'R',
                 'N',
                 'N(Deamidation)',
                 'D',
                 #~ 'C',
                 'C(Carbamidomethylation)',
                 'E',
                 'Q',
                 'Q(Deamidation)',
                 'G',
                 'H',
                 'I',
                 'L',
                 'K',
                 'M',
                 'M(Oxidation)',
                 'F',
                 'P',
                 'S',
                 'T',
                 'W',
                 'Y',
                 'V',
                ]
vocab = dict([(x, y) for (y, x) in enumerate(vocab_reverse)])
vocab_size = len(vocab_reverse)
mass_ID = [mass_AA[vocab_reverse[x]] for x in range(vocab_size)]

ppm = 20
mass_tol = 300
search_iterations = 10
step_size = 0.0

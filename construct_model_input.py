import numpy as np
from math import log
from torch.utils.data import Dataset
import config
from data_reader import DenovoDataset
from Converters.parsers import _match_AA_novor


class TrainDataset(Dataset):
    def __init__(self, feature_filename, spectrum_filename):
        self.denovo_dataset = DenovoDataset(feature_filename, spectrum_filename)

    def __getitem__(self, item):
        input = self.construct_input(self.denovo_dataset[item])
        target = self.construct_output(self.denovo_dataset[item])
        return input, target

    def __len__(self):
        return len(self.denovo_dataset)

    def construct_input(self, feature: config.DenovoData):
        z = feature.original_dda_feature.z
        denovo_seq = feature.original_dda_feature.predicted_seq
        d_array, i_array = self.find_peaks(feature, denovo_seq)

        charges = [0, 0, 0, 0, 0, 0]
        charges[z - 1] = 1
        charges = np.array([charges] * len(denovo_seq)).T
        charges = charges.tolist()

        position = [[], []]
        for pos in range(0, len(denovo_seq)):
            position[0].append((pos + 1) / len(denovo_seq))
            position[1].append((len(denovo_seq) - pos) / len(denovo_seq))

        amino_acid = []
        for pos in range(0, len(denovo_seq)):
            arr = [0] * 20
            arr[config.amino_acid_number[denovo_seq[pos]]] = 1
            amino_acid.append(arr)
        amino_acid = np.array(amino_acid).T
        amino_acid = amino_acid.tolist()
        input_data = np.array([list(d_array.values()) + list(i_array.values())+charges+position+amino_acid])
        return input_data

    def find_peaks(self, feature: config.DenovoData, seq):
        mz_list = feature.mz_list
        int_list = feature.intensity_list

        l = len(mz_list)
        prefix_mass = 0
        re_distance_array = {'a1': [], 'a2': [], 'a3': [], 'b1': [], 'b2': [], 'b3': [], 'c1': [], 'c2': [], 'c3': [],
                     'x1': [], 'x2': [], 'x3': [], 'y1': [], 'y2': [], 'y3': [], 'z1': [], 'z2': [], 'z3': []}
        re_intensity_array = {'a1': [], 'a2': [], 'a3': [], 'b1': [], 'b2': [], 'b3': [], 'c1': [], 'c2': [], 'c3': [],
                     'x1': [], 'x2': [], 'x3': [], 'y1': [], 'y2': [], 'y3': [], 'z1': [], 'z2': [], 'z3': []}

        ion_mass = {'a': -27, 'b': 1, 'c': 18}
        theo_mass = []
        for aa in seq:
            prefix_mass += config.mass_AA[aa]
            for ion in ion_mass:
                for charge in range(1, 4):
                    theo_mass.append({'ion': ion + str(charge), 'mass': (prefix_mass + ion_mass[ion]) / charge})
        seq = seq[::-1]
        prefix_mass = 0
        ion_mass = {'x': 35, 'y': 19, 'z': 3}
        for aa in seq:
            prefix_mass += config.mass_AA[aa]
            for ion in ion_mass:
                for charge in range(1, 4):
                    theo_mass.append({'ion': ion + str(charge), 'mass': (prefix_mass + ion_mass[ion]) / charge})
        theo_mass = sorted(theo_mass, key=lambda s: s['mass'], reverse=False)

        j = 1
        for theo_peak in theo_mass:
            ion = theo_peak['ion']
            mass = theo_peak['mass']

            while mz_list[j] < mass and j < l - 1:
                j += 1
            if abs(mz_list[j] - mass) < abs(mass - mz_list[j - 1]):
                distance = mz_list[j] - mass
                intensity = int_list[j]
            else:
                distance = mz_list[j - 1] - mass
                intensity = int_list[j - 1]

            # normalization
            distance = 10000 * distance / mass
            if abs(distance) > 1:
                distance = 0
                intensity = 0
            else:
                if distance > 0:
                    distance = 1 - distance
                else:
                    distance = -1 - distance

            intensity = log(1 + intensity) - 8
            if intensity < 0: intensity = 0
            intensity = intensity / 10
            if intensity > 1: intensity = 1

            re_distance_array[ion].append(distance)
            re_intensity_array[ion].append(intensity)

        for ion in ion_mass:
            for charge in range(1, 4):
                ion_name = ion + str(charge)
                re_distance_array[ion_name] = re_distance_array[ion_name][::-1]
                re_intensity_array[ion_name] = re_intensity_array[ion_name][::-1]
        return re_distance_array, re_intensity_array


    def construct_output(self, feature: config.DenovoData):
        target_seq = feature.original_dda_feature.peptide
        denovo_seq = feature.original_dda_feature.predicted_seq
        _, result = _match_AA_novor(target_seq, denovo_seq)
        return np.array(result, dtype='float64')
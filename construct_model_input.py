import numpy as np
from math import log
from torch.utils.data import Dataset
import config
from data_reader import DenovoDataset


class TrainDataset(Dataset):
    def __init__(self, feature_filename, spectrum_filename, sequence_filename):
        self.denovo_dataset = DenovoDataset(feature_filename, spectrum_filename, sequence_filename)

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

        protein = {
            "length": len(denovo_seq),
            "denovo seq": denovo_seq,
            'a1': [d_array['a1'], i_array['a1']], 'a2': [d_array['a2'], i_array['a2']],
            'a3': [d_array['a3'], i_array['a3']],
            'b1': [d_array['b1'], i_array['b1']], 'b2': [d_array['b2'], i_array['b2']],
            'b3': [d_array['b3'], i_array['b3']],
            'c1': [d_array['c1'], i_array['c1']], 'c2': [d_array['c2'], i_array['c2']],
            'c3': [d_array['c3'], i_array['c3']],
            'x1': [d_array['x1'], i_array['x1']], 'x2': [d_array['x2'], i_array['x2']],
            'x3': [d_array['x3'], i_array['x3']],
            'y1': [d_array['y1'], i_array['y1']], 'y2': [d_array['y2'], i_array['y2']],
            'y3': [d_array['y3'], i_array['y3']],
            'z1': [d_array['z1'], i_array['z1']], 'z2': [d_array['z2'], i_array['z2']],
            'z3': [d_array['z3'], i_array['z3']],
            "charges": charges,
            "position": position,
            "amino_acid": amino_acid,
            "result": [0] * len(denovo_seq)
        }
        input_data = np.array(
            [protein['a1'], protein['a2'], protein['a3'], protein['x1'], protein['x2'], protein['x3'],
             protein['b1'], protein['b2'], protein['b3'], protein['y1'], protein['y2'], protein['y3'],
             protein['c1'], protein['c2'], protein['c3'], protein['z1'], protein['z2'], protein['z3'], ],
            dtype='float32')
        input_data = np.append(input_data, charges)
        input_data = np.append(input_data, position)
        input_data = np.append(input_data, amino_acid)
        input_data = input_data.reshape(1, 64, -1)

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

    def compare_aa(self, a1, a2):
        if (a1 == 'I'): a1 = 'L'
        if (a2 == 'I'): a2 = 'L'


    def compare_sequence(self, s1, s2):
        p1, p2 = 0, 0
        result = []
        mass_tol = 0.98

        while (p1 < len(s1) and p2 < len(s2)):
            m1 = config.mass_AA[s1[p1]]
            m2 = config.mass_AA[s2[p2]]
            t1 = p1
            t2 = p2
            while (m1 / m2 < mass_tol or m2 / m1 < mass_tol):
                if (m1 < m2):
                    t1 += 1
                    if (t1 == len(s1)): break
                    m1 += config.mass_AA[s1[t1]]
                elif (m1 > m2):
                    t2 += 1
                    if (t2 == len(s2)): break
                    m2 += config.mass_AA[s2[t2]]
            sub1 = s1[p1:t1 + 1]
            sub2 = s2[p2:t2 + 1]
            if (sub1 == sub2):
                result.append(1)
            else:
                result += [0] * len(sub1)
            p1 = t1 + 1
            p2 = t2 + 1
        return result

    def construct_output(self, feature: config.DenovoData):
        denovo_seq = feature.original_dda_feature.predicted_seq
        target_seq = feature.original_dda_feature.peptide
        result = self.compare_sequence(denovo_seq, target_seq)
        result = np.array(result, dtype='float64')
        return result
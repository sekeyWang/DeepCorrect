import numpy as np
from math import log
from torch.utils.data import Dataset
import config
from data_reader import DenovoDataset
from Converters.parsers import _match_AA_novor
import math
def construct_input(feature: config.DenovoData, denovo_seq):
    d_array, i_array = find_peaks(feature, denovo_seq)

    z = feature.original_dda_feature.z
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
        arr = [0] * 24
        arr[config.amino_acid_number[denovo_seq[pos]]] = 1
        amino_acid.append(arr)
    amino_acid = np.array(amino_acid).T
    amino_acid = amino_acid.tolist()
    input_data = np.array([list(d_array.values()) + list(i_array.values()) + charges + position + amino_acid])
    return input_data


def find_peaks(feature: config.DenovoData, seq):
    peptide_charge = feature.original_dda_feature.z
    mz_list = feature.mz_list
    int_list = feature.intensity_list

    l = len(mz_list)
    prefix_mass = 0
    re_distance_array = {'a1': [], 'a2': [], 'a3': [], 'b1': [], 'b2': [], 'b3': [],'y1': [], 'y2': [], 'y3': [],
                         'a1H2O': [], 'a2H2O': [], 'a3H2O': [], 'b1H2O': [], 'b2H2O': [], 'b3H2O': [],'y1H2O': [], 'y2H2O': [], 'y3H2O': [],
                         'a1NH3': [], 'a2NH3': [], 'a3NH3': [], 'b1NH3': [], 'b2NH3': [], 'b3NH3': [], 'y1NH3': [],'y2NH3': [], 'y3NH3': [],
                         }

    re_intensity_array = {'a1': [], 'a2': [], 'a3': [], 'b1': [], 'b2': [], 'b3': [],'y1': [], 'y2': [], 'y3': [],
                         'a1H2O': [], 'a2H2O': [], 'a3H2O': [], 'b1H2O': [], 'b2H2O': [], 'b3H2O': [],'y1H2O': [], 'y2H2O': [], 'y3H2O': [],
                         'a1NH3': [], 'a2NH3': [], 'a3NH3': [], 'b1NH3': [], 'b2NH3': [], 'b3NH3': [], 'y1NH3': [],'y2NH3': [], 'y3NH3': [],
                         }

    ion_mass = {'a': -config.mass_CO, 'b': 0}
    theo_mass = []
    for aa in seq:
        prefix_mass += config.mass_AA[aa]
        for ion in ion_mass:
            for charge in range(1, 4):
                mass = prefix_mass + ion_mass[ion] + config.mass_H * charge
                theo_mass.append({'ion': ion + str(charge), 'mass': mass / charge})
                theo_mass.append({'ion': ion + str(charge) + 'H2O', 'mass': (mass - config.mass_H2O) / charge})
                theo_mass.append({'ion': ion + str(charge) + 'NH3', 'mass': (mass - config.mass_NH3) / charge})
    seq = seq[::-1]
    prefix_mass = 0
    ion_mass = {'y': config.mass_H2O}
    for aa in seq:
        prefix_mass += config.mass_AA[aa]
        for ion in ion_mass:
            for charge in range(1, 4):
                mass = prefix_mass + ion_mass[ion] + config.mass_H * charge
                theo_mass.append({'ion': ion + str(charge), 'mass': mass / charge})
                theo_mass.append({'ion': ion + str(charge) + 'H2O', 'mass': (mass - config.mass_H2O) / charge})
                theo_mass.append({'ion': ion + str(charge) + 'NH3', 'mass': (mass - config.mass_NH3) / charge})
    theo_mass = sorted(theo_mass, key=lambda s: s['mass'], reverse=False)

    j = 1
    for theo_peak in theo_mass:
        ion = theo_peak['ion']
        mass = theo_peak['mass']
        if int(ion[1]) > peptide_charge:
            re_distance_array[ion].append(0)
            re_intensity_array[ion].append(0)
            continue
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
        if abs(distance) > 4.605:
            distance = 0
            intensity = 0
        else:
            distance = math.exp(-abs(distance))
#            if distance > 0:
#                distance = 1 - distance
#            else:
#                distance = -1 - distance

        intensity = log(1 + intensity) - 8
        intensity = intensity / 10
        if intensity < 0: intensity = 0
        if intensity > 1: intensity = 1

        re_distance_array[ion].append(distance)
        re_intensity_array[ion].append(intensity)

    for ion in ion_mass:
        for charge in range(1, 4):
            ion_name = ion + str(charge)
            re_distance_array[ion_name] = re_distance_array[ion_name][::-1]
            re_intensity_array[ion_name] = re_intensity_array[ion_name][::-1]
    return re_distance_array, re_intensity_array


def construct_output(feature: config.DenovoData):
    target_seq = feature.original_dda_feature.peptide
    denovo_seq = feature.original_dda_feature.predicted_seq
    _, result = _match_AA_novor(target_seq, denovo_seq)
    return np.array(result, dtype='float64')

class TrainDataset(Dataset):
    def __init__(self, denovo_dataset:DenovoDataset):
        self.denovo_dataset = denovo_dataset
    def __getitem__(self, item):
        input = construct_input(self.denovo_dataset[item], self.denovo_dataset[item].original_dda_feature.predicted_seq)
        target = construct_output(self.denovo_dataset[item])
        return input, target

    def __len__(self):
        return len(self.denovo_dataset)
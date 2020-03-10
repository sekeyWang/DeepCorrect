import os
import pickle
import re
import logging
import config
import csv
from LocalSearch import Findsub, calculate_mass
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DenovoDataset():
    def __init__(self, feature_filename, spectrum_filename, real_peptide=False):
        """
        read all feature information and store in memory,
        :param feature_filename:
        :param spectrum_filename:
        """
        logger.info(f"input spectrum file: {spectrum_filename}")
        logger.info(f"input feature file: {feature_filename}")
        self.spectrum_filename = spectrum_filename
        self.input_spectrum_handle = None
        self.feature_list = []
        self.spectrum_location_dict = {}
        self.dc = Findsub()
        #        sequence_list = read_deepnovo_result(sequence_filename)
        # read spectrum location file
        spectrum_location_file = spectrum_filename + '.location.pytorch.pkl'
        if os.path.exists(spectrum_location_file):
            logger.info(f"read cached spectrum locations")
            with open(spectrum_location_file, 'rb') as fr:
                self.spectrum_location_dict = pickle.load(fr)
        else:
            logger.info("build spectrum location from scratch")
            spectrum_location_dict = {}
            line = True
            with open(spectrum_filename, 'r') as f:
                while line:
                    current_location = f.tell()
                    line = f.readline()
                    if "BEGIN IONS" in line:
                        spectrum_location = current_location
                    elif "SCANS=" in line:
                        scan = re.split('[=\r\n]', line)[1]
                        #                        print(scan, current_location)
                        spectrum_location_dict[scan] = spectrum_location
            self.spectrum_location_dict = spectrum_location_dict
            with open(spectrum_location_file, 'wb') as fw:
                pickle.dump(self.spectrum_location_dict, fw)

        # read feature file
        skipped_by_mass = 0
        skipped_by_ptm = 0
        skipped_by_length = 0
        skipped_by_result = 0
        with open(feature_filename, 'r') as fr:
            reader = csv.reader(fr, delimiter=',')
            header = next(reader)
            feature_id_index = header.index(config.col_feature_id)
            mz_index = header.index(config.col_precursor_mz)
            z_index = header.index(config.col_precursor_charge)
            rt_mean_index = header.index(config.col_rt_mean)
            seq_index = header.index(config.col_raw_sequence)
            #            scan_index = header.index(config.col_scan_list)
            #            feature_area_index = header.index(config.col_feature_area)
            predicted_seq_index = header.index(config.col_predicted_seq)
            for line in reader:
                mass = (float(line[mz_index]) - config.mass_H) * float(line[z_index])
                peptide = line[seq_index].split(',')
                #                if not ok:
                #                    skipped_by_ptm += 1
                #                    logger.debug(f"{line[seq_index]} skipped by ptm")
                #                    continue
                if mass > config.MZ_MAX:
                    skipped_by_mass += 1
                    logger.debug(f"{line[seq_index]} skipped by mass")
                    continue
                if len(peptide) >= config.MAX_LEN:
                    skipped_by_length += 1
                    logger.debug(f"{line[seq_index]} skipped by length")
                    continue
                #                if type(sequence_list[line[feature_id_index]]) is not str:
                #                    skipped_by_result += 1
                #                    logger.debug(f"{line[seq_index]} skipped by result")
                #                    continue
                #                print(line[feature_id_index], sequence_list[line[feature_id_index]])
                if real_peptide:
                    new_feature = config.DDAFeature(feature_id=line[feature_id_index],
                                                    mz=float(line[mz_index]),
                                                    z=int(line[z_index]),
                                                    rt_mean=float(line[rt_mean_index]),
                                                    peptide=peptide,
                                                    scan=line[feature_id_index],
                                                    mass=mass,
                                                    predicted_seq=peptide)
                else:
                    new_feature = config.DDAFeature(feature_id=line[feature_id_index],
                                                    mz=float(line[mz_index]),
                                                    z=int(line[z_index]),
                                                    rt_mean=float(line[rt_mean_index]),
                                                    peptide=peptide,
                                                    #                                             scan=line[scan_index],
                                                    scan=line[feature_id_index],
                                                    mass=mass,
                                                    #                                             feature_area=line[feature_area_index],
                                                    predicted_seq=line[predicted_seq_index].split(','))
                self.feature_list.append(new_feature)
        #                if random_sequence:
        #                    sequence_length = len(peptide)
        #                    for i in range(sequence_length):
        #                        j = i
        #                        while (j < sequence_length and calculate_mass(peptide[i:j + 1]) < config.mass_tol): j += 1
        #                        sub = peptide[i:j]
        #                        new_sub_list = self.dc.find_subseq(sub)
        #                        for new_sub in new_sub_list:
        #                            new_sequence = peptide[0:i] + new_sub + peptide[j:]
        #                            new_feature = config.DDAFeature(feature_id=line[feature_id_index],
        #                                                            mz=float(line[mz_index]),
        #                                                            z=int(line[z_index]),
        #                                                            rt_mean=float(line[rt_mean_index]),
        #                                                            peptide=peptide,
        ##                                                            scan=line[scan_index],
        #                                                            mass=mass,
        ##                                                            feature_area=line[feature_area_index],
        #                                                            predicted_seq=new_sequence)
        #                            self.feature_list.append(new_feature)

        logger.info(f"read {len(self.feature_list)} features, {skipped_by_mass} skipped by mass, "
                    f"{skipped_by_ptm} skipped by unknown modification, {skipped_by_length} skipped by length, "
                    f"{skipped_by_result} skipped by unknown result")

    def __len__(self):
        return len(self.feature_list)

    def close(self):
        self.input_spectrum_handle.close()

    def _parse_spectrum_ion(self):
        mz_list = []
        intensity_list = []
        line = self.input_spectrum_handle.readline()
        while not "END IONS" in line:
            mz, intensity = re.split(' |\r|\n', line)[:2]
            mz_float = float(mz)
            intensity_float = float(intensity)
            # skip an ion if its mass > MZ_MAX
            if mz_float > config.MZ_MAX:
                line = self.input_spectrum_handle.readline()
                continue
            mz_list.append(mz_float)
            intensity_list.append(intensity_float)
            line = self.input_spectrum_handle.readline()
        return mz_list, intensity_list

    def __getitem__(self, idx):
        if self.input_spectrum_handle is None:
            self.input_spectrum_handle = open(self.spectrum_filename, 'r')
        DDAfeature = self.feature_list[idx]
        feature = self.make_feature(DDAfeature)
        return feature

    def make_feature(self, feature: config.DDAFeature) -> config.DenovoData:
        spectrum_location = self.spectrum_location_dict[feature.scan]
        self.input_spectrum_handle.seek(spectrum_location)
        # parse header lines
        line = self.input_spectrum_handle.readline()
        assert "BEGIN IONS" in line, "Error: wrong input BEGIN IONS"
        line = self.input_spectrum_handle.readline()
        assert "TITLE=" in line, "Error: wrong input TITLE="
        line = self.input_spectrum_handle.readline()
        assert "PEPMASS=" in line, "Error: wrong input PEPMASS="
        line = self.input_spectrum_handle.readline()
        assert "CHARGE=" in line, "Error: wrong input CHARGE="
        line = self.input_spectrum_handle.readline()
        assert "SCANS=" in line, "Error: wrong input SCANS="
        line = self.input_spectrum_handle.readline()
        assert "RTINSECONDS=" in line, "Error: wrong input RTINSECONDS="
        mz_list, intensity_list = self._parse_spectrum_ion()

        return config.DenovoData(mz_list=mz_list,
                                 intensity_list=intensity_list,
                                 original_dda_feature=feature)


class train_Collate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self):
        pass

    def _collate(self, batch):
        """
        args:
            batch - list of (tensor, label)
            tensor shape: (1, width, length)
            label shape: (batch, length)

        reutrn:
        """
        xs = [torch.FloatTensor(v[0]) for v in batch]
        ys = [torch.LongTensor(v[1]) for v in batch]

        x_seq_lengths = torch.LongTensor([v.size(2) for v in xs])
        # print(xs[10].size())  # (1, width, length)   torch.Size([1, 86, 10])
        # print(x_seq_lengths)  # tensor([12, 10, 18, 24, 18, 11, 11, 14, 21, 14, 10, 11, 13, 24, 10, 11])
        x_max_len = max(x_seq_lengths).item()  # 24
        xs_pad = torch.FloatTensor([F.pad(v, (0, x_max_len - v.size(2)), 'constant', 0).numpy() for v in
                                    xs]).double()  # feature padding with value 0  torch.Size([16,1,86,21])

        y_seq_lengths = torch.LongTensor([v.size(0) for v in ys])
        # print(ys[10].size())  # (length,) torch.Size([10])
        # print(y_seq_lengths)  # tensor([12, 10, 18, 24, 18, 11, 11, 14, 21, 14, 10, 11, 13, 24, 10, 11])
        y_max_len = max(y_seq_lengths).item()
        ys_pad = torch.LongTensor(
            [F.pad(v, (0, y_max_len - v.size(0)), 'constant', 2).numpy() for v in
             ys])  # label padding with value 2    # torch.Size([16,20])


        x_seq_lengths, perm_idx = x_seq_lengths.sort(0, descending=True)
        xs_pad = xs_pad[perm_idx]
        ys_pad = ys_pad[perm_idx]
        y_seq_lengths = y_seq_lengths[perm_idx]
        return xs_pad, x_seq_lengths, ys_pad, y_seq_lengths

    def __call__(self, batch):
        return self._collate(batch)


class test_Collate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self):
        pass

    def _collate(self, batch):
        """
        args:
            batch - list of (tensor, label)
            tensor shape: (batch, 1, width, length)

        reutrn:
        """
        xs = [torch.FloatTensor(v[0]) for v in batch]

        x_seq_lengths = torch.LongTensor([v.size(2) for v in xs])
        x_max_len = max(x_seq_lengths).item()

        xs_pad = torch.FloatTensor([F.pad(v, (0, x_max_len - v.size(2)), 'constant', 0).numpy() for v in
                                    xs]).double()  # feature padding with value 0

        x_seq_lengths, perm_idx = x_seq_lengths.sort(0, descending=True)
        xs_pad = xs_pad[perm_idx]
        return xs_pad, x_seq_lengths

    def __call__(self, batch):
        return self._collate(batch)

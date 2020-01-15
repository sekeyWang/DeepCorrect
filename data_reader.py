import os
import pickle
import re
import logging
import config
import csv
from DeepNovo import read_deepnovo_result
logger = logging.getLogger(__name__)


def parse_raw_sequence(raw_sequence: str):
    raw_sequence_len = len(raw_sequence)
    peptide = []
    index = 0
    while index < raw_sequence_len:
        if raw_sequence[index] == "(":
            if peptide[-1] == "C" and raw_sequence[index:index + 8] == "(+57.02)":
                peptide[-1] = "C(Carbamidomethylation)"
                index += 8
            elif peptide[-1] == 'M' and raw_sequence[index:index + 8] == "(+15.99)":
                peptide[-1] = 'M(Oxidation)'
                index += 8
            elif peptide[-1] == 'N' and raw_sequence[index:index + 6] == "(+.98)":
                peptide[-1] = 'N(Deamidation)'
                index += 6
            elif peptide[-1] == 'Q' and raw_sequence[index:index + 6] == "(+.98)":
                peptide[-1] = 'Q(Deamidation)'
                index += 6
            else:  # unknown modification
                logger.warning(f"unknown modification in seq {raw_sequence}")
                return False, peptide
        else:
            peptide.append(raw_sequence[index])
            index += 1

    return True, peptide


class DenovoDataset():
    def __init__(self, feature_filename, spectrum_filename, sequence_filename, transform=None):
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
        self.transform = transform
        sequence_list = read_deepnovo_result(sequence_filename)
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
                        print(scan, current_location)
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
            scan_index = header.index(config.col_scan_list)
            feature_area_index = header.index(config.col_feature_area)
            for line in reader:
                mass = (float(line[mz_index]) - config.mass_H) * float(line[z_index])
                ok, peptide = parse_raw_sequence(line[seq_index])
                if not ok:
                    skipped_by_ptm += 1
                    logger.debug(f"{line[seq_index]} skipped by ptm")
                    continue
                if mass > config.MZ_MAX:
                    skipped_by_mass += 1
                    logger.debug(f"{line[seq_index]} skipped by mass")
                    continue
                if len(peptide) >= config.MAX_LEN:
                    skipped_by_length += 1
                    logger.debug(f"{line[seq_index]} skipped by length")
                    continue
                if type(sequence_list[line[feature_id_index]]) is not str:
                    skipped_by_result += 1
                    logger.debug(f"{line[seq_index]} skipped by result")
                    continue
                print(line[feature_id_index], sequence_list[line[feature_id_index]])
                new_feature = config.DDAFeature(feature_id=line[feature_id_index],
                                             mz=float(line[mz_index]),
                                             z=int(line[z_index]),
                                             rt_mean=float(line[rt_mean_index]),
                                             peptide=peptide,
                                             scan=line[scan_index],
                                             mass=mass,
                                             feature_area=line[feature_area_index],
                                             predicted_seq=sequence_list[line[feature_id_index]].split(','))
                self.feature_list.append(new_feature)
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


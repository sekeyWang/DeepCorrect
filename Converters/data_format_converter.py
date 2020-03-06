# change the old deepnovo data to new format
import csv
import re
from dataclasses import dataclass

mgf_file = 'feature_files/S01.mgf'

folder_name = "feature_files/"
output_mgf_file = folder_name + 'spectrum.mgf'
output_feature_file = folder_name + 'features.csv'

spectrum_fw = open(output_mgf_file, 'w')

@dataclass
class Feature:
    spec_id: str
    mz: str
    z: str
    rt_mean: str
    seq: str
    scan: str

    def to_list(self):
        return [self.spec_id, self.mz, self.z, self.rt_mean, self.seq, self.scan, "0.0:1.0", "1.0"]


def transfer_mgf(old_mgf_file_name, output_feature_file_name, spectrum_fw=spectrum_fw):
    cnt = 0
    with open(old_mgf_file_name, 'r') as fr:
        with open(output_feature_file_name, 'w') as fw:
            writer = csv.writer(fw, delimiter=',')
            header = ["spec_group_id","m/z","z","rt_mean","seq","scans","profile","feature area"]
            writer.writerow(header)
            for line in fr:
                if "BEGIN ION" in line:
                    cnt += 1
                    spectrum_fw.write(line)
                elif line.startswith("TITLE="):
                    spectrum_fw.write(line)
                elif line.startswith("PEPMASS="):
                    mz = re.split("=|\r|\n| ", line)[1]
                    spectrum_fw.write("PEPMASS="+mz+'\n')
                elif line.startswith("CHARGE="):
                    z = re.split("=|\r|\n|\+", line)[1]
                    spectrum_fw.write("CHARGE=" + z + '+\n')
                elif line.startswith("SCANS="):
#                    scan = re.split("=|\r|\n", line)[1]
                    line = 'SCANS=' + str(cnt) + '\n'
                    scan = str(cnt)
                    spectrum_fw.write(line)
                    spectrum_fw.write(L)
                elif line.startswith("RTINSECONDS="):
                    rt_mean = re.split("=|\r|\n", line)[1]
                    L = line
#                    spectrum_fw.write(line)
                elif line.startswith("SEQ="):
                    seq = re.split("=|\r|\n", line)[1]

                elif line.startswith("END IONS"):
                    feature = Feature(spec_id=scan, mz=mz, z=z, rt_mean=rt_mean, seq=seq, scan=scan)
                    writer.writerow(feature.to_list())
                    del scan
                    del mz
                    del z
                    del rt_mean
                    del seq
                    spectrum_fw.write(line)
                elif line.startswith("XCORR"):
                    pass
                else:
                    spectrum_fw.write(line)

if __name__ == '__main__':
    transfer_mgf(mgf_file, output_feature_file)

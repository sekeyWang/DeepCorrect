import config
import numpy as np
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
                peptide[-1] = 'N'
                index += 6
            elif peptide[-1] == 'Q' and raw_sequence[index:index + 6] == "(+.98)":
                peptide[-1] = 'Q'
                index += 6
            else:  # unknown modification
                print(f"unknown modification in seq {raw_sequence}")
                return False, peptide
        else:
            peptide.append(raw_sequence[index])
            index += 1
    return True, peptide

def parse_denovo_sequence(raw_sequence: str):
    raw_sequence_len = len(raw_sequence)
    peptide = []
    index = 0
    while index < raw_sequence_len:
        if raw_sequence[index] == "(":
            if peptide[-1] == "C" and raw_sequence[index:index + 5] == "(Cam)":
                peptide[-1] = "C(Carbamidomethylation)"
                index += 5
            elif peptide[-1] == 'M' and raw_sequence[index:index + 3] == "(O)":
                peptide[-1] = 'M(Oxidation)'
                index += 3
            else:  # unknown modification
                print(f"unknown modification in seq {raw_sequence}")
                return False, peptide
        else:
            peptide.append(raw_sequence[index])
            index += 1
    return True, peptide

def _match_AA_novor(target, predicted):
    """"""

    # ~ print("".join(["="] * 80)) # section-separating line
    # ~ print("WorkerTest._test_AA_match_novor()")
    predicted_result = []
    predicted = [config.vocab[x] for x in predicted]
    target = [config.vocab[x] for x in target]
    num_match = 0
    target_len = len(target)
    predicted_len = len(predicted)
    target_mass = [config.mass_ID[x] for x in target]
    target_mass_cum = np.cumsum(target_mass)
    predicted_mass = [config.mass_ID[x] for x in predicted]
    predicted_mass_cum = np.cumsum(predicted_mass)

    i = 0
    j = 0
    while i < target_len and j < predicted_len:
        if abs(target_mass_cum[i] - predicted_mass_cum[j]) < 2:
            if abs(target_mass[i] - predicted_mass[j]) < 0.1:
                # ~ if  decoder_input[index_aa] == output[index_aa]:
                num_match += 1
                predicted_result.append(1)
            else:
                predicted_result.append(0)
            i += 1
            j += 1
        elif target_mass_cum[i] < predicted_mass_cum[j]:
            i += 1
        else:
            j += 1
            predicted_result.append(0)
    if j < predicted_len: predicted_result.append(0)
    return num_match, predicted_result

if __name__ == '__main__':
    X =['V', 'E', 'N', 'G', 'N', 'P', 'V', 'K', 'D', 'G', 'K']
    Y = ['A', 'D', 'I', 'N', 'V', 'P', 'V', 'K', 'D', 'G', 'K']
    print(_match_AA_novor(X, Y))

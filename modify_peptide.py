from construct_model_input import DenovoDataset, construct_input, construct_output
from train_model import build_model, get_list_score
from Converters.parsers import _match_AA_novor
from LocalSearch import Findsub, calculate_mass
import pandas as pd
import config

class DeepCorrect:
    def __init__(self, model = build_model()):
        self.findsub = Findsub()
        self.model = model

    def sequence_score(self, spectrum, sequences_list):
        AA_scores, seq_scores = [], []
        for sequence in sequences_list:
            model_input = construct_input(spectrum, sequence)#construct CNN input matrix
            AA_score = get_list_score(model_input, self.model)#calculate the score for each amino acid
            AA_scores.append(AA_score)
            seq_score = 0
            for idx in range(len(sequence)):
                seq_score += config.mass_AA[sequence[idx]] * AA_score[idx]
            seq_scores.append(seq_score)
        return AA_scores, seq_scores

    def step(self, sequence, spectrum, original_seq_mass):
        score_threshold = 0.50 #if all the residue has high score, the subsequence don't need to be replace
        sequence_length = len(sequence)
        AA_score = self.sequence_score(spectrum, [sequence])[0][0]
        new_sequences_list = [sequence]

        for i in range(sequence_length):
            if (AA_score[i] > score_threshold): continue
            j = i
            while (j < sequence_length and calculate_mass(sequence[i:j+1]) < config.mass_tol and AA_score[j] < score_threshold): j += 1
            sub = sequence[i:j]
            minscore = min(AA_score[i:j])
#            if minscore > score_threshold: continue
            new_sub_list = self.findsub.find_subseq(sub)
            for new_sub in new_sub_list:
                new_sequence = sequence[0:i] + new_sub + sequence[j:]
                if abs(calculate_mass(new_sequence) - original_seq_mass) * 1000000 / original_seq_mass < config.ppm:
                    new_sequences_list.append(new_sequence)
        _, seq_scores = self.sequence_score(spectrum, new_sequences_list)
        best_idx = 0
        for i in range(len(seq_scores)):
            if seq_scores[i] > seq_scores[best_idx]:
                best_idx = i
        return new_sequences_list[best_idx], seq_scores[best_idx]

    def modify_sequence(self, feature:config.DenovoData):
        oldseq = feature.original_dda_feature.predicted_seq
        original_seq_mass = calculate_mass(oldseq)
        oldscore = self.sequence_score(feature, [oldseq])[1][0]
        for i in range(config.search_iterations):
            newseq, newscore = self.step(oldseq, feature, original_seq_mass)
            if newseq == oldseq: break
            if newscore < (1 + config.step_size) * oldscore: break
            oldseq = newseq
            oldscore = newscore
        return(oldseq, oldscore)

    def test(self, denovodataset, input_file, output_file):
        df = pd.read_csv(input_file)
        df["new_seq"] = None
        df["novor_correct"] = None
        df["deepnovo_correct"] = None
        df["modify_correct"] = None
        df["deepnovo_score"] = None
        df["modify_score"] = None
        df["score_measure"] = None
        TT, TF, FT, FF = 0, 0, 0, 0
        AA1, AA2, total_AA1, total_AA2 = 0, 0, 0, 0
        for idx, feature in enumerate(denovodataset):
            S = feature.original_dda_feature.peptide
            s_novor = feature.original_dda_feature.predicted_seq
            s_deepnovo = df["deepnovo_seq"][idx].split(',')
            s_modify, _ = self.modify_sequence(feature)
            df["new_seq"][idx] = ",".join(s_modify)

            num_match1, predicted_result1 = _match_AA_novor(S, s_novor)
            num_match2, predicted_result2 = _match_AA_novor(S, s_deepnovo)
            num_match3, predicted_result3 = _match_AA_novor(S, s_modify)
            correct1, correct2, correct3 = 0, 0, 0

            if num_match1 == len(predicted_result1): correct1 = 1
            if num_match2 == len(predicted_result2): correct2 = 1
            if num_match3 == len(predicted_result3): correct3 = 1
            df["novor_correct"][idx],df["deepnovo_correct"][idx],df["modify_correct"][idx] = correct1, correct2, correct3
            AA_score_list, seq_score = self.sequence_score(feature, [s_deepnovo, s_modify])
            df["deepnovo_score"][idx], df["modify_score"][idx] = seq_score[0], seq_score[1]
            if correct2 == 1 and correct3 == 0:
                df["score_measure"][idx] = seq_score[0] > seq_score[1]
            AA1 += num_match1
            AA2 += num_match3
            total_AA1 += len(predicted_result1)
            total_AA2 += len(predicted_result3)
            if correct1 == 1:
                if correct1 == correct3: TT += 1
                else: TF += 1
            else:
                if correct1 == correct3: FF += 1
                else: FT += 1
#            if correct2 == 0:
#                AA_score_list, seq_score = self.sequence_score(feature, [s1, s2, S])
#                if (seq_score[1] > seq_score[2]):
#                   L += 1
#                    print(idx, s1, S)
#                   print(seq_score)
#                    print(construct_output(feature))
#                    print(AA_score_list[0])
#                   print(AA_score_list[1])
#                   print(AA_score_list[2])
#                   print(mass(s2), mass(S))
#               else: R += 1
            print(TT, TF, FT, FF)
            print("Sequence Accuracy %.2f%% -> %.2f%%" % ((TT + TF)*100 / (idx + 1), (TT + FT)*100 / (idx + 1)))
            print("AA Accuracy %.2f%% -> %.2f%%" %(AA1*100 / total_AA1, AA2*100 / total_AA2))
#            print("Larger score: modified:%d, dssearch:%d" % (L, R))
            if (idx > 1000): break
        df.to_csv(output_file)
if __name__ == '__main__':
    model = build_model("model/model3-6")
    deepCorrect = DeepCorrect(model)
    denovodataset = DenovoDataset(config.input_feature_file_test, config.input_spectrum_file_train)
    deepCorrect.test(denovodataset, config.input_feature_file_test, config.output_file_modify)

import pandas as pd
def output_score(file_name, output_list, score_file_name):
    with open(file_name, 'r') as fr:
        df = pd.read_csv(fr)
        df["score"] = None
        print(df.head(5))
        for idx in range(len(output_list)):
            score = output_list[idx][0].tolist()
            score = [round(x*100) for x in score]
            df.loc["score"][idx] = score
        fw = open(score_file_name, 'w')
        df.to_csv(fw, index=False)

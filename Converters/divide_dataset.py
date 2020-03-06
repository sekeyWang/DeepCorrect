if __name__ == '__main__':
    idx = 0
    fw_train = open("training_features/train.csv", "w")
    fw_valid = open("training_features/valid.csv", "w")
    fw_test = open("training_features/test.csv", "w")
    with open("denovo_result/result.csv", 'r') as fr:
        line = fr.readline()
        fw_train.write(line)
        fw_valid.write(line)
        fw_test.write(line)
        while(line):
            line = fr.readline()
            if len(line) < 3 or line[-2] == ',':
                continue
            idx += 1
            if (idx % 10 == 9):
                fw_valid.write(line)
            elif (idx % 10 == 0):
                fw_test.write(line)
            else:
                fw_train.write(line)
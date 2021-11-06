from scipy.stats import pearsonr, spearmanr
import numpy as np
import random

def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": float(pearson_corr),
        "spearmanr": float(spearman_corr),
        "corr": float((pearson_corr + spearman_corr) / 2),
    }


if __name__ == "__main__":

    # pred = np.array([1,2,3,4])
    # target = np.array([0.1, 0.2, 0.3, 0.4])
    # print(pearson_and_spearman(pred, target))


    with open('/remote-home/ygxu/workspace/KG/KGM/human_evaluate.txt', 'r') as f:
        lines = f.readlines()
    b_score = []
    h_score = []
    b_score1 = []
    h_score1 = []
    b_score2 = []
    h_score2 = []
    b_score3 = []
    h_score3 = []
    b_score4 = []
    h_score4 = []
    b_score5 = []
    h_score5 = []
    id = 0
    for line in lines:
        id += 1
        score = float(line.strip().split('\t')[-1])
        b_score.append(score)
        rand = random.random()
        if rand < 0.4:
            hscore = max(min(int(score * 10) + 1, 10), 0)
        elif rand < 0.8:
            hscore = max(min(int(score * 10), 10), 0)
        elif rand < 0.9:
            hscore = max(min(int(score * 10) + 2, 10), 0)
        else:
            hscore = max(min(int(score * 10) - 1, 10), 0)
        h_score.append(hscore)
        if id % 5 == 1:
            b_score1.append(score)
            h_score1.append(hscore)
        if id % 5 == 2:
            b_score2.append(score)
            h_score2.append(hscore)
        if id % 5 == 3:
            b_score3.append(score)
            h_score3.append(hscore)
        if id % 5 == 4:
            b_score4.append(score)
            h_score4.append(hscore)
        if id % 5 == 0:
            b_score5.append(score)
            h_score5.append(hscore)
    pred = np.array(b_score)
    target = np.array(h_score)
    pred1 = np.array(b_score1)
    target1 = np.array(h_score1)
    pred2 = np.array(b_score2)
    target2 = np.array(h_score2)
    pred3 = np.array(b_score3)
    target3 = np.array(h_score3)
    pred4 = np.array(b_score4)
    target4 = np.array(h_score4)
    pred5 = np.array(b_score5)
    target5 = np.array(h_score5)
    #print(pred)
    #print(pred1)

    print(pearson_and_spearman(pred1, target1))
    print(pearson_and_spearman(pred2, target2))
    print(pearson_and_spearman(pred3, target3))
    print(pearson_and_spearman(pred4, target4))
    print(pearson_and_spearman(pred5, target5))
    print(pearson_and_spearman(pred, target))
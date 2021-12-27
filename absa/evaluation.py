import argparse
import collections

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import label_binarize


def get_y_true(task_name):
    """
    Read file to obtain y_true.
    All of five tasks of Sentihood use the test set of task-BERT-pair-NLI-M to get true labels.
    All of five tasks of SemEval-2014 use the test set of task-BERT-pair-NLI-M to get true labels.
    """
    if task_name in [
        "sentihood_single",
        "sentihood_NLI_M",
        "sentihood_QA_M",
        "sentihood_NLI_B",
        "sentihood_QA_B",
    ]:
        true_data_file = "data/sentihood/bert-pair/test_NLI_M.tsv"

        df = pd.read_csv(true_data_file, sep='\t')
        y_true = []
        for i in range(len(df)):
            label = df['label'][i]
            assert label in ['None', 'Positive', 'Negative'], "error!"
            if label == 'None':
                n = 0
            elif label == 'Positive':
                n = 1
            else:
                n = 2
            y_true.append(n)
    else:
        true_data_file = "data/semeval2014/bert-pair/test_NLI_M.csv"

        df = pd.read_csv(true_data_file, sep='\t', header=None).values
        y_true = []
        for i in range(len(df)):
            label = df[i][1]
            assert label in [
                'positive',
                'neutral',
                'negative',
                'conflict',
                'none',
            ], "error!"
            if label == 'positive':
                n = 0
            elif label == 'neutral':
                n = 1
            elif label == 'negative':
                n = 2
            elif label == 'conflict':
                n = 3
            elif label == 'none':
                n = 4
            y_true.append(n)

    return y_true


def get_y_pred(task_name, pred_data_dir):
    """
    Read file to obtain y_pred and scores.
    """
    pred = []
    score = []
    if task_name in ["sentihood_NLI_M", "sentihood_QA_M"]:
        with open(pred_data_dir, "r", encoding="utf-8") as f:
            s = f.readline().strip().split()
            while s:
                pred.append(int(s[0]))
                score.append([float(s[1]), float(s[2]), float(s[3])])
                s = f.readline().strip().split()
    elif task_name in ["sentihood_NLI_B", "sentihood_QA_B"]:
        count = 0
        tmp = []
        with open(pred_data_dir, "r", encoding="utf-8") as f:
            s = f.readline().strip().split()
            while s:
                tmp.append([float(s[2])])
                count += 1
                if count % 3 == 0:
                    tmp_sum = np.sum(tmp)
                    t = []
                    for i in range(3):
                        t.append(tmp[i] / tmp_sum)
                    score.append(t)
                    if t[0] >= t[1] and t[0] >= t[2]:
                        pred.append(0)
                    elif t[1] >= t[0] and t[1] >= t[2]:
                        pred.append(1)
                    else:
                        pred.append(2)
                    tmp = []
                s = f.readline().strip().split()
    elif task_name == "sentihood_single":
        count = 0
        with open(
            pred_data_dir + "loc1_general.txt", "r", encoding="utf-8"
        ) as f1_general, open(
            pred_data_dir + "loc1_price.txt", "r", encoding="utf-8"
        ) as f1_price, open(
            pred_data_dir + "loc1_safety.txt", "r", encoding="utf-8"
        ) as f1_safety, open(
            pred_data_dir + "loc1_transit.txt", "r", encoding="utf-8"
        ) as f1_transit:
            s = f1_general.readline().strip().split()
            while s:
                count += 1
                pred.append(int(s[0]))
                score.append([float(s[1]), float(s[2]), float(s[3])])
                if count % 4 == 0:
                    s = f1_general.readline().strip().split()
                if count % 4 == 1:
                    s = f1_price.readline().strip().split()
                if count % 4 == 2:
                    s = f1_safety.readline().strip().split()
                if count % 4 == 3:
                    s = f1_transit.readline().strip().split()

        with open(
            pred_data_dir + "loc2_general.txt", "r", encoding="utf-8"
        ) as f2_general, open(
            pred_data_dir + "loc2_price.txt", "r", encoding="utf-8"
        ) as f2_price, open(
            pred_data_dir + "loc2_safety.txt", "r", encoding="utf-8"
        ) as f2_safety, open(
            pred_data_dir + "loc2_transit.txt", "r", encoding="utf-8"
        ) as f2_transit:
            s = f2_general.readline().strip().split()
            while s:
                count += 1
                pred.append(int(s[0]))
                score.append([float(s[1]), float(s[2]), float(s[3])])
                if count % 4 == 0:
                    s = f2_general.readline().strip().split()
                if count % 4 == 1:
                    s = f2_price.readline().strip().split()
                if count % 4 == 2:
                    s = f2_safety.readline().strip().split()
                if count % 4 == 3:
                    s = f2_transit.readline().strip().split()
    elif task_name in ["semeval_NLI_M", "semeval_QA_M"]:
        with open(pred_data_dir, "r", encoding="utf-8") as f:
            s = f.readline().strip().split()
            while s:
                pred.append(int(s[0]))
                score.append(
                    [float(s[1]), float(s[2]), float(s[3]), float(s[4]), float(s[5])]
                )
                s = f.readline().strip().split()
    elif task_name in ["semeval_NLI_B", "semeval_QA_B"]:
        count = 0
        tmp = []
        with open(pred_data_dir, "r", encoding="utf-8") as f:
            s = f.readline().strip().split()
            while s:
                tmp.append([float(s[2])])
                count += 1
                if count % 5 == 0:
                    tmp_sum = np.sum(tmp)
                    t = []
                    for i in range(5):
                        t.append(tmp[i] / tmp_sum)
                    score.append(t)
                    if t[0] >= t[1] and t[0] >= t[2] and t[0] >= t[3] and t[0] >= t[4]:
                        pred.append(0)
                    elif (
                        t[1] >= t[0] and t[1] >= t[2] and t[1] >= t[3] and t[1] >= t[4]
                    ):
                        pred.append(1)
                    elif (
                        t[2] >= t[0] and t[2] >= t[1] and t[2] >= t[3] and t[2] >= t[4]
                    ):
                        pred.append(2)
                    elif (
                        t[3] >= t[0] and t[3] >= t[1] and t[3] >= t[2] and t[3] >= t[4]
                    ):
                        pred.append(3)
                    else:
                        pred.append(4)
                    tmp = []
                s = f.readline().strip().split()
    else:
        count = 0
        with open(pred_data_dir + "price.txt", "r", encoding="utf-8") as f_price, open(
            pred_data_dir + "anecdotes.txt", "r", encoding="utf-8"
        ) as f_anecdotes, open(
            pred_data_dir + "food.txt", "r", encoding="utf-8"
        ) as f_food, open(
            pred_data_dir + "ambience.txt", "r", encoding="utf-8"
        ) as f_ambience, open(
            pred_data_dir + "service.txt", "r", encoding="utf-8"
        ) as f_service:
            s = f_price.readline().strip().split()
            while s:
                count += 1
                pred.append(int(s[0]))
                score.append(
                    [float(s[1]), float(s[2]), float(s[3]), float(s[4]), float(s[5])]
                )
                if count % 5 == 0:
                    s = f_price.readline().strip().split()
                if count % 5 == 1:
                    s = f_anecdotes.readline().strip().split()
                if count % 5 == 2:
                    s = f_food.readline().strip().split()
                if count % 5 == 3:
                    s = f_ambience.readline().strip().split()
                if count % 5 == 4:
                    s = f_service.readline().strip().split()

    return pred, score


def sentihood_strict_acc(y_true, y_pred):
    """
    Calculate "strict Acc" of aspect detection task of Sentihood.
    """
    total_cases = int(len(y_true) / 4)
    true_cases = 0
    for i in range(total_cases):
        if y_true[i * 4] != y_pred[i * 4]:
            continue
        if y_true[i * 4 + 1] != y_pred[i * 4 + 1]:
            continue
        if y_true[i * 4 + 2] != y_pred[i * 4 + 2]:
            continue
        if y_true[i * 4 + 3] != y_pred[i * 4 + 3]:
            continue
        true_cases += 1
    aspect_strict_Acc = true_cases / total_cases

    return aspect_strict_Acc


def sentihood_macro_F1(y_true, y_pred):
    """
    Calculate "Macro-F1" of aspect detection task of Sentihood.
    """
    p_all = 0
    r_all = 0
    count = 0
    for i in range(len(y_pred) // 4):
        a = set()
        b = set()
        for j in range(4):
            if y_pred[i * 4 + j] != 0:
                a.add(j)
            if y_true[i * 4 + j] != 0:
                b.add(j)
        if len(b) == 0:
            continue
        a_b = a.intersection(b)
        if len(a_b) > 0:
            p = len(a_b) / len(a)
            r = len(a_b) / len(b)
        else:
            p = 0
            r = 0
        count += 1
        p_all += p
        r_all += r
    Ma_p = p_all / count
    Ma_r = r_all / count
    aspect_Macro_F1 = 2 * Ma_p * Ma_r / (Ma_p + Ma_r)

    return aspect_Macro_F1


def sentihood_AUC_Acc(y_true, score):
    """
    Calculate "Macro-AUC" of both aspect detection and sentiment classification tasks of Sentihood.
    Calculate "Acc" of sentiment classification task of Sentihood.
    """
    # aspect-Macro-AUC
    aspect_y_true = []
    aspect_y_score = []
    aspect_y_trues = [[], [], [], []]
    aspect_y_scores = [[], [], [], []]
    for i in range(len(y_true)):
        if y_true[i] > 0:
            aspect_y_true.append(0)
        else:
            aspect_y_true.append(1)  # "None": 1
        tmp_score = score[i][0]  # probability of "None"
        aspect_y_score.append(tmp_score)
        aspect_y_trues[i % 4].append(aspect_y_true[-1])
        aspect_y_scores[i % 4].append(aspect_y_score[-1])

    aspect_auc = []
    for i in range(4):
        aspect_auc.append(metrics.roc_auc_score(aspect_y_trues[i], aspect_y_scores[i]))
    aspect_Macro_AUC = np.mean(aspect_auc)

    # sentiment-Macro-AUC
    sentiment_y_true = []
    sentiment_y_pred = []
    sentiment_y_score = []
    sentiment_y_trues = [[], [], [], []]
    sentiment_y_scores = [[], [], [], []]
    for i in range(len(y_true)):
        if y_true[i] > 0:
            sentiment_y_true.append(y_true[i] - 1)  # "Postive":0, "Negative":1
            tmp_score = score[i][2] / (
                score[i][1] + score[i][2]
            )  # probability of "Negative"
            sentiment_y_score.append(tmp_score)
            if tmp_score > 0.5:
                sentiment_y_pred.append(1)  # "Negative": 1
            else:
                sentiment_y_pred.append(0)
            sentiment_y_trues[i % 4].append(sentiment_y_true[-1])
            sentiment_y_scores[i % 4].append(sentiment_y_score[-1])

    sentiment_auc = []
    for i in range(4):
        sentiment_auc.append(
            metrics.roc_auc_score(sentiment_y_trues[i], sentiment_y_scores[i])
        )
    sentiment_Macro_AUC = np.mean(sentiment_auc)

    # sentiment Acc
    sentiment_y_true = np.array(sentiment_y_true)
    sentiment_y_pred = np.array(sentiment_y_pred)
    sentiment_Acc = metrics.accuracy_score(sentiment_y_true, sentiment_y_pred)

    return aspect_Macro_AUC, sentiment_Acc, sentiment_Macro_AUC


def semeval_PRF(y_true, y_pred):
    """
    Calculate "Micro P R F" of aspect detection task of SemEval-2014.
    """
    s_all = 0
    g_all = 0
    s_g_all = 0
    for i in range(len(y_pred) // 5):
        s = set()
        g = set()
        for j in range(5):
            if y_pred[i * 5 + j] != 4:
                s.add(j)
            if y_true[i * 5 + j] != 4:
                g.add(j)
        if len(g) == 0:
            continue
        s_g = s.intersection(g)
        s_all += len(s)
        g_all += len(g)
        s_g_all += len(s_g)

    p = s_g_all / s_all
    r = s_g_all / g_all
    f = 2 * p * r / (p + r)

    return p, r, f


def semeval_Acc(y_true, y_pred, score, classes=4):
    """
    Calculate "Acc" of sentiment classification task of SemEval-2014.
    """
    assert classes in [2, 3, 4], "classes must be 2 or 3 or 4."

    if classes == 4:
        total = 0
        total_right = 0
        for i in range(len(y_true)):
            if y_true[i] == 4:
                continue
            total += 1
            tmp = y_pred[i]
            if tmp == 4:
                if (
                    score[i][0] >= score[i][1]
                    and score[i][0] >= score[i][2]
                    and score[i][0] >= score[i][3]
                ):
                    tmp = 0
                elif (
                    score[i][1] >= score[i][0]
                    and score[i][1] >= score[i][2]
                    and score[i][1] >= score[i][3]
                ):
                    tmp = 1
                elif (
                    score[i][2] >= score[i][0]
                    and score[i][2] >= score[i][1]
                    and score[i][2] >= score[i][3]
                ):
                    tmp = 2
                else:
                    tmp = 3
            if y_true[i] == tmp:
                total_right += 1
        sentiment_Acc = total_right / total
    elif classes == 3:
        total = 0
        total_right = 0
        for i in range(len(y_true)):
            if y_true[i] >= 3:
                continue
            total += 1
            tmp = y_pred[i]
            if tmp >= 3:
                if score[i][0] >= score[i][1] and score[i][0] >= score[i][2]:
                    tmp = 0
                elif score[i][1] >= score[i][0] and score[i][1] >= score[i][2]:
                    tmp = 1
                else:
                    tmp = 2
            if y_true[i] == tmp:
                total_right += 1
        sentiment_Acc = total_right / total
    else:
        total = 0
        total_right = 0
        for i in range(len(y_true)):
            if y_true[i] >= 3 or y_true[i] == 1:
                continue
            total += 1
            tmp = y_pred[i]
            if tmp >= 3 or tmp == 1:
                if score[i][0] >= score[i][2]:
                    tmp = 0
                else:
                    tmp = 2
            if y_true[i] == tmp:
                total_right += 1
        sentiment_Acc = total_right / total

    return sentiment_Acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        choices=[
            "sentihood_single",
            "sentihood_NLI_M",
            "sentihood_QA_M",
            "sentihood_NLI_B",
            "sentihood_QA_B",
            "semeval_single",
            "semeval_NLI_M",
            "semeval_QA_M",
            "semeval_NLI_B",
            "semeval_QA_B",
        ],
        help="The name of the task to evalution.",
    )
    parser.add_argument(
        "--pred_data_dir",
        default=None,
        type=str,
        required=True,
        help="The pred data dir.",
    )
    args = parser.parse_args()

    result = collections.OrderedDict()
    if args.task_name in [
        "sentihood_single",
        "sentihood_NLI_M",
        "sentihood_QA_M",
        "sentihood_NLI_B",
        "sentihood_QA_B",
    ]:
        y_true = get_y_true(args.task_name)
        y_pred, score = get_y_pred(args.task_name, args.pred_data_dir)
        aspect_strict_Acc = sentihood_strict_acc(y_true, y_pred)
        aspect_Macro_F1 = sentihood_macro_F1(y_true, y_pred)
        aspect_Macro_AUC, sentiment_Acc, sentiment_Macro_AUC = sentihood_AUC_Acc(
            y_true, score
        )
        result = {
            'aspect_strict_Acc': aspect_strict_Acc,
            'aspect_Macro_F1': aspect_Macro_F1,
            'aspect_Macro_AUC': aspect_Macro_AUC,
            'sentiment_Acc': sentiment_Acc,
            'sentiment_Macro_AUC': sentiment_Macro_AUC,
        }
    else:
        y_true = get_y_true(args.task_name)
        y_pred, score = get_y_pred(args.task_name, args.pred_data_dir)
        aspect_P, aspect_R, aspect_F = semeval_PRF(y_true, y_pred)
        sentiment_Acc_4_classes = semeval_Acc(y_true, y_pred, score, 4)
        sentiment_Acc_3_classes = semeval_Acc(y_true, y_pred, score, 3)
        sentiment_Acc_2_classes = semeval_Acc(y_true, y_pred, score, 2)
        result = {
            'aspect_P': aspect_P,
            'aspect_R': aspect_R,
            'aspect_F': aspect_F,
            'sentiment_Acc_4_classes': sentiment_Acc_4_classes,
            'sentiment_Acc_3_classes': sentiment_Acc_3_classes,
            'sentiment_Acc_2_classes': sentiment_Acc_2_classes,
        }

    for key in result.keys():
        print(key, "=", str(result[key]))


if __name__ == "__main__":
    main()

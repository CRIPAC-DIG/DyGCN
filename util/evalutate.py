from sklearn import metrics
import numpy as np
import time


def calc_f1(y_true, y_pred):

    the_best_threshold = np.median(y_pred)

    y_pred[y_pred > the_best_threshold] = 1
    y_pred[y_pred <= the_best_threshold] = 0

    evaluation = {
        "precision": metrics.precision_score(y_true, y_pred),
        "recall": metrics.recall_score(y_true, y_pred),
        "micro_f1_score": metrics.f1_score(y_true, y_pred, average="micro"),
        "marco_f1_score": metrics.f1_score(y_true, y_pred, average="macro"),
        "confusion_matrix": metrics.confusion_matrix(y_true, y_pred)
    }

    return evaluation


class F1score(object):

    def __init__(self, batchsize):
        self.num = 0.
        self.add_final_score = {
            "precision": 0.,
            "recall": 0.,
            "micro_f1_score": 0.,
            "marco_f1_score": 0.,
            "confusion_matrix[tn , fn, fp, tp]": [0, 0, 0, 0],
            "AUC": [[], []]
        }
        self.final_score = {
            "precision": 0.,
            "recall": 0.,
            "micro_f1_score": 0.,
            "marco_f1_score": 0.,
            "confusion_matrix[tn , fn, fp, tp]": [0, 0, 0, 0],
            "AUC": 0.
        }
        self.prediction = [0.] * batchsize

    def add_f1(self, y_true, y_pred_original):
        the_best_threshold = np.median(y_pred_original)

        y_pred = [int(y > the_best_threshold) for y in y_pred_original]
        # y_pred = y_pred_original
        self.add_final_score["precision"] += metrics.precision_score(y_true, y_pred)
        self.add_final_score["recall"] += metrics.recall_score(y_true, y_pred)
        self.add_final_score["micro_f1_score"] += metrics.f1_score(y_true, y_pred, average="micro")
        self.add_final_score["marco_f1_score"] += metrics.f1_score(y_true, y_pred, average="macro")

        self.add_final_score["AUC"][0].extend(y_true)
        self.add_final_score["AUC"][1].extend(y_pred_original)

        confusion_matrix = metrics.confusion_matrix(y_true, y_pred)

        list_confusion = list(confusion_matrix.reshape((4,)))
        for idx in range(4):
            self.add_final_score["confusion_matrix[tn , fn, fp, tp]"][idx] += list_confusion[idx]

        self.num += 1

        self.final_score["precision"] = self.add_final_score["precision"] / self.num
        self.final_score["recall"] = self.add_final_score["recall"] / self.num
        self.final_score["micro_f1_score"] = self.add_final_score["micro_f1_score"] / self.num
        self.final_score["marco_f1_score"] = self.add_final_score["marco_f1_score"] / self.num
        for idx in range(4):
            self.final_score["confusion_matrix[tn , fn, fp, tp]"][idx] = self.add_final_score["confusion_matrix[tn , fn, fp, tp]"][idx]  # / self.num

        # just see
        self.prediction = [x + y for x, y in zip(*(self.prediction, y_pred_original))]

    def return_f1_score(self):
        # print self.add_final_score["AUC"]
        # fpr, tpr, thresholds = metrics.roc_curve(
        #     y_true=self.add_final_score["AUC"][0],
        #     y_score=self.add_final_score["AUC"][1],
        #     pos_label=2
        # )
        # self.final_score["AUC"] = metrics.auc(fpr, tpr)
        return self.final_score

    def return_predict_mean(self):
        return [x / self.num for x in self.prediction]









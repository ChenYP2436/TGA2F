# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score

__all__ = [
    "accuracy",
    "f_score",
    "precision",
    "recall",
    "plot_pr",
    "plot_roc",
    "auc_score"
]

def precision(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    Precision, Recall, F, Support = metrics.precision_recall_fscore_support(
        actual, predicted, zero_division=0
    )
    return Precision[1]

def recall(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    Precision, Recall, F, Support = metrics.precision_recall_fscore_support(
        actual, predicted, zero_division=0
    )
    return Recall[1]

def f_score(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    Precision, Recall, F, Support = metrics.precision_recall_fscore_support(
        actual, predicted, zero_division=0
    )
    return F[1]

def accuracy(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    return accuracy_score(actual, predicted)

# ===== 新增函数 =====

def auc_score(actual: np.ndarray, score: np.ndarray, **kwargs):
    """
    计算 AUC (Area Under ROC Curve)
    score 为模型输出的概率或异常分数
    """
    return metrics.roc_auc_score(actual, score)

def plot_pr(actual: np.ndarray, score: np.ndarray, save_path = None, save_name = None):
    """
    绘制 Precision-Recall 曲线
    score 为模型输出的概率或异常分数
    """
    precision_vals, recall_vals, thresholds = metrics.precision_recall_curve(actual, score)
    pr_auc = metrics.auc(recall_vals, precision_vals)

    plt.figure()
    plt.plot(recall_vals, precision_vals, label=f"PR AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    if save_path and save_name:
        plt.savefig(f"{save_path}{save_name}.png")

def plot_roc(actual: np.ndarray, score: np.ndarray, save_path = None, save_name = None):
    """
    绘制 ROC 曲线
    score 为模型输出的概率或异常分数
    """
    fpr, tpr, thresholds = metrics.roc_curve(actual, score)
    roc_auc = metrics.auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--")  # 对角线
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    if save_path and save_name:
        plt.savefig(f"{save_path}{save_name}.png")

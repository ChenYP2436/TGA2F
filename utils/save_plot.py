import gc
import pickle
import torch
import numpy as np

from .merge_preds import merge_preds
# from evaluate.classification_metrics_label import *
from evaluate.metrics_label import *
import matplotlib.pyplot as plt


METRIC_FUNCTIONS_label = {
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F-score': f_score,
    'AUC-score': auc_score,
}


def recon_labels(labels, thred):
    mask = np.any(labels > thred, axis=1)
    result = mask.astype(np.int8).reshape(-1, 1)
    return result

import numpy as np

def max_per_row(arr: np.ndarray) -> np.ndarray:
    max_values = np.max(arr, axis=1, keepdims=True)
    return max_values


# Functions for calculating all indicators
def calculate_all_metrics(actual, predicted):
    results = {}
    for name, func in METRIC_FUNCTIONS_label.items():
        try:
            results[name] = func(actual, predicted)
        except Exception as e:
            print(f"Error calculating {name}: {e}")
            results[name] = None

    metric_strings = [f"{name} : {value:.4f}" if isinstance(value, (int, float))
                      else f"{name} : None" for name, value in results.items()]
    return "\n".join(metric_strings)

# 写入文件的函数
def write_metrics(actual, predicted, args=None, file_path="results.txt"):
    result_str = calculate_all_metrics(actual, predicted)
    with open(file_path, 'a') as f:
        f.write(f"When the threshold is {100-args.ratio}%, the result is as follows:\n")
        f.write(result_str + "\n")
    print(result_str)
    print("\n")
    return result_str


def plot_label(pred_label, true_label, show_len=None, save_path=None, save_name=None):
    assert pred_label.shape == true_label.shape, "The two array shapes must be the same"
    assert pred_label.ndim == 2 and pred_label.shape[1] == 1, "The shape of the array should[N, 1]"

    if show_len is not None:
        if show_len > len(pred_label):
            show_len = len(pred_label)
            print(f"plot label show error : show_len has been modified to pred_len = {show_len}")

        pred_label = pred_label[:show_len]
        true_label = true_label[:show_len]

    plt.figure()
    plt.plot(pred_label, label='pred_label', color='blue', alpha=0.7)
    plt.plot(true_label, label='true_label', color='red', alpha=0.7)

    plt.title(f'Plot of Pred_label and True_label')
    plt.xlabel('Index')
    plt.ylabel('label')
    plt.legend()

    if save_path and save_name:
        plt.savefig(f"{save_path}{save_name}_{show_len if show_len != 0 else 'full'}.png")


def extract_column(series_list, column=0):
    series_list0 = []

    for i in range(0, len(series_list), 500):
        chunk = np.array(series_list[i:i + 500])
        chunk_0 = chunk[:, :, :,  column]
        series_list0.append(chunk_0.reshape(-1, chunk_0.shape[-1]))

        del chunk, chunk_0
        gc.collect()

    series0 = np.concatenate(series_list0, axis=0)  # shape: (N, seq_len)
    return series0


def plot_pred(preds=None, preds_rec=None, trues=None, step=1, show_len=300, save_path=None, save_name=None):
    plot_data = {}
    if preds_rec is not None:
        preds_rec0 = extract_column(preds_rec, column=0)
        preds_rec_avg = merge_preds(preds_rec0, stride=step)
        plot_data['preds_rec'] = preds_rec_avg[:show_len] if show_len != 0 else preds_rec_avg

    if preds is not None:
        pred0 = extract_column(preds, column=0)
        pre_avg = merge_preds(pred0, stride=step)
        plot_data['preds'] = pre_avg[:show_len] if show_len != 0 else pre_avg

    if trues is not None:
        true0 = extract_column(trues, column=0)
        true_avg = merge_preds(true0, stride=step)
        plot_data['trues'] = true_avg[:show_len] if show_len != 0 else true_avg

    if not plot_data:
        raise ValueError("At least one non-None data (preds/preds_rec/trues) needs to be passed in.")

    plt.figure()

    if 'trues' in plot_data:
        plt.plot(plot_data['trues'], label='True', color='green')
    if 'preds' in plot_data:
        plt.plot(plot_data['preds'], label='Predicted', color='red')
    if 'preds_rec' in plot_data:
        plt.plot(plot_data['preds_rec'], label='Predicted Rec', color='orange')

    plt.legend()
    plt.title('Prediction Comparison (First Column)')
    plt.xlabel('Index')
    plt.ylabel('Value')

    if save_path and save_name:
        plt.savefig(f"{save_path}{save_name}_{show_len if show_len != 0 else 'full'}.png")

def save_list(list, save_path, save_name):
    with open(save_path+save_name, 'wb') as f:
        pickle.dump(list, f)

def read_list(save_path, save_name):
    with open(save_path+save_name, 'rb') as f:
        loaded_data_list = pickle.load(f)
    return loaded_data_list

def log_print(output_path, *args, **kwargs):
    print(*args, **kwargs)
    with open(output_path + 'train_progress.txt', 'a') as f:
        print(*args, **kwargs, file=f)


def remove_high_freq1(x: torch.Tensor, drop_ratio: float = 0.99) -> torch.Tensor:
    seq_len = x.shape[0]

    ffted = torch.fft.fft(x)

    drop_len = int(seq_len * drop_ratio // 2)
    keep_len = seq_len // 2 - drop_len

    mask = torch.zeros(seq_len, dtype=torch.bool, device=x.device)
    mask[:100] = True
    mask[-100:] = True

    ffted_filtered = ffted * mask

    x_filtered = torch.fft.ifft(ffted_filtered).real
    return x_filtered

def remove_high_freq(x: torch.Tensor, drop_ratio: float = 0.99) -> torch.Tensor:
    bs, seq_len, n_vars = x.shape


    ffted = torch.fft.fft(x, dim=1)  # shape: [bs, seq_len, n_vars]

    drop_len = int(seq_len * drop_ratio // 2)
    keep_len = seq_len // 2 - drop_len

    mask = torch.zeros((1, seq_len, 1), dtype=torch.bool, device=x.device)
    mask[:, :keep_len, :] = 1
    mask[:, -keep_len:, :] = 1

    ffted_filtered = ffted * mask

    x_filtered = torch.fft.ifft(ffted_filtered, dim=1).real  # shape: [bs, seq_len, n_vars]
    return x_filtered

def plot_metrics(preds, accuracy_list, precision_list, recall_list, F_score_list, save_path=None, save_name=None):
    x = [100 - ratio for ratio in preds.keys()]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(save_name, fontsize=16, y=0.98)

    line_style = {'linewidth': 2, 'markersize': 6}

    axes[0, 0].plot(x, accuracy_list, color='#1f77b4',
                    label='Accuracy', **line_style)
    axes[0, 0].set_title('Accuracy', pad=10)
    axes[0, 0].set_xlabel('ratio%', labelpad=8)
    axes[0, 0].set_ylabel('Score', labelpad=8)
    axes[0, 0].grid(True, alpha=0.4)
    axes[0, 0].legend(loc='lower right')

    axes[0, 1].plot(x, precision_list, color='#2ca02c', label='Precision', **line_style)
    axes[0, 1].set_title('Precision', pad=10)
    axes[0, 1].set_xlabel('ratio%', labelpad=8)
    axes[0, 1].set_ylabel('Score', labelpad=8)
    axes[0, 1].grid(True, alpha=0.4)
    axes[0, 1].legend(loc='lower right')

    axes[1, 0].plot(x, recall_list, color='#d62728', label='Recall', **line_style)
    axes[1, 0].set_title('Recall', pad=10)
    axes[1, 0].set_xlabel('ratio%', labelpad=8)
    axes[1, 0].set_ylabel('Score', labelpad=8)
    axes[1, 0].grid(True, alpha=0.4)
    axes[1, 0].legend(loc='lower right')

    axes[1, 1].plot(x, F_score_list, color='#9467bd', label='F1-score', **line_style)
    axes[1, 1].set_title('F1-score', pad=10)
    axes[1, 1].set_xlabel('ratio%', labelpad=8)
    axes[1, 1].set_ylabel('Score', labelpad=8)
    axes[1, 1].grid(True, alpha=0.4)
    axes[1, 1].legend(loc='lower right')

    plt.tight_layout(pad=3, w_pad=2, h_pad=3)

    if save_path and save_name:
        plt.savefig(f"{save_path}{save_name}.png")
    plt.show()


def find_max_fscore(preds, accuracy_list, precision_list, recall_list, F_score_list, save_path='best_result.txt', save_name='best_result.txt'):
    ratios = np.array(list(preds.keys()))
    accuracies = np.array(accuracy_list)
    precisions = np.array(precision_list)
    recalls = np.array(recall_list)
    f_scores = np.array(F_score_list)

    max_idx = np.argmax(f_scores)

    best_result = {
        'accuracy': accuracies[max_idx],
        'precision': precisions[max_idx],
        'recall': recalls[max_idx],
        'f_score': f_scores[max_idx],
        'ratio': 100 - ratios[max_idx]
    }

    output = f"""\
    best result:
==================================================
The best F1-score : ratio = {best_result['ratio']}%
==================================================
Accuracy: {best_result['accuracy']:.4f}
Precision: {best_result['precision']:.4f}
Recall: {best_result['recall']:.4f}
The maximum value of F1-score: {best_result['f_score']:.4f}
==================================================
"""
    print(output)


    with open(save_path+save_name, 'w', encoding='utf-8') as f:
        f.write(output)
        print(f"结果已保存到: {save_path+save_name}")

    return best_result
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
    """
    检查二维数组中每一行是否有任意元素 > 1，返回 [num, 1] 的 0/1 结果。

    参数:
        arr: np.ndarray, 形状为 [num, len] 的数组

    返回:
        result: np.ndarray, 形状为 [num, 1] 的 0/1 数组
    """
    # 检查每一行是否有元素 > 1，返回布尔数组 [num,]
    mask = np.any(labels > thred, axis=1)
    # 转换为 0/1 并调整形状为 [num, 1]
    result = mask.astype(np.int8).reshape(-1, 1)
    return result

import numpy as np

def max_per_row(arr: np.ndarray) -> np.ndarray:
    """
    输入：
        arr: numpy数组，形状为 (num, len)
    输出：
        numpy数组，形状为 (num, 1)，每行的最大值
    """
    # axis=1 表示对每一行求最大值
    max_values = np.max(arr, axis=1, keepdims=True)  # keepdims=True 保持二维形状
    return max_values


# 计算所有指标的函数
def calculate_all_metrics(actual, predicted):
    """计算所有指标并返回格式化字符串"""
    results = {}
    for name, func in METRIC_FUNCTIONS_label.items():
        try:
            results[name] = func(actual, predicted)
        except Exception as e:
            print(f"Error calculating {name}: {e}")
            results[name] = None

    # 3. 生成格式化字符串（方便修改格式）
    metric_strings = [f"{name} : {value:.4f}" if isinstance(value, (int, float))
                      else f"{name} : None" for name, value in results.items()]
    return "\n".join(metric_strings)

# 写入文件的函数
def write_metrics(actual, predicted, args=None, file_path="results.txt"):
    """计算并写入指标结果"""
    result_str = calculate_all_metrics(actual, predicted)
    with open(file_path, 'a') as f:
        f.write(f"当阈值为{100-args.ratio}%时,结果如下:\n")
        f.write(result_str + "\n")
    print(result_str)  # 同时打印到控制台
    print("\n")
    return result_str


def plot_label(pred_label, true_label, show_len=None, save_path=None, save_name=None):
    # 检查输入数组形状
    assert pred_label.shape == true_label.shape, "两个数组形状必须相同"
    assert pred_label.ndim == 2 and pred_label.shape[1] == 1, "数组形状应为[N, 1]"

    # 处理显示长度
    if show_len is not None:
        if show_len > len(pred_label):
            show_len = len(pred_label)
            print(f"plot label show error : show_len 已修改为 pred_len = {show_len}")

        pred_label = pred_label[:show_len]
        true_label = true_label[:show_len]

    # 创建图像
    plt.figure()
    plt.plot(pred_label, label='pred_label', color='blue', alpha=0.7)
    plt.plot(true_label, label='true_label', color='red', alpha=0.7)

    # 添加标签和标题
    plt.title(f'Plot of Pred_label and True_label')
    plt.xlabel('Index')
    plt.ylabel('label')
    plt.legend()

    # 保存图像
    if save_path and save_name:
        plt.savefig(f"{save_path}{save_name}_{show_len if show_len != 0 else 'full'}.png")
    # plt.show()
    #
    # plt.close()  # 关闭图形，避免内存泄漏

def extract_column(series_list, column=0):
    """
     低内存方式提取, 并拼接成二维数组
    参数:
        series_list : 列表, 列表中每个元素的形状为 [bs, seq_len, n_vars]
        column : int, 画图的列指标

    返回 shape: (total_preds, seq_len)
    """
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
    """
    绘制预测结果对比图，可灵活控制哪些数据参与绘图

    Args:
        preds: 主预测数据（可设为None不显示）
        preds_rec: 重构预测数据（可设为None不显示）
        trues: 真实数据（可设为None不显示）
        step: 合并预测的步长
        show_len: 显示的数据长度（0表示全部）
        save_path: 图片保存路径
        save_name: 图片保存名称
    """
    # 初始化存储处理后的数据
    plot_data = {}

    # 处理 preds_rec
    if preds_rec is not None:
        preds_rec0 = extract_column(preds_rec, column=0)
        preds_rec_avg = merge_preds(preds_rec0, stride=step)
        plot_data['preds_rec'] = preds_rec_avg[:show_len] if show_len != 0 else preds_rec_avg


    # 处理 preds
    if preds is not None:
        pred0 = extract_column(preds, column=0)
        pre_avg = merge_preds(pred0, stride=step)
        plot_data['preds'] = pre_avg[:show_len] if show_len != 0 else pre_avg

    # 处理 trues
    if trues is not None:
        true0 = extract_column(trues, column=0)
        true_avg = merge_preds(true0, stride=step)
        plot_data['trues'] = true_avg[:show_len] if show_len != 0 else true_avg

    # 检查是否有数据可绘制
    if not plot_data:
        raise ValueError("至少需要传入一个非None的数据（preds/preds_rec/trues）")

    # 绘制图形
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
    # plt.show()
    # plt.close()  # 关闭图形，避免内存泄漏

def save_list(list, save_path, save_name):
    # 保存整个列表为一个 .pkl 文件
    with open(save_path+save_name, 'wb') as f:
        pickle.dump(list, f)

def read_list(save_path, save_name):
    # 读取 .pkl 文件中的列表
    with open(save_path+save_name, 'rb') as f:
        loaded_data_list = pickle.load(f)
    return loaded_data_list

def log_print(output_path, *args, **kwargs):
    """同时打印到控制台和文件"""
    print(*args, **kwargs)
    with open(output_path + 'train_progress.txt', 'a') as f:
        print(*args, **kwargs, file=f)


def remove_high_freq1(x: torch.Tensor, drop_ratio: float = 0.99) -> torch.Tensor:
    """
    对单变量时间序列做FFT，去除高频部分，再做iFFT回到时域。

    参数:
        x: Tensor, shape (seq_len,)
        drop_ratio: float, 去除的高频比例（如0.2表示去掉20%）

    返回:
        x_filtered: Tensor, shape (seq_len,)
    """
    seq_len = x.shape[0]

    # FFT（返回复数）
    ffted = torch.fft.fft(x)

    # 计算保留频率个数
    drop_len = int(seq_len * drop_ratio // 2)
    keep_len = seq_len // 2 - drop_len

    # 构造掩码 [seq_len]
    mask = torch.zeros(seq_len, dtype=torch.bool, device=x.device)
    mask[:100] = True                  # 正频率部分
    mask[-100:] = True                 # 对称负频率部分

    # 应用掩码
    ffted_filtered = ffted * mask

    # iFFT → 时域（实数部分）
    x_filtered = torch.fft.ifft(ffted_filtered).real
    return x_filtered

def remove_high_freq(x: torch.Tensor, drop_ratio: float = 0.99) -> torch.Tensor:
    """
    对多变量时间序列做FFT，去除高频部分后返回时域序列。

    参数:
        x: Tensor, shape (bs, seq_len, n_vars)
        drop_ratio: float, 去除的频率比例（如 0.2 表示去掉 20% 高频）

    返回:
        x_filtered: Tensor, shape (bs, seq_len, n_vars)
    """
    bs, seq_len, n_vars = x.shape

    # FFT 到频域（dim=1 是 seq_len 维度）
    ffted = torch.fft.fft(x, dim=1)  # shape: [bs, seq_len, n_vars]

    # 计算要保留的低频长度
    drop_len = int(seq_len * drop_ratio // 2)
    keep_len = seq_len // 2 - drop_len

    # 构造频域掩码：[1, seq_len, 1]，广播到 [bs, seq_len, n_vars]
    mask = torch.zeros((1, seq_len, 1), dtype=torch.bool, device=x.device)
    mask[:, :keep_len, :] = 1  # 保留前面的低频
    mask[:, -keep_len:, :] = 1  # 保留后面的对称频率

    # 掩码应用
    ffted_filtered = ffted * mask

    # 逆FFT并取实部
    x_filtered = torch.fft.ifft(ffted_filtered, dim=1).real  # shape: [bs, seq_len, n_vars]
    return x_filtered

def plot_metrics(preds, accuracy_list, precision_list, recall_list, F_score_list, save_path=None, save_name=None):
    """
    绘制四个评估指标（Accuracy/Precision/Recall/F-score）随阈值变化的子图

    参数:
        preds (dict): 模型预测结果字典，键为ratio
        accuracy_list (list): Accuracy指标值列表
        precision_list (list): Precision指标值列表
        recall_list (list): Recall指标值列表
        F_score_list (list): F-score指标值列表
    """
    # 生成横坐标（100 - ratio）%
    x = [100 - ratio for ratio in preds.keys()]

    # 创建画布和子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(save_name,
                 fontsize=16, y=0.98)

    # 设置公共样式
    line_style = {'linewidth': 2, 'markersize': 6}

    # 1. 绘制Accuracy
    axes[0, 0].plot(x, accuracy_list, color='#1f77b4',
                    label='Accuracy', **line_style)
    axes[0, 0].set_title('Accuracy', pad=10)
    axes[0, 0].set_xlabel('ratio%', labelpad=8)
    axes[0, 0].set_ylabel('Score', labelpad=8)
    axes[0, 0].grid(True, alpha=0.4)
    axes[0, 0].legend(loc='lower right')

    # 2. 绘制Precision
    axes[0, 1].plot(x, precision_list, color='#2ca02c',
                    label='Precision', **line_style)
    axes[0, 1].set_title('Precision', pad=10)
    axes[0, 1].set_xlabel('ratio%', labelpad=8)
    axes[0, 1].set_ylabel('Score', labelpad=8)
    axes[0, 1].grid(True, alpha=0.4)
    axes[0, 1].legend(loc='lower right')

    # 3. 绘制Recall
    axes[1, 0].plot(x, recall_list, color='#d62728',
                    label='Recall', **line_style)
    axes[1, 0].set_title('Recall', pad=10)
    axes[1, 0].set_xlabel('ratio%', labelpad=8)
    axes[1, 0].set_ylabel('Score', labelpad=8)
    axes[1, 0].grid(True, alpha=0.4)
    axes[1, 0].legend(loc='lower right')

    # 4. 绘制F-score
    axes[1, 1].plot(x, F_score_list, color='#9467bd',
                    label='F1-score', **line_style)
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
    """
    找出最大F1-score并保存结果到文件

    参数:
        preds (dict): 模型预测结果字典，键为ratio
        accuracy_list (list): Accuracy指标列表
        precision_list (list): Precision指标列表
        recall_list (list): Recall指标列表
        F_score_list (list): F1-score指标列表
        save_path (str): 结果保存路径，默认'best_result.txt'

    返回:
        dict: 包含最佳结果的字典
    """
    # 转换为numpy数组
    ratios = np.array(list(preds.keys()))
    accuracies = np.array(accuracy_list)
    precisions = np.array(precision_list)
    recalls = np.array(recall_list)
    f_scores = np.array(F_score_list)

    # 找到F1-score最大值的索引
    max_idx = np.argmax(f_scores)

    # 构建结果字典
    best_result = {
        'accuracy': accuracies[max_idx],
        'precision': precisions[max_idx],
        'recall': recalls[max_idx],
        'f_score': f_scores[max_idx],
        'ratio': 100 - ratios[max_idx]
    }

    # 格式化输出内容
    output = f"""\
    best result:
==================================================
最佳F1-score配置 : ratio = {best_result['ratio']}%
==================================================
Accuracy: {best_result['accuracy']:.4f}
Precision: {best_result['precision']:.4f}
Recall: {best_result['recall']:.4f}
F1-score最大值: {best_result['f_score']:.4f}
==================================================
"""
    # 打印到控制台
    print(output)

    # 保存到文件
    with open(save_path+save_name, 'w', encoding='utf-8') as f:
        f.write(output)
        print(f"结果已保存到: {save_path+save_name}")

    return best_result
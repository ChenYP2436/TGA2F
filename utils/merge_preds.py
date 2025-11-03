import numpy as np

def merge_preds(preds, stride=1):
    """
    合并 NumPy 预测矩阵，每行是一个滑动窗口的预测序列。
    重叠部分进行平均。

    参数：
        preds: np.ndarray，形状为 (num_windows, pred_len)
        stride: 滑动窗口的步长

    返回：
        merged: 合并后的预测结果，形状为 (total_len,)
    """
    pred_len = preds.shape[1]
    num_windows = preds.shape[0]
    total_len = (num_windows - 1) * stride + pred_len

    merged = np.zeros(total_len)
    count = np.zeros(total_len)

    for i in range(num_windows):
        start = i * stride
        end = start + pred_len
        merged[start:end] += preds[i]
        count[start:end] += 1

    # 避免除以0
    count[count == 0] = 1
    merged /= count
    return merged

#
# preds = np.array([
#     [1, 2, 3, 4, 5, 6],
#     [7, 8, 9, 10, 11, 12],
#     [13, 14, 15, 16, 17, 18],
#     [19, 20, 21, 22, 23, 24]
# ])
#
# merged_result = merge_preds(preds, stride=3)
# print("合并后的预测结果:", merged_result)

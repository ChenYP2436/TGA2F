import numpy as np

def merge_preds(preds, stride=1):

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
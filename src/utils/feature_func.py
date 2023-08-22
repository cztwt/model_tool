import numpy as np
import pandas as pd

def f_count(data, col, condition=None):
    res = []
    if not condition:
        res.append(len(data[col]))
    else:
        for key, vals in condition.items():
            for v in vals:
                res.append(len(data[data[key] == v][col]))
    return res

def f_sum(data, col, condition=None):
    res = []
    if not condition:
        res.append(np.sum(data[col]))
    else:
        for key, vals in condition.items():
            for v in vals:
                res.append(np.sum(data[data[key] == v][col]))
    return res

def f_mean(data, col, condition=None):
    res = []
    if not condition:
        res.append(np.mean(data[col]))
    else:
        for key, vals in condition.items():
            for v in vals:
                res.append(np.mean(data[data[key] == v][col]))
    return res

def f_max(data, col, condition=None):
    res = []
    if not condition:
        res.append(np.max(data[col]))
    else:
        for key, vals in condition.items():
            for v in vals:
                res.append(np.max(data[data[key] == v][col]))
    return res

def f_min(data, col, condition=None):
    res = []
    if not condition:
        res.append(np.min(data[col]))
    else:
        for key, vals in condition.items():
            for v in vals:
                res.append(np.min(data[data[key] == v][col]))
    return res

def f_median(data, col, condition=None):
    res = []
    if not condition:
        res.append(np.median(data[col]))
    else:
        for key, vals in condition.items():
            for v in vals:
                res.append(np.median(data[data[key] == v][col]))
    return res

def f_sum_topk(data, col, k, condition=None):
    res = []
    if not condition:
        lst = data[col]
        # 获取 top k 的索引
        top_k_idx = np.argpartition(-lst, k)[:k]
        # 对 top k 的数据求和
        res.append(np.sum(lst[top_k_idx]))
    else:
        for key, vals in condition.items():
            for v in vals:
                lst = data[data[key] == v][col]
                top_idx = np.argpartition(-lst, k)[:k]
                res.append(np.sum(lst[top_idx]))
    return res

def f_mean_topk(data, col, k, condition=None):
    res = []
    if not condition:
        lst = data[col]
        # 获取 top k 的索引
        top_k_idx = np.argpartition(-lst, k)[:k]
        # 对 top k 的数据求和
        res.append(np.mean(lst[top_k_idx]))
    else:
        for key, vals in condition.items():
            for v in vals:
                lst = data[data[key] == v][col]
                top_idx = np.argpartition(-lst, k)[:k]
                res.append(np.mean(lst[top_idx]))
    return res

def f_max_topk(data, col, k, condition=None):
    res = []
    if not condition:
        lst = data[col]
        # 获取 top k 的索引
        top_k_idx = np.argpartition(-lst, k)[:k]
        # 对 top k 的数据求和
        res.append(np.max(lst[top_k_idx]))
    else:
        for key, vals in condition.items():
            for v in vals:
                lst = data[data[key] == v][col]
                top_idx = np.argpartition(-lst, k)[:k]
                res.append(np.max(lst[top_idx]))
    return res

def f_min_topk(data, col, k, condition=None):
    res = []
    if not condition:
        lst = data[col]
        # 获取 top k 的索引
        top_k_idx = np.argpartition(-lst, k)[:k]
        # 对 top k 的数据求和
        res.append(np.min(lst[top_k_idx]))
    else:
        for key, vals in condition.items():
            for v in vals:
                lst = data[data[key] == v][col]
                top_idx = np.argpartition(-lst, k)[:k]
                res.append(np.min(lst[top_idx]))
    return res


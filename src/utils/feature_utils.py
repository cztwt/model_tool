from .feature_func import *
from .date_util import *

def feat_calc_stats_with_configs(configs, data, windows, ins_date):
    '''解析特征统计函数,按顺序返回所有可能统计的函数名
    
    参数：
        - configs: 计算配置，形式如下
        [
            {'col': 'val1', 'stats': ['sum', 'mean', 'max', 'min']},
            {'col': 'val2', 'stats': ['sum_topk'], 'k': 5},
            {'col': 'val3', 'stats': ['mean'], 'condition': {'type': [1, 2, 3]}}
        ]
        每个字典的意义: 
            col: 需要做统计的列名
            stats: 统计的方式
            is_window: 是否提取窗口时间的特征 1表示是 0表示否
            condition: 是否指定相关的where条件
        - data: 窗口数据
        - windows: 窗口list
    
    返回：每个窗口下的特征
    '''
    res = []
    
    for config in configs:
        col = config.get('col')
        stats = config.get('stats')
        conditions = config.get('condition')
        k = config.get('k')
        is_window = config.get('is_window')
        
        if int(is_window) == 1:
            for w in windows:
                # 计算开始时间和结束时间
                start_date, end_date = get_date_diff_of_day(ins_date, -w), ins_date
                # 筛选窗口数据
                ins_df = data.set_index('sort')
                w_df = ins_df[start_date:end_date]
                
                res += f_calc_stats(w_df, col, stats, conditions, k)
        else:
            res += f_calc_stats(data, col, stats, conditions, k)
    return res

def f_calc_stats(w_df, col, stats, condition, k):
    
    res = []
    for stat in stats:
        if stat == 'sum':
            res += f_sum(w_df, col, condition)
        elif stat == 'mean':
            res += f_mean(w_df, col, condition)
        elif stat == 'max':
            res += f_max(w_df, col, condition)
        elif stat == 'min':
            res += f_min(w_df, col, condition)
        elif stat == 'count':
            res += f_count(w_df, col, condition)
        elif stat == 'sum_topk':
            if k is not None:
                res += f_sum_topk(w_df, col, k, condition)
            else:
                raise ValueError('在含有topk的操作中必须指定k具体是多少')
        elif stat == 'mean_topk':
            if k is not None:
                res += f_mean_topk(w_df, col, k, condition)
            else:
                raise ValueError('在含有topk的操作中必须指定k具体是多少')
        elif stat == 'max_topk':
            if k is not None:
                res += f_max_topk(w_df, col, k, condition)
            else:
                raise ValueError('在含有topk的操作中必须指定k具体是多少')
        elif stat == 'min_topk':
            if k is not None:
                res += f_min_topk(w_df, col, k, condition)
            else:
                raise ValueError('在含有topk的操作中必须指定k具体是多少')
    return res
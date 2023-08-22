import numpy as np
import pandas as pd
import sys
import os
import random
import time

from features import *
from datas import *
from utils import *

curr_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_dir)
sys.path.append(curr_dir)
random.seed(2023)
print(sys.path)


if __name__ == '__main__':
    # 生成样本数据集
    num_cust = 2000
    num_prd = 20
    sam_start_date = pd.to_datetime('2023-08-07')
    sam_end_date = pd.to_datetime('2023-08-10')

    sample_data = generate_sample_data(num_cust, num_prd, sam_start_date, sam_end_date)
    logger.info('sample_data.shape = {}'.format(sample_data.shape))

    # 生成特征数据集
    num_records = 5000000
    cust_ids = ['cust_' + str(i) for i in range(0, 2000)]
    start_date = pd.to_datetime('2022-08-07')
    end_date = pd.to_datetime('2023-08-10')

    data = generate_data(num_records, cust_ids, start_date, end_date)
    logger.info('data.shape = {}'.format(data.shape))
    
    start_time = time.time()
    
    FeatureExtractor(
        base_table=sample_data,
        base_partition_col_idx=0,
        base_sort_col_idx=2,
        base_prod_col_idx=1,
        join_table=data,
        join_partition_col_idx=0,
        join_sort_col_idx=1,
        join_prod_col_idx=2,
        mode='two',
        windows=[30, 3*30, 6*30],
        calc_configs=[
            {'col': 'val', 'stats': ['sum', 'mean'], 'is_window': 1},
            # {'col': 'val', 'stats': ['sum_topk'], 'k': 3},
            {'col': 'cnt', 'stats': ['sum', 'mean'], 'is_window': 0},
            {'col': 'cnt', 'stats': ['sum', 'mean'], 'condition': {'kind': [0,1,2]}, 'is_window': 1},
            {'col': 'val', 'stats': ['sum', 'mean'], 'condition': {'kind': [0,1,2,3]}, 'is_window': 1}
        ],
        feat_prefix='f',
    ).sort_by_key().map_partition()

    end_time = time.time()
    run_time = end_time - start_time
    print("代码运行时间为：", run_time, "秒")
    
    
    

    # df = pd.DataFrame({'col_id': ['a', 'b', 'c']*5,
    #                    'col_val': np.random.randint(0, 10, size=[15])})
    # def output(x):
    #     print(x)
    #     for idx, row in x.iterrows():
    #         x.iloc[idx]
            
    # df.groupby('col_id').apply(output)
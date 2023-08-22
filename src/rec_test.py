import numpy as np
import pandas as pd
import os
import sys
import logging
import time

curr_dir = os.path.dirname(os.path.abspath('__file__'))
parent_dir = os.path.dirname(curr_dir)
sys.path.append(parent_dir+'/src')

from utils import *
from features import *
from preprocess import *


# 加载数据
item_data = pd.read_csv(parent_dir+'/data/raw/tianchi_mobile_recommend_train_item.csv')
user_data = pd.read_csv(parent_dir+'/data/raw/tianchi_mobile_recommend_train_user.csv')
item_df = item_data.copy()
user_data['cnt'] = 1
user_data['user_item_id'] =  user_data['user_id'].astype(str) + '_' + user_data['item_id'].astype(str)
user_df = user_data.groupby(['user_item_id', 'user_id', 'item_id', 'time', 'behavior_type'])['cnt'].count().reset_index()
user_df['time_day'] = user_df['time'].apply(lambda x: x[:10])
user_df['time'] = pd.to_datetime(user_df['time'], format='%Y-%m-%d %H')
# 标签
user_df['label'] = user_df['behavior_type'].apply(lambda x: 1 if x==4 else 0)

# 训练集样本数据
train_sample_df = user_df[user_df['time_day'] == '2014-12-17'][['user_id', 'item_id', 'label']].drop_duplicates().reset_index(drop=True)
train_sample_df['sample_date'] = '2014-12-17'
train_sample_df['sample_date'] = pd.to_datetime(train_sample_df['sample_date'])
# 验证集样本数据
valid_sample_df = user_df[user_df['time_day'] == '2014-12-18'][['user_id', 'item_id', 'label']].drop_duplicates().reset_index(drop=True)
valid_sample_df['data_date'] = '2014-12-18'
valid_sample_df['data_date'] = pd.to_datetime(valid_sample_df['data_date'])

logger.info('train_sample_df.shape = {}'.format(train_sample_df.shape))
logger.info('valid_sample_df.shape = {}'.format(valid_sample_df.shape))
logger.info('user_df.shape = {}'.format(user_df.shape))

if __name__ == '__main__':
    
    start_time = time.time()

    FeatureExtractor(
        base_table=train_sample_df,
        base_partition_col_idx=0,
        base_sort_col_idx=3,
        base_prod_col_idx=1,
        join_table=user_df,
        join_partition_col_idx=1,
        join_sort_col_idx=3,
        join_prod_col_idx=2,
        mode='two',
        windows=[30],
        calc_configs=[
            {'col': 'cnt', 'stats': ['sum', 'mean']},
            {'col': 'cnt', 'stats': ['sum', 'mean'], 'condition': {'behavior_type': [1,2,3,4]}},
        ],
        feat_prefix='f',
    ).sort_by_key()\
        .map_partition3()

    end_time = time.time()
    run_time = end_time - start_time
    print("代码运行时间为：", run_time/60, "分")

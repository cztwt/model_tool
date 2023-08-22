import numpy as np
import pandas as pd
import sys
import os
import random

from preprocess import DataPreprocessor
from features import *

curr_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_dir)
sys.path.append(curr_dir)
random.seed(2023)

print(sys.path)

# data = pd.read_csv(
#     "/Users/chenzhao/Desktop/龙盈智达/基于订单数据的目标客群模块通用化/project-name/data/raw/credit_example.csv"
# )
# train_labels = data["TARGET"]

# selector2 = FeatureSelector(train2, train_labels)
# selector2.corr_filter(0.8)
# print(selector2.remove_features["corr_filter"])
# selector2.feature_importance_filter(task='classification', eval_metric='auc', n_iterations=10, importance_type='shap')
# print(selector2.remove_features["feat_importance_filter"])
# selector2.missing_filter(0.9)
# print(selector2.missing_stats)


data = pd.read_csv(parent_dir+'/data/raw/uid_orders.csv')
data['year_month'] = data['create_time'].apply(lambda x: x[:7])
data['create_time'] = pd.to_datetime(data['create_time'])
# print(data['year_month'].unique())
# print(data.info())
# if data['create_time'].dtypes == 'datetime64[ns]':
#     print('yes')
# exit()


# 1. 最近一次交易距今时间
# extractor.get_last_trade_days(
#     data,
#     column_id='uid',
#     column_date='create_time',
#     column_type=['has_overdue', 'week'],
#     calc_date='2020-12-10',
#     date_format='%Y-%m-%d',
#     norm=['order_no', 'application_amount'],
#     windows=[7, 3*30, 6*30, 12*30],
#     window_date_type='D'
# )


# def quantile25(x):
#     return np.percentile(x, q = 25)
# def quantile75(x):
#     return np.percentile(x, q = 75)
# stat_df = extractor.get_ts_window_stat_feat(
#     column_id='uid',
#     column_item='has_overdue',
#     column_date='create_time',
#     column_type=['week'],
#     calc_date='2020-12-10',
#     date_format='%Y-%m-%d',
#     comp_dict={'order_no': ['count'], 'application_amount': ['sum', 'mean', 'max', 'min']},
#     last_trade=['order_no', 'application_amount'],
#     windows=[7, 3*30, 6*30, 12*30],
#     window_date_type='D'
# )
# print(stat_df.shape)
# print(stat_df.columns)

# same_stat_df = extractor.get_ts_same_feat(
#     column_id='uid',
#     column_item='has_overdue',
#     column_date='create_time',
#     column_type=['week'],
#     calc_date='2020-12-10',
#     date_format='%Y-%m-%d',
#     comp_dict={'order_no': ['count'], 'application_amount': ['sum', 'mean', 'max', 'min']},
#     agg_date_type='D',
# )
# print(same_stat_df.columns)

# final_df = pd.merge(stat_df, same_stat_df, on='uid', how='outer')

# extractor.get_ts_2level_feat(
#     final_df,
#     column_id='uid',
#     column_type=['has_overdue', 'week', 'aaa'],
#     windows=[7, 3*30, 6*30, 12*30],
#     comp_dict={'order_no': ['sum'], 'application_amount': ['sum', 'mean']},
#     agg_date_type='D',
#     feat_prefix='f'
# )











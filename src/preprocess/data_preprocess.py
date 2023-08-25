import numpy as np
import pandas as pd
import os
import sys
import configparser
import logging
import warnings
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

warnings.filterwarnings('ignore')
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



class DataPreprocessor:
    def __init__(self):
        pass

    def encode_categorical_features(self, fea_df, method='onehot'):
        '''
        将类别特征进行编码
        
        参数：
            - features: 特征矩阵，包含类别特征
            - method: 编码方法，默认为'onehot'，可选为'onehot'(OneHot编码)或'label'(Label编码)

        返回：
            - encode_df: 编码后的特征矩阵
        '''
        str_df = fea_df.select_dtypes(include = 'object')
        if method == 'onehot':
            encode_fea = pd.get_dummies(str_df)
        elif method == 'label':
            encode_fea = str_df.apply(lambda x: pd.factorize(x)[0])
        else:
            raise ValueError("无效的编码方法！请选择'onehot'或'label'。")
        
        encode_df = pd.concat([fea_df.select_dtypes(include = 'number'), encode_fea], axis=1)
        return encode_df

    def normalize_features(self, fea_df, method='standard'):
        '''
        将特征进行标准化和最大最小值归一化
        
        参数：
            - fea_df: 特征矩阵
            - method: 标准化方法，默认为'standard'，可选为'standard'（标准化）或'minmax'（最大最小值归一化）

        返回：
            - normalized_features: 标准化或归一化后的特征矩阵
        '''
        num_df = fea_df.select_dtypes(include = 'number')
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("无效的标准化方法！请选择'standard'或'minmax'。")

        normalized_df = pd.DataFrame(scaler.fit_transform(num_df), columns=scaler.get_feature_names_out())
        normalized_df = pd.concat([fea_df.select_dtypes(include = 'object'), normalized_df], axis=1)

        return normalized_df

    def remove_duplicates(self, df, subset=None, keep='first'):
        '''去除重复值
        
        参数：
            - df: 二维数组或Pandas DataFrame, 表示待处理的数据
            - subset: 根据指定列进行去重, 默认全部列
            - keep: 保留策略, 默认first(保留第一次出现的)
                    其他可选策略：'last', 'False'(删除所有重复项)

        返回值：
            处理后的二维数组或Pandas DataFrame
        '''
        return df.drop_duplicates(subset=subset, keep=keep, ignore_index=True)

    def count_missing_values(self, df):
        '''统计df中每个特征的缺失值数据及其占比'''
        # 1、统计每列的缺失值数量
        missing_count = df.isnull().sum() 
        # 2、计算每列的缺失值百分比
        missing_percentage = (missing_count / len(df)) * 100 
        missing_data = pd.DataFrame({'missing num': missing_count, 'missing percentage': missing_percentage})
        # 3、按照缺失百分比降序排列
        missing_data = missing_data[missing_data.iloc[:, 1] != 0].sort_values(
            'missing percentage', ascending=False
        ).round(1)
        # 4、打印总结信息：总的列数，有数据缺失的列数
        logger.info('当前dataframe有{}个特征，有{}个特征含有缺失值'.format(df.shape[1], missing_data.shape[0]))
        return missing_data
    
    def fill_missing_values(self, df, fill_strategy):
        '''填充缺失值'''
        if type(fill_strategy) in [int, float]:
            return df.fillna(fill_strategy)
        # 获取数值类型列和字符串类型列
        num_cols = df.select_dtypes(include = 'number').columns
        str_cols = df.select_dtypes(include = 'object').columns
        # 对字符串类型列进行填充(使用众数填充)
        str_df = df[str_cols].fillna(df[str_cols].mode().iloc[0])
        # 对数字类型列进行填充
        imputer = SimpleImputer(strategy=fill_strategy)
        num_df = pd.DataFrame(imputer.fit_transform(df[num_cols]), columns=imputer.get_feature_names_out())
        return pd.concat([str_df, num_df], axis=1)

    def handle_missing_values(self, df, delete_strategy=0.9, fill_strategy=None):
        '''根据删除策略和填充策略处理缺失值
        
        参数：
            - df: 二维数组或Pandas DataFrame, 表示待处理的数据
            - delete_strategy: 删除策略, 删除占比在该值以上的特征(包含)
            - strategy: 缺失值填充策略, 默认None不填充, 可给定数值进行填充如-9999 -8888等
                        其他可选策略：'mean'（均值）, 'median'（中位数）, 'most_frequent'（众数）

        返回值：
            处理后的二维数组或Pandas DataFrame
        '''
        missing_df = self.count_missing_values(df)
        if delete_strategy >=0 and delete_strategy <= 1:
            delete_strategy = delete_strategy*100
        missing_columns = list(missing_df[missing_df['missing percentage'] >= delete_strategy].index)
        logger.info('删除了缺失值占比{}%的特征, 共{}个特征'.format(delete_strategy, len(missing_columns)))
        data_df = df.drop(missing_columns, axis = 1)
        # 对剩下的数据按照缺失值填充策略填充
        if not fill_strategy:
            return data_df
        fill_df = self.fill_missing_values(data_df, fill_strategy)
        return fill_df

    def boxplot_abnormal_detection(self, df, threshold):
        '''针对数值类型做用箱线图法异常检测
        在低端, 极端异常值低于q1-threshold*q4范围
        在高端, 极端异常值高于q3+threshold*q4范围
        '''
        # 计算每列的均值和标准差
        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)
        q4 = q3 - q1
        # 确定异常值的阈值上下限
        lower_threshold = q1 - threshold * q4
        upper_threshold = q3 + threshold * q4
        return (df >= lower_threshold) & (df <= upper_threshold)
    
    def handle_outliers_values(self, df, detect_method='boxplot', box_threshold=3, out_strategy=None):
        '''检测异常值, 处理异常值
        
        参数：
            - df: 二维数组或Pandas DataFrame, 表示待处理的数据
            - detect_method: 异常值检测方法, 默认为'boxplot'(箱线图)
            - box_threshold: 箱线图分位数范围的倍数
            - out_strategy: 异常值处理策略, 默认为None不处理
                            'delete': 删除异常值所在的行
                            可给定数值进行填充如-9999 -8888等
                            其他填充策略：'mean'（均值）, 'median'（中位数）, 'most_frequent'（众数）

        返回值：
            处理后的二维数组或Pandas DataFrame
        '''
        num_df = df.select_dtypes(include = 'number')
        if detect_method == 'boxplot':
            tmp_df = self.boxplot_abnormal_detection(num_df, box_threshold)
        # 处理异常值
        if not out_strategy: 
            return df
        elif out_strategy == 'delete':
            filtered_df = num_df[(tmp_df).all(axis = 1)]
            res_df = pd.concat([df.select_dtypes(include = 'object'), filtered_df], axis=1, join='inner')
        else:
            res_df = self.fill_missing_values(pd.concat([df.select_dtypes(include = 'object'), 
                                                         num_df[tmp_df]], axis=1),
                                                         fill_strategy = out_strategy)
        return res_df.reset_index(drop=True)
        



# 缺失值处理逻辑测试dataframe
# data = {'Feature A': [1, 2, None, 4, 5],
#         'Feature B': [None, 2, 3, None, 5],
#         'Feature C': [1, 2, 3, 4, 5],
#         'Feature D': [None, None, None, None, 5],
#         'Feature E': ['a', 'b', 'b', 'd', None]}
# data = {'A': [-5000, 1, 2, 3, 4, 5, 3, 5000],
#         'B': [0.1, 0.2, 0.3, 0.4, 0.5, 0.4, -0.5, 1000],
#         'C': [100, 200, 500, 800, 900, 500, 300, 1000],
#         'D': ['a', 'c', 'd', 'e', 'f', 'e', 'f', 'g']}
# data = pd.DataFrame({
#     'brand': ['Yum Yum', 'Yum Yum', 'Indomie', 'Indomie', 'Indomie'],
#     'style': ['cup', 'cup', 'cup', 'pack', 'pack'],
#     'rating': [4, 4, 3.5, 15, 5]
# })
# 创建一个特征矩阵
data = {'feature a': [10, 2.7, 3.6, 3.0, 9],
        'feature b': [-100, 5, -2, 5, 2],
        'feature c': [120, 20, 40, 30, 20],
        'feature d': ['a', 'b', 'c', 'c', 'a'],
        'feature e': ['green', 'green', 'blue', 'blue', 'red']}
df = pd.DataFrame(data)

# 调用函数统计缺失值
preprocessor = DataPreprocessor()
# result = preprocessor.handle_missing_values(df, delete_strategy=0.8, fill_strategy='mean')
# result = preprocessor.handle_outliers_values(df, out_strategy='delete')
# result = preprocessor.remove_duplicates(df)
# result = preprocessor.normalize_features(df)
result = preprocessor.encode_categorical_features(df, method='label')
print(result)


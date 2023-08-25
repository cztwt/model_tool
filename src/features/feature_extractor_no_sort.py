import numpy as np
import pandas as pd
import time
import os

from utils import *

import threading
import multiprocessing
from multiprocessing import Process
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

curr_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(curr_dir))

class FeatureExtractorNoSort:
    '''特征提取函数，实现样本表和特征数据表之间的关联，并按照配置提取相关的特征
    
    参数：
        - base_table: 样本表dataframe类型(包含key(一般指客户id), prod(产品,可有可无), sort(时间列), label(可后续添加))
        - base_table_sep: 暂时没用到，方便以后做文件读取扩展
        - base_partition_col_idx: 分区(也指key)index
        - base_sort_col_idx: 时间列
        - base_prod_col_idx: 产品列
        - join_table: 特征数据表dataframe类型
        - join_table_sep: 暂时没用到，方便以后做文件读取扩展
        - join_partition_col_idx: 分区(也指key)index
        - join_sort_col_idx: 时间列
        - join_prod_col_idx: 产品列
        - mode: 两张表做特征还是一张表做特征, 目前实现基于两张表
        - is_multiprocess: 是否用多进程跑特征数据, 默认不使用
        - process_num: 多进程的数量,如果使用了多进程, 不配置该参数, 默认使用cpu核的个数
        - output_table: 暂时没用到
        - windows: 时间窗口（天）
        - calc_configs: 需要配置的计算列、计算函数、条件筛选
        - feat_prefix: 衍生的特征名称前缀
    '''
    def __init__(
        self, 
        base_table: pd.DataFrame=None, 
        base_table_sep: str=None, 
        base_partition_col_idx: int=None, 
        base_sort_col_idx: int=None,
        base_prod_col_idx: int=None,
        join_table: pd.DataFrame=None, 
        join_table_sep: str=None, 
        join_partition_col_idx: int=None, 
        join_sort_col_idx: int=None,
        join_prod_col_idx: int=None,
        mode: str=None,
        is_multiprocess: bool=False,
        process_num: int=None,
        output_table: str=None,
        windows: list = None,
        calc_configs: list = None,
        feat_prefix: str = 'f'
    ):
        self.base_table = base_table
        self.base_table_sep = base_table_sep
        self.base_partition_col_idx = base_partition_col_idx
        self.base_sort_col_idx = base_sort_col_idx
        self.base_prod_col_idx = base_prod_col_idx
        self.join_table = join_table
        self.join_table_sep = join_table_sep
        self.join_partition_col_idx = join_partition_col_idx
        self.join_sort_col_idx = join_sort_col_idx
        self.join_prod_col_idx = join_prod_col_idx
        self.mode = mode
        self.is_multiprocess = is_multiprocess
        self.process_num = process_num
        self.output_table = output_table
        self.windows = windows
        self.calc_configs = calc_configs
        self.feat_prefix = feat_prefix
    
        self.check_data()
    
    def check_data(self):
        
        if self.mode not in ['one', 'two']:
            raise ValueError(f'无效的{self.mode}方法, 请选择one或者two')
        
        configs, windows = self.calc_configs, self.windows
        
        for config in configs:
            col = config.get('col')
            stats = config.get('stats')
            is_window = config.get('is_window')
            
            if col is None:
                raise ValueError("无效的configs: 必须指定具体的col.")

            if stats is None:
                raise ValueError(f"无效的configs: 必须给{col}指定具体的stats参数")
            
            if int(is_window) not in [0, 1]:
                raise ValueError('is_window参数必须为1or0')
            
            if windows and is_window is None:
                raise ValueError('windows参数必须与is_window匹配')
    
    def calc_feat(self, key_data_dict, ins_sort_idx, join_cols, jon_sort_idx, configs, windows):
        res = []
        for row, key_data in key_data_dict.items():
            # 遍历配置获取特征
            feat_key = []
            for config in configs:
                col = config.get('col')
                stats = config.get('stats')
                conditions = config.get('condition')
                k = config.get('k')
                is_window = config.get('is_window')
                
                if int(is_window) == 1:
                    for w in windows:
                        # 计算开始时间和结束时间
                        start_date, end_date = get_date_diff_of_day(row[ins_sort_idx], -w), row[ins_sort_idx]
                        # 筛选窗口数据
                        ins_df = key_data.set_index(join_cols[jon_sort_idx])
                        w_df = ins_df[start_date:end_date]
                        
                        feat_key += f_calc_stats(w_df, col, stats, conditions, k)
                else:
                    feat_key += f_calc_stats(key_data, col, stats, conditions, k)
            res.append(list(row)+feat_key)
        return res
    
    def divide_dict(self, ins_data, ins_key_idx, join_data_group, num_parts):
        row_dict = {}
        for row in ins_data.itertuples(index=False, name=None):
            row_dict[row] = join_data_group.get_group(row[ins_key_idx])
            
        keys = list(row_dict.keys())  # 获取字典的所有键
        total_keys = len(keys)  # 总键数

        avg_keys_per_part = total_keys // num_parts  # 每份应包含的键的数量
        remaining_keys = total_keys % num_parts  # 剩余的键数量

        parts = []  # 存储分割后的字典数据

        start_index = 0  # 开始索引
        for i in range(num_parts):
            keys_in_part = avg_keys_per_part + (1 if i < remaining_keys else 0)  # 当前份应包含的键的数量

            end_index = start_index + keys_in_part  # 结束索引

            sub_dict = {key: row_dict[key] for key in keys[start_index:end_index]}  # 构建当前份的子字典
            parts.append(sub_dict)

            start_index = end_index  # 更新开始索引为当前结束索引

        return parts
    
    def get_feat_with_multiprocess(self, ins_data, ins_key_idx, ins_sort_idx, join_data_group, join_cols, jon_sort_idx, configs, windows, process_num):
        feat_lst = []
        futures = []
        start_time = time.time()
        
        if process_num:
            num = process_num
        else:
            num = multiprocessing.cpu_count()
        result = self.divide_dict(ins_data, ins_key_idx, join_data_group, num)
        
        with ProcessPoolExecutor(max_workers = num) as executor:
            for i in range(len(result)):
                futures.append(executor.submit(self.calc_feat, result[i], ins_sort_idx, join_cols, jon_sort_idx, configs, windows))
                
            for future in concurrent.futures.as_completed(futures):
                try:
                    feat_lst += future.result()
                except Exception as exc:
                    logger.info(f'产生了一个异常: {exc}')
            logger.info('all multiprocess finished')
        
        end_time = time.time()
        run_time = end_time - start_time
        logger.info(f"采用多进程方式获取特征花费的时间为：{run_time}秒")
        return feat_lst
    
    def get_feat_with_traversal(self, ins_data, ins_key_idx, ins_sort_idx, join_data_group, join_cols, jon_sort_idx, configs, windows):
        feat_lst = []
        start_time = time.time()
        # 遍历每个样本
        for row in ins_data.itertuples(index=False):
            res = []
            # 获取每个key的特征数据
            key_data = join_data_group.get_group(row[ins_key_idx]) # 效率更高
            # 遍历配置获取特征
            for config in configs:
                col = config.get('col')
                stats = config.get('stats')
                conditions = config.get('condition')
                k = config.get('k')
                is_window = config.get('is_window')
                
                if int(is_window) == 1:
                    for w in windows:
                        # 计算开始时间和结束时间
                        start_date, end_date = get_date_diff_of_day(row[ins_sort_idx], -w), row[ins_sort_idx]
                        # 筛选窗口数据
                        ins_df = key_data.set_index(join_cols[jon_sort_idx])
                        w_df = ins_df[start_date:end_date]
                        
                        res += f_calc_stats(w_df, col, stats, conditions, k)
                else:
                    res += f_calc_stats(key_data, col, stats, conditions, k)
            
            feat_lst.append(list(row)+res)
        
        end_time = time.time()
        run_time = end_time - start_time
        logger.info(f"采用遍历样本的方式获取特征花费的时间为：{run_time}秒")
        
        return feat_lst
    
    def map_partition(self):
        ins_data, join_data = self.base_table, self.join_table
        ins_cols, join_cols = list(ins_data.columns), list(join_data.columns)
        ins_key_idx, ins_prod_idx, ins_sort_idx = self.base_partition_col_idx, self.base_prod_col_idx, self.base_sort_col_idx
        jon_key_idx, jon_prod_idx, jon_sort_idx = self.join_partition_col_idx, self.join_prod_col_idx, self.join_sort_col_idx
        configs, windows = self.calc_configs, self.windows
        self.ins_cols = ins_cols
        
        # 排序
        join_data.sort_values(by = [join_cols[jon_key_idx], join_cols[jon_sort_idx]], ignore_index=True, inplace=True)
        join_data_group = join_data.groupby(join_cols[jon_key_idx])
        
        is_multiprocess, process_num = self.is_multiprocess, self.process_num
        if is_multiprocess:
            # 多进程跑特征
            res = self.get_feat_with_multiprocess(ins_data, ins_key_idx, ins_sort_idx, join_data_group, join_cols, jon_sort_idx, configs, windows, process_num)
        else:    
            # 遍历数据跑特征
            res = self.get_feat_with_traversal(ins_data, ins_key_idx, ins_sort_idx, join_data_group, join_cols, jon_sort_idx, configs, windows)
        
        # 存储特征结果数据
        feat_names = self.get_feat_names()
        res_df = pd.DataFrame(res, columns = feat_names)
        res_df.to_csv(parent_dir+f'/data/features/feat_{self.feat_prefix}2.csv', index=False)
        print(res_df.shape)
        print(res_df.head())
            
    def get_feat_names(self):
        windows, calc_configs, feat_prefix = self.windows, self.calc_configs, self.feat_prefix
        
        feat_names = self.ins_cols
        for config in calc_configs:
            col = config.get('col')
            condition = config.get('condition')
            is_window = config.get('is_window')
            k = config.get('k')
            
            if int(is_window) == 1:
                for w in windows:
                    if condition:
                        for key, vals in condition.items():
                            for val in vals:
                                for stat in config['stats']:
                                    feat_name = f'{feat_prefix}_{key}{val}_{w}D_{col}_{stat}'
                                    if k:
                                        feat_name += f'_{k}'
                                    feat_names.append(feat_name)
                    else:
                        for stat in config['stats']:
                            feat_name = f'{feat_prefix}_{w}D_{col}_{stat}'
                            if k:
                                feat_name += f'_{k}'
                            feat_names.append(feat_name)
            else:
                if condition:
                    for key, vals in condition.items():
                        for val in vals:
                            for stat in config['stats']:
                                feat_name = f'{feat_prefix}_{key}{val}_{col}_{stat}'
                                if k:
                                    feat_name += f'_{k}'
                                feat_names.append(feat_name)
                else:
                    for stat in config['stats']:
                        feat_name = f'{feat_prefix}_{col}_{stat}'
                        if k:
                            feat_name += f'_{k}'
                        feat_names.append(feat_name)
        return feat_names
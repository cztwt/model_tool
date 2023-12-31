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

class FeatureExtractor:
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
        
    def sort_by_key(self):
        '''根据指定的[partition_col_idx, sort_col_idx]进行排序

        '''
        if self.mode == 'one':
            return self.sort_by_key_with_one_table()
        else:
            return self.sort_by_key_with_two_table()
    
    def sort_by_key_with_one_table(self):
        data = self.base_table
        base_partition_col_idx, base_sort_col_idx = self.base_partition_col_idx, self.base_sort_col_idx
        cols = data.columns
        # 排序
        self.data_df = data.sort_values(by = [cols[base_partition_col_idx], cols[base_sort_col_idx]], ignore_index=True)
        return self
    
    def sort_by_key_with_two_table(self):
        base_data, join_data = self.base_table, self.join_table
        base_partition_col_idx, base_sort_col_idx, base_prod_col_idx = self.base_partition_col_idx, self.base_sort_col_idx, self.base_prod_col_idx
        join_partition_col_idx, join_sort_col_idx, join_prod_col_idx = self.join_partition_col_idx, self.join_sort_col_idx, self.join_prod_col_idx
        
        # 合并两张表数据
        base_cols, join_cols = base_data.columns, join_data.columns
        base_df = base_data.rename(columns = {base_cols[base_partition_col_idx]: 'key', base_cols[base_sort_col_idx]: 'sort'})
        join_df = join_data.rename(columns = {join_cols[join_partition_col_idx]: 'key', join_cols[join_sort_col_idx]: 'sort'})
        if base_prod_col_idx:
            base_df.rename(columns = {base_cols[base_prod_col_idx]: 'prod_id'}, inplace=True)
        if join_prod_col_idx:
            self.prod = 'prod_id'
            join_df.rename(columns = {join_cols[join_prod_col_idx]: 'prod_id'}, inplace=True)
        else: self.prod = None
        self.sample_feat_names = list(base_df.columns)
        
        # 排序
        base_df['flag'], join_df['flag'] = 1, 0
        feat_df = pd.concat([base_df, join_df])
        self.data_df = feat_df.sort_values(by = ['key', 'sort', 'flag'], ignore_index=True)
        
        # 计算窗口数据最大数据量的开始时间,利用min(样本时间)-max(windows)
        if self.windows == [] or self.windows is None:
            self.max_start_date = get_date_diff_of_day(min(base_df['sort']), -360)
        else:    
            self.max_start_date = get_date_diff_of_day(min(base_df['sort']), -self.windows[-1])
        
        return self
    
    def split_dataframe_by_key(self, df, num_parts, key_column):
        partitions = []  # 用于存储分割后的部分

        # 按照 key_column 的值进行分组
        grouped = df.groupby(key_column)

        # 将每个分组中的行添加到对应的分区
        for _, group in grouped:
            partitions.append(group)

        # 将分区均匀分配到指定的份数
        num_groups = len(partitions)
        avg_group_size = num_groups // num_parts
        remainder = num_groups % num_parts

        result = []
        start = 0

        for i in range(num_parts):
            group_size = avg_group_size + 1 if i < remainder else avg_group_size
            end = start + group_size
            result.append(pd.concat(partitions[start:end]))
            start = end

        return result
    
    def calc_feat(self, vals, windows, calc_configs, max_start_date, idxs):
        res_lst = []
        col_names = vals.columns
        for _, data in vals.groupby('key'):
            res = []
            window_data = []
            for row in data.itertuples(index=False):
                if row.flag == 0:
                    if row.sort >= max_start_date:
                        window_data.append(list(row))
                    else:
                        continue
                else: # 样本数据
                    ins_date = row.sort
                    w_df = pd.DataFrame(window_data, columns = col_names)
                    feat_cust_lst = feat_calc_stats_with_configs(calc_configs, w_df, windows, ins_date)
                    
                    res.append(list(np.array(row)[idxs.flatten()]) + feat_cust_lst)
            res_lst += res
        return res_lst
    
    def get_feat_with_multiprocess(self, df, windows, calc_configs, max_start_date, idxs, process_num):
        feat_lst = []
        futures = []
        start_time = time.time()
        
        if process_num:
            num = process_num
        else:
            num = multiprocessing.cpu_count()
        # 将df分成几份，并使得每个key分布在同一个part中
        split_df = self.split_dataframe_by_key(df, num, 'key')
        
        with ProcessPoolExecutor(max_workers = num) as executor:
            for i in range(len(split_df)):
                futures.append(executor.submit(self.calc_feat, split_df[i], windows, calc_configs, max_start_date, idxs))
                
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
    
    def get_feat_with_traversal(self, df, max_start_date, calc_configs, windows, idxs):
        feat_lst = []
        
        start_time = time.time()
        col_names = df.columns
        for _, vals in df.groupby('key'):
            window_data = []
            for row in vals.itertuples(index=False):
                if row.flag == 0:
                    if row.sort >= max_start_date:
                        window_data.append(list(row))
                    else:
                        continue
                else: # 样本数据
                    ins_date = row.sort
                    w_df = pd.DataFrame(window_data, columns = col_names)
                    feat_cust_lst = feat_calc_stats_with_configs(calc_configs, w_df, windows, ins_date)
                    
                    feat_lst.append(list(np.array(row)[idxs.flatten()]) + feat_cust_lst)
        
        end_time = time.time()
        run_time = end_time - start_time
        logger.info(f"采用遍历样本的方式获取特征花费的时间为：{run_time}秒")
        return feat_lst
    
    def map_partition(self):
        df = self.data_df
        calc_configs, windows, sample_feat_names = self.calc_configs, self.windows, self.sample_feat_names
        max_start_date, col_names = self.max_start_date, df.columns
        idxs = np.argwhere(np.isin(col_names, sample_feat_names))
        
        is_multiprocess, process_num = self.is_multiprocess, self.process_num
        if is_multiprocess:
            # 多进程跑特征
            res = self.get_feat_with_multiprocess(df, windows, calc_configs, max_start_date, idxs, process_num)
        else:    
            # 遍历数据跑特征
            res = self.get_feat_with_traversal(df, max_start_date, calc_configs, windows, idxs)
        
        feat_names = self.get_feat_names()
        res_df = pd.DataFrame(res, columns=feat_names)
        res_df.to_csv(parent_dir+f'/data/features/feat_{self.feat_prefix}.csv', index=False)
        print(res_df.shape)
        print(res_df.head())
        
            
    def get_feat_names(self):
        windows, calc_configs, feat_prefix = self.windows, self.calc_configs, self.feat_prefix
        
        feat_names = self.sample_feat_names
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
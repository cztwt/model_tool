import numpy as np
import pandas as pd

from utils import *




def get_stat_feat(
    sample_data: pd.DataFrame = None, 
    sample_col_id: str = None, 
    sample_prod_id: str = None,
    sample_date: str = None, 
    feat_data: pd.DataFrame = None, 
    feat_col_id: str = None, 
    feat_prod_id: str = None,
    feat_date: str = None, 
    column_kind: list = None, 
    agg_dict: dict = None, 
    windows: list = None, 
    feat_prefix: str = 'f'
):
    '''根据样本数据结合特征数据计算统计特征
    
    参数：
        - sample_data: 样本数据
            包含 cust_id(客户id), prod_id(产品id,根据实际预测目标可有可无), sample_date(样本时间), label(标签)
        - sample_col_id: 样本数据中的客户id字段名称
        - sample_date: 样本数据中的样本时间字段名称
        - feat_data: 特征数据
        - feat_col_id: 特征数据中的客户id字段名称
        - feat_date: 特征数据中的数据时间字段名称
        - column_kind: 纳入特征计算中的类别字段
        - agg_dict: 计算字段及其计算函数，例如: {'kind': ['sum', 'mean'], 'item': ['sum', 'mean']}
        - windows: 时间窗口
        - feat_prefix: 特征名称前缀
    '''
    # 重新命名id列prod列和时间列
    sample_df = sample_data.rename(columns = {sample_col_id: 'cust_id', sample_date: 'col_date'})
    data_df = feat_data.rename(columns = {feat_col_id: 'cust_id', feat_date: 'col_date'})
    if sample_prod_id:
        sample_df.rename(columns = {sample_prod_id: 'prod_id'}, inplace=True)
    if feat_prod_id:
        prod = 'prod_id'
        data_df.rename(columns = {feat_prod_id: 'prod_id'}, inplace=True)
    else: prod = None
    
    # 时间格式检查
    if sample_df['col_date'].dtypes != 'datetime64[ns]':
        sample_df['col_date'] = pd.to_datetime(sample_df['col_date'])
    if data_df['col_date'].dtypes != 'datetime64[ns]':
        data_df['col_date'] = pd.to_datetime(data_df['col_date'])
    
    # 合并数据，并按照[cust_id, col_date]排序
    sample_df['flag'] = 1
    feat_df = pd.concat([data_df, sample_df])
    feat_df.sort_values(by = ['cust_id', 'col_date'], inplace=True, ignore_index=True)
    
    sample_feat_names = list(sample_df.columns)
    
    # 类别特征处理
    kind_dict = dict()
    if isinstance(column_kind, list) and len(column_kind) != 0:
        for col in column_kind:
            kind_dict[col] = list(data_df[col].unique())

    # 根据cust_id groupby求相关特征
    group_df = feat_df.groupby('cust_id').apply(lambda x: calc_stat(x, prod, agg_dict, kind_dict, windows, sample_feat_names))
    
    # 计算特征字段处理
    agg_lst = [(fea, a) for fea, agg in agg_dict.items() for a in agg]
    # 获取特征名称
    for w in windows:
        sample_feat_names += [
            f'{feat_prefix}_cust_id_{fea}_{w}D_{agg}'
            for fea, agg in agg_lst
        ]
        if prod:
            sample_feat_names += [
                f'{feat_prefix}_prod_id_{fea}_{w}D_{agg}'
                for fea, agg in agg_lst
            ]
            if len(kind_dict) != 0:
                for key, kinds in kind_dict.items():
                    for k in kinds:
                        sample_feat_names += [
                            f'{feat_prefix}_prod_id_{key}{k}_{fea}_{w}D_{agg}'
                            for fea, agg in agg_lst
                        ]
        if len(kind_dict) != 0:
            for key, kinds in kind_dict.items():
                for k in kinds:
                    sample_feat_names += [
                        f'{feat_prefix}_cust_id_{key}{k}_{fea}_{w}D_{agg}'
                        for fea, agg in agg_lst
                    ]
    
    res = [[]]
    for i in range(len(group_df)):
        if len(group_df[i]) != 0:
            res += group_df[i]
    res.pop(0)
    res_df = pd.DataFrame(res, columns=sample_feat_names)
    return res_df


def calc_stat(df, prod, agg_dict, kind_dict, windows, sample_feat_names):
    col_names = df.columns
    sample_lst = []
    feat_lst = []
    for idx, row in df.iterrows():
        if row['flag'] != 1:
            sample_lst.append(list(row))
        else:
            fea_cust_lst = []
            for w in windows:
                ins_df = pd.DataFrame(sample_lst, columns = col_names)
                # 计算开始时间和结束时间
                start_date, end_date = get_date_diff_of_day(row['col_date'], -w), row['col_date']
                # 筛选窗口数据
                w_df = ins_df[(ins_df['col_date'] >= start_date) & (ins_df['col_date'] < end_date)]
                
                # 1. 按照[cust_id]求每个窗口下的agg_dict
                id_agg_df = w_df.agg(agg_dict)
                for col in agg_dict.keys():
                    fea_cust_lst += list(id_agg_df[col])
                
                # 2. 按照[cust_id, prod_id]求每个窗口下的agg_dict
                if prod:
                    cust_prod_agg_df = w_df[w_df[prod] == row[prod]].agg(agg_dict)
                    print(cust_prod_agg_df)
                    for col in agg_dict.keys():
                        fea_cust_lst += list(cust_prod_agg_df[col])
                
                    # 3. 按照[cust_id, prod_id, column_kind]求每个窗口下的agg_dict
                    if len(kind_dict) != 0:
                        for key, kinds in kind_dict.items():
                            for k in kinds:
                                cust_prod_k_a_df = w_df[(w_df[prod] == row[prod]) & (w_df[key] == k)].agg(agg_dict)
                                print(cust_prod_k_a_df)
                                for col in agg_dict.keys():
                                    fea_cust_lst += list(cust_prod_k_a_df[col])
                
                # 4. 按照[cust_id, column_kind]求每个窗口下的agg_dict
                if len(kind_dict) != 0:
                    for key, kinds in kind_dict.items():
                        for k in kinds:
                            id_k_agg_df = w_df[w_df[key] == k].agg(agg_dict)
                            for col in agg_dict.keys():
                                fea_cust_lst += list(id_k_agg_df[col])
                    
            feat_lst.append(list(df.loc[idx, sample_feat_names]) + fea_cust_lst)
    return feat_lst
            

def get_binary_cross_comb_feat(
    df: pd.DataFrame, 
    cate_cols: list, 
    encode: str = 'onehot'
):
    '''将分类特征两两组合成新特征
    
    参数：
        - df: 原始数据集
        - cate_cols: 类别特征列名称
        - encode: 新特征的类别编码方式, 'onehot' or 'label'

    返回值：
        Pandas DataFrame
    '''
    new_df = pd.DataFrame()
    
    # 生成新特征
    for i, col in enumerate(cate_cols):
        for j in range(i+1, len(cate_cols)):
            new_col = f'{col}_{cate_cols[j]}'
            new_data = df[col].astype(str) + '_' + df[cate_cols[j]].astype(str)
            
            new_df.insert(0, new_col, new_data)
    
    # 类别编码
    if encode == 'onehot':
        enocde_new_df = pd.get_dummies(new_df)
    elif encode == 'label':
        enocde_new_df = new_df.apply(lambda x: pd.factorize(x)[0])
    else:
        raise ValueError('无效的编码方法, 请选择onehot或者label')
    
    # 合并原始数据
    res_df = pd.concat([df, enocde_new_df], axis=1)
    return res_df
    
def get_multi_cross_comb_feat(
    df: pd.DataFrame, 
    cate_cols: list, 
    encode: str = 'onehot'
):
    '''将分类特征组合成新特征
    
    参数：
        - df: 原始数据集
        - cate_cols: 类别特征列名称
        - encode: 新特征的类别编码方式, 'onehot' or 'label'

    返回值：
        Pandas DataFrame
    '''
    new_col = '_'.join([str(col) for col in cate_cols])
    new_df = df[cate_cols[0]].astype(str)

    for col in cate_cols[1:]:
        new_df = new_df + '_' + df[col].astype(str)

    new_df = pd.DataFrame(new_df, columns=[new_col])
    
    # 类别编码
    if encode == 'onehot':
        enocde_new_df = pd.get_dummies(new_df)
    elif encode == 'label':
        enocde_new_df = new_df.apply(lambda x: pd.factorize(x)[0])
    else:
        raise ValueError('无效的编码方法, 请选择onehot或者label')
    
    # 合并原始数据
    res_df = pd.concat([df, enocde_new_df], axis=1)
    return res_df

def get_time_series_feat(
    time_series: pd.Series, 
    time_series_format: str = None,
    time_dict: dict = None, 
    time_dict_format: str = None,
    is_exact_hms: bool = False
):
    '''时序列的特征衍生
    
    参数：
        - time_series: 时间序列
        - time_series_format: 如果时间序列中时间为字符串的格式，则需要该字段进行匹配
        - time_dict: 手动输入的时间, 支持多个时间收入
        - time_dict_fromat: 如果手动输入的时间为字符串，则需要该字段进行匹配
        - is_exact_hms: 是否提取时分秒相关特征

    返回值：
        Pandas DataFrame
    '''
    new_df = pd.DataFrame()
    
    if time_series.dtypes != 'datetime64[ns]':
        if not time_series_format:
            raise ValueError('time_series_format不能为空.')
        time_series = pd.to_datetime(time_series, format=time_series_format)
    col_name = time_series.name
    
    # 年、季度、月、日提取
    new_df[f'{col_name}_year'] = time_series.dt.year
    new_df[f'{col_name}_quarter'] = time_series.dt.quarter
    new_df[f'{col_name}_month'] = time_series.dt.month
    new_df[f'{col_name}_day'] = time_series.dt.day
    
    # 提取时分秒
    if is_exact_hms:
        new_df[f'{col_name}_hour'] = time_series.dt.hour
        new_df[f'{col_name}_minute'] = time_series.dt.minute
        new_df[f'{col_name}_second'] = time_series.dt.second
    
    # 周期提取
    new_df[f'{col_name}_weekofyear'] = time_series.dt.isocalendar().week
    new_df[f'{col_name}_dayofweek'] = time_series.dt.day_of_week+1
    new_df[f'{col_name}_weekend'] = (new_df[f'{col_name}_dayofweek'] > 5).astype(int) # 是否周末
    
    if is_exact_hms:
        new_df[f'{col_name}_hour_sec'] = (new_df[f'{col_name}_hour'] // 6).astype(int)
        new_df[f'{col_name}_day_or_night'] = (new_df[f'{col_name}_hour'] // 12).astype(int)
    
    # 时间差特征
    for key, val in time_dict.items():
        if isinstance(val, str):
            if not time_dict_format:
                raise ValueError('time_dict_format不能为空.')
            time_input = str_to_date(val, time_dict_format)            
        
        new_df[f'{col_name}_diff_days_{key}'] = time_series.apply(lambda x: get_days_diff(x, time_input))
        new_df[f'{col_name}_diff_months_{key}'] = time_series.apply(lambda x: get_months_diff(x, time_input))
        
    return new_df        


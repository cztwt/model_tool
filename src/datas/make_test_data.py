import pandas as pd
import numpy as np

def generate_sample_data(num_cust, num_prd, start_date, end_date):
    # 生成 cust_id 和 prd_id 的随机数据
    cust_ids = np.repeat([f'cust_{i}' for i in range(num_cust)], num_prd)
    prd_ids = [f'prd_{i}' for i in range(num_prd)] * num_cust

    # 生成 sample_date 的时间范围
    date_range = pd.date_range(start_date, end_date)
    sample_dates = np.random.choice(date_range, num_cust * num_prd, replace=True)

    # 创建 DataFrame
    data = pd.DataFrame({
        'cust_id': cust_ids,
        'prd_id': prd_ids,
        'sample_date': sample_dates
    })

    return data

def generate_data(num_records, cust_ids, start_date, end_date):
    # 随机选择 cust_id
    cust_selected = np.random.choice(cust_ids, num_records)

    # 生成 data_date 的时间范围
    date_range = pd.date_range(start_date, end_date)
    data_dates = np.random.choice(date_range, num_records, replace=True)
    
    # 生成 kind, cnt 和 val 的随机数据
    prods = np.random.choice(['prd_0', 'prd_1', 'prd_2'], num_records)
    kinds = np.random.choice([0, 1, 2], num_records)
    items = np.random.choice([0, 1, 2, 3], num_records)
    cnts = np.random.randint(1, 100, num_records)
    vals = np.random.uniform(1, 1000, num_records)

    # 创建 DataFrame
    data = pd.DataFrame({
        'cust_id': cust_selected,
        'data_date': data_dates,
        'item_id': prods,
        'kind': kinds,
        'item': items,
        'cnt': cnts,
        'val': vals
    })

    return data
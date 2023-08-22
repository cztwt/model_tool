import os
import logging
import warnings
import yaml


warnings.filterwarnings('ignore')
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 当前目录
curr_dir = os.path.dirname(os.path.abspath(__file__))
# 上级目录
parent_dir = os.path.dirname(curr_dir)


def parse_config(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def parse_data_config(file_path):
    '''解析数据配置文件,并存储为字典'''
    data, db_param = parse_config(file_path), dict()
    hive_db = {
        'host': data['hive_data_source']['host'],
        'port': data['hive_data_source']['port'],
        'username': data['hive_data_source']['username'],
        'password': data['hive_data_source']['password'],
        'database': data['hive_data_source']['database']
    }
    db_param['hive_db'] = hive_db
    mysql_db = {
        'host': data['mysql_data_source']['host'],
        'port': data['mysql_data_source']['port'],
        'username': data['mysql_data_source']['username'],
        'password': data['mysql_data_source']['password'],
        'database': data['mysql_data_source']['database']
    }
    db_param['mysql_db'] = mysql_db
    return db_param
